"""Exact and regressed episode scorers for the five controlled credit experiments."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import subprocess

import numpy as np
import torch

from .agents import ValueNetwork
from .buffers import JointEpisodeBuffer
from .dense_rewards import DenseReward
from .trajectory_logging import capture_lbf_state, to_jsonable


STATE_SCHEMA = "lbf-native-episode-v1"
ORACLE_ROLLOUTS = 3
MIXER_MIN_EPISODES = 50


class StdScaledScorer:
    """Scale mixer outputs without changing its stored targets or training units."""

    def __init__(self, reducer, divisor):
        self.reducer, self.divisor = reducer, divisor

    def transform(self, inputs):
        return self.reducer.transform(inputs) / self.divisor


class TotalEpisodeBuffer(JointEpisodeBuffer):
    """Regression readiness depends on total stored episodes, not class quotas."""

    @property
    def is_ready(self):
        return len(self.win_buffer) + len(self.loss_buffer) >= MIXER_MIN_EPISODES


def capture_state(env):
    base = env.unwrapped
    state = capture_lbf_state(env)
    state["food_spawned"] = float(base._food_spawned)
    state["max_episode_steps"] = int(base._max_episode_steps)
    for record, player in zip(state["players"], base.players):
        record["history"] = deepcopy(player.history)
        record["current_step"] = player.current_step
    return to_jsonable(state)


def observations_from_state(state, config):
    """Match native flat full-sight LBF observations, including observer-first order."""
    foods = np.asarray(state["food_grid"])
    food_slots = np.full((config.n_food, 3), [-1, -1, 0], dtype=np.float32)
    coords = np.argwhere(foods > 0)
    if len(coords) > config.n_food:
        raise ValueError("State contains more food than the critic input supports")
    for slot, (y, x) in enumerate(coords):
        food_slots[slot] = (y, x, foods[y, x])
    players = state["players"]
    result = []
    for ego in range(config.n_agents):
        order = [ego] + [i for i in range(config.n_agents) if i != ego]
        player_slots = [(*players[i]["position"], players[i]["level"]) for i in order]
        result.append(np.concatenate((food_slots.ravel(), np.asarray(player_slots).ravel())).astype(np.float32))
    return result


def splice_states(factual, donor, ego):
    result = []
    for fact, replacement in zip(factual, donor):
        state = deepcopy(fact)
        state["players"][ego]["position"] = list(replacement["players"][ego]["position"])
        state["players"][ego]["level"] = replacement["players"][ego]["level"]
        result.append(state)
    if not result:
        raise ValueError("Cannot splice an empty trajectory")
    return result


def joint_occupancy(states):
    if not states:
        raise ValueError("Joint occupancy requires at least one pre-action state")
    # Exact keys, no hashing collisions or dense enumeration. Static experiment
    # configuration is fixed; native episode fields include time and reward scale.
    counts = Counter()
    originals = {}
    for state in states:
        key = json.dumps(state, sort_keys=True, separators=(",", ":"))
        counts[key] += 1
        originals[key] = state
    return [(originals[key], count / len(states)) for key, count in counts.items()]


def terminal_score(state, config):
    helper = DenseReward("terminal_distance", config.grid_size, config.n_agents,
                         config.n_food, terminal_alpha=config.terminal_distance_alpha)
    return float(helper.terminal_reward(observations_from_state(state, config)).sum())


def average_pbrs_score(states, config):
    """Mean pre-action cooperative potential over a native episode."""
    if not states:
        raise ValueError("Average potential needs at least one pre-action state")
    helper = DenseReward("pbrs", config.grid_size, config.n_agents, config.n_food)
    # DenseReward.potential averages across agents; the requested trajectory
    # score normalizes the joint distance by max_distance only.
    return float(config.n_agents * config.terminal_distance_alpha * np.mean([
        helper.potential(observations_from_state(state, config)) for state in states
    ]))


def mean_td_advantage_score(values, team_rewards, gamma):
    """Mean one-step TD residual, with zero bootstrap at episode end."""
    values = np.asarray(values, dtype=np.float64)
    rewards = np.asarray(team_rewards, dtype=np.float64)
    if values.ndim != 1 or rewards.shape != values.shape or not len(values):
        raise ValueError("Mean advantage needs one value and team reward per pre-action state")
    if not np.isfinite(values).all() or not np.isfinite(rewards).all():
        raise ValueError("Mean advantage values and rewards must be finite")
    next_values = np.r_[values[1:], 0.0]
    return float(np.mean(rewards + gamma * next_values - values))


def mean_gae_advantage_score(values, team_rewards, gamma, gae_lambda, final_value=0.0):
    """Mean GAE; optional final value handles an exhausted replay recording."""
    return float(np.mean(trajectory_gae_advantages(values, team_rewards, gamma, gae_lambda, final_value)))


def trajectory_gae_advantages(values, team_rewards, gamma, gae_lambda, final_value=0.0):
    """Unstandardized teacher GAE at each transition, before score negation."""
    values = np.asarray(values, dtype=np.float64)
    rewards = np.asarray(team_rewards, dtype=np.float64)
    if values.ndim != 1 or rewards.shape != values.shape or not len(values):
        raise ValueError("Mean advantage needs one value and team reward per pre-action state")
    if not np.isfinite(values).all() or not np.isfinite(rewards).all() or not np.isfinite(final_value):
        raise ValueError("Mean advantage values and rewards must be finite")
    if not 0 <= gae_lambda <= 1:
        raise ValueError("GAE lambda must lie in [0, 1]")
    deltas = rewards + gamma * np.r_[values[1:], final_value] - values
    gae = 0.0
    advantages = np.empty_like(deltas)
    for t in range(len(deltas) - 1, -1, -1):
        gae = float(deltas[t]) + gamma * gae_lambda * gae
        advantages[t] = gae
    return advantages


def occupancy_goal_context(states, final_state, config):
    """Initial food goals and terminal remaining food, in fixed row-major slots."""
    if not states:
        raise ValueError("Goal context needs at least one pre-action state")
    slots = []
    for state in (states[0], final_state):
        coords = np.argwhere(np.asarray(state["food_grid"]) > 0)
        if len(coords) > config.n_food:
            raise ValueError("State contains more food than goal context supports")
        block = np.full((config.n_food, 2), -1., dtype=np.float32)
        block[:len(coords)] = coords / max(1, config.grid_size - 1)
        slots.append(block.ravel())
    return np.concatenate(slots)


def splice_average_pbrs_states(factual_states, donor_states, ego):
    """Hold factual food/history fixed; replace ego positions at each factual step.

    Donors shorter than the factual episode hold their final pre-action position,
    so factual and counterfactual averages use the same number of states.
    """
    if not factual_states or not donor_states:
        raise ValueError("Average-potential splice needs nonempty trajectories")
    result = []
    for t, fact in enumerate(factual_states):
        replacement = donor_states[min(t, len(donor_states) - 1)]["players"][ego]
        state = deepcopy(fact)
        state["players"][ego]["position"] = list(replacement["position"])
        state["players"][ego]["level"] = replacement["level"]
        result.append(state)
    return result


def terminal_donor_splice(final_state, donor_states, factual_length, ego):
    """Use Pipeline 1's common-prefix donor endpoint in the factual food world."""
    if factual_length < 1 or not donor_states:
        raise ValueError("Terminal donor splice needs nonempty factual and donor trajectories")
    hybrid = deepcopy(final_state)
    endpoint = donor_states[min(factual_length, len(donor_states)) - 1]["players"][ego]
    hybrid["players"][ego]["position"] = list(endpoint["position"])
    hybrid["players"][ego]["level"] = endpoint["level"]
    return hybrid


def evaluate_finalized_critic(agents, config):
    """Held-out environment-return calibration under the finalized PPO actors."""
    import gymnasium as gym
    from auxillary.reset_oracle import _sample_policy_action, _is_done
    np_state, torch_state = np.random.get_state(), torch.random.get_rng_state()
    env = gym.make(config.env_name)
    rng = np.random.default_rng(config.seed + 2_000_003)
    errors, episode_returns = [], []
    try:
        for episode in range(config.batch_size):
            obs, _ = env.reset(seed=config.seed + 2_000_003 + episode)
            predictions, rewards = [], []
            for _ in range(config.max_ep_len):
                flats = [np.asarray(o).flatten() for o in obs]
                with torch.no_grad():
                    values = []
                    for i, agent in enumerate(agents):
                        x = np.concatenate(flats) if config.centralized_critic else flats[i]
                        x = torch.as_tensor(x, dtype=torch.float32)
                        values.append(float(agent.critic_net(agent.critic_input(x))))
                predictions.append(float(np.mean(values)))
                actions = [_sample_policy_action(agent, flats[i], rng) for i, agent in enumerate(agents)]
                obs, reward, terminated, truncated, _ = env.step(actions)
                rewards.append(float(sum(reward)))
                if _is_done(terminated, truncated):
                    break
            targets = np.zeros(len(rewards))
            continuation = 0.
            for t in reversed(range(len(rewards))):
                continuation = rewards[t] + config.gamma * continuation
                targets[t] = continuation
            errors.extend((np.asarray(predictions)-targets).tolist())
            episode_returns.append(float(sum(rewards)))
    finally:
        env.close()
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
    return {"episodes": config.batch_size, "states": len(errors),
            "mean_environment_return": float(np.mean(episode_returns)),
            "value_rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "value_mae": float(np.mean(np.abs(errors))),
            "validation_seed_offset": 2_000_003}


def save_finalized_critic(path, agents, config, preparation_returns, validation=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1, "state_schema": STATE_SCHEMA,
        "seed": config.seed, "env_name": config.env_name,
        "n_agents": config.n_agents, "n_food": config.n_food,
        "grid_size": config.grid_size, "obs_dim_flat": config.obs_dim_flat,
        "critic_obs_dim": agents[0].critic_obs_dim,
        "centralized_critic": config.centralized_critic,
        "shared_critic": config.shared_critic, "reward_target": "environment_team_return",
        "aggregation": "mean_agent_conditioned_shared" if config.shared_critic else "mean_separate",
        "gamma": config.gamma, "training_episodes": len(preparation_returns),
        "final_100_mean_return": float(np.mean(preparation_returns[-100:])),
        "training_config": {key: value for key, value in vars(config).items()
                            if isinstance(value, (str, int, float, bool, type(None)))},
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "held_out_validation": validation,
    }
    critics = [agents[0].critic_net] if config.shared_critic else [a.critic_net for a in agents]
    payload = {"metadata": metadata,
               "critics": [{k: v.detach().cpu() for k, v in net.state_dict().items()} for net in critics],
               "actors": [{k: v.detach().cpu() for k, v in agent.actor_net.state_dict().items()} for agent in agents]}
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)
    digest = sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".manifest.json").write_text(json.dumps({**metadata, "sha256": digest}, indent=2) + "\n")
    return metadata


class FrozenPPOScorer:
    def __init__(self, path, config):
        self.path = str(Path(path).resolve())
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        self.metadata = payload["metadata"]
        self.sha256 = sha256(Path(self.path).read_bytes()).hexdigest()
        expected = {"schema_version": 1, "state_schema": STATE_SCHEMA,
                    "seed": config.seed, "env_name": config.env_name,
                    "obs_dim_flat": config.obs_dim_flat, "n_agents": config.n_agents,
                    "n_food": config.n_food, "grid_size": config.grid_size,
                    "gamma": config.gamma, "reward_target": "environment_team_return"}
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise ValueError(f"Critic checkpoint mismatch for {key}: expected {value!r}, got {self.metadata.get(key)!r}")
        self.config = config
        self.shared = self.metadata["shared_critic"]
        self.centralized = self.metadata["centralized_critic"]
        input_dim = self.metadata["critic_obs_dim"] + (config.n_agents if self.shared else 0)
        expected_count = 1 if self.shared else config.n_agents
        if len(payload["critics"]) != expected_count:
            raise ValueError("Checkpoint has an invalid critic count")
        # Loading must not change actor initialization/training RNG.
        with torch.random.fork_rng():
            self.nets = []
            for weights in payload["critics"]:
                net = ValueNetwork(input_dim)
                net.load_state_dict(weights)
                net.eval().requires_grad_(False)
                self.nets.append(net)

    def values(self, states):
        all_obs = [observations_from_state(state, self.config) for state in states]
        predictions = []
        with torch.no_grad():
            for ego in range(self.config.n_agents):
                inputs = [np.concatenate(obs) if self.centralized else obs[ego] for obs in all_obs]
                x = torch.as_tensor(np.asarray(inputs), dtype=torch.float32)
                if self.shared:
                    identity = torch.zeros((len(states), self.config.n_agents))
                    identity[:, ego] = 1
                    x = torch.cat((x, identity), dim=1)
                predictions.append(self.nets[0 if self.shared else ego](x).numpy())
        return np.mean(predictions, axis=0)

    def score(self, states):
        occupancy = joint_occupancy(states)
        values = self.values([state for state, _ in occupancy])
        return float(np.dot(values, [weight for _, weight in occupancy]))

    def temporal_credit_weights(self, states, team_rewards):
        """Share factual teacher |GAE| weights across agents; zero GAE is uniform."""
        advantages = trajectory_gae_advantages(
            self.values(states), team_rewards, self.config.gamma, self.config.ppo_gae_lambda)
        magnitudes = np.abs(advantages)
        peak = float(magnitudes.max())
        if peak == 0:
            return np.full(len(magnitudes), 1.0 / len(magnitudes))
        # Rescale first to avoid overflow while summing finite magnitudes.
        magnitudes /= peak
        return magnitudes / magnitudes.sum()

    def advantage_score(self, states, team_rewards, bootstrap_state=None):
        if team_rewards is None:
            raise ValueError("Pipeline 3.5 requires the actual environment team reward sequence")
        values = self.values(states)
        if bootstrap_state is not None and not getattr(self.config, "gae_advantage_score", False):
            raise ValueError("Replay truncation bootstrap requires GAE scoring")
        final_value = float(self.values([bootstrap_state])[0]) if bootstrap_state is not None else 0.0
        score = (mean_gae_advantage_score(values, team_rewards, self.config.gamma, self.config.ppo_gae_lambda,
                                        final_value=final_value)
                 if getattr(self.config, "gae_advantage_score", False)
                 else mean_td_advantage_score(values, team_rewards, self.config.gamma))
        return -score if getattr(self.config, "negate_mean_advantage_score", False) else score


class OnlinePPOScorer(FrozenPPOScorer):
    """Use live MAPPO critic networks with the same joint-state scoring rule."""

    def __init__(self, agents, config):
        if agents is None or len(agents) != config.n_agents:
            raise ValueError("Online critic scoring requires all live MAPPO agents")
        self.config = config
        self.shared = config.shared_critic
        self.centralized = config.centralized_critic
        self.nets = ([agents[0].critic_net] if self.shared else
                     [agent.critic_net for agent in agents])


def load_random_trajectory_bank(config, path, require_actions=False):
    """Cache uniform joint-policy donor states/actions without disturbing training RNG."""
    import gymnasium as gym
    path = Path(path)
    metadata = {"schema": "uniform-joint-pre-action-v2", "seed": config.seed,
                "env_name": config.env_name, "episodes": config.n_samples,
                "horizon": config.max_ep_len, "n_agents": config.n_agents,
                "action_dim": config.action_dim}
    if path.exists():
        saved = json.loads(path.read_text())
        legacy_metadata = {**metadata, "schema": "uniform-joint-pre-action-v1"}
        episodes = saved.get("episodes", [])
        if (saved.get("metadata") == metadata and len(episodes) == config.n_samples):
            return episodes
        has_actions = all("actions" in step for episode in episodes for step in episode)
        if (saved.get("metadata") == legacy_metadata and len(episodes) == config.n_samples
                and (has_actions or not require_actions)):
            return episodes
        if saved.get("metadata") not in (metadata, legacy_metadata):
            raise ValueError(f"Incompatible random trajectory bank: {path}")
        # Upgrade a legacy position-only cache when the trajectory regressor
        # needs actions for its occupancy signatures.
    rng = np.random.default_rng(config.seed + 3_000_003)
    np_state, torch_state = np.random.get_state(), torch.random.get_rng_state()
    env = gym.make(config.env_name)
    episodes = []
    try:
        for episode in range(config.n_samples):
            env.reset(seed=config.seed + 3_000_003 + episode)
            states = []
            for _ in range(config.max_ep_len):
                states.append({"players": [{"position": [int(x) for x in player.position],
                                            "level": int(player.level)}
                                           for player in env.unwrapped.players]})
                actions = rng.integers(config.action_dim, size=config.n_agents).tolist()
                states[-1]["actions"] = actions
                _, _, terminated, truncated, _ = env.step(actions)
                if bool(np.all(terminated)) or bool(np.all(truncated)):
                    break
            episodes.append(states)
    finally:
        env.close()
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"metadata": metadata, "episodes": episodes}))
    temporary.replace(path)
    return episodes


class CreditPipeline:
    def __init__(self, config, encoder, reducer, density, shaper, buffer, checkpoint=None, agents=None, scorer=None):
        self.config, self.encoder, self.reducer = config, encoder, reducer
        self.density, self.shaper, self.buffer = density, shaper, buffer
        self.pipeline = config.pipeline
        self.mean_advantage_score = getattr(config, "mean_advantage_score", False)
        self.score_std_scaling = getattr(config, "score_std_scaling", False)
        if self.score_std_scaling and not (self.mean_advantage_score and config.gae_advantage_score):
            raise ValueError("Score std scaling requires mean GAE advantage scoring")
        self.score_std = None
        self.score_std_samples = 0
        self.score_divisor = 1.0
        self._score_batch_advantages = []
        self.reset_oracle_mixer_cf = getattr(config, "reset_oracle_mixer_cf", False)
        self.action_replay_mixer_cf = getattr(config, "action_replay_mixer_cf", False)
        self.action_replay_direct_score = getattr(config, "action_replay_direct_score", False)
        if self.action_replay_direct_score and not self.action_replay_mixer_cf:
            raise ValueError("Direct replay scoring requires --action-replay-mixer-cf")
        if self.action_replay_mixer_cf:
            if not (self.pipeline == 5 and self.mean_advantage_score and config.gae_advantage_score
                    and config.negate_mean_advantage_score and config.occupancy_goal_input):
                raise ValueError("Action-replay mixer CF requires negative-GAE Pipeline 5.5 with occupancy goals")
            if (self.reset_oracle_mixer_cf or getattr(config, "random_donor_bank", None)
                    or getattr(config, "oracle_cf_diagnostic", False)
                    or getattr(config, "occupancy_cf_diagnostic", False)
                    or getattr(config, "occupancy_mean_cf_diagnostic", False)):
                raise ValueError("Action replay cannot be combined with donors, reactive reset CF, or splice diagnostics")
        if self.reset_oracle_mixer_cf and not (self.pipeline == 5 and self.mean_advantage_score
                                                and config.gae_advantage_score
                                                and config.negate_mean_advantage_score
                                                and config.occupancy_goal_input
                                                and not config.online_critic_score
                                                and not getattr(config, "random_donor_bank", None)):
            raise ValueError("Reset-oracle mixer CF requires frozen negative-GAE Pipeline 5.5 with occupancy goals and no donor bank")
        if self.mean_advantage_score and self.pipeline not in (3, 5):
            raise ValueError("Mean advantage scoring applies only to pipelines 3 and 5")
        if getattr(config, "negate_mean_advantage_score", False) and not self.mean_advantage_score:
            raise ValueError("Negated mean advantage requires mean advantage scoring")
        if getattr(config, "gae_advantage_score", False) and not self.mean_advantage_score:
            raise ValueError("GAE advantage requires mean advantage scoring")
        if getattr(config, "online_critic_score", False) and not self.mean_advantage_score:
            raise ValueError("Online critic scoring requires mean advantage scoring")
        self.mixer = self.pipeline in (2, 5)
        self.frozen = (scorer if scorer is not None else
                       OnlinePPOScorer(agents, config) if getattr(config, "online_critic_score", False)
                       else FrozenPPOScorer(checkpoint, config) if self.pipeline in (3, 4, 5)
                       else None)
        self.oracle_rng = np.random.default_rng(config.seed + 1_000_003)
        self.fit_version = 0
        self.random_bank = (load_random_trajectory_bank(
            config, config.random_donor_bank, require_actions=self.pipeline in (2, 5))
                            if getattr(config, "random_donor_bank", None) else None)
        self.random_donor_trajectories = None
        self.random_donor_signatures = None
        if self.pipeline in (2, 5) and self.random_bank is not None and not config.terminal_goal_input:
            self.random_donor_trajectories = []
            self.random_donor_signatures = []
            for donor in self.random_bank:
                per_agent = [[] for _ in range(config.n_agents)]
                for step in donor:
                    # Food fields do not enter the occupancy encoder; zero food
                    # slots preserve the native observation offsets for players.
                    donor_state = {"food_grid": np.zeros((config.grid_size, config.grid_size)),
                                   "players": step["players"]}
                    donor_obs = observations_from_state(donor_state, config)
                    for agent in range(config.n_agents):
                        per_agent[agent].append((donor_obs[agent], int(step["actions"][agent])))
                self.random_donor_trajectories.append(per_agent)
                self.random_donor_signatures.append(
                    np.concatenate(encoder.encode_per_agent(per_agent, config.n_agents, config)))
            self.random_donor_signatures = np.asarray(self.random_donor_signatures, dtype=np.float32)

    def finish_score_batch(self):
        """Freeze factual population std for the next rollout batch; no centering."""
        if not self.score_std_scaling or not self._score_batch_advantages:
            return
        advantages = np.concatenate(self._score_batch_advantages)
        self.score_std = float(np.std(advantages, ddof=0))
        self.score_std_samples = len(advantages)
        # No variability supplies no useful scale; retain neutral scoring units.
        self.score_divisor = self.score_std + 1e-8 if self.score_std > 1e-8 else 1.0
        self._score_batch_advantages.clear()

    def scaled_advantage_score(self, states, team_rewards, bootstrap_state=None):
        kwargs = {} if bootstrap_state is None else {"bootstrap_state": bootstrap_state}
        return self.frozen.advantage_score(states, team_rewards, **kwargs) / self.score_divisor

    def target(self, states, final_state, team_rewards=None):
        if self.pipeline in (1, 2):
            return (average_pbrs_score(states, self.config) if self.config.average_pbrs_score
                    else terminal_score(final_state, self.config))
        if self.mean_advantage_score:
            return self.frozen.advantage_score(states, team_rewards)
        return self.frozen.score(states)

    def _normalize(self, raw):
        # Same multi-sample scale update as the existing compute_all_rewards.
        max_abs = float(np.max(np.abs(raw)))
        if self.shaper.running_max_abs is None:
            self.shaper.running_max_abs = max(1.0, max_abs)
        else:
            self.shaper.running_max_abs = .99 * self.shaper.running_max_abs + .01 * max_abs
        reward = raw / max(1e-8, self.shaper.running_max_abs)
        if self.shaper.clip_val is not None:
            reward = np.clip(reward, -self.shaper.clip_val, self.shaper.clip_val)
        return reward

    def process(self, ep, trajectories, states, final_state, env_return, running_avg,
                env, initial_snapshot, initial_obs, agents, team_rewards=None):
        c = self.config
        signatures = (self.encoder.encode_per_agent_from_state(final_state, c)
                      if self.mixer and c.terminal_goal_input else
                      self.encoder.encode_per_agent(trajectories, c.n_agents, c)
                      if self.mixer else None)
        goal_context = (occupancy_goal_context(states, final_state, c)
                        if self.mixer and c.occupancy_goal_input else None)
        joint = (np.concatenate([*signatures, goal_context]) if goal_context is not None else
                 np.concatenate(signatures) if self.mixer else np.empty(0, dtype=np.float32))
        if self.score_std_scaling:
            self._score_batch_advantages.append(trajectory_gae_advantages(
                self.frozen.values(states), team_rewards, c.gamma, c.ppo_gae_lambda))
        raw_target = self.target(states, final_state, team_rewards)
        target = raw_target / self.score_divisor
        scoring_reducer = (StdScaledScorer(self.reducer, self.score_divisor)
                           if self.score_std_scaling else self.reducer)
        prediction_before_fit = (float(scoring_reducer.transform(joint[None])[0, 0])
                                 if self.mixer and self.fit_version else None)
        store_episode = self.pipeline != 3 and (self.random_bank is None or self.pipeline in (2, 5))
        if store_episode:
            self.buffer.store(trajectories, joint, env_return)
            record = (self.buffer.win_buffer if env_return > 0 else self.buffer.loss_buffer)[-1]
            record.states, record.final_state, record.target, record.episode = states, final_state, raw_target, ep
        losses = list(self.buffer.loss_buffer)
        wins = list(self.buffer.win_buffer)
        enough_episodes = len(wins) + len(losses) >= MIXER_MIN_EPISODES
        if self.mixer:
            should_fit = (enough_episodes and ep % c.fit_freq == 0 and
                          (c.beta_mode != "dynamic" or running_avg >= c.beta_threshold - .1))
            if should_fit:
                records = wins + losses
                self.reducer.fit(np.asarray([r.joint_signature for r in records]),
                                 np.asarray([r.target for r in records]))
                self.density._is_fitted = True
                self.fit_version += 1
            occupancy_dim = self.encoder.joint_dim(c.n_agents) if c.occupancy_goal_input else None
            self.shaper.update_baselines(np.asarray([r.joint_signature[:occupancy_dim] for r in wins]),
                                        np.asarray([r.joint_signature[:occupancy_dim] for r in losses]),
                                        [r.raw_trajectories for r in losses])
            if self.random_donor_trajectories is not None:
                self.shaper._X_loss = self.random_donor_signatures
                self.shaper._loss_trajs = self.random_donor_trajectories
        eligible = env_return <= 0 or c.shape_wins
        ready = (not self.mixer or
                 (ep >= c.warmup_episodes and enough_episodes and getattr(self.density, "_is_fitted", False)))
        if self.random_bank is not None:
            ready = ready and running_avg > c.beta_threshold
        available = (self.pipeline == 3 or self.reset_oracle_mixer_cf or self.action_replay_mixer_cf
                     or self.random_bank is not None or bool(losses))
        ledger = {"episode": ep, "target": target, "eligible": eligible,
                  "ready": ready, "available": available, "scored": False,
                  "donor_source": ("recorded_action_replay" if self.action_replay_mixer_cf else
                                   "native_reset_rollout" if self.reset_oracle_mixer_cf else
                                   "uniform_random_bank" if self.random_bank is not None else
                                   "random_reset_rollout" if self.pipeline == 3 else "loss_buffer"),
                  "requested_samples": ORACLE_ROLLOUTS if self.pipeline == 3 or self.reset_oracle_mixer_cf or self.action_replay_mixer_cf
                                       else c.n_samples, "actual_samples": 0,
                  "prediction_before_fit": prediction_before_fit,
                  "buffered_episodes": len(wins) + len(losses),
                  "buffered_wins": len(wins), "buffered_losses": len(losses),
                  "fit_version": self.fit_version, "raw_credit": [0.] * c.n_agents}
        if self.mean_advantage_score:
            kind = "gae" if getattr(c, "gae_advantage_score", False) else "td"
            sign = "negative_" if getattr(c, "negate_mean_advantage_score", False) else ""
            ledger["score_type"] = f"{sign}mean_{kind}_advantage"
            ledger["critic_source"] = "online" if getattr(c, "online_critic_score", False) else "frozen"
        if self.score_std_scaling:
            ledger.update(unscaled_target=raw_target, score_std=self.score_std,
                          score_std_samples=self.score_std_samples, score_divisor=self.score_divisor)
        if not eligible or not ready or not available:
            ledger["skip_reason"] = "episode_ineligible" if not eligible else "scorer_not_ready" if not ready else "no_loss_donors"
            return np.zeros(c.n_agents), ledger
        if self.action_replay_mixer_cf:
            from .action_replay import action_replay_counterfactuals
            rollouts, diagnostics = action_replay_counterfactuals(
                c, states, trajectories, self.oracle_rng, ORACLE_ROLLOUTS)
            samples, critic_samples = [], []
            for group in rollouts:
                mixer_group, critic_group = [], []
                for rollout in group:
                    cf_agents = self.encoder.encode_per_agent(rollout["trajectories"], c.n_agents, c)
                    cf_goal = occupancy_goal_context(rollout["states"], rollout["final_state"], c)
                    cf_joint = np.concatenate([*cf_agents, cf_goal])
                    mixer_group.append(float(scoring_reducer.transform(cf_joint[None])[0, 0]))
                    bootstrap = (rollout["final_state"] if rollout["diagnostics"]["bootstrap_final_value"]
                                 else None)
                    critic_group.append(self.scaled_advantage_score(
                        rollout["states"], rollout["rewards"], bootstrap_state=bootstrap))
                samples.append(mixer_group)
                critic_samples.append(critic_group)
            factual = float(scoring_reducer.transform(joint[None])[0, 0])
            means = np.mean(samples, axis=1)
            raw = factual - means
            donor_ids = []
            ledger.update(action_replay=diagnostics, mixer_on_action_replay_cf=True,
                          replay_critic_cf_scores=critic_samples,
                          replay_critic_cf_means=np.mean(critic_samples, axis=1).tolist(),
                          replay_critic_raw_credit=(target - np.mean(critic_samples, axis=1)).tolist())
            if self.action_replay_direct_score:
                # Keep fitting, readiness, RNG consumption and replay identical;
                # only the score supplying the applied credit changes.
                ledger.update(mixer_factual_score=factual, mixer_cf_scores=samples,
                              mixer_raw_credit=raw.tolist(), credit_score_source="direct_teacher")
                factual, samples = target, critic_samples
                means = np.mean(samples, axis=1)
                raw = factual - means
            rewards = self._normalize(raw)
        elif self.reset_oracle_mixer_cf:
            from auxillary.reset_oracle import compute_oracle_counterfactual
            if initial_snapshot is None or initial_obs is None or agents is None:
                raise ValueError("Reset-oracle mixer CF needs the initial simulator snapshot, observations, and agents")
            mixer_samples = [[] for _ in range(c.n_agents)]

            def observe_rollout(ego, oracle_score, cf_trajectories, cf_states, cf_final_state):
                cf_agents = self.encoder.encode_per_agent(cf_trajectories, c.n_agents, c)
                cf_goal = occupancy_goal_context(cf_states, cf_final_state, c)
                cf_joint = np.concatenate([*cf_agents, cf_goal])
                mixer_samples[ego].append(float(scoring_reducer.transform(cf_joint[None])[0, 0]))

            factual = float(scoring_reducer.transform(joint[None])[0, 0])
            oracle_means, _, oracle_samples = compute_oracle_counterfactual(
                env, initial_snapshot, initial_obs, agents, c, None, None, None,
                target, self.oracle_rng, K=ORACLE_ROLLOUTS,
                state_capture=capture_state, return_samples=True,
                reward_aware_scorer=self.scaled_advantage_score,
                rollout_observer=observe_rollout)
            samples = mixer_samples
            means = np.mean(samples, axis=1)
            raw = factual - means
            rewards = self._normalize(raw)
            donor_ids = []
            ledger["oracle_cf_scores"] = oracle_samples
            ledger["oracle_cf_means"] = np.asarray(oracle_means).tolist()
            ledger["oracle_raw_credit"] = (target - oracle_means).tolist()
            ledger["mixer_on_native_reset_cf"] = True
        elif self.mixer and c.terminal_goal_input:
            donors = self.random_bank if self.random_bank is not None else losses
            indices = np.random.choice(len(donors), size=min(c.n_samples, len(donors)), replace=False)
            donor_ids = ([int(i) for i in indices] if self.random_bank is not None else
                         [losses[i].episode for i in indices])
            cf_signatures = []
            exact_cf_scores = []
            for ego in range(c.n_agents):
                exact_agent_scores = []
                for idx in indices:
                    donor = donors[idx]
                    donor_states = donor if self.random_bank is not None else donor.states
                    hybrid = terminal_donor_splice(final_state, donor_states, len(states), ego)
                    cf_signatures.append(self.encoder.encode_from_state(hybrid, c))
                    exact_agent_scores.append(terminal_score(hybrid, c))
                exact_cf_scores.append(exact_agent_scores)
            values = scoring_reducer.transform(np.vstack([joint, *cf_signatures])).reshape(-1)
            factual = float(values[0])
            samples = values[1:].reshape(c.n_agents, len(indices)).tolist()
            means = np.mean(samples, axis=1)
            raw = factual - means
            rewards = self._normalize(raw)
            ledger["exact_factual_score"] = target
            ledger["exact_cf_scores"] = exact_cf_scores
        elif self.mixer and c.occupancy_goal_input:
            donors = self.shaper._loss_trajs
            indices = np.random.choice(len(donors), size=min(c.n_samples, len(donors)), replace=False)
            donor_ids = ([int(i) for i in indices] if self.random_bank is not None else
                         [losses[i].episode for i in indices])
            cf_signatures = []
            for ego in range(c.n_agents):
                for idx in indices:
                    cf_occupancy = self.shaper._construct_counterfactual_signature_for_loss_idx(
                        trajectories, ego, idx, self.encoder, c, agent_signatures=signatures)
                    cf_signatures.append(np.concatenate([cf_occupancy, goal_context]))
            values = scoring_reducer.transform(np.vstack([joint, *cf_signatures])).reshape(-1)
            factual = float(values[0])
            samples = values[1:].reshape(c.n_agents, len(indices)).tolist()
            means = np.mean(samples, axis=1)
            raw = factual - means
            running_scale_before = self.shaper.running_max_abs
            rewards = self._normalize(raw)
            self.shaper.last_reward_diagnostics = {
                "sampled_donor_indices": indices.astype(int).tolist(),
                "sampled_cf_scores": samples,
                "sampled_cf_signatures": [np.asarray(cf_signatures[ego*len(indices):(ego+1)*len(indices)])
                                          for ego in range(c.n_agents)],
                "running_scale_before": running_scale_before,
                "running_scale_after": float(self.shaper.running_max_abs),
                "unclipped_normalized_rewards": (raw / max(1e-8, self.shaper.running_max_abs)).tolist(),
            }
        elif self.mixer:
            rewards, factual, means, raw = self.shaper.compute_all_rewards(
                joint, scoring_reducer, self.density, episode_trajectories=trajectories,
                encoder=self.encoder, config=c, agent_signatures=signatures,
                return_raw_scores=True)
            diag = self.shaper.last_reward_diagnostics
            samples = diag["sampled_cf_scores"]
            donor_ids = ([int(i) for i in diag["sampled_donor_indices"]]
                         if self.random_bank is not None else
                         [losses[i].episode for i in diag["sampled_donor_indices"]])
        elif self.pipeline == 3:
            from auxillary.reset_oracle import compute_oracle_counterfactual
            factual = target
            scorer_kwargs = ({"reward_aware_scorer": self.scaled_advantage_score}
                             if self.mean_advantage_score else {"episode_scorer": self.frozen.score})
            means, raw, samples = compute_oracle_counterfactual(
                env, initial_snapshot, initial_obs, agents, c, None, None, None,
                factual, self.oracle_rng, K=ORACLE_ROLLOUTS,
                state_capture=capture_state, return_samples=True, **scorer_kwargs)
            rewards = self._normalize(raw)
            donor_ids = []
        else:
            factual = target
            donors = self.random_bank if self.random_bank is not None else losses
            indices = np.random.choice(len(donors), size=min(c.n_samples, len(donors)), replace=False)
            donor_ids = [int(i) if self.random_bank is not None else losses[i].episode for i in indices]
            samples = []
            for ego in range(c.n_agents):
                scores = []
                for idx in indices:
                    donor = donors[idx]
                    donor_states = donor if self.random_bank is not None else donor.states
                    if self.pipeline == 1:
                        if c.average_pbrs_score:
                            hybrid = splice_average_pbrs_states(states, donor_states, ego)
                            scores.append(average_pbrs_score(hybrid, c))
                        else:
                            hybrid = terminal_donor_splice(final_state, donor_states, len(states), ego)
                            scores.append(terminal_score(hybrid, c))
                    else:
                        scores.append(self.frozen.score(splice_states(states, donor.states, ego)))
                samples.append(scores)
            means = np.mean(samples, axis=1)
            raw = factual - means
            rewards = self._normalize(raw)
        ledger.update(scored=True, factual_score=float(factual), cf_scores=samples,
                      cf_means=np.asarray(means).tolist(), raw_credit=np.asarray(raw).tolist(),
                      normalized_credit=np.asarray(rewards).tolist(), donor_episodes=donor_ids,
                      actual_samples=len(samples[0]), running_scale=self.shaper.running_max_abs)
        if self.mixer:
            ledger["prediction_minus_factual_target"] = float(ledger.get("mixer_factual_score", factual) - target)
        return rewards, ledger
