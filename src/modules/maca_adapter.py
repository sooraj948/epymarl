"""Online random-donor MACA credit for full-sight native LBF episodes."""

from collections import deque
from contextlib import contextmanager
import copy
import importlib.metadata
import json
from pathlib import Path
import random
import re
import subprocess
import sys

import gymnasium as gym
import numpy as np
import torch

from components.episode_buffer import EpisodeBatch
from maca.config import MACAConfig
from maca.credit_pipelines import (
    CreditPipeline, FrozenPPOScorer, TotalEpisodeBuffer, observations_from_state,
)
from maca.density import PassthroughDensityEstimator
from maca.encoders import StateActionOccupancyEncoder
from maca.reducers import TrajectoryReturnRegressor
from maca.shaper import CounterfactualRewardShaper
from maca.trajectory_logging import to_jsonable


def state_from_observations(observations, config):
    """Reconstruct the public board; reject partial or reordered observations."""
    obs = np.asarray(observations, dtype=np.float32)
    if obs.shape != (config.n_agents, config.obs_dim_flat) or not np.isfinite(obs).all():
        raise ValueError("MACA needs finite native flat full-sight LBF observations")
    if not np.equal(obs, np.round(obs)).all():
        raise ValueError("MACA needs unnormalized native LBF coordinates and levels")
    foods = obs[0, :3 * config.n_food].reshape(-1, 3)
    grid = np.zeros((config.grid_size, config.grid_size), dtype=np.int32)
    for row, col, level in foods.astype(int):
        if (row, col, level) == (-1, -1, 0):
            continue
        if not (0 <= row < config.grid_size and 0 <= col < config.grid_size and level > 0):
            raise ValueError("Invalid food slot in LBF observation")
        if grid[row, col]:
            raise ValueError("Duplicate food in LBF observation")
        grid[row, col] = level
    players = []
    for agent in range(config.n_agents):
        row, col, level = obs[agent, 3 * config.n_food:3 * config.n_food + 3].astype(int)
        if not (0 <= row < config.grid_size and 0 <= col < config.grid_size and level > 0):
            raise ValueError("MACA requires full-sight observations of every player")
        players.append({"position": [int(row), int(col)], "level": int(level)})
    # Native LBF can overlap players after a blocked movement chain. Preserve
    # that observed board exactly; no-replay splice does not require a reset.
    state = {"food_grid": grid, "players": players}
    if not np.array_equal(np.asarray(observations_from_state(state, config)), obs):
        raise ValueError("Observation does not match native full-sight LBF ordering")
    return state


class EPyMARLScorer(FrozenPPOScorer):
    """Reuse MACA score formulas, with values supplied by the live EPyMARL critic."""

    def __init__(self, critic, config):
        self.critic, self.config = critic, config
        self.reward_mean, self.reward_std = 0.0, 1.0

    def set_reward_normalization(self, mean, std):
        """Freeze the policy's environment-reward transform for this batch."""
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError("MACA needs finite reward moments and a positive standard deviation")
        self.reward_mean, self.reward_std = float(mean), float(std)

    def advantage_score(self, states, team_rewards, bootstrap_state=None):
        if team_rewards is None:
            raise ValueError("MACA teacher requires the environment reward sequence")
        rewards = (np.asarray(team_rewards, dtype=np.float64) - self.reward_mean) / self.reward_std
        return super().advantage_score(states, rewards, bootstrap_state=bootstrap_state)

    def values(self, states):
        observations = np.asarray([
            observations_from_state(state, self.config) for state in states
        ], dtype=np.float32)
        device = next(iter(self.critic.parameters())).device
        values_batch = EpisodeBatch(
            {"state": {"vshape": self.config.n_agents * self.config.obs_dim_flat}},
            {}, 1, len(states), device=device,
        )
        values_batch.update({"state": torch.as_tensor(
            observations.reshape(1, len(states), -1), device=device,
        )})
        with torch.no_grad():
            values = self.critic(values_batch)[0, :, :, 0].mean(dim=-1)
        return values.cpu().numpy()


class MACAAdapter:
    def __init__(self, args, scheme, critic, logger):
        if not args.common_reward or args.reward_scalarisation != "sum":
            raise ValueError("MACA requires common_reward=True and reward_scalarisation=sum")
        if args.standardise_returns:
            raise ValueError("MACA currently requires standardise_returns=False")
        if (args.critic_type not in {"cv_critic", "cv_critic_ns"}
                or args.mac not in {"basic_mac", "non_shared_mac"}
                or args.obs_last_action or args.obs_individual_obs):
            raise ValueError("MACA supports state-only centralized MAPPO critics and basic/non-shared actors")
        if args.env != "gymma" or args.env_args.get("pretrained_wrapper"):
            raise ValueError("MACA currently supports native full-sight LBF through gymma")
        if not (args.buffer_size == args.batch_size == args.batch_size_run):
            raise ValueError("MACA requires buffer_size=batch_size=batch_size_run for fresh episode batches")
        allowed_env = {"key", "time_limit", "pretrained_wrapper", "seed"}
        if set(args.env_args) - allowed_env:
            raise ValueError("Additional environment constructor settings are not supported by the MACA donor bank")
        key = args.env_args["key"]
        match = re.fullmatch(r"(?:lbforaging:)?Foraging-(\d+)x(\d+)-(\d+)p-(\d+)f(?:-coop)?-v3", key)
        if match is None or match[1] != match[2]:
            raise ValueError("MACA currently requires a square full-sight native Foraging v3 task")
        self.config = c = MACAConfig.from_env_name(key)
        c.seed, c.gamma = args.seed, args.gamma
        c.max_ep_len = int(args.env_args["time_limit"])
        c.n_agents, c.action_dim, c.batch_size = args.n_agents, args.n_actions, args.batch_size
        if (c.n_agents != int(match[3]) or c.action_dim != 6
                or scheme["obs"]["vshape"] != c.obs_dim_flat
                or scheme["state"]["vshape"] != c.n_agents * c.obs_dim_flat):
            raise ValueError("MACA environment dimensions do not match native LBF observations")
        allowed = {
            "beta", "beta_threshold", "beta_ramp", "fit_freq", "mixer_epochs",
            "mixer_batch_size", "n_samples", "buffer_capacity", "warmup_episodes",
            "shape_wins", "ppo_gae_lambda",
        }
        if set(args.maca) - allowed:
            raise ValueError(f"Unknown MACA settings: {sorted(set(args.maca) - allowed)}")
        for name, value in args.maca.items():
            setattr(c, name, value)
        c.shape_wins = bool(args.maca.get("shape_wins", False))
        for name in ("fit_freq", "mixer_epochs", "mixer_batch_size", "n_samples", "buffer_capacity"):
            value = getattr(c, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"MACA {name} must be a positive integer")
        if c.mixer_batch_size < 2 or c.buffer_capacity < 50:
            raise ValueError("MACA needs mixer_batch_size>=2 and buffer_capacity>=50")
        if (not np.isfinite([c.beta, c.beta_threshold, c.beta_ramp, c.ppo_gae_lambda]).all()
                or c.beta < 0 or c.beta_ramp < 0 or not 0 <= c.ppo_gae_lambda <= 1
                or c.warmup_episodes < 0 or c.max_ep_len < 1):
            raise ValueError("Invalid MACA reward schedule or teacher GAE settings")
        c.pipeline = 5
        c.encoder_type, c.reducer_type = "occupancy", "trajectory_return"
        c.baseline_mode, c.beta_mode, c.r_int_injection = "multi_sample", "dynamic", "uniform"
        c.online_critic_score = c.mean_advantage_score = c.gae_advantage_score = True
        c.negate_mean_advantage_score = c.occupancy_goal_input = True
        c.centralized_critic, c.shared_critic = True, args.critic_type == "cv_critic"
        self.standardise_rewards = args.standardise_rewards
        self.output = Path(args.local_results_path) / "maca" / args.unique_token
        self.output.mkdir(parents=True, exist_ok=True)
        c.random_donor_bank = str(self.output / "random_donors.json")
        self.logger = logger
        self.episodes = 0
        self.returns = deque(maxlen=100)
        self.last_ledgers = []
        self.rng = (
            np.random.RandomState(args.seed + 4_000_003).get_state(),
            torch.Generator().manual_seed(args.seed + 4_000_003).get_state(),
            random.Random(args.seed + 4_000_003).getstate(),
        )
        with self.isolated_rng():
            # Validate the task before generating the donor cache.
            env = gym.make(key)
            try:
                if env.unwrapped.sight < c.grid_size:
                    raise ValueError("MACA requires full-sight LBF")
                obs, _ = env.reset(seed=c.seed)
                state_from_observations(obs, c)
            finally:
                env.close()
            encoder = StateActionOccupancyEncoder(c.grid_size, c.action_dim)
            reducer = TrajectoryReturnRegressor(
                encoder.joint_dim(c.n_agents) + 4 * c.n_food,
                lr=c.mixer_lr, epochs=c.mixer_epochs, batch_size=c.mixer_batch_size, seed=c.seed,
            )
            self.pipeline = CreditPipeline(
                c, encoder, reducer, PassthroughDensityEstimator(),
                CounterfactualRewardShaper(
                    baseline_mode=c.baseline_mode, per_agent_dim=encoder.per_agent_dim,
                    n_agents=c.n_agents, clip_val=None, n_samples=c.n_samples,
                ),
                TotalEpisodeBuffer(buffer_capacity=c.buffer_capacity),
                scorer=EPyMARLScorer(critic, c),
            )
        self._write_manifest(args)

    @contextmanager
    def isolated_rng(self):
        policy_rng = (np.random.get_state(), torch.random.get_rng_state(), random.getstate())
        np.random.set_state(self.rng[0])
        torch.random.set_rng_state(self.rng[1])
        random.setstate(self.rng[2])
        try:
            yield
        finally:
            self.rng = (np.random.get_state(), torch.random.get_rng_state(), random.getstate())
            np.random.set_state(policy_rng[0])
            torch.random.set_rng_state(policy_rng[1])
            random.setstate(policy_rng[2])

    def _write_manifest(self, args):
        repo = Path(__file__).resolve().parents[2]
        def git(*command):
            try:
                return subprocess.check_output(["git", "-C", str(repo), *command], text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                return None
        versions = {}
        for name in ("torch", "numpy", "gymnasium", "lbforaging", "scikit-learn", "sacred"):
            versions[name] = importlib.metadata.version(name)
        manifest = {
            "variant": "online_random_donor_splice", "policy_advantages": "epymarl_nstep",
            "teacher_score": "negative_mean_gae_without_advantage_standardization",
            "teacher_reward_units": "standardized_environment_rewards" if args.standardise_rewards else "raw_environment_rewards",
            "policy_shaping": "add_individual_credit_in_raw_units_then_apply_environment_reward_transform",
            "reward_moments_source": "environment_rewards_only_once_per_ppo_batch",
            "python": sys.version,
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "versions": versions, "device": args.device, "config": vars(args),
            "maca_config": vars(self.config),
        }
        (self.output / "manifest.json").write_text(json.dumps(to_jsonable(manifest), indent=2) + "\n")

    def prepare(self, batch, t_env):
        c = self.config
        shaped = torch.zeros(
            (batch.batch_size, batch.max_seq_length - 1, c.n_agents),
            dtype=batch["reward"].dtype, device=batch["reward"].device,
        )
        ledgers = []
        with self.isolated_rng():
            for b in range(batch.batch_size):
                filled = batch["filled"][b, :, 0].detach().cpu().numpy()
                count = int(filled.sum())
                if count < 2 or not np.array_equal(filled, np.r_[np.ones(count), np.zeros(len(filled) - count)]):
                    raise ValueError("MACA requires complete contiguous episodes and their final observation")
                length = count - 1
                if length > c.max_ep_len or bool(batch["terminated"][b, :length - 1].any()):
                    raise ValueError("MACA received an invalid episode boundary")
                obs = batch["obs"][b, :count].detach().cpu().numpy()
                if not np.array_equal(batch["state"][b, :count].detach().cpu().numpy(), obs.reshape(count, -1)):
                    raise ValueError("MACA critic expects concatenated native observations as global state")
                actions = batch["actions"][b, :length, :, 0].detach().cpu().numpy()
                if (actions < 0).any() or (actions >= c.action_dim).any():
                    raise ValueError("Invalid requested LBF action")
                states = [state_from_observations(o, c) for o in obs]
                trajectories = [[(obs[t, i].copy(), int(actions[t, i])) for t in range(length)]
                                for i in range(c.n_agents)]
                rewards = batch["reward"][b, :length, 0].detach().cpu().numpy()
                if not np.isfinite(rewards).all():
                    raise ValueError("MACA received nonfinite environment rewards")
                env_return = float(rewards.sum())
                self.returns.append(env_return)
                running_avg = float(np.mean(self.returns))
                credit, ledger = self.pipeline.process(
                    self.episodes, trajectories, states[:-1], states[-1], env_return,
                    running_avg, None, None, None, None, team_rewards=rewards,
                )
                fraction = (float(running_avg > c.beta_threshold) if c.beta_ramp == 0 else
                            float(np.clip((running_avg - c.beta_threshold) / c.beta_ramp, 0, 1)))
                weight = c.beta * fraction
                for i in range(c.n_agents):
                    agent_rewards = self.pipeline.shaper.spread_reward(
                        rewards, float(credit[i]), mode="uniform", dynamic_beta=weight,
                    )
                    shaped[b, :length, i] = torch.as_tensor(agent_rewards, device=shaped.device)
                ledger.update(
                    t_env=t_env, episode_length=length, env_return=env_return,
                    running_return=running_avg, effective_weight=weight,
                    applied_credit=(np.asarray(credit) * weight).tolist(),
                    reward_mean=self.pipeline.frozen.reward_mean,
                    reward_std=self.pipeline.frozen.reward_std,
                )
                ledgers.append(ledger)
                self.episodes += 1
            self.pipeline.finish_score_batch()
        if not torch.isfinite(shaped).all():
            raise ValueError("MACA produced nonfinite shaped rewards")
        self.last_ledgers = ledgers
        with (self.output / "credit_episodes.jsonl").open("a") as stream:
            stream.writelines(json.dumps(to_jsonable(row)) + "\n" for row in ledgers)
        for name, value in {
            "episodes": self.episodes, "fit_version": self.pipeline.fit_version,
            "scored_fraction": np.mean([r["scored"] for r in ledgers]),
            "effective_beta": np.mean([r["effective_weight"] for r in ledgers]),
        }.items():
            self.logger.log_stat("maca/" + name, float(value), t_env)
        for agent in range(c.n_agents):
            self.logger.log_stat(f"maca/applied_credit_agent_{agent}", float(np.mean([
                r["applied_credit"][agent] for r in ledgers
            ])), t_env)
        return shaped

    def state_dict(self):
        excluded = {"config", "frozen", "encoder", "reducer", "density", "shaper", "buffer"}
        return copy.deepcopy({
            "version": 1,
            "standardise_rewards": self.standardise_rewards,
            "reward_moments": (self.pipeline.frozen.reward_mean, self.pipeline.frozen.reward_std),
            "config": {k: v for k, v in vars(self.config).items() if k != "random_donor_bank"},
            "episodes": self.episodes, "returns": list(self.returns), "rng": self.rng,
            "pipeline": {k: v for k, v in vars(self.pipeline).items() if k not in excluded},
            "buffer": self.pipeline.buffer, "shaper": vars(self.pipeline.shaper),
            "density": vars(self.pipeline.density), "mixer": self.pipeline.reducer.net.state_dict(),
            "mixer_rng": self.pipeline.reducer._generator.get_state(),
        })

    def load_state_dict(self, state):
        expected = {k: v for k, v in vars(self.config).items() if k != "random_donor_bank"}
        if (state["version"] != 1 or state["config"] != expected
                or state.get("standardise_rewards", False) != self.standardise_rewards):
            raise ValueError("MACA checkpoint configuration does not match this run")
        state = copy.deepcopy(state)
        self.episodes, self.rng = state["episodes"], state["rng"]
        self.returns = deque(state["returns"], maxlen=100)
        self.pipeline.__dict__.update(state["pipeline"])
        self.pipeline.buffer = state["buffer"]
        self.pipeline.shaper.__dict__.update(state["shaper"])
        self.pipeline.density.__dict__.update(state["density"])
        self.pipeline.reducer.net.load_state_dict(state["mixer"])
        self.pipeline.reducer._generator.set_state(state["mixer_rng"])
        self.pipeline.frozen.set_reward_normalization(*state.get("reward_moments", (0.0, 1.0)))
