"""Physical counterfactuals from observed initial boards and recorded joint actions.

Only ego behavior is resampled. Teammates replay requested actions, so their
positions and the food state can diverge under the native joint dynamics.
No agent policies or factual simulator snapshots are accepted by this module.
"""

from collections import Counter
import random
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch

from .trajectory_logging import capture_lbf_state


MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def reconstruct_initial_board(env, state, *, seed):
    """Initialize native LBF from public t=0 configuration, using a fresh RNG.

    reset() initializes wrappers and bookkeeping; its sampled layout is replaced.
    Food reward normalization is derived from the board, not a hidden snapshot.
    """
    base = env.unwrapped
    food = np.asarray(state["food_grid"], dtype=np.int32)
    players = state["players"]
    if state["step"] != 0 or state.get("game_over", False):
        raise ValueError("Action replay requires an initial, nonterminal board")
    if food.shape != base.field.shape or len(players) != len(base.players):
        raise ValueError("Initial board does not match the replay environment")
    if np.any(food < 0) or not np.any(food > 0):
        raise ValueError("Initial board must contain positive food levels")
    positions = [tuple(p["position"]) for p in players]
    if len(set(positions)) != len(positions):
        raise ValueError("Initial board contains overlapping agents")
    for position, player in zip(positions, players):
        if (len(position) != 2 or any(int(x) != x for x in position)
                or not (0 <= position[0] < food.shape[0] and 0 <= position[1] < food.shape[1])
                or food[position] != 0 or player["level"] <= 0):
            raise ValueError("Initial board contains an invalid player position or level")
    if state.get("max_episode_steps", base._max_episode_steps) != base._max_episode_steps:
        raise ValueError("Initial board episode limit differs from replay environment")

    env.reset(seed=seed)
    base.field = food.copy()
    base._food_spawned = float(food.sum())
    base.current_step = 0
    base._game_over = False
    for player, recorded, position in zip(base.players, players, positions):
        player.setup(position, int(recorded["level"]), base.field_size)
        player.reward = 0
        player.current_step = None
    base._gen_valid_moves()
    return base._make_gym_obs()


def _done(value):
    return bool(np.all(value))


def _rollout(env, initial_state, actions, factual_positions, ego, rng, config):
    obs = reconstruct_initial_board(env, initial_state, seed=int(rng.integers(2**32)))
    base = env.unwrapped
    trajectories = [[] for _ in range(config.n_agents)]
    states, rewards = [], []
    invalid = np.zeros(config.n_agents, dtype=int)
    blocked = np.zeros(config.n_agents, dtype=int)
    diverged = np.zeros(config.n_agents, dtype=int)
    distance = np.zeros(config.n_agents, dtype=int)
    first_divergence = None
    limit = min(len(actions), config.max_ep_len)
    reason = "record_exhausted" if len(actions) < config.max_ep_len else "configured_time_limit"
    for t in range(limit):
        states.append(capture_lbf_state(env))
        positions = np.asarray([p.position for p in base.players])
        distances = np.abs(positions - factual_positions[t]).sum(axis=1)
        diverged += distances > 0
        distance += distances
        if first_divergence is None and any(distances[j] > 0 for j in range(config.n_agents) if j != ego):
            first_divergence = t
        joint = list(actions[t])
        joint[ego] = int(rng.integers(config.action_dim))
        intended = []
        for j, (player, action) in enumerate(zip(base.players, joint)):
            valid = action in {a.value for a in base._valid_actions[player]}
            invalid[j] += not valid
            dy, dx = MOVES.get(action, (0, 0)) if valid else (0, 0)
            intended.append((player.position[0] + dy, player.position[1] + dx))
            trajectories[j].append((np.asarray(obs[j]).flatten().copy(), action))
        obs, reward, terminated, truncated, _ = env.step(joint)
        rewards.append(float(sum(reward)))
        for j, player in enumerate(base.players):
            blocked[j] += intended[j] != tuple(positions[j]) and intended[j] != tuple(player.position)
        if _done(terminated) or _done(truncated):
            reason = ("environment_time_limit" if base.current_step >= base._max_episode_steps
                      else "environment_terminal")
            break

    final_state = capture_lbf_state(env)
    length = len(states)
    # Preserve the existing finite-horizon score convention at configured/native
    # limits. A missing factual suffix alone is not an environment terminal.
    bootstrap = reason == "record_exhausted"
    diagnostics = {
        "length": length, "return": float(sum(rewards)), "stop_reason": reason,
        "bootstrap_final_value": bootstrap,
        "invalid_requested_actions": invalid.tolist(), "collision_blocked_moves": blocked.tolist(),
        "non_ego_position_divergence_fraction": [None if j == ego else float(diverged[j] / length)
                                                  for j in range(config.n_agents)],
        "non_ego_mean_manhattan_divergence": [None if j == ego else float(distance[j] / length)
                                               for j in range(config.n_agents)],
        "first_non_ego_divergence_step": first_divergence,
    }
    return {"trajectories": trajectories, "states": states, "final_state": final_state,
            "rewards": rewards, "diagnostics": diagnostics}


def action_replay_counterfactuals(config, factual_states, factual_trajectories, rng, samples):
    """Return per-ego native rollouts, isolated from the factual environment/RNG.

    The public API deliberately accepts neither policies nor simulator snapshots.
    """
    count = len(factual_states)
    if (not count or samples < 1 or len(factual_trajectories) != config.n_agents
            or any(len(traj) != count for traj in factual_trajectories)):
        raise ValueError("Action replay needs aligned, nonempty factual states and actions")
    actions = [[int(traj[t][1]) for traj in factual_trajectories] for t in range(count)]
    if any(not 0 <= action < config.action_dim for joint in actions for action in joint):
        raise ValueError("Factual action is outside the environment action set")
    factual_positions = np.asarray([[p["position"] for p in state["players"]]
                                    for state in factual_states])
    np_state, torch_state, python_state = np.random.get_state(), torch.random.get_rng_state(), random.getstate()
    start = perf_counter()
    env = None
    try:
        env = gym.make(config.env_name)
        obs = reconstruct_initial_board(env, factual_states[0], seed=int(rng.integers(2**32)))
        if any(not np.array_equal(np.asarray(obs[j]).flatten(), np.asarray(factual_trajectories[j][0][0]).flatten())
               for j in range(config.n_agents)):
            raise ValueError("Reconstructed initial observations differ from factual observations")
        rollouts = [[_rollout(env, factual_states[0], actions, factual_positions, ego, rng, config)
                     for _ in range(samples)] for ego in range(config.n_agents)]
    finally:
        if env is not None:
            env.close()
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        random.setstate(python_state)
    lengths = [r["diagnostics"]["length"] for group in rollouts for r in group]
    return rollouts, {
        "simulator_steps": sum(lengths), "elapsed_seconds": perf_counter() - start,
        "stop_reasons": dict(Counter(r["diagnostics"]["stop_reason"] for group in rollouts for r in group)),
        "samples": [[r["diagnostics"] for r in group] for group in rollouts],
    }
