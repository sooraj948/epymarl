"""Null controls for testing whether intrinsic rewards contain useful information."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PermutedIntrinsicRewardSchedule:
    """Pre-scaled per-agent bonuses assigned to fixed active episode slots."""

    rewards: np.ndarray
    source_episode: np.ndarray
    source_path: str
    source_sha256: str
    active_episodes: int


def _deranged_donors(active_episodes: np.ndarray, seed: int) -> np.ndarray:
    """Return a reproducible permutation with no episode retaining itself."""
    if len(active_episodes) < 2:
        raise ValueError("A shuffled-reward control requires at least two shaped episodes")

    rng = np.random.default_rng(seed)
    for _ in range(100):
        donors = rng.permutation(active_episodes)
        if np.all(donors != active_episodes):
            return donors

    # Deterministic fallback that is always a derangement for distinct entries.
    return np.roll(active_episodes, 1)


def load_permuted_intrinsic_schedule(
    source_path: str,
    *,
    n_episodes: int,
    n_agents: int,
    seed: int,
    env_name: str | None = None,
) -> PermutedIntrinsicRewardSchedule:
    """Reconstruct and permute the applied bonuses from a previous result file.

    ``critic_absorption`` stores ``delta_r = beta * r_int / T``. Multiplying
    it by the stored episode length recovers the already-scaled episode bonus.
    Whole agent vectors are shuffled among the original active episode slots,
    preserving timing, coverage, magnitude, signs, and agent coupling.
    """
    path = Path(source_path)
    raw_bytes = path.read_bytes()
    data = json.loads(raw_bytes)
    config = data.get("config", {})

    source_seed = config.get("seed")
    if source_seed is not None and int(source_seed) != int(seed):
        raise ValueError(
            f"Source seed {source_seed} does not match control-run seed {seed}; "
            "use paired seed files for a controlled comparison"
        )
    source_env = config.get("env_name")
    if env_name and source_env and source_env != env_name:
        raise ValueError(f"Source environment {source_env!r} does not match {env_name!r}")
    source_n_episodes = config.get("n_episodes")
    if source_n_episodes is not None and int(source_n_episodes) != int(n_episodes):
        raise ValueError(
            f"Source has {source_n_episodes} episodes but control run requests {n_episodes}"
        )

    entries = (data.get("trackers") or {}).get("critic_absorption", [])
    if not entries:
        raise ValueError("Source result has no critic_absorption records to shuffle")

    original = np.zeros((n_episodes, n_agents), dtype=np.float64)
    seen: set[tuple[int, int]] = set()
    for entry in entries:
        if len(entry) < 8:
            raise ValueError("critic_absorption entries must contain at least 8 fields")
        episode = int(entry[0])
        agent_id = int(entry[1])
        if not 0 <= episode < n_episodes:
            raise ValueError(f"Source episode {episode} is outside [0, {n_episodes})")
        if not 0 <= agent_id < n_agents:
            raise ValueError(f"Source agent {agent_id} is outside [0, {n_agents})")
        key = (episode, agent_id)
        if key in seen:
            raise ValueError(f"Duplicate critic_absorption record for episode/agent {key}")
        seen.add(key)
        delta_r = float(entry[2])
        episode_length = int(entry[7])
        original[episode, agent_id] = delta_r * episode_length

    active = np.flatnonzero(np.any(np.abs(original) > 1e-12, axis=1))
    donors = _deranged_donors(active, seed)

    permuted = np.zeros_like(original)
    permuted[active] = original[donors]
    source_episode = np.full(n_episodes, -1, dtype=np.int64)
    source_episode[active] = donors

    return PermutedIntrinsicRewardSchedule(
        rewards=permuted,
        source_episode=source_episode,
        source_path=str(path.resolve()),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        active_episodes=len(active),
    )
