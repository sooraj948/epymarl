"""Selective full-trajectory logging for non-positive policy-gradient cosine."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch


ACTION_NAMES = ("NONE", "NORTH", "SOUTH", "WEST", "EAST", "LOAD")


def to_jsonable(value: Any) -> Any:
    """Convert training values to strict JSON (including non-finite floats)."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def action_name(action: int) -> str:
    action = int(action)
    return ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else f"UNKNOWN_{action}"


def capture_lbf_state(env) -> Dict[str, Any]:
    """Capture a renderer-independent snapshot of an LBF environment."""
    base = env.unwrapped
    valid_actions = getattr(base, "_valid_actions", {})
    players = []
    for agent_id, player in enumerate(base.players):
        valid = valid_actions.get(player, ())
        players.append(
            {
                "agent_id": agent_id,
                "position": list(player.position),
                "level": int(player.level),
                "score": float(player.score),
                "last_reward": float(player.reward),
                "valid_actions": sorted(
                    (getattr(item, "name", str(item)) for item in valid),
                    key=lambda name: ACTION_NAMES.index(name) if name in ACTION_NAMES else 999,
                ),
            }
        )
    return {
        "step": int(base.current_step),
        "food_grid": np.asarray(base.field, dtype=int).tolist(),
        "players": players,
        "game_over": bool(getattr(base, "_game_over", False)),
    }


def gradient_flip_agents(
    diagnostics_by_agent: Mapping[int, Mapping[int, Mapping[str, Any]]],
    episode: int,
) -> list[int]:
    """Return agents whose finite pre-update cosine is non-positive."""
    affected = []
    for agent_id, by_episode in diagnostics_by_agent.items():
        values = by_episode.get(int(episode), {})
        cosine = values.get("preupdate_gradient_cosine")
        if cosine is not None and np.isfinite(cosine) and float(cosine) <= 0.0:
            affected.append(int(agent_id))
    return sorted(affected)


class GradientFlipTrajectoryLogger:
    """Buffer one PPO batch and persist only episodes meeting the trigger."""

    schema_version = 1

    def __init__(self, output_dir: str, seed: int):
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.pending: Dict[int, Dict[str, Any]] = {}
        self.index = []

    def retain(self, episode: int, payload: Dict[str, Any]) -> None:
        self.pending[int(episode)] = payload

    def flush_batch(self, diagnostics_by_agent, forensic_records_for_episode) -> None:
        """Write triggered episodes, then release every trajectory in the batch."""
        for episode, payload in self.pending.items():
            forensic_records = forensic_records_for_episode(episode)
            # Diagnostic A exposes the cosine only for scorer-eligible episodes;
            # keep this qualitative sample set aligned with that ledger.
            if not forensic_records:
                continue
            affected = gradient_flip_agents(diagnostics_by_agent, episode)
            if not affected:
                continue

            per_agent_diagnostics = {
                str(agent_id): diagnostics_by_agent[agent_id].get(episode, {})
                for agent_id in diagnostics_by_agent
            }
            document = {
                "schema_version": self.schema_version,
                "trigger": {
                    "metric": "preupdate_gradient_cosine",
                    "condition": "finite cosine <= 0",
                    "affected_agents": affected,
                    "per_agent_diagnostics": per_agent_diagnostics,
                },
                "reward_forensics": forensic_records,
                **payload,
            }
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"episode_{episode}.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(to_jsonable(document), handle, indent=2, allow_nan=False)

            cosines = {
                str(agent_id): to_jsonable(
                    per_agent_diagnostics[str(agent_id)].get(
                        "preupdate_gradient_cosine"
                    )
                )
                for agent_id in diagnostics_by_agent
            }
            self.index.append(
                {
                    "episode": episode,
                    "affected_agents": affected,
                    "preupdate_gradient_cosine": cosines,
                    "path": os.path.abspath(path),
                }
            )
        self.pending.clear()


class StratifiedTrajectorySampler:
    """Keep at most one example per category in each of ten training windows."""

    schema_version = 1
    categories = ("win", "loss", "negative_cosine")

    def __init__(self, output_dir: str, seed: int, total_episodes: int):
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.total_episodes = int(total_episodes)
        self.rng = np.random.default_rng(seed)
        self.start_episode = None
        self.pending = {}
        self.index = []
        self.counts = {category: 0 for category in self.categories}

    def activate(self, episode: int) -> None:
        if self.start_episode is None:
            self.start_episode = int(episode)

    def window(self, episode: int) -> int:
        span = max(1, self.total_episodes - self.start_episode)
        return min(9, 10 * (int(episode) - self.start_episode) // span)

    def should_capture(self, episode: int) -> bool:
        if self.start_episode is None:
            return False
        window = self.window(episode)
        if all(self.counts[category] >= window + 1 for category in self.categories):
            return False
        end = self.start_episode + math.ceil((window + 1) * (self.total_episodes - self.start_episode) / 10)
        remaining = max(1, end - episode)
        # A rising coin probability gives each window multiple chances while
        # ensuring its last eligible episode is inspected.
        return self.rng.random() < min(1.0, 12.0 / remaining)

    def retain(self, episode: int, payload: Dict[str, Any]) -> None:
        self.pending[int(episode)] = payload

    def flush_batch(self, diagnostics_by_agent) -> None:
        for episode, payload in self.pending.items():
            window = self.window(episode)
            summary = payload["episode_summary"]
            outcome = "win" if summary["environment_return"] > 0 else "loss"
            affected = [
                agent_id for agent_id, by_episode in diagnostics_by_agent.items()
                if (cosine := by_episode.get(episode, {}).get("preupdate_gradient_cosine"))
                is not None and np.isfinite(cosine) and cosine < 0
            ]
            category = "negative_cosine" if affected else outcome
            selected = [category] if self.counts[category] < window + 1 else []
            if not selected:
                continue
            for category in selected:
                self.counts[category] += 1
            document = {
                "schema_version": self.schema_version,
                "sample_categories": selected,
                "training_window": window,
                "trigger": {
                    "metric": "preupdate_gradient_cosine",
                    "condition": "finite cosine < 0" if affected else "none",
                    "affected_agents": affected,
                    "per_agent_diagnostics": {
                        str(i): by_episode.get(episode, {})
                        for i, by_episode in diagnostics_by_agent.items()
                    },
                },
                **payload,
            }
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"episode_{episode}.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(to_jsonable(document), handle, indent=2, allow_nan=False)
            self.index.append({"episode": episode, "categories": selected,
                               "training_window": window, "path": os.path.abspath(path)})
        self.pending.clear()
