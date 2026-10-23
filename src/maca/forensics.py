"""Opt-in, observational diagnostics for the MACA intrinsic-reward pipeline."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _float(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _nearest(query: np.ndarray, reference: np.ndarray) -> tuple[float, int]:
    if reference is None or len(reference) == 0:
        return float("nan"), -1
    distances = np.linalg.norm(reference - query, axis=1)
    index = int(np.argmin(distances))
    return float(distances[index]), index


def _leave_one_out_nearest(signatures: np.ndarray) -> np.ndarray:
    if signatures is None or len(signatures) < 2:
        return np.asarray([], dtype=float)
    distances = np.linalg.norm(
        signatures[:, None, :] - signatures[None, :, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _percentile(value: float, reference: np.ndarray) -> float:
    if not np.isfinite(value) or len(reference) == 0:
        return float("nan")
    return float(100.0 * np.mean(reference <= value))


class RewardForensics:
    """Collect Diagnostics A-C without changing model fitting or training RNG."""

    schema_version = 1

    def __init__(
        self,
        seed: int,
        n_agents: int,
        action_dim: int,
        exhaustive_donors: bool = True,
    ):
        self.seed = int(seed)
        self.n_agents = int(n_agents)
        self.action_dim = int(action_dim)
        self.exhaustive_donors = bool(exhaustive_donors)
        self.fit_id = 0
        self.ledger: List[Dict[str, Any]] = []
        self.prequential: List[Dict[str, Any]] = []
        self._ledger_by_key: Dict[tuple[int, int], Dict[str, Any]] = {}
        self._win_signatures = np.empty((0, 0))
        self._loss_signatures = np.empty((0, 0))
        self._factual_signatures = np.empty((0, 0))
        self._factual_loo = np.asarray([], dtype=float)
        self._factual_score_min = float("nan")
        self._factual_score_max = float("nan")

    def before_store(
        self,
        episode: int,
        joint_signature: np.ndarray,
        env_return: float,
        reducer,
        density,
    ) -> None:
        """Score a trajectory before it can enter a future scorer fit."""
        if not getattr(density, "_is_fitted", False):
            return
        prediction = _float(density.score(reducer.transform(joint_signature[None, :])))
        self.prequential.append(
            {
                "seed": self.seed,
                "episode": int(episode),
                "scorer_fit_id": self.fit_id,
                "prediction": prediction,
                "env_return": float(env_return),
                "error": prediction - float(env_return),
            }
        )

    def after_fit(self, buffer, reducer, density) -> None:
        """Cache the fitted model's factual support for deterministic OOD ranks."""
        self.fit_id += 1
        wins, losses = buffer.get_all_signatures()
        self._win_signatures = np.asarray(wins, dtype=float)
        self._loss_signatures = np.asarray(losses, dtype=float)
        available = [x for x in (self._win_signatures, self._loss_signatures) if len(x)]
        self._factual_signatures = (
            np.vstack(available) if available else np.empty((0, 0), dtype=float)
        )
        self._factual_loo = _leave_one_out_nearest(self._factual_signatures)
        if len(self._factual_signatures):
            scores = np.asarray(
                density.score(reducer.transform(self._factual_signatures)), dtype=float
            ).reshape(-1)
            self._factual_score_min = float(scores.min())
            self._factual_score_max = float(scores.max())

    def _counterfactual_ood(self, signatures: np.ndarray) -> Dict[str, Any]:
        nearest_distances: List[float] = []
        percentiles: List[float] = []
        nearest_ids: List[str] = []
        for signature in np.asarray(signatures):
            distance, index = _nearest(signature, self._factual_signatures)
            nearest_distances.append(distance)
            percentiles.append(_percentile(distance, self._factual_loo))
            if index < 0:
                nearest_ids.append("")
            elif index < len(self._win_signatures):
                nearest_ids.append(f"fit-{self.fit_id}:win-{index}")
            else:
                nearest_ids.append(
                    f"fit-{self.fit_id}:loss-{index - len(self._win_signatures)}"
                )
        return {
            "sampled_cf_nearest_factual_distance": nearest_distances,
            "sampled_cf_ood_percentile": percentiles,
            "sampled_cf_nearest_factual_id": nearest_ids,
            "sampled_cf_ood_percentile_median": (
                float(np.nanmedian(percentiles)) if percentiles else float("nan")
            ),
        }

    def _exhaustive_donor_metrics(
        self,
        agent_id: int,
        joint_signature: np.ndarray,
        factual_score: float,
        episode_trajectories,
        agent_signatures,
        encoder,
        config,
        reducer,
        density,
        shaper,
        sampled_reward: float,
    ) -> Dict[str, Any]:
        n_loss = len(shaper._X_loss) if shaper._X_loss is not None else 0
        if n_loss == 0:
            return {}
        signatures = np.vstack(
            [
                shaper._construct_counterfactual_signature_for_loss_idx(
                    episode_trajectories,
                    agent_id,
                    donor_index,
                    encoder,
                    config,
                    agent_signatures=agent_signatures,
                )
                for donor_index in range(n_loss)
            ]
        )
        scores = np.asarray(density.score(reducer.transform(signatures)), dtype=float).reshape(-1)
        rewards = factual_score - scores
        exhaustive_reward = float(rewards.mean())
        distance, loss_index = _nearest(signatures.mean(axis=0), self._loss_signatures)
        return {
            "exhaustive_donor_count": int(n_loss),
            "exhaustive_cf_mean": float(scores.mean()),
            "exhaustive_cf_std": float(scores.std()),
            "exhaustive_reward_mean": exhaustive_reward,
            "exhaustive_positive_fraction": float(np.mean(rewards > 0.0)),
            "sampled_minus_exhaustive_reward": float(sampled_reward - exhaustive_reward),
            "sampled_exhaustive_sign_disagreement": bool(
                np.sign(sampled_reward) != np.sign(exhaustive_reward)
            ),
            "exhaustive_cf_score_min": float(scores.min()),
            "exhaustive_cf_score_max": float(scores.max()),
            "exhaustive_mean_signature_nearest_loss_distance": distance,
            "exhaustive_mean_signature_nearest_loss_id": (
                f"fit-{self.fit_id}:loss-{loss_index}" if loss_index >= 0 else ""
            ),
        }

    def record_scored_episode(
        self,
        *,
        episode: int,
        ppo_batch_id: int,
        env_return: float,
        episode_trajectories,
        agent_signatures,
        joint_signature: np.ndarray,
        terminal_credit: np.ndarray,
        max_terminal_distance: float,
        running_return_100: float,
        dynamic_beta: float,
        normalized_rewards: np.ndarray,
        factual_score: float,
        mean_cf_scores: np.ndarray,
        raw_rewards: np.ndarray,
        shaper,
        reducer,
        density,
        encoder,
        config,
    ) -> None:
        details = shaper.last_reward_diagnostics or {}
        donor_indices = details.get("sampled_donor_indices", [])
        donor_ids = [f"fit-{self.fit_id}:loss-{idx}" for idx in donor_indices]
        loss_returns = [0.0 for _ in donor_indices]
        scale_before = details.get("running_scale_before")
        scale_after = details.get("running_scale_after")
        unclipped = details.get("unclipped_normalized_rewards", normalized_rewards)
        signs = np.sign(np.asarray(raw_rewards, dtype=float))
        opposing = bool(np.any(signs > 0) and np.any(signs < 0))
        support_masks = [np.abs(np.asarray(sig)) > 1e-12 for sig in agent_signatures]
        if len(support_masks) >= 2:
            union = np.logical_or(support_masks[0], support_masks[1]).sum()
            overlap = float(np.logical_and(support_masks[0], support_masks[1]).sum() / max(union, 1))
        else:
            overlap = float("nan")

        for agent_id in range(self.n_agents):
            cf_scores = np.asarray(details.get("sampled_cf_scores", [[]])[agent_id], dtype=float)
            cf_signatures = np.asarray(
                details.get("sampled_cf_signatures", [[]])[agent_id], dtype=float
            )
            action_histogram = np.bincount(
                [int(step[1]) for step in episode_trajectories[agent_id]],
                minlength=self.action_dim,
            ).astype(int)
            record: Dict[str, Any] = {
                "seed": self.seed,
                "episode": int(episode),
                "agent_id": int(agent_id),
                "ppo_batch_id": int(ppo_batch_id),
                "scorer_fit_id": self.fit_id,
                "episode_length": len(episode_trajectories[agent_id]),
                "env_return": float(env_return),
                "scorer_ready": True,
                "reward_eligible": True,
                "reward_applied": bool(dynamic_beta != 0.0),
                "factual_score": float(factual_score),
                "sampled_donor_ids": donor_ids,
                "sampled_donor_returns": loss_returns,
                "sampled_cf_scores": cf_scores.tolist(),
                "sampled_cf_mean": float(cf_scores.mean()),
                "sampled_cf_std": float(cf_scores.std()),
                "sampled_cf_min": float(cf_scores.min()),
                "sampled_cf_max": float(cf_scores.max()),
                "sampled_positive_fraction": float(np.mean((factual_score - cf_scores) > 0.0)),
                "raw_rint": float(raw_rewards[agent_id]),
                "running_scale_before": None if scale_before is None else float(scale_before),
                "running_scale_after": float(scale_after),
                "normalized_rint": float(normalized_rewards[agent_id]),
                "reward_clipped": bool(
                    not np.isclose(float(unclipped[agent_id]), float(normalized_rewards[agent_id]))
                ),
                "running_return_100": float(running_return_100),
                "dynamic_beta": float(dynamic_beta),
                "applied_episode_bonus": float(dynamic_beta * normalized_rewards[agent_id]),
                "injected_reward_per_step": float(
                    dynamic_beta * normalized_rewards[agent_id]
                    / max(len(episode_trajectories[agent_id]), 1)
                ),
                "terminal_proximity_credit": float(terminal_credit[agent_id]),
                "agent_terminal_distance": float(-terminal_credit[agent_id] * max_terminal_distance),
                "agent_action_histogram": action_histogram.tolist(),
                "agent_occupancy_support": int(support_masks[agent_id].sum()),
                "agent_teammate_overlap": overlap,
                "agents_have_opposing_rint_signs": opposing,
                "factual_score_outside_fit_range": bool(
                    factual_score < self._factual_score_min
                    or factual_score > self._factual_score_max
                ),
            }
            record.update(self._counterfactual_ood(cf_signatures))
            if dynamic_beta != 0.0 and self.exhaustive_donors:
                record.update(
                    self._exhaustive_donor_metrics(
                        agent_id,
                        joint_signature,
                        float(factual_score),
                        episode_trajectories,
                        agent_signatures,
                        encoder,
                        config,
                        reducer,
                        density,
                        shaper,
                        float(raw_rewards[agent_id]),
                    )
                )
            self.ledger.append(record)
            self._ledger_by_key[(int(episode), int(agent_id))] = record

    def add_gae(
        self,
        episode: int,
        agent_id: int,
        plain_advantages,
        shaped_advantages,
    ) -> None:
        record = self._ledger_by_key.get((int(episode), int(agent_id)))
        if record is None:
            return
        plain = np.asarray(plain_advantages, dtype=float)
        shaped = np.asarray(shaped_advantages, dtype=float)
        record.update(
            {
                "plain_adv_s0": float(plain[0]),
                "shaped_adv_s0": float(shaped[0]),
                "plain_adv_mean": float(plain.mean()),
                "shaped_adv_mean": float(shaped.mean()),
                "raw_adv_sign_flip_fraction": float(np.mean(np.sign(plain) != np.sign(shaped))),
            }
        )

    def add_ppo(self, agent_id: int, diagnostics: Dict[int, Dict[str, Any]]) -> None:
        for episode, values in diagnostics.items():
            record = self._ledger_by_key.get((int(episode), int(agent_id)))
            if record is not None:
                record.update(values)

    def episode_records(self, episode: int) -> List[Dict[str, Any]]:
        """Return all available per-agent forensic records for one episode."""
        return [
            self._ledger_by_key[(int(episode), agent_id)]
            for agent_id in range(self.n_agents)
            if (int(episode), agent_id) in self._ledger_by_key
        ]

    def trackers(self) -> Dict[str, Any]:
        return {
            "reward_forensics_schema_version": self.schema_version,
            "reward_forensics": self.ledger,
            "prequential_scorer": self.prequential,
        }
