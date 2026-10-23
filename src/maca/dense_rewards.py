"""Hand-engineered dense reward baselines for cooperative LBF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DenseReward:
    """Compute transition rewards from flat LBF observations.

    The returned value is a single shared team reward.  For raw distance this
    is the sum of the per-agent proximity rewards described in the diagnostic.
    """

    mode: str
    grid_size: int
    n_agents: int
    n_food: int
    gamma: float = 0.99
    alpha: float = 1.0
    raw_alpha: float = 0.1

    terminal_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"none", "pbrs", "raw_distance", "terminal_distance"}:
            raise ValueError(f"Unknown dense reward mode: {self.mode}")
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if self.n_agents < 1 or self.n_food < 1:
            raise ValueError("n_agents and n_food must be positive")

    @property
    def max_distance(self) -> float:
        return float(2 * (self.grid_size - 1))

    def _joint_positions(self, observations: Sequence[np.ndarray]):
        if len(observations) != self.n_agents:
            raise ValueError(
                f"Expected {self.n_agents} observations, got {len(observations)}"
            )

        # Every fully-observable LBF observation contains the same foods.  The
        # first observation lists its observer first, while each observation's
        # self slot provides an ordering-independent source for player position.
        first = np.asarray(observations[0]).reshape(-1)
        expected_dim = 3 * (self.n_food + self.n_agents)
        if first.size < expected_dim:
            raise ValueError(
                f"Expected an LBF observation with at least {expected_dim} values, "
                f"got {first.size}"
            )

        foods = []
        for food_idx in range(self.n_food):
            y, x, level = first[3 * food_idx : 3 * food_idx + 3]
            if level > 0 and y >= 0 and x >= 0:
                foods.append((float(y), float(x)))

        self_offset = 3 * self.n_food
        players = []
        for observation in observations:
            flat = np.asarray(observation).reshape(-1)
            if flat.size < self_offset + 2:
                raise ValueError("LBF observation is missing the self-position slot")
            players.append((float(flat[self_offset]), float(flat[self_offset + 1])))

        return players, foods

    @staticmethod
    def _manhattan(left, right) -> float:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    def potential(self, observations: Sequence[np.ndarray]) -> float:
        players, foods = self._joint_positions(observations)
        if not foods:
            return 0.0
        best_team_distance = min(
            sum(self._manhattan(player, food) for player in players)
            for food in foods
        )
        return -best_team_distance / (self.n_agents * self.max_distance)

    def transition_reward(
        self,
        observations: Sequence[np.ndarray],
        next_observations: Sequence[np.ndarray],
        done: bool,
    ) -> float:
        if self.mode == "none":
            return 0.0
        if self.mode == "pbrs":
            current_potential = self.potential(observations)
            next_potential = 0.0 if done else self.potential(next_observations)
            return self.alpha * (
                self.gamma * next_potential - current_potential
            )

        if self.mode == "terminal_distance":
            return 0.0

        players, foods = self._joint_positions(next_observations)
        if not foods:
            return 0.0
        normalized_distances = (
            min(self._manhattan(player, food) for food in foods) / self.max_distance
            for player in players
        )
        return -self.raw_alpha * sum(normalized_distances)

    def terminal_reward(
        self,
        observations: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Per-agent terminal credit: negative Manhattan distance to nearest food.

        Returns an array of shape (n_agents,).  Non-zero only in
        ``terminal_distance`` mode.
        """
        if self.mode != "terminal_distance":
            return np.zeros(self.n_agents)
        credit = self.diagnostic_terminal_credit(observations)
        return self.terminal_alpha * credit

    def diagnostic_terminal_credit(
        self,
        observations: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Per-agent negative normalized Manhattan distance to nearest food.

        Always computed regardless of mode — used for online comparison
        against MACA's R_int.  Returns shape (n_agents,) in [-1, 0].
        """
        players, foods = self._joint_positions(observations)
        if not foods:
            return np.zeros(self.n_agents)
        result = np.empty(self.n_agents)
        for i, player in enumerate(players):
            nearest = min(self._manhattan(player, food) for food in foods)
            result[i] = -nearest / self.max_distance
        return result
