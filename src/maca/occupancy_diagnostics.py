"""Distances between normalized state-action occupancy histograms."""

import numpy as np


METRICS = ("total_variation", "jensen_shannon_bits", "hellinger")


def pairwise_distances(left, right):
    """Return bounded distances for every pair of occupancy histograms."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Occupancies must be two matrices with matching bin counts")
    for values in (left, right):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Occupancies must be finite and nonnegative")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Each occupancy histogram must sum to one")
    p, q = left[:, None, :], right[None, :, :]
    midpoint = (p + q) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        p_term = np.where(p > 0, p * np.log2(p / midpoint), 0.0)
        q_term = np.where(q > 0, q * np.log2(q / midpoint), 0.0)
    return {
        "total_variation": 0.5 * np.sum(np.abs(p - q), axis=-1),
        "jensen_shannon_bits": 0.5 * np.sum(p_term + q_term, axis=-1),
        "hellinger": np.sqrt(np.maximum(0.0, 1.0 - np.sum(np.sqrt(p * q), axis=-1))),
    }


def compare_splices_to_oracles(splice_signatures, oracle_signatures, n_agents, per_agent_dim):
    """Expected splice/oracle distance and independent oracle/oracle baseline.

    Each splice_signatures[ego] matrix contains the actual scored splice inputs;
    each oracle_signatures[ego] matrix contains the native reset-rollout inputs.
    Their trailing goal context is deliberately excluded from these distances.
    """
    results = []
    for ego in range(n_agents):
        splice = np.asarray(splice_signatures[ego])
        oracle = np.asarray(oracle_signatures[ego])
        if oracle.shape[0] < 2:
            raise ValueError("At least two oracle rollouts are needed for a baseline")
        roles = {}
        for agent in range(n_agents):
            start, end = agent * per_agent_dim, (agent + 1) * per_agent_dim
            cross = pairwise_distances(splice[:, start:end], oracle[:, start:end])
            within = pairwise_distances(oracle[:, start:end], oracle[:, start:end])
            upper = np.triu_indices(len(oracle), k=1)
            roles["replaced" if agent == ego else f"partner_{agent}"] = {
                "splice_oracle": {metric: float(cross[metric].mean()) for metric in METRICS},
                "oracle_oracle": {metric: float(within[metric][upper].mean()) for metric in METRICS},
            }
        results.append(roles)
    return results


def compare_mean_splices_to_oracles(splice_signatures, oracle_signatures,
                                    n_agents, per_agent_dim):
    """Compare mean of scored splices with one mean of reset-oracle occupancies."""
    results = []
    for ego in range(n_agents):
        splice = np.asarray(splice_signatures[ego], dtype=np.float64)
        oracle = np.asarray(oracle_signatures[ego], dtype=np.float64)
        if splice.shape[0] != 16 or oracle.shape[0] != 8:
            raise ValueError("Mean diagnostic requires 16 scored splices and 8 oracle rollouts")
        roles = {}
        for agent in range(n_agents):
            start, end = agent * per_agent_dim, (agent + 1) * per_agent_dim
            splice_mean = splice[:, start:end].mean(axis=0, keepdims=True)
            oracle_mean = oracle[:, start:end].mean(axis=0, keepdims=True)
            distances = pairwise_distances(splice_mean, oracle_mean)
            roles["replaced" if agent == ego else f"partner_{agent}"] = {
                metric: float(distances[metric][0, 0]) for metric in METRICS
            }
        results.append(roles)
    return results
