from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_shared_suite import align_metric, metric_series


def test_duplicate_steps_keep_last_observation_and_nonfinite_rejected():
    steps, values = metric_series({"return_mean": {"steps": [3, 1, 3], "values": [.3, .1, .4]}}, "return_mean")
    np.testing.assert_array_equal(steps, [1, 3])
    np.testing.assert_array_equal(values, [.1, .4])
    with pytest.raises(ValueError, match="Nonfinite"):
        metric_series({"return_mean": {"steps": [1], "values": [float("nan")]}}, "return_mean")


def test_alignment_common_support_mean_and_sample_standard_deviation():
    runs = []
    for method in ["mappo", "coma"]:
        for seed in range(4):
            steps = [seed, seed + 10]
            values = [.1 * x + .05 * seed + (method == "coma") for x in steps]
            runs.append(SimpleNamespace(method=method, metrics={"return_mean": {"steps": steps, "values": values}}))
    grid, groups = align_metric(runs, "return_mean", points=8)
    np.testing.assert_array_equal(grid, np.arange(3, 11))
    np.testing.assert_allclose(groups["mappo"]["mean"], .1 * grid + .075)
    np.testing.assert_allclose(groups["coma"]["mean"], .1 * grid + 1.075)
    expected_std = np.std([0., .05, .1, .15], ddof=1)
    np.testing.assert_allclose(groups["mappo"]["std"], expected_std)
    assert groups["mappo"]["n"] == 4
