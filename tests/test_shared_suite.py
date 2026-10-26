import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_shared_suite as suite


def test_full_suite_protocol(tmp_path):
    args = suite.parse_args([])
    stages = suite.stages_for(args, tmp_path)
    assert args.max_parallel == 8
    assert [len(stage["jobs"]) for stage in stages] == [19, 8, 8]
    for index, stage in enumerate(stages):
        assert ("-coop-" in stage["environment"]) == (index == 0)
        for job in stage["jobs"]:
            assert job.seed in suite.SEEDS
            assert job.budget_steps == (4_000_000 if index == 2 else 2_000_000)
            assert "env_args.time_limit=50" in job.command
            assert "test_nepisode=100" in job.command
            # No common override changes any stock algorithm hyperparameter.
            assert not any(s.startswith(("entropy_coef=", "critic_type=", "use_rnn=",
                                         "standardise_rewards=", "hidden_dim=")) for s in job.command)
            if job.method == "mappo":
                assert "--config=mappo" in job.command
                assert index != 0 or job.seed != 1587
            elif job.method == "coma":
                assert "--config=coma" in job.command
                assert not any(s.startswith("maca.") for s in job.command)
            else:
                assert index == 0
                assert "--config=maca_mappo" in job.command
                assert any(f"maca.beta={beta}" in job.command for beta in suite.BETAS)
    assert sum(job.budget_steps for stage in stages for job in stage["jobs"]) == 86_000_000
    # Each beta has its own method name, preventing seed-log collisions.
    for stage in stages:
        keys = [(job.method, job.seed) for job in stage["jobs"]]
        assert len(keys) == len(set(keys))


def test_smoke_and_seed_selection(tmp_path):
    stages = suite.stages_for(suite.parse_args(["--smoke"]), tmp_path)
    assert [len(s["jobs"]) for s in stages] == [5, 2, 2]
    for stage in stages:
        for job in stage["jobs"]:
            assert job.seed == 1587 and job.budget_steps == 6000
            assert "save_model=False" in job.command
    selected = suite.stages_for(suite.parse_args(["--seeds", "2398", "3370"]), tmp_path)
    assert [len(s["jobs"]) for s in selected] == [10, 4, 4]


def test_task_registration_and_native_observations():
    suite.preflight()


def test_explicit_ten_by_ten_maca_only(tmp_path):
    args = suite.parse_args(["--tasks", "10x10_3p3f", "--methods", "maca"])
    stages = suite.stages_for(args, tmp_path)
    assert args.max_parallel == 8
    assert len(stages) == 1 and stages[0]["task"] == "10x10_3p3f"
    jobs = stages[0]["jobs"]
    assert len(jobs) == 12
    assert {job.seed for job in jobs} == set(suite.SEEDS)
    assert {job.method for job in jobs} == {"maca_beta_0p08", "maca_beta_0p1", "maca_beta_0p15"}
    for job in jobs:
        assert job.budget_steps == 4_000_000
        assert "--config=maca_mappo" in job.command
        assert "env_args.key=lbforaging:Foraging-10x10-3p-3f-v3" in job.command
        assert "env_args.time_limit=50" in job.command
        assert "use_cuda=False" in job.command


def test_tmux_forwards_task_and_method_selection(tmp_path, monkeypatch):
    calls = []
    args = suite.parse_args(["--tasks", "10x10_3p3f", "--methods", "maca", "--tmux"])
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(suite.subprocess, "run", lambda *a, **kw: None)
    def capture(command, **kwargs):
        calls.append(command)
        return "@42\n"
    monkeypatch.setattr(suite.subprocess, "check_output", capture)
    assert suite.launch_tmux(args) == 0
    assert len(calls) == 1
    command = calls[0]
    assert command[command.index("-n") + 1] == "maca-10x10"
    assert "--tasks 10x10_3p3f --methods maca" in command[-1]


def test_stages_run_sequentially_and_completed_resume(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(suite, "preflight", lambda: None)
    monkeypatch.setattr(suite.tempfile, "mkdtemp", lambda **kw: str(tmp_path))
    def fake_pool(jobs, output, parallel):
        calls.append(Path(output).name)
        Path(output).mkdir()
        (Path(output) / "summary.json").write_text(json.dumps({"status": "completed"}))
        assert parallel == 8
    monkeypatch.setattr(suite, "run_pool", fake_pool)
    assert suite.main([]) == 0
    assert calls == [task[0] for task in suite.TASKS]
    calls.clear()
    assert suite.main(["--resume", str(tmp_path)]) == 0
    assert not calls
    assert json.loads((tmp_path / "suite_status.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("args", [["--max-parallel", "0"], ["--seeds", "1", "1"],
                                  ["--tmux", "--dry-run"]])
def test_bad_arguments(args):
    with pytest.raises(SystemExit):
        suite.parse_args(args)
