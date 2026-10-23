import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_comparison as launcher


def test_seed_method_environment_and_override_routing(tmp_path):
    args = launcher.parse_args(["--max-parallel", "8", "--seeds", "1587", "2398",
                                "--env", "Foraging-10x10-3p-2f-coop-v3", "t_max=1234", "maca.beta=0.05"])
    jobs = launcher.jobs_for(args, "test", tmp_path)
    assert {(j.method, j.seed) for j in jobs} == {
        (m, s) for m in ["baseline", "maca", "coma"] for s in [1587, 2398]}
    for job in jobs:
        assert "env_args.key=lbforaging:Foraging-10x10-3p-2f-coop-v3" in job.command
        assert "t_max=1234" in job.command
        assert ("maca.beta=0.05" in job.command) == (job.method == "maca")
        if job.method == "coma":
            assert "--config=coma" in job.command
            # Keep stock COMA's sharing, recurrence and normalization settings.
            assert not any(s.startswith(("use_rnn=", "mac=", "standardise_rewards=")) for s in job.command)


@pytest.mark.parametrize("argv", [
    ["--max-parallel", "0"], ["--seeds", "1", "1"], ["--methods", "maca", "maca"],
    ["seed=1"], ["--env", "rware:example-v2"],
    ["--env", "lbforaging:Foraging-10x8-3p-2f-coop-v3"],
    ["batch_size_run=1"], ["--env", "one", "env_args.key=two"],
])
def test_invalid_launches_rejected(argv):
    with pytest.raises(SystemExit):
        launcher.parse_args(argv)


def test_other_gymma_environments_without_maca():
    args = launcher.parse_args(["--methods", "baseline", "coma", "--env", "rware:example-v2"])
    assert args.env == "rware:example-v2"
    assert launcher.parse_args(["--benchmark"]).common["t_max"] == "20000"
    assert launcher.parse_args(["--benchmark"]).common["save_model"] == "False"
    assert launcher.parse_args(["--env-name", "Foraging-5x5-2p-2f-coop-v3"]).env == "lbforaging:Foraging-5x5-2p-2f-coop-v3"


def test_v1_v2_ancestor_limits(tmp_path, monkeypatch):
    proc = tmp_path / "proc/self/cgroup"
    proc.parent.mkdir(parents=True)
    proc.write_text("5:cpu,cpuacct:/team/session\n6:memory:/team/session\n2:pids:/team/session\n0::/team/session\n")
    files = {
        "cpu/team/cpu.cfs_quota_us": "50000", "cpu/team/cpu.cfs_period_us": "100000",
        "memory/team/memory.limit_in_bytes": "536870912", "memory/team/memory.usage_in_bytes": "268435456",
        "pids/team/pids.max": "128", "pids/team/pids.current": "12",
        "team/session/cpu.max": "200000 100000", "team/session/memory.max": "max",
    }
    for relative, content in files.items():
        path = tmp_path / "sys/fs/cgroup" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    real_path = Path
    monkeypatch.setattr(launcher, "Path", lambda value: tmp_path / value.lstrip("/")
                        if str(value).startswith(("/proc/", "/sys/")) else real_path(value))
    limits = launcher.cgroup_limits()
    assert limits["cpu_quota_cores"] == .5
    assert limits["memory_limits"][0]["limit"] == 536870912
    assert limits["pid_limits"][0]["used"] == 12


def worker_command(events, marker, delay):
    code = f"""
import json,os,time
from pathlib import Path
path=Path({str(events)!r})
def record(kind):
 fd=os.open(path,os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
 os.write(fd,(json.dumps(dict(marker={marker!r},kind=kind,time=time.monotonic()))+'\\n').encode());os.close(fd)
record('start');time.sleep({delay!r});record('end')
"""
    return [sys.executable, "-c", code]


def test_pool_fills_slots_across_methods(tmp_path):
    events = tmp_path / "events.jsonl"
    jobs = [launcher.Job("baseline", 1, worker_command(events, "slow", .5), 100),
            launcher.Job("baseline", 2, worker_command(events, "fast", .1), 100),
            launcher.Job("maca", 1, worker_command(events, "maca", .1), 100)]
    summary = launcher.run_pool(jobs, tmp_path, 2, sample_interval=.05)
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    timestamps = {(row["marker"], row["kind"]): row["time"] for row in rows}
    assert timestamps["maca", "start"] < timestamps["slow", "end"]
    active = peak = 0
    for row in sorted(rows, key=lambda row: row["time"]):
        active += 1 if row["kind"] == "start" else -1
        peak = max(peak, active)
    assert active == 0 and peak <= 2
    assert summary["status"] == "completed" and len(summary["completed_jobs"]) == 3
    assert summary["budget_steps_per_second"] > 0


def test_failure_kills_worker_group_and_leaves_summary(tmp_path):
    ids = tmp_path / "ids.json"
    code = f"""
import json,os,subprocess,sys,time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
Path({str(ids)!r}).write_text(json.dumps([os.getpid(),child.pid]))
time.sleep(30)
"""
    jobs = [launcher.Job("baseline", 1, [sys.executable, "-c", code], 100),
            launcher.Job("coma", 1, [sys.executable, "-c", "import time;time.sleep(.4);raise SystemExit(7)"], 100),
            launcher.Job("maca", 1, [sys.executable, "-c", "print('should not start')"], 100)]
    with pytest.raises(RuntimeError, match="coma seed 1 failed"):
        launcher.run_pool(jobs, tmp_path, 2, sample_interval=.05)
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "failed"
    assert not (tmp_path / "maca_seed_1.log").exists()
    for pid in json.loads(ids.read_text()):
        stat = Path("/proc") / str(pid) / "stat"
        assert not stat.exists() or stat.read_text().split()[2] == "Z"


def test_dry_run_from_external_directory(tmp_path):
    before = set((ROOT / "results/comparisons").iterdir())
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_comparison.py"),
                             "--dry-run", "--seeds", "1587", "--max-parallel", "8"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "3 jobs, one shared pool of 3" in result.stdout
    assert "--config=coma" in result.stdout
    assert before == set((ROOT / "results/comparisons").iterdir())


def test_cancelled_pool_terminates_descendants(tmp_path):
    ids = tmp_path / "worker_ids.json"
    worker = f"""
import json,os,subprocess,sys,time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
Path({str(ids)!r}).write_text(json.dumps([os.getpid(),child.pid]))
time.sleep(30)
"""
    code = f"""
import signal,sys
sys.path.insert(0,{str(ROOT / 'scripts')!r})
from run_comparison import Job,run_pool
def cancel(*args): raise KeyboardInterrupt
signal.signal(signal.SIGTERM,cancel)
try: run_pool([Job('maca',1,[sys.executable,'-c',{worker!r}],100)],{str(tmp_path)!r},1,.05)
except KeyboardInterrupt: sys.exit(130)
"""
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 5
        while not ids.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert ids.exists()
        process.terminate()
        assert process.wait(timeout=8) == 130
        assert json.loads((tmp_path / "summary.json").read_text())["status"] == "cancelled"
        for pid in json.loads(ids.read_text()):
            stat = Path("/proc") / str(pid) / "stat"
            assert not stat.exists() or stat.read_text().split()[2] == "Z"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
