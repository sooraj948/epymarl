"""Compare recurrent QMIX, individual MACA regularization and summed MACA credit."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from run_comparison import Job, ROOT, run_pool
from run_shared_suite import SEEDS, TASKS, preflight


CONFIGS = {"qmix": "qmix_lbf", "individual_aux": "maca_qmix_aux", "team_sum": "maca_qmix_sum"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=[t[0] for t in TASKS], default=[t[0] for t in TASKS])
    parser.add_argument("--methods", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--beta", type=float, default=.1)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="6k steps, one seed, ten evaluation episodes per variant")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tmux", action="store_true", help="Run in a new window of the existing epymarl-maca session")
    args = parser.parse_args(argv)
    if args.max_parallel < 1 or not math.isfinite(args.beta) or args.beta < 0:
        parser.error("Use positive parallelism and a finite nonnegative beta")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error("Use distinct seeds in [0, 2**32)")
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.methods)) != len(args.methods):
        parser.error("Tasks and methods must be distinct")
    if args.tmux and args.dry_run:
        parser.error("Use --dry-run separately from --tmux")
    return args


def stages_for(args, output):
    output = Path(output)
    stages = []
    for task, environment, budget in TASKS:
        if task not in args.tasks:
            continue
        jobs = []
        for method in args.methods:
            for seed in args.seeds[:1] if args.smoke else args.seeds:
                steps = 6000 if args.smoke else budget
                settings = {
                    "seed": seed, "env_args.key": environment, "env_args.time_limit": 50,
                    "common_reward": True, "reward_scalarisation": "sum", "t_max": steps,
                    "use_cuda": False, "test_nepisode": 10 if args.smoke else 100,
                    "test_interval": 6000 if args.smoke else 50000,
                    "log_interval": 1000 if args.smoke else 10000,
                    "runner_log_interval": 1000 if args.smoke else 10000,
                    "learner_log_interval": 1000 if args.smoke else 10000,
                    "save_model": not args.smoke, "save_model_interval": 100000,
                    "label": output.name, "name": f"{CONFIGS[method]}_{output.name}",
                    "local_results_path": str(output / task),
                }
                if method != "qmix":
                    settings["maca.beta"] = args.beta
                command = [sys.executable, "-u", str(ROOT / "src/main.py"),
                           f"--config={CONFIGS[method]}", "--env-config=gymma", "with"]
                command.extend(f"{key}={value}" for key, value in settings.items())
                jobs.append(Job(method, seed, command, steps))
        stages.append({"task": task, "environment": environment, "jobs": jobs})
    return stages


def launch_tmux(args):
    subprocess.run(["tmux", "has-session", "-t", "=epymarl-maca"], check=True)
    forwarded = ["--tasks", *args.tasks, "--methods", *args.methods, "--seeds", *map(str, args.seeds),
                 "--beta", str(args.beta), "--max-parallel", str(args.max_parallel)]
    if args.smoke:
        forwarded.append("--smoke")
    parent = ROOT / "results/qmix_suite"
    parent.mkdir(parents=True, exist_ok=True)
    fd, log = tempfile.mkstemp(prefix="launch_", suffix=".log", dir=parent)
    os.close(fd)
    command = shlex.join([sys.executable, "-u", str(Path(__file__).resolve()), *forwarded])
    window = subprocess.check_output([
        "tmux", "new-window", "-d", "-P", "-F", "#{window_id}", "-t", "epymarl-maca:",
        "-n", "qmix-lbf", "-c", str(ROOT), f"exec {command} >> {shlex.quote(log)} 2>&1",
    ], text=True).strip()
    print(f"Started in epymarl-maca window {window}; console log: {log}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.tmux:
        launch_tmux(args)
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parent = ROOT / "results/qmix_suite"
    if args.dry_run:
        output = parent / f"{stamp}_dry_run"
    else:
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=stamp + "_", dir=parent))
    stages = stages_for(args, output)
    manifest = [{**s, "jobs": [vars(j) for j in s["jobs"]]} for s in stages]
    print(f"{sum(len(s['jobs']) for s in stages)} jobs; output: {output}", flush=True)
    if args.dry_run:
        for stage in stages:
            for job in stage["jobs"]:
                print(shlex.join(job.command))
        return 0
    (output / "commands.json").write_text(json.dumps(manifest, indent=2) + "\n")
    preflight()
    for stage in stages:
        run_pool(stage["jobs"], output / stage["task"], args.max_parallel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
