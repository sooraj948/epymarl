"""Run shared MAPPO, three no-replay MACA betas, and stock COMA on three LBF tasks."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile

from run_comparison import Job, ROOT, print_resources, resources, run_pool


SEEDS = (1587, 2398, 2875, 3370)
BETAS = (0.08, 0.1, 0.15)
TASKS = (
    ("8x8_2p2f_coop", "lbforaging:Foraging-8x8-2p-2f-coop-v3", 2_000_000),
    ("8x8_3p2f", "lbforaging:Foraging-8x8-3p-2f-v3", 2_000_000),
    ("10x10_3p3f", "lbforaging:Foraging-10x10-3p-3f-v3", 4_000_000),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--tasks", nargs="+", choices=[task[0] for task in TASKS],
                        help="Select task groups; defaults to all three")
    parser.add_argument("--methods", nargs="+", choices=["mappo", "coma", "maca"],
                        help="Explicit methods on selected tasks; maca includes all three betas")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="6,000 steps per variant/task, one seed, ten evaluation episodes, no checkpoints")
    parser.add_argument("--resume", type=Path, help="Resume a suite directory, skipping completed task groups")
    parser.add_argument("--tmux", action="store_true",
                        help="Start in a new detached window of the existing epymarl-maca session")
    args = parser.parse_args(argv)
    if args.max_parallel < 1 or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error("Use positive parallelism and distinct seeds in [0, 2**32)")
    if args.dry_run and args.tmux:
        parser.error("Use --dry-run separately from --tmux")
    if args.resume:
        args.resume = args.resume.resolve()
        if not (args.resume / "commands.json").is_file():
            parser.error("--resume must name an existing suite directory containing commands.json")
    return args


def stages_for(args, output):
    output = Path(output)
    stages = []
    seeds = args.seeds[:1] if args.smoke else args.seeds
    for index, (task, environment, budget) in enumerate(TASKS):
        if args.tasks and task not in args.tasks:
            continue
        variants = [("mappo", "mappo", None), ("coma", "coma", None)]
        if index == 0 or (args.methods and "maca" in args.methods):
            variants += [("maca_beta_" + str(beta).replace(".", "p"), "maca_mappo", beta) for beta in BETAS]
        if args.methods:
            variants = [variant for variant in variants if
                        ("maca" if variant[2] is not None else variant[0]) in args.methods]
        jobs = []
        for method, config, beta in variants:
            for seed in seeds:
                # The historical successful 1587 result exists only for the first MAPPO task.
                if index == 0 and method == "mappo" and seed == 1587 and not args.smoke:
                    continue
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
                    "label": output.name, "name": f"{method}_{output.name}",
                    "local_results_path": str(output / task),
                }
                if beta is not None:
                    settings["maca.beta"] = beta
                command = [sys.executable, "-u", str(ROOT / "src/main.py"),
                           f"--config={config}", "--env-config=gymma", "with"]
                command += [f"{key}={value}" for key, value in settings.items()]
                jobs.append(Job(method, seed, command, steps))
        stages.append({"task": task, "environment": environment, "jobs": jobs})
    return stages


def serialized_stages(stages):
    return [{**stage, "jobs": [vars(job) for job in stage["jobs"]]} for stage in stages]


def preflight():
    """Verify exact registered task variants and full-sight observation reconstruction."""
    import gymnasium as gym
    sys.path.insert(0, str(ROOT / "src"))
    from maca.config import MACAConfig
    from modules.maca_adapter import state_from_observations
    for _, key, _ in TASKS:
        env = gym.make(key)
        try:
            config = MACAConfig.from_env_name(key)
            assert env.unwrapped.sight >= config.grid_size, f"Partial sight: {key}"
            assert bool(env.unwrapped.force_coop) == ("-coop-" in key), f"Cooperation mismatch: {key}"
            assert env.unwrapped._max_episode_steps == 50, f"Native horizon mismatch: {key}"
            observations, _ = env.reset(seed=SEEDS[0])
            state_from_observations(observations, config)
        finally:
            env.close()


def launch_tmux(args):
    """Use a fresh window; never type into or interrupt an existing pane."""
    subprocess.run(["tmux", "has-session", "-t", "=epymarl-maca"], check=True)
    forwarded = ["--max-parallel", str(args.max_parallel), "--seeds", *map(str, args.seeds)]
    if args.smoke:
        forwarded.append("--smoke")
    if args.resume:
        forwarded += ["--resume", str(args.resume)]
    if args.tasks:
        forwarded += ["--tasks", *args.tasks]
    if args.methods:
        forwarded += ["--methods", *args.methods]
    launch_dir = ROOT / "results/shared_suite"
    launch_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # A permanent console log makes the outcome visible even after tmux closes the window.
    fd, log_name = tempfile.mkstemp(prefix=f"launch_{stamp}_", suffix=".log", dir=launch_dir)
    os.close(fd)
    command = shlex.join([sys.executable, "-u", str(Path(__file__).resolve()), *forwarded])
    shell_command = f"exec {command} >> {shlex.quote(log_name)} 2>&1"
    window_name = "maca-10x10" if args.tasks == ["10x10_3p3f"] and args.methods == ["maca"] else "shared-suite"
    window = subprocess.check_output([
        "tmux", "new-window", "-d", "-P", "-F", "#{window_id}",
        "-t", "epymarl-maca:", "-n", window_name, "-c", str(ROOT), shell_command,
    ], text=True).strip()
    print(f"Started in epymarl-maca window {window}; console log: {log_name}", flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.tmux:
        try:
            return launch_tmux(args)
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"Could not start in the existing tmux session: {error}", file=sys.stderr)
            return 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parent = ROOT / "results/shared_suite"
    if args.resume:
        output = args.resume
    elif args.dry_run:
        output = parent / f"{stamp}_dry_run"
    else:
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=stamp + "_", dir=parent))
    stages = stages_for(args, output)
    manifest = serialized_stages(stages)
    count = sum(len(stage["jobs"]) for stage in stages)
    print(f"{count} jobs; sequential task groups; at most {args.max_parallel} simultaneous trainers", flush=True)
    print(f"Suite output: {output}", flush=True)
    if args.dry_run:
        for stage in stages:
            print(f"\n{stage['task']}: {len(stage['jobs'])} jobs", flush=True)
            for job in stage["jobs"]:
                print(shlex.join(job.command), flush=True)
        return 0
    command_file = output / "commands.json"
    if args.resume:
        if json.loads(command_file.read_text()) != manifest:
            raise ValueError("Resume arguments do not match the saved suite; use its original seeds and smoke setting")
    else:
        command_file.write_text(json.dumps(manifest, indent=2) + "\n")
    preflight()
    print_resources(resources(), args.max_parallel, 10)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def cancel(signum, frame):
        raise KeyboardInterrupt
    for sig in previous:
        signal.signal(sig, cancel)
    state = {"status": "running", "completed_tasks": [], "active_task": None}
    state_file = output / "suite_status.json"
    def record():
        state_file.write_text(json.dumps(state, indent=2) + "\n")
    try:
        for stage in stages:
            stage_output = output / stage["task"]
            summary_path = stage_output / "summary.json"
            if args.resume and summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if summary["status"] == "completed":
                    print(f"Already completed: {stage['task']}", flush=True)
                    state["completed_tasks"].append(stage["task"])
                    continue
                raise RuntimeError(
                    f"Incomplete task group at {stage_output}. Move that directory aside before resuming; "
                    "completed task groups will be preserved and this group rerun in full."
                )
            state["active_task"] = stage["task"]
            record()
            print(f"Starting task group {stage['task']} ({len(stage['jobs'])} jobs)", flush=True)
            run_pool(stage["jobs"], stage_output, args.max_parallel)
            state["completed_tasks"].append(stage["task"])
        state.update(status="completed", active_task=None)
        record()
    except KeyboardInterrupt:
        state["status"] = "cancelled"
        record()
        return 130
    except (OSError, RuntimeError) as error:
        state["status"] = "failed"
        state["error"] = str(error)
        record()
        print(str(error), file=sys.stderr)
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(f"All {count} jobs completed. Results: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
