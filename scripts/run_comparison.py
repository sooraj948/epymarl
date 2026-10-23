"""A bounded process pool and resource report for EPyMARL experiment runs."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

import psutil
import yaml


ROOT = Path(__file__).resolve().parents[1]
METHODS = {"baseline": "maca_baseline_ns", "maca": "maca_mappo_ns", "coma": "coma"}
DEFAULT_ENV = "lbforaging:Foraging-8x8-2p-2f-coop-v3"


def read_number(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def cgroup_limits():
    """Read ancestor limits in both v1 and v2, including this cluster's v1 layout."""
    quotas, memory, pids = [], [], []
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        _, controllers, relative = line.split(":", 2)
        mounts = ([(Path("/sys/fs/cgroup"), "v2")] if not controllers else
                  [(Path("/sys/fs/cgroup") / name, name) for name in controllers.split(",")])
        for mount, controller in mounts:
            current = mount / relative.lstrip("/")
            while current == mount or mount in current.parents:
                if controller == "v2":
                    try:
                        quota, period = (current / "cpu.max").read_text().split()
                        if quota != "max":
                            quotas.append(int(quota) / int(period))
                    except (OSError, ValueError, ZeroDivisionError):
                        pass
                    pairs = [("memory.max", "memory.current", memory), ("pids.max", "pids.current", pids)]
                else:
                    if controller == "cpu":
                        quota = read_number(current / "cpu.cfs_quota_us")
                        period = read_number(current / "cpu.cfs_period_us")
                        if quota is not None and quota > 0 and period:
                            quotas.append(quota / period)
                    pairs = []
                    if controller == "memory":
                        pairs.append(("memory.limit_in_bytes", "memory.usage_in_bytes", memory))
                    if controller == "pids":
                        pairs.append(("pids.max", "pids.current", pids))
                for maximum, usage, destination in pairs:
                    limit, used = read_number(current / maximum), read_number(current / usage)
                    if limit is not None and 0 < limit < 2**60:
                        destination.append({"limit": limit, "used": used, "path": str(current)})
                if current == mount:
                    break
                current = current.parent
    return {"cpu_quota_cores": min(quotas) if quotas else None,
            "memory_limits": memory, "pid_limits": pids}


def resources():
    limits = cgroup_limits()
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    scheduler = {}
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "PBS_NP", "NSLOTS"):
        value = os.environ.get(name, "")
        if value.isdigit():
            scheduler[name] = int(value)
    candidates = [float(affinity)] + list(scheduler.values())
    if limits["cpu_quota_cores"] is not None:
        candidates.append(limits["cpu_quota_cores"])
    vm = psutil.virtual_memory()
    available = [vm.available] + [max(0, item["limit"] - item["used"])
                                 for item in limits["memory_limits"] if item["used"] is not None]
    swap = psutil.swap_memory()
    return {**limits, "affinity_logical_cpus": affinity,
            "host_physical_cores": psutil.cpu_count(logical=False),
            "effective_cpu_cores": min(candidates), "scheduler_cpu_limits": scheduler,
            "host_available_bytes": vm.available, "memory_headroom_bytes": min(available),
            "host_load_average": list(os.getloadavg()),
            "host_swap_in_bytes": swap.sin, "host_swap_out_bytes": swap.sout}


def print_resources(snapshot, parallel, env_workers):
    quota = snapshot["cpu_quota_cores"]
    print(f"CPU affinity: {snapshot['affinity_logical_cpus']} logical CPUs; "
          f"host physical cores: {snapshot['host_physical_cores']}; "
          f"effective detected ceiling: {snapshot['effective_cpu_cores']:g}", flush=True)
    print(f"CPU quota: {quota if quota is not None else 'none detected'}; "
          f"scheduler limits: {snapshot['scheduler_cpu_limits'] or 'none detected'}", flush=True)
    print(f"Available memory headroom: {snapshot['memory_headroom_bytes'] / 1024**3:.1f} GiB; "
          f"host load average: {snapshot['host_load_average']}", flush=True)
    print(f"Pool: at most {parallel} trainers + {parallel * env_workers} environment workers. "
          "Shared host capacity is not an exclusive reservation.", flush=True)


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    overrides = [s for s in argv if "=" in s and not s.startswith("--")]
    options = [s for s in argv if s not in overrides]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1587, 2398, 2875, 3370])
    parser.add_argument("--methods", choices=list(METHODS), nargs="+", default=list(METHODS))
    parser.add_argument("--max-parallel", type=int, default=4,
                        help="Shared concurrent training-job limit, across all methods (default: 4)")
    parser.add_argument("--env", "--env-name", dest="env",
                        help="Gymma task ID, e.g. lbforaging:Foraging-10x10-3p-2f-coop-v3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-resources", action="store_true", help="Report detected CPU/memory limits and exit")
    parser.add_argument("--benchmark", action="store_true",
                        help="Short 20,000-step runs, no checkpoints; compare summaries at parallelism 4 and 8")
    args = parser.parse_args(options)
    if args.max_parallel < 1 or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error("Use positive --max-parallel and distinct seeds in [0, 2**32)")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Methods must be distinct")
    common, maca = {}, {}
    fixed = {"seed", "name", "learner", "label", "evaluate", "checkpoint_path", "load_step", "save_replay", "local_results_path"}
    for override in overrides:
        key, value = override.split("=", 1)
        if not key or key in fixed:
            parser.error(f"The launcher fixes {key}; use src/main.py for individual/evaluation runs")
        (maca if key.startswith("maca.") else common)[key] = value
    if args.env and "env_args.key" in common and args.env != common["env_args.key"]:
        parser.error("Specify the environment with --env or env_args.key, without conflicting values")
    args.env = args.env or common.pop("env_args.key", DEFAULT_ENV)
    if args.env.startswith("Foraging-"):
        args.env = "lbforaging:" + args.env
    if "maca" in args.methods and not re.fullmatch(
            r"lbforaging:Foraging-(\d+)x\1-\d+p-\d+f-coop-v3", args.env):
        parser.error("MACA supports square full-sight cooperative LBF v3 tasks; "
                     "use --methods baseline coma for other gymma environments")
    defaults = {
        "t_max": "20000" if args.benchmark else "1000000", "use_cuda": "False",
        "common_reward": "True", "reward_scalarisation": "sum", "env_args.time_limit": "50",
        "test_nepisode": "20", "test_interval": "10000", "log_interval": "5000",
        "runner_log_interval": "5000", "learner_log_interval": "5000",
        "save_model": "False" if args.benchmark else "True", "save_model_interval": "100000",
    }
    args.common = {**defaults, **common, "env_args.key": args.env}
    args.maca = maca
    for method in args.methods:
        config = {}
        for filename in ("default.yaml", "envs/gymma.yaml", f"algs/{METHODS[method]}.yaml"):
            config.update(yaml.safe_load((ROOT / "src/config" / filename).read_text()))
        config.update({k: yaml.safe_load(v) for k, v in args.common.items() if "." not in k})
        if method == "maca" and not config["buffer_size"] == config["batch_size"] == config["batch_size_run"]:
            parser.error("MACA needs buffer_size=batch_size=batch_size_run")
        if int(config["batch_size_run"]) < 1 or int(config["t_max"]) < 1:
            parser.error("Use positive batch_size_run and t_max")
    args.env_workers = int(config["batch_size_run"])
    return args


@dataclass
class Job:
    method: str
    seed: int
    command: list
    budget_steps: int


def jobs_for(args, run_id, output):
    jobs = []
    for method in args.methods:
        config = METHODS[method]
        for seed in args.seeds:
            settings = {**args.common, "seed": str(seed), "label": run_id,
                        "name": f"{config}_{run_id}", "local_results_path": str(output)}
            if method == "maca":
                settings.update(args.maca)
            command = [sys.executable, "-u", str(ROOT / "src/main.py"),
                       f"--config={config}", "--env-config=gymma", "with"]
            command += [f"{key}={value}" for key, value in settings.items()]
            jobs.append(Job(method, seed, command, int(args.common["t_max"])))
    return jobs


def tree_memory(pid):
    """PSS avoids counting fork-shared PyTorch pages once per environment worker."""
    rss, pss = 0, 0
    try:
        parent = psutil.Process(pid)
        processes = [parent] + parent.children(recursive=True)
    except psutil.Error:
        return rss, pss
    for process in processes:
        try:
            mem = process.memory_full_info()
            rss += mem.rss
            pss += getattr(mem, "pss", mem.rss)
        except (psutil.Error, OSError):
            pass
    return rss, pss


def stop_processes(active):
    for item in active.values():
        try:
            os.killpg(item["process"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for item in active.values():
        process = item["process"]
        try:
            process.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        item["stream"].close()


def run_pool(jobs, output, parallel, sample_interval=1.0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    active, completed = {}, []
    pending = iter(jobs)
    exhausted = False
    start = time.monotonic()
    peak_rss = peak_pss = 0
    next_sample = start
    before = resources()
    status = "failed"
    try:
        while active or not exhausted:
            while len(active) < parallel and not exhausted:
                job = next(pending, None)
                if job is None:
                    exhausted = True
                    break
                log = output / f"{job.method}_seed_{job.seed}.log"
                stream = log.open("w")
                env = {**os.environ, **{name: "1" for name in (
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
                try:
                    process = subprocess.Popen(job.command, cwd=ROOT, env=env, start_new_session=True,
                                               stdout=stream, stderr=subprocess.STDOUT)
                except BaseException:
                    stream.close()
                    raise
                active[process.pid] = {"job": job, "process": process, "stream": stream,
                                       "start": time.monotonic(), "log": str(log)}
                print(f"Started {job.method} seed {job.seed} (PID {process.pid}); log: {log}", flush=True)
            now = time.monotonic()
            if now >= next_sample:
                memories = [tree_memory(pid) for pid in active]
                peak_rss = max(peak_rss, sum(r for r, _ in memories))
                peak_pss = max(peak_pss, sum(p for _, p in memories))
                next_sample = now + sample_interval
            for pid, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                job = item["job"]
                completed.append({"method": job.method, "seed": job.seed, "exit_code": code,
                                  "seconds": time.monotonic() - item["start"],
                                  "budget_steps": job.budget_steps, "log": item["log"]})
                if code != 0:
                    raise RuntimeError(f"{job.method} seed {job.seed} failed ({code}); see {item['log']}")
                item["stream"].close()
                del active[pid]
                print(f"Completed {job.method} seed {job.seed}", flush=True)
            if active:
                time.sleep(.1)
        status = "completed"
    except KeyboardInterrupt:
        status = "cancelled"
        raise
    finally:
        stop_processes(active)
        seconds = time.monotonic() - start
        summary = {"status": status, "max_parallel": parallel, "wall_seconds": seconds,
                   "completed_jobs": completed, "peak_tree_rss_bytes": peak_rss,
                   "peak_tree_pss_bytes": peak_pss, "resources_before": before,
                   "resources_after": resources(),
                   "budget_steps_per_second": sum(j.budget_steps for j in jobs) / seconds
                   if status == "completed" else None}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    args = parse_args(argv)
    slots = min(args.max_parallel, len(args.seeds) * len(args.methods))
    print_resources(resources(), slots, args.env_workers)
    if args.check_resources:
        return 0
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.dry_run:
        output = ROOT / "results/comparisons" / (run_id + "_dry_run")
    else:
        parent = ROOT / "results/comparisons"
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=run_id + "_", dir=parent))
    jobs = jobs_for(args, output.name, output)
    print(f"{len(jobs)} jobs, one shared pool of {slots}; environment: {args.env}", flush=True)
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job.command))
        return 0
    (output / "commands.json").write_text(json.dumps([vars(job) for job in jobs], indent=2) + "\n")
    print(f"Comparison output: {output}", flush=True)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def cancel(signum, frame):
        raise KeyboardInterrupt
    for sig in previous:
        signal.signal(sig, cancel)
    try:
        summary = run_pool(jobs, output, slots)
    except KeyboardInterrupt:
        print("Cancelled; active trainers and environment workers terminated.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(f"Completed in {summary['wall_seconds']:.1f}s; "
          f"budget throughput {summary['budget_steps_per_second']:.1f} steps/s; "
          f"peak sampled pool PSS {summary['peak_tree_pss_bytes'] / 1024**3:.2f} GiB.\n"
          f"Summary: {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
