"""Compare exact rollout and learner-state hashes across fresh, same-seed processes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile

from run_comparison import Job, ROOT, run_pool
from run_shared_suite import parse_args as suite_args, stages_for


def digest(value):
    import numpy as np
    import torch
    result = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            visit(item.detach().cpu().contiguous().numpy())
        elif isinstance(item, np.ndarray):
            result.update(f"array:{item.dtype}:{item.shape}:".encode())
            result.update(item.tobytes())
        elif isinstance(item, np.generic):
            visit(item.item())
        elif isinstance(item, dict):
            result.update(b"dict:")
            for key in sorted(item, key=lambda key: (str(type(key)), repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(f"sequence:{len(item)}:".encode())
            for child in item:
                visit(child)
        else:
            result.update(f"{type(item).__name__}:{item!r};".encode())
    visit(value)
    return result.hexdigest()


def learner_state(model):
    import random
    import numpy as np
    import torch
    state = {
        "actor": model.mac.agent.state_dict(), "critic": model.critic.state_dict(),
        "target_critic": model.target_critic.state_dict(),
        "actor_optimizer": model.agent_optimiser.state_dict(),
        "critic_optimizer": model.critic_optimiser.state_dict(),
        "critic_updates": model.critic_training_steps,
        "target_update": model.last_target_update_step,
        "numpy_rng": np.random.get_state(), "torch_rng": torch.random.get_rng_state(),
        "python_rng": random.getstate(),
    }
    if hasattr(model, "old_mac"):
        state["old_actor"] = model.old_mac.agent.state_dict()
    if hasattr(model, "rew_ms"):
        state["reward_moments"] = vars(model.rew_ms)
    if getattr(model, "adapter", None) is not None:
        adapter = model.adapter
        state["maca"] = {
            "mixer": adapter.pipeline.reducer.net.state_dict(), "rng": adapter.rng,
            "shuffle_rng": adapter.pipeline.reducer._generator.get_state(),
            "episodes": adapter.episodes, "returns": list(adapter.returns),
            "fit_version": adapter.pipeline.fit_version, "credit": adapter.last_ledgers,
            "credit_scale": adapter.pipeline.shaper.running_max_abs,
        }
    return state


def worker(trace_path, arguments):
    """Instrument the normal entrypoint without changing actions, updates or seeds."""
    sys.path.insert(0, str(ROOT / "src"))
    import yaml
    from learners import REGISTRY
    from runners.parallel_runner import ParallelRunner
    from utils.logging import Logger
    trace = []
    original_run = ParallelRunner.run
    def run(self, test_mode=False):
        batch = original_run(self, test_mode=test_mode)
        trace.append({"kind": "evaluation" if test_mode else "training_rollout",
                      "t_env": self.t_env, "sha256": digest(batch.data.transition_data)})
        return batch
    ParallelRunner.run = run
    config_name = next(arg.split("=", 1)[1] for arg in arguments if arg.startswith("--config="))
    config = yaml.safe_load((ROOT / "src/config/algs" / f"{config_name}.yaml").read_text())
    cls = REGISTRY[config["learner"]]
    original_train = cls.train
    def train(self, batch, t_env, episode_num):
        result = original_train(self, batch, t_env, episode_num)
        trace.append({"kind": "learner_update", "t_env": t_env, "sha256": digest(learner_state(self))})
        return result
    cls.train = train
    original_log = Logger.log_stat
    def log(self, key, value, t, to_sacred=True):
        # Wall-clock measurements deliberately cannot match across repetitions.
        if not key.endswith("_seconds"):
            trace.append({"kind": "metric", "key": key, "t_env": int(t), "sha256": digest(value)})
        return original_log(self, key, value, t, to_sacred)
    Logger.log_stat = log
    sys.argv = [str(ROOT / "src/main.py"), *arguments]
    try:
        runpy.run_path(str(ROOT / "src/main.py"), run_name="__main__")
    finally:
        Path(trace_path).write_text(json.dumps(trace, indent=2) + "\n")


def check_native_food_spacing(resets=100):
    """Native food spacing prevents a player belonging to two loading groups."""
    import gymnasium as gym
    import numpy as np
    from run_shared_suite import TASKS
    minimum, checked = None, 0
    for _, key, _ in TASKS:
        env = gym.make(key)
        try:
            for repeat in range(resets):
                env.reset(seed=1587 + repeat)
                foods = np.argwhere(env.unwrapped.field > 0)
                for i, food in enumerate(foods):
                    for other in foods[i + 1:]:
                        distance = int(np.abs(food - other).sum())
                        minimum = distance if minimum is None else min(minimum, distance)
                        assert distance > 2, f"Overlapping loading neighborhoods in {key}"
                checked += 1
        finally:
            env.close()
    return {"checked_resets": checked, "minimum_food_manhattan_distance": minimum,
            "overlapping_loading_neighborhoods": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1587)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--max-parallel", type=int, default=4)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.max_parallel < 1:
        parser.error("Steps and parallelism must be positive")
    parent = ROOT / "results/reproducibility"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(tempfile.mkdtemp(prefix=stamp + "_", dir=parent))
    jobs, pairs = [], {}
    for repeat in range(2):
        run_output = output / f"repeat_{repeat}"
        for stage in stages_for(suite_args(["--smoke", "--seeds", str(args.seed)]), run_output):
            for job in stage["jobs"]:
                key = f"{stage['task']}_{job.method}"
                trace_path = output / f"{key}_repeat_{repeat}.json"
                pairs.setdefault(key, []).append(trace_path)
                arguments = [f"t_max={args.steps}" if arg.startswith("t_max=") else arg for arg in job.command[3:]]
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(trace_path), *arguments]
                jobs.append(Job(f"{key}_repeat_{repeat}", args.seed, command, args.steps))
    (output / "commands.json").write_text(json.dumps([vars(job) for job in jobs], indent=2) + "\n")
    print(f"Reproducibility audit: {output}", flush=True)
    run_pool(jobs, output, args.max_parallel)
    comparisons = {}
    for key, paths in pairs.items():
        left, right = [json.loads(path.read_text()) for path in paths]
        mismatch = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
        exact = left == right
        comparisons[key] = {
            "exact": exact, "events": [len(left), len(right)], "first_mismatch": mismatch,
            "left": left[mismatch] if mismatch is not None else None,
            "right": right[mismatch] if mismatch is not None else None,
        }
        print(f"{key}: {'EXACT' if exact else 'DIFFERENT'}", flush=True)
    report = {
        "seed": args.seed, "steps_per_run": args.steps, "max_parallel": args.max_parallel,
        "comparisons": comparisons, "native_food_spacing": check_native_food_spacing(),
        "seed_helper": "unchanged src/maca/utils.py set_seeds from original MACA repository",
        "seed_helper_sha256": hashlib.sha256((ROOT / "src/maca/utils.py").read_bytes()).hexdigest(),
        "scope": "Fresh-process CPU runs on this runtime; timestamps, paths and wall-clock metrics excluded",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {output / 'report.json'}", flush=True)
    return 0 if all(row["exact"] for row in comparisons.values()) else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(main())
