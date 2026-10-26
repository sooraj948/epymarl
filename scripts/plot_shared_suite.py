"""Plot completed suite returns across seeds, including explicitly imported historical runs."""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ["mappo", "maca_beta_0p08", "maca_beta_0p1", "maca_beta_0p15", "coma"]
LABELS = {"mappo": "MAPPO", "coma": "COMA", "maca_beta_0p08": "MACA β=0.08",
          "maca_beta_0p1": "MACA β=0.10", "maca_beta_0p15": "MACA β=0.15"}
COLORS = dict(zip(METHODS, ["#2563eb", "#d97706", "#15803d", "#9333ea", "#dc2626"]))
TITLES = {"8x8_2p2f_coop": "8×8 · 2 agents · 2 foods\nForced cooperation",
          "8x8_3p2f": "8×8 · 3 agents · 2 foods\nStandard LBF",
          "10x10_3p3f": "10×10 · 3 agents · 3 foods\nStandard LBF"}
METRICS = {"return_mean": "Training return", "test_return_mean": "Evaluation return"}


@dataclass
class Run:
    task: str
    method: str
    seed: int
    folder: Path
    config: dict
    metrics: dict
    imported: bool = False


def metric_series(metrics, metric):
    row = metrics[metric]
    if not row["steps"] or len(row["steps"]) != len(row["values"]):
        raise ValueError(f"Missing or inconsistent {metric} observations")
    # Keep the last observation if the same step was recorded twice.
    points = {}
    for step, value in zip(row["steps"], row["values"]):
        step, value = float(step), float(value)
        if not np.isfinite([step, value]).all():
            raise ValueError(f"Nonfinite {metric} observation")
        points[step] = value
    steps = np.asarray(sorted(points))
    return steps, np.asarray([points[step] for step in steps])


def align_metric(runs, metric, points=401):
    """Interpolate within support shared by every run; never invent endpoint values."""
    series = [(run, *metric_series(run.metrics, metric)) for run in runs]
    start = max(steps[0] for _, steps, _ in series)
    stop = min(steps[-1] for _, steps, _ in series)
    if stop <= start:
        raise ValueError("Runs have no shared time support")
    grid = np.linspace(start, stop, points)
    groups = {}
    for method in METHODS:
        selected = [(run, steps, values) for run, steps, values in series if run.method == method]
        if not selected:
            continue
        values = np.stack([np.interp(grid, steps, values) for _, steps, values in selected])
        groups[method] = {"mean": values.mean(axis=0), "std": values.std(axis=0, ddof=1),
                          "n": len(selected)}
    return grid, groups


def discover_runs(suite, sacred):
    manifest = json.loads((suite / "commands.json").read_text())
    environments = {stage["environment"]: stage["task"] for stage in manifest}
    budgets = {stage["task"]: stage["jobs"][0]["budget_steps"] for stage in manifest}
    expected = {(stage["task"], job["method"], job["seed"]) for stage in manifest for job in stage["jobs"]}
    expected.add(("8x8_2p2f_coop", "mappo", 1587))
    found = {}
    for config_file in sacred.glob("*/*/*/config.json"):
        config = json.loads(config_file.read_text())
        folder = config_file.parent
        provenance_file = folder / "import_provenance.json"
        provenance = json.loads(provenance_file.read_text()) if provenance_file.exists() else {}
        imported = provenance.get("suite_label") == suite.name
        if config.get("label") != suite.name and not imported:
            continue
        environment = config["env_args"]["key"]
        if environment not in environments:
            raise ValueError(f"Unexpected suite environment at {folder}")
        method = provenance["method"] if imported else config["name"].split("_" + suite.name)[0]
        task, seed = environments[environment], int(config["seed"])
        key = (task, method, seed)
        if key not in expected or key in found:
            raise ValueError(f"Unexpected or duplicate suite run {key}: {folder}")
        status = json.loads((folder / "run.json").read_text())["status"]
        if status != "COMPLETED":
            raise ValueError(f"Run is not completed: {folder} ({status})")
        found[key] = Run(task, method, seed, folder, config,
                         json.loads((folder / "metrics.json").read_text()), imported)
    if set(found) != expected:
        raise ValueError(f"Missing suite runs: {sorted(expected - set(found))}")
    for task in budgets:
        for method in METHODS:
            seeds = {run.seed for run in found.values() if run.task == task and run.method == method}
            if seeds and seeds != {1587, 2398, 2875, 3370}:
                raise ValueError(f"Expected all four seeds for {task}/{method}: {seeds}")
    tasks = list(budgets)
    runs = sorted(found.values(), key=lambda run: (tasks.index(run.task), METHODS.index(run.method), run.seed))
    return runs, budgets


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_panel(ax, grid, groups, budget, metric, title=None, zoom=None):
    for method, stats in groups.items():
        mean, std = stats["mean"], stats["std"]
        color = COLORS[method]
        ax.fill_between(grid / 1e6, mean - std, mean + std, color=color, alpha=.12, linewidth=0)
        ax.plot(grid / 1e6, mean, color=color, linewidth=2,
                linestyle="--" if method == "coma" else "-", label=LABELS[method])
    ax.set_xlim(0, (zoom if zoom is not None else budget) / 1e6)
    ax.set_ylim(-.035, 1.035)
    ax.set_xlabel("Training environment steps (millions)")
    ax.set_ylabel(METRICS[metric])
    if title:
        ax.set_title(title, fontsize=12, pad=10)
    ax.grid(alpha=.19, linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def finish_figure(fig, output, stem, methods, title, historical):
    handles = [Line2D([0], [0], color=COLORS[method], lw=2,
                      linestyle="--" if method == "coma" else "-", label=LABELS[method]) for method in methods]
    fig.suptitle(title, fontsize=16, y=.99)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .95),
               ncol=len(handles), frameon=False, fontsize=10)
    caption = "Mean ±1 sample SD across 4 seeds · Linear time alignment within common observed support · No smoothing"
    if historical:
        caption += "\n8×8 cooperative MAPPO includes historical seed 1587 (GPU); the other runs used the current CPU runtime."
    fig.text(.5, .014, caption, ha="center", va="bottom", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, .065 if historical else .045, 1, .88))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "results/shared_suite/20261025T224414Z_wwvf5p9f")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    suite = args.suite.resolve()
    output = args.output.resolve() if args.output else suite / "plots"
    runs, budgets = discover_runs(suite, ROOT / "results/sacred")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": "#334155", "text.color": "#0f172a",
                         "pdf.fonttype": 42})
    index, raw, aggregated, summaries = [], [], [], []
    aligned = {}
    for run in runs:
        index.append({"task": run.task, "method": LABELS[run.method], "seed": run.seed,
                      "sacred_run_id": run.folder.name, "sacred_folder": str(run.folder),
                      "historical_import": run.imported,
                      "device": "GPU" if run.config["use_cuda"] else "CPU"})
        for metric in METRICS:
            steps, values = metric_series(run.metrics, metric)
            for step, value in zip(steps, values):
                raw.append({"task": run.task, "method": LABELS[run.method], "seed": run.seed,
                            "metric": metric, "step": int(step), "value": value})
    for task in budgets:
        selected = [run for run in runs if run.task == task]
        for metric in METRICS:
            grid, groups = align_metric(selected, metric)
            aligned[task, metric] = grid, groups
            for method, stats in groups.items():
                for step, mean, std in zip(grid, stats["mean"], stats["std"]):
                    aggregated.append({"task": task, "method": LABELS[method], "metric": metric,
                                       "step": step, "mean": mean, "sample_std": std, "n_seeds": stats["n"]})
                summaries.append({"task": task, "method": LABELS[method], "metric": metric,
                                  "last_common_step": grid[-1], "mean": stats["mean"][-1],
                                  "sample_std": stats["std"][-1], "n_seeds": stats["n"]})
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9), squeeze=False)
    for col, task in enumerate(budgets):
        for row, metric in enumerate(METRICS):
            draw_panel(axes[row, col], *aligned[task, metric], budgets[task], metric,
                       TITLES[task] if row == 0 else None)
    finish_figure(fig, output, "all_environments_consolidated", METHODS,
                  "LBF — consolidated training and evaluation returns", historical=True)
    for task in budgets:
        historical = task == "8x8_2p2f_coop"
        fig, axes = plt.subplots(2 if historical else 1, 2,
                                 figsize=(12, 9 if historical else 5.4), squeeze=False)
        for col, metric in enumerate(METRICS):
            draw_panel(axes[0, col], *aligned[task, metric], budgets[task], metric,
                       "Full training budget")
            if historical:
                draw_panel(axes[1, col], *aligned[task, metric], budgets[task], metric,
                           "First 500k steps", zoom=500000)
        methods = list(aligned[task, "return_mean"][1])
        finish_figure(fig, output, f"{task}_consolidated", methods,
                      TITLES[task].replace("\n", " — "), historical)
    write_csv(output / "runs.csv", index)
    write_csv(output / "returns_raw.csv", raw)
    write_csv(output / "returns_mean_std.csv", aggregated)
    write_csv(output / "final_common_returns.csv", summaries)
    metadata = {
        "suite": str(suite), "run_count": len(runs), "seeds": [1587, 2398, 2875, 3370],
        "band": "sample standard deviation across seeds, ddof=1; not a confidence interval",
        "alignment": "401 uniform points per task/metric on shared observed time support; linear interpolation; no extrapolation",
        "smoothing": "none", "reward_units": "raw summed environment reward; no intrinsic rewards",
        "evaluation": "prescribed soft_policies selector samples actions",
        "historical_run": next(row for row in index if row["historical_import"]),
    }
    (output / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Plotted {len(runs)} completed runs. Output: {output}")
    for row in summaries:
        if row["metric"] == "test_return_mean":
            print(f"{row['task']} | {row['method']} | {row['mean']:.4f} ± {row['sample_std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
