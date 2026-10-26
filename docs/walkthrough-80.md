# Walkthrough 80 Consolidated results and folder names

## Major design choices

1. **Move the historical completed MAPPO seed 1587 into this repository without rewriting its experiment metadata.** Its five original files were verified by SHA-256 before and after the transfer. Copy the three referenced shared Sacred source files as well, preserving hashes. Record import provenance separately and leave an old-path symlink so earlier references still resolve to the same data rather than a duplicate run.
2. **Select the exact completed comparison cohort.** Match the 35 current-suite runs by Sacred label and the imported run by explicit provenance. Require 36 unique completed runs and all four seeds for each of the nine method/task combinations. Exclude smoke runs, seeding audits and the earlier separate-network comparison.
3. **Aggregate original environment returns across seeds.** Plot training and evaluation separately. Use mean and sample standard deviation across the four seeds, without smoothing. Preserve individual raw observations in CSV. The bands are not confidence intervals and do not represent within-evaluation-episode variability.
4. **Align by training environment steps without extrapolation.** Runs log at slightly different timesteps because episodes have different lengths. For each task and metric, use the time interval observed by every included seed/method and interpolate linearly onto 401 common points. Do not fill missing leading observations with zero or extend endpoints to the nominal budget.
5. **Mark historical runtime differences.** The imported seed ran with CUDA/Python 3.8/PyTorch 2.4.1; the new suite ran on CPU/Python 3.10/PyTorch 2.5.1. Preserve this provenance in the run index, metadata and figure captions. Evaluation remains the prescribed sampled-action evaluation, and all plotted returns exclude MACA intrinsic shaping.

## Historical run transfer

The completed run originally at:

```text
/scratch/cluster/sooraj/epymarl/results/sacred/mappo/lbforaging:Foraging-8x8-2p-2f-coop-v3/2
```

now lives at:

```text
/scratch/cluster/sooraj/epymarl-maca/results/sacred/mappo/lbforaging:Foraging-8x8-2p-2f-coop-v3/8
```

Its original `config.json`, `run.json`, `metrics.json`, `info.json` and `cout.txt` are unchanged. `import_provenance.json` records the source/destination IDs, original file checksums, copied Sacred sources and associated suite label. The original path is a relative symlink to the destination. The imported `config.json` therefore retains its historical `default_label`; the plotting script uses explicit import provenance to include it with the current suite.

This is Sacred run **8**, seed **1587**. The directory number is an observer-assigned run ID, not a seed or checkpoint timestep.

## Reading folder and file names

The current suite root is:

```text
results/shared_suite/20261025T224414Z_wwvf5p9f/
```

| Name or pattern | Meaning |
| --- | --- |
| `20261025T224414Z` | Suite launch timestamp: 2026-10-25 22:44:14 UTC |
| `wwvf5p9f` | Random suffix to avoid output-directory collisions |
| `8x8_2p2f_coop` | 8-by-8 grid, two players, two foods, forced cooperation |
| `8x8_3p2f` | 8-by-8 grid, three players, two foods, standard LBF without forced cooperation |
| `10x10_3p3f` | 10-by-10 grid, three players, three foods, standard LBF |
| `mappo_seed_2398.log` | Vanilla shared MAPPO, seed 2398, console log |
| `maca_beta_0p08_seed_1587.log` | MACA with configured beta 0.08, seed 1587; `p` replaces the decimal point |
| `coma_seed_3370.log` | Prescribed COMA, seed 3370 |
| `results/sacred/<algorithm>/<environment>/<number>/` | Saved metrics/configuration for observer run ID `<number>`; read its seed from `config.json` |
| `models/<method>_<suite>_seed<seed>_<environment>_<local-start-time>/<steps>/` | Checkpoint for that training run at `<steps>` total training environment steps |
| `maca/<method>_<suite>_seed<seed>_<environment>_<local-start-time>/` | MACA credit ledgers, donor bank and run manifest |

For example, `.../models/.../100500/` is a snapshot at 100,500 environment steps, not episode 100,500. A model-folder timestamp is a local run-start timestamp; the suite timestamp ending in `Z` is UTC.

`plots/runs.csv` maps every task/method/seed to its actual Sacred directory, numerical observer ID, runtime device and historical-import flag. It is the direct lookup table for all 36 results.

## Generated plots and data

Output directory:

```text
results/shared_suite/20261025T224414Z_wwvf5p9f/plots/
```

| Artifact | Contents |
| --- | --- |
| `all_environments_consolidated.png` / `.pdf` | Three task columns; training and evaluation rows; all applicable methods |
| `8x8_2p2f_coop_consolidated.png` / `.pdf` | Full-budget training/evaluation plus first-500k-step detail for MAPPO, COMA and all three MACA betas |
| `8x8_3p2f_consolidated.png` / `.pdf` | Training/evaluation comparison of MAPPO and COMA |
| `10x10_3p3f_consolidated.png` / `.pdf` | Training/evaluation comparison of MAPPO and COMA |
| `runs.csv` | Exact run/seed/folder index |
| `returns_raw.csv` | Every selected per-seed logged training/evaluation return |
| `returns_mean_std.csv` | Aligned means, sample standard deviations and seed counts |
| `final_common_returns.csv` | Values at the last common observed timestep for each task/metric, which can precede the nominal budget |
| `plot_metadata.json` | Cohort, alignment, band definition, reward units and historical-run provenance |

Recreate them with:

```bash
cd /scratch/cluster/sooraj/epymarl-maca
../.TPCA/bin/python scripts/plot_shared_suite.py
```

Optional `--suite <directory>` and `--output <directory>` select input/output locations. The script requires the current four-seed comparison matrix and its explicit historical import. It does not modify source run data.

Validation: two plotting tests pass, checking common-support interpolation, the mean/sample-SD calculation, duplicate-step handling and nonfinite rejection. The real-data cohort has 36 unique completed runs with four seeds in all nine groups. Source-run and copied-source checksums were verified again after plotting. The overall and detailed small-task PNGs were visually inspected; PNG and PDF versions of all four figures were generated. Matplotlib is now listed in `requirements-maca.txt`.
