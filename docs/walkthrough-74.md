# Walkthrough 74 Single repository MACA setup

## Major design choices

1. **Use this fork as the only active codebase.** The original MACA library is copied into `src/maca/`; the old MACA repository remains a historical reference. No separate package installation or second source checkout is required to import it.
2. **Preserve the imported algorithm exactly.** All 16 tracked MACA Python files match source commit `893f188e8bb918ec099150510a6c0edb7829022c` byte for byte. The adapter and learner integration remain the next step in [walkthrough 73](walkthrough-73.md).
3. **Preserve existing experiment checkouts and results.** This repository occupies a new `epymarl-maca/` directory. Local experiment results are excluded through `.gitignore`.
4. **Keep the agreed experiment design.** The initial integration uses online random-donor splice, `mappo_ns` and EPyMARL n-step PPO. The mixer's GAE teacher and the policy's n-step advantages remain a major issue for later controlled comparison, documented in walkthrough 73.

## Repository and source provenance

| Item | Value |
| --- | --- |
| Local checkout | `/scratch/cluster/sooraj/epymarl-maca` |
| Fork remote `origin` | `git@github.com:sooraj948/epymarl.git` |
| Initial local branch | `main` |
| EPyMARL source commit | `cbc38c09588064eab978501d0f12c2cf58fa7fc2` |
| MACA source commit | `893f188e8bb918ec099150510a6c0edb7829022c` |
| Imported source | Original tracked `maca/*.py` files, now `src/maca/*.py` |

GitHub was unreachable during setup: HTTPS could not resolve `github.com`, and SSH failed because the environment's current UID has no account entry. The checkout was therefore cloned from the existing local EPyMARL repository with `--no-hardlinks`, preserving Git history and independent object storage, then its remote was set to the user's fork. The remote branch contents could not be verified. Publishing remains pending GitHub access.

The imported root `AGENTS.md` preserves the user's development and documentation conventions. The original integration plan is in `docs/walkthrough-73.md`. Only this fork should receive future integration changes.

## Runtime and dependencies

`src/main.py` already places `src/` on Python's import path. Code inside EPyMARL can therefore import `maca` directly. Shell import checks use `PYTHONPATH=src` to reproduce that source layout.

The copied code uses built-in generic type annotations and requires Python 3.9 or newer. The available Python 3.10 runtime successfully imports the package and selected credit pipeline; the existing `venv` uses Python 3.8 and is unsuitable for the unchanged source. No existing virtual environment was modified and no training run was launched. Select the experiment runtime before executing the integration plan.

`requirements-maca.txt` includes EPyMARL's base requirements plus scikit-learn and native LBF. It records dependency intent; setup did not install or upgrade packages. Optional historical UMAP and Wasserstein variants may require additional packages when selected.

## Verification

The setup checks established that:

- All 16 imported source files are identical to the originals and parse under Python 3.10.
- `maca`, `MACAConfig`, `StateActionOccupancyEncoder`, `TrajectoryReturnRegressor` and `CreditPipeline` import from this checkout.
- The fork remote points to the supplied SSH URL.
- The EPyMARL training source remains at the inspected baseline, with no learner integration enabled by the import.
- New documentation, requirements and repository metadata pass whitespace checks. The copied MACA source retains its historical formatting.

The import check can be repeated with a compatible runtime:

```bash
cd /scratch/cluster/sooraj/epymarl-maca
PYTHONPATH=src python3.10 -c 'import maca; from maca.credit_pipelines import CreditPipeline; print(maca.__file__)'
```

## Publish and continue

After GitHub access is available, publish the local setup commit:

```bash
cd /scratch/cluster/sooraj/epymarl-maca
git push origin main
```

If the remote has advanced, fetch and review the difference before reconciling history. Preserve remote changes and use a normal push.

Continue with the adapter and learner in walkthrough 73. In particular, the copied random-donor loader still needs its auxiliary helper dependency removed before the selected pipeline can run independently. Import success does not constitute a completed training integration.
