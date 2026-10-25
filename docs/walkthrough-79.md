# Walkthrough 79 Smaller MACA scope and original seeding

## Major design choices

1. **Run MACA only on the existing 8x8 two-player cooperative task.** Both larger standard LBF tasks run vanilla shared MAPPO and prescribed COMA only. Keep all four seeds for every experiment except the historical completed first-task MAPPO seed 1587. The current matrix is 35 jobs and 86M budgeted training steps.
2. **Default to four simultaneous trainers.** Environment groups remain sequential, and freed slots refill across methods/betas/seeds within a group. Each trainer still has ten environment workers, so the default ceiling is four trainers plus forty environment workers. `--max-parallel` can override the default explicitly; no automatic expansion is performed.
3. **Reuse the original MACA seeding implementation unchanged.** `src/main.py` now calls `src/maca/utils.py:set_seeds()`, whose source is byte-identical to `Unsupervised-Multi-agent-Credit-Assignment/maca/utils.py`. That helper seeds NumPy, CPU PyTorch and CUDA, selects deterministic cuDNN kernels, disables cuDNN benchmarking, and requests deterministic PyTorch operations with the original `warn_only=True` behavior. It retains the original `CUBLAS_WORKSPACE_CONFIG` default. Do not introduce a new hash-seed policy, loading tie-break rule or environment patch.
4. **Retain EPyMARL's worker and evaluation behavior.** Worker `i` receives `seed+i`; resets then continue each environment's own RNG stream. Existing Sacred Python-RNG seeding remains. MACA's dedicated donor, mixer-shuffle and isolated RNG streams remain. Prescribed `soft_policies` action selection is unchanged, including its stochastic evaluation behavior.
5. **Check numerical repeatability directly and state its scope.** Audit fresh processes, not just seed arguments or plots. Compare exact training/evaluation batch hashes, learner and optimizer states, running reward moments, RNG states, MACA mixer states and credits, and non-timing metrics. Positive short-run results do not establish bitwise equivalence for every full-budget run, checkpoint continuation, hardware or dependency version.

## Current experiment matrix

| Order | Full-sight native LBF task | Methods | Steps per run | Jobs |
| --- | --- | --- | ---: | ---: |
| 1 | `Foraging-8x8-2p-2f-coop-v3` | MAPPO, COMA, MACA beta 0.08/0.1/0.15 | 2M | 19 |
| 2 | `Foraging-8x8-3p-2f-v3` | MAPPO, COMA | 2M | 8 |
| 3 | `Foraging-10x10-3p-3f-v3` | MAPPO, COMA | 4M | 8 |

Seeds remain 1587, 2398, 2875 and 3370. Only the first task's vanilla MAPPO excludes 1587. All task horizons remain 50 steps. The first task retains forced cooperation; neither larger task does. All MAPPO/COMA hyperparameters, MACA reward processing, beta schedule and checkpoint/evaluation intervals from walkthrough 78 remain.

```bash
cd /scratch/cluster/sooraj/epymarl-maca
./run_shared_suite.sh --dry-run
./run_shared_suite.sh --tmux
```

The wrapper defaults to `.TPCA` and the new four-trainer pool without additional flags. The tmux option requires access to the existing `epymarl-maca` session and creates a new detached window. A process that already loaded an earlier queue will not adopt these edits; the new matrix applies to new invocations. Resume manifests deliberately reject a previously saved incompatible matrix rather than silently changing its meaning.

## Seeding audit

Original reference: `/scratch/cluster/sooraj/Unsupervised-Multi-agent-Credit-Assignment/maca/utils.py:43`.

Direct comparison against that original helper confirmed **identical NumPy and PyTorch random draws and determinism flags for all four requested seeds**. Original source files and installed LBF code were not modified.

`tests/test_seeding.py` verifies that actual subprocess workers receive the intended offset seeds by comparing their reset observations to directly constructed Gymma environments seeded with `1587+i`. It checks that independent workers have different initial boards, repeated runner construction reproduces them exactly, and subsequent resets advance the RNG rather than repeating the first board.

`scripts/check_reproducibility.py` instruments the ordinary entrypoint without changing its seeds, actions or learning updates. It runs each of the nine current method/task variants twice in separate processes at seed 1587, with 6,000 training steps per run and at most four concurrent trainers. It hashes:

- Every training and evaluation batch, including observations, state, actions, rewards, availability, termination and filled masks.
- Actor, critic, target critic, old actor where present, optimizer states, reward moments and update counters after each learner update.
- NumPy, PyTorch and Python RNG state; for MACA, mixer parameters/buffers, isolated RNG state, shuffle generator state, running return history, fit version, credit scale and applied-credit ledgers.
- Every logged numerical metric except wall-clock timing.

Output paths, timestamps and wall-clock timings are intentionally excluded. Hash equality is byte equality of numerical data, rather than a tolerance comparison. The diagnostic does not claim to compare simulator checkpoint continuation or to exhaustively prove all future trajectories deterministic.

```bash
../.TPCA/bin/python scripts/check_reproducibility.py
# Optional other seed, budget or concurrency:
../.TPCA/bin/python scripts/check_reproducibility.py --seed 2398 --steps 6000 --max-parallel 4
```

On 2026-10-25, **all nine pairs matched exactly** across all recorded events. Report: `results/reproducibility/20261025T223624Z_ehog0bc6/report.json`; detailed traces and process logs are alongside it. All 18 runs completed, including actual mixer fitting and nonzero MACA shaping. The audit took about 64.9 seconds on this runtime.

The helper preserves the old `warn_only=True` deterministic-operation setting. It is the requested original setup, not an unconditional guarantee that every future operation has a deterministic implementation. Historical custom MACA used a different rollout/training implementation; retaining its seed helper does not make EPyMARL trajectories identical to historical custom-trainer trajectories. The earlier successful EPyMARL seed-1587 run also used different Python/PyTorch/GPU runtime settings.

## Correction to the loading-order concern

An initial synthetic probe of LBF's object-identity set ordering produced different loading outcomes on a manually constructed board. **That board is unreachable under the native spawn rules:** its two food positions were Manhattan distance two apart, whereas native spawning excludes food pairs at that distance or closer. A player therefore cannot belong to two foods' adjacent loading groups in these native tasks. Loading groups are disjoint, so the initial example does not demonstrate a reproducibility defect for the requested tasks. No loading-order patch was made.

The corrected audit records the synthetic probe as unreachable and separately checks 300 native resets across the three requested tasks. The minimum observed food distance was three and no overlapping loading neighborhoods occurred. The original seeding and upstream environment behavior remain.

## Evaluation behavior and validation

The supplied MAPPO and COMA configs select `soft_policies`. Its `SoftPoliciesSelector.select_action()` always calls `Categorical.sample()` and ignores `test_mode`; `test_greedy=True` does not make this selector greedy. Earlier documentation and chat descriptions of greedy evaluation were incorrect. Preserve the prescribed selector; evaluation is unshaped but stochastic, consumes the policy RNG, and uses the same runner's environment streams. A changed evaluation schedule can therefore change later training trajectories. The fixed schedule was identical in each audit pair.

All **38 tests pass**, including the revised scope/parallelism matrix, sequential groups, worker seed offsets and continuing reset streams. Shell syntax and Git whitespace checks pass. Full-suite tmux access remains restricted by the implementation sandbox; completed diagnostics are not full training runs.
