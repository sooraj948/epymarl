# Walkthrough 82 Shared MAPPO experiments and optional post-shaping standardization

## Major design choices

1. **Add post-shaping standardization as an option.** `maca.reward_standardization=environment` remains the default; `shaped` estimates running reward moments after individual credit has been added. Both options require `standardise_rewards=True` to normalize; requesting `shaped` with normalization disabled is rejected. The separate-network config keeps its existing normalization-disabled default.
2. **Use one normalizer shared by agents.** In `shaped` mode, pool all agents' valid per-timestep shaped rewards into one scalar stream. Exclude padding and transitions after termination. Update moments exactly once per PPO batch and reuse normalized rewards throughout all four PPO epochs. Do not normalize each agent independently: a common scale preserves relative credit magnitudes and keeps the shared critic/teacher in common units. Per-agent normalization is a valid distinct experiment, not a requirement of shared critics.
3. **Resolve teacher timing without circular dependence.** Current shaped moments cannot be estimated before current credit exists. The teacher therefore uses the previous batch's frozen moments and current critic to score environment-reward GAE. After credit is produced, update the running moments using current shaped rewards and normalize the policy/critic rewards with those updated moments. Initial teacher moments are mean zero, variance one, using the existing running-stat initialization. No second environment-only update occurs in this mode.
4. **Preserve the established algorithm and default behavior.** Shared MAPPO retains recurrent actors, agent-conditioned centralized critic, hidden size 128, entropy 0.001, learning rate 0.0003, four PPO epochs, clip 0.2, soft target updates at 0.01 and five-step policy targets. MACA retains online random-donor no-replay splice, occupancy/food-goal regression, signed uniform injection, loss-only shaping and dynamic beta. Stock COMA and seeding remain prescribed. The small PPO normalization hook preserves the exact default formula.
5. **Provide a targeted tmux script.** `run_maca_10x10_shaped_norm.sh` runs only betas 0.08/0.1 on full-sight standard `Foraging-10x10-3p-3f-v3`, horizon 50, four seeds and 4M steps per run. Eight jobs can run simultaneously. Names include `_shaped_norm` and the tmux window is `maca-10x10-shaped-norm`, distinguishing this cohort from earlier environment-statistics runs.
6. **Persist and identify the normalization convention.** Checkpoints save mode, running moments and the teacher transform and reject cross-mode loading. Older checkpoints without a mode are treated as `environment`. Manifests identify the moment source and teacher timing; Sacred logs record current policy mean/std and valid sample count. Ledger `reward_mean`/`reward_std` fields identify the frozen teacher transform, which precedes the policy update in shaped mode.
7. **Keep historical evidence and experiment scope explicit.** The original default suite remains 35 jobs, with MACA only on the small task. A separate 10x10 three-beta sweep and the new two-beta normalization sweep are explicit selections. Figures use raw environment returns, historical seed provenance is retained, and short reproducibility checks do not imply cross-runtime or simulator-resume guarantees.

This consolidates walkthroughs 78–82 under the five-walkthrough rule. Earlier filenames redirect here. Full pre-consolidation documents 78–81 remain available with, for example, `git show 02ceac6:docs/walkthrough-80.md`. The original separate-network integration, installation and resource benchmarks remain in [walkthrough 77](walkthrough-77.md).

## Reward processing and options

For team reward `r[t]`, actual episode length `T`, normalized signed credit `c[i]` and scheduled coefficient `b`:

```text
r_shaped[t,i] = r[t] + b * c[i] / T
```

| Choice | Moment source and update | Teacher transform | Policy/critic reward |
| --- | --- | --- | --- |
| `environment` (default) | Raw team rewards, once per batch, retaining upstream padding treatment | Current environment moments | `(r_shaped - mean_environment) / std_environment` |
| `shaped` | All agents' valid shaped rewards, pooled to one stream, once per batch | Previous shaped moments | `(r_shaped - mean_shaped_updated) / std_shaped_updated` |

Both use EPyMARL's existing `RunningMeanStd` initialization and variance convention. These are running statistics over accumulated batches, not independent per-batch standardizers. In shaped mode each valid agent-transition is a sample; common environment rewards are therefore represented once for every agent. This changes sample counts and excludes padded samples relative to the legacy mode, even before nonzero credit is available. A particular batch need not have exactly zero mean/unit variance under running statistics. Agent-specific rewards and advantages remain separate.

The `environment` choice keeps intrinsic credit out of the statistics directly, although changed policies can indirectly change collected environment rewards. Its shaped reward stream need not have unit variance. The `shaped` choice allows beta to influence both numerator and denominator and centers the pooled shaped stream. Neither convention is assumed empirically superior without the comparison.

The teacher averages the current critic's agent predictions and computes negative mean GAE without advantage standardization. Environment rewards are transformed into that teacher's chosen units before subtraction from values. The online critic is itself trained on shaped returns; there is no independent environment-only teacher critic. Critic parameters and teacher moments remain fixed throughout batch scoring. MACA preparation runs before the first critic update and its result is cached for all PPO epochs. The current batch's intrinsic credit is not fed recursively into its own teacher score.

Beta is the maximum coefficient of the existing dynamic schedule:

```text
b = beta * clip((recent_raw_return_mean + 0.01) / 0.35, 0, 1)
```

Only episodes with raw return <= 0 receive shaping by default; positive partial successes are also excluded. The existing mixer uses 16 donor samples, a total-episode readiness quota of 50, refitting every 100 episodes and 20 fitting epochs; `warmup_episodes=0` does not imply immediate credit. Donor counterfactuals splice occupancy features rather than replaying recorded actions through a simulator. Mixer fitting, donor draws and initialization use isolated RNG streams.

**Open experimental issue:** policy advantages remain EPyMARL five-step targets, whereas the teacher uses GAE with lambda 0.95. Terminal/bootstrap conventions are retained. Reward/advantage estimator differences from the historical custom trainer are not resolved here. The historical custom PPO standardized actor advantages; this EPyMARL configuration does not. Reward scaling alone therefore does not imply a particular conversion of beta between implementations.

Beta zero retains the exact stock-PPO bypass, including stock environment normalization. Positive-beta/zero-credit updates in the default environment mode also match stock shared PPO exactly. The new shaped mode intentionally uses its pooled valid-sample convention rather than claiming bitwise stock equivalence.

## Launchers and experiment matrices

Run commands from `/scratch/cluster/sooraj/epymarl-maca`. All wrappers select `../.TPCA/bin/python`, with `MACA_PYTHON` as an override; activation is not required. `--tmux` creates a fresh detached window in the **existing** `epymarl-maca` session and prints its window ID and persistent console log. Omit `--tmux` to run directly in a terminal or an already attached pane. Attach with `tmux attach -t epymarl-maca`.

```bash
# New post-shaping normalization: 8 jobs, 32M budgeted steps.
./run_maca_10x10_shaped_norm.sh --dry-run
./run_maca_10x10_shaped_norm.sh --tmux

# Existing environment-statistics 10x10 sweep: 12 jobs, 48M steps.
./run_maca_10x10.sh --tmux

# Original default comparison: 35 jobs, 86M steps.
./run_shared_suite.sh --tmux
```

The new wrapper is equivalent to:

```bash
./run_shared_suite.sh --tasks 10x10_3p3f --methods maca --betas 0.08 0.1 \
  --maca-reward-standardization shaped --tmux
```

The direct Sacred override is `maca.reward_standardization=shaped`, declared in both MACA algorithm YAMLs. `environment` preserves the earlier rule. The generic launcher supports `--tasks`, explicit `--methods mappo coma maca`, `--betas`, `--seeds` and `--max-parallel`. Explicit MACA selection enables selected betas on any selected supported task. Without selectors, the default cohort below is retained. No NaN, infinite, duplicate or negative beta values are accepted.

| Task, all full sight and horizon 50 | Default methods | Steps/run | Default jobs |
| --- | --- | ---: | ---: |
| `lbforaging:Foraging-8x8-2p-2f-coop-v3` | MAPPO, COMA, MACA 0.08/0.1/0.15 | 2M | 19 |
| `lbforaging:Foraging-8x8-3p-2f-v3` | MAPPO, COMA | 2M | 8 |
| `lbforaging:Foraging-10x10-3p-3f-v3` | MAPPO, COMA | 4M | 8 |

Seeds are 1587, 2398, 2875 and 3370. Only the first task's vanilla MAPPO omits already completed historical seed 1587. Both additional 10x10 launchers use **all four** seeds; earlier image exclusions were plotting choices. The new sweep uses beta 0.08 and 0.1; the existing 10x10 wrapper defaults to all three. Both larger tasks use standard LBF without forced cooperation, matching the requested codebase task variants. The README prescribes a 50-step horizon and supplies no task-specific MAPPO/COMA hyperparameters.

The original shared-suite proposal had 19/20/20 jobs, 59 total and 158M steps, with MACA on all tasks. That was narrowed before the full comparison to the 35-job default above. Concurrency briefly changed from eight to four during the seeding audit and was then restored to **eight**. Environments run sequentially; within a task, freed slots refill across methods/betas/seeds. Each trainer has ten environment workers; the default ceiling is eight trainers plus eighty workers, with one BLAS/OpenMP thread per trainer. Changing source does not alter a queue already loaded by a running launcher.

## Shared architecture, evaluation and runtime

Vanilla uses stock `mappo.yaml`; `maca_mappo.yaml` matches its algorithm settings except learner/name and MACA options. Shared actors use agent identity, and `cv_critic` appends an agent one-hot to centralized state, allowing different agent value predictions despite shared parameters. Hidden size is 128, learning rate 0.0003, entropy coefficient 0.001, PPO epochs 4, clipping 0.2, five-step returns, soft target tau 0.01, reward normalization on and return normalization off. Shared actor/critic implementations are unchanged. The base PPO learner now exposes a small normalization hook with the same default formula.

Stock COMA retains its YAML's shared recurrent actors, hidden size 64, entropy 0.001, ten-step returns, hard target updates every 200 updates and reward standardization. Its architecture is not forced to match MAPPO. Earlier separate/feedforward MAPPO configs and `run_maca_comparison.sh` remain available as documented in walkthrough 77.

Training is CPU, with summed common environment reward and ten fresh episodes per update. Evaluation samples 100 episodes every 50k training steps; diagnostics log every 10k; checkpoints save every 100k, including the usual initial save. Collected/logged returns and gating remain original environment rewards and exclude intrinsic shaping. `--smoke` uses the first selected seed, 6,000 steps per selected variant, ten evaluation episodes, 1,000-step diagnostics and no model saves.

**Evaluation correction:** the prescribed `soft_policies` selector always calls `Categorical.sample()` and ignores `test_mode`. `test_greedy=True` does not make it greedy; earlier chat/documentation descriptions were incorrect. It consumes policy RNG and advances the same runner's environment streams, so changing evaluation frequency can change subsequent training trajectories. The selector and fixed schedule are preserved.

## Output names, provenance and resume

Every invocation creates `results/shared_suite/<unique-id>/`, containing `commands.json`, `suite_status.json` and task directories. Task directories contain per-method/beta/seed `.log` files, `summary.json` with sampled resource usage, `models/` and MACA manifests/credit ledgers/donor banks in `maca/`. Shaped-normalization logs include names such as `maca_beta_0p1_shaped_norm_seed_2398.log`. Policy normalization diagnostics are `maca/policy_reward_mean`, `maca/policy_reward_std` and `maca/normalization_samples`. Manifests specify teacher units, moment source, timing, config, Git revision/status, package versions and device.

| Name/pattern | Meaning |
| --- | --- |
| `20261025T224414Z_wwvf5p9f` | Launch at 2026-10-25 22:44:14 UTC plus random collision-avoidance suffix |
| `8x8_2p2f_coop`, `8x8_3p2f`, `10x10_3p3f` | Grid size, player count (`p`), food count (`f`), optional forced-cooperation suffix |
| `mappo_seed_2398.log`, `coma_seed_3370.log` | Algorithm and seed |
| `maca_beta_0p08_seed_1587.log` | MACA beta 0.08 (`p` replaces decimal point), seed 1587 |
| `results/sacred/<algorithm>/<environment>/<number>/` | Observer run ID, **not seed**; `config.json` gives seed and suite label |
| `models/<method>_<suite>_seed<seed>_<environment>_<local-start-time>/<steps>/` | Checkpoint at that many total training environment steps |
| `maca/<method>_<suite>_seed<seed>_<environment>_<local-start-time>/` | Credit ledgers, random donor bank and manifest |

`100500/` means 100,500 environment steps, not episodes. Model-folder start times are local, while suite names ending in `Z` are UTC. Sacred metrics remain in `metrics.json` under the base-config algorithm folder; identify a cohort by its suite label and method-specific name. The plot index described below maps run IDs to seeds.

```bash
./run_shared_suite.sh --resume results/shared_suite/<unique-id> --tmux
./run_maca_10x10.sh --resume results/shared_suite/<unique-id> --tmux
./run_maca_10x10_shaped_norm.sh --resume results/shared_suite/<unique-id> --tmux
```

Resume requires the original seeds, selectors, beta list, mode and smoke setting; incompatible command manifests are rejected. Completed task groups are skipped. It is not individual checkpoint continuation: preserve/move an incomplete task directory aside before rerunning that group in full. Process groups are terminated on failure/cancellation. Checkpoints preserve models, optimizers, target critic, update counters, MACA buffers/mixer/RNG/teacher transform and reward moments, but not simulator state or exact rollout continuation. Legacy environment-mode checkpoints without the new mode field are accepted; cross-mode loading is rejected.

Implementation sandbox access to the existing tmux socket has been restricted (`Operation not permitted`). The host-side launcher works without an interactive attach; success requires a reported window/console log. Smoke completion does not imply full training has launched.

## Original seeding and reproducibility evidence

`src/main.py` calls the unchanged `src/maca/utils.py:set_seeds()`, byte-identical to `/scratch/cluster/sooraj/Unsupervised-Multi-agent-Credit-Assignment/maca/utils.py:43`. It seeds NumPy, CPU PyTorch and CUDA, sets cuDNN deterministic mode, disables benchmarking, requests deterministic operations with the original `warn_only=True` behavior and retains `CUBLAS_WORKSPACE_CONFIG`. Sacred's Python-RNG seeding remains. Worker `i` receives `seed+i` and resets continue its private environment RNG stream. There is no new hash-seed policy, reset reseeding, loading tie rule or environment patch.

Direct reference comparison matched NumPy/PyTorch draws and flags for all four seeds. Actual subprocess tests compare worker reset observations against directly seeded `1587+i` Gymma instances, confirm distinct workers and repeatable runner creation, and verify continuing rather than repeating reset streams.

```bash
../.TPCA/bin/python scripts/check_reproducibility.py
../.TPCA/bin/python scripts/check_reproducibility.py --seed 2398 --steps 6000 --max-parallel 4
```

The audit instruments the normal entrypoint without altering actions, seeds or updates. It hashes every training/evaluation batch (state, observations, availability, actions, rewards, termination and filled masks); actor/critic/target/old-actor parameters; optimizer states, reward moments and update counters; NumPy/PyTorch/Python RNG; MACA mixer parameters/buffers, isolated and shuffle RNGs, return history, fit version, credit scale and ledgers; and non-timing numerical metrics. Timestamps, paths and wall-clock timings are excluded. Comparisons are byte-exact, not tolerance based.

On 2026-10-25, all nine method/task pairs matched across 18 fresh-process runs at seed 1587 and 6,000 steps, max concurrency four, including real fitting/nonzero credit. Report and traces: `results/reproducibility/20261025T223624Z_ehog0bc6/report.json`. Wall time was about 64.9s. This verifies that tested CPU runtime, not all future full-budget trajectories, arbitrary operations under `warn_only`, simulator checkpoint continuation, historical custom-trainer trajectories or different hardware/dependency versions.

Two distinct native-LBF observations were investigated. First, blocked movement chains can genuinely overlap players; the adapter now preserves those valid observations, tested with a native three-player regression case. Second, a synthetic object-identity loading-order example initially appeared nondeterministic, but its foods were Manhattan distance two apart, which native spawning forbids. Native food neighborhoods are disjoint, so that example does not demonstrate nondeterminism in these tasks. The corrected report marks it unreachable; 300 native resets across the tasks had minimum food distance three and no overlapping loading neighborhoods. Neither investigation modified environment dynamics or original source code.

## Historical run transfer and plotting

Historical small-task MAPPO seed 1587 moved from:

```text
/scratch/cluster/sooraj/epymarl/results/sacred/mappo/lbforaging:Foraging-8x8-2p-2f-coop-v3/2
```

to Sacred run **8** in the new repository:

```text
/scratch/cluster/sooraj/epymarl-maca/results/sacred/mappo/lbforaging:Foraging-8x8-2p-2f-coop-v3/8
```

The five original files (`config.json`, `run.json`, `metrics.json`, `info.json`, `cout.txt`) were SHA-256 checked before/after moving. Three referenced shared Sacred source files were copied with hash verification. `import_provenance.json` records source/destination IDs, checksums, suite association and runtime. The old path is a relative symlink, not a duplicate. Original metadata, including `default_label`, remains untouched; plots include the run through explicit provenance. That run used CUDA/Python 3.8/PyTorch 2.4.1 versus the current CPU/Python 3.10/PyTorch 2.5.1. The 2M budget matches the original successful baseline.

The completed default suite is `results/shared_suite/20261025T224414Z_wwvf5p9f/`: 35 completed runs plus the historical import gives 36. The separate original 10x10 MACA sweep is `20261026T142720Z_d79nn6d3`. Scientific plots use original environment returns, with evaluation/training separate. Aggregate plots use means and sample SD (`ddof=1`) across seeds, not confidence intervals or within-episode variation. For each task/metric, linear interpolation uses 401 uniform points inside support observed by every included seed/method, without smoothing, leading-zero filling or endpoint extrapolation. Final-common-return summaries can precede nominal budgets. Per-seed plots use original observations, with unfinished curves explicitly marked and not extended.

```bash
../.TPCA/bin/python scripts/plot_shared_suite.py
# Optional --suite <directory> and --output <directory>
```

The exporter validates the exact original cohort: 36 unique completed runs, four seeds in each of nine method/task groups, excluding smoke/audit/separate-network runs. Output is `<suite>/plots/`:

| Artifact | Meaning |
| --- | --- |
| `all_environments_consolidated.png` | Three task columns, training/evaluation rows |
| `8x8_2p2f_coop_consolidated.png` | Full budget plus first-500k-step detail, all methods |
| `8x8_3p2f_consolidated.png`, `10x10_3p3f_consolidated.png` | MAPPO/COMA comparisons |
| `runs.csv` | Task/method/seed, Sacred ID/path, device and historical-import lookup |
| `returns_raw.csv`, `returns_mean_std.csv` | Individual observations and aligned aggregate data |
| `final_common_returns.csv`, `plot_metadata.json` | Last-common-step values and statistical/provenance conventions |

Original export generated PNG/PDF versions of four figures, and Matplotlib was added to requirements. PDFs in that suite's plot folder were subsequently removed at user request; later ad hoc plots generated PNGs only. Comparisons excluding 1587 remove it from **every** compared method; the prior exclusion was a plotting choice. Later requests included MACA-0.1/MAPPO comparisons with/without 1587, cross-seed variance (`sample_SD**2` of seed-mean returns, not logged within-episode variance), and per-seed evaluation figures. The latest grouping puts each seed in its own panel with methods as curves. Historical runtime differences are marked. The plotting tests verify interpolation support, mean/sample SD, duplicate-step handling and nonfinite rejection; overall/detail PNGs were visually inspected and source/cohort checksums/counts verified.

## Files and validation history

| File | Purpose |
| --- | --- |
| `src/config/algs/maca_mappo.yaml`, `maca_mappo_ns.yaml` | Default normalization option declaration; shared config otherwise matches stock MAPPO |
| `src/learners/ppo_learner.py` | Small overridable normalization hook, original default math |
| `src/learners/maca_ppo_learner.py` | Mode validation, deferred pooled update, cached per-agent rewards, diagnostics and checkpoint mode/moments |
| `src/modules/maca_adapter.py` | Shared/state-only critic support, teacher transform, native observation reconstruction, manifests and adapter checkpoint metadata |
| `scripts/run_shared_suite.py` and wrappers | Matrix/task/method/beta/mode selection, bounded pool, tmux forwarding, dry run/smoke/resume |
| `scripts/check_reproducibility.py`, `tests/test_seeding.py` | Fresh-process hashes and actual worker seed/reset validation |
| `scripts/plot_shared_suite.py`, `tests/test_result_plots.py` | Exact-cohort plots and statistical checks |

Validation chronology, preserving prior evidence:

| Work | Evidence |
| --- | --- |
| Initial shared integration, 2026-10-25 | 37 tests; zero-beta and positive-beta/zero-credit default-mode parameter equivalence, once-per-batch scoring, teacher units, padding/RNG isolation and normalized checkpoint round-trip |
| Initial smoke | Failed `20261025T220740Z_60g8j3eh` exposed the overlap rejection; after correction, `20261025T220921Z_aghnfg2r` completed all 15 method/task jobs at 6k steps, all MACA mixers fitted once/nonzero credit |
| Initial five-job resource samples | Task-group times 13.92s/14.72s/15.16s, peak PSS <=2.67 GiB; execution checks, not proof of learning/eight-job throughput |
| Scope/seeding refinement | 38 tests; nine exact audit pairs and direct helper checks described above; shell/whitespace checks passed |
| Plot export | Two plotting tests; 36 unique completed records/four seeds per group, checksum verification and visual inspection |
| Original 10x10 launcher | Nine launcher tests and smoke `20261026T142505Z_od3f8oap`: three betas, 130 episodes/run, one mixer fit and 14 nonzero-shaped episodes/run |
| New post-shaping option, 2026-10-26 | All 49 tests pass, including shared pooled moments, padding/termination exclusion, no environment double-update, previous teacher moments, per-agent targets, epoch caching, beta-zero bypass, invalid-option rejection and mode-safe checkpoints with exact next fixed-batch actor/critic/moment updates |
| New post-shaping smoke | `20261026T185137Z_f_ld4orn`: both beta 0.08/0.1 seed-1587 jobs completed 6k steps; each stored 130 episodes, fitted once and shaped 14 episodes; finite positive teacher scales and manifest timing/source verified |

The new full dry run selects exactly eight 4M-step jobs, 32M budgeted steps, with `_shaped_norm` names, the correct full-sight non-cooperative environment, and tmux forwarding of both beta list and mode. Shell/whitespace checks pass. Smoke runs establish execution and real credit application, not full-budget learning performance or a completed full training launch.
