# Walkthrough 77 MACA and COMA experiments with a shared job pool

## Major design choices

1. **Maintain a single EPyMARL fork containing MACA.** `src/maca/` holds the imported algorithm components; EPyMARL owns rollout collection, actors, critics, optimization and evaluation. The original MACA repository remains historical material and is not an import dependency.
2. **Keep the selected online random-donor MACA variant.** Separate feedforward actors and centralized critics use `mappo_ns`. The mixer learns negative mean unstandardized GAE and provides signed per-agent reward shaping. EPyMARL's policy still uses five-step PPO targets. This estimator difference remains a major open experimental issue.
3. **Use stock COMA as requested.** The launcher selects the existing `coma.yaml` without modifying its algorithm settings: shared recurrent actors, `basic_mac`, the COMA action-value critic, reward standardization, ten-step returns, and entropy coefficient 0.001. It does not substitute `coma_ns`. Environment, CPU selection, budget, evaluation schedule and checkpoint frequency follow the shared launch protocol.
4. **Fill one bounded pool across methods.** Jobs are queued by method then seed, but any completed job immediately frees a slot for the next job. There is no barrier between baseline, MACA and COMA. Fewer seeds can therefore run different methods concurrently. The portable default is four training jobs; this system's pilot supports selecting eight.
5. **Measure rather than infer concurrency from host RAM alone.** The launcher reports CPU affinity, scheduler and cgroup limits, memory headroom and host load. It samples proportional set size (PSS) across the training process trees, avoiding repeated counting of fork-shared PyTorch pages. Benchmark summaries report elapsed time, budgeted throughput and memory alongside before/after resource snapshots.
6. **Isolate policy RNG and keep evaluation unshaped.** MACA initialization, donor generation, fitting and sampling have independent NumPy, CPU PyTorch and Python RNG state. Episodes are scored once before each PPO update. Collected rewards and evaluation returns remain original summed team rewards. Beta zero bypasses the adapter and reproduces the matched baseline update in tests.

This consolidates walkthroughs 73 through 77 under the repository's five-walkthrough rule. Earlier files link here. Original full historical versions remain in Git; for example, `git show 20aa69a:docs/walkthrough-75.md` and `git show eddadea:docs/walkthrough-76.md` recover the integration and first launcher documents.

## Run all three methods

```bash
cd /scratch/cluster/sooraj/epymarl-maca
./run_maca_comparison.sh --max-parallel 8
```

Defaults are four seeds (`1587 2398 2875 3370`), three methods (`baseline maca coma`), CPU, one million environment steps **per job**, and the 50-step native `Foraging-8x8-2p-2f-coop-v3` task. This schedules 12 jobs with at most eight trainers and 80 environment workers at once. The workers spend much of their time waiting for actions; 88 processes do not imply 88 continuously busy CPU cores. Numerical library threads are limited to one per trainer.

The name `baseline` selects `maca_baseline_ns.yaml`, `maca` selects `maca_mappo_ns.yaml`, and `coma` selects the original `coma.yaml`. With no `--max-parallel` argument, the pool size remains four. There is no automatic reuse of earlier experiments; each invocation has a unique output directory and shared Sacred label.

Select fewer seeds, a different supported LBF task and a larger budget:

```bash
./run_maca_comparison.sh --seeds 1587 2398 --max-parallel 8 --env Foraging-10x10-3p-2f-coop-v3 t_max=2000000 maca.beta=0.05
```

This schedules six jobs and starts all six together. Ordinary Sacred `key=value` overrides apply to every selected method. `maca.*` overrides apply only to MACA. `--env-name` is an alias for `--env`; `env_args.key=...` also works. Bare `Foraging-...` identifiers receive the `lbforaging:` prefix automatically. Conflicting environment options, repeated seeds/methods and invalid concurrency are rejected before launching.

Select only a subset of methods:

```bash
./run_maca_comparison.sh --methods baseline maca --max-parallel 8
./run_maca_comparison.sh --methods coma --seeds 1587 2398 --max-parallel 2
```

The launcher accepts registered Gymma task identifiers. MACA remains specific to square, full-sight, native cooperative LBF v3 tasks: its state reconstruction, occupancy encoder and donors cannot automatically generalize to another environment family. For other Gymma tasks, select `--methods baseline coma` and install the relevant optional environment dependencies. Additional LBF constructor overrides and pretrained wrappers remain unsupported by this MACA adapter so the factual and donor tasks cannot silently differ.

Use `--dry-run` to inspect exact commands without training or output-directory creation. Seed, learner, experiment name/label, evaluation and checkpoint continuation are controlled by the launcher. Use the direct entrypoint for individually configured evaluation or restart runs. `MACA_PYTHON` can select another compatible interpreter using its absolute executable path.

## Check whether eight jobs are viable

```bash
./run_maca_comparison.sh --check-resources --max-parallel 8
./run_maca_comparison.sh --benchmark --max-parallel 4
./run_maca_comparison.sh --benchmark --max-parallel 8
```

Run the two benchmarks sequentially with the same methods, seeds, environment and overrides. Benchmark mode changes the default budget to 20,000 steps per job and disables checkpoint saving. It retains the training algorithms, mixer fitting, environment horizon and evaluation schedule. It can be overridden, for example with `--benchmark t_max=40000` for a longer pilot.

The session inspected on 2026-10-23 had affinity to 128 logical CPUs on a host with 64 physical cores, about 229.4 GiB of available memory, no detected CPU quota and no scheduler CPU allocation variables. These are shared capacities, not an exclusive reservation. The resource reader checks both cgroup v1 and v2 and their ancestors; reading only v2 `cpu.max`/`memory.max` would have missed this system's v1 controllers.

The same 12 jobs, each with a 20,000-step budget, were measured sequentially:

| Pool size | Total elapsed | Budgeted steps per second | Peak sampled pool PSS |
| --- | ---: | ---: | ---: |
| 4 | 63.30 seconds | 3,791.5 | 2.45 GiB |
| 8 | 36.96 seconds | 6,494.2 | 4.25 GiB |

Eight gave 1.71 times the total throughput and reduced elapsed time by about 42 percent. Host swap-in and swap-out counters did not increase during either pilot. All 24 pilot jobs completed, and runtime configurations confirmed stock COMA in all eight COMA runs. This supports eight on the inspected system and workload; repeat the pilot when task size, host load, interpreter or resource allocation changes.

The measurement includes startup, evaluation, rollout collection, credit fitting, learning and resource monitoring. Throughput uses the requested budgets, not an exact final transition count: EPyMARL completes whole batches and runs while `t_env <= t_max`, so actual counts can exceed the budget slightly. PSS is sampled, and the short pilot does not establish the maximum memory footprint of a full training history.

While a long comparison runs, use `vmstat 1`, `free -h` or `top` in another terminal. Prefer a larger pool when total throughput improves, memory remains comfortably below the effective limit, and there is no sustained new paging. Existing used swap is different from ongoing swap-in/out. Do not run the two throughput pilots simultaneously.

Pilot summaries are in `results/comparisons/20261023T210607Z_in93bv8w/summary.json` (four) and `results/comparisons/20261023T210810Z_z0nvj_mv/summary.json` (eight). These local experiment artifacts are ignored by Git.

## Runtime and installation

The current workspace has a ready Python 3.10 environment at `../.TPCA`. The old Python 3.8 virtual environment was left unchanged. The copied MACA annotations require Python 3.9 or newer; Python 3.10 is the tested runtime. With package downloads unavailable during setup, the new environment used available system numerical libraries and locally available pure-Python packages. `pip check` reports no broken requirements.

On another machine:

```bash
python3.10 -m venv ../.TPCA
../.TPCA/bin/python -m pip install -r requirements-maca.txt
```

The minimal requirements avoid EPyMARL's old PyYAML 5.3.1 pin and include native LBF, Sacred, scikit-learn, pytest and psutil. Tested versions are PyTorch 2.5.1, NumPy 2.2.1, SciPy 1.15.1, scikit-learn 1.6.1, Gymnasium 1.1.1, LBF 2.0.0, Sacred 0.8.7, PyYAML 6.0.2, pytest 8.3.5 and psutil 6.1.1. Optional historical UMAP/Wasserstein variants and other EPyMARL environments can require additional dependencies.

For direct single runs, without the comparison launcher:

```bash
../.TPCA/bin/python src/main.py --config=maca_mappo_ns --env-config=gymma with seed=1587
../.TPCA/bin/python src/main.py --config=maca_baseline_ns --env-config=gymma with seed=1587
../.TPCA/bin/python src/main.py --config=coma --env-config=gymma with seed=1587 use_cuda=False t_max=1000000 env_args.key=lbforaging:Foraging-8x8-2p-2f-coop-v3 env_args.time_limit=50
```

The comparison launcher applies common budget/evaluation/checkpoint settings explicitly; a direct stock COMA command otherwise retains its own run-control defaults as well as its algorithm defaults.

## MACA settings and the PPO hook

| Setting | MACA default |
| --- | --- |
| Actor and critic structure | Separate feedforward actors and centralized critics, hidden width 64 |
| Policy update | Existing EPyMARL PPO, four epochs, five-step critic targets |
| Rewards | Common summed team rewards, no running reward or return standardization |
| Parallel episodes, batch size and buffer size | 10 each; keep these equal for fresh episode batches |
| Teacher | Live training critics averaged across agents, negative mean unstandardized GAE, lambda 0.95 |
| Representation | Per-agent normalized position/requested-action occupancy plus initial/final food coordinates |
| Mixer | Existing 128/64 regressor, Adam 0.001, 20 fitting epochs, minibatch 64 |
| Random donors | 16 uniform joint-policy trajectories, fixed by seed and task |
| Credit scale | Existing exponential moving maximum-absolute-credit normalization; no clipping |
| Beta cap, threshold and ramp | 0.1, -0.01 and 0.35 |
| Return window | Latest 100 training episodes |
| Readiness, fit interval and warmup | At least 50 stored episodes; fit every 100 zero-based episode indices; additional warmup 0 |
| Storage | 500 winning and 500 losing episodes |
| Eligibility | Nonpositive environment return; `maca.shape_wins=True` includes positive returns |
| Injection | Uniform across the episode's actual transitions |

The beta cap is an initial setting, not a tuned optimum. At a zero running return its effective value is `0.1 * 0.01 / 0.35`, approximately 0.002857. The first default fit normally occurs at index 100, the 101st training episode. Readiness, eligibility and the performance gate still determine whether credit is delivered.

For each agent, `r_shaped[t,i] = r_team[t] + effective_beta * normalized_credit[i] / episode_length`. Signed credits are preserved. Each episode's own `filled` entries determine its transition count and final observation; padding and the final observation receive no credit. Warmup, fit frequency and running return count training episodes, not PPO epochs, evaluations or timesteps. Episodes are processed in deterministic sampled batch order, which need not reproduce historical sequential collection order.

Full-sight observations reconstruct public food/player coordinates and levels while preserving observer-first agent order. They round-trip to native observations; hidden simulator history is not fabricated. Teacher values come from the actual current EPyMARL critic with no gradients and fixed parameters during scoring of a whole batch. Teacher GAE retains the historical finite-horizon zero bootstrap at episode end. Its factual reward sequence is unshaped, while the online critics themselves learn from shaped rewards, matching the selected online variant.

`MACAPPOLearner.train` prepares rewards once, then delegates to `PPOLearner.train`. Its `train_critic_sequential` override supplies cached rewards on every PPO epoch. Both critic targets and policy advantages therefore use shaped rewards. The base PPO learner, runners, controllers, main entrypoint and episode buffer remain unchanged. The only environment-registry adjustment imports SMAClite when selected, allowing LBF to run without that unrelated dependency.

## Major open issue N step PPO and GAE

**Retaining EPyMARL n-step PPO may be a major experimental issue.** The mixer teacher's negative mean GAE is an episode regression target; it does not mean the policy trains with GAE. Historical MACA trained its policy with GAE on shaped rewards and standardized advantages across each agent's transition batch. EPyMARL instead uses n-step residual advantages and updates the critic/recomputes advantages during PPO epochs. Reward normalization, clipping defaults, episode ordering and bootstrap conventions can also differ from old experiments.

The current comparison asks whether MACA improves this matched EPyMARL n-step baseline. It does not reproduce the historical trainer or establish equal performance with GAE. Discuss a controlled study of n-step baseline, n-step plus MACA, GAE baseline and GAE plus MACA if results differ materially. Agree explicitly on frozen versus recomputed advantages, standardization and terminal/time-limit bootstrapping; replacing a return formula alone does not reproduce the historical PPO semantics.

Stock COMA is deliberately a different algorithm baseline with shared recurrent actors and reward standardization. The MACA/base MAPPO comparison matches its settings apart from credit; the COMA comparison does not isolate the credit mechanism alone.

## Files and outputs

| File | Responsibility |
| --- | --- |
| `run_maca_comparison.sh` | Executable wrapper selecting the runtime |
| `scripts/run_comparison.py` | Seed/method/environment selection, common and MACA-only overrides, shared queue, resources, benchmark summaries and cleanup |
| `src/learners/maca_ppo_learner.py` | PPO reward hook, timing, MACA checkpoints and old-policy synchronization on load |
| `src/modules/maca_adapter.py` | Native trajectory/state conversion, live critic bridge, selected pipeline, independent RNG, manifests and credit logs |
| `src/maca/credit_pipelines.py` | Injected scorer support; random-bank loader no longer imports an auxiliary experiment script for its done helper |
| `src/learners/__init__.py` | Registered MACA learner with MACA-specific imports deferred until construction |
| `src/config/algs/maca_baseline_ns.yaml`, `maca_mappo_ns.yaml` | Matched unstandardized CPU MAPPO configurations |
| `src/config/algs/coma.yaml` | Unchanged upstream COMA settings |
| `tests/test_maca_integration.py`, `test_comparison_launcher.py` | Algorithm bridge, configuration, resource, scheduling and cleanup checks |

Each comparison writes `results/comparisons/<unique ID>/commands.json`, per-method/seed console logs and `summary.json`. The summary records success/failure/cancellation, completed-job exit codes and elapsed times, pool PSS/RSS peaks, budgeted throughput and resource snapshots. Process groups allow failure or cancellation to terminate trainers and environment workers; queued jobs are not launched after failure. Numerical threads are limited for child jobs without changing the caller's environment.

The launcher sets `local_results_path` to that comparison directory, so its MACA manifests/credit logs are under `<comparison>/maca/` and models under `<comparison>/models/`. Direct runs retain `results/maca/` and `results/models/`. Sacred metrics remain under `results/sacred/<base algorithm name>/<task>/<run>/`; the resolved configuration's label identifies the comparison. The existing soft-policy selector samples during evaluation for all methods using it.

MACA's per-run `manifest.json` records the fork commit/dirty status, package versions, device and resolved settings. `random_donors.json` records donor metadata and requested actions. `credit_episodes.jsonl` has one row per training episode with original return, teacher target, readiness/eligibility, fit version, counterfactual scores and signed applied credit. Sacred also records MACA counters, scored fraction, effective beta, per-agent credit and preparation/learner timings. Original training/evaluation returns remain unshaped.

The launch protocol saves models every 100,000 environment steps, with EPyMARL's usual initial checkpoint. MACA adds `maca_state.th` to normal model/optimizer files, saving mixer weights and shuffle RNG, episode records/targets, donor data, normalization, counters, return window and independent RNG. Target-critic state and update counters are also preserved. Task/configuration mismatches are rejected. Exact trajectory continuation is not guaranteed because the original model checkpoints do not save all environment, runner and training-buffer state. COMA's own checkpoint behavior is unchanged.

Evaluate a MACA checkpoint using its parent directory containing numeric timestep directories and the same environment, seed and MACA settings:

```bash
../.TPCA/bin/python src/main.py --config=maca_mappo_ns --env-config=gymma with seed=1587 checkpoint_path='results/comparisons/<ID>/models/<unique run token>' evaluate=True
```

Evaluation does not fit the mixer or update credit statistics. Per-agent preparation is never performed once per PPO epoch. No new packaging project, editable installation or source-repository `PYTHONPATH` is required; `src/main.py` already puts `src/` on the import path. Unit tests use the same source root.

## Validation and design history

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../.TPCA/bin/python -m pytest -q tests
```

All 24 checks pass. They cover native state/observation round trips, critic values and negative-GAE calculations, gradient-free scoring, signed uniform injection and padding, policy RNG isolation, exact beta-zero update agreement, real readiness/nonzero credit, one preparation per batch, finite policy updates, checkpoint round trips with identical next credits, matched MAPPO configs, invalid-setting rejection, seed/method/environment routing, unchanged stock COMA choices, v1/v2 ancestor limits, cross-method queue filling, bounded concurrency, failure/cancellation cleanup including descendants, and dry runs from another directory.

End-to-end validation progressed as follows:

| Check | Result |
| --- | --- |
| Initial 5-step MACA run, accelerated fit settings, 600-step requested budget | 122 episodes, 73 fits and 73 episodes with nonzero credit |
| Normal 50-step MACA run, 6,000-step requested budget | 130 episodes, one fit, 30 nonzero-credit episodes, model saved at 6,500 steps |
| Matched baseline, 500-step requested budget | Completed |
| Saved MACA model evaluation | Completed; checkpoint hash unchanged and no credit episodes generated |
| Stock COMA smoke run | Completed |
| All three methods on 10x10, three-agent cooperative LBF | Completed |
| Twelve-job pilots at four and eight concurrency | All 24 jobs completed; measurements above |

These checks establish execution and credit delivery, not learning improvement. Native multi-agent LBF can trigger Gymnasium's scalar-reward warning before gymma sums its reward list; existing PyTorch checkpoint calls can emit future-default warnings. Neither prevented validation. Shell syntax, whitespace checks and `pip check` also pass.

## Source provenance and repository setup

The fork is `git@github.com:sooraj948/epymarl.git`, based on inspected EPyMARL commit `cbc38c09588064eab978501d0f12c2cf58fa7fc2`. Sixteen tracked MACA files were copied from `893f188e8bb918ec099150510a6c0edb7829022c` into `src/maca/`. They initially matched byte for byte; subsequent integration changes are versioned here. Commit `f4a180a` imported source and plans, `20aa69a` integrated MACA, and `eddadea` added the first fixed-four-seed launcher. That launcher used separate method stages; the current shared pool supersedes those barriers.

The original two-maintained-repository/editable-package proposal was abandoned in favor of this single fork. Historical replay/reset-oracle, frozen critic, dense reward and diagnostic experiments remain outside the first integration. Their old trainers and results remain in the original repository; future active code and tests belong here. Do not import old training scripts or `auxillary` experiment scripts to obtain small helpers.

For a fresh setup, create the GitHub fork, clone it and import the tracked MACA library and relevant documentation once:

```bash
git clone git@github.com:sooraj948/epymarl.git epymarl-maca
cd epymarl-maca
git rev-parse HEAD
# For an initial import only, when src/maca does not already exist:
cp -R ../Unsupervised-Multi-agent-Credit-Assignment/maca src/maca
```

During the original setup GitHub DNS/SSH access was unavailable, so the existing local EPyMARL checkout was cloned with `--no-hardlinks` and `origin` was set to the supplied fork. Git history and independent object storage were preserved; the original checkouts and results stayed intact. Local result artifacts are ignored. The root `AGENTS.md` preserves the user's development conventions. Import checks used `PYTHONPATH=src` under Python 3.10; `git fsck` and the setup checks passed.

Stage explicit source paths, review diffs and commit locally; publish with `git push origin main` from a shell with GitHub access. An upstream remote is optional, and changes do not require coordinated commits in another repository. Keep the selected baseline version fixed during a study and record its actual fork commit and dirty state in every experiment manifest. Documentation is updated after implementation and validation.
