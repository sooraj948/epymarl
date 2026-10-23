# Walkthrough 75 Online random donor MACA integration

## Major design choices

1. **Extend EPyMARL through a registered learner subclass.** `MACAPPOLearner` prepares per-agent shaped rewards once per rollout batch, then uses the existing `PPOLearner` critic and actor updates. The base PPO learner, controllers, runners and episode buffer are unchanged. Native random-donor generation and the existing MACA encoder, regressor, normalization and spreading code remain in use.
2. **Keep separate feedforward actors and centralized critics.** Both new configurations derive their settings from `mappo_ns`, with hidden width 64 and the same optimization settings. Both use summed team rewards without running reward/return standardization and default to CPU.
3. **Use online random-donor splice only.** The mixer learns negative mean unstandardized GAE from the current training critics, using normalized position-action occupancies and initial/final food coordinates. Counterfactual signatures replace the ego occupancy using uniformly generated donor trajectories while retaining factual goal context. This integration does not perform native action replay or frozen-critic experiments.
4. **Keep n-step PPO and document its importance.** The policy uses EPyMARL's `q_nstep=5` targets. The mixer teacher uses GAE with lambda 0.95. This difference from historical MACA policy training can materially affect the outcome and remains a major open issue for the later four-condition n-step/GAE comparison in walkthrough 73.
5. **Make credit independent of policy RNG and evaluation.** MACA initialization, donor generation, fitting and sampling use separately maintained NumPy, CPU PyTorch and Python RNG state. Teacher scoring has no gradients. Evaluation does not process episodes through the credit pipeline. A zero beta bypasses the adapter and matches the baseline update exactly in the focused test.
6. **Use the native LBF horizon by default.** The registered `Foraging-8x8-2p-2f-coop-v3` task has a native 50-step limit, so both new configs explicitly set `time_limit=50`. The old plan's illustrative 200-step wrapper setting would not extend that native limit. Changing the wrapper limit can shorten episodes; native environment constructor overrides are intentionally rejected by this adapter so factual and donor tasks stay consistent.

## Run MACA

The current workspace has a ready Python 3.10 environment at `../.TPCA`. From the fork:

```bash
cd /scratch/cluster/sooraj/epymarl-maca
../.TPCA/bin/python src/main.py --config=maca_mappo_ns --env-config=gymma with seed=1587
```

Run the matched baseline with:

```bash
../.TPCA/bin/python src/main.py --config=maca_baseline_ns --env-config=gymma with seed=1587
```

Both use a one-million-environment-step budget. EPyMARL completes whole parallel rollout batches and its existing loop runs while `t_env <= t_max`, so the final count can exceed that budget by approximately one rollout batch. Episodes terminate at the native horizon or earlier. Ten environments supply ten fresh episodes per update; `buffer_size`, `batch_size` and `batch_size_run` must remain equal when changing parallelism.

Example budget and beta overrides:

```bash
../.TPCA/bin/python src/main.py --config=maca_mappo_ns --env-config=gymma with seed=2398 t_max=2000000 maca.beta=0.05
```

Select another square, full-sight native cooperative LBF v3 task by overriding `env_args.key`. The adapter checks dimensions, observation ordering and available sight. It currently rejects additional constructor settings, pretrained wrappers, recurrent/shared architectures and reward standardization; supporting those requires matching donor and critic behavior explicitly.

## Default settings and credit timing

| Setting | Default |
| --- | --- |
| Environment | `lbforaging:Foraging-8x8-2p-2f-coop-v3` |
| Native and wrapper horizon | 50 |
| Device | CPU |
| Batch episodes and parallel environments | 10 |
| Policy target | EPyMARL n-step, `q_nstep=5` |
| Teacher score | Negative mean GAE, lambda 0.95 |
| Beta cap | 0.1, an initial setting rather than a tuned optimum |
| Dynamic beta threshold and ramp | -0.01 and 0.35 |
| Uniform-random donor trajectories | 16, fixed per seed and task |
| Mixer readiness | At least 50 stored episodes |
| Fit frequency | Every 100 episode indices, with zero-based indexing |
| Mixer | Existing 128/64 regressor, Adam 0.001, 20 fitting epochs, minibatch 64 |
| Buffer capacity | 500 winning and 500 losing episodes |
| Additional warmup | 0 episodes |
| Credit eligibility | Nonpositive environment-return episodes; `maca.shape_wins=True` includes positive returns |
| Injection | Signed, uniform across the episode's actual transitions |

With these defaults, the first fit normally occurs at index 100, the 101st training episode. Eligibility and the performance gate still determine which episodes receive credit. For an all-failure return window of zero, effective beta is `0.1 * (0 - (-0.01)) / 0.35`, approximately 0.002857. Credit need not be large immediately after the first fit.

Each valid transition receives `team_reward[t] + effective_beta * normalized_credit[i] / episode_length`. Final observations and padded timesteps receive no additional reward. Mixer targets use factual environment rewards; the live training critics themselves learn from shaped rewards, as in the selected online MACA variant. The adapter scores each episode once before optimizer updates and reuses those rewards across all four PPO epochs.

## Files and entrypoints

| File | Responsibility |
| --- | --- |
| `src/learners/maca_ppo_learner.py` | Reward preparation, existing critic-update hook, MACA checkpoint handling and timing |
| `src/modules/maca_adapter.py` | Observation/state conversion, live critic bridge, selected pipeline construction, RNG isolation, logging and adapter state |
| `src/maca/credit_pipelines.py` | Optional scorer injection and removal of the random-bank loader's auxiliary-script helper import |
| `src/learners/__init__.py` | New learner registry entry |
| `src/config/algs/maca_mappo_ns.yaml` | Runnable MACA experiment |
| `src/config/algs/maca_baseline_ns.yaml` | Matched PPO baseline |
| `src/envs/__init__.py` | Import SMAClite only when selected, so LBF does not require unrelated SMAClite dependencies |
| `requirements-maca.txt` | Minimal LBF/MACA dependencies, avoiding the old base file's PyYAML 5.3.1 pin |
| `tests/test_maca_integration.py` | Nine focused integration checks |

`src/main.py` loads the algorithm and environment YAMLs, and `src/run.py` creates the registered learner. `MACAPPOLearner.train` calls `adapter.prepare` before delegating to `PPOLearner.train`. Its `train_critic_sequential` override passes the cached rewards into the existing critic update. The actor then uses the resulting existing n-step advantages.

## Outputs and checkpoints

- `results/sacred/maca_mappo_ns/<task>/<run>/` contains Sacred's resolved configuration, status and normal EPyMARL metrics. Training/evaluation returns remain original summed environment rewards. The existing soft-policy action selector samples during evaluation; baseline and MACA use the same behavior.
- `results/maca/<unique run token>/manifest.json` records the fork commit/dirty state, resolved MACA and EPyMARL settings, package versions and device. `random_donors.json` stores donor metadata and requested actions. `credit_episodes.jsonl` stores one record per training episode, including teacher target, readiness/eligibility, counterfactual scores, applied per-agent credit and original return.
- Sacred also records `maca/episodes`, `maca/fit_version`, `maca/scored_fraction`, `maca/effective_beta`, applied credit by agent, and separate MACA preparation and PPO update times.
- `results/models/<unique run token>/<t_env>/` contains normal EPyMARL weights/optimizers plus `maca_state.th`. Model saving defaults to every 100,000 environment steps, with EPyMARL's usual initial checkpoint.

MACA state saves mixer weights and shuffle-generator state, stored episode records/targets, donor data, running normalization, counters, return history and independent RNG state. The subclass also saves target-critic state and target-update counters and synchronizes the old policy on load. Configuration mismatches are rejected. Exact continuation of simulator trajectories is not guaranteed because EPyMARL model checkpoints do not contain all environment/runner states.

For evaluation, set `checkpoint_path` to the parent directory containing numeric timestep directories and use the same environment, seed and MACA settings as training:

```bash
../.TPCA/bin/python src/main.py --config=maca_mappo_ns --env-config=gymma with seed=1587 checkpoint_path='results/models/<unique run token>' evaluate=True
```

## Runtime preparation

The old Python 3.8 environment was left unchanged. A new `../.TPCA` Python 3.10 environment uses the available system numerical libraries and locally available pure-Python environment/test packages; downloads were unavailable during implementation. `pip check` reports no broken requirements. This environment runs the commands above without accessing the original MACA source repository.

On another machine, use a compatible Python 3.10 environment and install the minimal requirements normally:

```bash
python3.10 -m venv ../.TPCA
../.TPCA/bin/python -m pip install -r requirements-maca.txt
```

Tested versions: PyTorch 2.5.1, NumPy 2.2.1, Gymnasium 1.1.1, LBF 2.0.0, scikit-learn 1.6.1, Sacred 0.8.7, PyYAML 6.0.2, SciPy 1.15.1 and pytest 8.3.5. Optional historical MACA variants are not part of this minimal dependency set.

## Validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../.TPCA/bin/python -m pytest -q tests/test_maca_integration.py
```

All nine checks passed. They cover observation reconstruction and teacher values/negative-GAE calculations, gradient-free scoring, signed injection and padding, initialization/processing RNG isolation, exact beta-zero parameter-update agreement, real pipeline readiness and nonzero credit, one preparation per PPO batch, policy updates, checkpoint round trips with identical next credits, rejection of unsupported settings, and matched configuration values.

End-to-end checks through the actual Sacred entrypoint also completed:

| Check | Result |
| --- | --- |
| Short 5-step run, 2 environments, 600-step requested budget, accelerated fit settings | 122 training episodes, 73 fits and 73 episodes with nonzero applied credit |
| Normal 50-step task, 10 environments, default MACA settings, 6,000-step requested budget | 130 training episodes, one fit and 30 episodes with nonzero applied credit; saved model at 6,500 steps |
| Matched baseline entrypoint, 500-step requested budget | Completed |
| Evaluation of the normal-run checkpoint | Completed, checkpoint hash unchanged and no training credit records generated |

The end-to-end checks verify execution and credit delivery, not learning improvement. Gymnasium reports its scalar-reward warning for native multi-agent LBF's list rewards; the gymma wrapper sums those rewards as configured. Existing PyTorch checkpoint calls can report future-default warnings. These did not prevent validation.

## Major open issue N step policy training and GAE teacher scoring

The initial experiment asks whether MACA improves the matched EPyMARL n-step baseline. It does not reproduce the historical MACA trainer: EPyMARL uses n-step targets, updates its critic during each PPO epoch, and does not apply the historical per-agent GAE advantage standardization. Reward normalization, clipping defaults and episode ordering can also differ from old experiments.

The mixer teacher's negative mean GAE is an episode-level regression target, not the policy update estimator. A disappointing or changed result should prompt the planned controlled comparison of n-step baseline, n-step plus MACA, GAE baseline and GAE plus MACA with all other settings matched. That discussion should explicitly address advantage freezing/standardization and terminal versus time-limit bootstrapping. This remains an important design question rather than an issue resolved by successful integration tests.
