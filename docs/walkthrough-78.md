# Walkthrough 78 Shared MAPPO MACA and stock COMA suite

## Major design choices

1. **Anchor MACA to the successful stock shared MAPPO.** Vanilla uses the existing `mappo.yaml`. New `maca_mappo.yaml` matches every algorithm setting except learner/name and adds the existing MACA options. Both use shared recurrent actors, hidden size 128, an agent-conditioned shared centralized critic, entropy coefficient 0.001, reward standardization, soft target updates at 0.01, and five-step PPO. No advantage standardization is added. Stock COMA keeps every algorithm setting in `coma.yaml`.
2. **Keep per-agent shaping and consistent reward units.** Environment-only running reward moments update once per PPO batch using the unchanged base learner. MACA adds signed individual credit in raw reward units, then applies that same mean/std to each agent's shaped reward. The teacher transforms environment rewards with the same frozen batch moments before combining them with critic values. Separate agent rewards, return targets and advantages remain separate despite shared network parameters. Evaluation and return-based gating use original summed environment rewards.
3. **Use online random-donor no-replay splice.** The existing occupancy/goal regressor, random donor bank, negative mean GAE teacher, uniform temporal injection, loss-only shaping and dynamic beta schedule remain. Beta values 0.08, 0.1 and 0.15 are the maximum coefficients in that schedule. No recorded-action replay, simulator counterfactual replay, or frozen external critic is introduced.
4. **Use the final requested task variants.** Keep forced cooperation only for the existing 8x8 two-player task. Both larger tasks use native standard LBF without `-coop`, with full sight and a 50-step horizon. The README prescribes 50 steps for LBF and supplies no task-specific MAPPO/COMA overrides; retain the stock algorithm hyperparameters and change only the environment and requested training budget.
5. **Run task groups sequentially with up to eight trainers within each group.** Fill slots across methods/betas/seeds until the group finishes; then advance to the next environment. Skip the historical successful seed 1587 only for the first task's vanilla MAPPO. All other experiments use all four seeds. Default suite size is 59 jobs and 158M budgeted training steps.
6. **Preserve existing runs and tmux panes.** The new launcher writes a unique suite directory. `--tmux` creates a detached window in the existing `epymarl-maca` session rather than sending commands to a potentially busy pane. It records a persistent console log. Earlier separate-network configs and launcher remain available.

## Experiment matrix

All environments expose native flat full-sight observations. Each task runs vanilla MAPPO, stock COMA and MACA at each of the three betas.

| Order | Environment | Steps per run | Vanilla seeds | COMA and each MACA beta seeds | Jobs |
| --- | --- | ---: | --- | --- | ---: |
| 1 | `lbforaging:Foraging-8x8-2p-2f-coop-v3` | 2,000,000 | 2398, 2875, 3370 | 1587, 2398, 2875, 3370 | 19 |
| 2 | `lbforaging:Foraging-8x8-3p-2f-v3` | 2,000,000 | 1587, 2398, 2875, 3370 | 1587, 2398, 2875, 3370 | 20 |
| 3 | `lbforaging:Foraging-10x10-3p-3f-v3` | 4,000,000 | 1587, 2398, 2875, 3370 | 1587, 2398, 2875, 3370 | 20 |

The first task's 2M budget matches the successful historical run, recorded under `/scratch/cluster/sooraj/epymarl/results/sacred/mappo/lbforaging:Foraging-8x8-2p-2f-coop-v3/2/`. Its original seed-1587 results are retained in place rather than copied or rerun. Historical CUDA/Python/PyTorch settings differ from the current CPU/Python-3.10 runtime; learning curves across those runtimes are not bitwise reproducibility checks.

Common launch settings are CPU, summed team environment reward, ten fresh episodes per update, 100 greedy evaluation episodes every 50k training steps, diagnostic logging every 10k steps, and checkpoints every 100k steps. The pool caps the number of trainers at eight; each trainer starts ten environment workers. Thread-count environment variables remain one per trainer as in the existing resource-tested launcher.

## Reward handling and shared critic

For raw team reward `r[t]`, episode length `T`, normalized signed MACA credit `c[i]`, and effective scheduled beta `b`, the policy/critic target reward is:

```text
r_shaped[t,i] = r[t] + b * c[i] / T
r_train[t,i]  = (r_shaped[t,i] - environment_mean) / environment_std
```

The moments are computed from unshaped environment rewards exactly once by upstream PPO, including its existing treatment of padding. They are common to all agents and exclude intrinsic credit, so normalization does not independently recenter each agent's shaping. Padding is ignored by the unchanged PPO masks. The beta schedule and win/loss classification use raw episode returns.

The teacher uses `(r[t] - environment_mean) / environment_std` with the live critic's average across agents. It computes negative mean GAE without standardizing those advantages. Critic parameters and reward moments remain fixed throughout all episode scoring in a batch. The online critic is trained on shaped normalized returns, as the previous online scorer used a critic trained on shaped raw returns. An independent environment-only teacher critic is not introduced.

Preparation occurs inside the first `train_critic_sequential` call: upstream reward statistics have been updated, but no critic optimization has occurred. Prepared per-agent rewards are cached across all PPO epochs. The upstream PPO learner, shared actor, shared critic and COMA implementations are unchanged. The shared critic already appends an agent one-hot identity; its output can represent a different shaped value for each agent.

The teacher's reward transform is new when using standardized MAPPO. It avoids subtracting normalized critic values from raw rewards. It can change mixer target scales relative to the earlier unstandardized experiments; manifests explicitly record teacher units, policy shaping convention and moment source. Credit ledgers include the frozen reward mean/std. MACA checkpoints now also preserve reward moments and the teacher transform.

**Major open issue remains:** the policy uses EPyMARL five-step targets while the mixer teacher uses GAE with lambda 0.95. Existing terminal/bootstrap conventions also remain. This integration does not resolve those estimator differences. The teacher GAE is not the policy advantage estimator.

## Commands and output

```bash
cd /scratch/cluster/sooraj/epymarl-maca
./run_shared_suite.sh --dry-run
./run_shared_suite.sh --tmux
```

The second command launches in a new detached `shared-suite` window of the existing `epymarl-maca` session, prints its window ID and console-log path, then returns. To run in the current terminal instead, omit `--tmux`. The wrapper selects `../.TPCA/bin/python` without requiring activation; `MACA_PYTHON` can override it. `--max-parallel` defaults to eight, and `--seeds` can select a subset.

Each run creates `results/shared_suite/<unique-id>/` with `commands.json`, `suite_status.json`, and one directory per task. Task directories contain per-method/beta/seed logs, resource `summary.json`, models and MACA artifacts. Sacred continues writing to `results/sacred/<base-config-name>/<environment>/<run-id>/`; the suite label and method-specific run name identify the appropriate runs. Existing pool behavior terminates active process groups on failure or cancellation.

```bash
./run_shared_suite.sh --resume results/shared_suite/<unique-id> --tmux
```

Resume requires the original seed selection and smoke setting and skips completed task groups. It does not resume individual checkpoints. If a task group failed, move its entire task directory aside before resuming; that group will be rerun in full while already completed groups remain untouched. An interrupted group without a completed summary must likewise be preserved elsewhere before reuse. Simulator state and exact rollout continuation are not checkpointed.

## Files and validation

| File | Change |
| --- | --- |
| `src/config/algs/maca_mappo.yaml` | Stock shared MAPPO plus existing MACA options |
| `src/modules/maca_adapter.py` | Shared actor/critic support; standard LBF support; teacher reward transform; normalization metadata/checkpoint state |
| `src/learners/maca_ppo_learner.py` | Prepare after reward-stat update, preserve per-agent shaping and cache across epochs, checkpoint reward statistics |
| `scripts/run_shared_suite.py`, `run_shared_suite.sh` | Sequential task groups, bounded pool, exact experiment matrix, tmux launch, dry run, smoke and task-level resume |
| `tests/test_maca_integration.py`, `tests/test_shared_suite.py` | Shared/normalized update equivalence, individual rewards, teacher units, checkpoint state, native movement edge case, matrix and sequential groups |

Validation completed on 2026-10-25:

- All **37 tests** pass across MACA integration, the existing comparison launcher and the new suite. Zero beta and positive-beta/zero-credit shared MAPPO updates match stock PPO actor/critic parameters exactly. Tests verify moments update once, individual targets survive normalization, teacher reward units match critic values, RNG is isolated, and normalized checkpoints round-trip.
- `./run_shared_suite.sh --smoke` completed all **15 jobs**, using one seed and 6,000 steps for each method/beta/task. Every MACA run fitted its mixer once and applied nonzero shaping. Successful smoke suite: `results/shared_suite/20261025T220921Z_aghnfg2r/`. Five-job task-group wall times were 13.92s, 14.72s and 15.16s; peak sampled PSS was at most 2.67 GiB. This checks execution and credit application, not learning performance or eight-job throughput.
- The first smoke attempt exposed an invalid adapter assumption: native LBF can overlap players after a blocked movement chain. The adapter now preserves these genuine observations rather than rejecting them, with a native three-player regression case. No environment dynamics were modified. The original failed smoke is retained at `results/shared_suite/20261025T220740Z_60g8j3eh/`.
- Full-sight, force-cooperation flags and native horizons are checked before the suite starts. Stock MAPPO and COMA algorithm files are unchanged. Shell syntax and Git whitespace checks pass.

The sandbox used for implementation cannot connect to the existing tmux Unix socket (`Operation not permitted`). The launcher therefore supports the exact host-side `--tmux` command without requiring an interactive attach; a launch is successful only when the command reports the new window and console log. Smoke results do not imply the full suite has started.
