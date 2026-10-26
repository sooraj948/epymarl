# Walkthrough 83: Recurrent QMIX for LBF with individual and summed MACA credit

## Major design choices

1. **Use EPyMARL QMIX as the base learner.** Retain its Adam optimizer, double Q-learning, monotonic mixing network, replay capacity, batch size, exploration schedule, reward standardization and target-update rule. Stock `qmix.yaml` remains available. The new `qmix_lbf` baseline and both shaped variants share a 64-unit GRU, agent identity and previous-action input. Recurrence and previous action are explicit changes from the upstream EPyMARL feedforward preset, including for fully observable LBF.
2. **Implement both requested injection routes.** `maca_qmix_aux` keeps the team TD target on environment rewards and directly regularizes the actual action-selecting agent utilities against individual shaped Monte Carlo returns. `maca_qmix_sum` uses a single shared terminal bonus equal to the sum of individual credits and has no auxiliary loss. There is no separate prediction head or second critic.
3. **Score factual trajectories with mean target-network team Q for recorded actions.** Reset recurrent history at episode start and unroll observations/previous actions in order. Mix the selected utilities through the target QMIX mixer; average only valid pre-action steps. Exclude the final observation and padding. Higher Q is better; this replaces negative mean GAE as the regression label. Scoring is detached and uses the existing target networks, with no extra target-update schedule.
4. **Reuse the existing credit mechanism.** Keep state-action occupancy signatures, food-goal context, trajectory-score regression, uniform-random donor bank, agentwise occupancy splicing, signed credit normalization, loss-only eligibility and dynamic beta. Counterfactuals are evaluated by the learned trajectory regressor; they are not fabricated recurrent inputs to the QMIX scorer and are not simulator rollouts.
5. **Compute credit once at collection.** Prepare each new complete training episode before replay insertion, store its applied individual credits as episode constants, and retain original environment rewards. Replay sampling never refits MACA, recomputes credit, advances episode counters or changes beta. Evaluation bypasses this preparation. Regression labels also remain fixed at collection, including across target-network versions.
6. **Start with terminal credit and finite-episode auxiliary returns.** Add each bonus once at the last actual transition. The auxiliary return has no bootstrap, including at the configured horizon. This avoids a separate one-step TD estimator for the trajectory-dependent reward. Inputs remain causal local histories; an RNN is not a guarantee that they identify the globally determined credit.
7. **Use one common environment-reward scale.** Retain the stock replay reward normalizer, updated once per learner batch from original rewards. Its padding convention is unchanged. Convert bonuses with the same current standard deviation, without subtracting the reward mean from a bonus. Do not fit moments to shaped rewards or independently normalize agents. Replayed bonuses remain fixed in raw units; their normalized values follow the current replay moments.
8. **Keep the auxiliary objective narrow and inspectable.** Only episodes with nonzero applied individual credit enter the auxiliary loss; otherwise there is no additional return-regression objective. Average over their valid agent-transitions, with `auxiliary_weight=1.0` by default. `maca.beta=0` bypasses the adapter and produces exactly the same tested parameter updates as stock QMIX with the same recurrent architecture.
9. **Persist learner and shaping state.** Save target agent/mixer weights, optimizer, reward moments, target-update counters, credit regressor, donor/episode buffers and isolated shaping RNG. Checkpoint configuration includes architecture, environment, optimizer settings and shaping mode; incompatible loads are rejected. Environment state, full policy replay and global sampling RNG are not saved, so this is not an exact simulator-resume facility.

## Prescribed parameters and deliberate differences

The source is the repository's `default.yaml` plus `qmix.yaml`, matching the
[upstream QMIX preset](https://github.com/uoe-agents/epymarl/blob/main/src/config/algs/qmix.yaml).
The target interval is **200 learner updates in the actual EPyMARL implementation**,
despite the older YAML comment describing episodes. No task-specific tuning is added.

| Setting | All three new variants |
| --- | --- |
| Optimizer / learning rate | Adam / 0.0005 |
| Discount / gradient clipping | 0.99 / norm 10 |
| Shared utility network | Linear-ReLU, GRUCell(64), action-value output |
| Utility inputs | Native observation, previous action one-hot, agent identity |
| Mixing network | Monotonic QMIX; embedding 32; two hypernetwork layers, width 64 |
| Replay / training batch | 5,000 complete episodes / 32 episodes |
| Collection | One environment, one new episode per loop; one update once replay is ready |
| Exploration | Epsilon 1.0 to 0.05 over 50,000 environment steps |
| Evaluation epsilon | 0.0, greedy action selection |
| Target updates | Hard copy every 200 learner updates |
| Rewards / returns | Environment reward standardization on / return standardization off |
| MACA defaults | Max beta 0.1, threshold -0.01, ramp 0.35, 16 donors, fit every 100 collected episodes, 20 fitting epochs |
| Auxiliary coefficient | 1.0, applied only in `individual_aux` mode |

The dynamic coefficient uses the established last-100-episode raw return average:

```text
effective_beta = beta * clip((recent_return_mean + 0.01) / 0.35, 0, 1)
stored_credit[i] = effective_beta * normalized_signed_credit[i]
```

MACA readiness still requires 50 total stored episodes and a fitting boundary.
With the default zero-based 100-episode fit frequency, the first fit occurs on
episode index 100, after collecting 101 episodes. Positive partial returns are
ineligible unless `shape_wins=True`. `warmup_episodes=0` does not bypass readiness.

## Objectives

Let `u[i,t]` be the actual agent utility for its recorded action, `c[i]` the
stored applied credit, `T` the episode length, and `mu`, `sigma` the current
shared environment-reward moments. The factual trajectory label is:

```text
score(episode) = mean_t Q_tot_target(state[t], history[t], recorded_joint_action[t])
```

Individual auxiliary mode uses:

```text
r_norm[t] = (r_environment[t] - mu) / sigma
r_individual[i,t] = r_norm[t] + 1[t == T-1] * c[i] / sigma
G[i,t] = sum_{k=t}^{T-1} gamma^(k-t) * r_individual[i,k]
L_aux = mean_over_active_valid_agent_transitions (u[i,t] - stop_gradient(G[i,t]))^2
L_total = L_QMIX(environment rewards) + auxiliary_weight * L_aux
```

The auxiliary gradients reach the same shared recurrent utility network used for
greedy actions. They do not directly reach the mixing network. The team TD loss
still trains both utilities and mixer. Averaging instead of summing auxiliary
losses prevents the coefficient from automatically increasing with agent count.
These are behavior-policy trajectory returns used as a regularizer, not an
independently justified Bellman target for an agent's QMIX utility.

Summed mode uses:

```text
r_team[t] = r_norm[t] + 1[t == T-1] * sum_i c[i] / sigma
L_total = L_QMIX(r_team)
```

Use a sum, not an agent average, for this requested experiment. Positive and
negative credits can cancel. Logs include both signed sum and absolute credit
sum; the per-episode ledger retains every individual credit. Summing retains
trajectory dependence, and terminal placement does not establish a Markov reward
function or policy-invariance guarantee.

## Scorer timing and experimental limits

The teacher is the lagged QMIX target pair, not an independent environment-only
critic. Both auxiliary utility training and summed reward shaping can influence
subsequent teacher snapshots. Factual scoring itself uses `no_grad` and cannot
backpropagate through credit generation.

Before the first target refresh, scores come from the initial target network.
With replay batch 32 and hard refresh every 200 learner updates, this persists
through roughly the first 231 collected episodes. The current implementation
does not add a teacher warmup rule or relabel the old regression records. Fits
therefore mix labels from different network versions and reward scales. These
are explicit initial experimental choices, not evidence of calibrated scores.

Individual targets can conflict with the utility ordering preferred by the team
loss. The monotonic mixer architecture remains in place, but fitting arbitrary
individual returns need not improve team behavior. Summed bonuses can instead
lose their individual discrimination through cancellation. Compare both with
the matched baseline using original environment returns. The smoke runs below
establish execution and credit application, not learning improvement.

## Files and launch commands

| File | Responsibility |
| --- | --- |
| `src/config/algs/qmix_lbf.yaml` | Recurrent matched baseline |
| `src/config/algs/maca_qmix_aux.yaml` | Individual utility regularization |
| `src/config/algs/maca_qmix_sum.yaml` | Summed shared bonus |
| `src/learners/qmix_lbf_learner.py` | Terminal rewards, Monte Carlo targets, auxiliary loss, checkpoints |
| `src/modules/qmix_maca_adapter.py` | Recurrent Q scoring and collection-time MACA |
| `src/learners/q_learner.py` | Default-preserving reward and auxiliary-loss hooks |
| `src/modules/maca_adapter.py` | Shared environment validation, donor construction and persistence |
| `src/maca/credit_pipelines.py` | Optional externally supplied factual score |
| `src/run.py` | Learner scheme extension and pre-insertion episode preparation |
| `scripts/run_qmix_suite.py`, `run_qmix_suite.sh` | Bounded comparison launcher |
| `tests/test_qmix_lbf.py` | Behavioral, numerical, replay and checkpoint checks |

From the repository root:

```bash
# Review commands without launching training.
./run_qmix_suite.sh --dry-run

# Short three-variant execution check on the small cooperative task.
./run_qmix_suite.sh --smoke --tasks 8x8_2p2f_coop --max-parallel 3

# Full selected-task comparison, in a new window of the existing tmux session.
./run_qmix_suite.sh --tasks 10x10_3p3f --tmux

# Direct individual-regularization run; all full budgets are opt-in launches.
../.TPCA/bin/python src/main.py --config=maca_qmix_aux --env-config=gymma with \
  env_args.key=lbforaging:Foraging-8x8-2p-2f-coop-v3 env_args.time_limit=50 \
  seed=1587 use_cuda=False t_max=2000000
```

The default suite is 36 jobs: three methods, four seeds (1587, 2398, 2875,
3370), and the three existing full-sight, horizon-50 tasks. The two 8x8 tasks
use 2M steps each; the 10x10 task uses 4M, totaling 96M budgeted steps.
Task groups run sequentially, with at most eight trainers at once. All four
seeds are included for the new QMIX baseline. Larger tasks retain standard LBF
without forced cooperation. `--methods`, `--tasks`, `--seeds`, `--beta` and
`--max-parallel` select a smaller comparison. The wrapper uses `../.TPCA/bin/python`
or `MACA_PYTHON`, without requiring shell activation.

Results go under `results/qmix_suite/<run>/`: command manifests, task summaries,
console logs, models, MACA manifests, donor banks and episode credit ledgers.
Sacred continues to store raw environment-return metrics under `results/sacred/`.

## Validation performed

- **66 tests pass** across the repository, including 17 new QMIX tests. They cover prescribed settings, matched architectures, exact baseline/beta-zero updates, taken-action recurrent scoring, padding exclusion, terminal placement, finite returns, active-only auxiliary gradients, summed-credit cancellation, common normalization, replay immutability, RNG isolation, target refreshes, incompatible checkpoint rejection and an identical next fixed-batch update after checkpoint restoration. Existing MAPPO checks pass.
- **Nine 6k-step jobs pass:** every method on every supported task, seed 1587. The small cooperative task applied nonzero credit to 21 episodes in each shaped run; the three-agent 8x8 task applied it to 17 auxiliary / 18 summed episodes; the 10x10 task applied it to 13 / 14 episodes. Each shaped run performed one real regressor fit. These short runs precede the first target refresh.
- **Two 16k-step jobs pass:** individual and summed modes on the small cooperative task. Each collected 321 episodes, fitted three times, applied nonzero credit to 218 episodes, and recorded both initial and update-200 target versions. Model saving was exercised. The unit tests verify the exact checkpoint continuation contract.
- Shell syntax and Git whitespace checks pass. Full-budget training was not launched.

Evidence directories are `results/qmix_suite/20261026T211909Z_s7_bc0v8`
(small task), `results/qmix_suite/20261026T212043Z_bp_1mb9e` (larger tasks),
and `results/qmix_suite/target_refresh_smoke_l2rz8c28` (longer target-refresh checks).
