# Walkthrough 73 EPyMARL integration plan

## Major design choices

1. **Maintain one EPyMARL fork containing the MACA source.** Copy this repository's `maca/` directory into the fork at `src/maca/`. All new code, tests, documentation and experiments live in that fork. EPyMARL owns environment interaction, actors, critics, batching, optimization and evaluation. MACA owns trajectory encoding, donor selection, episode scoring and credit. Preserve the original repository as a historical reference; it is not a runtime dependency or a second maintained implementation.
2. **Integrate online random-donor splice first.** This is the user-selected initial variant: position-action occupancy, initial/final food context, a regressor taught by negative mean GAE from the current critics, random-donor counterfactual signatures and uniform reward injection. Native action replay, reactive policy counterfactuals and frozen critics are later extensions.
3. **Use `mappo_ns`.** The user selected separate feedforward actors and centralized critics, the closest existing EPyMARL architecture to the current MACA experiments. Train fresh matched baseline runs; the existing shared recurrent `mappo` results are a different condition.
4. **Keep EPyMARL n-step PPO initially. This is a major open issue.** The user explicitly requested retaining n-step PPO and documenting its possible importance. The mixer teacher still uses negative mean GAE. These two advantage estimators serve different purposes, but changing the policy estimator from historical MACA GAE may materially change results. See the dedicated discussion below.
5. **Use CPU and original team reward units for the first comparison.** Set `use_cuda=False`, `common_reward=True`, `reward_scalarisation="sum"`, `standardise_rewards=False` and `standardise_returns=False` in both new comparison configurations. These are proposed implementation defaults, not requirements of MACA. Keep other PPO settings matched. Do not infer equivalence to stock reward-standardized EPyMARL or historical MACA runs.
6. **Compute credit once before each PPO update.** Process each fresh rollout episode once, hold teacher critic parameters fixed while processing that batch and reuse the resulting shaped rewards across PPO epochs. Preserve original episode rewards for logging, evaluation and mixer teacher rewards.

This is an implementation plan. The training integration has not been implemented. Repository inspection used MACA commit `893f188` and EPyMARL commit `cbc38c0`. The single-repository approach supersedes the earlier proposal for two maintained repositories and an editable MACA dependency.

## Git and GitHub setup

Use the GitHub Fork button on `uoe-agents/epymarl` to create `sooraj948/epymarl`, unless that fork already exists. Clone the personal fork into a fresh directory so the existing baseline checkout and its results remain intact. The following commands assume the destination does not already exist.

```bash
cd /scratch/cluster/sooraj
git clone git@github.com:sooraj948/epymarl.git epymarl-maca
cd epymarl-maca
git rev-parse HEAD
cp -R ../Unsupervised-Multi-agent-Credit-Assignment/maca src/maca
mkdir -p docs
cp ../Unsupervised-Multi-agent-Credit-Assignment/docs/walkthrough-73.md docs/walkthrough-73.md
printf '\n/results/\n' >> .gitignore
git add src/maca docs/walkthrough-73.md .gitignore
git diff --cached --stat
git commit -m "Import MACA source and integration plan"
git push origin main
```

These commands import the source; they do not complete the training integration. Record the original MACA commit `893f188` and the initial fork commit in the imported documentation. If the fork starts from a newer EPyMARL version than `cbc38c0`, check the relevant hooks again and train the comparison baseline on that version. Freeze the selected version during the first study.

The fork is the only repository receiving future implementation changes. Copy relevant credit-pipeline tests and algorithm documentation into it as needed. Keep old trainers, plots and result artifacts in the original repository as historical material. A second remote, packaging project, submodule, second active branch in MACA or coordinated pair of pull requests is unnecessary. Adding an upstream remote later is optional if updates from EPyMARL become useful.

Stage explicit source paths, review diffs, and commit integration changes in this fork. Record its commit and dirty status in each experiment manifest. The imported source is versioned by the same commit as its adapter and learner.

## Runtime and dependency preparation

The repository instructions select `../.TPCA` for experiments. That path is absent in the inspected workspace; `../venv` exists. Restore or identify the required compatible runtime before running checks. Do not silently switch interpreters or alter the environment used by ongoing experiments.

With MACA at `src/maca/`, `src/main.py` can import it alongside EPyMARL's other modules. No editable installation, separate `pyproject.toml` or `PYTHONPATH` change is needed for that entrypoint. Configure focused tests to use the same source root.

Avoid importing the old `train.py`, `train_credit_pipelines.py` or `auxillary` experiment scripts. Move the small selected scheduling/helper functions into the copied MACA modules. Verify compatible PyTorch, NumPy, SciPy, scikit-learn, Gymnasium and LBF versions and add necessary dependencies to the fork's requirements. Do not upgrade all packages opportunistically.

## File changes

The minimum training extension remains a learner, an adapter, a registry entry and a configuration. A maintainable first version also needs the copied MACA source, a matched baseline configuration and focused checks. Every path below is inside the fork.

| File | Planned change |
| --- | --- |
| `src/maca/` | Copy the existing MACA library and retain its module boundaries. |
| `src/maca/credit_pipelines.py` | Accept an optional injected scorer, expose the selected reward-weight helper copied from the old trainer and remove the random-bank loader's dependency on `auxillary.reset_oracle._is_done`. |
| `src/modules/maca_adapter.py` | Build the selected MACA components; convert batches to trajectories and states; implement teacher scoring with the EPyMARL critic; apply credit and maintain MACA state. |
| `src/learners/maca_ppo_learner.py` | Subclass `PPOLearner`; prepare shaped rewards once in `train`; substitute them in `train_critic_sequential`; save/load the adapter state. |
| `src/learners/__init__.py` | Register the new learner. Keep MACA-specific dependency imports inside its construction path where practical so unrelated baseline learners remain usable. |
| `src/config/algs/maca_baseline_ns.yaml` | Add the fresh, explicitly configured baseline derived from `mappo_ns.yaml`. |
| `src/config/algs/maca_mappo_ns.yaml` | Match that baseline's PPO settings and select the new learner plus the selected MACA variant. EPyMARL does not automatically inherit one algorithm YAML from another. |
| Requirements and focused tests | Declare necessary dependencies; verify scorer injection, copied helper behavior, trajectory conversion, reward placement, RNG isolation and one complete PPO update. |
| Experiment instructions and this walkthrough | Record commands, source provenance, resolved settings, checks and the outstanding GAE question. Update implementation documentation last. |

Keep MACA algorithm components such as `encoders.py`, `reducers.py`, `buffers.py` and `shaper.py` unchanged unless a concrete compatibility failure requires a fix. The first version should not need changes to EPyMARL's main entrypoint, run loop, runners, controllers, environment wrappers, episode buffer or base PPO learner.

## Entrypoints and ownership

```text
src/main.py
  load the selected algorithm and environment configuration
  -> src/run.py run_sequential
     construct the registered learner
     collect a fresh rollout batch with original rewards
     -> MACAPPOLearner.train
        -> adapter.prepare
           reconstruct trajectories and score episodes
           cache shaped rewards
        -> PPOLearner.train
           -> MACAPPOLearner.train_critic_sequential
              pass cached rewards to the existing critic update
           existing PPO actor update
```

The existing `PPOLearner.train` calls `self.train_critic_sequential` inside each PPO epoch. Override that method to supply the cached `[batch, time, agents]` tensor. Both the critic targets and policy advantages then use shaped rewards through the existing implementation. Do not update only actor advantages while leaving the critic on a different objective.

When beta is zero, bypass MACA initialization and processing as far as practical and delegate to baseline behavior. This provides a clean equivalence check without MACA consuming the policy's random numbers.

## Batch conversion and scoring

1. Require the supported full-sight flat-observation LBF task and validate observation ordering, agent/action counts, horizon and food slots. Fail clearly for unsupported partial observations or wrappers that change the representation.
2. Recover each episode's true transition count and final observation from its own `filled` entries, checked against termination and padding. The final stored observation is not another transition. Never use the longest episode's length for every batch member.
3. Construct each agent's pre-action `(observation, requested_action)` sequence. Reconstruct global food/player coordinates and levels from full-sight observations, preserving observer-first player order. Include only state fields needed by this variant and the encoder; do not fabricate hidden simulator history.
4. Obtain teacher values from EPyMARL's current centralized critics under `torch.no_grad`, with parameters fixed until all episodes in the batch are scored. Average across agent critics before computing the existing negative mean GAE target. Preserve the selected finite-horizon teacher convention and test its boundary separately from the policy's bootstrap convention.
5. Inject a scorer exposing the existing pipeline's required value and score operations. Do not try to load EPyMARL weights into MACA's `ValueNetwork` or pretend the two checkpoint formats are compatible.
6. Process each episode once in a defined deterministic batch order. Maintain MACA's own episode counter and running return window. Warmup, fit frequency and beta scheduling count training episodes, not timesteps, PPO epochs or evaluation episodes.
7. Keep `buffer_size = batch_size = batch_size_run` for the first parallel-runner configuration so each update consumes a fresh complete rollout batch. EPyMARL can permute sampled episodes; the adapter must not rely on completion order or rescore old episodes. Strict reproduction of the historical sequential episode order is outside this initial comparison.
8. Use the existing random-donor bank generation, occupancy encoder, food context, trajectory regressor, normalization and dynamic beta schedule. Extract the selected settings from a concrete recent run manifest rather than silently using the older `MACAConfig` defaults. Expose only needed MACA settings; derive environment dimensions and reuse existing constants.

The injected reward is:

```text
r_shaped[b,t,i] = r_team[b,t] + beta[b] * credit[b,i] / episode_length[b]
```

Use the existing spreading helper for uniform injection, preserve positive/negative credit, and leave padding at zero. Keep raw rewards in the collected batch and logs. The teacher uses the factual unshaped team reward sequence, while the live online critics evolve under shaped training, matching the selected online variant's design.

MACA must use independent RNG state, including initialization, mixer fitting, donor selection and cache creation. Preserve policy RNG state around MACA work and advance MACA's own RNG state across calls. Evaluation must neither fit the mixer nor update donors, counters or running statistics.

## State and experiment records

Save the mixer weights, stored episode records and targets, donor identity, normalization statistics, episode counters, return window, fit version, score-batch statistics and MACA RNG state alongside EPyMARL's model files. Validate the task/configuration on load. Preserve the existing regressor's optimizer lifecycle rather than changing fitting behavior during integration.

Saving MACA state supports continuing its learning history. Exact continuation of environment trajectories additionally requires simulator, runner and training-buffer state, which the existing minimal EPyMARL model checkpoint does not provide. Do not claim bitwise restart equivalence from model checkpoints alone.

Log original team returns versus environment steps as the primary comparison. Also record training episodes, effective beta, per-agent credit, skipped-credit reasons, fit version, and rollout/MACA/learner timings. Record the resolved configuration, fork commit and dirty status, package versions and device. Use the same environment-step budget, evaluation behavior and seeds across conditions; MACA's episode-based schedules remain explicitly episode-based.

## Major open issue N step PPO versus GAE

**This may be a major experimental issue and must be revisited before drawing broad conclusions about MACA.**

Historical MACA policy training computes GAE on shaped rewards, then standardizes policy advantages across each agent's transition batch. EPyMARL's current PPO learner computes n-step bootstrapped targets and uses the resulting critic residuals as policy advantages; it also updates the critic and recomputes advantages during PPO epochs. The selected first integration retains those EPyMARL behaviors.

The mixer teacher remains negative mean unstandardized GAE on factual environment rewards. Its GAE target describes an episode score; it does not mean the policy update also uses GAE. Logs and plots must state both estimators independently.

Changing the policy estimator can alter credit propagation, variance, and the effect of uniform reward injection. Changes in recurrent/parameter-sharing choices, reward standardization, advantage standardization, bootstrap conventions and update timing also prevent direct attribution of differences between historical MACA results and the new EPyMARL runs solely to MACA quality.

The first comparison answers: **Does the selected MACA credit improve a matched EPyMARL n-step PPO baseline?** It does not establish equivalence to the original MACA trainer or that MACA works equally well under GAE.

If the n-step results are weak or differ materially from historical runs, discuss a second controlled study with four conditions: n-step baseline, n-step plus MACA, GAE baseline and GAE plus MACA. Keep all other settings matched. Implement GAE as a shared learner capability for both baseline and MACA in that later study. Agree on frozen-versus-recomputed advantages, standardization and terminal/time-limit bootstrapping explicitly; swapping a return formula alone does not reproduce historical PPO semantics.

## Validation and implementation order

1. Create the fork, copy the MACA source and plan, and prepare the compatible runtime. Resolve the exact recent MACA settings and freeze the new baseline configuration.
2. Make the changes inside `src/maca/`. Copy and run relevant existing credit-pipeline and reward-spreading tests. Verify `src/main.py` can import MACA without the original repository or auxiliary experiment scripts on its import path.
3. Implement and test full-sight observation/state reconstruction, including round trips to native observations, different episode lengths, missing food slots, final states and padding.
4. Add the scorer bridge and compare its predicted values and negative mean GAE with direct calculations using the same EPyMARL critic. Verify no critic gradients or parameter changes during scoring.
5. Add the learner and registry entry. Check beta-zero agreement with the matched baseline using the same initialization and seed. Verify scoring occurs once per episode, not once per PPO epoch, and verify signed per-agent credit placement plus a finite nonzero-credit update.
6. Verify evaluation does not mutate MACA state and run a save/load round trip for MACA learning state. Distinguish this from exact simulator continuation.
7. Run short matched baseline and MACA jobs, long enough for mixer readiness and nonzero credit. Check original reward logging and compare fixed-budget elapsed times before launching a seed study.
8. Review diffs, commit changes in the single fork, and update its documentation last with the resolved settings, commands, validation and remaining GAE issue.

After implementation, the planned entrypoints are:

```bash
cd /scratch/cluster/sooraj/epymarl-maca
../.TPCA/bin/python src/main.py --config=maca_baseline_ns --env-config=gymma with env_args.key=lbforaging:Foraging-8x8-2p-2f-coop-v3 env_args.time_limit=200 seed=1587
../.TPCA/bin/python src/main.py --config=maca_mappo_ns --env-config=gymma with env_args.key=lbforaging:Foraging-8x8-2p-2f-coop-v3 env_args.time_limit=200 seed=1587
```

The new algorithm configurations must explicitly set the same chosen `t_max`, batch sizes, CPU device, reward units and PPO parameters. These commands are planned interfaces; the configs and runtime must exist before they can run.
