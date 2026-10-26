# Walkthrough 81 MACA-only 10x10 experiment

## Major design choices

1. **Run only the requested additional MACA experiment.** New `run_maca_10x10.sh` selects `10x10_3p3f` and `maca` explicitly through the existing shared launcher. Default `run_shared_suite.sh` behavior is unchanged: its earlier matrix still limits MACA to the small task.
2. **Match the completed 10x10 baseline environment and budget.** Use `lbforaging:Foraging-10x10-3p-3f-v3`, full sight, standard LBF without forced cooperation, a 50-step horizon and 4M training environment steps per run.
3. **Keep the existing MACA and shared MAPPO implementations.** Use `maca_mappo.yaml`, online random-donor no-replay splice, shared recurrent actors, shared agent-conditioned centralized critic, low entropy, original seed helper, and the established shaping/standardization order. Beta 0.08, 0.1 and 0.15 retain the existing dynamic schedule.
4. **Use all four seeds and at most eight concurrent trainers.** Seeds 1587, 2398, 2875 and 3370 give 12 runs and 48M budgeted steps. The previous exclusion of seed 1587 applied to comparison images, not this new experiment. Eight trainers start first; remaining jobs fill slots as earlier jobs finish.
5. **Reuse tmux launch and process cleanup.** `--tmux` opens a detached `maca-10x10` window in the existing `epymarl-maca` session. Preserve task/method selectors in the detached command so it cannot accidentally launch the default 35-run suite. Write a unique output directory, commands, status, per-run logs, resource summary and persistent console log.

## Launch

```bash
cd /scratch/cluster/sooraj/epymarl-maca
./run_maca_10x10.sh --dry-run
./run_maca_10x10.sh --tmux
```

The wrapper selects `../.TPCA/bin/python` automatically. No virtualenv activation is required. To run directly in the current terminal or an already attached tmux pane, omit `--tmux`. Optional `--seeds 2398 2875 3370` restricts the seed list, and `--max-parallel 4` reduces concurrency.

The existing session must already exist for `--tmux`. Attach with `tmux attach -t epymarl-maca` when needed. Full training is launched by the command above; the implementation validation below ran only short smoke jobs.

The generic equivalent is:

```bash
./run_shared_suite.sh --tasks 10x10_3p3f --methods maca --tmux
```

Explicit `--methods maca` enables all three beta variants on selected tasks. With no selectors, the original default task/method matrix is preserved. A resumed targeted experiment must use the same selectors; invoking `run_maca_10x10.sh --resume <suite-directory> --tmux` preserves them automatically.

## Output and fixed settings

Each invocation creates `results/shared_suite/<unique-id>/10x10_3p3f/`. Logs are named, for example, `maca_beta_0p1_seed_2398.log`. Models and MACA artifacts are in the task's `models/` and `maca/` directories; Sacred results are under `results/sacred/maca_mappo/lbforaging:Foraging-10x10-3p-3f-v3/<run-id>/`, selected by the new suite label.

Training stays on CPU. Each trainer uses ten fresh episodes per update. Evaluation samples 100 episodes every 50k training steps, diagnostic logging occurs every 10k steps, and checkpoints are saved every 100k steps. No additional hyperparameters or algorithm changes are introduced.

## Validation

- All nine shared-launcher tests pass, including exact selection of 12 MACA-only 4M-step jobs and forwarding of task/method selectors into the tmux command.
- Shell syntax and dry-run checks pass. The dry run contains only the requested 10x10 standard task and all three beta/four-seed combinations.
- `./run_maca_10x10.sh --smoke` completed all three beta runs at seed 1587 and 6,000 steps. Each run stored 130 episodes, fitted its mixer once and applied nonzero shaping to 14 episodes. Smoke output: `results/shared_suite/20261026T142505Z_od3f8oap/`.

This validates the launcher and actual credit application; it does not establish full-budget learning performance.
