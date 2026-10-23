# Walkthrough 76 Four seed baseline and MACA launcher

## Major design choices

1. **Use the previous four seeds for both conditions.** `run_maca_comparison.sh` runs `maca_baseline_ns` and `maca_mappo_ns` with seeds `1587`, `2398`, `2875` and `3370`.
2. **Run the baseline stage before MACA.** Four seeds run in parallel within each stage. The second stage starts after all baseline runs succeed. With the default ten environments per training process, this creates up to 40 environment workers plus four training processes. Numerical library threads are limited to one per training process.
3. **Keep shared settings matched.** Ordinary Sacred overrides apply to both methods; `maca.*` overrides apply only to MACA. The existing defaults remain CPU, one million environment steps per run, the native 50-step LBF horizon and MACA beta cap 0.1. Fixed seeds and method selection cannot be overridden through the launcher.
4. **Keep each invocation separate.** Unique comparison directories preserve eight console logs. A shared comparison ID is recorded as the Sacred `label`, connecting the eight runs. Every invocation starts fresh; there is no automatic reuse of previous experiments.
5. **Clean up workers on failure or cancellation.** Each run has its own process group. A failing job stops the launcher and terminates remaining jobs and their environment workers; interruption also cleans up active process groups. Training code and optimization are unchanged.

## Run

```bash
cd /scratch/cluster/sooraj/epymarl-maca
./run_maca_comparison.sh
```

This schedules eight runs, each with its own one-million-step budget. It does not split one million steps between seeds. EPyMARL's existing whole-batch stopping behavior can exceed the requested budget slightly.

Preview the commands without starting training or creating log directories:

```bash
./run_maca_comparison.sh --dry-run
```

Change the budget for both methods, and beta for MACA only:

```bash
./run_maca_comparison.sh t_max=2000000 maca.beta=0.05
```

The script resolves its own repository directory and defaults to `../.TPCA/bin/python`. Set `MACA_PYTHON` to an absolute executable path to select another compatible runtime. Additional Sacred overrides use the same `key=value` syntax as `src/main.py`. Use the direct entrypoint for evaluation, checkpoint continuation or individually selected seeds.

## Outputs

Console logs are saved under `results/comparisons/<unique comparison ID>/` as `maca_baseline_ns_seed_<seed>.log` and `maca_mappo_ns_seed_<seed>.log`. The directory is printed at startup. Normal metrics, credit ledgers and checkpoints remain under `results/sacred/`, `results/maca/` and `results/models/`, as described in walkthrough 75.

## Validation

`bash -n` and a dry run passed. Temporary executable stubs verified all eight method/seed combinations, the four-process concurrency limit, baseline completion before the MACA stage, shared override propagation, MACA-only override routing, one-thread environment settings and invocation from another directory. Additional checks verified seed-override rejection, baseline failure preventing the MACA stage, and cancellation terminating both the training stubs and simulated environment workers.

No full experiment was started during launcher validation. The policy still uses n-step PPO while the mixer teacher uses GAE; the open experimental issue documented in walkthrough 75 remains unchanged.
