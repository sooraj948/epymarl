#!/usr/bin/env bash
# Run the matched baseline, then online random-donor MACA, on the same four seeds.
set -euo pipefail

COMPARISON_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPARISON_PYTHON="${MACA_PYTHON:-$COMPARISON_ROOT/../.TPCA/bin/python}"
COMPARISON_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
COMPARISON_LOGS="$COMPARISON_ROOT/results/comparisons/$COMPARISON_ID"
SEEDS=(1587 2398 2875 3370)
COMMON_OVERRIDES=()
MACA_OVERRIDES=()
ACTIVE_PIDS=()
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: ./run_maca_comparison.sh [--dry-run] [key=value ...]

Runs baseline seeds 1587, 2398, 2875, 3370 in parallel, then the same MACA seeds.
At most four training processes run at once (ten environment workers each).
Defaults: CPU, 1,000,000 environment steps per run, MACA beta cap 0.1.

Examples:
  ./run_maca_comparison.sh
  ./run_maca_comparison.sh t_max=2000000 maca.beta=0.05
  ./run_maca_comparison.sh --dry-run

Ordinary Sacred overrides apply to both methods; maca.* overrides apply to MACA.
Seed, configuration, learner, label, evaluation and checkpoint selection are fixed
to keep this launcher a fresh paired comparison. Each invocation starts new runs.
MACA_PYTHON can select a different compatible Python executable.
EOF
}

for argument in "$@"; do
    case "$argument" in
        --help|-h) usage; exit 0 ;;
        --dry-run) DRY_RUN=true ;;
        maca.*=*) MACA_OVERRIDES+=("$argument") ;;
        seed=*|name=*|learner=*|label=*|evaluate=*|checkpoint_path=*|load_step=*|save_replay=*)
            printf 'This comparison fixes %s; use src/main.py directly for individual runs.\n' "${argument%%=*}" >&2
            exit 2 ;;
        *=*) COMMON_OVERRIDES+=("$argument") ;;
        *) usage >&2; printf 'Invalid argument: %s\n' "$argument" >&2; exit 2 ;;
    esac
done

if [[ ! -x "$COMPARISON_PYTHON" ]]; then
    printf 'Python executable not found: %s\nSet MACA_PYTHON to a compatible Python executable.\n' "$COMPARISON_PYTHON" >&2
    exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
    printf 'This launcher requires setsid to clean up training processes and environment workers.\n' >&2
    exit 2
fi

cleanup() {
    for pid in "${ACTIVE_PIDS[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    for pid in "${ACTIVE_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd -- "$COMPARISON_ROOT"
if ! "$DRY_RUN"; then
    mkdir -p -- "$COMPARISON_ROOT/results/comparisons"
    COMPARISON_LOGS="$(mktemp -d "$COMPARISON_ROOT/results/comparisons/$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
    COMPARISON_ID="${COMPARISON_LOGS##*/}"
fi
printf 'Comparison: %s\nConsole logs: %s\n' "$COMPARISON_ID" "$COMPARISON_LOGS"

for config in maca_baseline_ns maca_mappo_ns; do
    ACTIVE_PIDS=()
    printf 'Starting %s for seeds: %s\n' "$config" "${SEEDS[*]}"
    for seed in "${SEEDS[@]}"; do
        command=("$COMPARISON_PYTHON" -u "$COMPARISON_ROOT/src/main.py"
                 "--config=$config" --env-config=gymma with
                 "seed=$seed" "label=$COMPARISON_ID" "${COMMON_OVERRIDES[@]}")
        if [[ "$config" == maca_mappo_ns ]]; then
            command+=("${MACA_OVERRIDES[@]}")
        fi
        printf ' '
        printf ' %q' "${command[@]}"
        printf '\n'
        if ! "$DRY_RUN"; then
            OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
                setsid "${command[@]}" > "$COMPARISON_LOGS/${config}_seed_${seed}.log" 2>&1 &
            ACTIVE_PIDS+=("$!")
        fi
    done
    for index in "${!ACTIVE_PIDS[@]}"; do
        if wait "${ACTIVE_PIDS[$index]}"; then
            unset 'ACTIVE_PIDS[index]'
            printf 'Completed %s seed %s\n' "$config" "${SEEDS[$index]}"
        else
            printf 'Failed %s seed %s; see %s/%s_seed_%s.log\n' \
                "$config" "${SEEDS[$index]}" "$COMPARISON_LOGS" "$config" "${SEEDS[$index]}" >&2
            exit 1
        fi
    done
done

if "$DRY_RUN"; then
    printf 'Dry run complete; no training was launched.\n'
else
    printf 'Completed all eight runs. Console logs: %s\n' "$COMPARISON_LOGS"
    printf 'Metrics: results/sacred/  MACA credit: results/maca/  Models: results/models/\n'
fi
