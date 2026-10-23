#!/usr/bin/env bash
# Shared job pool for MAPPO baseline, MACA and stock COMA comparisons.
set -euo pipefail
COMPARISON_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPARISON_PYTHON="${MACA_PYTHON:-$COMPARISON_ROOT/../.TPCA/bin/python}"
if [[ ! -x "$COMPARISON_PYTHON" ]]; then
    printf 'Python executable not found: %s\nSet MACA_PYTHON to a compatible executable.\n' "$COMPARISON_PYTHON" >&2
    exit 2
fi
exec "$COMPARISON_PYTHON" "$COMPARISON_ROOT/scripts/run_comparison.py" "$@"
