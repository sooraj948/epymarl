#!/usr/bin/env bash
# Sequential LBF tasks, each with a bounded pool of shared MAPPO/MACA/COMA jobs.
set -euo pipefail
SUITE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUITE_PYTHON="${MACA_PYTHON:-$SUITE_ROOT/../.TPCA/bin/python}"
if [[ ! -x "$SUITE_PYTHON" ]]; then
    printf 'Python executable not found: %s\nSet MACA_PYTHON to Python 3.10 or newer.\n' "$SUITE_PYTHON" >&2
    exit 2
fi
exec "$SUITE_PYTHON" "$SUITE_ROOT/scripts/run_shared_suite.py" "$@"
