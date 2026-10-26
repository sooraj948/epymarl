#!/usr/bin/env bash
set -euo pipefail
QMIX_SUITE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QMIX_SUITE_PYTHON="${MACA_PYTHON:-$QMIX_SUITE_ROOT/../.TPCA/bin/python}"
exec "$QMIX_SUITE_PYTHON" "$QMIX_SUITE_ROOT/scripts/run_qmix_suite.py" "$@"
