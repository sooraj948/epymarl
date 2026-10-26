#!/usr/bin/env bash
# MACA 10x10 beta sweep with one shared normalizer fitted after reward shaping.
set -euo pipefail
SHAPED_NORM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SHAPED_NORM_ROOT/run_maca_10x10.sh" --betas 0.08 0.1 --maca-reward-standardization shaped "$@"
