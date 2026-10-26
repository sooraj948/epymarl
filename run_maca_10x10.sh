#!/usr/bin/env bash
# Shared MAPPO + no-replay MACA: 10x10, three agents, three foods, all three betas.
set -euo pipefail
MACA_10X10_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$MACA_10X10_ROOT/run_shared_suite.sh" --tasks 10x10_3p3f --methods maca "$@"
