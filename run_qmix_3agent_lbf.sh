#!/usr/bin/env bash
# QMIX baseline and both MACA modes on three-agent LBF, using beta 0.1 only.
set -euo pipefail
QMIX_3AGENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$QMIX_3AGENT_ROOT/run_qmix_suite.sh" "$@" \
  --tasks 8x8_3p2f 10x10_3p3f \
  --methods qmix individual_aux team_sum \
  --beta 0.1
