#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Phase 2 acceptance on the assigned tray (needs chips and TT_METAL_HOME):
#   crops      one core draws 32x32 tiles of every scene            (node 0)
#   grid       the whole 1920x1088 frame on all worker cores         (node 0)
#   strict     exact depth ties under LESS_EQUAL, no tolerance       (node 0)
#   multichip  1, 2, 4 and 8 chips, identical to the single-chip frame (nodes 0-7)
# Each step prints per-scene results and fails on the first mismatch.
set -euo pipefail
: "${TT_METAL_HOME:?set TT_METAL_HOME to a built tt-metal checkout}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export TT_RASTER_TIMEOUT="${TT_RASTER_TIMEOUT:-900}"
bin="${TT_RASTER_BUILD_DIR:-$root/build}/tools"
scripts/run.sh 0 "$bin/ttr_device" all --mode crops
scripts/run.sh 0 "$bin/ttr_device" all --mode grid
scripts/run.sh 0 "$bin/ttr_device" coplanar_lequal --mode grid --strict
scripts/run.sh 0,1,2,3,4,5,6,7 "$bin/ttr_multichip" all --chips 1,2,4,8 --bench 5
