#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Phase 2 acceptance on the assigned tray (needs chips and TT_METAL_HOME):
#   crops      one core draws 32x32 tiles of every scene            (node 0)
#   grid       the whole 1920x1088 frame on all worker cores, also against the golden PNGs (node 0)
#   strict     exact depth ties under LESS_EQUAL, no tolerance       (node 0)
#   shader     every fragment shader instruction against its definition, in units of the last place (node 0)
#   multichip  1, 2, 4 and 8 chips, identical to the single-chip frame (nodes 0-7)
# Each step prints per-scene results and fails on the first mismatch. TT_RASTER_NODE, TT_RASTER_MC_NODES and
# TT_RASTER_MC_CHIPS select other chips (a chip subset must be a connected part of the tray mesh).
set -euo pipefail
: "${TT_METAL_HOME:?set TT_METAL_HOME to a built tt-metal checkout}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export TT_RASTER_TIMEOUT="${TT_RASTER_TIMEOUT:-900}"
bin="${TT_RASTER_BUILD_DIR:-$root/build}/tools"
node="${TT_RASTER_NODE:-0}"
mc_nodes="${TT_RASTER_MC_NODES:-0,1,2,3,4,5,6,7}"
mc_chips="${TT_RASTER_MC_CHIPS:-1,2,4,8}"
scripts/run.sh "$node" "$bin/ttr_device" all --mode crops
scripts/run.sh "$node" "$bin/ttr_device" all --mode grid --golden
scripts/run.sh "$node" "$bin/ttr_device" coplanar_lequal --mode grid --strict
scripts/run.sh "$node" "$bin/ttr_shader_ops" all
scripts/run.sh "$mc_nodes" "$bin/ttr_multichip" all --chips "$mc_chips" --bench 5
