#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run a tt-raster binary on the assigned tray with the environment tt-metal needs.
# tt-metal writes generated/ output into the current directory, so run from a scratch
# directory outside the repository (never from the repository root).
#
#   scripts/run.sh <node-list> <binary> [args...]
#   scripts/run.sh 0 build/spikes/a_lane_map/spike_a
#   scripts/run.sh 0,1,2,3 build/tools/probe_device mesh
#
# Chips are always selected by device node id through TT_VISIBLE_DEVICES; only nodes 0-7
# (the assigned tray) are accepted. A timeout kills a hung program.
set -euo pipefail

: "${TT_METAL_HOME:?set TT_METAL_HOME to a built tt-metal checkout}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

nodes="${1:?usage: run.sh <node-list> <binary> [args...]}"
bin="${2:?usage: run.sh <node-list> <binary> [args...]}"
shift 2

IFS=',' read -ra ids <<<"$nodes"
for n in "${ids[@]}"; do
    if ! [[ "$n" =~ ^[0-7]$ ]]; then
        echo "run.sh: node '$n' is outside the assigned tray (nodes 0-7)" >&2
        exit 2
    fi
done

[[ "$bin" = /* ]] || bin="$root/$bin"
scratch="${TT_RASTER_RUN_DIR:-${TMPDIR:-/tmp}/tt-raster-run}"
mkdir -p "$scratch"
cd "$scratch"

export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
export TT_VISIBLE_DEVICES="$nodes"
exec timeout -k 10 "${TT_RASTER_TIMEOUT:-300}" "$bin" "$@"
