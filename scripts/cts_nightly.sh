#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Nightly Vulkan CTS run for cron: the drawing groups (scripts/cts_groups.txt) on lavapipe alone, then through the Tensix offload on every
# chip of the assigned tray that no other process holds. Results are kept per night; the exit status is 1 if either run has new failures
# (cases that fail and are not in scripts/cts_expected_failures.txt) or cases that never ran.
#
#   scripts/cts_nightly.sh [--no-chip] [--keep N]
#   0 2 * * *  TT_METAL_HOME=$HOME/experiments/tt-metal /path/to/tt-raster/scripts/cts_nightly.sh >> $HOME/.cache/tt-raster/cts/cron.log 2>&1
#
# Results: $TT_RASTER_CTS_RESULTS (default ~/.cache/tt-raster/cts)/<UTC date>/{lavapipe,offload}/ with results.json and the CTS logs, and
# summary.txt. Needs the CTS and the Mesa build of scripts/run_cts.sh and a host build (scripts/build.sh). Only nodes 0-7 are ever used.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_root="${TT_RASTER_CTS_RESULTS:-$HOME/.cache/tt-raster/cts}"
keep=14
use_chip=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-chip) use_chip=0; shift ;;
        --keep) keep="$2"; shift 2 ;;
        *) echo "usage: cts_nightly.sh [--no-chip] [--keep N]" >&2; exit 2 ;;
    esac
done
mkdir -p "$results_root"
exec 9>"$results_root/.lock"
flock -n 9 || { echo "cts_nightly: another run is in progress"; exit 0; }

day="$(date -u +%Y%m%d)"
out="$results_root/$day"
mkdir -p "$out"
summary="$out/summary.txt"
: > "$summary"
rc=0

echo "== lavapipe alone ($day)" | tee -a "$summary"
python3 "$root/scripts/cts_farm.py" --backend off --workers 8 --out "$out/lavapipe" 2>&1 | tail -n 25 | tee -a "$summary"
[[ ${PIPESTATUS[0]} -eq 0 ]] || rc=1

if [[ $use_chip -eq 1 ]]; then
    # a chip another process has open (another session on the tray) is left alone
    nodes=()
    for n in 0 1 2 3 4 5 6 7; do
        if [[ -e /dev/tenstorrent/$n ]] && ! lsof "/dev/tenstorrent/$n" >/dev/null 2>&1; then nodes+=("$n"); fi
    done
    if [[ ${#nodes[@]} -eq 0 ]]; then
        echo "== offload: no free chip in nodes 0-7, skipped" | tee -a "$summary"
    else
        node_list="$(IFS=,; echo "${nodes[*]}")"
        echo "== offload on chips $node_list" | tee -a "$summary"
        python3 "$root/scripts/cts_farm.py" --backend tensix --nodes "$node_list" --baseline "$out/lavapipe/results.json" --out "$out/offload" 2>&1 | tail -n 25 | tee -a "$summary"
        [[ ${PIPESTATUS[0]} -eq 0 ]] || rc=1
    fi
fi

# keep the last N nights
ls -1d "$results_root"/[0-9]*/ 2>/dev/null | sort | head -n "-$keep" | while read -r old; do rm -rf -- "$old"; done
echo "== exit status $rc" | tee -a "$summary"
exit $rc
