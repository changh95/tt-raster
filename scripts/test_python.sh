#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Python-side checks against real Newton data: the Newton bridge and the Isaac Lab renderer (against
# Isaac Lab stand-ins). Needs a Python with Newton and Warp (default: ~/experiments/venv-newton) and
# the built libttraster.so; Newton runs on the CPU.
#   scripts/test_python.sh                       CPU backend of tt-raster
#   TT_METAL_HOME=... TT_RASTER_NODE=3 scripts/test_python.sh    also the Tensix backend on that node
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="${TT_RASTER_PYTHON:-$HOME/experiments/venv-newton/bin/python}"

for t in test_newton_bridge test_isaaclab_shim test_pipeline test_frame_api; do
    echo "== $t (reference backend)"
    "$py" "$root/validation/$t.py" --backend reference 2>&1 | grep -E "^(ok|FAIL|skip|PASS)|failed|Error|Traceback" 
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "$t failed"; exit 1; }
done

if [[ -n "${TT_METAL_HOME:-}" ]]; then
    node="${TT_RASTER_NODE:-3}"
    for t in test_newton_bridge test_isaaclab_shim test_pipeline test_frame_api; do
        echo "== $t (tensix backend, node $node)"
        "$root/scripts/run.sh" "$node" "$py" "$root/validation/$t.py" --backend tensix 2>&1 | grep -E "^(ok|FAIL|skip|PASS)|failed|Error|Traceback"
        [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "$t failed"; exit 1; }
    done
fi
echo "python tests OK"
