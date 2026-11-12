#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Python-side checks: the core frame API (core/tests/test_frame_api.py) and, for the Isaac Lab plugin, the Newton bridge and the
# Isaac Lab renderer (against Isaac Lab stand-ins, with real Newton data). Needs a Python with Newton and Warp (default: ~/experiments/venv-newton) and
# the built libttraster.so; Newton runs on the CPU.
#   scripts/test_python.sh                       CPU backend of tt-raster
#   TT_METAL_HOME=... TT_RASTER_NODE=3 scripts/test_python.sh    also the Tensix backend on that node
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="${TT_RASTER_PYTHON:-$HOME/experiments/venv-newton/bin/python}"
testpath() {  # where a test lives
    if [[ "$1" == test_frame_api ]]; then echo "$root/core/tests/$1.py"; else echo "$root/plugins/isaaclab/validation/$1.py"; fi
}

for t in test_newton_bridge test_isaaclab_shim test_pipeline test_frame_api; do
    echo "== $t (reference backend)"
    "$py" "$(testpath "$t")" --backend reference 2>&1 | grep -E "^(ok|FAIL|skip|PASS)|failed|Error|Traceback" 
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "$t failed"; exit 1; }
done

echo "== test_gaussian (Gaussian splatting plugin, reference backend)"
TTRASTER_GS_LIB="${TT_RASTER_BUILD_DIR:-$root/build}/plugins/gaussian/libttraster_gaussian.so" "$py" "$root/plugins/gaussian/tests/test_gaussian.py" --backend reference 2>&1 | grep -E "^(ok|FAIL|PASS)|Error|Traceback"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "test_gaussian failed"; exit 1; }

echo "== test_server (device server, CPU reference backend)"
"$py" "$root/core/tests/test_server.py" 2>&1 | grep -E "^(ok|FAIL|PASS)|Error|Traceback"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "test_server failed"; exit 1; }

if [[ -n "${TT_METAL_HOME:-}" ]]; then
    node="${TT_RASTER_NODE:-3}"
    for t in test_newton_bridge test_isaaclab_shim test_pipeline test_frame_api; do
        echo "== $t (tensix backend, node $node)"
        "$root/scripts/run.sh" "$node" "$py" "$(testpath "$t")" --backend tensix 2>&1 | grep -E "^(ok|FAIL|skip|PASS)|failed|Error|Traceback"
        [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "$t failed"; exit 1; }
    done
    echo "== test_gaussian (tensix backend, node $node)"
    TTRASTER_GS_LIB="${TT_RASTER_BUILD_DIR:-$root/build}/plugins/gaussian/libttraster_gaussian.so" "$root/scripts/run.sh" "$node" "$py" "$root/plugins/gaussian/tests/test_gaussian.py" --backend tensix 2>&1 | grep -E "^(ok|FAIL|PASS)|Error|Traceback"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "test_gaussian (tensix) failed"; exit 1; }
fi
echo "python tests OK"
