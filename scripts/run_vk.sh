#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run a Vulkan program on lavapipe (Mesa's conformant CPU Vulkan driver) from the `scripts/build_mesa.sh --llvm` build.
#
#   scripts/run_vk.sh [--threads N] [--size WxH] [--backend off|reference|tensix] [--node N] <program> [args...]
#
#   scripts/run_vk.sh vulkaninfo --summary
#   scripts/run_vk.sh --threads 4 vkcube --c 300         # in a virtual X server (X11 swapchain)
#   scripts/run_vk.sh --no-x <program>                   # without a virtual X server (headless programs)
#
# By default Vulkan runs entirely on the CPU. --backend reference|tensix turns on the Tensix offload of lavapipe's llvmpipe (api/vulkan/README.md):
# whole frames of untextured triangles with simple shaders go to libttraster (reference: its CPU backend; tensix: the chip on --node,
# default 5, TT_METAL_HOME must be set), everything else stays on the CPU path.
# LP_NUM_THREADS (--threads) sets the number of rasterizer threads of llvmpipe/lavapipe (default: all cores).
set -euo pipefail

prefix="${TT_RASTER_MESA_LLVM_PREFIX:-$HOME/experiments/mesa-install-llvm}"
size="1920x1080"
use_x=1
backend="off"
node="${TT_RASTER_NODE:-5}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads) export LP_NUM_THREADS="$2"; shift 2 ;;
        --size) size="$2"; shift 2 ;;
        --no-x) use_x=0; shift ;;
        --backend) backend="$2"; shift 2 ;;
        --node) node="$2"; shift 2 ;;
        *) break ;;
    esac
done
[[ $# -gt 0 ]] || { echo "run_vk.sh: no program given" >&2; exit 2; }
[[ -f "$prefix/share/vulkan/icd.d/lvp_icd.x86_64.json" ]] || { echo "run_vk.sh: no lavapipe in $prefix (scripts/build_mesa.sh --llvm)" >&2; exit 2; }

export LD_LIBRARY_PATH="$prefix/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES="$prefix/share/vulkan/icd.d/lvp_icd.x86_64.json"
prog="$(command -v "$1")"
shift
launch=()
if [[ "$backend" != "off" ]]; then
    export TTR_LP="$backend"
    export TTRASTER_LIB="${TTRASTER_LIB:-${TT_RASTER_BUILD_DIR:-$root/build}/core/libttraster.so}"
    [[ "$backend" == "tensix" ]] && launch=("$root/scripts/run.sh" "$node")
fi
if [[ $use_x == 1 ]]; then
    exec "${launch[@]}" "$(command -v xvfb-run)" -a -s "-screen 0 ${size}x24" "$prog" "$@"
fi
exec "${launch[@]}" "$prog" "$@"
