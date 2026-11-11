#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run a Vulkan program on lavapipe (Mesa's conformant CPU Vulkan driver) from the `scripts/build_mesa.sh --llvm` build.
#
#   scripts/run_vk.sh [--threads N] [--size WxH] <program> [args...]
#
#   scripts/run_vk.sh vulkaninfo --summary
#   scripts/run_vk.sh --threads 4 vkcube --c 300         # in a virtual X server (X11 swapchain)
#   scripts/run_vk.sh --no-x <program>                   # without a virtual X server (headless programs)
#
# No device is involved yet: Vulkan runs entirely on the CPU (api/vulkan/README.md says what is planned for Tensix).
# LP_NUM_THREADS (--threads) sets the number of rasterizer threads of llvmpipe/lavapipe (default: all cores).
set -euo pipefail

prefix="${TT_RASTER_MESA_LLVM_PREFIX:-$HOME/experiments/mesa-install-llvm}"
size="1920x1080"
use_x=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads) export LP_NUM_THREADS="$2"; shift 2 ;;
        --size) size="$2"; shift 2 ;;
        --no-x) use_x=0; shift ;;
        *) break ;;
    esac
done
[[ $# -gt 0 ]] || { echo "run_vk.sh: no program given" >&2; exit 2; }
[[ -f "$prefix/share/vulkan/icd.d/lvp_icd.x86_64.json" ]] || { echo "run_vk.sh: no lavapipe in $prefix (scripts/build_mesa.sh --llvm)" >&2; exit 2; }

export LD_LIBRARY_PATH="$prefix/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES="$prefix/share/vulkan/icd.d/lvp_icd.x86_64.json"
prog="$(command -v "$1")"
shift
if [[ $use_x == 1 ]]; then
    exec "$(command -v xvfb-run)" -a -s "-screen 0 ${size}x24" "$prog" "$@"
fi
exec "$prog" "$@"
