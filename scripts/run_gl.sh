#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run an OpenGL program (X11/GLX) on the Mesa built by scripts/build_mesa.sh, rasterized by the Tensix fast path,
# inside a virtual X server.
#
#   scripts/run_gl.sh <node> [--size WxH] [--backend tensix|reference|off] [--driver llvmpipe] <program> [args...]
#
#   scripts/run_gl.sh 5 glxgears -geometry 1920x1080
#   scripts/run_gl.sh 5 --backend off glxgears        # the same Mesa, stock softpipe (for comparisons)
#   scripts/run_gl.sh 5 --driver llvmpipe glxgears    # the conformant CPU driver (OpenGL 4.6) of the `build_mesa.sh --llvm` build
#   scripts/run_gl.sh 5 --driver llvmpipe --backend tensix glxgears   # llvmpipe with the Tensix offload (TTR_LP)
#
# `node` is a device node 0-7 of the assigned tray (see scripts/run.sh). TTR_GL_DEBUG=1 prints the fast path's
# decisions and per-frame timings.
set -euo pipefail

: "${TT_METAL_HOME:?set TT_METAL_HOME to a built tt-metal checkout}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="${TT_RASTER_MESA_PREFIX:-$HOME/experiments/mesa-install}"
build="${TT_RASTER_BUILD_DIR:-$root/build}"

node="${1:?usage: run_gl.sh <node> [--size WxH] [--backend tensix|reference|off] <program> [args...]}"
shift
size="1920x1080"
backend="tensix"
backend_set=0
driver=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --driver) driver="$2"; shift 2 ;;
        --size) size="$2"; shift 2 ;;
        --backend) backend="$2"; backend_set=1; shift 2 ;;
        *) break ;;
    esac
done
[[ $# -gt 0 ]] || { echo "run_gl.sh: no program given" >&2; exit 2; }

if [[ "$driver" == "llvmpipe" ]]; then
    prefix="${TT_RASTER_MESA_LLVM_PREFIX:-$HOME/experiments/mesa-install-llvm}"
    [[ $backend_set == 1 ]] || backend="off"   # llvmpipe alone, unless --backend tensix|reference asks for the Tensix offload (TTR_LP)
elif [[ -n "$driver" ]]; then
    echo "run_gl.sh: unknown driver '$driver' (llvmpipe, or leave it out for the tensix/softpipe Mesa)" >&2; exit 2
fi
lib="$prefix/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBGL_DRIVERS_PATH="$lib/dri"
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER="${driver:-softpipe}"
# Below about 300,000 pixels (550 x 550) stock softpipe is faster than a chip frame (api/gl/README.md); set 0 to always use the chip.
export TTR_GL_MIN_PIXELS="${TTR_GL_MIN_PIXELS:-300000}"
export TTRASTER_LIB="${TTRASTER_LIB:-$build/core/libttraster.so}"
if [[ "$backend" == "off" ]]; then
    unset TTR_GL TTR_LP
elif [[ "$driver" == "llvmpipe" ]]; then
    unset TTR_GL
    export TTR_LP="$backend"
else
    export TTR_GL="$backend"
fi

prog="$(command -v "$1")"
shift
exec "$root/scripts/run.sh" "$node" "$(command -v xvfb-run)" -a -s "-screen 0 ${size}x24" "$prog" "$@"
