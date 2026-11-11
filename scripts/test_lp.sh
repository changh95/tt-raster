#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# The Tensix offload of llvmpipe (api/gl/llvmpipe_tensix, patch 0002) through OpenGL and through Vulkan (lavapipe): the same programs run on
# llvmpipe/lavapipe alone and with the offload, and the images are compared (at most 0.2 % of the pixels may differ by more than 4 levels,
# the edge pixels where the two rasterizers resolve the fill rule differently; 0.5 % for the Vulkan scene of 3000 small triangles). With the CPU backend of libttraster always; with
# TT_METAL_HOME set also on a chip (TT_RASTER_NODE, default 5), and there also with compiled shaders (TTR_SHADER_COMPILE=1).
# Needs the `scripts/build_mesa.sh --llvm` build, the host build (libttraster.so), vulkan-tools' loader, glslangValidator, xvfb, a C compiler
# with GL and X11 headers, and a Python with numpy (TT_RASTER_PYTHON).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="${TT_RASTER_MESA_LLVM_PREFIX:-$HOME/experiments/mesa-install-llvm}"
build="${TT_RASTER_BUILD_DIR:-$root/build}"
py="${TT_RASTER_PYTHON:-$HOME/experiments/venv-newton/bin/python}"
node="${TT_RASTER_NODE:-5}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail=0

gcc -O1 -o "$work/gl_scene" "$root/api/gl/tests/gl_scene.c" -lGL -lX11 -lm
"$root/api/vulkan/tests/build.sh" "$work/vk" >/dev/null

gl() {  # gl <out prefix> <backend: off|reference|tensix> <gl_scene args...>
    local out="$1" be="$2"; shift 2
    local env=(LD_LIBRARY_PATH="$prefix/lib/x86_64-linux-gnu" LIBGL_DRIVERS_PATH="$prefix/lib/x86_64-linux-gnu/dri" LIBGL_ALWAYS_SOFTWARE=1
               GALLIUM_DRIVER=llvmpipe TTRASTER_LIB="$build/core/libttraster.so" TTR_LP_DEBUG=1)
    [[ "$be" == off ]] || env+=(TTR_LP="$be")
    local launch=()
    [[ "$be" == tensix ]] && launch=("$root/scripts/run.sh" "$node")
    env "${env[@]}" "${launch[@]}" "$(command -v xvfb-run)" -a -s "-screen 0 400x400x24" "$work/gl_scene" "$@" --size 256 --dump "$out" >"$out.log" 2>&1
}

cmp_gl() {  # cmp_gl <name> <backend> <gl_scene args...>
    local name="$1" be="$2"; shift 2
    gl "$work/${name}_off" off "$@"
    gl "$work/${name}_$be" "$be" "$@"
    local taken
    taken="$(grep -c 'frame rendered' "$work/${name}_$be.log" || true)"
    if "$py" "$root/api/gl/tests/compare_ppm.py" "$work/${name}_off0.ppm" "$work/${name}_${be}0.ppm" --max-bad 0.002 >"$work/cmp.txt"; then
        echo "ok   GL $name ($be, $taken frame(s) offloaded): $(sed 's/.*ppm: //' "$work/cmp.txt")"
    else
        echo "FAIL GL $name ($be): $(sed 's/.*ppm: //' "$work/cmp.txt")"; fail=1
    fi
}

cmp_vk() {  # cmp_vk <name> <backend> <vk_offscreen args...>   (MAX_BAD: share of pixels that may differ by more than 4 levels)
    local name="$1" be="$2"; shift 2
    TTR_SHADER_COMPILE="${COMPILE:-}" "$root/scripts/run_vk.sh" --no-x --backend off "$work/vk/vk_offscreen" "$@" --out "$work/${name}_off.ppm" >/dev/null 2>&1
    TTR_SHADER_COMPILE="${COMPILE:-}" TTR_LP_DEBUG=1 "$root/scripts/run_vk.sh" --no-x --backend "$be" --node "$node" "$work/vk/vk_offscreen" "$@" \
        --out "$work/${name}_$be.ppm" >"$work/${name}_$be.log" 2>&1
    local taken
    taken="$(grep -c 'frame rendered' "$work/${name}_$be.log" || true)"
    if "$py" "$root/api/gl/tests/compare_ppm.py" "$work/${name}_off.ppm" "$work/${name}_$be.ppm" --max-bad "${MAX_BAD:-0.002}" >"$work/cmp.txt" && [[ "$taken" -gt 0 ]]; then
        echo "ok   Vulkan $name ($be${COMPILE:+, compiled shaders}, $taken frame(s) offloaded): $(sed 's/.*ppm: //' "$work/cmp.txt")"
    else
        echo "FAIL Vulkan $name ($be): offloaded $taken, $(sed 's/.*ppm: //' "$work/cmp.txt")"; fail=1
    fi
}

cmp_present() {  # cmp_present <name> <backend> <vk_present args...>: frames through an X11 swapchain, read back from the X server
    local name="$1" be="$2"; shift 2
    "$root/scripts/run_vk.sh" --size 640x480 --backend off "$work/vk/vk_present" "$@" --out "$work/${name}_off.ppm" >"$work/${name}_off.log" 2>&1
    TTR_LP_DEBUG=1 "$root/scripts/run_vk.sh" --size 640x480 --backend "$be" --node "$node" "$work/vk/vk_present" "$@" \
        --out "$work/${name}_$be.ppm" >"$work/${name}_$be.log" 2>&1
    local taken
    taken="$(grep -c 'frame rendered' "$work/${name}_$be.log" || true)"
    if "$py" "$root/api/gl/tests/compare_ppm.py" "$work/${name}_off.ppm" "$work/${name}_$be.ppm" --max-bad "${MAX_BAD:-0.002}" >"$work/cmp.txt" && [[ "$taken" -gt 0 ]] &&
       grep -q "differ in 0 of" "$work/${name}_$be.log"; then
        echo "ok   Vulkan $name ($be, $taken frame(s) offloaded, window equals the copy of the image): $(sed 's/.*ppm: //' "$work/cmp.txt")"
    else
        echo "FAIL Vulkan $name ($be): offloaded $taken, $(sed 's/.*ppm: //' "$work/cmp.txt"); $(grep differ "$work/${name}_$be.log")"; fail=1
    fi
}

backends=(reference)
[[ -n "${TT_METAL_HOME:-}" ]] && backends+=(tensix)
for be in "${backends[@]}"; do
    cmp_gl default "$be"
    cmp_gl lighting "$be" --lighting
    cmp_gl blend "$be" --blend
    cmp_gl cull "$be" --cull
    cmp_gl depthmask "$be" --depthmask
    cmp_gl nodepth "$be" --nodepth
    cmp_gl ortho "$be" --ortho --depth --continue --frames 3
    cmp_gl phong "$be" --fs phong --sphere
    cmp_gl lambert "$be" --fs lambert
    cmp_gl smooth "$be" --fs smooth
    cmp_gl stripes "$be" --fs stripes     # gl_FragCoord (which llvmpipe's NIR flips itself) and a uniform loaded from the buffer
    cmp_gl discard "$be" --fs discard     # DEMOTE
    cmp_gl fog "$be" --fs fog
    cmp_gl cond "$be" --fs cond
    cmp_gl loop "$be" --fs loop
    cmp_gl uniform "$be" --fs uniform --continue --frames 3
    MAX_BAD=0.005 cmp_vk triangles "$be" --size 320x240 --tris 3000   # 3000 triangles of about 25 pixels: edge pixels are a larger share
    cmp_vk shader "$be" --size 320x240 --tris 0 --layers 2
    cmp_present swapchain "$be" --tris 2000 --frames 4
done
if [[ -n "${TT_METAL_HOME:-}" ]]; then
    COMPILE=1 cmp_vk shader_compiled tensix --size 640x360 --tris 0 --layers 3
fi

[[ $fail == 0 ]] && echo "llvmpipe/lavapipe offload OK" || { echo "llvmpipe/lavapipe offload FAILED"; exit 1; }
