#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# The conformant baseline of the OpenGL and Vulkan story (docs/EVALUATION_GL_VULKAN.md), checked on the CPU drivers of the
# `scripts/build_mesa.sh --llvm` build:
#   * lavapipe reports Vulkan >= 1.3 and a conformance version; llvmpipe reports OpenGL >= 4.5;
#   * vkcube renders a few frames through the X11 swapchain;
#   * api/vulkan/tests/vk_offscreen (vertex buffer, render pass, depth test) gives the same image with 1 and 8 rasterizer threads;
#   * the `tensix` GL driver still starts from the same build (softpipe with the fast path, CPU backend of the renderer).
# Needs: the --llvm Mesa build, vulkan-tools (vulkaninfo, vkcube), glslangValidator, libvulkan-dev, xvfb, mesa-utils (glxinfo).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="${TT_RASTER_MESA_LLVM_PREFIX:-$HOME/experiments/mesa-install-llvm}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

info="$("$root/scripts/run_vk.sh" --no-x vulkaninfo --summary 2>&1)"
check "lavapipe is the Vulkan device" 'grep -q "driverName *= llvmpipe" <<<"$info"'
check "Vulkan API version >= 1.3" 'grep -E "apiVersion *= .*\(1\.([3-9]|[1-9][0-9])\." <<<"$info" >/dev/null'
check "lavapipe reports a conformance version" 'grep -E "conformanceVersion *= [1-9]" <<<"$info" >/dev/null'
echo "     $(grep -E 'apiVersion|conformanceVersion|driverInfo' <<<"$info" | head -3 | tr -s ' \t' ' ' | tr '\n' ';')"

gl="$(LD_LIBRARY_PATH="$prefix/lib/x86_64-linux-gnu" LIBGL_DRIVERS_PATH="$prefix/lib/x86_64-linux-gnu/dri" LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
    xvfb-run -a glxinfo -B 2>&1 || true)"
check "llvmpipe offers OpenGL >= 4.5 (compatibility profile)" 'grep -E "OpenGL version string: (4\.[5-9]|[5-9]\.)" <<<"$gl" >/dev/null'

check "vkcube renders 60 frames" '"$root/scripts/run_vk.sh" --size 640x480 vkcube --c 60 --width 640 --height 480 >"$work/vkcube.log" 2>&1'

"$root/api/vulkan/tests/build.sh" "$work/vk" >/dev/null
for t in 1 8; do
    "$root/scripts/run_vk.sh" --no-x --threads "$t" "$work/vk/vk_offscreen" --size 640x480 --tris 2000 --frames 2 --out "$work/o$t.ppm" >/dev/null 2>&1
done
check "vk_offscreen: identical images with 1 and 8 threads" 'cmp -s "$work/o1.ppm" "$work/o8.ppm"'
check "vk_offscreen: the image is not empty" '[[ "$(python3 -c "
d=open(\"$work/o1.ppm\",\"rb\").read()[15:]
print(len(set(d[::97])))")" -gt 10 ]]'

tx="$(LD_LIBRARY_PATH="$prefix/lib/x86_64-linux-gnu" LIBGL_DRIVERS_PATH="$prefix/lib/x86_64-linux-gnu/dri" LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=tensix TTR_GL=reference \
    TTRASTER_LIB="${TT_RASTER_BUILD_DIR:-$root/build}/core/libttraster.so" xvfb-run -a glxinfo -B 2>&1 || true)"
check "the tensix GL driver starts from the same build" 'grep -q "OpenGL renderer string: tensix" <<<"$tx"'

[[ $fail == 0 ]] && echo "Vulkan/OpenGL baseline OK" || { echo "Vulkan/OpenGL baseline FAILED"; exit 1; }
