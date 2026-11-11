#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build a Mesa (GLX on X11) with the tt-raster Tensix fast path (api/gl/README.md).
#
#   scripts/build_mesa.sh          softpipe only: the `tensix` driver (GL 3.3), small and quick
#   scripts/build_mesa.sh --llvm   llvmpipe + softpipe + lavapipe: the conformant CPU drivers (OpenGL 4.6, Vulkan 1.4) next to the
#                                  `tensix` driver, one Mesa for both APIs (docs/EVALUATION_GL_VULKAN.md, api/vulkan/README.md)
#
# Environment:
#   TT_RASTER_MESA_SRC     Mesa checkout (default ~/experiments/refs/mesa; cloned at tag mesa-26.3.0 if missing)
#   TT_RASTER_MESA_BUILD   build directory   (default ~/experiments/mesa-build, with --llvm ~/experiments/mesa-build-llvm)
#   TT_RASTER_MESA_PREFIX  install prefix    (default ~/experiments/mesa-install, with --llvm ~/experiments/mesa-install-llvm)
#   TT_RASTER_LLVM_CONFIG  llvm-config for --llvm (default /usr/bin/llvm-config-20)
#   --llvm also needs glslang >= 12.2 on PATH (Ubuntu 22.04 packages 11.8: take a release from
#   https://github.com/KhronosGroup/glslang/releases) and libllvm-dev; `vulkan-tools` (vulkaninfo, vkcube) to run Vulkan programs.
#   TT_RASTER_MESA_VENV    Python venv with meson and mako (default ~/experiments/venv-mesa, created if missing)
#
# System packages needed (Debian/Ubuntu): flex bison libx11-dev libxext-dev libxcb1-dev libx11-xcb-dev
# libxcb-glx0-dev libxcb-dri2-0-dev libxcb-dri3-dev libxcb-present-dev libxcb-sync-dev libxcb-randr0-dev
# libxcb-shm0-dev libxshmfence-dev libxrandr-dev libxfixes-dev libxxf86vm-dev libdrm-dev libexpat1-dev
# python3-venv; to run GL programs without a display also xvfb mesa-utils libgl-dev.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llvm=0
if [[ "${1:-}" == "--llvm" ]]; then llvm=1; shift; fi
src="${TT_RASTER_MESA_SRC:-$HOME/experiments/refs/mesa}"
if [[ $llvm == 1 ]]; then
    build="${TT_RASTER_MESA_BUILD:-$HOME/experiments/mesa-build-llvm}"
    prefix="${TT_RASTER_MESA_PREFIX:-$HOME/experiments/mesa-install-llvm}"
else
    build="${TT_RASTER_MESA_BUILD:-$HOME/experiments/mesa-build}"
    prefix="${TT_RASTER_MESA_PREFIX:-$HOME/experiments/mesa-install}"
fi
venv="${TT_RASTER_MESA_VENV:-$HOME/experiments/venv-mesa}"

if [[ ! -d "$venv" ]]; then
    python3 -m venv "$venv"
    "$venv/bin/pip" install -q meson mako pyyaml packaging
fi
export PATH="$venv/bin:$PATH"

if [[ ! -d "$src/.git" ]]; then
    mkdir -p "$(dirname "$src")"
    git clone --depth 1 --branch mesa-26.3.0 https://gitlab.freedesktop.org/mesa/mesa.git "$src"
fi

# Apply the patch once (it touches softpipe and the software driver helpers) and copy the new files.
patch_file="$root/api/gl/patches/0001-softpipe-tensix-fast-path.patch"
if git -C "$src" apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "patch already applied"
else
    git -C "$src" apply "$patch_file"
fi
cp "$root/api/gl/softpipe_tensix/"sp_tensix*.[ch] "$root/core/include/ttraster/ttraster_isa.h" "$src/src/gallium/drivers/softpipe/"

if [[ ! -d "$build" ]]; then
    common=(-Dbuildtype=release -Dplatforms=x11 -Dglx=dri -Degl=enabled -Dgbm=disabled -Dgles1=disabled -Dgles2=enabled
        -Dvideo-codecs= -Dlmsensors=disabled -Dvalgrind=disabled -Dzstd=disabled -Dgallium-rusticl=false -Dtools= -Dbuild-tests=false)
    if [[ $llvm == 1 ]]; then
        native="$(mktemp)"
        printf "[binaries]\nllvm-config = '%s'\n" "${TT_RASTER_LLVM_CONFIG:-/usr/bin/llvm-config-20}" > "$native"
        meson setup "$build" "$src" --native-file "$native" --prefix="$prefix" "${common[@]}" -Dgallium-drivers=llvmpipe,softpipe \
            -Dvulkan-drivers=swrast -Dvulkan-layers= -Dllvm=enabled -Dshared-llvm=enabled
        rm -f "$native"
    else
        meson setup "$build" "$src" --prefix="$prefix" "${common[@]}" -Dgallium-drivers=softpipe -Dvulkan-drivers= -Dllvm=disabled
    fi
fi
meson configure "$build" -Degl=enabled >/dev/null   # builds made before EGL was added pick it up here
ninja -C "$build"
ninja -C "$build" install
if [[ $llvm == 1 ]]; then
    echo "Mesa (llvmpipe, softpipe+tensix, lavapipe) installed in $prefix; run GL programs with scripts/run_gl.sh --driver llvmpipe|tensix, Vulkan programs with scripts/run_vk.sh"
else
    echo "Mesa installed in $prefix; run GL programs with scripts/run_gl.sh"
fi
