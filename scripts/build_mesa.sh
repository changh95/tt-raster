#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build a Mesa (softpipe only, GLX on X11) with the tt-raster Tensix fast path (api/gl/README.md).
#
#   scripts/build_mesa.sh
#
# Environment:
#   TT_RASTER_MESA_SRC     Mesa checkout (default ~/experiments/refs/mesa; cloned at tag mesa-26.3.0 if missing)
#   TT_RASTER_MESA_BUILD   build directory   (default ~/experiments/mesa-build)
#   TT_RASTER_MESA_PREFIX  install prefix    (default ~/experiments/mesa-install)
#   TT_RASTER_MESA_VENV    Python venv with meson and mako (default ~/experiments/venv-mesa, created if missing)
#
# System packages needed (Debian/Ubuntu): flex bison libx11-dev libxext-dev libxcb1-dev libx11-xcb-dev
# libxcb-glx0-dev libxcb-dri2-0-dev libxcb-dri3-dev libxcb-present-dev libxcb-sync-dev libxcb-randr0-dev
# libxcb-shm0-dev libxshmfence-dev libxrandr-dev libxfixes-dev libxxf86vm-dev libdrm-dev libexpat1-dev
# python3-venv; to run GL programs without a display also xvfb mesa-utils libgl-dev.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="${TT_RASTER_MESA_SRC:-$HOME/experiments/refs/mesa}"
build="${TT_RASTER_MESA_BUILD:-$HOME/experiments/mesa-build}"
prefix="${TT_RASTER_MESA_PREFIX:-$HOME/experiments/mesa-install}"
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
    meson setup "$build" "$src" --prefix="$prefix" -Dbuildtype=release -Dgallium-drivers=softpipe -Dvulkan-drivers= \
        -Dllvm=disabled -Dplatforms=x11 -Dglx=dri -Degl=enabled -Dgbm=disabled -Dgles1=disabled -Dgles2=enabled \
        -Dvideo-codecs= -Dlmsensors=disabled -Dvalgrind=disabled -Dzstd=disabled -Dgallium-rusticl=false -Dtools= \
        -Dbuild-tests=false
fi
meson configure "$build" -Degl=enabled >/dev/null   # builds made before EGL was added pick it up here
ninja -C "$build"
ninja -C "$build" install
echo "Mesa installed in $prefix; run GL programs with scripts/run_gl.sh"
