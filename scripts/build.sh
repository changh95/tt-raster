#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Configure and build tt-raster.
#   scripts/build.sh [extra cmake --build args]                       host library, tools and tests only
#   TT_METAL_HOME=/path/to/tt-metal scripts/build.sh [...]            also the device spikes and renderer
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${TT_RASTER_BUILD_DIR:-$root/build}"

args=(-S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release)
if [[ -n "${TT_METAL_HOME:-}" ]]; then
    # Must match the toolchain tt-metal itself was built with (ABI).
    toolchain="$(sed -n 's/^CMAKE_TOOLCHAIN_FILE:FILEPATH=//p' "$TT_METAL_HOME/build_Release/CMakeCache.txt")"
    args+=(-DCMAKE_TOOLCHAIN_FILE="$toolchain" -DTT_METAL_HOME="$TT_METAL_HOME")
fi

cmake "${args[@]}"
cmake --build "$build" "$@"
