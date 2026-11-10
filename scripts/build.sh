#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Configure and build tt-raster against a built tt-metalium checkout.
#   TT_METAL_HOME=/path/to/tt-metal scripts/build.sh [extra cmake --build args]
set -euo pipefail

: "${TT_METAL_HOME:?set TT_METAL_HOME to a built tt-metal checkout}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${TT_RASTER_BUILD_DIR:-$root/build}"

# Must match the toolchain tt-metal itself was built with (ABI).
toolchain="$(sed -n 's/^CMAKE_TOOLCHAIN_FILE:FILEPATH=//p' "$TT_METAL_HOME/build_Release/CMakeCache.txt")"

cmake -S "$root" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$toolchain" \
    -DTT_METAL_HOME="$TT_METAL_HOME"
cmake --build "$build" "$@"
