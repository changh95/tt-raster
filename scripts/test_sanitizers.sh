#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# The host tests under AddressSanitizer and UndefinedBehaviorSanitizer (two extra builds in a scratch directory, nothing is
# written to the repository's build directory). With TT_METAL_HOME set, also the shader fuzz on a chip (and four chips)
# under AddressSanitizer: the host side of the device path.
#
#   scripts/test_sanitizers.sh                      host tests only
#   TT_METAL_HOME=... TT_RASTER_NODE=5 scripts/test_sanitizers.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="${TT_RASTER_SAN_DIR:-${TMPDIR:-/tmp}/tt-raster-sanitizers}"
node="${TT_RASTER_NODE:-5}"
mkdir -p "$work"

build() {  # build <name> <sanitizer flags>
    CXXFLAGS="$2 -fno-omit-frame-pointer -g" CFLAGS="$2 -fno-omit-frame-pointer -g" LDFLAGS="$2" \
        TT_RASTER_BUILD_DIR="$work/$1" "$root/scripts/build.sh" >"$work/$1.build.log" 2>&1
}

echo "== AddressSanitizer: host tests"
build asan "-fsanitize=address"
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir "$work/asan" --output-on-failure | grep -E "tests passed|Failed|\*\*\*"

echo "== UndefinedBehaviorSanitizer: host tests"
build ubsan "-fsanitize=undefined -fno-sanitize-recover=undefined"
UBSAN_OPTIONS=print_stacktrace=1 ctest --test-dir "$work/ubsan" --output-on-failure | grep -E "tests passed|Failed|\*\*\*"

if [[ -n "${TT_METAL_HOME:-}" && -x "$work/asan/core/tools/ttr_shader_fuzz" ]]; then
    echo "== AddressSanitizer: shader fuzz on a chip and on four chips"
    export ASAN_OPTIONS=detect_leaks=0
    "$root/scripts/run.sh" "$node" "$work/asan/core/tools/ttr_shader_fuzz" --iters 8 --seed 1 | grep -E "total|^OK|FAILED"
    "$root/scripts/run.sh" 0,1,2,3 "$work/asan/core/tools/ttr_shader_fuzz" --chips 4 --iters 6 --seed 9 | grep -E "total|^OK|FAILED"
fi
echo "sanitizer tests OK"
