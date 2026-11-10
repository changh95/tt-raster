#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run the host tests (CPU reference, setup, oracle, goldens). No device needed.
#   scripts/test.sh [ctest args]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${TT_RASTER_BUILD_DIR:-$root/build}"
ctest --test-dir "$build" --output-on-failure "$@"
