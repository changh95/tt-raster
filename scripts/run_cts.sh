#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run a part of the Khronos Vulkan CTS (dEQP-VK) on lavapipe, with or without the Tensix offload, and print the results by status.
#
#   scripts/run_cts.sh [--backend off|reference|tensix] [--node N] [--threads N] [--log FILE.qpa] <case pattern> [deqp-vk arguments]
#
#   scripts/run_cts.sh 'dEQP-VK.draw.simple_draw.*'
#   scripts/run_cts.sh --backend reference 'dEQP-VK.rasterization.primitive_size.*'
#
# The CTS (vulkan-cts-1.4.7.1) is built by hand in ~/experiments/refs/vk-gl-cts (TT_RASTER_CTS_DIR): `python3 external/fetch_sources.py`,
# `cmake -S . -B build -G Ninja -DDEQP_TARGET=default`, `ninja -C build deqp-vk`. With the offload on, TTR_LP_DEBUG output of the driver goes to
# stderr of the CTS process (it is not logged). The exit status is 1 if any case failed.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cts="${TT_RASTER_CTS_DIR:-$HOME/experiments/refs/vk-gl-cts}/build/external/vulkancts/modules/vulkan"
backend="off"
node="${TT_RASTER_NODE:-5}"
log=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) backend="$2"; shift 2 ;;
        --node) node="$2"; shift 2 ;;
        --threads) export LP_NUM_THREADS="$2"; shift 2 ;;
        --log) log="$2"; shift 2 ;;
        *) break ;;
    esac
done
pattern="${1:?usage: run_cts.sh [--backend off|reference|tensix] [--node N] [--threads N] [--log FILE] <case pattern> [deqp-vk args]}"
shift
[[ -x "$cts/deqp-vk" ]] || { echo "run_cts.sh: no deqp-vk in $cts" >&2; exit 2; }
[[ -n "$log" ]] || log="$(mktemp --suffix=.qpa)"

prefix="${TT_RASTER_MESA_LLVM_PREFIX:-$HOME/experiments/mesa-install-llvm}"
export LD_LIBRARY_PATH="$prefix/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES="$prefix/share/vulkan/icd.d/lvp_icd.x86_64.json"
launch=()
if [[ "$backend" != "off" ]]; then
    export TTR_LP="$backend"
    export TTRASTER_LIB="${TTRASTER_LIB:-${TT_RASTER_BUILD_DIR:-$root/build}/core/libttraster.so}"
    [[ "$backend" == "tensix" ]] && launch=("$root/scripts/run.sh" "$node")
fi

cd "$cts"
set +e
"${launch[@]}" "$cts/deqp-vk" --deqp-archive-dir="$cts" --deqp-case="$pattern" --deqp-log-filename="$log" --deqp-log-images=disable --deqp-log-shader-sources=disable \
    --deqp-surface-type=fbo --deqp-visibility=hidden "$@" > "$log.out" 2>&1
rc=$?
set -e
echo "deqp-vk exit status $rc, log $log"
python3 - "$log.out" <<'PY'
import collections, re, sys
status = collections.Counter()
failed = []
cur = None
for line in open(sys.argv[1], errors="replace"):
    m = re.match(r"Test case '([^']+)'", line)
    if m:
        cur = m.group(1)
        continue
    m = re.match(r"\s+(\w+) \(.*\)\s*$", line)
    if m and cur:
        status[m.group(1)] += 1
        if m.group(1) not in ("Pass", "NotSupported", "QualityWarning", "CompatibilityWarning", "Waiver"):
            failed.append((cur, m.group(1)))
        cur = None
print("  ".join(f"{k}: {v}" for k, v in sorted(status.items())) or "no results")
for name, st in failed[:25]:
    print(f"  {st}  {name}")
if len(failed) > 25:
    print(f"  ... and {len(failed) - 25} more")
sys.exit(1 if failed else 0)
PY
