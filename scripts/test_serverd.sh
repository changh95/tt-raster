#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# The device server under its supervisor (scripts/serverd.sh): a hung frame ends the server with status 3, the supervisor "resets" the chip
# (a stand-in command here) and starts a new server, and the client finds it. The first part runs the server on the CPU reference (no chip);
# with TT_METAL_HOME set a second part serves a real chip (TT_RASTER_NODE, default 5) to client processes and compares their frames with the
# CPU reference (colour identical, depth within the rasterizer's own ambiguity of 1e-6).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="${TT_RASTER_PYTHON:-$HOME/experiments/venv-newton/bin/python}"
build="${TT_RASTER_BUILD_DIR:-$root/build}"
dir="$(mktemp -d /tmp/ttr-sd-XXXXXX)"   # short: a Unix socket path holds at most 107 characters
sock="$dir/s.sock"
sup=""
cleanup() { [[ -n "$sup" ]] && kill -TERM "$sup" 2>/dev/null; wait 2>/dev/null; rm -rf -- "$dir"; }
trap cleanup EXIT

cat > "$dir/client.py" <<PY
import ctypes as C, os, sys, time
sys.argv = ["x"]
os.environ["TT_RASTER_BUILD_DIR_NAME"] = os.path.basename("$build")
sys.path.insert(0, "$root/core/tests")
import numpy as np
import test_server as T
mode = os.environ["CLIENT_MODE"]
ref = T.lib.ttr_context_create(b"reference", 1)
expect = [T.render(ref, s, s % 2 == 1)[:2] for s in range(4)]
os.environ["TTR_SERVER"] = "$sock"
os.environ["TTR_SERVER_RETRY_MS"] = "200"
ctx = T.lib.ttr_context_create(b"tensix", 1)
def same(i, got):
    rgba, depth = got[0], got[1]
    return np.array_equal(rgba, expect[i % 4][0]) and float(np.abs(depth - expect[i % 4][1]).max()) <= 1e-6
if mode == "restart":
    # frames are always right; the server hangs on its second frame (test hook), is restarted, and serves again
    served_after_restart, fallbacks, seen_hang = 0, 0, False
    deadline = time.time() + 60
    i = 0
    while time.time() < deadline and served_after_restart < 3:
        got = T.render(ctx, i % 4, (i % 4) % 2 == 1)
        assert same(i, got), "frame %d differs from the reference" % i
        fb = got[2].startswith(b"cpu fallback")
        fallbacks += fb
        if fb: seen_hang = True
        elif seen_hang: served_after_restart += 1
        i += 1
        time.sleep(0.1)
    print("frames %d, fallbacks %d, served after the restart %d" % (i, fallbacks, served_after_restart))
    sys.exit(0 if served_after_restart >= 3 and fallbacks >= 1 else 1)
else:
    bad = 0
    for i in range(12):
        got = T.render(ctx, i % 4, (i % 4) % 2 == 1)
        bad += (not same(i, got)) + got[2].startswith(b"cpu fallback")
    print("12 frames through the chip server, %d wrong or on the CPU" % bad)
    sys.exit(1 if bad else 0)
PY

echo "== supervisor: a hung frame, a (stand-in) reset, a new server"
TT_RASTER_SERVERD_BACKEND=reference TT_RASTER_RESET_CMD="echo reset-called" TTR_SERVERD_TEST_HANG_AFTER=1 \
    "$root/scripts/serverd.sh" 5 --socket "$sock" --frame-timeout 1 >"$dir/sup.log" 2>&1 &
sup=$!
for _ in $(seq 1 100); do [[ -S "$sock" ]] && break; sleep 0.1; done
CLIENT_MODE=restart "$py" "$dir/client.py"
grep -q "hung device" "$dir/sup.log" && grep -q "reset-called" "$dir/sup.log" && echo "ok   the supervisor saw the hang and ran its reset command"
kill -TERM "$sup"; wait "$sup" 2>/dev/null || true; sup=""
echo "ok   the supervisor stops on SIGTERM"

if [[ -n "${TT_METAL_HOME:-}" ]]; then
    node="${TT_RASTER_NODE:-5}"
    echo "== a chip (node $node) behind the server, three client processes"
    "$root/scripts/serverd.sh" "$node" --socket "$sock" --frame-timeout 60 >"$dir/chip.log" 2>&1 &
    sup=$!
    for _ in $(seq 1 120); do [[ -S "$sock" ]] && break; sleep 0.5; done
    [[ -S "$sock" ]] || { echo "FAIL the chip server did not start"; tail -5 "$dir/chip.log"; exit 1; }
    pids=()
    for k in 1 2 3; do CLIENT_MODE=chip "$py" "$dir/client.py" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done
    echo "ok   three processes shared one chip"
    kill -TERM "$sup"; wait "$sup" 2>/dev/null || true; sup=""
fi
echo "serverd tests OK"
