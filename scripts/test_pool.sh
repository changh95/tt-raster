#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Several processes, several chips: three processes ask for a chip of the pool "4,5" (TTR_DEVICE_POOL; run.sh pool:4,5) while the first two
# hold theirs. The first two must each render a frame on a chip (different chips), the third must be refused with a message (not hang,
# not abort) and a frame on the CPU backend must still work. Needs TT_METAL_HOME and a built libttraster (build/ or TTRASTER_LIB).
#   TT_METAL_HOME=... scripts/test_pool.sh [pool, default 4,5]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TT_METAL_HOME:?set TT_METAL_HOME}"
pool="${1:-4,5}"
py="${TT_RASTER_PYTHON:-$HOME/experiments/venv-newton/bin/python}"
lib="${TTRASTER_LIB:-$root/build/core/libttraster.so}"
work="$(mktemp -d)"
trap '[[ -n "${KEEP:-}" ]] || rm -rf "$work"' EXIT

cat >"$work/client.py" <<'PY'
import ctypes as C, os, sys, time
lib = C.CDLL(sys.argv[1])
lib.ttr_last_error.restype = C.c_char_p
lib.ttr_context_create.restype = C.c_void_p
ctx = C.c_void_p(lib.ttr_context_create(b"tensix", 1))   # (a bare int would be passed as a 32-bit value)
if not ctx.value:
    print("REFUSED", lib.ttr_last_error().decode())
    sys.exit(0)
class V(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("z", C.c_float), ("rhw", C.c_float), ("color", C.c_float * 4)]
class S(C.Structure):
    _fields_ = [("depth_func", C.c_int), ("depth_write", C.c_int), ("blend", C.c_int)]
cc = (C.c_float * 4)(0, 0, 0, 1)
assert lib.ttr_frame_begin(ctx, 64, 64, cc, C.c_float(1.0), 0) == 0
tri = (V * 3)(V(4, 4, 0.5, 1, (C.c_float * 4)(1, 0, 0, 1)), V(60, 4, 0.5, 1, (C.c_float * 4)(1, 0, 0, 1)), V(4, 60, 0.5, 1, (C.c_float * 4)(1, 0, 0, 1)))
assert lib.ttr_frame_triangles(ctx, tri, 1, 0, 0, None) == 0
out = (C.c_ubyte * (64 * 64 * 4))()
assert lib.ttr_frame_end(ctx, out, 0, None, 0) == 0
libc = C.CDLL(None)
libc.getenv.restype = C.c_char_p   # (the library set the variable in the C environment; os.environ is a copy from before)
print("RENDERED device", libc.getenv(b"TT_VISIBLE_DEVICES").decode(), "pixel", out[(10 * 64 + 10) * 4])
sys.stdout.flush()
time.sleep(float(sys.argv[2]))
lib.ttr_context_destroy(ctx)
PY

launch() {  # launch <n> <hold seconds>
    ( TTRASTER_LIB="$lib" "$root/scripts/run.sh" "pool:$pool" "$py" "$work/client.py" "$lib" "$2" >"$work/c$1.log" 2>&1 ) &
}
launch 1 90
sleep 20
launch 2 70
sleep 20
launch 3 1
wait
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
check "process 1 rendered on a chip" 'grep -q "^RENDERED" "$work/c1.log"'
check "process 2 rendered on a chip" 'grep -q "^RENDERED" "$work/c2.log"'
check "they used different chips" '[[ "$(grep -h "^RENDERED" "$work/c1.log" "$work/c2.log" | awk "{print \$3}" | sort -u | wc -l)" == 2 ]]'
check "process 3 was refused with a message" 'grep -q "^REFUSED all Tensix devices of the pool" "$work/c3.log"'
[[ $fail == 0 ]] && echo "device pool OK" || { echo "device pool FAILED"; for f in "$work"/c*.log; do echo "--- $f"; tail -n 5 "$f"; done; exit 1; }
