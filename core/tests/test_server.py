# SPDX-License-Identifier: Apache-2.0
"""The device server (core/tools/ttr_serverd) and the remote backend of libttraster (TTR_SERVER): frames rendered through the server are
bit-identical to the in-process CPU reference, several clients share one server, a missing, killed or hung server costs a client nothing
but the CPU time of the frame, damaged messages are refused, a second server on one socket does not start. The server runs with
--backend reference, so the test needs no chip (the same protocol serves a chip).

    python core/tests/test_server.py      (numpy; TTRASTER_LIB / TTR_SERVERD or the build/ directory)
"""
import ctypes as C
import os
import random
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
build = os.path.join(here, "..", "..", os.environ.get("TT_RASTER_BUILD_DIR_NAME", "build"))
LIB = os.environ.get("TTRASTER_LIB") or os.path.join(build, "core", "libttraster.so")
SERVERD = os.environ.get("TTR_SERVERD") or os.path.join(build, "core", "tools", "ttr_serverd")

FAILS = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


class Vertex(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("z", C.c_float), ("rhw", C.c_float), ("color", C.c_float * 4)]


class ShaderDesc(C.Structure):
    _fields_ = [("code", C.c_void_p), ("n_instr", C.c_uint32), ("consts", C.c_void_p), ("n_consts", C.c_uint32), ("num_inputs", C.c_uint32),
                ("num_regs", C.c_uint32), ("out_reg", C.c_uint32), ("fragcoord_reg", C.c_int32)]


lib = C.CDLL(LIB)
lib.ttr_last_error.restype = C.c_char_p
lib.ttr_context_create.restype = C.c_void_p
lib.ttr_context_create.argtypes = [C.c_char_p, C.c_uint32]
lib.ttr_context_destroy.argtypes = [C.c_void_p]
lib.ttr_context_report.restype = C.c_char_p
lib.ttr_context_report.argtypes = [C.c_void_p]
lib.ttr_frame_begin.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p, C.c_float, C.c_int]
lib.ttr_frame_triangles.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_int, C.c_int, C.c_void_p]
lib.ttr_frame_end.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_void_p, C.c_int]
lib.ttr_frame_shader.argtypes = [C.c_void_p, C.c_void_p]
lib.ttr_frame_shader.restype = C.c_uint32
lib.ttr_frame_triangles_shaded.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p, C.c_int, C.c_int, C.c_void_p]

W, H = 96, 64
OP_MOV = 1
CONST = 1 << 12


def triangles(seed, n=24):
    rng = random.Random(seed)
    tris = []
    for _ in range(n):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        z = rng.uniform(0.1, 0.9)
        col = [rng.random() for _ in range(3)] + [1.0]
        tris.append([Vertex(cx + rng.uniform(-30, 30), cy + rng.uniform(-30, 30), z + rng.uniform(-0.05, 0.05), 1.0, (C.c_float * 4)(*col)) for _ in range(3)])
    return tris


def render(ctx, seed, shaded=False):
    cc = (C.c_float * 4)(0.1, 0.2, 0.3, 1.0)
    assert lib.ttr_frame_begin(ctx, W, H, cc, 1.0, 0) == 0
    tris = triangles(seed)
    arr = (Vertex * (3 * len(tris)))(*[v for t in tris for v in t])
    if not shaded:
        assert lib.ttr_frame_triangles(ctx, arr, len(tris), 0, 0, None) == 0
    else:
        # r0, r1 = varyings; colour = (r0, r1, constant 0.25, 1)
        words = []
        for dst, src in ((4, 0), (5, 1)):
            words += [OP_MOV | (dst << 16), src, 0, 0]
        words += [OP_MOV | (6 << 16), CONST | 0, 0, 0, OP_MOV | (7 << 16), CONST | 1, 0, 0]
        code = (C.c_uint32 * len(words))(*words)
        consts = (C.c_float * 2)(0.25, 1.0)
        d = ShaderDesc(C.cast(code, C.c_void_p), len(words) // 4, C.cast(consts, C.c_void_p), 2, 2, 8, 4, -1)
        sid = lib.ttr_frame_shader(ctx, C.byref(d))
        assert sid != 0, lib.ttr_last_error()
        rng = random.Random(seed + 1)
        attrs = (C.c_float * (len(tris) * 6))(*[rng.random() for _ in range(len(tris) * 6)])
        assert lib.ttr_frame_triangles_shaded(ctx, arr, len(tris), sid, attrs, 0, 0, None) == 0, lib.ttr_last_error()
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    depth = np.zeros((H, W), dtype=np.float32)
    assert lib.ttr_frame_end(ctx, rgba.ctypes.data, 0, depth.ctypes.data, 0) == 0, lib.ttr_last_error()
    return rgba, depth, lib.ttr_context_report(ctx)


def start_server(sock, extra=(), env=None):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.Popen([SERVERD, "--backend", "reference", "--socket", sock, *extra], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=e)
    for _ in range(100):
        if os.path.exists(sock):
            return p
        if p.poll() is not None:
            break
        time.sleep(0.05)
    raise RuntimeError("server did not start: " + p.stderr.read().decode())


def stop(p):
    if p.poll() is None:
        p.terminate()
    try:
        p.wait(5)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def main():
    tmp = tempfile.mkdtemp(prefix="ttr-server-test-")
    sock = os.path.join(tmp, "s.sock")
    os.environ["TTR_SERVER_RETRY_MS"] = "200"

    ref = lib.ttr_context_create(b"reference", 1)
    expected = {(s, sh): render(ref, s, sh)[:2] for s in range(4) for sh in (False, True)}

    # -- a frame through the server equals the in-process reference ------------------------------------------------------
    server = start_server(sock)
    os.environ["TTR_SERVER"] = sock
    ctx = lib.ttr_context_create(b"tensix", 1)   # TTR_SERVER set: the remote backend
    check(bool(ctx), "a context on the remote backend")
    for s in range(4):
        for sh in (False, True):
            rgba, depth, report = render(ctx, s, sh)
            check(np.array_equal(rgba, expected[(s, sh)][0]) and np.array_equal(depth, expected[(s, sh)][1]) and not report.startswith(b"cpu fallback"),
                  f"frame {s} ({'shader' if sh else 'fixed function'}) through the server equals the reference")

    # -- several clients at once -----------------------------------------------------------------------------------------
    results = []

    def client(k):
        c = lib.ttr_context_create(b"tensix", 1)
        good = True
        for i in range(12):
            s = (k + i) % 4
            rgba, depth, report = render(c, s, i % 2 == 0)
            good &= bool(np.array_equal(rgba, expected[(s, i % 2 == 0)][0]) and np.array_equal(depth, expected[(s, i % 2 == 0)][1]) and not report.startswith(b"cpu fallback"))
        results.append(good)
        lib.ttr_context_destroy(c)

    threads = [threading.Thread(target=client, args=(k,)) for k in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    check(len(results) == 4 and all(results), "four clients render through one server, every frame equal to the reference")

    # -- damaged messages are refused, and the server goes on serving ------------------------------------------------------
    s = socket.socket(socket.AF_UNIX)
    s.connect(sock)
    s.sendall(struct.pack("<IIIIQ", 0x53525454, 1, 2, 1, 10) + b"not a frame")
    hdr = s.recv(24)
    check(len(hdr) == 24 and struct.unpack("<IIIIQ", hdr)[2] == 4, "a message that is not a frame gets an error reply")
    s.close()
    s = socket.socket(socket.AF_UNIX)
    s.connect(sock)
    s.sendall(b"garbage that is no header at all....")
    s.settimeout(3)
    try:
        closed = s.recv(1) == b""
    except socket.timeout:
        closed = False
    except ConnectionResetError:   # the server closed with our unread bytes pending
        closed = True
    check(closed, "a connection that sends garbage is closed")
    s.close()
    rgba, depth, report = render(ctx, 1)
    check(np.array_equal(rgba, expected[(1, False)][0]) and not report.startswith(b"cpu fallback"), "the server still serves after damaged messages")

    # -- a second server on the same socket does not start -----------------------------------------------------------------
    second = subprocess.run([SERVERD, "--backend", "reference", "--socket", sock], capture_output=True, timeout=20)
    check(second.returncode == 1 and b"already listening" in second.stderr, "a second server on the same socket refuses to start")

    # -- the server goes away: the client renders on the CPU, and comes back when a server is there again ---------------------
    stop(server)
    rgba, depth, report = render(ctx, 2)
    check(np.array_equal(rgba, expected[(2, False)][0]) and np.array_equal(depth, expected[(2, False)][1]) and report.startswith(b"cpu fallback"),
          "with the server gone a frame is rendered on the CPU, with the same pixels")
    server = start_server(sock)
    time.sleep(0.4)
    rgba, depth, report = render(ctx, 3)
    check(np.array_equal(rgba, expected[(3, False)][0]) and not report.startswith(b"cpu fallback"), "the client uses a restarted server again")
    stop(server)

    # -- a hung device: the watchdog ends the server with status 3, the client finishes the frame on the CPU -------------------
    server = start_server(sock, ["--frame-timeout", "1"], {"TTR_SERVERD_TEST_HANG_AFTER": "1"})
    for _ in range(20):   # the client finds the new server once its retry delay is over; the first frame it serves passes
        rgba, depth, report = render(ctx, 0)
        if not report.startswith(b"cpu fallback"):
            break
        time.sleep(0.25)
    check(not report.startswith(b"cpu fallback"), "the frame before the hang is served")
    t0 = time.time()
    rgba, depth, report = render(ctx, 1)   # this one never returns on the server
    took = time.time() - t0
    check(np.array_equal(rgba, expected[(1, False)][0]) and report.startswith(b"cpu fallback"), "a frame the server never answered is rendered on the CPU, with the same pixels")
    rc = server.wait(10)
    check(rc == 3 and took < 8, f"the watchdog stopped the hung server with status 3 after about {took:.1f} s")
    check(not os.path.exists(sock), "and removed its socket")

    lib.ttr_context_destroy(ctx)
    lib.ttr_context_destroy(ref)
    print("PASS" if not FAILS else f"FAIL: {len(FAILS)} check(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
