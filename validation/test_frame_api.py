# SPDX-License-Identifier: Apache-2.0
"""The frame-level Raster API of libttraster (ttr_context_*, ttr_frame_*): orientation, depth functions, culling,
shading modes, errors. Uses the CPU backend; the same calls run on a chip with backend "tensix".

    venv-newton/bin/python validation/test_frame_api.py [--backend reference|tensix]
"""
import argparse
import ctypes as C
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import ttraster  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


class State(C.Structure):
    _fields_ = [("depth_func", C.c_int), ("depth_write", C.c_int), ("blend", C.c_int)]


class Vertex(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("z", C.c_float), ("rhw", C.c_float), ("color", C.c_float * 4)]


def vert(x, y, z=0.5, w=1.0, c=(1, 1, 1, 1)):
    return Vertex(x, y, z, 1.0 / w, (C.c_float * 4)(*c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="reference")
    args = ap.parse_args()
    lib = ttraster._l()
    lib.ttr_context_create.restype = C.c_void_p
    lib.ttr_context_create.argtypes = [C.c_char_p, C.c_uint32]
    lib.ttr_context_destroy.argtypes = [C.c_void_p]
    lib.ttr_frame_begin.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p, C.c_float, C.c_int]
    lib.ttr_frame_triangles.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_int, C.c_int, C.c_void_p]
    lib.ttr_frame_end.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_void_p, C.c_int]
    LESS, LEQUAL, ALWAYS = 0, 1, 2
    LINEAR, FLAT, PERSP = 0, 1, 2
    W, H = 64, 48
    ctx = lib.ttr_context_create(args.backend.encode(), 1)
    check(bool(ctx), f"context created ({args.backend})")

    def frame(tris, shading=LINEAR, cull=0, depth_func=LESS, clear=(0, 0, 0, 1), clear_depth=1.0, flip=0, want_depth=True, state=None):
        """tris: list of 3-tuples of Vertex. Returns (rgba [H, W, 4] uint8, depth [H, W] float32 or None)."""
        cc = (C.c_float * 4)(*clear)
        assert lib.ttr_frame_begin(ctx, W, H, cc, clear_depth, depth_func) == 0
        arr = (Vertex * (3 * len(tris)))(*[v for t in tris for v in t])
        if tris:
            assert lib.ttr_frame_triangles(ctx, arr, len(tris), shading, cull, C.byref(state) if state else None) == 0, ttraster._l().ttr_last_error()
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        depth = np.zeros((H, W), dtype=np.float32) if want_depth else None
        assert lib.ttr_frame_end(ctx, rgba.ctypes.data, 0, depth.ctypes.data if want_depth else None, flip) == 0
        return rgba, depth

    red = (1.0, 0.0, 0.0, 1.0)
    # Clear only.
    img, z = frame([], clear=(0.2, 0.4, 0.6, 1.0), clear_depth=0.75)
    check(tuple(img[5, 5]) == (51, 102, 153, 255) and abs(z[5, 5] - 0.75) < 1e-6, "an empty frame is the clear colour and depth")
    # A flat rectangle (two triangles) x in [8, 40], y in [4, 20]: y goes down, rows are not flipped by default.
    quad = [(vert(8, 4, c=red), vert(40, 4, c=red), vert(40, 20, c=red)), (vert(8, 4, c=red), vert(40, 20, c=red), vert(8, 20, c=red))]
    img, z = frame(quad, depth_func=LESS)
    inside = img[4:20, 8:40]
    check(np.all(inside == np.array([255, 0, 0, 255])) and np.all(img[24:, :, :3] == 0), "rectangle fills pixels 8..39 x 4..19, y down")
    check(abs(float(z[10, 20]) - 0.5) < 1e-6 and z[30, 5] == 1.0, "depth plane holds z inside and the clear depth outside")
    flipped, _ = frame(quad, flip=1)
    check(np.array_equal(flipped, img[::-1]), "flip_y reverses the rows")
    # Depth functions: a nearer green triangle drawn first, then a farther red one over it.
    green = (0.0, 1.0, 0.0, 1.0)
    near = [(vert(8, 4, z=0.2, c=green), vert(40, 4, z=0.2, c=green), vert(40, 20, z=0.2, c=green))]
    far = [(vert(8, 4, z=0.8, c=red), vert(40, 4, z=0.8, c=red), vert(40, 20, z=0.8, c=red))]
    img, _ = frame(near + far, depth_func=LESS)
    check(tuple(img[8, 30]) == (0, 255, 0, 255), "LESS keeps the nearer triangle")
    img, _ = frame(near + far, depth_func=ALWAYS)
    check(tuple(img[8, 30]) == (255, 0, 0, 255), "ALWAYS (depth test off): the later triangle wins")
    same = [(vert(8, 4, z=0.5, c=green), vert(40, 4, z=0.5, c=green), vert(40, 20, z=0.5, c=green)),
            (vert(8, 4, z=0.5, c=red), vert(40, 4, z=0.5, c=red), vert(40, 20, z=0.5, c=red))]
    check(tuple(frame(same, depth_func=LESS)[0][8, 30]) == (0, 255, 0, 255), "LESS: an equal depth does not win")
    check(tuple(frame(same, depth_func=LEQUAL)[0][8, 30]) == (255, 0, 0, 255), "LEQUAL: the later triangle wins an equal depth")
    # Culling by the sign of the window-space area.
    tri = [(vert(8, 4, c=red), vert(40, 4, c=red), vert(40, 20, c=red))]   # positive area (y down)
    check(np.any(frame(tri, cull=0)[0][:, :, 0] > 0), "no culling draws the triangle")
    check(not np.any(frame(tri, cull=1)[0][:, :, 0] > 0), "culling positive area removes it")
    check(np.any(frame(tri, cull=2)[0][:, :, 0] > 0), "culling negative area keeps it")
    # Shading: flat takes the first vertex, linear interpolates, perspective differs when w varies.
    cols = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
    t = [(vert(4, 4, c=cols[0], w=1), vert(60, 8, c=cols[1], w=1), vert(30, 44, c=cols[2], w=8))]
    flat = frame(t, shading=FLAT)[0]
    check(tuple(flat[20, 30]) == (255, 0, 0, 255), "flat shading uses the colour of the first vertex")
    lin = frame(t, shading=LINEAR)[0].astype(int)
    persp = frame(t, shading=PERSP)[0].astype(int)
    check(int(np.abs(lin - persp).max()) > 20, "perspective shading differs from linear when w varies")
    eq = [(vert(4, 4, c=cols[0], w=2), vert(60, 8, c=cols[1], w=2), vert(30, 44, c=cols[2], w=2))]
    check(int(np.abs(frame(eq, shading=LINEAR)[0].astype(int) - frame(eq, shading=PERSP)[0].astype(int)).max()) <= 1,
          "with equal w the two agree")
    # Own state: depth writes off, blending, depth test off without depth writes (GL semantics).
    half_red = (1.0, 0.0, 0.0, 0.5)
    base = [(vert(8, 4, z=0.6, c=green), vert(40, 4, z=0.6, c=green), vert(40, 20, z=0.6, c=green))]
    over = [(vert(8, 4, z=0.3, c=half_red), vert(40, 4, z=0.3, c=half_red), vert(40, 20, z=0.3, c=half_red))]
    BLEND_ALPHA = 1

    def two_batches(state):
        """The green triangle with the default state, then the half-red one with `state`."""
        cc = (C.c_float * 4)(0, 0, 0, 1)
        assert lib.ttr_frame_begin(ctx, W, H, cc, 1.0, LESS) == 0
        for tris, st in ((base, None), (over, state)):
            arr = (Vertex * (3 * len(tris)))(*[v for t in tris for v in t])
            assert lib.ttr_frame_triangles(ctx, arr, len(tris), LINEAR, 0, C.byref(st) if st else None) == 0
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        depth = np.zeros((H, W), dtype=np.float32)
        assert lib.ttr_frame_end(ctx, rgba.ctypes.data, 0, depth.ctypes.data, 0) == 0
        return rgba, depth

    img, z = two_batches(State(LESS, 0, BLEND_ALPHA))
    check(abs(int(img[8, 30][0]) - 128) <= 1 and abs(int(img[8, 30][1]) - 128) <= 1, "alpha blending over the green triangle")
    check(abs(float(z[8, 30]) - 0.6) < 1e-6, "a blended batch with depth writes off leaves the depth alone")
    img, z = two_batches(State(ALWAYS, 0, 0))
    check(abs(int(img[8, 30][0]) - 255) <= 0 and int(img[8, 30][1]) == 0, "depth test off: the later triangle is drawn over")
    check(abs(float(z[8, 30]) - 0.6) < 1e-6, "depth test off does not write depth")
    # Errors and misuse.
    cc = (C.c_float * 4)(0, 0, 0, 1)
    check(lib.ttr_frame_begin(ctx, 4096, 16, cc, 1.0, LESS) != 0, "a target above 2048 pixels is refused")
    arr = (Vertex * 3)(*tri[0])
    check(lib.ttr_frame_triangles(ctx, arr, 1, LINEAR, 0, None) != 0, "triangles outside a frame are refused")
    lib.ttr_frame_begin(ctx, W, H, cc, 1.0, LESS)
    lib.ttr_frame_triangles(ctx, arr, 1, LINEAR, 1, None)
    check(lib.ttr_frame_triangles(ctx, arr, 1, LINEAR, 2, None) != 0, "mixed cull modes in one frame are refused")
    lib.ttr_context_destroy(ctx)
    print(f"   backend: {args.backend}")
    if FAILS:
        print(f"{len(FAILS)} check(s) failed")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
