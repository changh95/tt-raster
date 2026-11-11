# SPDX-License-Identifier: Apache-2.0
"""Pipelined rendering (Scene.submit / Scene.collect) must give exactly the images of synchronous render() calls,
whatever the poses did between submit and collect, and misuse must fail with an error, not a crash.

    venv-newton/bin/python validation/test_pipeline.py [--backend reference|tensix] [--worlds 16]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import scene_spec as S  # noqa: E402
import ttraster  # noqa: E402
from compare_scene import build_scene  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def poses_for(sc, W, rep):
    p = np.zeros((W, 4, 7), dtype=np.float32)
    p[..., 6] = 1
    for w in range(W):
        x, th = S.world_params(w + rep * 7)
        p[w, 1] = (*S.RAIL["center"], 0, 0, 0, 1)
        p[w, 2] = (x, 0, 1.0, 0, 0, 0, 1)
        p[w, 3] = (x, 0, 1.0, *S.quat_y(th))
    return p.reshape(-1, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="reference")
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()
    W, N = args.worlds, args.size
    sc = build_scene(W, args.backend, 1, 4, 16)
    cam = ttraster.Camera(width=N, height=N, focal_length=S.CAMERA["focal_length"],
                          horizontal_aperture=S.CAMERA["horizontal_aperture"], vertical_aperture=S.CAMERA["horizontal_aperture"],
                          near_clip=S.CAMERA["near"], far_clip=S.CAMERA["far"])
    cams = np.tile(np.array([*S.CAMERA["position"], *S.CAMERA_QUAT], dtype=np.float32), (W, 1))
    outputs = ("depth", "shape", "rgb")
    frames = 6

    # Reference: synchronous frames.
    sync = []
    for k in range(frames):
        sc.set_instance_poses(poses_for(sc, W, k))
        out, _ = sc.render(cam, cams, outputs)
        sync.append(out)
    check(not all(np.array_equal(sync[0]["depth"], sync[k]["depth"]) for k in range(1, frames)), "the frames differ from each other")

    # Pipelined: two in flight, poses change between submits, outputs double-buffered.
    bufs = [{}, {}]
    tickets = []
    results = {}
    for k in range(frames):
        if len(tickets) == 2:  # both slots busy: collect the oldest first
            t, kk = tickets.pop(0)
            bufs[kk % 2], _ = sc.collect(t)   # the arrays come back and are reused by a later frame
            results[kk] = {n: a.copy() for n, a in bufs[kk % 2].items()}
        sc.set_instance_poses(poses_for(sc, W, k))
        t = sc.submit(cam, cams, outputs, out=bufs[k % 2])
        tickets.append((t, k))
        if k % 2 == 0:  # nothing may leak from a later pose change into a frame already submitted
            sc.set_instance_poses(poses_for(sc, W, 100 + k))
    while tickets:
        t, kk = tickets.pop(0)
        bufs[kk % 2], _ = sc.collect(t)
        results[kk] = {n: a.copy() for n, a in bufs[kk % 2].items()}
    same = all(np.array_equal(results[k][n], sync[k][n]) for k in range(frames) for n in outputs)
    check(len(bufs[0]) == 3 and len(bufs[1]) == 3, "output arrays were reused across frames")
    check(same, f"{frames} pipelined frames are bit-identical to synchronous frames (all outputs)")

    # Misuse.
    sc.set_instance_poses(poses_for(sc, W, 0))
    a = sc.submit(cam, cams, outputs, out={})
    b = sc.submit(cam, cams, outputs, out={})
    try:
        sc.submit(cam, cams, outputs, out={})
        check(False, "a third frame in flight is refused")
    except RuntimeError as e:
        check("collect" in str(e), "a third frame in flight is refused (" + str(e) + ")")
    try:
        sc.render(cam, cams, outputs)
        check(False, "synchronous render is refused while frames are in flight")
    except RuntimeError as e:
        check("in flight" in str(e), "synchronous render is refused while frames are in flight")
    sc.collect(a)
    try:
        sc.collect(a)
        check(False, "collecting a ticket twice fails")
    except RuntimeError:
        check(True, "collecting a ticket twice fails")
    sc.collect(b)
    out, _ = sc.render(cam, cams, outputs)
    check(np.array_equal(out["depth"], sync[0]["depth"]), "synchronous rendering works again after collecting")
    # A scene destroyed with a frame in flight waits for it instead of crashing.
    sc.submit(cam, cams, outputs, out={})
    sc.close()
    check(True, "closing a scene with a frame in flight does not crash")
    print(f"   backend: {args.backend}")
    if FAILS:
        print(f"{len(FAILS)} check(s) failed")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
