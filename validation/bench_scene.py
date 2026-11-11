# SPDX-License-Identifier: Apache-2.0
"""Throughput of libttraster on the shared cartpole-like scene (rendering only, poses change every frame).

    scripts/run.sh 0,1,2,3,4,5,6,7 venv-newton/bin/python validation/bench_scene.py --worlds 256 --backend tensix --chips 8
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import scene_spec as S  # noqa: E402
import ttraster  # noqa: E402
from compare_scene import build_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--backend", default="reference")
    ap.add_argument("--chips", type=int, default=1)
    ap.add_argument("--outputs", default="depth")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--pipeline", action="store_true",
                    help="two frames in flight (submit/collect): reports the time per frame in steady state")
    args = ap.parse_args()
    W, N = args.worlds, args.size
    sc = build_scene(W, args.backend, args.chips, 4, 16)
    cam = ttraster.Camera(width=N, height=N, focal_length=S.CAMERA["focal_length"],
                          horizontal_aperture=S.CAMERA["horizontal_aperture"], vertical_aperture=S.CAMERA["horizontal_aperture"],
                          near_clip=S.CAMERA["near"], far_clip=S.CAMERA["far"])
    cams = np.tile(np.array([*S.CAMERA["position"], *S.CAMERA_QUAT], dtype=np.float32), (W, 1))
    outputs = tuple(args.outputs.split(","))
    rng = np.random.default_rng(1)
    base = np.zeros((sc.num_instances, 7), dtype=np.float32)
    base[:, 6] = 1
    walls, last = [], None
    bufs = {}  # persistent output arrays, as an RL loop would keep them
    for rep in range(args.reps + 1):
        poses = base.copy().reshape(W, 4, 7)
        for w in range(W):
            x, th = S.world_params(w + rep * 7)
            poses[w, 0] = (0, 0, 0, 0, 0, 0, 1)
            poses[w, 1] = (*S.RAIL["center"], 0, 0, 0, 1)
            poses[w, 2] = (x, 0, 1.0, 0, 0, 0, 1)
            poses[w, 3] = (x, 0, 1.0, *S.quat_y(th))
        sc.set_instance_poses(poses.reshape(-1, 7))
        t0 = time.perf_counter()
        bufs, st = sc.render(cam, cams, outputs, out=bufs)
        walls.append((time.perf_counter() - t0) * 1000)
        last = st
    walls = sorted(walls[1:])  # the first call compiles kernels and warms caches
    med = walls[len(walls) // 2]
    if args.pipeline:
        def poses_at(rep):
            poses = base.copy().reshape(W, 4, 7)
            for w in range(W):
                x, th = S.world_params(w + rep * 7)
                poses[w, 0] = (0, 0, 0, 0, 0, 0, 1)
                poses[w, 1] = (*S.RAIL["center"], 0, 0, 0, 1)
                poses[w, 2] = (x, 0, 1.0, 0, 0, 0, 1)
                poses[w, 3] = (x, 0, 1.0, *S.quat_y(th))
            return poses.reshape(-1, 7)

        frames = max(args.reps, 8) * 2
        all_poses = [poses_at(k) for k in range(frames + 2)]   # generated outside the timed loop (Python is slow at it)
        pbufs, tickets, lat = [{}, {}], [], []
        t_start = None
        for k in range(frames + 2):
            if k == 2:   # steady state: the two slots are warm
                t_start = time.perf_counter()
            if len(tickets) == 2:
                t, kk, ts = tickets.pop(0)
                pbufs[kk % 2], pst = sc.collect(t)
                lat.append((time.perf_counter() - ts) * 1000)
            if k < frames + 2:
                sc.set_instance_poses(all_poses[k])
                ts = time.perf_counter()
                tickets.append((sc.submit(cam, cams, outputs, out=pbufs[k % 2]), k, ts))
        while tickets:
            t, kk, ts = tickets.pop(0)
            pbufs[kk % 2], pst = sc.collect(t)
            lat.append((time.perf_counter() - ts) * 1000)
        per_frame = (time.perf_counter() - t_start) * 1000 / (frames)
        print("   last pipelined frame:", {k: round(v, 1) if isinstance(v, float) else v for k, v in pst.items()})
        print(f"{args.backend} chips={args.chips} worlds={W} {N}x{N} outputs={args.outputs} PIPELINED (2 in flight): "
              f"{per_frame:.1f} ms per frame ({W / per_frame * 1000:.0f} images/s), latency submit->collect {np.median(lat):.1f} ms")
        return
    print(f"{args.backend} chips={args.chips} worlds={W} {N}x{N} outputs={args.outputs}: frame {med:.1f} ms median "
          f"({W / med * 1000:.0f} images/s)  vertex {last['vertex_ms']:.1f} raster {last['raster_ms']:.1f} "
          f"assemble {last['assemble_ms']:.1f} build {last['build_ms']:.1f} ms (C++ total {last['total_ms']:.1f}), {last['triangles']} triangles, {last['passes']} pass(es)")
    rep = sc.backend_report()
    if rep:
        print("   backend:", rep)


if __name__ == "__main__":
    main()
