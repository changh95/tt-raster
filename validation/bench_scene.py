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
        out, st = sc.render(cam, cams, outputs)
        walls.append((time.perf_counter() - t0) * 1000)
        last = st
    walls = sorted(walls[1:])  # the first call compiles kernels and warms caches
    med = walls[len(walls) // 2]
    print(f"{args.backend} chips={args.chips} worlds={W} {N}x{N} outputs={args.outputs}: frame {med:.1f} ms median "
          f"({W / med * 1000:.0f} images/s)  vertex {last['vertex_ms']:.1f} raster {last['raster_ms']:.1f} "
          f"assemble {last['assemble_ms']:.1f} ms, {last['triangles']} triangles, {last['passes']} pass(es)")
    rep = sc.backend_report()
    if rep:
        print("   backend:", rep)


if __name__ == "__main__":
    main()
