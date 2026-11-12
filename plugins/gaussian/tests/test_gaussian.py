# SPDX-License-Identifier: Apache-2.0
"""The Python binding of the Gaussian-splatting plugin: scenes from numpy and PLY, the float reference against the backend path,
validation errors.

    python plugins/gaussian/tests/test_gaussian.py [--backend reference|tensix]      (numpy; TTRASTER_GS_LIB or the build directory)
"""
import argparse
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import ttraster_gs as gs  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def psnr(a, b):
    d = a[..., :3].astype(np.float64) - b[..., :3].astype(np.float64)
    mse = (d * d).mean()
    return float("inf") if mse == 0 else 10 * np.log10(255.0 * 255.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="reference")
    args = ap.parse_args()

    scene = gs.Scene.synthetic(20000, seed=3, sh_degree=3)
    check(len(scene) == 20000 and scene.sh_degree == 3, "a synthetic scene of 20,000 Gaussians with SH degree 3")
    xyz, scale, rot, opacity, sh = scene.arrays()
    check(sh.shape == (20000, 16, 3) and np.allclose(np.linalg.norm(rot, axis=1), 1, atol=1e-3), "arrays come back with the expected shapes, unit quaternions")

    again = gs.Scene(xyz, scale, rot, opacity, sh)
    cam = gs.Camera.look_at(eye=(0, 0.6, 4), target=(0, -0.1, 0), up=(0, 1, 0), fov_x=55, width=320, height=240)
    ref = gs.Renderer("cpu")
    a = ref.render(scene, cam, (0.05, 0.05, 0.08))
    b = ref.render(again, cam, (0.05, 0.05, 0.08))
    check(a.shape == (240, 320, 4) and np.array_equal(a, b), "a scene rebuilt from its arrays renders the same image")
    check(ref.last_stats["splats"] > 5000, f"the camera sees {ref.last_stats['splats']} splats")
    check(a[..., 3].min() == 255 and a[..., :3].std() > 5, "the image is opaque and not empty")

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.ply")
        scene.save_ply(path)
        loaded = gs.Scene.load_ply(path)
        c = ref.render(loaded, cam, (0.05, 0.05, 0.08))
        check(len(loaded) == len(scene) and psnr(a, c) > 55, f"save and load a PLY: the image is the same (PSNR {psnr(a, c):.1f} dB)")
        try:
            gs.Scene.load_ply(os.path.join(d, "missing.ply"))
            check(False, "a missing PLY is an error")
        except ValueError as e:
            check("cannot open" in str(e), "a missing PLY is an error")

    be = gs.Renderer(args.backend)
    d = be.render(scene, cam, (0.05, 0.05, 0.08))
    p = psnr(a, d)
    check(p > 40, f"the {args.backend} backend against the float reference: PSNR {p:.2f} dB")

    try:
        gs.Scene(xyz, scale * 0, rot, opacity, sh)
        check(False, "zero scales are refused")
    except ValueError as e:
        check("scale" in str(e), "zero scales are refused")
    try:
        gs.Scene(xyz, scale, rot, opacity * 0 + 2.0, sh)
        check(False, "opacity above 1 is refused")
    except ValueError as e:
        check("opacity" in str(e), "opacity above 1 is refused")

    print("PASS" if not FAILS else f"FAIL: {len(FAILS)} check(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
