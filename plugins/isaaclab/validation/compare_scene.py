# SPDX-License-Identifier: Apache-2.0
"""Render the shared test scene with libttraster and compare with the Newton Warp CPU reference.

    venv-newton/bin/python plugins/isaaclab/validation/newton_reference.py ref.npz --worlds 8 --size 96
    venv-newton/bin/python plugins/isaaclab/validation/compare_scene.py ref.npz [--backend reference|tensix] [--png OUT.png]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import scene_spec as S  # noqa: E402
import ttraster  # noqa: E402


def build_scene(worlds, backend, n_chips, capsule_rings, capsule_segments):
    sc = ttraster.Scene(worlds, backend=backend, n_chips=n_chips)
    ground = sc.add_ground(200.0)
    rail = sc.add_box(*S.RAIL["half"])
    cart = sc.add_box(*S.CART["half"])
    pole = sc.add_capsule(S.POLE["radius"], S.POLE["half_height"], capsule_rings, capsule_segments)
    for w in range(worlds):
        x, theta = S.world_params(w)
        sc.add_instance(ground, w, 0, S.GROUND_COLOR, (0, 0, 0, 0, 0, 0, 1))
        sc.add_instance(rail, w, 1, S.RAIL["color"], (*S.RAIL["center"], 0, 0, 0, 1))
        sc.add_instance(cart, w, 2, S.CART["color"], (x, 0.0, 1.0, 0, 0, 0, 1))
        sc.add_instance(pole, w, 3, S.POLE["color"], (x, 0.0, 1.0, *S.quat_y(theta)))
    return sc


def metrics(ref, out, far):
    hit_ref = ref["shape"] != 0xFFFFFFFF
    hit = out["shape"][..., 0] != 0xFFFFFFFF
    r = {}
    r["shape agreement (pixels with the same id)"] = float((ref["shape"] == out["shape"][..., 0]).mean())
    ious = []
    for s in np.unique(ref["shape"][hit_ref]):
        a, b = ref["shape"] == s, out["shape"][..., 0] == s
        ious.append((int(s), float((a & b).sum() / max((a | b).sum(), 1))))
    r["IoU per shape id"] = ious
    both = hit_ref & hit
    d_ref = np.where(ref["depth"] < 0, 0, ref["depth"])
    err = np.abs(out["depth"][..., 0] - d_ref)[both]
    r["depth: mean / 99% / max abs error over common hits [m]"] = (float(err.mean()), float(np.percentile(err, 99)), float(err.max()))
    rel = (err / np.maximum(d_ref[both], 1e-6))
    r["depth: median / 99% relative error"] = (float(np.median(rel)), float(np.percentile(rel, 99)))
    ferr = np.abs(out["forward_depth"][..., 0] - ref["forward_depth"])[both]
    r["forward depth: mean / max abs error [m]"] = (float(ferr.mean()), float(ferr.max()))
    nd = np.clip((out["normal"] * ref["normal"]).sum(-1), -1, 1)[both]
    r["normals: mean / 99% angle error [deg]"] = (float(np.degrees(np.arccos(nd)).mean()), float(np.percentile(np.degrees(np.arccos(nd)), 99)))
    for name in ("albedo", "rgb"):
        key = "albedo" if name == "albedo" else "color"
        a = ref[key][..., :3].astype(np.float64)[both]
        b = out[name][..., :3].astype(np.float64)[both]
        mse = ((a - b) ** 2).mean()
        r[f"{name}: PSNR over common hits [dB]"] = float(10 * np.log10(255.0**2 / mse)) if mse > 0 else float("inf")
        full = ((ref[key][..., :3].astype(np.float64) - out[name][..., :3].astype(np.float64)) ** 2).mean()
        r[f"{name}: PSNR over the whole image [dB]"] = float(10 * np.log10(255.0**2 / full)) if full > 0 else float("inf")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--backend", default="reference")
    ap.add_argument("--chips", type=int, default=1)
    ap.add_argument("--capsule-rings", type=int, default=8)
    ap.add_argument("--capsule-segments", type=int, default=24)
    ap.add_argument("--png")
    args = ap.parse_args()
    ref = dict(np.load(args.ref))
    W, N = int(ref["worlds"]), int(ref["size"])
    ref["depth"] = np.where(ref["depth"] < 0, 0, ref["depth"])
    # Newton numbers shapes across all worlds (4 per world here); compare per-world local ids.
    miss = ref["shape"] == 0xFFFFFFFF
    ref["shape"] = np.where(miss, 0xFFFFFFFF, ref["shape"] - 4 * np.arange(W, dtype=np.uint32)[:, None, None])
    sc = build_scene(W, args.backend, args.chips, args.capsule_rings, args.capsule_segments)
    cam = ttraster.Camera(width=N, height=N, focal_length=S.CAMERA["focal_length"],
                          horizontal_aperture=S.CAMERA["horizontal_aperture"], vertical_aperture=S.CAMERA["horizontal_aperture"],
                          near_clip=S.CAMERA["near"], far_clip=S.CAMERA["far"])
    poses = np.tile(np.array([*S.CAMERA["position"], *S.CAMERA_QUAT], dtype=np.float32), (W, 1))
    outputs = ("depth", "forward_depth", "shape", "albedo", "rgb", "normal")
    out, st = sc.render(cam, poses, outputs)
    # Newton marks shape misses with 0xFFFFFFFF and depth misses with <= 0.
    for k, v in metrics(ref, out, S.CAMERA["far"]).items():
        print(f"{k}: {v}")
    print("stats:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in st.items()})
    if args.png:
        from PIL import Image
        k = min(W, 4)
        def row(a): return np.concatenate(list(a[:k]), axis=1)
        def gray(d): return np.repeat((np.clip(d, 0, 12) / 12 * 255).astype(np.uint8)[..., None], 3, -1)
        d_err = np.abs(out["depth"][..., 0] - ref["depth"]) * (ref["shape"] != 0xFFFFFFFF)
        def hot(e): return np.stack([np.clip(e * 255 * 20, 0, 255)] * 1 + [np.zeros_like(e)] * 2, -1).astype(np.uint8)
        rows = [row(ref["color"][..., :3]), row(out["rgb"][..., :3]), row(gray(ref["depth"])), row(gray(out["depth"][..., 0])),
                row(((ref["normal"] * 0.5 + 0.5) * 255).astype(np.uint8)), row(((out["normal"] * 0.5 + 0.5) * 255).astype(np.uint8)),
                row(hot(d_err))]
        Image.fromarray(np.concatenate(rows, axis=0)).resize((k * N * 3, len(rows) * N * 3), Image.NEAREST).save(args.png)
        print("wrote", args.png)


if __name__ == "__main__":
    main()
