# SPDX-License-Identifier: Apache-2.0
"""integrations/isaaclab_ttraster against Isaac Lab stand-ins, with real Newton model data.

    venv-newton/bin/python validation/test_isaaclab_shim.py [--backend reference|tensix]

Drives the renderer the way the camera sensor does (initialize, create_render_data, set_outputs,
update_camera, render) and compares the buffers with Newton's own tiled-camera output for the same
model, state and camera. See validation/isaaclab_stubs.py for what is stubbed.
"""
import argparse
import os
import sys
import types
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import warp as wp  # noqa: E402

wp.config.quiet = True

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, os.path.join(HERE, "..", "integrations", "isaaclab_ttraster"))
import isaaclab_stubs as stubs  # noqa: E402
import test_newton_bridge as TB  # noqa: E402
from ttraster import newton_bridge as NB  # noqa: E402

stubs.install()
from isaaclab_ttraster.renderers.ttraster_renderer import TtRasterRenderer  # noqa: E402
from isaaclab_ttraster.renderers.ttraster_renderer_cfg import TtRasterRendererCfg  # noqa: E402

check = TB.check


class Proxy:
    """ProxyArray stand-in: .warp is the array."""

    def __init__(self, warp_array):
        self.warp = warp_array
        self.torch = None


class FakeProvider:
    def __init__(self, backend):
        self.backend_ = backend
        self.calls = 0

    def create_mapping(self, labels):
        return list(labels)

    def get_transforms(self, poses, mapping, count):
        self.calls += 1
        poses.transforms = self.backend_.published
        return True


class FakeBackend:
    def __init__(self, model):
        self.model = model
        self.state_0 = model.state()
        self.geometry_offsets = None
        self.published = None


class FakeSim:
    def __init__(self, model):
        self.cfg = types.SimpleNamespace(physics=None)
        self.device = "cpu"
        self.requires_usd_stage = False
        self.requires_newton_model = False
        self.backend = FakeBackend(model)
        self.provider = FakeProvider(self.backend)

    def get_or_create_backend(self, cfg):
        return self.backend

    def get_scene_data_provider(self):
        return self.provider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="reference")
    args = ap.parse_args()
    W, N, FAR = 4, 96, 20.0

    problems, checked = stubs.check_against_isaaclab_source(TtRasterRenderer)
    if checked:
        check(not problems, "renderer overrides the real BaseRenderer interface (parameters match): " + "; ".join(problems))
    else:
        print("skip  Isaac Lab sources not found (set ISAACLAB_SRC): interface not compared with the real base class")

    model = TB.build_model(W)
    body_q = TB.pose_bodies(model, W, np.random.default_rng(7))
    sim = FakeSim(model)
    sim.backend.published = wp.array(body_q, dtype=wp.transformf, device="cpu")
    cfg = TtRasterRendererCfg(backend=args.backend, depth_clipping_behavior="max")
    r = TtRasterRenderer(cfg, sim=sim)
    check(sim.requires_newton_model and not sim.requires_usd_stage, "renderer asks for the Newton model, not the USD stage")
    r.initialize()
    check(set(map(str, r.supported_output_types())) >= {"rgb", "rgba", "depth", "normals", "albedo"}, "supported outputs published")

    spawn = types.SimpleNamespace(clipping_range=(0.1, FAR), distortion=None)
    kinds = ["rgba", "rgb", "depth", "distance_to_camera", "distance_to_image_plane", "normals", "albedo"]
    cam_cfg = types.SimpleNamespace(width=N, height=N, data_types=kinds, spawn=spawn, background_color=None, isp_cfg=None)
    spec = stubs.CameraRenderSpec(cfg=cam_cfg, device="cpu", num_instances=W, camera_prim_paths=("/World/envs/env_0/cam",), view_count=W)
    rd = r.create_render_data(spec)

    bufs = {
        "rgba": wp.zeros((W, N, N, 4), dtype=wp.uint8, device="cpu"), "rgb": wp.zeros((W, N, N, 3), dtype=wp.uint8, device="cpu"),
        "depth": wp.zeros((W, N, N, 1), dtype=wp.float32, device="cpu"),
        "distance_to_camera": wp.zeros((W, N, N, 1), dtype=wp.float32, device="cpu"),
        "distance_to_image_plane": wp.zeros((W, N, N, 1), dtype=wp.float32, device="cpu"),
        "normals": wp.zeros((W, N, N, 3), dtype=wp.float32, device="cpu"), "albedo": wp.zeros((W, N, N, 4), dtype=wp.uint8, device="cpu"),
    }
    r.set_outputs(rd, {k: Proxy(v) for k, v in bufs.items()})

    cam_pos = np.array([-4.0, 0.0, 1.8], dtype=np.float32)
    pos = wp.array(np.tile(cam_pos, (W, 1)), dtype=wp.vec3f, device="cpu")
    ori = wp.array(np.tile([0, 0, 0, 1.0], (W, 1)).astype(np.float32), dtype=wp.quatf, device="cpu")
    f = 24.0 / 20.955 * N
    intr = wp.array(np.tile(np.array([[f, 0, N / 2], [0, f, N / 2], [0, 0, 1]], dtype=np.float32), (W, 1, 1)), dtype=wp.mat33f, device="cpu")
    r.update_camera(rd, Proxy(pos), Proxy(ori), Proxy(intr))
    r.render(rd)
    check(sim.provider.calls == 1, "body transforms were requested from the scene data provider")

    cam_pose = NB.camera_poses_opengl(cam_pos[None], np.array([[0, 0, 0, 1.0]]))[0]
    ref = TB.newton_render(model, body_q, cam_pos, cam_pose[3:], N, FAR)
    hit_ref = ref["shape"] != 0xFFFFFFFF

    rgba = bufs["rgba"].numpy()
    rgb3 = bufs["rgb"].numpy()
    dist = bufs["distance_to_camera"].numpy()[..., 0]
    fwd = bufs["depth"].numpy()[..., 0]
    fwd2 = bufs["distance_to_image_plane"].numpy()[..., 0]
    nrm = bufs["normals"].numpy()
    alb = bufs["albedo"].numpy()
    hit = alb[..., 3] > 0  # albedo is transparent black where nothing is hit
    check(bool(np.all((fwd < FAR) == hit)), "planar depth is below the far clip exactly where the albedo is opaque")
    both = hit & hit_ref
    check(float((hit == hit_ref).mean()) > 0.985, f"hit mask agreement with Newton {(hit == hit_ref).mean():.4f} > 0.985")
    rel = np.abs(dist - np.where(ref["depth"] < 0, 0, ref["depth"]))[both] / np.maximum(ref["depth"][both], 1e-6)
    check(float(np.median(rel)) < 2e-3, f"distance_to_camera median relative error {np.median(rel):.2e} < 2e-3")
    check(np.array_equal(fwd, fwd2), "depth and distance_to_image_plane are the same planar depth")
    check(bool(np.all(fwd[both] <= dist[both] * 1.000001)) and bool(np.all(fwd[both] > 0)), "planar depth <= ray distance on hits")
    ang = np.degrees(np.arccos(np.clip((nrm * ref["normal"]).sum(-1), -1, 1)))[both]
    check(float(ang.mean()) < 2.0, f"normals mean angle error {ang.mean():.3f} deg < 2")
    p = TB.psnr(rgba[..., :3][both], ref["color"][..., :3][both])
    check(p > 36.0, f"rgba PSNR over common hits {p:.1f} dB > 36")
    check(np.array_equal(rgb3, rgba[..., :3]), "rgb (3 channel) equals the colour channels of rgba")
    miss_rgba = rgba[~hit]
    check(miss_rgba.size > 0 and bool(np.all(miss_rgba == np.array([238, 238, 238, 255], dtype=np.uint8))),
          "pixels that hit nothing take Isaac Lab's default background colour (0xFFEEEEEE)")
    check(bool(np.all(nrm[~hit] == 0)), "normals are zero where nothing is hit")
    # "max" depth clipping: background depth becomes the far clip (a planar depth buffer from this config).
    check(bool(np.all(alb[~hit] == 0)), "albedo is transparent black where nothing is hit")
    check(bool(np.all(dist[~hit] == np.float32(FAR))) and bool(np.all(fwd[~hit] == np.float32(FAR))),
          "depth_clipping_behavior='max' fills the background depth with the far clip")

    # A second frame after the bodies moved: poses are re-read from the provider.
    body_q2 = TB.pose_bodies(model, W, np.random.default_rng(11))
    sim.backend.published = wp.array(body_q2, dtype=wp.transformf, device="cpu")
    dist_before = bufs["distance_to_camera"].numpy().copy()
    r.render(rd)
    dist2 = bufs["distance_to_camera"].numpy()[..., 0]
    hit2 = bufs["albedo"].numpy()[..., 3] > 0
    ref2 = TB.newton_render(model, body_q2, cam_pos, cam_pose[3:], N, FAR)
    both2 = hit2 & (ref2["shape"] != 0xFFFFFFFF)
    rel2 = np.abs(dist2 - np.where(ref2["depth"] < 0, 0, ref2["depth"]))[both2] / np.maximum(ref2["depth"][both2], 1e-6)
    changed = float(np.abs(dist2 - dist_before[..., 0]).max())
    check(float(np.median(rel2)) < 2e-3 and changed > 0.5,
          f"second frame follows the new body poses (median depth error {np.median(rel2):.2e}, largest change vs frame 1 {changed:.2f} m)")

    # Unsupported requests fail loudly instead of returning empty buffers.
    bad = stubs.CameraRenderSpec(cfg=types.SimpleNamespace(**{**vars(cam_cfg), "data_types": ["semantic_segmentation"]}),
                                 device="cpu", num_instances=W, camera_prim_paths=("/x",), view_count=W)
    try:
        r.create_render_data(bad)
        check(False, "segmentation request is rejected")
    except NotImplementedError:
        check(True, "segmentation request is rejected")
    dist_spawn = types.SimpleNamespace(clipping_range=(0.1, FAR), distortion=object())
    try:
        r.create_render_data(stubs.CameraRenderSpec(cfg=types.SimpleNamespace(**{**vars(cam_cfg), "spawn": dist_spawn}),
                                                    device="cpu", num_instances=W, camera_prim_paths=("/x",), view_count=W))
        check(False, "lens distortion is rejected")
    except NotImplementedError:
        check(True, "lens distortion is rejected")

    r.cleanup(rd)
    r.close()
    r.close()  # idempotent
    print(f"   backend: {args.backend}")
    if TB.FAILS:
        print(f"{len(TB.FAILS)} check(s) failed")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
