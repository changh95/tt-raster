# SPDX-License-Identifier: Apache-2.0
"""NewtonScene (plugins/isaaclab/python/ttraster/newton_bridge.py) against Newton's Warp ray tracer on a model built the way
Isaac Lab builds one: global ground plane, bodies with several shape types and offsets, per-world copies.

    venv-newton/bin/python plugins/isaaclab/validation/test_newton_bridge.py [--backend reference|tensix] [--size 96]

Exit status 0 when every check passes. Newton runs on the CPU.
"""
import argparse
import math
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import warp as wp  # noqa: E402

wp.config.quiet = True
import newton  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import ttraster  # noqa: E402
from ttraster import newton_bridge as NB  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def build_model(worlds):
    b = newton.ModelBuilder()
    b.add_ground_plane(color=(0.125, 0.125, 0.15))  # outside any world: a global shape
    mesh = newton.Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
                       np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], dtype=np.int32))
    for _ in range(worlds):
        b.begin_world()
        # Box on its own body, with a shape offset and rotation relative to the body.
        a = b.add_body(xform=wp.transform((0.0, -1.0, 0.5), wp.quat_identity()), is_kinematic=True)
        b.add_shape_box(a, xform=wp.transform((0.1, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.4)),
                        hx=0.3, hy=0.2, hz=0.25, color=(0.85, 0.25, 0.20))
        # Sphere and cylinder on one body.
        s = b.add_body(xform=wp.transform((0.5, 0.0, 0.6), wp.quat_identity()), is_kinematic=True)
        b.add_shape_sphere(s, radius=0.35, color=(0.2, 0.4, 0.9))
        b.add_shape_cylinder(s, xform=wp.transform((0.0, 0.9, -0.2), wp.quat_identity()), radius=0.15, half_height=0.3,
                             color=(0.9, 0.8, 0.2))
        c = b.add_body(xform=wp.transform((1.0, 1.0, 0.5), wp.quat_identity()), is_kinematic=True)
        b.add_shape_capsule(c, radius=0.12, half_height=0.4, color=(0.25, 0.65, 0.30))
        e = b.add_body(xform=wp.transform((1.2, -0.8, 0.6), wp.quat_identity()), is_kinematic=True)
        b.add_shape_ellipsoid(e, rx=0.35, ry=0.2, rz=0.3, color=(0.7, 0.3, 0.7))
        k = b.add_body(xform=wp.transform((1.8, 0.2, 0.5), wp.quat_identity()), is_kinematic=True)
        b.add_shape_cone(k, radius=0.3, half_height=0.35, color=(0.3, 0.8, 0.8))
        m = b.add_body(xform=wp.transform((0.3, 0.9, 0.2), wp.quat_identity()), is_kinematic=True)
        b.add_shape_mesh(m, mesh=mesh, scale=(0.8, 0.8, 0.6), color=(0.9, 0.5, 0.2))
        b.end_world()
    return b.finalize(device="cpu")


def pose_bodies(model, worlds, rng):
    q = model.state().body_q.numpy().reshape(worlds, -1, 7)
    for w in range(worlds):
        for i in range(q.shape[1]):
            ang = rng.uniform(-0.8, 0.8)
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            qq = np.array([*(axis * math.sin(ang / 2)), math.cos(ang / 2)])
            q[w, i, 3:] = qq
            q[w, i, :3] += rng.uniform(-0.12, 0.12, size=3)
    return q.reshape(-1, 7)


def newton_render(model, body_q, cam_pos, cam_quat_gl, size, far):
    state = model.state()
    state.body_q.assign(body_q)
    model.bvh_refit_shapes(state)
    sensor = newton.sensors.SensorTiledCamera(model)
    sensor.default_render_config.max_distance = far
    sensor.utils.create_default_light(enable_shadows=False)
    rays = sensor.utils.compute_camera_rays_pinhole(size, size, focal_length=24.0, horizontal_aperture=20.955,
                                                    vertical_aperture=20.955)
    W = model.world_count
    cam = wp.array([[wp.transform(tuple(cam_pos), tuple(cam_quat_gl))] * W], dtype=wp.transformf, device="cpu")
    color = sensor.utils.create_color_image_output(size, size)
    depth = sensor.utils.create_depth_image_output(size, size)
    fdepth = wp.zeros((W, 1, size, size), dtype=wp.float32, device="cpu")
    normal = sensor.utils.create_normal_image_output(size, size)
    shape = sensor.utils.create_shape_index_image_output(size, size)
    sensor.update(state, cam, rays, color_image=color, depth_image=depth, forward_depth_image=fdepth, normal_image=normal,
                  shape_index_image=shape)
    return dict(color=color.numpy().view(np.uint8).reshape(W, size, size, 4), depth=depth.numpy().reshape(W, size, size),
                normal=normal.numpy().reshape(W, size, size, 3), shape=shape.numpy().reshape(W, size, size).astype(np.uint32))


def psnr(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="reference")
    ap.add_argument("--size", type=int, default=96)
    args = ap.parse_args()
    W, N, FAR = 4, args.size, 20.0

    # -- pure-math helpers ------------------------------------------------------------------------
    q = NB.camera_poses_opengl([[-5, 0, 2]], [[0, 0, 0, 1]])[0]
    ref = np.array([-5, 0, 2, 0.5, -0.5, -0.5, 0.5])
    check(np.allclose(q, ref, atol=1e-6), "world-convention identity camera -> OpenGL pose (looks along +x, z up)")
    rng = np.random.default_rng(3)
    a = rng.normal(size=(5, 4)); a /= np.linalg.norm(a, axis=1, keepdims=True)
    v = rng.normal(size=(5, 3))
    rot_a = NB.quat_rotate(a, v)
    check(np.allclose(np.linalg.norm(rot_a, axis=1), np.linalg.norm(v, axis=1), atol=1e-6), "quat_rotate keeps lengths")
    b2 = rng.normal(size=(5, 4)); b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
    check(np.allclose(NB.quat_rotate(NB.quat_mul(a, b2), v), NB.quat_rotate(a, NB.quat_rotate(b2, v)), atol=1e-6),
          "quat_mul composes rotations")

    # -- the model and the two renderers ---------------------------------------------------------------
    model = build_model(W)
    body_q = pose_bodies(model, W, np.random.default_rng(7))
    cam_pos = np.array([-4.0, 0.0, 1.8], dtype=np.float32)
    cam_q_world = np.array([0, 0, 0, 1], dtype=np.float32)
    cam_pose = NB.camera_poses_opengl(cam_pos[None], cam_q_world[None])[0]
    ref_img = newton_render(model, body_q, cam_pos, cam_pose[3:], N, FAR)

    scene = NB.NewtonScene(model, backend=args.backend)
    check(scene.skipped == [], "every shape type of the test model is mirrored")
    check(scene.num_instances == W * 7 + W * 0 + W, "instance count = per-world shapes + the global ground in every world "
          f"({scene.num_instances})")
    scene.update(body_q)
    intr = np.array([[[24.0 / 20.955 * N, 0, N / 2], [0, 24.0 / 20.955 * N, N / 2], [0, 0, 1]]] * W, dtype=np.float32)
    cam = NB.camera_from_intrinsics(N, N, intr, 0.1, FAR)
    check(abs(cam.focal_length / cam.horizontal_aperture * N - intr[0, 0, 0]) < 1e-3, "intrinsics -> focal length / aperture")
    poses = np.tile(cam_pose, (W, 1))
    out, _ = scene.render(cam, poses, ("depth", "shape", "rgb", "normal"))

    # -- compare -----------------------------------------------------------------------------------------
    hit_ref = ref_img["shape"] != 0xFFFFFFFF
    shape = out["shape"][..., 0]
    hit = shape != 0xFFFFFFFF
    agree = float((ref_img["shape"] == shape).mean())
    check(agree > 0.985, f"shape id agreement {agree:.4f} > 0.985")
    ids = np.unique(ref_img["shape"][hit_ref])
    print("   shapes seen by Newton:", ids.tolist(), " by ttraster:", np.unique(shape[hit]).tolist())
    for s in ids:
        ra, rb = ref_img["shape"] == s, shape == s
        iou = float((ra & rb).sum() / max((ra | rb).sum(), 1))
        if ra.sum() >= 20:  # slivers of a few pixels are decided by the coverage rule at the edge
            check(iou > 0.93, f"shape {int(s)} IoU {iou:.3f} > 0.93 ({int(ra.sum())} px in Newton)")
    both = hit_ref & hit
    d = np.abs(out["depth"][..., 0] - np.where(ref_img["depth"] < 0, 0, ref_img["depth"]))[both]
    rel = d / np.maximum(ref_img["depth"][both], 1e-6)
    check(float(np.median(rel)) < 2e-3, f"depth median relative error {np.median(rel):.2e} < 2e-3")
    ang_full = np.degrees(np.arccos(np.clip((out["normal"] * ref_img["normal"]).sum(-1), -1, 1)))
    ang = ang_full[both]
    per_shape = {int(s): float(ang_full[both & (ref_img["shape"] == s)].mean()) for s in ids if (both & (ref_img["shape"] == s)).sum() > 20}
    worst = sorted(per_shape.items(), key=lambda kv: -kv[1])[:4]
    print("   worst normal errors per shape [deg]:", [(k, round(v, 2)) for k, v in worst])
    check(float(ang.mean()) < 2.0, f"normals mean angle error {ang.mean():.3f} deg < 2")
    p = psnr(out["rgb"][..., :3][both], ref_img["color"][..., :3][both])
    check(p > 36.0, f"rgb PSNR over common hits {p:.1f} dB > 36")
    # Per-pixel lighting (a fragment shader on perspective-interpolated normals) against the same Newton image.
    scene.scene.set_lighting(True)
    out_pp, _ = scene.render(cam, poses, ("rgb",))
    scene.scene.set_lighting(False)
    p_pp = psnr(out_pp["rgb"][..., :3][both], ref_img["color"][..., :3][both])
    check(p_pp > 36.0 and p_pp >= p - 0.1, f"per-pixel lighting: rgb PSNR {p_pp:.1f} dB against Newton (per-vertex {p:.1f} dB)")
    print(f"   rgb PSNR over common hits: per-vertex {p:.2f} dB, per-pixel {p_pp:.2f} dB")
    # Interior pixels (same shape id in both images, and in the neighbourhood): the PSNR over the whole image is dominated by
    # edge pixels where the two renderers disagree about coverage, so look at the shading where they agree.
    same = (shape == ref_img["shape"]) & hit
    inner = same.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            inner &= np.roll(np.roll(same, dy, axis=1), dx, axis=2)
    pi_v = psnr(out["rgb"][..., :3][inner], ref_img["color"][..., :3][inner])
    pi_p = psnr(out_pp["rgb"][..., :3][inner], ref_img["color"][..., :3][inner])
    check(pi_p > pi_v and pi_p > 45.0, f"interior pixels ({int(inner.sum())}): per-pixel {pi_p:.1f} dB against per-vertex {pi_v:.1f} dB")
    if args.backend != "reference":
        # The device's transcendentals differ from the CPU interpreter's in the last bits: within one level.
        host = NB.NewtonScene(model, backend="reference")
        host.update(body_q)
        host.scene.set_lighting(True)
        out_host, _ = host.render(cam, poses, ("rgb",))
        diff = np.abs(out_pp["rgb"].astype(int) - out_host["rgb"].astype(int))
        check(int(diff.max()) <= 1 and float((diff > 0).mean()) < 0.02,
              f"per-pixel lighting on the device against the CPU interpreter: largest difference {int(diff.max())} level, {float((diff > 0).mean()):.4f} of the values differ")
        host.close()
    print(f"   backend: {args.backend}")
    scene.close()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
