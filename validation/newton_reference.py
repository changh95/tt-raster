# SPDX-License-Identifier: Apache-2.0
"""Render the shared test scene with Newton's Warp ray tracer (SensorTiledCamera) on the CPU and
save depth, forward depth, normals, albedo, colour and shape ids.

    venv-newton/bin/python validation/newton_reference.py OUT.npz [--worlds 8] [--size 96]
"""
import argparse
import time
import warnings

import numpy as np
import warp as wp

warnings.filterwarnings("ignore")
wp.config.quiet = True
import newton  # noqa: E402

import scene_spec as S  # noqa: E402


def build_model(worlds, device):
    builder = newton.ModelBuilder()
    shape_labels = []
    for w in range(worlds):
        builder.begin_world()
        builder.add_ground_plane(color=S.GROUND_COLOR)
        rail = builder.add_body(xform=wp.transform(S.RAIL["center"], wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(rail, hx=S.RAIL["half"][0], hy=S.RAIL["half"][1], hz=S.RAIL["half"][2], color=S.RAIL["color"])
        cart = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(cart, hx=S.CART["half"][0], hy=S.CART["half"][1], hz=S.CART["half"][2], color=S.CART["color"])
        pole = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_capsule(pole, radius=S.POLE["radius"], half_height=S.POLE["half_height"], color=S.POLE["color"])
        builder.end_world()
    return builder.finalize(device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--size", type=int, default=96)
    args = ap.parse_args()
    device = "cpu"
    W, N = args.worlds, args.size

    model = build_model(W, device)
    state = model.state()
    # Body order per world: rail, cart, pole (the ground plane has no body).
    q = state.body_q.numpy().reshape(W, 3, 7)
    for w in range(W):
        x, theta = S.world_params(w)
        q[w, 0] = (*S.RAIL["center"], 0, 0, 0, 1)
        q[w, 1] = (x, 0.0, 1.0, 0, 0, 0, 1)
        q[w, 2] = (x, 0.0, 1.0, *S.quat_y(theta))
    state.body_q.assign(q.reshape(-1, 7))
    model.bvh_refit_shapes(state)

    sensor = newton.sensors.SensorTiledCamera(model)
    sensor.default_render_config.max_distance = S.CAMERA["far"]  # Isaac Lab sets the far clip per camera
    sensor.utils.create_default_light(enable_shadows=False)
    rays = sensor.utils.compute_camera_rays_pinhole(
        N, N, focal_length=S.CAMERA["focal_length"], horizontal_aperture=S.CAMERA["horizontal_aperture"],
        vertical_aperture=S.CAMERA["horizontal_aperture"])
    cam = wp.array([[wp.transform(S.CAMERA["position"], S.CAMERA_QUAT)] * W], dtype=wp.transformf, device=device)
    color = sensor.utils.create_color_image_output(N, N)
    depth = sensor.utils.create_depth_image_output(N, N)
    fdepth = wp.zeros((W, 1, N, N), dtype=wp.float32, device=device)
    normal = sensor.utils.create_normal_image_output(N, N)
    albedo = sensor.utils.create_albedo_image_output(N, N)
    shape = sensor.utils.create_shape_index_image_output(N, N)
    kw = dict(color_image=color, depth_image=depth, forward_depth_image=fdepth, normal_image=normal,
              albedo_image=albedo, shape_index_image=shape)
    sensor.update(state, cam, rays, **kw)  # first call compiles the kernels
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        sensor.update(state, cam, rays, **kw)
    wp.synchronize()
    ms = (time.perf_counter() - t0) / reps * 1000.0

    def u8(a):
        return a.numpy().view(np.uint8).reshape(W, N, N, 4)

    np.savez_compressed(
        args.out, depth=depth.numpy().reshape(W, N, N), forward_depth=fdepth.numpy().reshape(W, N, N),
        normal=normal.numpy().reshape(W, N, N, 3), albedo=u8(albedo), color=u8(color),
        shape=shape.numpy().reshape(W, N, N).astype(np.uint32), worlds=W, size=N)
    print(f"newton warp cpu: {W} worlds of {N}x{N}, {ms:.1f} ms per frame on the CPU")


if __name__ == "__main__":
    main()
