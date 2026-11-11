# SPDX-License-Identifier: Apache-2.0
"""An animated comparison on a simulated cartpole: physics by MuJoCo Warp (through Newton's SolverMuJoCo), every
frame rendered twice from the same body transforms, by Newton's Warp ray tracer (CPU) and by tt-raster (Tensix chip
or the CPU backend), with the difference of the two.

    scripts/run.sh 5 venv-newton/bin/python plugins/isaaclab/validation/cartpole_gif.py OUT.gif [--backend tensix] [--worlds 2] [--frames 120]

The camera is the Isaac Lab cartpole camera (96 x 96, offset (-5, 0, 2), 24 mm focal length, clip 0.1 to 20 m).
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import warp as wp  # noqa: E402

wp.config.quiet = True
import newton  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import scene_spec as S  # noqa: E402
import ttraster  # noqa: E402
from ttraster import newton_bridge as NB  # noqa: E402

CART_Z = 1.4
RAIL_HALF = (0.03, 1.5, 0.03)        # along y: the camera looks along +x, so the cart slides across the image
CART_HALF = (0.18, 0.30, 0.12)
GROUND = (0.42, 0.48, 0.42)
CAMERA_POS = (-4.0, 0.0, 1.4)


def build_cartpole():
    """One cartpole: a static rail, a cart on a prismatic joint along x, a pole on a revolute joint about y."""
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.density = 200.0
    # The rail is a shape on the world body: static.
    b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, CART_Z), wp.quat_identity()), hx=RAIL_HALF[0], hy=RAIL_HALF[1],
                    hz=RAIL_HALF[2], color=S.RAIL["color"], cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    cart = b.add_link(xform=wp.transform((0.0, 0.0, CART_Z), wp.quat_identity()), label="cart")
    b.add_shape_box(cart, hx=CART_HALF[0], hy=CART_HALF[1], hz=CART_HALF[2], color=S.CART["color"])
    # The pole hangs from the cart's centre; its capsule extends 0.5 m along the pole's local z.
    pole = b.add_link(xform=wp.transform((0.0, 0.0, CART_Z), wp.quat_identity()), label="pole")
    b.add_shape_capsule(pole, xform=wp.transform((0.0, 0.0, S.POLE["half_height"]), wp.quat_identity()), radius=S.POLE["radius"],
                        half_height=S.POLE["half_height"], color=S.POLE["color"])
    j_cart = b.add_joint_prismatic(parent=-1, child=cart, axis=wp.vec3(0.0, 1.0, 0.0), parent_xform=wp.transform((0.0, 0.0, CART_Z), wp.quat_identity()),
                                   child_xform=wp.transform_identity(), limit_lower=-1.0, limit_upper=1.0, limit_ke=2.0e4, limit_kd=2.0e2,
                                   damping=2.0, armature=0.1, label="slider")
    j_pole = b.add_joint_revolute(parent=cart, child=pole, axis=wp.vec3(1.0, 0.0, 0.0), parent_xform=wp.transform_identity(),
                                  child_xform=wp.transform_identity(), damping=0.05, armature=0.01, label="hinge")
    b.add_articulation([j_cart, j_pole], label="cartpole")
    for shape in range(b.shape_count):
        b.shape_flags[shape] &= ~newton.ShapeFlags.COLLIDE_SHAPES
    return b


def depth_rgb(d, far=8.0):
    """Distance along the ray as a colour ramp (near = warm, far = dark blue); misses are black."""
    t = np.clip(d / far, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    img = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    img[d <= 0] = 0
    return img


def normal_rgb(n):
    return ((np.clip(n, -1, 1) * 0.5 + 0.5) * 255).astype(np.uint8)


def build_model(worlds, device):
    one = build_cartpole()
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_ground_plane(color=GROUND)   # a global shape
    b.replicate(one, worlds, spacing=(0.0, 0.0, 0.0))
    return b.finalize(device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--backend", default="tensix")
    ap.add_argument("--chips", type=int, default=1)
    ap.add_argument("--worlds", type=int, default=2)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--mujoco-cpu", action="store_true", help="use MuJoCo's C engine instead of MuJoCo Warp")
    args = ap.parse_args()
    W, N, FAR, device = args.worlds, 96, S.CAMERA["far"], "cpu"

    model = build_model(W, device)
    q = model.joint_q.numpy().reshape(W, -1)   # per world: [cart x, pole angle]
    # Start with the pole nearly upright (unstable) and tilted differently in every world; it falls and swings, the cart reacts.
    for w in range(W):
        q[w, 0] = -0.5 * (-1) ** w
        q[w, 1] = (0.25 + 0.2 * w) * (-1) ** w
    model.joint_q.assign(q.reshape(-1))
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=args.mujoco_cpu)
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    print(f"physics: {'MuJoCo (C)' if args.mujoco_cpu else 'MuJoCo Warp'} on {device}, {W} worlds, {model.body_count} bodies", flush=True)

    # Newton's tiled camera (the reference).
    sensor = newton.sensors.SensorTiledCamera(model)
    sensor.default_render_config.max_distance = FAR
    sensor.utils.create_default_light(enable_shadows=False)
    rays = sensor.utils.compute_camera_rays_pinhole(N, N, focal_length=S.CAMERA["focal_length"],
                                                    horizontal_aperture=S.CAMERA["horizontal_aperture"],
                                                    vertical_aperture=S.CAMERA["horizontal_aperture"])
    cam = wp.array([[wp.transform(CAMERA_POS, S.CAMERA_QUAT)] * W], dtype=wp.transformf, device=device)
    color = sensor.utils.create_color_image_output(N, N)
    depth_img = sensor.utils.create_depth_image_output(N, N)
    normal_img = sensor.utils.create_normal_image_output(N, N)
    clear = newton.sensors.SensorTiledCamera.ClearData(clear_color=0xFFEEEEEE)   # Isaac Lab's background colour

    # tt-raster, fed by the same body transforms.
    scene = NB.NewtonScene(model, backend=args.backend, n_chips=args.chips)
    camera = ttraster.Camera(width=N, height=N, focal_length=S.CAMERA["focal_length"],
                             horizontal_aperture=S.CAMERA["horizontal_aperture"], vertical_aperture=S.CAMERA["horizontal_aperture"],
                             near_clip=S.CAMERA["near"], far_clip=S.CAMERA["far"], background=0xFFEEEEEE)
    cam_poses = np.tile(np.array([*CAMERA_POS, *S.CAMERA_QUAT], dtype=np.float32), (W, 1))
    out = {}

    frame_dt = 1.0 / args.fps
    sim_dt = frame_dt / args.substeps
    frames = []
    t0 = time.time()
    scale = args.scale
    mine_name = "Tensix chip" if args.backend == "tensix" else "CPU backend"
    labels = ("Newton RGB", f"tt-raster RGB", "Newton depth", "tt-raster depth", "Newton normals", "tt-raster normals")
    print(f'renderers: Newton Warp (CPU) against tt-raster ({mine_name})', flush=True)
    for f in range(args.frames):
        # Physics: MuJoCo Warp.
        for _ in range(args.substeps):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, sim_dt)
            state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        # Reference render.
        model.bvh_refit_shapes(state_0)
        sensor.update(state_0, cam, rays, color_image=color, depth_image=depth_img, normal_image=normal_img, clear_data=clear)
        ref_rgb = color.numpy().view(np.uint8).reshape(W, N, N, 4)[..., :3]
        ref_depth = np.where(depth_img.numpy().reshape(W, N, N) < 0, 0.0, depth_img.numpy().reshape(W, N, N))
        ref_normal = normal_img.numpy().reshape(W, N, N, 3)
        # tt-raster render.
        scene.update(body_q)
        res, _ = scene.render(camera, cam_poses, ("rgb", "depth", "normal"), out=out)
        out = res
        panels = [ref_rgb, res["rgb"][..., :3], depth_rgb(ref_depth), depth_rgb(res["depth"][..., 0]), normal_rgb(ref_normal),
                  normal_rgb(res["normal"])]
        rows = []
        for w in range(W):
            rows.append(np.concatenate([np.array(Image.fromarray(p[w]).resize((N * scale, N * scale), Image.NEAREST)) for p in panels], axis=1))
        canvas = Image.new("RGB", (len(panels) * N * scale, W * N * scale + 20), (24, 24, 24))
        canvas.paste(Image.fromarray(np.concatenate(rows, axis=0)), (0, 20))
        d = ImageDraw.Draw(canvas)
        for k, label in enumerate(labels):
            d.text((k * N * scale + 4, 4), label, fill=(255, 255, 255))
        frames.append(canvas)
        if f % 20 == 0:
            print(f"frame {f}/{args.frames}  ({time.time() - t0:.0f} s)  pole angles {np.round(body_q.reshape(W, -1, 7)[:, -1, 3:5], 2).tolist()}", flush=True)
    quant = [im.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in frames]
    quant[0].save(args.out, save_all=True, append_images=quant[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {args.out}: {len(quant)} frames, {os.path.getsize(args.out) / 1e6:.2f} MB")
    scene.close()


if __name__ == "__main__":
    main()
