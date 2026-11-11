# SPDX-License-Identifier: Apache-2.0
"""Per-step time of the Isaac Lab renderer, synchronous against asynchronous capture, with simulated other work (a sleep)
between renders. Run it once per mode (a process opens the chip once):

    TT_METAL_HOME=... scripts/run.sh 5 venv-newton/bin/python plugins/isaaclab/validation/bench_isaaclab_shim.py [--async] [--work-ms 20] [--coarse]

`render() blocks` is the time the camera sensor waits in render(); `step` includes the simulated work. --coarse tessellates the
shapes coarsely (about 350,000 triangles for 1,024 environments instead of 3.2 million). Newton on the CPU builds the model, so
the CUDA driver warning is expected.
"""
import argparse
import os
import sys
import time
import types
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "python"), os.path.join(HERE, "..", "isaaclab_ttraster")]
import warp as wp  # noqa: E402

wp.config.quiet = True
import isaaclab_stubs as stubs  # noqa: E402
import test_isaaclab_shim as T  # noqa: E402
import test_newton_bridge as TB  # noqa: E402
stubs.install()
from isaaclab_ttraster.renderers.ttraster_renderer import TtRasterRenderer  # noqa: E402
from isaaclab_ttraster.renderers.ttraster_renderer_cfg import TtRasterRendererCfg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--async", dest="asy", action="store_true")
ap.add_argument("--work-ms", type=float, default=20.0)
ap.add_argument("--envs", type=int, default=1024)
ap.add_argument("--size", type=int, default=96)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--chips", type=int, default=1)
ap.add_argument("--coarse", action="store_true")
a = ap.parse_args()
W, N = a.envs, a.size
model = TB.build_model(W)
sim = T.FakeSim(model)
states = [wp.array(TB.pose_bodies(model, W, np.random.default_rng(i)), dtype=wp.transformf, device="cpu") for i in range(4)]
sim.backend.published = states[0]
r = TtRasterRenderer(TtRasterRendererCfg(backend="tensix", num_chips=a.chips, async_rendering=a.asy, **(dict(sphere_rings=5, sphere_segments=8, capsule_rings=3, capsule_segments=8, round_segments=8) if a.coarse else {})), sim=sim)
r.initialize()
spawn = types.SimpleNamespace(clipping_range=(0.1, 20.0), distortion=None)
cam_cfg = types.SimpleNamespace(width=N, height=N, data_types=["distance_to_camera"], spawn=spawn, background_color=None, isp_cfg=None)
spec = stubs.CameraRenderSpec(cfg=cam_cfg, device="cpu", num_instances=W, camera_prim_paths=("/c",), view_count=W)
rd = r.create_render_data(spec)
buf = wp.zeros((W, N, N, 1), dtype=wp.float32, device="cpu")
r.set_outputs(rd, {"distance_to_camera": T.Proxy(buf)})
pos = wp.array(np.tile(np.array([-4.0, 0, 1.8], np.float32), (W, 1)), dtype=wp.vec3f, device="cpu")
ori = wp.array(np.tile([0, 0, 0, 1.0], (W, 1)).astype(np.float32), dtype=wp.quatf, device="cpu")
f = 24.0 / 20.955 * N
intr = wp.array(np.tile(np.array([[f, 0, N / 2], [0, f, N / 2], [0, 0, 1]], np.float32), (W, 1, 1)), dtype=wp.mat33f, device="cpu")
r.update_camera(rd, T.Proxy(pos), T.Proxy(ori), T.Proxy(intr))
t_render, t_step = [], []
for k in range(a.steps + 6):
    t0 = time.perf_counter()
    time.sleep(a.work_ms / 1e3)             # physics step and policy
    t1 = time.perf_counter()
    sim.backend.published = states[k % 4]
    r.render(rd)
    t2 = time.perf_counter()
    if k >= 6:
        t_render.append(t2 - t1)
        t_step.append(t2 - t0)
st = r.last_stats
sync_t = []
for _ in range(5):
    t0 = time.perf_counter(); r._sync_scene(); sync_t.append(time.perf_counter() - t0)
print("last stats:", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in st.items()}, "| _sync_scene %.1f ms" % (np.median(sync_t) * 1e3))
print(f"async={a.asy} envs={W} {N}x{N} work={a.work_ms} ms: render() blocks {np.median(t_render)*1e3:.1f} ms, step {np.median(t_step)*1e3:.1f} ms (median of {a.steps})")

r.close()
