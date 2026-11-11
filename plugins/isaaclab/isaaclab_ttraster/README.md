# isaaclab_ttraster

Isaac Lab (3.x) tiled-camera renderer backend that rasterizes on Tenstorrent Tensix cores with
[tt-raster](../../README.md). It is a `BaseRenderer` subclass with a `RendererCfg`, built the way
`isaaclab_newton.renderers.NewtonWarpRenderer` is: the scene is the Newton model of the simulation, body
transforms come from the scene data provider each frame, and the camera outputs are written into the buffers
the camera sensor binds with `set_outputs`.

**Status: not run inside a live Isaac Lab.** Isaac Lab needs PyTorch, CUDA and USD, which the development host
does not have. What was tested (`scripts/test_python.sh`):

- the renderer drives a real `newton.Model` / `body_q` through the Isaac Lab call sequence (`initialize`,
  `create_render_data`, `set_outputs`, `update_camera`, `render`) against stand-ins for the Isaac Lab classes
  (`plugins/isaaclab/validation/isaaclab_stubs.py`), and its buffers match Newton's own tiled-camera output for the same state
  (shape id agreement 99.8 %, distance median relative error 6e-4, normals 0.23 degrees, RGB PSNR 38 dB on a scene
  with every supported shape type);
- the method names and parameter names of the real `BaseRenderer` (parsed from the Isaac Lab sources, no import) match
  the ones the renderer overrides.

What that does not prove: that the Isaac Lab classes the renderer imports (`SceneDataFormat.Transform`,
`SimulationContext.get_or_create_backend`, the config machinery, the `ProxyArray` buffers on a CUDA device) behave
as the stand-ins do. Expect small fixes on first contact.

## Use

```python
from isaaclab_ttraster.renderers import TtRasterRendererCfg

camera = TiledCameraCfg(..., renderer_cfg=TtRasterRendererCfg(backend="tensix", num_chips=1))
```

`ttraster` (the Python package in `plugins/isaaclab/python/ttraster` of this repository) and `libttraster.so` must be importable
(`PYTHONPATH`, `TTRASTER_LIB`); `backend="reference"` runs the same pipeline on the CPU without a device.

## Supported

| Isaac Lab output | Notes |
| --- | --- |
| `rgba`, `rgb` | per-vertex Newton lighting (hemispheric ambient plus one directional light), no shadows; misses take the camera's `background_color` (default 0xFFEEEEEE) |
| `albedo` | RGBA8, transparent black where nothing is hit (as Newton) |
| `distance_to_camera` | distance along the pixel ray |
| `depth`, `distance_to_image_plane` | distance along the camera axis; `depth_clipping_behavior="max"` fills misses with the far clip |
| `normals` | world-space unit normals, zero where nothing is hit |
| `semantic_segmentation`, `instance_segmentation` | the per-pixel shape index (Newton's global shape index, as `shape_index_image`) goes through Isaac Lab's own `NewtonSegmentationMapper` (`isaaclab_newton.renderers.segmentation`), exactly as in the Newton Warp renderer: colorized RGBA or int32 ids, `info["idToLabels"]`; the config fields `semantic_filter`, `colorize_semantic_segmentation`, `colorize_instance_segmentation` and `semantic_segmentation_mapping` are the Newton renderer's. **Only the plumbing is tested** (against a stand-in mapper; the real one needs the USD stage) |

Shapes: plane, sphere, capsule, box, ellipsoid, cylinder, cone, triangle meshes (flat shaded when the mesh has no
normals). One shared intrinsics matrix and a centred principal point are required. The shape id the renderer
produces internally is Newton's global shape index.

## Not supported (rejected with `NotImplementedError` where a camera asks for it)

HDR colour, lens distortion, ISP pipelines, textures, shadows, deforming
geometry (`geometry_offsets`), principal point offsets. Attribute interpolation is screen-linear and lighting
per-vertex by default; `per_pixel_lighting=True` lights per pixel on perspective-interpolated normals (about 3 times the frame time on the
device, almost no gain in image difference to Newton: `docs/PERF.md`).

## Asynchronous rendering

`TtRasterRendererCfg(async_rendering=True)` follows the delayed-capture pattern of Isaac Lab (`OVRTXRendererCfg.async_rendering`): the chip
renders while the physics step and the policy run, and every camera returns its previous capture. `prepare_capture` snapshots the pose, calibration
and frame indices of the capture being submitted, `render()` submits it and publishes the previous one, and `read_output` puts the snapshot that
belongs to the published image into `camera.data.info[output]["capture"]`. The first capture and the first one after `reset()` wait for a fresh
image (the second repeats the first, without new `capture` metadata). The price is one step of latency in the observations; the gain is that a
step costs the longer of the simulation and one frame instead of their sum (1,024 environments, 96 x 96, 356,000 triangles, x8 chip: 67 ms
down to 31 ms per step with 20 ms of other work; `docs/PERF.md`, `plugins/isaaclab/validation/bench_isaaclab_shim.py`). One camera per renderer. Checked
against the Isaac Lab stand-ins, bit for bit against synchronous renders of the same states (`plugins/isaaclab/validation/test_isaaclab_shim.py`).

## Cost

The data path reads poses to the host, rasterizes, and copies images back to the output buffers (a host to
device copy when those live on a GPU). On the x8-link chip a 1,024-environment, 96 x 96 depth frame takes about 33
ms in the renderer (`docs/PERF.md`); an Isaac Lab step adds its own copies and the GPU synchronization.
