# plugin: Isaac Lab

Batched-camera rendering for reinforcement-learning environments, on top of `core/`.

- `src/scene/`: the scene layer (meshes, instances, one pinhole camera per world, vertex stage, output assembly), `src/capi/` and
  `include/ttraster/ttraster_scene.h`: its C interface (`libttraster_scene.so`).
- `python/ttraster/`: ctypes binding and the Newton bridge.
- `isaaclab_ttraster/`: the Isaac Lab renderer package (`TtRasterRendererCfg`, `async_rendering`, segmentation); see its README.
- `tests/`: C++ tests of the scene layer; `validation/`: Python tests against Newton Warp and Isaac Lab stand-ins, benchmarks.

Build with `scripts/build.sh` (the plugin is part of the CMake project, `-DTT_RASTER_BUILD_PLUGINS=OFF` leaves it out); test with
`scripts/test.sh` and `scripts/test_python.sh`.
