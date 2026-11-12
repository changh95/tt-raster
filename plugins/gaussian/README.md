# plugin: Gaussian splatting

3D Gaussian Splatting (Kerbl et al., 2023) on top of `core/`: scene and camera types, EWA projection, spherical-harmonics colour, a float CPU reference renderer, the same splats drawn through any backend of the core (CPU rasterizer, Tensix chips, device server), PLY reading and writing, synthetic scenes. See `docs/GAUSSIANS.md` for the design, the tests and the measurements (the CPU reference is the fast path on this host; the chip path works and is slower and less exact).

- `src/gs/`: `gaussians.h` (types), `project.cpp`, `render.{h,cpp}`, `ply.{h,cpp}`.
- `src/capi/`, `include/ttraster/ttraster_gs.h`: the C interface (`libttraster_gaussian.so`); `python/ttraster_gs/`: ctypes binding.
- `tools/ttr_gsplat.cpp`: `ttr_gsplat --demo 200000 --size 960x540 --backend tensix --out /tmp/gs` (or `--ply model.ply`, `--synth N`).
- `tests/`: `test_gaussian.cpp` (in `ctest`), `test_gaussian.py`.

Build with `scripts/build.sh` (part of the CMake project; `-DTT_RASTER_BUILD_PLUGINS=OFF` leaves it out).
