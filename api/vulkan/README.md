# Vulkan

**Status: lavapipe on the CPU, packaged and tested; no Tensix acceleration yet.** The evaluation that led to this plan, with the
measurements, is `docs/EVALUATION_GL_VULKAN.md`.

## What works today

`scripts/build_mesa.sh --llvm` builds one Mesa with llvmpipe (OpenGL 4.6 compatibility), lavapipe (Vulkan, API 1.4,
`conformanceVersion = 1.3.1.1` as it reports itself) and the softpipe `tensix` driver of `api/gl`. Run Vulkan programs with
`scripts/run_vk.sh <program>`; `scripts/test_vk.sh` checks the baseline (device, version, conformance version, vkcube,
`api/vulkan/tests/vk_offscreen` giving the same image with 1 and 8 threads, and that the `tensix` GL driver still starts).

This is the conformance baseline of `docs/FULL_COVERAGE_PLAN.md`: a frame that Tensix cannot run correctly always has the CPU path.
It was not run through the Vulkan CTS here; the statements about conformance are Mesa upstream's.

## Why the OpenGL fast path does not carry over

The fast path of `api/gl` hooks softpipe. Lavapipe cannot sit on softpipe: it creates its screen with llvmpipe regardless of
`GALLIUM_DRIVER` (`sw_screen_create_vk`), and `lvp_device.c` uses llvmpipe-only functions (texture handles for every descriptor,
`llvmpipe_get_mem_file_size`). The seam for Tensix offload under Vulkan is llvmpipe's setup and rasterizer (`lp_setup`), which lavapipe
shares with OpenGL on llvmpipe.

## Plan (each step has a gate, `docs/EVALUATION_GL_VULKAN.md`)

1. A shader compiler from NIR to SFPI. Measured: compiled code is about 40 times faster than the interpreter on a Mad chain and 6.3
   times on a Mad/Frc chain; without it one chip is worth 4 to 8 CPU threads of llvmpipe.
2. Frame-level offload at llvmpipe's seam: whole frames of untextured triangles with ALU shaders to the core's frame API
   (`core/include/ttraster/ttraster.h`), everything else on the CPU path. Gate: a target workload beats llvmpipe on the same host.
3. Texture unit, compute, a device server for several processes, a CTS farm.

## Files

- `tests/vk_offscreen.c`, `tests/tri.vert`, `tests/tri.frag`, `tests/build.sh`: a headless Vulkan test (vertex buffer, render pass with
  clear, graphics pipeline with depth test, read-back, PPM output, timing).
