# Vulkan

**Status: lavapipe on the CPU is the conformant baseline, and frames of untextured triangles with simple shaders can be offloaded to Tensix
(whole frame or nothing).** The evaluation that led to this, with the measurements, is `docs/EVALUATION_GL_VULKAN.md`.

## What works today

`scripts/build_mesa.sh --llvm` builds one Mesa with llvmpipe (OpenGL 4.6 compatibility), lavapipe (Vulkan, API 1.4,
`conformanceVersion = 1.3.1.1` as it reports itself) and the softpipe `tensix` driver of `api/gl`. Run Vulkan programs with
`scripts/run_vk.sh <program>`; `scripts/test_vk.sh` checks the baseline (device, version, conformance version, vkcube,
`api/vulkan/tests/vk_offscreen` giving the same image with 1 and 8 threads, and that the `tensix` GL driver still starts).

This is the conformance baseline of `docs/FULL_COVERAGE_PLAN.md`: a frame that Tensix cannot run correctly always has the CPU path.
It was not run through the Vulkan CTS here; the statements about conformance are Mesa upstream's.

## The Tensix offload

`api/gl/llvmpipe_tensix/lp_tensix.c` plus the hooks of `api/gl/patches/0002-llvmpipe-tensix-hooks.patch` (the same files serve OpenGL on llvmpipe).
Enable with `scripts/run_vk.sh --backend tensix [--node N]` (or `reference` for the CPU backend of the renderer): `TTR_LP`, `TTRASTER_LIB`,
`TTR_LP_DEBUG=1` logs every decision, `TTR_SHADER_COMPILE=1` compiles the shaders (`core/src/tensix/shader_codegen.*`). What it takes and what it
refuses are listed in `docs/EVALUATION_GL_VULKAN.md` ("Vulkan on Tensix"); a draw it cannot take makes the frame so far go into the surfaces and
llvmpipe draw the rest, so images are always right. `scripts/test_lp.sh` is the check (images against lavapipe alone, CPU backend and chip).
Measured at 1920 x 1080: with 1,000 or more shader instructions per pixel one chip beats 64 lavapipe threads (1.4x at 32 layers of a
131-instruction shader); geometry-heavy frames and small ones lose or tie.

## Why the OpenGL fast path of softpipe did not carry over

The softpipe fast path hooks softpipe. Lavapipe cannot sit on softpipe: it creates its screen with llvmpipe regardless of
`GALLIUM_DRIVER` (`sw_screen_create_vk`), and `lvp_device.c` uses llvmpipe-only functions (texture handles for every descriptor,
`llvmpipe_get_mem_file_size`). The offload above is the same idea on llvmpipe's set-up stage, which lavapipe shares.

## Next

1. A NIR front end of our own (today llvmpipe's NIR goes through `nir_to_tgsi`; push constants, uniform buffers and other system values are
   not wired for it), a lazy depth plane, a swapchain test (vkcube with the offload), uniform buffers and push constants.
2. The software texture unit, then compute, a device server for several processes, a CTS farm (docs/FULL_COVERAGE_PLAN.md), each behind its gate.

## Files

- `tests/vk_offscreen.c`, `tests/tri.*`, `tests/alu.*`, `tests/build.sh`: a headless Vulkan test (vertex buffer, render pass with clear, graphics
  pipelines with and without depth test, full-screen quads with a 131-instruction shader, read-back, PPM output, timing).
- `../gl/llvmpipe_tensix/`, `../gl/patches/0002-llvmpipe-tensix-hooks.patch`: the offload.
