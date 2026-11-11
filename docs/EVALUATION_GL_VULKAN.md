# OpenGL and Vulkan on Tensix: how far can it go?

_2026-11-11. Measured on this development host (64 hardware threads, one Blackhole chip on an x8 link, tt-metal v0.81.1, Mesa 26.3.0 with LLVM 20.1.8). Numbers marked "estimate" are projections from a measurement, and say from which one. Spec-sheet GPU figures are general knowledge, not measured here._

## The question

Two separate questions hide in "full OpenGL and Vulkan support":

1. **Can we support the whole API, 100 %?** Meaning: every command of OpenGL 4.6 and Vulkan 1.4 produces the right pixels, and ideally passes the Khronos conformance tests (CTS).
2. **Can it be as good as a GPU?** Meaning: speed and feature coverage in hardware terms (textures, MSAA, compute, ray tracing, frame latency).

## Short answers

| | Answer | Why |
| --- | --- | --- |
| 100 % API coverage, conformant | **Yes, as a hybrid: the CPU path is always correct and Tensix runs the frames it can match.** It cannot be "100 % on Tensix". | Mesa's llvmpipe (OpenGL 4.6 compatibility here, conformant for 4.5 upstream) and lavapipe (Vulkan, `conformanceVersion = 1.3.1.1`, API 1.4) already exist, build on this host and run (measured below). Tensix work then only has to be pixel-identical to them, frame by frame, with whole-frame fallback. |
| GPU-class speed and features | **No, not in general.** Only for a narrow class: untextured, shader-ALU-heavy rendering, and only with a shader compiler instead of the current interpreter. | No texture units, no ROPs, no hardware rasterizer; a fixed cost of about 2 to 5 ms per frame; today's shader interpreter is 6 to 40 times slower than the compiled code the same hardware can run (measured). |
| Beats a CPU renderer on this host | **Not today** at 64 threads; **plausible** for shader-heavy untextured frames once shaders are compiled, and already true against few CPU threads. | Measured crossover below. |

## What exists

- **OpenGL, today** (`api/gl`): Mesa softpipe with a fast path. GL 3.3 compatibility profile, a subset runs on the chip, everything else on softpipe, so the output is right (about 1.5 to 2 % of the pixels differ by one colour level from stock softpipe, 0.01 to 0.03 % by more). Not run through Khronos CTS or piglit; it is not a conformance claim.
- **Vulkan, today**: nothing of ours. What Mesa gives on the CPU is measured below.
- **New in this change**: `scripts/build_mesa.sh --llvm` builds llvmpipe + lavapipe + our softpipe `tensix` driver into one Mesa; `scripts/run_gl.sh --driver llvmpipe`, `scripts/run_vk.sh`, `scripts/test_vk.sh`, and `api/vulkan/tests/vk_offscreen` (a headless Vulkan test). `scripts/test_vk.sh` checks: lavapipe is the device, API >= 1.3, a conformance version is reported, llvmpipe offers OpenGL >= 4.5, vkcube renders through the X11 swapchain, the headless test gives the same image with 1 and 8 threads, and the `tensix` GL driver still starts from the same build.

Facts read off the build: `vulkaninfo`: `apiVersion = 1.4.363`, `driverName = llvmpipe`, `conformanceVersion = 1.3.1.1`; `glxinfo`: llvmpipe `OpenGL version string: 4.6 (Compatibility Profile)`, softpipe 3.3.

**Why Vulkan cannot simply reuse the OpenGL fast path.** Our fast path hooks softpipe. Lavapipe cannot sit on softpipe: its screen creation ignores `GALLIUM_DRIVER` and takes llvmpipe (`sw_screen_create_vk` with `sw_vk = true`), and `lvp_device.c` calls llvmpipe-only functions (`llvmpipe_create_texture_handle`, `llvmpipe_get_mem_file_size`; all descriptors go through llvmpipe texture handles). The seam for Vulkan, and for OpenGL 4.x, is therefore llvmpipe's setup and rasterizer, as `docs/FULL_COVERAGE_PLAN.md` says (option B).

## Measurements: one chip against llvmpipe on the same host

`api/gl/tests/gl_bench.c`, 1920 x 1080, one frame = clear + draws + `glFinish` + one-pixel read-back (no window copy). Chip = the softpipe `tensix` fast path on one chip (x8 link). llvmpipe = `LP_NUM_THREADS` rasterizer threads. ms per frame, lower is better.

| Workload | Chip | llvmpipe 1 | 4 | 16 | 64 |
| --- | --- | --- | --- | --- | --- |
| 301,088 small triangles, depth test | 155 | 57 | 65 | 58 | 55 |
| 1 full-screen layer, 34-instruction shader | 6.1 | 11.0 | 5.6 | 1.6 | 1.0 |
| 1 layer, 131-instruction shader | 10.8 | 50.6 | 15.0 | 4.0 | 2.9 |
| 4 layers, 131-instruction shader | 27.8 | 198.8 | 52.1 | 13.2 | 7.4 |
| 8 layers, pass-through colour (fill rate) | 8.2 | 22.0 | 12.8 | 4.1 | 2.6 |
| glxgears (FPS, window copy included, noisy) | 160 | 197 | 251 | 442 | 673 |

`vk_offscreen` on lavapipe, the same host: 1,058 triangles 7.3 ms (1 thread), 4.7 (4), 2.6 (16), 2.1 (64); 301,088 triangles 64 ms at every thread count (setup and binning are not parallel).

What this says:

- **One chip is worth about 4 to 8 CPU threads** of llvmpipe on shader-heavy or fill-heavy frames, and loses to 16 or more threads. This host has 64.
- **Geometry-bound frames lose by 3x** through the Mesa path (155 against 57 ms). The time is on the host: Mesa's vertex pipeline plus the per-triangle copy into the frame; the core's own frame path takes about 30 ms for 356,000 triangles in the scene benchmark (`docs/PERF.md`), so this is a driver cost, not the chip's.
- **Small frames lose** because of the fixed 2 to 5 ms per frame (host stages, upload, read-back).

## Why the chip is where it is: interpreter against compiled shaders

The shader interpreter issues every instruction through runtime-indexed register access on the RISC-V core (docs/PERF.md, "Interpreter cost"). To see what a shader compiler could gain, the same programs were built into the kernel as straight SFPI code (registers and constants known at compile time), `core/tools/ttr_shader_bench --compiled` (a timing experiment: `RASTER_EXP_SH_COMPILED_K` in `core/src/tensix/kernels/ckernel_sfpu_shader.h`; the result image is not compared, the operations are the interpreter's own). 1920 x 1088 quad, device time above the 0-instruction frame, per instruction and 32-lane vector on the busiest core (18 records of 32 vectors), K = 128:

| Program | Interpreter | Compiled | Ratio |
| --- | --- | --- | --- |
| dependent Mad chain | 82 ns | about 2 ns (+0.146 ms in all; noise is +-0.1 ms) | about 40x |
| Mad / Frc chain (the benchmark shader of `gl_bench`; a Frc is about 8 SFPU operations) | 75 ns per instruction (5.5 ms above the empty frame) | 12 ns per instruction (0.88 ms) | 6.3x |

(Correction: this benchmark first assumed two triangle records per tile and so printed nanosecond figures 1.9 times too small, here and in PERF.md and in earlier reports; the records per tile were measured now, 2132 for 2040 tiles. The device times in ms and all ratios were never affected.)

A compiled SFPU operation costs about 2 ns (2.7 cycles at 1.35 GHz): about 1.9 T lane-operations per second per chip for simple operations, against 0.05 T lane-instructions per second for the interpreter. On the benchmark shader the compiled chip reaches about 0.3 T lane-instructions per second, which is about 60 llvmpipe threads (one thread did 5 G per second: 50.6 ms for 2 M pixels x 131 instructions), against about 10 threads for the interpreter.

**Estimate** (from the table above and the compiled/interpreted ratio of the Mad/Frc chain): with a compiled shader the chip frame would be the 4.5 ms fixed part plus about 0.9 ms per 131-instruction layer, against 1.5 ms per layer for 64 llvmpipe threads. The break-even is then about five such layers (about 650 shader instructions per pixel at 1080p); beyond that one chip wins, and eight chips split the frame. Below it the fixed cost loses.

## Against a GPU

An entry-level discrete GPU is in the 5 to 10 TFLOP/s FP32 class (spec sheets), that is 2.5 to 5 T fused lane-operations per second: a compiled Tensix chip (1.9 T on simple operations, less on shaders with `floor`, `fract` or transcendentals) is within a factor of two to ten of that for pure ALU work. Everything else is where a GPU is far ahead, and no kernel work changes it:

| Feature | GPU | Tensix here |
| --- | --- | --- |
| Texture sampling | dedicated units | none; the SFPU cannot gather per-lane addresses; a software unit (data-movement cores fetch texels into L1, the SFPU filters) is the plan's workstream 2 and is not started |
| Rasterizer, ROPs, blending, depth | fixed function | software: about 230 M triangles per second per chip for tiny triangles (324,000 in 1.4 ms), blending in the SFPU |
| Frame latency | well under 1 ms | 2 to 5 ms fixed (host stages, upload, read-back over PCIe); not hidden for presenting windows (see PERF.md, overlap rejected) |
| Compute, memory model, atomics | native | needs the plan's workstream 4; none exists |
| MSAA, stencil, queries, fp64, geometry and tessellation | native | none today |
| Ray tracing | hardware on some | not applicable |

## Verdict by requirement

| Requirement | Reachable | Route |
| --- | --- | --- |
| Vulkan 1.3/1.4 conformance | yes, inherited | lavapipe on the CPU; Tensix frames must equal its output |
| OpenGL 4.5/4.6 conformance | yes, inherited | llvmpipe on the CPU (or Zink on lavapipe) |
| Tensix executes whole frames of untextured geometry with ALU shaders | yes, partly done | OpenGL subset exists on softpipe; same subset on llvmpipe's seam for both APIs |
| Beats llvmpipe at 64 threads | only for shader-heavy untextured frames, with a shader compiler, estimated break-even about 650 instructions per pixel; more chips lower it | NIR to SFPI compiler (workstream 1) |
| Textured frames faster than the CPU | unproven; needs the software texture unit and then a measurement; the plan itself marks it a kill gate | workstream 2 |
| Compute shaders on Tensix | long-term | workstream 4 |
| GPU-class performance on general workloads | no | structural |

## What to build, in this order

1. **Keep the conformant baseline integrated** (done here): one Mesa with llvmpipe, lavapipe and our `tensix` driver, scripts and a test that checks the baseline.
2. **Shader compiler, NIR to SFPI** (the decisive piece: 6x to 39x on measured programs; JIT build time per shader is tens of seconds, so interpret first and compile in the background, cache by shader hash). Gate: the 131-instruction case above reaches the break-even estimate, measured, not projected.
3. **Frame-level offload at llvmpipe's seam** (`lp_setup`: post-clip triangles with float vertices, the fragment shader's NIR, the state in the variant key; translate it with a NIR front end for the existing instruction set). Gate from the plan: a target workload beats llvmpipe on the same host. Until then Vulkan stays on lavapipe.
4. **Texture unit**, **compute**, **device server** (several processes at once), **CTS farm**: as in `docs/FULL_COVERAGE_PLAN.md`, in that order, each behind its own gate.

Until step 2 passes its gate there is no reason to promise GPU-like results. What can be promised today: complete API coverage on the CPU, a Tensix path for the OpenGL subset, and a measured route to making Tensix worth using for ALU-heavy untextured work.

## Caveats

- The CTS was not run (OpenGL, Vulkan, or ours). Conformance statements are Mesa upstream's, and this build reports `conformanceVersion = 1.3.1.1`; I did not verify submissions.
- One host, one chip on an x8 link, a 64-thread CPU. On a host with fewer cores the chip wins sooner; with a faster CPU later. llvmpipe numbers use `LP_NUM_THREADS`; glxgears numbers include the virtual X server and are noisy (the 4-thread value is below the 2-thread value in repeated runs).
- The compiled-shader numbers come from a benchmark kernel, not from a general compiler; the ratio for real shaders depends on register pressure and on how many operations are transcendental (those cost the same SFPU sequences in both).
- The break-even is an estimate. The fixed part of a frame (4.5 ms at 1080p) would have to shrink for small frames; nothing here shows it can fall below about 2 ms.
