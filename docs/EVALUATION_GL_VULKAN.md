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
| Beats a CPU renderer on this host | **Yes, for shader-heavy untextured frames, with compiled shaders (measured: one chip beats 64-thread llvmpipe from about 8 layers of a 131-instruction shader at 1080p, by 1.3x at 16 layers); no for geometry-heavy and small frames.** Interpreted shaders never get there. | Measured crossover below. |

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

### Measured with a first shader compiler

`core/src/tensix/shader_codegen.cpp` turns a shader program into straight SFPI (`TTR_SHADER_COMPILE=1`, or `DeviceOptions::compile_shaders`): the generated header is built into the compute kernel for the frame's set of programs, cached by content (tt-metal caches the binaries), the interpreter stays as the fallback for programs that were not compiled and for kernels that do not build. Every case is the interpreter's own code, so results are bit-identical: the random-program fuzz (`ttr_shader_fuzz`, which compares with the CPU reference) gives 0 differing pixels in exact mode (20 + 8 + 2 frames, 1.2 M + 0.3 M + 4.2 M pixels, with blending, at 1920 x 1088), smooth mode and the random GLSL fuzz are unchanged, and the `shader` golden scene has no mismatch. Registers: the compiler cannot spill a vector value (8 local registers: "too few lregs to hold live values" when whole programs were kept local, which is what the 40x experiment above did), so a lifetime analysis keeps a result in a local variable across instructions only while the live values, the operands and the temporaries of the helper the instruction expands to (estimates per operation) fit in 7 registers; everything else goes through Dest rows with compile-time addresses (level 2). Level 1 sends every result through Dest (always builds), and the renderer tries level 2, then 1, then the interpreter, remembering what failed (about 1 program set in 25 of the random programs falls from 2 to 1). One more rule found by `ttr_shader_ops`: a product that is added to something later must go through Dest, because the kernels are built with `-ffast-math` and the compiler fuses `x * y + z` held in registers into one multiply-add, which the reference (and the interpreter, whose product passes through Dest) does not. Code size is the other limit: the kernel text of a core is 70,656 bytes, the interpreter takes about 12 KB, so programs are compiled in order while an estimated 36 KB are not exceeded; the rest run interpreted.

1920 x 1080 `gl_bench`, ms per frame (shader: 2 instructions per step, 131 instructions for `--alu 64`; layers are full-screen quads, no depth test):

| Workload | Chip, interpreter | Chip, compiled (level 1: all through Dest) | Chip, compiled (level 2: allocator) | llvmpipe 1 | 4 | 16 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 layer, 34 instructions | 6.6 | 5.6 | | 11.0 | 5.6 | 1.6 | 1.0 |
| 1 layer, 131 instructions | 10.8 | 7.2 | 5.7 | 50.6 | 15.0 | 4.0 | 2.9 |
| 4 layers | 29.6 | 14.0 | 8.6 | 198.8 | 52.1 | 13.2 | 7.4 |
| 8 layers | 51.5 | 17.8 | 13.2 | | 103.5 | 28.7 | 19.1 |
| 16 layers | | 26.0 | 21.2 | | | | 34.4 |

So **one chip with compiled shaders (level 2) beats 64 llvmpipe threads from about 6 layers (about 800 shader instructions per pixel): 1.45x at 8 layers, 1.6x at 16 layers; it beats 16 threads from 4 layers**, and the interpreter never does. A layer costs about 1.0 ms on the chip (the all-local experiment above promised 0.9), against 1.5 to 1.9 ms for 64 threads. The fixed part of the frame (about 5 ms at 1080p) is what loses the one-layer cases. The estimate made before this measurement was a break-even at about five layers; it was optimistic by the register handling. First build of a new program set takes about 9 s for a 143-instruction shader (0.9 s when cached): the build is synchronous, so an application whose shaders change from frame to frame should not enable it. Building in the background while the interpreter renders was tried (`program_preparation::prepare` on a second thread, the same kernels, one core, nothing enqueued) and does not work in this tt-metal version: run beside a render in the same process it fails with "SubDeviceManagerTracker is not initialized on MeshDevice" (it worked only when the render thread was idle), so it is not in the tree. What would work: a warm-up at application start, an offline pre-warm of the cache (the cache directory is content-addressed), or tt-metal's remote compile server; none is built.

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
| Beats llvmpipe at 64 threads | **measured:** shader-heavy untextured frames (from about 1,000 shader instructions per pixel at 1080p), with the compiler of this change; more chips would lower it | shader compiler, first version done (`shader_codegen.cpp`), NIR front end and better register handling to do |
| Textured frames faster than the CPU | unproven; needs the software texture unit and then a measurement; the plan itself marks it a kill gate | workstream 2 |
| Compute shaders on Tensix | long-term | workstream 4 |
| GPU-class performance on general workloads | no | structural |

## What to build, in this order

1. **Keep the conformant baseline integrated** (done here): one Mesa with llvmpipe, lavapipe and our `tensix` driver, scripts and a test that checks the baseline.
2. **Shader compiler** (the decisive piece). Done: ISA program to SFPI, bit-identical, cached, with fallback (above). Left: compile without stalling the first frame (see above: not possible beside a render with this tt-metal; needs a warm-up or an offline pre-warm), a NIR front end so that lavapipe and llvmpipe shaders reach the same instruction set (today the OpenGL fast path translates TGSI), a size model calibrated on kernel ELFs. Gate (met for the case measured above): a workload beats llvmpipe at 64 threads on this host.
3. **Frame-level offload at llvmpipe's seam** (`lp_setup`: post-clip triangles with float vertices, the fragment shader's NIR, the state in the variant key; translate it with a NIR front end for the existing instruction set). Gate from the plan: a target workload beats llvmpipe on the same host. Until then Vulkan stays on lavapipe.
4. **Texture unit**, **compute**, **device server** (several processes at once), **CTS farm**: as in `docs/FULL_COVERAGE_PLAN.md`, in that order, each behind its own gate.

Until step 2 passes its gate there is no reason to promise GPU-like results. What can be promised today: complete API coverage on the CPU, a Tensix path for the OpenGL subset, and a measured route to making Tensix worth using for ALU-heavy untextured work.

## Caveats

- The CTS was not run (OpenGL, Vulkan, or ours). Conformance statements are Mesa upstream's, and this build reports `conformanceVersion = 1.3.1.1`; I did not verify submissions.
- One host, one chip on an x8 link, a 64-thread CPU. On a host with fewer cores the chip wins sooner; with a faster CPU later. llvmpipe numbers use `LP_NUM_THREADS`; glxgears numbers include the virtual X server and are noisy (the 4-thread value is below the 2-thread value in repeated runs).
- The compiled-shader numbers come from a benchmark kernel, not from a general compiler; the ratio for real shaders depends on register pressure and on how many operations are transcendental (those cost the same SFPU sequences in both).
- The break-even is an estimate. The fixed part of a frame (4.5 ms at 1080p) would have to shrink for small frames; nothing here shows it can fall below about 2 ms.
