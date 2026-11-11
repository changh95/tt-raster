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

## Vulkan on Tensix: the llvmpipe offload (done, measured)

Order 3 of the plan below is built: `api/gl/llvmpipe_tensix/` (patch `api/gl/patches/0002-llvmpipe-tensix-hooks.patch`, files copied by `scripts/build_mesa.sh`) hooks llvmpipe's set-up stage, so it serves OpenGL on llvmpipe and, because lavapipe is built on llvmpipe, Vulkan. Enabled by `TTR_LP=tensix|reference` (`scripts/run_vk.sh --backend tensix`, `scripts/run_gl.sh --driver llvmpipe --backend tensix`). The design is the softpipe fast path of `api/gl` moved to the right seam:

- A full colour+depth clear (`lp_setup_clear`; lavapipe's render pass clear arrives there too) opens a frame; llvmpipe does not clear, the rasterizer frame starts at the first accepted triangle. A draw that finds no frame open (the clear was a `vkCmdClearColorImage` before a `loadOp = LOAD` render pass, as in the Vulkan CTS, or a partial clear) opens one if the surfaces hold a single value everywhere: llvmpipe's own pending work is finished first, the surfaces are scanned (stopping at the first pixel that differs), and that value becomes the frame's clear colour and depth, converted back exactly (a depth that would not survive the round trip is refused). A framebuffer without a depth buffer is fine (no depth test, no depth output), and so is a depth bias of zero, which Vulkan applications switch on all the time. llvmpipe's linear rasterizer, which draws axis-aligned rectangles and takes unclipped vertices without passing them through the triangle hook, is switched off while the offload is enabled (it is only used for framebuffers without depth, so the offload never saw those draws before).
- Triangles arrive from `retry_triangle_ccw`, after clipping, viewport transform and culling, as float window coordinates with 1/w (the same data softpipe's hook sees), and go to `libttraster`'s frame API per draw with the draw's depth and blend state.
- The fragment shader is llvmpipe's NIR: cloned, its extra colour outputs removed (llvmpipe lowers `gl_FragColor` to eight outputs), converted with `nir_to_tgsi` and then through the existing TGSI translator to the shader instruction set (and, with `TTR_SHADER_COMPILE=1`, to compiled SFPI). A pass-through shader needs no program. Uniforms are read from constant buffer 0 at every draw.
- The frame is rendered when llvmpipe flushes (a queue submit, a swap, a read-back) or when a draw the offload cannot do arrives (textures, images, storage buffers, stencil, alpha test, other blend factors, lines and points, scissor smaller than the framebuffer, an active query, `discard`, geometry/tessellation shaders): then the frame so far goes into the surfaces first and llvmpipe draws the rest, so the result is always right. The result is written straight into the colour surface (RGBA/BGRA, converted on the way) and the depth surface (Z24, Z32, Z32F, Z16) of llvmpipe.
- Lavapipe rebinds the same framebuffer before every draw; the hook keeps the frame open when the surfaces are unchanged.

`scripts/test_lp.sh` runs 11 OpenGL scenes and 3 Vulkan scenes on llvmpipe/lavapipe alone and with the offload (the CPU backend of the renderer, and a chip with compiled shaders) and compares the images: at most 0.2 % of the pixels differ by more than 4 levels (edge pixels; 0.5 % for 3,000 tiny Vulkan triangles); a scene the offload refuses is identical to llvmpipe. All pass on both backends.

Vulkan, 1920 x 1080, `api/vulkan/tests/vk_offscreen`, full-screen quads with a 131-instruction fragment shader (64 steps of `fract(x * 0.5 + y * 0.3 + 0.123)`, chosen so that rounding differences between drivers do not grow; images identical to lavapipe's), depth test off, median of three runs of 20 frames, ms per frame:

| Layers | lavapipe, 64 threads | lavapipe + one chip, compiled shaders |
| --- | --- | --- |
| 4 | 9.8 | 12.5 |
| 8 | 18.1 | 17.5 |
| 16 | 34.8 | 27.4 |
| 32 | 67.4 | 47.9 |

The crossover is at about 8 layers (about 1,000 shader instructions per pixel), as in OpenGL; at 32 layers the chip is 1.4x faster than 64 CPU threads. Geometry-heavy frames lose or tie, as before: 1,000 triangles 7.1 ms on the chip against 2.2 ms, 20,000 triangles 12.6 against 8.0, 100,000 triangles about 29 against 28 (measured with the first version of the depth copy: 8.5 vs 2.0, 16.0 vs 7.9). The interpreter never wins (4 layers 35.8 ms, 16 layers 124 ms). This is the workload class where one chip is worth more than a 64-core host; it is the only one so far.

Wired since the first version of this section: `gl_FragCoord` (llvmpipe's NIR flips `y` itself with a driver constant loaded from the buffer, so the translator must not flip it again), uniforms loaded with `LOAD` from constant buffer 0, and `discard` (`DEMOTE`); `stripes`, `fog`, `cond`, `loop`, `discard` and `uniform` are in `scripts/test_lp.sh` (40 comparisons pass on the CPU backend and on a chip). Not done in this offload (each is refused and runs on the CPU path): other system values, textures (needs the software texture unit), blending beyond five factor pairs, MSAA, stencil, render areas smaller than the image, several colour attachments, and shaders that write depth, use integer operations or read memory. Since then: uniform buffers and push constants are read from every constant buffer a program names (the CTS `ubo` group passes with the offload, 8,215 cases), and presenting through an X11 swapchain is tested (`api/vulkan/tests/vk_present.c`: the window equals a copy of the last image and equals lavapipe's, on the CPU backend and on a chip). Depth is written eagerly, converted on the host thread pool into the surface's format with the colour copy (`ttr_frame_end_depth`: Vulkan frames at 1080p, 1,000 triangles 7.1 ms, 20,000 triangles 12.6 ms on the chip against 2.2 and 8.0 for 64 CPU threads); a lazy depth plane (as in `api/gl`) was tried through a resource hook and deadlocked, because llvmpipe's rasterizer threads map the depth surface of every scene (documented here instead of fixed).

## Vulkan CTS through the offload (measured)

The Khronos Vulkan CTS 1.4.7.1 (`dEQP-VK`, built from the public sources, `scripts/run_cts.sh`) was run on lavapipe alone, on lavapipe with the offload on its CPU backend, and on lavapipe with the offload on a chip (node 5). The groups below are the ones that draw: everything that renders triangles with fixed-function state and fragment shaders. Cases the driver does not support are reported as NotSupported by the CTS itself and are the same with and without the offload. "Offload" counts: how many draws the offload took is in the debug output (`TTR_LP_DEBUG=1`), not in the CTS result.

| Group | Pass, lavapipe alone | Pass, offload (CPU backend) | Pass, offload (chip) | Not supported |
| --- | --- | --- | --- | --- |
| `draw.renderpass.*` | 10,760 | 10,756 | 10,756 | 1,706 |
| `draw.dynamic_rendering.*` | 13,910 | 13,910 | 13,910 | 3,099 |
| `rasterization.*` | 7,398 | 7,398 | 7,398 | 7,621 |
| `fragment_operations.*` | 116 | 116 | 116 | 35 |
| `dynamic_state.*` | 560 | 560 | 560 | 111 |
| `clipping.*`, `depth.*` | 308 | 308 | 308 | 8 |
| `ubo.*` | 8,215 | 8,215 | 8,215 | 5,025 |
| `renderpasses.renderpass1.*` | 16,940 | 16,940 | 16,940 | 2,858 |
| `renderpasses.renderpass2.*` | 11,858 | 11,858 | 11,858 | 20,822 |
| `renderpasses.dynamic_rendering.*` | 17,095 (32 fail) | 17,095 (the same 32 fail) | 17,095 (the same 32 fail) | 12,082 |
| `glsl.*` | 23,744 | 23,744 | 23,744 | 4,677 |

The 32 failures in `renderpasses.dynamic_rendering` (`m10_feedback_loop`, 4 samples, depth and depth-stencil formats) are lavapipe's own: the same 32 fail without the offload, and 4-sample framebuffers are not offloaded. The only cases that differ from lavapipe alone are the four `draw.renderpass.output_location` cases (`array.b8g8r8a8-unorm-highp`, its `-output-vec2` and `-output-vec3` variants, and `shuffle.inputs-outputs-mod`), on the CPU backend and on the chip alike. They draw the same smooth-coloured triangle strip once with the offload (one attachment) and once with llvmpipe (three attachments) and compare the images bit for bit. On such a strip (`vk_offscreen --strip`) 584 of 10,800 colour values differ by one level, 575 of them upwards on the offload's side: across 60 pixels the colour goes through exact x.5 values, and llvmpipe's float32 plane evaluation lands just below them where the renderer rounds up. Making them agree would mean evaluating llvmpipe's plane equations bit for bit on the device; it is one level in 5 % of the values, invisible, and not done.

What the CTS found, fixed here (each is now in the tree and in `scripts/test_lp.sh` or the CTS runs above):

- **No frame was ever opened.** CTS draw tests clear with `vkCmdClearColorImage` and render with `loadOp = LOAD`, which never reaches `lp_setup_clear`: the offload processed 0 frames. A draw that finds no frame open now opens one on the surfaces' content if that is a single value (see the section above).
- **llvmpipe's linear rasterizer bypassed the offload.** For framebuffers without a depth buffer llvmpipe draws axis-aligned quads as rectangles and takes unclipped vertices; neither passes the triangle hook. Switched off while the offload is on. Several scissor, large-viewport, strip and vertex-attribute-divisor cases failed because of it (86 of 10,760 in `draw.renderpass`).
- **Rasterization precision.** The library snaps vertices to a 1/16 pixel grid (4 subpixel bits, so that edge functions are exact in FP32 on the device); llvmpipe snapped to 1/256 and reported 8 bits. Reporting 8 failed 12 triangle cases in `rasterization.*` with the offload (edge pixels), reporting 4 while llvmpipe snapped to 8 failed 18 `late_fragment_tests` cases, and two scissor cases compared a render of each path. With llvmpipe itself set to 4 bits (`FIXED_ORDER 4`, which is also the minimum of Vulkan and OpenGL) and reporting 4, everything above agrees. This build's llvmpipe is therefore a 4-bit one with or without the offload.
- **Silently wrong: depth written by a shader.** The conversion to TGSI kept colour output 0 only, so `gl_FragDepth` writes vanished (two `renderpass1.low_resolution_z` cases). Shaders that write depth, stencil or the sample mask are refused.
- **Crashes on shapes the offload had not seen.** An input attachment (a bindless image load in NIR) crashed `nir_to_tgsi`: shaders are screened before the conversion. A frame without a depth buffer was cleared with the depth flag.
- **Several devices in one process.** The CTS creates a second Vulkan device inside a test; the second open of the chip in one process aborts in tt-metal and left the first renderer hanging. A per-device lock now refuses the second renderer with a message and that device runs on llvmpipe alone (`core/src/tensix/chip_lock.h`).

Reasons draws stayed on the CPU path in these runs (debug counters; not a conformance matter): primitives that are not triangles (lines and points: 1,277 draws in `draw.renderpass`), geometry and tessellation shaders, textures (all of `glsl.texture_functions`), layer and viewport-index varyings, integer and bitwise shader operations, scissor smaller than the framebuffer, and stencil.

## Texture unit: the kill-gate number (spike F, measured)

The plan's workstream 2 (a software texture unit: data-movement cores fetch texels, the SFPU filters) has a kill gate in `docs/FULL_COVERAGE_PLAN.md`: "textured frames worth offloading", decided by texel-rate numbers. `core/spikes/f_texel` measures what a data-movement RISC-V core of the chip can do per pixel with a 2D RGBA8 texture and an affine mapping of about one texel per pixel (the best case for caching; 12 x 10 = 120 worker cores, 1.35 GHz, one chip):

| What one core does per pixel | cycles | per core | 120 cores |
| --- | --- | --- | --- |
| texture in L1 (256 x 256, 256 KB), nearest: 1 load | 18 | 75 Mpixel/s | 9.0 Gpixel/s |
| L1, gather of the 4 bilinear texels and the 2 weights into an L1 ring (for the SFPU to filter) | 75 | 18 Mpixel/s | 2.1 Gpixel/s |
| L1, bilinear filtered on the RISC-V (3 packed RGBA8 lerps) | 91 | 15 Mpixel/s | 1.8 Gpixel/s |
| DRAM texture (2048 x 2048), one 64-byte NoC read per pixel, 32 in flight | 53 (1 core) | 25 Mpixel/s | 0.53 Gpixel/s (305 cycles each with all cores: the NoC and DRAM are the limit) |
| DRAM, four reads per pixel (footprint crosses a page corner) | 147 (1 core) | 9 Mpixel/s | 0.12 Gpixel/s |

Against it, llvmpipe on the same host (`gl_bench --layers 8 --tex 256`, 1920 x 1080, bilinear, one texel per pixel, GL_REPEAT; the numbers include the frame's clear and read-back): 64 threads 3.0 Gpixel/s, 8 threads 1.1 to 1.2, one thread 0.2 (256 x 256 and 1024 x 1024 textures alike).

Reading: a loads-and-stores RISC-V core spends about 8 cycles on every L1 access, so the gather alone costs 75 cycles a pixel and a chip tops out near 2 Gpixel/s with the whole texture in every core's L1, before the SFPU has filtered anything or the shader has run. That is about two thirds of 64 llvmpipe threads, 2 times 8 threads, and the DRAM-fed variant (textures larger than an L1, or any texture that is not resident) is 6 to 25 times slower than the CPU. **The gate is not met on this host: textured frames stay on the CPU path** (what the offload does today: it refuses them). It would flip only with several chips on one frame (a tray of eight is about 17 Gpixel/s in the best case, if the other stages scale, which they do not yet), a host with far fewer cores, or a faster texel path in silicon (the plan's own fallback: "raise texture units as a silicon request").

## Against a GPU

An entry-level discrete GPU is in the 5 to 10 TFLOP/s FP32 class (spec sheets), that is 2.5 to 5 T fused lane-operations per second: a compiled Tensix chip (1.9 T on simple operations, less on shaders with `floor`, `fract` or transcendentals) is within a factor of two to ten of that for pure ALU work. Everything else is where a GPU is far ahead, and no kernel work changes it:

| Feature | GPU | Tensix here |
| --- | --- | --- |
| Texture sampling | dedicated units | none; the SFPU cannot gather per-lane addresses; the fetch rate of a software unit is measured below (spike F): below 64-thread llvmpipe on this host even in the best case, so texturing stays on the CPU |
| Rasterizer, ROPs, blending, depth | fixed function | software: about 230 M triangles per second per chip for tiny triangles (324,000 in 1.4 ms), blending in the SFPU |
| Frame latency | well under 1 ms | 2 to 5 ms fixed (host stages, upload, read-back over PCIe); not hidden for presenting windows (see PERF.md, overlap rejected) |
| Compute, memory model, atomics | native | needs the plan's workstream 4; none exists |
| MSAA, stencil, queries, fp64, geometry and tessellation | native | none today |
| Ray tracing | hardware on some | not applicable |

## Verdict by requirement

| Requirement | Reachable | Route |
| --- | --- | --- |
| Vulkan 1.3/1.4 conformance | yes, inherited | lavapipe on the CPU; Tensix frames must equal its output (`scripts/test_lp.sh` compares them) |
| OpenGL 4.5/4.6 conformance | yes, inherited | llvmpipe on the CPU (or Zink on lavapipe) |
| Tensix executes whole frames of untextured geometry with ALU shaders | yes, partly done | OpenGL subset exists on softpipe; same subset on llvmpipe's seam for both APIs |
| Beats llvmpipe at 64 threads | **measured:** shader-heavy untextured frames (from about 1,000 shader instructions per pixel at 1080p), with the compiler of this change; more chips would lower it | shader compiler, first version done (`shader_codegen.cpp`), NIR front end and better register handling to do |
| Textured frames faster than the CPU | unproven; needs the software texture unit and then a measurement; the plan itself marks it a kill gate | workstream 2 |
| Compute shaders on Tensix | long-term | workstream 4 |
| GPU-class performance on general workloads | no | structural |

## What to build, in this order

1. **Keep the conformant baseline integrated** (done here): one Mesa with llvmpipe, lavapipe and our `tensix` driver, scripts and a test that checks the baseline.
2. **Shader compiler** (the decisive piece). Done: ISA program to SFPI, bit-identical, cached, with fallback (above). Left: compile without stalling the first frame (see above: not possible beside a render with this tt-metal; needs a warm-up or an offline pre-warm), a NIR front end so that lavapipe and llvmpipe shaders reach the same instruction set (today the OpenGL fast path translates TGSI), a size model calibrated on kernel ELFs. Gate (met for the case measured above): a workload beats llvmpipe at 64 threads on this host.
3. **Frame-level offload at llvmpipe's seam** (`lp_setup`: post-clip triangles with float vertices, the fragment shader's NIR through `nir_to_tgsi` and the existing translator). Done (section above); the gate (a workload beats llvmpipe on the same host) is met for shader-heavy untextured frames in both OpenGL and Vulkan.
4. **Texture unit**, **compute**, **device server** (several processes at once), **CTS farm**: as in `docs/FULL_COVERAGE_PLAN.md`, in that order, each behind its own gate.

Until step 2 passes its gate there is no reason to promise GPU-like results. What can be promised today: complete API coverage on the CPU, a Tensix path for the OpenGL subset, and a measured route to making Tensix worth using for ALU-heavy untextured work.

## Caveats

- The CTS was run for the drawing groups listed above, not in full (compute, ray tracing, mesh shaders, video, sparse resources, WSI and the rest were not, and the OpenGL CTS was not run). No submission was made. Conformance statements are Mesa upstream's, and this build reports `conformanceVersion = 1.3.1.1`, but its llvmpipe now snaps vertices to 4 subpixel bits instead of 8, so it is not the configuration upstream conformance was granted for.
- One host, one chip on an x8 link, a 64-thread CPU. On a host with fewer cores the chip wins sooner; with a faster CPU later. llvmpipe numbers use `LP_NUM_THREADS`; glxgears numbers include the virtual X server and are noisy (the 4-thread value is below the 2-thread value in repeated runs).
- The compiled-shader numbers come from a benchmark kernel, not from a general compiler; the ratio for real shaders depends on register pressure and on how many operations are transcendental (those cost the same SFPU sequences in both).
- The break-even is an estimate. The fixed part of a frame (4.5 ms at 1080p) would have to shrink for small frames; nothing here shows it can fall below about 2 ms.
