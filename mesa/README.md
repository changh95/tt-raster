# OpenGL through Mesa: the softpipe Tensix fast path

Mesa's `softpipe` (the reference software driver of the Gallium stack) gets a fast path that hands the rasterization
of whole frames to tt-raster: a Blackhole chip, or the CPU reference backend of the same pipeline. Mesa does
everything else: the OpenGL frontend (GL 3.3 compatibility profile as built here), GLSL and fixed-function state
to shaders, vertex shading, clipping, viewport transform and culling (Gallium's `draw` module on the host).

This is not a new driver and not Vulkan. It makes a subset of OpenGL fast, and everything outside the subset runs on
softpipe, unchanged and correct.

## What takes the fast path

A draw is accepted when all of this holds, otherwise softpipe draws it (after rendering and writing back what the
fast path has collected so far; the rest of that frame then runs on softpipe):

| Area | Accepted |
| --- | --- |
| Primitives | triangles, strips, fans, quads, polygons; no lines or points, no geometry shader, no stream output |
| Vertex stage | anything (fixed function, GLSL vertex shaders): it runs on the host in Mesa's draw module |
| Fragment shader | a colour pass-through (`gl_FragColor = varying`: what fixed-function rendering and lighting in the vertex shader produce), or any straight-line GLSL program the translator can express (below): arithmetic, `gl_FragCoord`, uniforms, `discard`, conditionals, unrolled loops. Varyings are interpolated perspective-correctly, linearly or flat. Not accepted: textures, derivatives, loops with run-time bounds, integer and bit operations, writes to depth, more than one output |
| Framebuffer | one 8-bit unorm RGBA/BGRA colour buffer (not sRGB) and a depth buffer (Z24S8, Z24X8, S8Z24, X8Z24, Z32, Z32F, Z16), single sample, at most 2048 x 2048 |
| Depth | test LESS, LEQUAL or ALWAYS, or off (then nothing is tested or written); depth writes on or off per draw; no stencil, no alpha test, no depth bounds |
| Blending | equation ADD with (SRC_ALPHA, ONE_MINUS_SRC_ALPHA), (ONE, ONE), (SRC_ALPHA, ONE), (ONE, ONE_MINUS_SRC_ALPHA) or (DST_COLOR, ZERO), the same factors for RGB and alpha; RGBA colour mask. Refused: other factors, separate alpha factors, logic op, colour write masks, two-sided colour, scissor, polygon stipple/smoothing, queries |
| State changes | depth function, depth writes and blending can change between draws of a frame: every triangle carries its state to the rasterizer (draws stay in API order) |
| Frame structure | a frame starts with a full colour+depth clear; draws before it run on softpipe |

A frame is collected, rendered once when something needs the pixels (a buffer swap, `glFlush`/`glFinish`, a read-back,
a different framebuffer, a draw the fast path refuses) and written straight into the colour buffer's memory. The depth
buffer is written lazily: only when something reads it (a softpipe draw, a CPU read, a framebuffer change, a partial
clear); in the usual loop (draw, swap, clear) the clear drops it unwritten.

GLSL is available for vertex shading (any GLSL vertex shader runs on the host) and for fragment shading: Mesa compiles the
fragment shader to TGSI as usual, `softpipe_tensix/sp_tensix_shader.c` translates that to the instruction set of the
rasterizer's shader interpreter (`include/ttraster/ttraster_isa.h`, executed per pixel on the SFPU, see docs/FINDINGS.md "G.6"),
and uniforms are read into the program's constant table at every draw.

What the translator expands: every TGSI vec4 instruction per component (swizzles, write masks, negate and absolute value, saturate);
`MOV ADD MUL MAD FMA MIN MAX LRP DIV`, `DP2 DP3 DP4`, `RCP RSQ SQRT EX2 LG2 POW SIN COS`, `FRC FLR CEIL TRUNC SSG`, the compares and
`CMP UCMP`, `KILL KILL_IF` (`discard`), `IF/ELSE/ENDIF` (if-converted: both sides run and the writes are selected by a lane mask),
`gl_FragCoord` (with Mesa's origin and pixel-centre conventions), constants and uniforms, boolean to float conversion. Loops the GLSL
compiler unrolled are straight-line code and work; a loop with run-time bounds is refused, and so are textures, derivatives, indirect
addressing and integer arithmetic. A refused shader means that draw (and the rest of the frame) runs on softpipe, with a message.
A frame with arithmetic fragment shaders has an RGBA8 target on the rasterizer (colour is quantized with every write, blending reads
the quantized colour), which is also what softpipe does.

## Files

- `softpipe_tensix/sp_tensix.{c,h}`: the fast path (new files in `src/gallium/drivers/softpipe`, MIT like Mesa);
  `softpipe_tensix/sp_tensix_shader.{c,h}`: the TGSI to shader-program translator.
- `patches/0001-softpipe-tensix-fast-path.patch`: the hooks (clear, draw, triangle setup, flush, framebuffer change,
  texture map; the `tensix` driver name; the `meson.build` entry). Applies to Mesa tag `mesa-26.3.0`.
- `tests/gl_scene.c`: a deterministic fixed-function / GLSL scene with a table of fragment shaders (`--fs NAME`, `--sphere`);
  `tests/compare_ppm.py`: image comparison; `tests/shaders_gif.py`: the animation below.
- `../scripts/build_mesa.sh`, `../scripts/run_gl.sh`, `../scripts/test_gl.sh`.

The rasterizer is reached through `libttraster.so` (`include/ttraster/ttraster.h`, "Raster API": `ttr_context_create`,
`ttr_frame_begin`, `ttr_frame_triangles`, `ttr_frame_shader`, `ttr_frame_triangles_shaded`, `ttr_frame_end`), loaded with `dlopen` so Mesa's build does not depend on
tt-metal. Perspective-correct colours are interpolated by the rasterizer itself (colour * 1/w and 1/w are interpolated,
the device multiplies by a Newton-refined reciprocal; see docs/FINDINGS.md).

## Use

```
scripts/build_mesa.sh                       # clones Mesa, applies the patch, builds softpipe + GLX, installs
scripts/run_gl.sh 5 glxgears -geometry 1920x1080
scripts/run_gl.sh 5 --backend off glxgears  # same Mesa, stock softpipe, for comparison
scripts/run_gl.sh 5 --backend reference ... # the fast path on the CPU backend (no chip)
scripts/test_gl.sh                          # image comparisons (TT_METAL_HOME set: also on a chip)
```

Environment: `TTR_GL=tensix|reference` turns the fast path on (`GALLIUM_DRIVER=tensix` is an alias for tensix),
`TTR_GL_CHIPS` sets the chip count, `TTRASTER_LIB` the path of `libttraster.so`, `TTR_GL_DEBUG=1` logs every
decision and per-frame timings, `TTR_GL_DEBUG=2` also prints the TGSI of every fragment shader and the translated program.
`GL_RENDERER` reads `tensix (softpipe)` when the path is on.

## Results

`scripts/test_gl.sh` renders a scene (smooth, flat, perspective, fan, strip, GLSL, lighting, culling, depth-tested
follow-up draws after a read-back) with stock softpipe, the fast path on the CPU backend and on a chip:

- Against stock softpipe about 1.5 to 2 % of the pixels differ by one colour level (rounding) and 0.01 to 0.03 % by more
  (pixels on triangle edges, where the two rasterizers resolve the fill rule differently); depth buffers agree the same way.
- CPU backend against chip: identical except perspective colours, where 0.6 % of the pixels differ by one level.
- Fragment shaders (`--fs`: Lambert and Phong lighting per pixel, procedural stripes from `gl_FragCoord`, `discard`, nested
  conditionals, an unrolled loop of `sin`, a uniform animated per frame, `smoothstep/mix/mod/floor`, write masks, fog from
  `gl_FragCoord.z`, an alpha from the shader): all 12 are translated and drawn on the fast path, on a chip within 0.6 to 1.6 %
  one-level differences (3.8 to 5.8 % for the one with `floor` and `mod`, whose steps move by a level on a rounding) and
  0.01 to 0.1 % larger ones (edges, the steps) against stock softpipe, and identical to the CPU backend except 0.0015 % of the
  pixels by one level. A shader with a loop of run-time bounds is refused and draws the stock result exactly.
- Refused draws (other blend factors, a loop of run-time bounds) give the stock result exactly or within the same edge noise.

Speed (glxgears, Xvfb, one x8 chip, stock softpipe is single-threaded on this host): 300 x 300 575 FPS stock and
310 FPS on Tensix (a frame costs about 2 ms of fixed chip latency), 1024 x 1024 79 and 173 FPS, 1920 x 1080 31 and 126 FPS
(`docs/PERF.md`).

![glxgears, stock softpipe and Tensix, recorded side by side](../docs/images/glxgears_stock_vs_tensix.gif)

Fragment shaders on a chip (six of the test shaders on a rotating sphere, `tests/shaders_gif.py`): Phong lighting per pixel,
stripes from `gl_FragCoord`, `discard`, a loop of `sin`, conditionals, a uniform animated per frame.

![GLSL fragment shaders drawn on a Tensix chip](../docs/images/gl_fragment_shaders.gif)

A Phong-lit sphere at 1024 x 1024 takes 73 ms per frame on stock softpipe and 9.7 ms on one chip (4.6 ms of it in the
rasterizer, 2.2 ms on the device; the rest is Mesa's own work and the read-back); `docs/PERF.md`.

## Limits and what comes next

Whole-frame batching assumes the usual loop. Missing for conformance-grade OpenGL: textures (Tensix has no texture units),
stencil (the eight Dest planes are full), MSAA, lines and points, textures in fragment shaders, loops with run-time bounds in
fragment shaders, other blend factors
and separate alpha factors (a table row each), colour write masks, scissor. Vulkan is a different frontend (lavapipe) and
is not covered.
