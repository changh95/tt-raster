# core: the renderer

Everything needed to rasterize a frame of screen-space triangles, with no knowledge of OpenGL, Vulkan, Isaac Lab or any other consumer.

- `include/ttraster/ttraster.h`: the C frame API (`ttr_context_*`, `ttr_frame_*`; built as `libttraster.so`). A caller opens a context on a
  backend (`"reference"`, the CPU, or `"tensix"`), describes a frame (target size, clear values, triangles with their depth/blend state, fragment
  shader programs) and reads pixels back. `include/ttraster/ttraster_isa.h` is the instruction set of the fragment shader interpreter.
- `src/ttraster/`: the frame format, triangle setup, binning, the CPU reference rasterizer, the shader interpreter, `Backend` (`backend.h`).
- `src/tensix/`: the Tensix renderer (host side, kernels, multi-chip), built when `TT_METAL_HOME` is set; `shader_codegen.*` compiles shader programs
  to straight SFPI (`TTR_SHADER_COMPILE=1`, off by default: the first build of a program set takes seconds).
- `tools/`, `tests/`, `spikes/`: tools, host tests with golden images, and the Phase 0 hardware spikes.

Consumers live outside: `api/` (OpenGL, Vulkan) and `plugins/` (Isaac Lab, later others). A consumer links `libttraster.so` or, from C++,
`ttraster_core`; the core never includes anything from them.
