/* SPDX-License-Identifier: Apache-2.0
 *
 * libttraster: the C interface of the renderer core. A caller (a graphics driver such as the Mesa and Vulkan drivers in api/,
 * or any program that has screen-space triangles) opens a context on a backend, describes a frame and gets pixels back; it
 * knows nothing about Tensix. Every function returns a negative value (or NULL) on failure; ttr_last_error() describes it.
 * Batched-camera scene rendering (Isaac Lab) is a plugin on top of this core: plugins/isaaclab/include/ttraster/ttraster_scene.h.
 */
#ifndef TTRASTER_H
#define TTRASTER_H

#include <stdint.h>
#include "ttraster_isa.h"

#ifdef __cplusplus
extern "C" {
#endif

const char* ttr_last_error(void);

/* ---- Raster API: a frame of screen-space triangles (for graphics drivers) -----------------------------------
 *
 * ttr_context owns a backend ("reference" or "tensix"). A frame is: begin (target size, clear values, depth
 * function), any number of ttr_frame_triangles() in API order, end (render everything at once and read back).
 * Window coordinates: x right, y down, pixel centres at +0.5, z in [0, 1], 1 = far. Targets are at most 2048
 * pixels wide and high. Colours are RGBA in [0, 1].
 * Everything the rasterizer does not support (blending, stencil, textures) must be filtered out by the caller.
 */
typedef struct ttr_context ttr_context;

typedef struct {
    float x, y, z, rhw;   /* window position, depth, 1/w (reserved for perspective-correct varyings) */
    float color[4];
} ttr_vertex;

enum { TTR_DEPTH_LESS = 0, TTR_DEPTH_LEQUAL = 1, TTR_DEPTH_ALWAYS = 2 /* depth test off: later triangles win */ };
enum { TTR_CULL_NONE = 0, TTR_CULL_POSITIVE_AREA = 1, TTR_CULL_NEGATIVE_AREA = 2 }; /* sign of the window-space area */
/* Colour interpolation of a batch of triangles: linear in screen space, the colour of each triangle's first vertex,
 * or perspective-correct (colour * rhw and rhw are interpolated, then divided; every vertex needs rhw > 0). */
enum { TTR_SHADE_LINEAR = 0, TTR_SHADE_FLAT = 1, TTR_SHADE_PERSPECTIVE = 2 };

ttr_context* ttr_context_create(const char* backend, uint32_t n_chips);
void ttr_context_destroy(ttr_context* ctx);

int ttr_frame_begin(
    ttr_context* ctx, uint32_t width, uint32_t height, const float clear_color[4], float clear_depth, int depth_func);
/* Depth and blend state of a batch of triangles. Blending: the colour is clamped, blended with the colour in the
 * target and clamped again (see BlendMode in core/src/ttraster/frame.h): ALPHA = (SRC_ALPHA, ONE_MINUS_SRC_ALPHA),
 * ADD = (ONE, ONE), PREMULTIPLIED = (ONE, ONE_MINUS_SRC_ALPHA), MULTIPLY = (DST_COLOR, ZERO),
 * ALPHA_ADD = (SRC_ALPHA, ONE), the same factors for
 * RGB and alpha. depth_func TTR_DEPTH_ALWAYS = depth test off. Batches with different state can follow each
 * other in one frame; triangles are drawn in submission order. */
enum {
    TTR_BLEND_NONE = 0, TTR_BLEND_ALPHA = 1, TTR_BLEND_ADD = 2, TTR_BLEND_PREMULTIPLIED = 3, TTR_BLEND_MULTIPLY = 4,
    TTR_BLEND_ALPHA_ADD = 5 /* (SRC_ALPHA, ONE) */
};
typedef struct {
    int depth_func;   /* TTR_DEPTH_* */
    int depth_write;  /* nonzero: passing fragments write depth */
    int blend;        /* TTR_BLEND_* */
} ttr_state;

/* verts: 3 per triangle. shading: TTR_SHADE_*. cull: TTR_CULL_* (one mode per frame). state: NULL = the frame's
 * depth function, depth writes on, no blending. */
int ttr_frame_triangles(
    ttr_context* ctx, const ttr_vertex* verts, uint32_t n_triangles, int shading, int cull, const ttr_state* state);
/* ---- Fragment shaders ------------------------------------------------------------------------------------------
 *
 * A fragment shader is a straight-line program (instruction set: ttraster_isa.h) with its constant table (uniforms and
 * literals). ttr_frame_shader() adds one to the open frame and returns its id (identical programs of one frame share an id; ids
 * restart at 1 with every ttr_frame_begin); 0 means refused (invalid program, or the frame's programs no longer fit the device's
 * program table), with ttr_last_error() saying why. Triangles with a shader take their colour from it: the varyings (num_inputs
 * floats per vertex, vertex-major per triangle: the n floats of vertex 0, then vertex 1, then vertex 2) are interpolated linearly
 * or, with `perspective` (every vertex needs rhw > 0), perspective-correctly, and the shader's colour goes through the same depth
 * and blend state as a fixed-function colour. A frame with shader triangles has an RGBA8 target: colour is quantized with every
 * write and blending reads the quantized colour (docs/FINDINGS.md). */
typedef struct {
    const uint32_t* code;   /* n_instr instructions of TTR_INSTR_WORDS words */
    uint32_t n_instr;
    const float* consts;
    uint32_t n_consts;
    uint32_t num_inputs;    /* varying components, in registers 0 .. num_inputs - 1 */
    uint32_t num_regs;      /* registers used (at least num_inputs, at most TTR_MAX_REGS) */
    uint32_t out_reg;       /* the colour R, G, B, A is in out_reg .. out_reg + 3 */
    int32_t fragcoord_reg;  /* >= 0: x, y (window pixel coordinates of the pixel centre), z and 1/w of the pixel in 4 registers from here */
} ttr_shader_desc;

uint32_t ttr_frame_shader(ttr_context* ctx, const ttr_shader_desc* desc);
/* verts: 3 per triangle (position, depth and rhw; the colour is not used). attrs: n_triangles * 3 * num_inputs floats.
 * cull and state as in ttr_frame_triangles. */
int ttr_frame_triangles_shaded(
    ttr_context* ctx, const ttr_vertex* verts, uint32_t n_triangles, uint32_t shader, const float* attrs, int perspective,
    int cull, const ttr_state* state);

/* Render the frame. rgba: width x height x 4 bytes (row pitch `stride_bytes`, 0 = width * 4); depth: width x height
 * floats or NULL. flip_y != 0 writes the bottom row of the render target first (OpenGL window orientation). */
int ttr_frame_end(ttr_context* ctx, uint8_t* rgba, uint32_t stride_bytes, float* depth, int flip_y);
/* ttr_frame_end() with a destination format for the colour: `color_flags` is a combination of TTR_OUT_* (0 = RGBA bytes, as
 * above). The rows are converted while they are copied, on the host thread pool, so a caller whose surface has another
 * layout (a BGRA window) needs no second pass. depth may be NULL, and then the depth plane is not read back at all. */
enum { TTR_OUT_SWAP_RB = 1 /* the word is B, G, R, A in memory order */, TTR_OUT_OPAQUE = 2 /* alpha forced to 255 */ };
int ttr_frame_end_ex(ttr_context* ctx, uint8_t* rgba, uint32_t stride_bytes, uint32_t color_flags, float* depth, int flip_y);
/* The same, with the depth plane written straight into a depth surface of a graphics driver, converted on the way (on the host thread
 * pool, like the colour). `bits_kept` formats keep the stencil bits of the words they write into: the surface is read and written. */
enum {
    TTR_DEPTH_F32 = 0,        /* 32-bit float */
    TTR_DEPTH_U32 = 1,        /* 32-bit unorm */
    TTR_DEPTH_U16 = 2,        /* 16-bit unorm */
    TTR_DEPTH_U24_LOW = 3,    /* 24-bit unorm in bits 0..23 of a 32-bit word, bits 24..31 (stencil) kept */
    TTR_DEPTH_U24_HIGH = 4,   /* 24-bit unorm in bits 8..31, bits 0..7 (stencil) kept */
    TTR_DEPTH_U24_LOW_X = 5,  /* in bits 0..23, bits 24..31 zero */
    TTR_DEPTH_U24_HIGH_X = 6  /* in bits 8..31, bits 0..7 zero */
};
typedef struct {
    uint8_t* data;
    uint32_t stride_bytes;
    int format; /* TTR_DEPTH_* */
} ttr_depth_target;
int ttr_frame_end_depth(
    ttr_context* ctx, uint8_t* rgba, uint32_t stride_bytes, uint32_t color_flags, const ttr_depth_target* depth, int flip_y);
/* The depth plane of the frame ended last (renders the frame again; the triangles stay in the context until the next
 * ttr_frame_begin). For a caller that ended the frame without depth and now needs it after all. 0 on success. */
int ttr_frame_depth(ttr_context* ctx, float* depth, int flip_y);
/* Phase timings of the last frame as text; valid until the next call. */
const char* ttr_context_report(ttr_context* ctx);

#ifdef __cplusplus
}
#endif
#endif
