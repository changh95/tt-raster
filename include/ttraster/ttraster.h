/* SPDX-License-Identifier: Apache-2.0
 *
 * libttraster: C interface for tiled-camera rendering on Tensix (or the CPU reference).
 *
 * A scene holds meshes and instances for a number of worlds (environments). One camera per world
 * is rendered per call, all through one rasterizer run. Conventions follow Isaac Lab / Newton Warp:
 * z-up world, cameras look down -z with +x right and +y up, quaternions are x, y, z, w, shape colours
 * are sRGB. Poses are 7 floats: x y z qx qy qz qw. Every function returns a negative value (or
 * NULL) on failure; ttr_last_error() describes it.
 */
#ifndef TTRASTER_H
#define TTRASTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ttr_scene ttr_scene;

typedef struct {
    uint32_t width, height;
    float focal_length, horizontal_aperture, vertical_aperture;
    float near_clip, far_clip;
    uint32_t background; /* RGBA8, R in the low byte: colour of the rgb image where nothing is hit */
} ttr_camera;

/* Output images, one per world: [world][height][width][channels]. NULL = not wanted. */
typedef struct {
    float* depth;           /* 1 channel: distance along the pixel's ray, 0 where nothing is hit */
    float* forward_depth;   /* 1: distance along the camera axis, 0 where nothing is hit */
    uint32_t* shape;        /* 1: shape id, 0xFFFFFFFF where nothing is hit */
    uint8_t* albedo;        /* 4: RGBA, sRGB */
    uint8_t* rgb;           /* 4: RGBA, shaded, sRGB */
    float* normal;          /* 3: unit world-space normal, 0 where nothing is hit */
} ttr_images;

typedef struct {
    double vertex_ms, raster_ms, assemble_ms, build_ms, total_ms;
    uint64_t triangles;
    uint32_t passes;
} ttr_stats;

/* Phase timings of the last ttr_render() call as text; valid until the next call. */
const char* ttr_backend_report(ttr_scene* scene);

/* backend: "reference" (CPU, always available) or "tensix" (needs a build with tt-metal);
 * n_chips is the number of chips for "tensix" (ignored otherwise). */
ttr_scene* ttr_scene_create(uint32_t num_worlds, const char* backend, uint32_t n_chips);
void ttr_scene_destroy(ttr_scene* scene);
const char* ttr_last_error(void);

/* Mesh ids (>= 0) are returned; the mesh winding is counter-clockwise seen from outside. */
int32_t ttr_add_box(ttr_scene* s, float hx, float hy, float hz);
int32_t ttr_add_sphere(ttr_scene* s, float radius, int rings, int segments);
int32_t ttr_add_capsule(ttr_scene* s, float radius, float half_height, int rings, int segments); /* axis +z */
int32_t ttr_add_ground(ttr_scene* s, float half_extent);                                          /* z = 0, normal +z */
int32_t ttr_add_mesh(
    ttr_scene* s, const float* positions, const float* normals, uint32_t num_vertices, const uint32_t* indices,
    uint32_t num_triangles);

/* Returns the instance id (>= 0). */
int32_t ttr_add_instance(
    ttr_scene* s, uint32_t mesh, uint32_t world, uint32_t shape_id, const float albedo_srgb[3], const float pose[7]);

/* poses: num_instances x 7, in instance id order. */
int ttr_set_instance_poses(ttr_scene* s, const float* poses);

/* camera_poses: num_worlds x 7 (camera to world). light_direction: 3 floats (the way the light
 * travels) or NULL for Newton's default (-1, 1, -1) / sqrt(3). Returns 0 on success. */
int ttr_render(
    ttr_scene* s, const ttr_camera* camera, const float* camera_poses, const float* light_direction, const ttr_images* out,
    ttr_stats* stats);

/* Pipelined rendering. ttr_submit() takes a snapshot of the instance poses, copies the camera arguments and
 * starts the frame on another thread; the caller may move poses on and submit the next frame right away. Up to
 * two frames are in flight (every ticket owns a slot until it is collected). The output buffers named in `out`
 * must stay valid and untouched until ttr_collect() of that ticket, and two frames in flight need different
 * buffers. Returns the ticket (>= 0), or -1 (e.g. no free slot: collect the oldest ticket first).
 * ttr_collect() waits for the frame, fills `stats` and returns 0, or -1 if the frame failed.
 * A synchronous ttr_render() is refused while frames are in flight. Meshes and instances must not be added
 * while frames are in flight. */
int ttr_submit(
    ttr_scene* s, const ttr_camera* camera, const float* camera_poses, const float* light_direction, const ttr_images* out);
int ttr_collect(ttr_scene* s, int ticket, ttr_stats* stats);

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
 * target and clamped again (see BlendMode in src/ttraster/frame.h): ALPHA = (SRC_ALPHA, ONE_MINUS_SRC_ALPHA),
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
/* Render the frame. rgba: width x height x 4 bytes (row pitch `stride_bytes`, 0 = width * 4); depth: width x height
 * floats or NULL. flip_y != 0 writes the bottom row of the render target first (OpenGL window orientation). */
int ttr_frame_end(ttr_context* ctx, uint8_t* rgba, uint32_t stride_bytes, float* depth, int flip_y);
/* Phase timings of the last frame as text; valid until the next call. */
const char* ttr_context_report(ttr_context* ctx);

#ifdef __cplusplus
}
#endif
#endif
