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

#ifdef __cplusplus
}
#endif
#endif
