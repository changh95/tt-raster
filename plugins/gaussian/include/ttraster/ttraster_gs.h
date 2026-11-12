/* SPDX-License-Identifier: Apache-2.0
 *
 * libttraster_gaussian: the C interface of the 3D Gaussian Splatting plugin. A scene is what a trained model is (positions, scales,
 * rotations, opacities, spherical-harmonics colour); a renderer turns a scene and a camera into an RGBA8 image, either with the float CPU
 * reference ("cpu") or through a backend of the renderer core ("reference": the core's CPU rasterizer, "tensix": a chip or several,
 * "remote": a device server, see TTR_SERVER). Conventions: gs/gaussians.h. Every function returns a negative value (or NULL) on failure;
 * ttr_gs_last_error() says why.
 */
#ifndef TTRASTER_GS_H
#define TTRASTER_GS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* ttr_gs_last_error(void);

typedef struct ttr_gs_scene ttr_gs_scene;

/* Copies the arrays: xyz 3n, scale 3n (linear), rot 4n (unit quaternions w, x, y, z), opacity n in [0, 1], sh n * (degree + 1)^2 * 3
 * (coefficient k of channel c of Gaussian i at (i * K + k) * 3 + c). */
ttr_gs_scene* ttr_gs_scene_create(uint32_t sh_degree, uint64_t n, const float* xyz, const float* scale, const float* rot, const float* opacity, const float* sh);
/* A 3DGS point_cloud.ply (binary little endian; logit opacity, log scale, unnormalised rotation as trained models store them). */
ttr_gs_scene* ttr_gs_scene_load_ply(const char* path);
ttr_gs_scene* ttr_gs_scene_synthetic(uint64_t n, uint32_t seed, uint32_t sh_degree);
int ttr_gs_scene_save_ply(const ttr_gs_scene* scene, const char* path);
uint64_t ttr_gs_scene_size(const ttr_gs_scene* scene);
uint32_t ttr_gs_scene_sh_degree(const ttr_gs_scene* scene);
/* Copies the scene's arrays out (the buffers must have the sizes of ttr_gs_scene_create's). */
int ttr_gs_scene_get(const ttr_gs_scene* scene, float* xyz, float* scale, float* rot, float* opacity, float* sh);
void ttr_gs_scene_destroy(ttr_gs_scene* scene);

typedef struct {
    float view[16]; /* world to camera, row-major; the camera looks along +z, x right, y down */
    float fx, fy, cx, cy;
    uint32_t width, height;
    float znear;
} ttr_gs_camera;

/* Camera at eye looking at target (up is the world's up), horizontal field of view in degrees. */
int ttr_gs_look_at(ttr_gs_camera* camera, const float eye[3], const float target[3], const float up[3], float fov_x_degrees, uint32_t width, uint32_t height);

typedef struct {
    uint64_t gaussians, splats; /* in the scene, and the ones the camera sees */
    double project_ms, render_ms;
} ttr_gs_stats;

typedef struct ttr_gs_renderer ttr_gs_renderer;
/* backend: "cpu" (float reference), "reference", "tensix" or "remote". n_chips: chips of the assigned tray for "tensix". */
ttr_gs_renderer* ttr_gs_renderer_create(const char* backend, uint32_t n_chips);
void ttr_gs_renderer_destroy(ttr_gs_renderer* renderer);
/* rgba: width * height * 4 bytes, top row first; background: 3 floats in [0, 1]; stats may be NULL. The renderer backends draw targets of
 * at most 2048 x 2048 pixels; "cpu" has no such limit. */
int ttr_gs_render(ttr_gs_renderer* renderer, const ttr_gs_scene* scene, const ttr_gs_camera* camera, const float background[3], uint8_t* rgba, ttr_gs_stats* stats);

#ifdef __cplusplus
}
#endif
#endif
