// SPDX-License-Identifier: Apache-2.0

#include "ttraster/ttraster.h"

#include <cstring>
#include <memory>
#include <string>

#include "scene/scene.h"
#ifdef TT_RASTER_WITH_TENSIX
#include "tensix/scene_backend.h"
#endif

namespace {
thread_local std::string g_error;
}

struct ttr_scene {
    ttr::scene::Scene scene;
    std::unique_ptr<ttr::scene::Backend> backend;
    explicit ttr_scene(uint32_t worlds) : scene(worlds) {}
};

extern "C" {

const char* ttr_last_error(void) { return g_error.c_str(); }

ttr_scene* ttr_scene_create(uint32_t num_worlds, const char* backend, uint32_t n_chips) {
    try {
        auto s = std::make_unique<ttr_scene>(num_worlds);
        const std::string b = backend ? backend : "reference";
        if (b == "reference") {
            s->backend = std::make_unique<ttr::scene::ReferenceBackend>();
        } else if (b == "tensix") {
#ifdef TT_RASTER_WITH_TENSIX
            s->backend = ttr::make_tensix_backend(n_chips == 0 ? 1 : n_chips);
#else
            g_error = "this libttraster was built without Tensix support";
            return nullptr;
#endif
        } else {
            g_error = "unknown backend '" + b + "'";
            return nullptr;
        }
        (void)n_chips;
        return s.release();
    } catch (const std::exception& e) {
        g_error = e.what();
        return nullptr;
    }
}

void ttr_scene_destroy(ttr_scene* scene) { delete scene; }

#define TTR_TRY(expr)                 \
    try {                             \
        return (expr);                \
    } catch (const std::exception& e) { \
        g_error = e.what();           \
        return -1;                    \
    }

int32_t ttr_add_box(ttr_scene* s, float hx, float hy, float hz) { TTR_TRY(static_cast<int32_t>(s->scene.add_mesh(ttr::scene::make_box(hx, hy, hz)))) }
int32_t ttr_add_sphere(ttr_scene* s, float r, int rings, int segs) { TTR_TRY(static_cast<int32_t>(s->scene.add_mesh(ttr::scene::make_sphere(r, rings, segs)))) }
int32_t ttr_add_capsule(ttr_scene* s, float r, float hh, int rings, int segs) { TTR_TRY(static_cast<int32_t>(s->scene.add_mesh(ttr::scene::make_capsule(r, hh, rings, segs)))) }
int32_t ttr_add_ground(ttr_scene* s, float e) { TTR_TRY(static_cast<int32_t>(s->scene.add_mesh(ttr::scene::make_ground(e)))) }

int32_t ttr_add_mesh(
    ttr_scene* s, const float* positions, const float* normals, uint32_t nv, const uint32_t* indices, uint32_t nt) {
    try {
        ttr::scene::Mesh m;
        m.position.assign(positions, positions + 3ull * nv);
        m.normal.assign(normals, normals + 3ull * nv);
        m.index.assign(indices, indices + 3ull * nt);
        for (uint32_t i : m.index) {
            if (i >= nv) {
                g_error = "mesh index out of range";
                return -1;
            }
        }
        return static_cast<int32_t>(s->scene.add_mesh(std::move(m)));
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int32_t ttr_add_instance(
    ttr_scene* s, uint32_t mesh, uint32_t world, uint32_t shape_id, const float albedo[3], const float pose[7]) {
    try {
        ttr::scene::Instance i;
        i.mesh = mesh;
        i.world = world;
        i.shape_id = shape_id;
        std::memcpy(i.albedo, albedo, sizeof i.albedo);
        std::memcpy(i.position, pose, 3 * sizeof(float));
        std::memcpy(i.rotation, pose + 3, 4 * sizeof(float));
        return static_cast<int32_t>(s->scene.add_instance(i));
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int ttr_set_instance_poses(ttr_scene* s, const float* poses) {
    for (size_t i = 0; i < s->scene.instances(); i++) {
        s->scene.set_pose(static_cast<uint32_t>(i), poses + 7 * i, poses + 7 * i + 3);
    }
    return 0;
}

int ttr_render(
    ttr_scene* s, const ttr_camera* camera, const float* camera_poses, const float* light_direction, const ttr_images* out,
    ttr_stats* stats) {
    try {
        ttr::scene::Camera cam;
        cam.width = camera->width;
        cam.height = camera->height;
        cam.focal_length = camera->focal_length;
        cam.horizontal_aperture = camera->horizontal_aperture;
        cam.vertical_aperture = camera->vertical_aperture;
        cam.near_clip = camera->near_clip;
        cam.far_clip = camera->far_clip;
        std::vector<ttr::scene::Pose> poses(s->scene.worlds());
        for (size_t w = 0; w < poses.size(); w++) {
            std::memcpy(poses[w].position, camera_poses + 7 * w, 3 * sizeof(float));
            std::memcpy(poses[w].rotation, camera_poses + 7 * w + 3, 4 * sizeof(float));
        }
        ttr::scene::Light light;
        if (light_direction) std::memcpy(light.direction, light_direction, sizeof light.direction);
        ttr::scene::Images img;
        img.depth = out->depth;
        img.forward_depth = out->forward_depth;
        img.shape = out->shape;
        img.albedo = out->albedo;
        img.rgb = out->rgb;
        img.normal = out->normal;
        ttr::scene::RenderStats st;
        s->scene.render(cam, poses, light, img, *s->backend, &st);
        if (stats) {
            stats->vertex_ms = st.vertex_ms;
            stats->raster_ms = st.raster_ms;
            stats->assemble_ms = st.assemble_ms;
            stats->triangles = st.triangles;
            stats->passes = st.passes;
        }
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

}  // extern "C"
