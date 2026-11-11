// SPDX-License-Identifier: Apache-2.0

#include "tensix/kernels/raster_layout.h"
#include "ttraster/ttraster_scene.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <memory>
#include <string>

#include "scene/scene.h"
#include "ttraster/frame.h"
#include "ttraster/parallel.h"
#ifdef TT_RASTER_WITH_TENSIX
#include "tensix/tensix_backend.h"
#endif

namespace {
thread_local std::string g_error;
}

struct ttr_scene {
    ttr::scene::Scene scene;
    std::unique_ptr<ttr::Backend> backend;
    std::string report;
    // Pipelined frames: ticket t runs in slot t % kRenderSlots until it is collected.
    struct Job {
        std::future<ttr::scene::RenderStats> result;
        int ticket = -1;
        bool busy = false;
    };
    Job jobs[ttr::kRenderSlots];
    int next_ticket = 0;
    bool per_pixel_lighting = false;
    explicit ttr_scene(uint32_t worlds) : scene(worlds) {}
    bool in_flight() const {
        for (const Job& j : jobs) {
            if (j.busy) return true;
        }
        return false;
    }
    ~ttr_scene() {
        for (Job& j : jobs) {
            if (j.busy) j.result.wait();   // never destroy the scene under a running frame
        }
    }
};

namespace {
void fill_stats(const ttr::scene::RenderStats& st, ttr_stats* stats) {
    if (!stats) return;
    stats->vertex_ms = st.vertex_ms;
    stats->raster_ms = st.raster_ms;
    stats->assemble_ms = st.assemble_ms;
    stats->build_ms = st.build_ms;
    stats->total_ms = st.total_ms;
    stats->triangles = st.triangles;
    stats->passes = st.passes;
}

// The arguments of a render call, copied out of the caller's structs.
struct RenderArgs {
    ttr::scene::Camera cam;
    std::vector<ttr::scene::Pose> poses;
    ttr::scene::Light light;
    ttr::scene::Images img;
};

RenderArgs make_args(ttr_scene* s, const ttr_camera* camera, const float* camera_poses, const float* light_direction, const ttr_images* out) {
    RenderArgs a;
    a.cam.width = camera->width;
    a.cam.height = camera->height;
    a.cam.focal_length = camera->focal_length;
    a.cam.horizontal_aperture = camera->horizontal_aperture;
    a.cam.vertical_aperture = camera->vertical_aperture;
    a.cam.near_clip = camera->near_clip;
    a.cam.far_clip = camera->far_clip;
    a.cam.background = camera->background;
    a.poses.resize(s->scene.worlds());
    for (size_t w = 0; w < a.poses.size(); w++) {
        std::memcpy(a.poses[w].position, camera_poses + 7 * w, 3 * sizeof(float));
        std::memcpy(a.poses[w].rotation, camera_poses + 7 * w + 3, 4 * sizeof(float));
    }
    if (light_direction) std::memcpy(a.light.direction, light_direction, sizeof a.light.direction);
    a.light.per_pixel = s->per_pixel_lighting;
    a.img.depth = out->depth;
    a.img.forward_depth = out->forward_depth;
    a.img.shape = out->shape;
    a.img.albedo = out->albedo;
    a.img.rgb = out->rgb;
    a.img.normal = out->normal;
    return a;
}
}  // namespace

extern "C" {

const char* ttr_last_error(void) { return g_error.c_str(); }
const char* ttr_backend_report(ttr_scene* s) { return s->report.c_str(); }

ttr_scene* ttr_scene_create(uint32_t num_worlds, const char* backend, uint32_t n_chips) {
    try {
        auto s = std::make_unique<ttr_scene>(num_worlds);
        const std::string b = backend ? backend : "reference";
        if (b == "reference") {
            s->backend = std::make_unique<ttr::ReferenceBackend>();
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

void ttr_set_lighting(ttr_scene* s, int per_pixel) { s->per_pixel_lighting = per_pixel != 0; }

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
        if (s->in_flight()) {
            g_error = "frames are in flight: ttr_collect() them before a synchronous ttr_render()";
            return -1;
        }
        const RenderArgs a = make_args(s, camera, camera_poses, light_direction, out);
        ttr::scene::RenderStats st;
        s->scene.render(a.cam, a.poses, a.light, a.img, *s->backend, &st);
        s->report = st.backend_report;
        fill_stats(st, stats);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int ttr_submit(
    ttr_scene* s, const ttr_camera* camera, const float* camera_poses, const float* light_direction, const ttr_images* out) {
    try {
        const int ticket = s->next_ticket;
        ttr_scene::Job& job = s->jobs[ticket % ttr::kRenderSlots];
        if (job.busy) {
            g_error = "no free slot: ttr_collect() ticket " + std::to_string(job.ticket) + " first";
            return -1;
        }
        auto args = std::make_shared<RenderArgs>(make_args(s, camera, camera_poses, light_direction, out));
        auto snap = std::make_shared<ttr::scene::Scene::Snapshot>(s->scene.snapshot());   // poses as of now
        const int slot = ticket % ttr::kRenderSlots;
        job.result = std::async(std::launch::async, [s, args, snap, slot]() {
            ttr::scene::RenderStats st;
            s->scene.render_snapshot(*snap, args->cam, args->poses, args->light, args->img, *s->backend, slot, &st);
            return st;
        });
        job.ticket = ticket;
        job.busy = true;
        s->next_ticket++;
        return ticket;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int ttr_collect(ttr_scene* s, int ticket, ttr_stats* stats) {
    ttr_scene::Job& job = s->jobs[((ticket % ttr::kRenderSlots) + ttr::kRenderSlots) % ttr::kRenderSlots];
    if (!job.busy || job.ticket != ticket) {
        g_error = "ttr_collect: ticket " + std::to_string(ticket) + " is not in flight";
        return -1;
    }
    job.busy = false;   // the slot is free again whatever happens below
    try {
        const ttr::scene::RenderStats st = job.result.get();
        s->report = st.backend_report;
        fill_stats(st, stats);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

}  // extern "C"
