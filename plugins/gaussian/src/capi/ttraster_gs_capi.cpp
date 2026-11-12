// SPDX-License-Identifier: Apache-2.0

#include "ttraster/ttraster_gs.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "gs/ply.h"
#include "gs/render.h"
#include "ttraster/remote.h"
#ifdef TT_RASTER_WITH_TENSIX
#include "tensix/tensix_backend.h"
#endif

namespace {
thread_local std::string g_error;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
}  // namespace

struct ttr_gs_scene {
    ttr::gs::GaussianScene scene;
};

struct ttr_gs_renderer {
    std::unique_ptr<ttr::Backend> backend;   // null: the float CPU reference
};

extern "C" {

const char* ttr_gs_last_error(void) { return g_error.c_str(); }

ttr_gs_scene* ttr_gs_scene_create(uint32_t sh_degree, uint64_t n, const float* xyz, const float* scale, const float* rot, const float* opacity, const float* sh) {
    try {
        auto s = std::make_unique<ttr_gs_scene>();
        s->scene.sh_degree = sh_degree;
        if (sh_degree > 3) { g_error = "sh_degree must be 0 to 3"; return nullptr; }
        const size_t K = s->scene.sh_coeffs();
        s->scene.xyz.assign(xyz, xyz + 3 * n);
        s->scene.scale.assign(scale, scale + 3 * n);
        s->scene.rot.assign(rot, rot + 4 * n);
        s->scene.opacity.assign(opacity, opacity + n);
        s->scene.sh.assign(sh, sh + n * K * 3);
        const std::string why = s->scene.validate();
        if (!why.empty()) { g_error = why; return nullptr; }
        return s.release();
    } catch (const std::exception& e) {
        g_error = e.what();
        return nullptr;
    }
}

ttr_gs_scene* ttr_gs_scene_load_ply(const char* path) {
    auto s = std::make_unique<ttr_gs_scene>();
    std::string err;
    if (!ttr::gs::read_ply(path, &s->scene, &err)) { g_error = err; return nullptr; }
    return s.release();
}

ttr_gs_scene* ttr_gs_scene_synthetic(uint64_t n, uint32_t seed, uint32_t sh_degree) {
    if (sh_degree > 3) { g_error = "sh_degree must be 0 to 3"; return nullptr; }
    auto s = std::make_unique<ttr_gs_scene>();
    s->scene = ttr::gs::make_synthetic_scene(static_cast<size_t>(n), seed, sh_degree);
    return s.release();
}

int ttr_gs_scene_save_ply(const ttr_gs_scene* scene, const char* path) {
    std::string err;
    if (!ttr::gs::write_ply(path, scene->scene, &err)) { g_error = err; return -1; }
    return 0;
}

uint64_t ttr_gs_scene_size(const ttr_gs_scene* scene) { return scene->scene.size(); }
uint32_t ttr_gs_scene_sh_degree(const ttr_gs_scene* scene) { return scene->scene.sh_degree; }

int ttr_gs_scene_get(const ttr_gs_scene* scene, float* xyz, float* scale, float* rot, float* opacity, float* sh) {
    const ttr::gs::GaussianScene& s = scene->scene;
    std::memcpy(xyz, s.xyz.data(), s.xyz.size() * sizeof(float));
    std::memcpy(scale, s.scale.data(), s.scale.size() * sizeof(float));
    std::memcpy(rot, s.rot.data(), s.rot.size() * sizeof(float));
    std::memcpy(opacity, s.opacity.data(), s.opacity.size() * sizeof(float));
    std::memcpy(sh, s.sh.data(), s.sh.size() * sizeof(float));
    return 0;
}

void ttr_gs_scene_destroy(ttr_gs_scene* scene) { delete scene; }

int ttr_gs_look_at(ttr_gs_camera* camera, const float eye[3], const float target[3], const float up[3], float fov_x_degrees, uint32_t width, uint32_t height) {
    if (width == 0 || height == 0 || !(fov_x_degrees > 0.0f && fov_x_degrees < 179.0f)) { g_error = "bad size or field of view"; return -1; }
    const ttr::gs::Camera c = ttr::gs::Camera::look_at(eye, target, up, fov_x_degrees, width, height);
    static_assert(sizeof(ttr_gs_camera) == sizeof(float) * 16 + sizeof(float) * 4 + 2 * sizeof(uint32_t) + sizeof(float), "ttr_gs_camera layout");
    std::memcpy(camera->view, c.view, sizeof c.view);
    camera->fx = c.fx; camera->fy = c.fy; camera->cx = c.cx; camera->cy = c.cy;
    camera->width = c.width; camera->height = c.height; camera->znear = c.znear;
    return 0;
}

ttr_gs_renderer* ttr_gs_renderer_create(const char* backend, uint32_t n_chips) {
    try {
        auto r = std::make_unique<ttr_gs_renderer>();
        const std::string b = backend ? backend : "cpu";
        if (b == "cpu") {
        } else if (b == "reference") {
            r->backend = std::make_unique<ttr::ReferenceBackend>();
        } else if (b == "remote" || (b == "tensix" && std::getenv("TTR_SERVER") != nullptr && std::getenv("TTR_SERVER")[0] != '\0')) {
            r->backend = std::make_unique<ttr::RemoteBackend>();
        } else if (b == "tensix") {
#ifdef TT_RASTER_WITH_TENSIX
            r->backend = ttr::make_tensix_backend(n_chips == 0 ? 1 : n_chips);
#else
            g_error = "this libttraster_gaussian was built without Tensix support";
            return nullptr;
#endif
        } else {
            g_error = "unknown backend '" + b + "'";
            return nullptr;
        }
        return r.release();
    } catch (const std::exception& e) {
        g_error = e.what();
        return nullptr;
    }
}

void ttr_gs_renderer_destroy(ttr_gs_renderer* renderer) { delete renderer; }

int ttr_gs_render(ttr_gs_renderer* renderer, const ttr_gs_scene* scene, const ttr_gs_camera* camera, const float background[3], uint8_t* rgba, ttr_gs_stats* stats) {
    try {
        ttr::gs::Camera cam;
        std::memcpy(cam.view, camera->view, sizeof cam.view);
        cam.fx = camera->fx; cam.fy = camera->fy; cam.cx = camera->cx; cam.cy = camera->cy;
        cam.width = camera->width; cam.height = camera->height; cam.znear = camera->znear > 0 ? camera->znear : 0.2f;
        auto t0 = Clock::now();
        const std::vector<ttr::gs::Splat> splats = ttr::gs::project(scene->scene, cam);
        const double project_ms = ms_since(t0);
        t0 = Clock::now();
        ttr::Image img;
        if (renderer->backend == nullptr) img = ttr::gs::to_image(ttr::gs::render_reference(splats, cam, background));
        else img = ttr::gs::render_backend(splats, cam, background, *renderer->backend);
        std::memcpy(rgba, img.rgba.data(), img.rgba.size());
        if (stats != nullptr) {
            stats->gaussians = scene->scene.size();
            stats->splats = splats.size();
            stats->project_ms = project_ms;
            stats->render_ms = ms_since(t0);
        }
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

}  // extern "C"
