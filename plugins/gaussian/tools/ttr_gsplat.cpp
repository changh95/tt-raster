// SPDX-License-Identifier: Apache-2.0
//
// Render a 3D Gaussian Splatting scene (a trained model's PLY, or a synthetic one) with the CPU reference and through a backend of the
// renderer core, and compare the two.
//
//   ttr_gsplat --synth 200000 --size 1280x720 --out /tmp/gs                       # reference only, writes /tmp/gs_ref.png
//   ttr_gsplat --ply point_cloud.ply --backend reference --out /tmp/gs            # + the shader-triangle path on the CPU backend
//   scripts/run.sh 5 build/plugins/gaussian/ttr_gsplat --synth 200000 --backend tensix --runs 5
//
// Options: --ply FILE | --synth N | --demo N (a chequered floor, a sphere and a torus) [--seed S] [--sh-degree D]; --size WxH (default 1280x720); --fov DEG (60); --azimuth DEG, --elevation DEG,
// --distance D (camera on a sphere around the scene's centre); --bg R,G,B (0 0 0); --backend reference|tensix|remote (none: the reference only);
// --runs N (timed device frames after one warm-up); --chips N (tensix: chips of the assigned tray, screen split between them); --no-ref (skip the float reference); --out PREFIX; --write-ply FILE (save the scene).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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

using namespace ttr;
using namespace ttr::gs;
using Clock = std::chrono::steady_clock;

namespace {
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
}

int main(int argc, char** argv) {
    std::string ply, backend_name, out_prefix;
    size_t synth = 0, demo = 0;
    uint32_t seed = 1, sh_degree = 3, W = 1280, H = 720;
    float fov = 60, azimuth = 30, elevation = 20, distance = 0;
    float bg[3] = {0, 0, 0};
    int runs = 3;
    size_t chips = 1;
    bool no_ref = false;
    std::string write_ply_path;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--ply") ply = next();
        else if (a == "--synth") synth = static_cast<size_t>(std::atoll(next()));
        else if (a == "--demo") demo = static_cast<size_t>(std::atoll(next()));
        else if (a == "--seed") seed = static_cast<uint32_t>(std::atoi(next()));
        else if (a == "--sh-degree") sh_degree = static_cast<uint32_t>(std::atoi(next()));
        else if (a == "--size") std::sscanf(next(), "%ux%u", &W, &H);
        else if (a == "--fov") fov = static_cast<float>(std::atof(next()));
        else if (a == "--azimuth") azimuth = static_cast<float>(std::atof(next()));
        else if (a == "--elevation") elevation = static_cast<float>(std::atof(next()));
        else if (a == "--distance") distance = static_cast<float>(std::atof(next()));
        else if (a == "--bg") std::sscanf(next(), "%f,%f,%f", &bg[0], &bg[1], &bg[2]);
        else if (a == "--backend") backend_name = next();
        else if (a == "--runs") runs = std::atoi(next());
        else if (a == "--chips") chips = static_cast<size_t>(std::atoi(next()));
        else if (a == "--no-ref") no_ref = true;
        else if (a == "--out") out_prefix = next();
        else if (a == "--write-ply") write_ply_path = next();
        else { std::fprintf(stderr, "ttr_gsplat: unknown option %s (see the header of tools/ttr_gsplat.cpp)\n", a.c_str()); return 2; }
    }
    GaussianScene scene;
    std::string err;
    if (!ply.empty()) {
        if (!read_ply(ply, &scene, &err)) { std::fprintf(stderr, "ttr_gsplat: %s\n", err.c_str()); return 1; }
    } else if (demo > 0) {
        scene = make_demo_scene(demo, seed);
    } else if (synth > 0) {
        scene = make_synthetic_scene(synth, seed, sh_degree);
    } else {
        std::fprintf(stderr, "ttr_gsplat: give --ply FILE, --synth N or --demo N\n");
        return 2;
    }
    if (!write_ply_path.empty() && !write_ply(write_ply_path, scene, &err)) { std::fprintf(stderr, "ttr_gsplat: %s\n", err.c_str()); return 1; }
    std::printf("scene: %zu Gaussians, SH degree %u\n", scene.size(), scene.sh_degree);

    // Camera on a sphere around the centre of the scene (the median position; the distance covers 90 % of the points when not given).
    std::vector<float> xs, ys, zs;
    for (size_t i = 0; i < scene.size(); i += std::max<size_t>(1, scene.size() / 20000)) { xs.push_back(scene.xyz[3 * i]); ys.push_back(scene.xyz[3 * i + 1]); zs.push_back(scene.xyz[3 * i + 2]); }
    auto median = [](std::vector<float> v) { std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return v[v.size() / 2]; };
    const float centre[3] = {median(xs), median(ys), median(zs)};
    if (distance <= 0) {
        std::vector<float> r;
        for (size_t i = 0; i < xs.size(); i++) r.push_back(std::sqrt((xs[i] - centre[0]) * (xs[i] - centre[0]) + (ys[i] - centre[1]) * (ys[i] - centre[1]) + (zs[i] - centre[2]) * (zs[i] - centre[2])));
        std::nth_element(r.begin(), r.begin() + r.size() * 9 / 10, r.end());
        distance = 2.2f * r[r.size() * 9 / 10] / std::tan(0.5f * fov * 3.14159265f / 180.0f) * 0.5f + r[r.size() * 9 / 10];
    }
    const float az = azimuth * 3.14159265f / 180.0f, el = elevation * 3.14159265f / 180.0f;
    const float eye[3] = {centre[0] + distance * std::cos(el) * std::sin(az), centre[1] + distance * std::sin(el), centre[2] + distance * std::cos(el) * std::cos(az)};
    const float up[3] = {0, 1, 0};
    const Camera cam = Camera::look_at(eye, centre, up, fov, W, H);

    auto t0 = Clock::now();
    ProjectStats ps;
    const std::vector<Splat> splats = project(scene, cam, &ps);
    std::printf("projection: %zu splats kept of %zu (behind %zu, faint %zu, degenerate %zu, off screen %zu), %.1f ms\n", ps.kept, ps.input, ps.behind, ps.faint,
                ps.degenerate, ps.off_screen, ms_since(t0));

    Image ref_img;
    if (!no_ref) {
        RenderStats rs;
        t0 = Clock::now();
        const FloatImage f = render_reference(splats, cam, bg, &rs);
        std::printf("reference (float, front to back): %zu tile records, bin %.1f ms, raster %.1f ms\n", rs.tile_records, rs.bin_ms, rs.raster_ms);
        ref_img = to_image(f);
        if (!out_prefix.empty()) write_png(out_prefix + "_ref.png", ref_img);
    }
    if (!backend_name.empty()) {
        std::unique_ptr<Backend> backend;
        if (backend_name == "reference") backend = std::make_unique<ReferenceBackend>();
        else if (backend_name == "remote") backend = std::make_unique<RemoteBackend>();
        else if (backend_name == "tensix") {
#ifdef TT_RASTER_WITH_TENSIX
            backend = make_tensix_backend(chips);
#else
            std::fprintf(stderr, "ttr_gsplat: built without Tensix support\n");
            return 2;
#endif
        } else { std::fprintf(stderr, "ttr_gsplat: unknown backend '%s'\n", backend_name.c_str()); return 2; }
        RenderStats st;
        Image img = render_backend(splats, cam, bg, *backend, &st);   // warm-up (kernel build on a chip)
        std::vector<double> total;
        for (int r = 0; r < runs; r++) {
            t0 = Clock::now();
            img = render_backend(splats, cam, bg, *backend, &st);
            total.push_back(ms_since(t0));
        }
        std::sort(total.begin(), total.end());
        if (!total.empty()) std::printf("%s backend: frame median %.1f ms (build %.1f ms, render %.1f ms) for %zu splats, %zu triangles; %s\n", backend_name.c_str(), total[total.size() / 2],
                                        st.build_ms, st.render_ms, st.splats, 2 * st.splats, backend->report().c_str());
        if (!out_prefix.empty()) write_png(out_prefix + "_" + backend_name + ".png", img);
        if (!no_ref) {
            uint32_t maxerr = 0;
            const double p = psnr_rgb(ref_img, img, &maxerr);
            std::printf("against the float reference: PSNR %.2f dB, max error %u levels\n", p, maxerr);
        }
    }
    return 0;
}
