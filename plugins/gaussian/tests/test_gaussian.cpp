// SPDX-License-Identifier: Apache-2.0
//
// 3D Gaussian Splatting: spherical harmonics, EWA projection against a Monte-Carlo estimate, culling, the PLY format, the reference renderer
// against closed forms, and the renderer through the core's backend against the reference.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <array>
#include <cstring>
#include <random>
#include <unistd.h>

#include "check.h"
#include "gs/ply.h"
#include "gs/render.h"

using namespace ttr;
using namespace ttr::gs;

namespace {

GaussianScene one_gaussian(float x, float y, float z, float sigma, float opacity, const float rgb[3], uint32_t degree = 0) {
    GaussianScene s;
    s.sh_degree = degree;
    s.xyz = {x, y, z};
    s.scale = {sigma, sigma, sigma};
    s.rot = {1, 0, 0, 0};
    s.opacity = {opacity};
    s.sh.assign(static_cast<size_t>(s.sh_coeffs()) * 3, 0.0f);
    for (int c = 0; c < 3; c++) s.sh[c] = (rgb[c] - 0.5f) / 0.28209479177387814f;
    return s;
}

Camera front_camera(uint32_t w, uint32_t h) {
    const float eye[3] = {0, 0, 0}, target[3] = {0, 0, 1}, up[3] = {0, -1, 0};   // world z forward, world y down: the view matrix is the identity
    return Camera::look_at(eye, target, up, 60.0f, w, h);
}

}  // namespace

int main() {
    // -- camera conventions -------------------------------------------------------------------------------------------------------------
    {
        const float eye[3] = {3, 2, -5}, target[3] = {0.5f, 0.2f, 0.1f}, up[3] = {0, 1, 0};
        const Camera c = Camera::look_at(eye, target, up, 70.0f, 640, 480);
        auto project_point = [&](const float p[3], float* x, float* y, float* z) {
            const float tx = c.view[0] * p[0] + c.view[1] * p[1] + c.view[2] * p[2] + c.view[3];
            const float ty = c.view[4] * p[0] + c.view[5] * p[1] + c.view[6] * p[2] + c.view[7];
            *z = c.view[8] * p[0] + c.view[9] * p[1] + c.view[10] * p[2] + c.view[11];
            *x = c.fx * tx / *z + c.cx; *y = c.fy * ty / *z + c.cy;
        };
        float x, y, z;
        project_point(target, &x, &y, &z);
        CHECK(std::fabs(x - 320) < 1e-3f && std::fabs(y - 240) < 1e-3f && z > 0);
        const float above[3] = {target[0], target[1] + 0.3f, target[2]};   // a point above the target (world up) lands higher on the screen
        project_point(above, &x, &y, &z);
        CHECK(y < 240);
        float pos[3];
        c.position(pos);
        CHECK(std::fabs(pos[0] - 3) < 1e-4f && std::fabs(pos[1] - 2) < 1e-4f && std::fabs(pos[2] + 5) < 1e-4f);
        const float right_pt[3] = {target[0] + 0.3f * c.view[0], target[1] + 0.3f * c.view[1], target[2] + 0.3f * c.view[2]};   // along the camera's x axis
        project_point(right_pt, &x, &y, &z);
        CHECK(x > 320);
    }

    // -- spherical harmonics: orthonormal basis (numerical quadrature over the sphere) -------------------------------------------------
    {
        constexpr int kN = 16;
        std::vector<float> sh(kN * 3, 0.0f);
        double gram[kN][kN] = {};
        const int nt = 400, np = 800;
        for (int it = 0; it < nt; it++) {
            const double theta = (it + 0.5) * M_PI / nt;
            for (int ip = 0; ip < np; ip++) {
                const double phi = (ip + 0.5) * 2 * M_PI / np;
                const float dir[3] = {static_cast<float>(std::sin(theta) * std::cos(phi)), static_cast<float>(std::sin(theta) * std::sin(phi)), static_cast<float>(std::cos(theta))};
                double basis[kN];
                for (int k = 0; k < kN; k++) {
                    std::fill(sh.begin(), sh.end(), 0.0f);
                    sh[k * 3] = 1.0f;
                    float rgb[3];
                    sh_to_rgb(3, sh.data(), dir, rgb);
                    basis[k] = rgb[0] - 0.5f;
                }
                const double w = std::sin(theta) * (M_PI / nt) * (2 * M_PI / np);
                for (int a = 0; a < kN; a++) for (int b = 0; b < kN; b++) gram[a][b] += basis[a] * basis[b] * w;
            }
        }
        double worst = 0;
        for (int a = 0; a < kN; a++) for (int b = 0; b < kN; b++) worst = std::max(worst, std::fabs(gram[a][b] - (a == b ? 1.0 : 0.0)));
        CHECK_MSG(worst < 2e-3, "SH basis orthonormal, worst deviation %g", worst);
        // the sign convention of the reference implementation: degree 1 is (-y, z, -x) times 0.4886
        float rgb[3];
        std::fill(sh.begin(), sh.end(), 0.0f);
        sh[1 * 3] = 1.0f;   // coefficient 1: -C1 * y
        const float dy[3] = {0, 1, 0};
        sh_to_rgb(1, sh.data(), dy, rgb);
        CHECK(std::fabs(rgb[0] - (0.5f - 0.4886025f)) < 1e-6f);
    }

    // -- EWA against a Monte-Carlo estimate of the projected covariance ----------------------------------------------------------------
    {
        // an anisotropic, rotated Gaussian off the axis, small enough for the linearization to hold
        GaussianScene s;
        s.sh_degree = 0;
        s.xyz = {0.4f, -0.2f, 3.0f};
        s.scale = {0.12f, 0.03f, 0.07f};
        const float ang = 0.9f;
        const float axis[3] = {0.3f, 0.5f, 0.81f};
        const float al = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
        s.rot = {std::cos(ang / 2), std::sin(ang / 2) * axis[0] / al, std::sin(ang / 2) * axis[1] / al, std::sin(ang / 2) * axis[2] / al};
        s.opacity = {0.8f};
        s.sh = {0, 0, 0};
        const Camera cam = front_camera(640, 480);
        const std::vector<Splat> sp = project(s, cam);
        CHECK(sp.size() == 1);
        if (sp.size() == 1) {
            // Monte Carlo: samples of the 3D Gaussian, exact perspective projection, sample covariance
            std::mt19937 rng(5);
            std::normal_distribution<double> g(0, 1);
            const double r = s.rot[0], x = s.rot[1], y = s.rot[2], z = s.rot[3];
            const double R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y), 2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
                                 2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)};
            double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
            const int N = 400000;
            for (int i = 0; i < N; i++) {
                double l[3] = {g(rng) * s.scale[0], g(rng) * s.scale[1], g(rng) * s.scale[2]};
                double p[3];
                for (int a = 0; a < 3; a++) p[a] = s.xyz[a] + R[a * 3] * l[0] + R[a * 3 + 1] * l[1] + R[a * 3 + 2] * l[2];
                const double u = cam.fx * p[0] / p[2], v = cam.fy * p[1] / p[2];
                sx += u; sy += v; sxx += u * u; syy += v * v; sxy += u * v;
            }
            const double mxu = sx / N, myv = sy / N;
            const double vxx = sxx / N - mxu * mxu, vyy = syy / N - myv * myv, vxy = sxy / N - mxu * myv;
            // the conic is the inverse of cov + 0.3 I
            const double a = sp[0].conic[2], b = -sp[0].conic[1], c = sp[0].conic[0];   // cov2 = inverse of conic: [[c, -b'], [-b', a]] / det
            const double det = static_cast<double>(sp[0].conic[0]) * sp[0].conic[2] - static_cast<double>(sp[0].conic[1]) * sp[0].conic[1];
            const double cxx = sp[0].conic[2] / det, cyy = sp[0].conic[0] / det, cxy = -sp[0].conic[1] / det;
            (void)a; (void)b; (void)c;
            CHECK_MSG(std::fabs(cxx - 0.3 - vxx) < 0.04 * vxx + 0.05, "cov xx %g vs Monte Carlo %g", cxx - 0.3, vxx);
            CHECK_MSG(std::fabs(cyy - 0.3 - vyy) < 0.04 * vyy + 0.05, "cov yy %g vs Monte Carlo %g", cyy - 0.3, vyy);
            CHECK_MSG(std::fabs(cxy - vxy) < 0.04 * std::sqrt(vxx * vyy) + 0.05, "cov xy %g vs Monte Carlo %g", cxy, vxy);
            CHECK(std::fabs(sp[0].mx - (cam.fx * 0.4f / 3.0f + cam.cx)) < 1e-3f && std::fabs(sp[0].my - (cam.fy * -0.2f / 3.0f + cam.cy)) < 1e-3f);
            CHECK(std::fabs(sp[0].depth - 3.0f) < 1e-6f);
            // radius: the square contains the ellipse where alpha >= 1/255
            const double lam1 = 0.5 * (cxx + cyy) + std::sqrt(0.25 * (cxx - cyy) * (cxx - cyy) + cxy * cxy);
            const double want = std::sqrt(2 * std::log(255 * 0.8) * lam1) + 1;
            CHECK_MSG(std::fabs(sp[0].radius - want) < 0.05, "radius %g vs %g", sp[0].radius, want);
        }
    }

    // -- culling ---------------------------------------------------------------------------------------------------------------------------
    {
        GaussianScene s = one_gaussian(0, 0, 3, 0.05f, 0.9f, std::array<float, 3>{0.5f, 0.5f, 0.5f}.data());
        auto add = [&](float x, float y, float z, float o) {
            s.xyz.insert(s.xyz.end(), {x, y, z}); s.scale.insert(s.scale.end(), {0.05f, 0.05f, 0.05f}); s.rot.insert(s.rot.end(), {1, 0, 0, 0});
            s.opacity.push_back(o); s.sh.insert(s.sh.end(), {0, 0, 0});
        };
        add(0, 0, -1, 0.9f);       // behind
        add(0, 0, 0.1f, 0.9f);     // closer than znear
        add(0, 0, 3, 0.003f);      // too faint ever to reach 1/255
        add(50, 0, 3, 0.9f);       // far off screen
        ProjectStats st;
        const std::vector<Splat> sp = project(s, front_camera(640, 480), &st);
        CHECK(sp.size() == 1 && st.kept == 1 && st.behind == 2 && st.faint == 1 && st.off_screen == 1 && st.input == 5);
        // threads do not change the result
        const std::vector<Splat> sp1 = project(make_synthetic_scene(30000, 3), front_camera(320, 240), nullptr, 1);
        const std::vector<Splat> sp8 = project(make_synthetic_scene(30000, 3), front_camera(320, 240), nullptr, 8);
        bool same = sp1.size() == sp8.size();
        for (size_t i = 0; same && i < sp1.size(); i++) same = std::memcmp(&sp1[i], &sp8[i], sizeof(Splat)) == 0;
        CHECK(same);
    }

    // -- PLY ---------------------------------------------------------------------------------------------------------------------------------
    {
        namespace fs = std::filesystem;
        const std::string dir = (fs::temp_directory_path() / ("ttr-gs-test-" + std::to_string(::getpid()))).string();
        fs::create_directories(dir);
        for (uint32_t degree : {0u, 1u, 2u, 3u}) {
            const GaussianScene a = make_synthetic_scene(500, 11, degree);
            CHECK(a.validate().empty());
            std::string err;
            CHECK(write_ply(dir + "/s.ply", a, &err));
            GaussianScene b;
            CHECK_MSG(read_ply(dir + "/s.ply", &b, &err), "%s", err.c_str());
            bool ok = b.size() == a.size() && b.sh_degree == degree && b.sh.size() == a.sh.size();
            double worst = 0;
            for (size_t i = 0; ok && i < a.xyz.size(); i++) worst = std::max(worst, static_cast<double>(std::fabs(a.xyz[i] - b.xyz[i])));
            for (size_t i = 0; ok && i < a.sh.size(); i++) worst = std::max(worst, static_cast<double>(std::fabs(a.sh[i] - b.sh[i])));
            for (size_t i = 0; ok && i < a.opacity.size(); i++) worst = std::max(worst, static_cast<double>(std::fabs(a.opacity[i] - b.opacity[i])));
            for (size_t i = 0; ok && i < a.scale.size(); i++) worst = std::max(worst, static_cast<double>(std::fabs(a.scale[i] - b.scale[i]) / a.scale[i]));
            for (size_t i = 0; ok && i < a.rot.size(); i++) worst = std::max(worst, static_cast<double>(std::fabs(a.rot[i] - b.rot[i])));
            CHECK_MSG(ok && worst < 1e-5, "PLY round trip, SH degree %u, worst difference %g", degree, worst);
        }
        // refused: not a PLY, ASCII format, a missing property, a truncated file
        std::FILE* f = std::fopen((dir + "/bad.ply").c_str(), "wb");
        std::fputs("hello\n", f);
        std::fclose(f);
        std::string err;
        GaussianScene g;
        CHECK(!read_ply(dir + "/bad.ply", &g, &err) && !err.empty());
        f = std::fopen((dir + "/ascii.ply").c_str(), "wb");
        std::fputs("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", f);
        std::fclose(f);
        CHECK(!read_ply(dir + "/ascii.ply", &g, &err) && err.find("binary_little_endian") != std::string::npos);
        f = std::fopen((dir + "/missing.ply").c_str(), "wb");
        std::fputs("ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n", f);
        std::fputs("....", f);
        std::fclose(f);
        CHECK(!read_ply(dir + "/missing.ply", &g, &err) && err.find("missing property") != std::string::npos);
        const GaussianScene a = make_synthetic_scene(50, 2, 0);
        write_ply(dir + "/t.ply", a, &err);
        fs::resize_file(dir + "/t.ply", fs::file_size(dir + "/t.ply") - 10);
        CHECK(!read_ply(dir + "/t.ply", &g, &err) && err.find("ends inside") != std::string::npos);
        std::error_code ec;
        fs::remove_all(dir, ec);
    }

    // -- the reference renderer against closed forms ----------------------------------------------------------------------------------------
    {
        const float rgb[3] = {0.8f, 0.4f, 0.1f}, bg[3] = {0.1f, 0.2f, 0.3f};
        const Camera cam = front_camera(64, 64);
        // one Gaussian on the axis: the pixel at the centre sees alpha = min(0.99, opacity exp(power))
        const GaussianScene s = one_gaussian(0, 0, 2, 0.05f, 0.7f, rgb);
        const std::vector<Splat> sp = project(s, cam);
        CHECK(sp.size() == 1);
        const FloatImage img = render_reference(sp, cam, bg);
        for (const auto& pix : {std::pair<int, int>{32, 32}, {31, 31}, {35, 30}, {40, 40}}) {
            const float dx = pix.first + 0.5f - sp[0].mx, dy = pix.second + 0.5f - sp[0].my;
            const float power = -0.5f * (sp[0].conic[0] * dx * dx + sp[0].conic[2] * dy * dy) - sp[0].conic[1] * dx * dy;
            float alpha = std::min(0.99f, 0.7f * std::exp(power));
            if (alpha < 1.0f / 255.0f) alpha = 0.0f;
            for (int c = 0; c < 3; c++) {
                const float want = alpha * rgb[c] + (1 - alpha) * bg[c];
                CHECK_MSG(std::fabs(img.rgb[(static_cast<size_t>(pix.second) * 64 + pix.first) * 3 + c] - want) < 2e-6f, "pixel (%d, %d) channel %d", pix.first, pix.second, c);
            }
        }
        CHECK(std::fabs(img.rgb[0] - bg[0]) < 1e-7f);   // the corner is untouched background
        // order: the nearer of two Gaussians covers the farther one; swapping the depths swaps the result
        const float red[3] = {1, 0, 0}, blue[3] = {0, 0, 1};
        GaussianScene two = one_gaussian(0, 0, 2.0f, 0.1f, 0.95f, red);
        GaussianScene far_blue = one_gaussian(0, 0, 3.0f, 0.2f, 0.95f, blue);
        two.xyz.insert(two.xyz.end(), far_blue.xyz.begin(), far_blue.xyz.end()); two.scale.insert(two.scale.end(), far_blue.scale.begin(), far_blue.scale.end());
        two.rot.insert(two.rot.end(), far_blue.rot.begin(), far_blue.rot.end()); two.opacity.push_back(0.95f); two.sh.insert(two.sh.end(), far_blue.sh.begin(), far_blue.sh.end());
        const FloatImage a = render_reference(project(two, cam), cam, bg);
        const size_t centre = (32u * 64 + 32) * 3;
        CHECK(a.rgb[centre] > 0.8f && a.rgb[centre + 2] < 0.15f);   // red in front
        std::swap(two.xyz[2], two.xyz[5]);
        const FloatImage b = render_reference(project(two, cam), cam, bg);
        CHECK(b.rgb[centre + 2] > 0.7f && b.rgb[centre] < 0.3f);    // now the blue one is nearer
        // behind opaque Gaussians nothing is visible: three layers of alpha 0.99 leave a transmittance of 1e-6, so forty blue ones behind them
        // add less than a level (the early stop at T < 1e-4 is this: it cannot change a pixel by more than that)
        GaussianScene stack = one_gaussian(0, 0, 2.0f, 0.5f, 1.0f, red);
        for (int i = 1; i < 43; i++) {
            const GaussianScene extra = one_gaussian(0, 0, 2.0f + 0.1f * i, 0.5f, 1.0f, i < 3 ? red : blue);
            stack.xyz.insert(stack.xyz.end(), extra.xyz.begin(), extra.xyz.end()); stack.scale.insert(stack.scale.end(), extra.scale.begin(), extra.scale.end());
            stack.rot.insert(stack.rot.end(), extra.rot.begin(), extra.rot.end()); stack.opacity.push_back(1.0f); stack.sh.insert(stack.sh.end(), extra.sh.begin(), extra.sh.end());
        }
        const FloatImage c = render_reference(project(stack, cam), cam, bg);
        CHECK_MSG(c.rgb[centre] > 0.985f && c.rgb[centre + 2] < 0.004f, "behind opaque Gaussians: r %g b %g", c.rgb[centre], c.rgb[centre + 2]);
        // 8-bit conversion
        const Image im = to_image(a);
        CHECK(im.rgba[(32u * 64 + 32) * 4 + 3] == 255 && im.width == 64);
    }

    // -- through the core's backend (the CPU reference rasterizer here; the same Frame goes to a chip) -----------------------------------------
    {
        const GaussianScene s = make_synthetic_scene(20000, 7, 3);
        const float eye[3] = {0, 0.6f, 4.0f}, target[3] = {0, -0.1f, 0}, up[3] = {0, 1, 0};
        const Camera cam = Camera::look_at(eye, target, up, 55.0f, 320, 240);
        const float bg[3] = {0.05f, 0.05f, 0.08f};
        const std::vector<Splat> sp = project(s, cam);
        CHECK(sp.size() > 5000);
        const Image ref = to_image(render_reference(sp, cam, bg));
        ReferenceBackend be;
        const Image dev = render_backend(sp, cam, bg, be);
        uint32_t maxerr = 0;
        const double p = psnr_rgb(ref, dev, &maxerr);
        std::printf("synthetic scene, %zu splats: PSNR %.2f dB, max error %u levels (backend path against the float reference)\n", sp.size(), p, maxerr);
        CHECK_MSG(p > 40.0, "PSNR %.2f dB", p);
        // an empty scene is the background
        const Image none = render_backend({}, cam, bg, be);
        CHECK(none.rgba[0] == static_cast<uint8_t>(std::lround(0.05f * 255)) && none.rgba[2] == static_cast<uint8_t>(std::lround(0.08f * 255)));
        Camera big = cam;
        big.width = 4096;
        bool threw = false;
        try { render_backend(sp, big, bg, be); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }
    return ttr_test::finish();
}
