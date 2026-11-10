// SPDX-License-Identifier: Apache-2.0
//
// Depth and colour: interpolation against barycentric coordinates, depth functions, clear
// values, flat shading, culling, clamping, and planes of clipped triangles.

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "check.h"
#include "ttraster/reference.h"
#include "ttraster/scenes.h"
#include "ttraster/setup.h"

using namespace ttr;

namespace {

Vertex V(float x, float y, float z, float r, float g, float b, float a = 1.0f) {
    Vertex v{};
    v.x = x;
    v.y = y;
    v.z = z;
    v.rhw = 1.0f;
    v.color[0] = r;
    v.color[1] = g;
    v.color[2] = b;
    v.color[3] = a;
    return v;
}

Triangle T(Vertex a, Vertex b, Vertex c, bool flat = false) {
    Triangle t;
    t.v[0] = a;
    t.v[1] = b;
    t.v[2] = c;
    t.flat = flat;
    return t;
}

// Exact barycentric weights of pixel centre (px+0.5, py+0.5) for the snapped vertices.
bool bary(const Triangle& t, int px, int py, double w[3]) {
    SnappedTriangle s;
    if (snap_triangle(t, CullMode::None, s) != SetupStatus::Ok) {
        return false;
    }
    const double x[3] = {s.x[0] / 16.0, s.x[1] / 16.0, s.x[2] / 16.0};
    const double y[3] = {s.y[0] / 16.0, s.y[1] / 16.0, s.y[2] / 16.0};
    const double X = px + 0.5, Y = py + 0.5;
    const double det = (x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0]);
    w[1] = ((X - x[0]) * (y[2] - y[0]) - (Y - y[0]) * (x[2] - x[0])) / det;
    w[2] = ((x[1] - x[0]) * (Y - y[0]) - (y[1] - y[0]) * (X - x[0])) / det;
    w[0] = 1.0 - w[1] - w[2];
    return true;
}

uint8_t channel(uint32_t packed, int c) { return static_cast<uint8_t>(packed >> (8 * c)); }

}  // namespace

int main() {
    // Smooth colour and depth match the exact barycentric interpolation (1 LSB, 1e-6 in depth).
    {
        Frame f;
        f.width = f.height = 96;
        const Triangle t = T(V(5.3f, 7.9f, 0.1f, 1, 0, 0, 0.25f), V(90.2f, 20.1f, 0.6f, 0, 1, 0, 0.5f), V(30.7f, 88.6f, 0.9f, 0, 0, 1, 1.0f));
        f.triangles.push_back(t);
        const Framebuffer fb = render_reference(f);
        int checked = 0, worst = 0;
        double worst_z = 0;
        for (int y = 0; y < 96; y++) {
            for (int x = 0; x < 96; x++) {
                double w[3];
                CHECK(bary(t, x, y, w));
                const bool inside = w[0] > 1e-9 && w[1] > 1e-9 && w[2] > 1e-9;
                const bool outside = w[0] < -1e-9 || w[1] < -1e-9 || w[2] < -1e-9;
                const size_t i = fb.index(x, y);
                if (inside) {
                    checked++;
                    CHECK(fb.cover[i] == 1);
                    for (int c = 0; c < 4; c++) {
                        const double want = w[0] * t.v[0].color[c] + w[1] * t.v[1].color[c] + w[2] * t.v[2].color[c];
                        const double q = std::floor(want * 255.0 + 0.5);
                        const int e = std::abs(static_cast<int>(channel(fb.color[i], c)) - static_cast<int>(q));
                        worst = e > worst ? e : worst;
                    }
                    const double wz = w[0] * 0.1 + w[1] * 0.6 + w[2] * 0.9;
                    worst_z = std::fmax(worst_z, std::fabs(fb.depth[i] - wz));
                } else if (outside) {
                    CHECK(fb.cover[i] == 0);
                    CHECK(fb.color[i] == pack_rgba8(f.clear_color) && fb.depth[i] == f.clear_depth && fb.id[i] == 0);
                }
            }
        }
        std::printf("smooth triangle: %d pixels checked, worst colour error %d LSB, worst depth error %.2e\n", checked, worst, worst_z);
        CHECK(checked > 3000);
        CHECK(worst <= 1);
        CHECK(worst_z < 1e-6);
    }

    // Depth functions. B is drawn after A at the same depth: Less keeps A, LessEqual takes B.
    {
        Frame f;
        f.width = f.height = 64;
        const Triangle a = T(V(2, 2, 0.5f, 1, 0, 0), V(60, 2, 0.5f, 1, 0, 0), V(2, 60, 0.5f, 1, 0, 0), true);
        const Triangle b = T(V(2, 2, 0.5f, 0, 1, 0), V(60, 2, 0.5f, 0, 1, 0), V(2, 60, 0.5f, 0, 1, 0), true);
        f.triangles = {a, b};
        f.depth_func = DepthFunc::Less;
        Framebuffer fb = render_reference(f);
        CHECK(channel(fb.color[fb.index(10, 10)], 0) == 255 && channel(fb.color[fb.index(10, 10)], 1) == 0);
        CHECK(fb.id[fb.index(10, 10)] == 1 && fb.cover[fb.index(10, 10)] == 2);
        f.depth_func = DepthFunc::LessEqual;
        fb = render_reference(f);
        CHECK(channel(fb.color[fb.index(10, 10)], 0) == 0 && channel(fb.color[fb.index(10, 10)], 1) == 255);
        CHECK(fb.id[fb.index(10, 10)] == 2);

        // A nearer triangle drawn first is not overwritten by a farther one, either way round.
        Triangle near_tri = a, far_tri = b;
        for (Vertex& v : near_tri.v) v.z = 0.25f;
        for (Vertex& v : far_tri.v) v.z = 0.75f;
        f.depth_func = DepthFunc::Less;
        f.triangles = {near_tri, far_tri};
        fb = render_reference(f);
        CHECK(fb.id[fb.index(10, 10)] == 1);
        f.triangles = {far_tri, near_tri};
        fb = render_reference(f);
        CHECK(fb.id[fb.index(10, 10)] == 2 && channel(fb.color[fb.index(10, 10)], 0) == 255);
    }

    // Flat shading uses the provoking vertex (v[0]); smooth does not.
    {
        Frame f;
        f.width = f.height = 64;
        f.triangles.push_back(T(V(2, 2, 0.5f, 1, 0, 0), V(60, 2, 0.5f, 0, 1, 0), V(2, 60, 0.5f, 0, 0, 1), true));
        const Framebuffer fb = render_reference(f);
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                if (fb.cover[fb.index(x, y)] != 0) {
                    CHECK(fb.color[fb.index(x, y)] == pack_rgba8(f.triangles[0].v[0].color));
                }
            }
        }
    }

    // Culling removes the triangle entirely.
    {
        Frame f;
        f.width = f.height = 64;
        f.triangles.push_back(T(V(2, 2, 0.5f, 1, 0, 0), V(60, 2, 0.5f, 1, 0, 0), V(2, 60, 0.5f, 1, 0, 0), true));
        f.cull = CullMode::Positive;
        CHECK(render_reference(f).cover[f.width * 10 + 10] == 0);
        f.cull = CullMode::Negative;
        CHECK(render_reference(f).cover[f.width * 10 + 10] == 1);
    }

    // Quantization: clamping, rounding, NaN.
    {
        CHECK(quantize_unorm8(-0.5f) == 0 && quantize_unorm8(0.0f) == 0 && quantize_unorm8(1.0f) == 255 &&
              quantize_unorm8(7.0f) == 255 && quantize_unorm8(NAN) == 0);
        CHECK(quantize_unorm8(0.5f) == 128);       // 127.5 + 0.5 -> 128
        CHECK(quantize_unorm8(1.0f / 255.0f) == 1);
        CHECK(quantize_unorm8(0.4999f / 255.0f) == 0);
        const float c[4] = {1.0f, 0.0f, 0.5f, 1.0f};
        CHECK(pack_rgba8(c) == 0xFF80'00FFu);
    }

    // The padded part of the target is rasterized and cropped: a triangle reaching past the
    // viewport edge still shades the padding, and padded sizes are as specified.
    {
        Frame f;
        f.width = 100;
        f.height = 40;
        CHECK(f.padded_width() == 128 && f.padded_height() == 64 && f.tiles_x() == 4 && f.tiles_y() == 2);
        f.triangles.push_back(T(V(0, 0, 0.5f, 1, 1, 1), V(200, 0, 0.5f, 1, 1, 1), V(0, 200, 0.5f, 1, 1, 1), true));
        const Framebuffer fb = render_reference(f);
        CHECK(fb.width == 128 && fb.height == 64 && fb.cover[fb.index(110, 10)] == 1);
    }

    // A clipped (guard-band) triangle keeps the plane of the original triangle: colour and depth at
    // the viewport corners match the exact barycentric values.
    {
        Frame f;
        CHECK(make_scene("fullscreen", 1920, 1088, f));
        CHECK(setup_frame(f).stats.clipped_sources == 1);
        const Framebuffer fb = render_reference(f);
        const Triangle& t = f.triangles[0];
        int worst = 0;
        double worst_z = 0;
        for (auto [x, y] : {std::pair<int, int>{0, 0}, {1919, 0}, {0, 1087}, {1919, 1087}, {960, 544}}) {
            double w[3];
            CHECK(bary(t, x, y, w));
            const size_t i = fb.index(x, y);
            for (int c = 0; c < 3; c++) {
                const double want = w[0] * t.v[0].color[c] + w[1] * t.v[1].color[c] + w[2] * t.v[2].color[c];
                worst = std::max(worst, std::abs(static_cast<int>(channel(fb.color[i], c)) - static_cast<int>(std::floor(want * 255.0 + 0.5))));
            }
            const double wz = w[0] * t.v[0].z + w[1] * t.v[1].z + w[2] * t.v[2].z;
            worst_z = std::fmax(worst_z, std::fabs(fb.depth[i] - wz));
        }
        std::printf("clipped fullscreen triangle: worst colour error %d LSB, depth error %.2e\n", worst, worst_z);
        CHECK(worst <= 1 && worst_z < 1e-6);
    }

    return ttr_test::finish();
}
