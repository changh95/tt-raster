// SPDX-License-Identifier: Apache-2.0
//
// Triangle setup: snapping, rejection, fill rule, bounding box, guard-band clipping.

#include <algorithm>
#include <cmath>
#include <set>

#include "check.h"
#include "ttraster/oracle.h"
#include "ttraster/reference.h"
#include "ttraster/setup.h"

using namespace ttr;

namespace {

Vertex V(float x, float y, float z = 0.5f) {
    Vertex v{};
    v.x = x;
    v.y = y;
    v.z = z;
    v.rhw = 1.0f;
    v.color[0] = v.color[1] = v.color[2] = v.color[3] = 1.0f;
    return v;
}

Triangle T(Vertex a, Vertex b, Vertex c) {
    Triangle t;
    t.v[0] = a;
    t.v[1] = b;
    t.v[2] = c;
    t.flat = true;
    return t;
}

Frame frame_with(uint32_t w, uint32_t h, Triangle t) {
    Frame f;
    f.width = w;
    f.height = h;
    f.triangles.push_back(t);
    return f;
}

uint64_t covered(const Frame& f) {
    const Framebuffer fb = render_reference(f);
    uint64_t n = 0;
    for (uint8_t c : fb.cover) {
        n += c;
    }
    return n;
}

}  // namespace

int main() {
    // Snapping rounds to the 1/16 grid, ties to even.
    {
        SnappedTriangle s;
        CHECK(snap_triangle(T(V(1.03125f, 2.0f), V(5.0f, 2.0f), V(1.0f, 9.0f)), CullMode::None, s) == SetupStatus::Ok);
        CHECK(s.x[0] == 16);    // 16.5 -> 16 (even)
        CHECK(snap_triangle(T(V(1.09375f, 2.0f), V(5.0f, 2.0f), V(1.0f, 9.0f)), CullMode::None, s) == SetupStatus::Ok);
        CHECK(s.x[0] == 18);    // 17.5 -> 18 (even)
        CHECK(s.x[1] == 80 && s.y[0] == 32 && s.y[2] == 144);
    }

    // Rejection: degenerate, collapsed by snapping, non-finite, out of range, culled.
    {
        SnappedTriangle s;
        CHECK(snap_triangle(T(V(0, 0), V(8, 8), V(16, 16)), CullMode::None, s) == SetupStatus::Degenerate);
        CHECK(snap_triangle(T(V(3.0f, 3.0f), V(3.01f, 3.0f), V(3.0f, 3.01f)), CullMode::None, s) == SetupStatus::Degenerate);
        CHECK(snap_triangle(T(V(NAN, 0), V(8, 0), V(0, 8)), CullMode::None, s) == SetupStatus::NonFinite);
        CHECK(snap_triangle(T(V(0, 0), V(8, 0), V(0, INFINITY)), CullMode::None, s) == SetupStatus::NonFinite);
        CHECK(snap_triangle(T(V(0, 0), V(8, 0), V(0, 2.0e6f)), CullMode::None, s) == SetupStatus::OutOfRange);
        const Triangle pos = T(V(0, 0), V(8, 0), V(0, 8));  // signed_area2 > 0 in y-down coordinates
        CHECK(snap_triangle(pos, CullMode::None, s) == SetupStatus::Ok && s.area2 > 0);
        CHECK(snap_triangle(pos, CullMode::Positive, s) == SetupStatus::Culled);
        CHECK(snap_triangle(pos, CullMode::Negative, s) == SetupStatus::Ok);
        const Triangle neg = T(V(0, 0), V(0, 8), V(8, 0));
        CHECK(snap_triangle(neg, CullMode::Negative, s) == SetupStatus::Culled);
        CHECK(snap_triangle(neg, CullMode::Positive, s) == SetupStatus::Ok && s.area2 < 0);
    }

    // The fill rule on a right triangle whose edges run exactly through pixel centres.
    // Vertices (0.5,0.5) (8.5,0.5) (0.5,8.5): the top edge (y = 0.5) and the left edge (x = 0.5)
    // are included, the hypotenuse x + y = 9 is not: pixels (i, j) with i, j >= 0 and i + j <= 7.
    {
        const Frame f = frame_with(32, 32, T(V(0.5f, 0.5f), V(8.5f, 0.5f), V(0.5f, 8.5f)));
        CHECK_MSG(covered(f) == 36, "got %llu", static_cast<unsigned long long>(covered(f)));
        // The same triangle with the vertex order reversed covers the same pixels.
        const Frame g = frame_with(32, 32, T(V(0.5f, 0.5f), V(0.5f, 8.5f), V(8.5f, 0.5f)));
        const Framebuffer a = render_reference(f), b = render_reference(g);
        CHECK(a.cover == b.cover);
    }

    // Two triangles sharing a diagonal that runs through pixel centres cover each pixel once.
    {
        Frame f;
        f.width = f.height = 32;
        f.triangles.push_back(T(V(0.5f, 0.5f), V(16.5f, 0.5f), V(16.5f, 16.5f)));
        f.triangles.push_back(T(V(0.5f, 0.5f), V(16.5f, 16.5f), V(0.5f, 16.5f)));
        const Framebuffer fb = render_reference(f);
        uint32_t twice = 0, once = 0;
        for (uint8_t c : fb.cover) {
            twice += c > 1;
            once += c == 1;
        }
        CHECK(twice == 0);
        CHECK_MSG(once == 16 * 16, "once %u", once);
    }

    // Bounding box and tile range, including a triangle that straddles tile corners.
    {
        Frame f = frame_with(256, 256, T(V(31.0f, 31.0f), V(65.0f, 31.0f), V(31.0f, 65.0f)));
        const SetupResult s = setup_frame(f);
        CHECK(s.tris.size() == 1);
        CHECK(s.tris[0].tile_x0 == 0 && s.tris[0].tile_x1 == 2);
        CHECK(s.tris[0].tile_y0 == 0 && s.tris[0].tile_y1 == 2);
        // A triangle that fits between pixel centres touches no pixel and is dropped.
        f = frame_with(256, 256, T(V(10.1f, 10.1f), V(10.4f, 10.1f), V(10.1f, 10.4f)));
        const SetupResult e = setup_frame(f);
        CHECK(e.tris.empty() && e.stats.off_screen == 1);
        // Fully off screen.
        f = frame_with(64, 64, T(V(-50, -50), V(-10, -50), V(-50, -10)));
        CHECK(setup_frame(f).tris.empty());
    }

    // Edge coefficients stay within 2^15 for every piece, whatever the input.
    {
        Frame f;
        f.width = 1920;
        f.height = 1088;
        f.triangles.push_back(T(V(-5000, -5000), V(9000, -3000), V(-2000, 7000)));
        f.triangles.push_back(T(V(-8, -8), V(4200, -8), V(-8, 2400)));
        f.triangles.push_back(T(V(900, 500), V(1000000, 400), V(950, 1000000)));
        const SetupResult s = setup_frame(f);
        CHECK(s.stats.clipped_sources == 3);
        const ClipRect g = guard_rect(1920, 1088);
        CHECK(g.x1 - g.x0 == 2048 * 16 && g.y1 - g.y0 == 2048 * 16);
        for (const SetupTriangle& t : s.tris) {
            for (const Edge& e : t.edge) {
                CHECK(std::abs(e.a) <= 32768 && std::abs(e.b) <= 32768);
            }
            for (int i = 0; i < 3; i++) {
                CHECK(t.x[i] >= g.x0 && t.x[i] <= g.x1 && t.y[i] >= g.y0 && t.y[i] <= g.y1);
            }
        }
    }

    // Clipping an edge is independent of its direction: the same polygon walked backwards gives
    // the same vertex set, which is what keeps neighbours watertight.
    {
        const ClipRect r{0, 0, 16000, 16000};
        std::vector<std::array<int64_t, 2>> poly = {{-5000, 3000}, {23000, -4000}, {9000, 31000}};
        auto fwd = clip_polygon(poly, r);
        std::reverse(poly.begin(), poly.end());
        auto bwd = clip_polygon(poly, r);
        std::set<std::array<int64_t, 2>> a(fwd.begin(), fwd.end()), b(bwd.begin(), bwd.end());
        CHECK(a == b);
        CHECK(fwd.size() >= 3);
        for (auto& p : fwd) {
            CHECK(p[0] >= r.x0 && p[0] <= r.x1 && p[1] >= r.y0 && p[1] <= r.y1);
        }
    }

    // A clipped triangle covers the same pixels as exact int64 coverage of its clipped pieces, and
    // clipping does not change coverage of a triangle whose far edges are outside the viewport.
    {
        const Frame f = frame_with(1920, 1088, T(V(-8, -8), V(4300, -8), V(-8, 2400)));
        const SetupResult s = setup_frame(f);
        CHECK(s.stats.clipped_sources == 1 && s.stats.pieces > 1);
        const Framebuffer fb = render_reference(f);
        CHECK(fb.cover == oracle_cover_from_setup(s, fb.width, fb.height));
        const InputOracle in = oracle_cover_from_input(f);
        CHECK(fb.cover == in.cover);  // identical even inside the clip mask: the cut edges are off screen
        uint64_t covered_px = 0;
        for (uint8_t c : fb.cover) {
            covered_px += c;
        }
        CHECK_MSG(covered_px == 1920ull * 1088ull, "covered %llu", static_cast<unsigned long long>(covered_px));
    }

    return ttr_test::finish();
}
