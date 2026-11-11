// SPDX-License-Identifier: Apache-2.0
//
// Perspective-correct colour interpolation (Triangle::perspective): the reference rasterizer against an analytic
// per-pixel computation in double precision, and the cases that must not take the perspective path.

#include <cmath>
#include <vector>

#include "check.h"
#include "ttraster/reference.h"
#include "ttraster/setup.h"

using namespace ttr;

namespace {

Vertex V(float x, float y, float z, float w, float r, float g, float b) {
    Vertex v{};
    v.x = x;
    v.y = y;
    v.z = z;
    v.rhw = 1.0f / w;
    v.color[0] = r;
    v.color[1] = g;
    v.color[2] = b;
    v.color[3] = 1.0f;
    return v;
}

Frame frame_of(uint32_t w, uint32_t h) {
    Frame f;
    f.width = w;
    f.height = h;
    f.clear_color[0] = f.clear_color[1] = f.clear_color[2] = 0.0f;
    return f;
}

// Colour of pixel (px, py) from exact barycentric weights of the 1/16-snapped vertices at the pixel centre.
void analytic(const Triangle& t, int px, int py, bool perspective, double out[3]) {
    double x[3], y[3];
    for (int i = 0; i < 3; i++) {
        x[i] = std::nearbyint(t.v[i].x * 16.0) / 16.0;
        y[i] = std::nearbyint(t.v[i].y * 16.0) / 16.0;
    }
    const double cx = px + 0.5, cy = py + 0.5;
    const double area = (x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0]);
    double l[3];
    l[0] = ((x[1] - cx) * (y[2] - cy) - (y[1] - cy) * (x[2] - cx)) / area;
    l[1] = ((x[2] - cx) * (y[0] - cy) - (y[2] - cy) * (x[0] - cx)) / area;
    l[2] = 1.0 - l[0] - l[1];
    double wsum = 0;
    for (int i = 0; i < 3; i++) wsum += l[i] * (perspective ? t.v[i].rhw : 1.0);
    for (int c = 0; c < 3; c++) {
        double s = 0;
        for (int i = 0; i < 3; i++) s += l[i] * t.v[i].color[c] * (perspective ? t.v[i].rhw : 1.0);
        out[c] = s / wsum;
    }
}

}  // namespace

int main() {
    // A big triangle whose far vertex is 8 times as far as the near ones: perspective and screen-linear differ a lot.
    Triangle tri;
    tri.v[0] = V(4.0f, 60.0f, 0.2f, 1.0f, 1.0f, 0.0f, 0.0f);
    tri.v[1] = V(60.0f, 56.0f, 0.2f, 1.0f, 0.0f, 1.0f, 0.0f);
    tri.v[2] = V(30.0f, 3.0f, 0.9f, 8.0f, 0.0f, 0.0f, 1.0f);
    tri.perspective = true;
    {
        Frame f = frame_of(64, 64);
        f.triangles.push_back(tri);
        const Framebuffer fb = render_reference(f);
        int covered = 0, over_one = 0, differs_from_linear = 0;
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                if (fb.cover[fb.index(x, y)] == 0) continue;
                covered++;
                double want[3], lin[3];
                analytic(tri, x, y, true, want);
                analytic(tri, x, y, false, lin);
                const uint32_t c = fb.color[fb.index(x, y)];
                for (int ch = 0; ch < 3; ch++) {
                    const double got = ((c >> (8 * ch)) & 0xff) / 255.0;
                    over_one += std::fabs(got - want[ch]) * 255.0 > 1.0;
                    differs_from_linear += std::fabs(got - lin[ch]) * 255.0 > 3.0;
                }
            }
        }
        CHECK_MSG(covered > 1500, "covered %d", covered);
        CHECK_MSG(over_one == 0, "%d channel values are more than one level from the analytic perspective colour", over_one);
        CHECK_MSG(differs_from_linear > 200, "the test does not tell perspective from linear (%d)", differs_from_linear);
    }

    // The same triangle, partly outside the target (guard-band clipping keeps the planes of the whole triangle).
    {
        Triangle big = tri;
        for (int i = 0; i < 3; i++) {
            big.v[i].x = (big.v[i].x - 30.0f) * 3.0f + 30.0f;
            big.v[i].y = (big.v[i].y - 30.0f) * 3.0f + 30.0f;
        }
        Frame f = frame_of(64, 64);
        f.triangles.push_back(big);
        const SetupResult s = setup_frame(f);
        const Framebuffer fb = render_reference(f);
        int bad = 0, n = 0;
        for (int y = 0; y < 64; y += 3) {
            for (int x = 0; x < 64; x += 3) {
                if (fb.cover[fb.index(x, y)] == 0) continue;
                n++;
                double want[3];
                analytic(big, x, y, true, want);
                const uint32_t c = fb.color[fb.index(x, y)];
                for (int ch = 0; ch < 3; ch++) bad += std::fabs(((c >> (8 * ch)) & 0xff) / 255.0 - want[ch]) * 255.0 > 1.5;
            }
        }
        CHECK(!s.tris.empty() && n > 100);
        CHECK_MSG(bad == 0, "%d channel values off on the cropped triangle", bad);
    }

    // Invalid rhw drops the triangle; flat triangles ignore the flag.
    {
        Frame f = frame_of(64, 64);
        Triangle bad = tri;
        bad.v[1].rhw = -1.0f;
        f.triangles.push_back(bad);
        bad.v[1].rhw = std::nanf("");
        f.triangles.push_back(bad);
        bad.v[1].rhw = 0.0f;
        f.triangles.push_back(bad);
        const SetupResult s = setup_frame(f);
        CHECK(s.tris.empty() && s.stats.non_finite == 3);
        Frame g = frame_of(64, 64);
        Triangle flat = bad;   // still has the invalid rhw, which does not matter for a flat triangle
        flat.flat = true;
        g.triangles.push_back(flat);
        const SetupResult sg = setup_frame(g);
        CHECK(sg.tris.size() == 1 && !sg.tris[0].perspective);
    }

    // With equal rhw at all vertices the perspective path is the linear one (up to one level).
    {
        Triangle eq = tri;
        for (int i = 0; i < 3; i++) eq.v[i].rhw = 0.25f;
        Triangle lin = eq;
        lin.perspective = false;
        Frame a = frame_of(64, 64), b = frame_of(64, 64);
        a.triangles.push_back(eq);
        b.triangles.push_back(lin);
        const Framebuffer fa = render_reference(a), fb = render_reference(b);
        int worst = 0;
        for (size_t i = 0; i < fa.color.size(); i++) {
            for (int ch = 0; ch < 3; ch++) {
                worst = std::max(worst, std::abs(static_cast<int>((fa.color[i] >> (8 * ch)) & 0xff) - static_cast<int>((fb.color[i] >> (8 * ch)) & 0xff)));
            }
        }
        CHECK_MSG(worst <= 1, "worst difference %d", worst);
    }
    return ttr_test::finish();
}
