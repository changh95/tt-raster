// SPDX-License-Identifier: Apache-2.0
//
// Viewports (a tiled-camera atlas): triangles are confined to their own rectangle, a frame of many
// viewports renders exactly what separate frames would, and the target may exceed kMaxTargetSize.

#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "check.h"
#include "ttraster/oracle.h"
#include "ttraster/reference.h"
#include "ttraster/scenes.h"
#include "ttraster/setup.h"

using namespace ttr;

namespace {

Triangle tri_at(float x0, float y0, float x1, float y1, float x2, float y2, float z, float r, float g, float b, uint32_t vp) {
    Triangle t;
    t.flat = true;
    t.viewport = vp;
    const float xs[3] = {x0, x1, x2}, ys[3] = {y0, y1, y2};
    for (int i = 0; i < 3; i++) {
        t.v[i].x = xs[i];
        t.v[i].y = ys[i];
        t.v[i].z = z;
        t.v[i].rhw = 1.0f;
        t.v[i].color[0] = r;
        t.v[i].color[1] = g;
        t.v[i].color[2] = b;
        t.v[i].color[3] = 1.0f;
    }
    return t;
}

}  // namespace

int main() {
    // 1. A huge triangle of viewport 0 does not touch viewport 1 (next to it) or the gap below it.
    {
        Frame f;
        f.width = 128;
        f.height = 96;
        f.viewports = {{0, 0, 64, 64}, {64, 0, 64, 64}};
        f.triangles.push_back(tri_at(-500, -500, 900, -500, -500, 900, 0.5f, 1, 0, 0, 0));  // covers all of viewport 0 and beyond
        f.triangles.push_back(tri_at(70, 10, 120, 10, 70, 50, 0.4f, 0, 1, 0, 1));
        const Framebuffer fb = render_reference(f);
        CHECK(fb.cover[fb.index(10, 10)] == 1 && fb.cover[fb.index(63, 63)] == 1);   // inside viewport 0
        CHECK(fb.cover[fb.index(64, 10)] == 0 || fb.id[fb.index(64, 10)] == 2);      // viewport 1 only has triangle 2
        uint64_t bleed = 0, in0 = 0;
        for (uint32_t y = 0; y < 96; y++) {
            for (uint32_t x = 0; x < 128; x++) {
                const size_t i = fb.index(x, y);
                if (fb.id[i] == 1) {
                    in0++;
                    bleed += !(x < 64 && y < 64);
                }
            }
        }
        CHECK_MSG(bleed == 0, "%llu pixels of viewport 0's triangle lie outside it", static_cast<unsigned long long>(bleed));
        CHECK(in0 == 64 * 64);
        CHECK(fb.cover[fb.index(10, 80)] == 0);  // below the viewports: untouched
    }

    // 2. An atlas wider than kMaxTargetSize: 40 cameras of 96 x 96 side by side (3840 x 96), every one
    //    with the same scene content, equals one camera rendered alone.
    {
        const uint32_t n = 40, size = 96;
        Frame single;
        make_scene("random_10k", size, size, single);
        single.triangles.resize(600);
        // Put the vertices on the 1/16 pixel grid first, so that adding a whole number of pixels in
        // FP32 below is exact and cannot change how a vertex snaps.
        for (Triangle& t : single.triangles)
            for (Vertex& v : t.v) {
                v.x = std::nearbyint(v.x * 16.0f) / 16.0f;
                v.y = std::nearbyint(v.y * 16.0f) / 16.0f;
            }
        // Same viewport clipping as in the atlas: clipping at a cut edge moves its cut points by up to
        // 1/32 pixel compared with leaving the triangle unclipped, so compare like with like.
        single.viewports = {{0, 0, static_cast<int32_t>(size), static_cast<int32_t>(size)}};
        const Framebuffer one = render_reference(single);

        Frame atlas;
        atlas.width = n * size;
        atlas.height = size;
        for (int c = 0; c < 4; c++) atlas.clear_color[c] = single.clear_color[c];
        for (uint32_t k = 0; k < n; k++) atlas.viewports.push_back({static_cast<int32_t>(k * size), 0, static_cast<int32_t>(size), static_cast<int32_t>(size)});
        for (uint32_t k = 0; k < n; k++) {
            for (Triangle t : single.triangles) {
                t.viewport = k;
                for (Vertex& v : t.v) v.x += static_cast<float>(k * size);
                atlas.triangles.push_back(t);
            }
        }
        const SetupResult setup = setup_frame(atlas);
        const TileBins bins = bin_triangles(setup, atlas.tiles_x(), atlas.tiles_y());
        const Framebuffer fb = render_reference(atlas, setup, bins);
        CHECK(atlas.padded_width() == 3840);
        uint64_t bad = 0, shown = 0;
        for (uint32_t k = 0; k < n; k++)
            for (uint32_t y = 0; y < size; y++)
                for (uint32_t x = 0; x < size; x++)
                {
                    const bool d = fb.color[fb.index(k * size + x, y)] != one.color[one.index(x, y)] ||
                                   fb.cover[fb.index(k * size + x, y)] != one.cover[one.index(x, y)];
                    bad += d;
                    if (d && shown++ < 3)
                        std::printf("mismatch cam %u (%u,%u): atlas cover %u id %u, single cover %u id %u\n", k, x, y,
                                    fb.cover[fb.index(k * size + x, y)], fb.id[fb.index(k * size + x, y)],
                                    one.cover[one.index(x, y)], one.id[one.index(x, y)]);
                }
        CHECK_MSG(bad == 0, "%llu atlas pixels differ from the single-camera render", static_cast<unsigned long long>(bad));
        CHECK(fb.cover == oracle_cover_from_setup(setup, fb.width, fb.height));
        std::printf("atlas %ux%u: %zu triangles, %zu pieces, %llu tile pairs\n", atlas.width, atlas.height, atlas.triangles.size(),
                    setup.tris.size(), static_cast<unsigned long long>(bins.entries));
    }

    // 3. Invalid viewports are rejected.
    {
        Frame f;
        f.width = f.height = 64;
        f.viewports = {{0, 0, 128, 64}};
        f.triangles.push_back(tri_at(1, 1, 30, 1, 1, 30, 0.5f, 1, 1, 1, 0));
        bool threw = false;
        try { (void)setup_frame(f); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }
    return ttr_test::finish();
}
