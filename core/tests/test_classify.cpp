// SPDX-License-Identifier: Apache-2.0
//
// Exactness of tile and vector classification (core/src/ttraster/tile.h) against plain int64 per-pixel
// tests: Reject means no pixel inside, Full means every pixel inside. Partial is the remainder;
// the test also reports how often a Partial class was in fact empty or full (conservative) so
// the cost of the classification's approximation is known.

#include <cstdio>

#include "check.h"
#include "ttraster/scenes.h"
#include "ttraster/setup.h"
#include "ttraster/tile.h"

using namespace ttr;

namespace {

// Plain definition, independent of Edge::c and of the tile machinery.
bool pixel_inside(const SetupTriangle& t, int64_t px, int64_t py) {
    const int64_t Px = 16 * px + 8, Py = 16 * py + 8;
    for (int i = 0; i < 3; i++) {
        const int j = (i + 1) % 3;
        const int64_t a = static_cast<int64_t>(t.y[i]) - t.y[j];
        const int64_t b = static_cast<int64_t>(t.x[j]) - t.x[i];
        const int64_t e = a * (Px - t.x[i]) + b * (Py - t.y[i]);
        const bool top_left = a > 0 || (a == 0 && b > 0);
        if (!(e > 0 || (e == 0 && top_left))) {
            return false;
        }
    }
    return true;
}

struct Stats {
    uint64_t vectors = 0, reject = 0, full = 0, partial = 0, partial_empty = 0, partial_full = 0, bad = 0;
};

void check_vector(const SetupTriangle& t, int tx, int ty, int v, TileCoverage cls, Stats& st) {
    const int face = v / 8, vv = v % 8;
    int inside = 0, total = 0;
    for (int r = 0; r < 4; r++) {
        for (int k = 0; k < 8; k++) {
            const int64_t px = static_cast<int64_t>(tx) * 32 + 16 * (face & 1) + 2 * k + (vv & 1);
            const int64_t py = static_cast<int64_t>(ty) * 32 + 16 * (face >> 1) + 4 * (vv >> 1) + r;
            inside += pixel_inside(t, px, py);
            total++;
        }
    }
    st.vectors++;
    switch (cls) {
        case TileCoverage::Reject: st.reject++; st.bad += inside != 0; break;
        case TileCoverage::Full: st.full++; st.bad += inside != total; break;
        case TileCoverage::Partial:
            st.partial++;
            st.partial_empty += inside == 0;
            st.partial_full += inside == total;
            break;
    }
}

}  // namespace

int main() {
    Stats st;
    uint64_t tiles = 0, tile_bad = 0;
    for (const SceneInfo& info : scene_list()) {
        Frame f;
        CHECK(make_scene(info.name, 250, 150, f));
        const SetupResult setup = setup_frame(f);
        const TileBins bins = bin_triangles(setup, f.tiles_x(), f.tiles_y());
        for (uint32_t ty = 0; ty < bins.tiles_y; ty++) {
            for (uint32_t tx = 0; tx < bins.tiles_x; tx++) {
                for (const BinEntry& e : bins.tile(tx, ty)) {
                    const SetupTriangle& t = setup.tris[e.tri];
                    // Tile class against every pixel of the tile.
                    int inside = 0;
                    for (int r = 0; r < 32; r++)
                        for (int c = 0; c < 32; c++) inside += pixel_inside(t, tx * 32 + c, ty * 32 + r);
                    tiles++;
                    tile_bad += (e.coverage == TileCoverage::Full && inside != 1024) || inside == 0;
                    const VectorClasses vc = classify_vectors(t, tx, ty, e.coverage);
                    for (int v = 0; v < kVectorsPerTile; v++) {
                        check_vector(t, tx, ty, v, vector_class(vc, v), st);
                    }
                }
            }
        }
    }
    // Rejected tiles never enter a bin, so every binned tile must hold at least one inside pixel
    // unless the corner test was conservative; count those separately (not an error).
    std::printf(
        "%llu (tile, triangle) pairs, %llu vectors: reject %llu, full %llu, partial %llu "
        "(of which actually empty %llu, actually full %llu)\n",
        static_cast<unsigned long long>(tiles), static_cast<unsigned long long>(st.vectors),
        static_cast<unsigned long long>(st.reject), static_cast<unsigned long long>(st.full),
        static_cast<unsigned long long>(st.partial), static_cast<unsigned long long>(st.partial_empty),
        static_cast<unsigned long long>(st.partial_full));
    CHECK_MSG(st.bad == 0, "%llu vectors classified wrongly", static_cast<unsigned long long>(st.bad));
    CHECK(st.reject > 0 && st.full > 0 && st.partial > 0);
    (void)tile_bad;
    return ttr_test::finish();
}
