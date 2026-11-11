// SPDX-License-Identifier: Apache-2.0
//
// Storage reuse across frames: setup_frame / bin_triangles into persistent buffers must give exactly
// what a fresh call gives, whatever the previous frame left behind (bigger, smaller, parallel or
// sequential setup path).

#include "check.h"
#include "ttraster/scenes.h"
#include "ttraster/setup.h"
#include "ttraster/tile.h"

using namespace ttr;

namespace {

bool same_triangle(const SetupTriangle& a, const SetupTriangle& b) {
    bool ok = a.source == b.source && a.piece == b.piece && a.clipped == b.clipped && a.flat == b.flat &&
              a.tile_x0 == b.tile_x0 && a.tile_x1 == b.tile_x1 && a.tile_y0 == b.tile_y0 && a.tile_y1 == b.tile_y1;
    for (int i = 0; i < 3; i++) {
        ok = ok && a.x[i] == b.x[i] && a.y[i] == b.y[i] && a.edge[i].a == b.edge[i].a && a.edge[i].b == b.edge[i].b &&
             a.edge[i].c == b.edge[i].c && a.edge[i].top_left == b.edge[i].top_left;
    }
    ok = ok && a.depth.c == b.depth.c && a.depth.gx == b.depth.gx && a.depth.gy == b.depth.gy;
    for (int ch = 0; ch < 4; ch++) {
        ok = ok && a.color[ch].c == b.color[ch].c && a.color[ch].gx == b.color[ch].gx && a.color[ch].gy == b.color[ch].gy;
    }
    return ok;
}

bool same_setup(const SetupResult& a, const SetupResult& b) {
    if (a.tris.size() != b.tris.size() || a.stats.input != b.stats.input || a.stats.accepted != b.stats.accepted ||
        a.stats.pieces != b.stats.pieces || a.stats.clipped_sources != b.stats.clipped_sources ||
        a.stats.off_screen != b.stats.off_screen || a.stats.degenerate != b.stats.degenerate ||
        a.stats.culled != b.stats.culled) {
        return false;
    }
    for (size_t i = 0; i < a.tris.size(); i++) {
        if (!same_triangle(a.tris[i], b.tris[i])) return false;
    }
    return true;
}

bool same_bins(const TileBins& a, const TileBins& b) {
    if (a.tiles_x != b.tiles_x || a.tiles_y != b.tiles_y || a.entries != b.entries || a.full_entries != b.full_entries ||
        a.rejected != b.rejected || a.bins.size() != b.bins.size()) {
        return false;
    }
    for (size_t i = 0; i < a.bins.size(); i++) {
        if (a.bins[i].size() != b.bins[i].size()) return false;
        for (size_t k = 0; k < a.bins[i].size(); k++) {
            if (a.bins[i][k].tri != b.bins[i][k].tri || a.bins[i][k].coverage != b.bins[i][k].coverage) return false;
        }
    }
    return true;
}

Frame big_frame(uint32_t w, uint32_t h) {
    // More than 4 chunks of 4096 triangles, so setup_frame takes its multi-threaded path.
    Frame f, g;
    CHECK(make_scene("random_10k", w, h, f));
    CHECK(make_scene("tiny_10k", w, h, g));
    f.triangles.insert(f.triangles.end(), g.triangles.begin(), g.triangles.end());
    return f;
}

}  // namespace

int main() {
    Frame big = big_frame(1920, 1088);
    Frame mid = big_frame(1280, 704);
    mid.triangles.resize(mid.triangles.size() * 3 / 4);
    Frame small;
    CHECK(make_scene("fan", 640, 384, small));
    CHECK(big.triangles.size() >= 4 * 4096);

    // Order matters: big -> small (sequential path) -> mid (parallel, other target) -> big -> mid.
    Frame* order[] = {&big, &small, &mid, &big, &mid};
    SetupResult res;
    SetupWorkspace ws;
    TileBins bins;
    for (size_t step = 0; step < sizeof order / sizeof order[0]; step++) {
        const Frame& f = *order[step];
        const SetupResult fresh = setup_frame(f);
        setup_frame(f, res, ws);
        CHECK_MSG(same_setup(res, fresh), "setup differs at step %zu", step);
        const TileBins fresh_bins = bin_triangles(fresh, f.tiles_x(), f.tiles_y());
        bin_triangles(res, f.tiles_x(), f.tiles_y(), bins);
        CHECK_MSG(same_bins(bins, fresh_bins), "bins differ at step %zu", step);
    }
    return ttr_test::finish();
}
