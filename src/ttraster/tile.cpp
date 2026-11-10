// SPDX-License-Identifier: Apache-2.0

#include "ttraster/tile.h"

#include <algorithm>

namespace ttr {

namespace {

// Pixel centre of pixel index p in subpixel units.
constexpr int64_t centre(int64_t p) { return p * kSubpixelScale + kSubpixelScale / 2; }

// Edge value at an integer subpixel point, with the top-left bias applied: inside iff >= 0.
int64_t biased_edge(const Edge& e, int64_t px, int64_t py) {
    return e.a * px + e.b * py + e.c - (e.top_left ? 0 : 1);
}

}  // namespace

TileCoverage classify_tile(const SetupTriangle& t, int tx, int ty) {
    // Extreme pixel centres of the tile: a linear function is extremal at a corner.
    const int64_t x_lo = centre(static_cast<int64_t>(tx) * kTileSize);
    const int64_t x_hi = centre(static_cast<int64_t>(tx) * kTileSize + kTileSize - 1);
    const int64_t y_lo = centre(static_cast<int64_t>(ty) * kTileSize);
    const int64_t y_hi = centre(static_cast<int64_t>(ty) * kTileSize + kTileSize - 1);
    const int64_t xs[2] = {x_lo, x_hi}, ys[2] = {y_lo, y_hi};

    bool full = true;
    for (const Edge& e : t.edge) {
        int inside = 0;
        for (int64_t px : xs) {
            for (int64_t py : ys) {
                inside += biased_edge(e, px, py) >= 0;
            }
        }
        if (inside == 0) {
            return TileCoverage::Reject;
        }
        full &= inside == 4;
    }
    return full ? TileCoverage::Full : TileCoverage::Partial;
}

// Tile centre in subpixel units: between pixel 15 and 16 of the tile.
static int64_t tile_centre(int t) {
    return static_cast<int64_t>(t) * kTileSize * kSubpixelScale + kTileSize / 2 * kSubpixelScale;
}

TileEdge make_tile_edge(const Edge& e, int tx, int ty) {
    TileEdge te;
    te.a = static_cast<float>(e.a);
    te.b = static_cast<float>(e.b);
    te.c = static_cast<float>(biased_edge(e, tile_centre(tx), tile_centre(ty)));  // one rounding to nearest
    return te;
}

TileParams make_tile_params(const SetupTriangle& t, int tx, int ty) {
    TileParams p{};
    p.coverage = classify_tile(t, tx, ty);
    const int64_t cx = tile_centre(tx), cy = tile_centre(ty);
    for (int i = 0; i < 3; i++) {
        p.edge[i] = make_tile_edge(t.edge[i], tx, ty);
    }

    const double tcx = static_cast<double>(cx) / kSubpixelScale;
    const double tcy = static_cast<double>(cy) / kSubpixelScale;
    auto fill = [&](const Plane& pl, float& c, float& gx, float& gy) {
        c = static_cast<float>(pl.c + pl.gx * tcx + pl.gy * tcy);
        gx = static_cast<float>(pl.gx / kSubpixelScale);  // per subpixel unit: exact scaling
        gy = static_cast<float>(pl.gy / kSubpixelScale);
    };
    fill(t.depth, p.depth_c, p.depth_gx, p.depth_gy);
    for (int ch = 0; ch < 4; ch++) {
        fill(t.color[ch], p.color_c[ch], p.color_gx[ch], p.color_gy[ch]);
    }
    return p;
}

TileBins bin_triangles(const SetupResult& setup, uint32_t tiles_x, uint32_t tiles_y) {
    TileBins b;
    b.tiles_x = tiles_x;
    b.tiles_y = tiles_y;
    b.bins.resize(static_cast<size_t>(tiles_x) * tiles_y);
    for (uint32_t i = 0; i < setup.tris.size(); i++) {
        const SetupTriangle& t = setup.tris[i];
        for (int ty = t.tile_y0; ty <= t.tile_y1; ty++) {
            for (int tx = t.tile_x0; tx <= t.tile_x1; tx++) {
                const TileCoverage c = classify_tile(t, tx, ty);
                if (c == TileCoverage::Reject) {
                    b.rejected++;
                    continue;
                }
                b.bins[static_cast<size_t>(ty) * tiles_x + tx].push_back({i, c});
                b.entries++;
                b.full_entries += c == TileCoverage::Full;
            }
        }
    }
    return b;
}

}  // namespace ttr
