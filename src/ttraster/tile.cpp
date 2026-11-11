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

namespace {

// Classify the box of pixel centres [x_lo, x_hi] x [y_lo, y_hi] (corner pixels given in subpixel centres).
TileCoverage classify_box(const SetupTriangle& t, int64_t x_lo, int64_t x_hi, int64_t y_lo, int64_t y_hi) {
    // A pixel centre outside the vertex bounding box is outside the triangle. The three edge tests
    // alone miss boxes near a vertex (each edge has a corner of the box on its inside).
    if (x_hi < std::min({t.x[0], t.x[1], t.x[2]}) || x_lo > std::max({t.x[0], t.x[1], t.x[2]}) ||
        y_hi < std::min({t.y[0], t.y[1], t.y[2]}) || y_lo > std::max({t.y[0], t.y[1], t.y[2]})) {
        return TileCoverage::Reject;
    }
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

}  // namespace

TileCoverage classify_vector(const SetupTriangle& t, int tx, int ty, int v) {
    const int face = v / 8, vv = v % 8;
    const int64_t col0 = static_cast<int64_t>(tx) * kTileSize + 16 * (face & 1) + (vv & 1);  // first column of the parity
    const int64_t row0 = static_cast<int64_t>(ty) * kTileSize + 16 * (face >> 1) + 4 * (vv >> 1);
    return classify_box(t, centre(col0), centre(col0 + 14), centre(row0), centre(row0 + 3));
}

VectorClasses classify_vectors(const SetupTriangle& t, int tx, int ty, TileCoverage tile_class) {
    VectorClasses c{{0, 0}};
    for (int v = 0; v < kVectorsPerTile; v++) {
        const TileCoverage k = tile_class == TileCoverage::Full ? TileCoverage::Full : classify_vector(t, tx, ty, v);
        c.word[v / 16] |= static_cast<uint32_t>(k) << (2 * (v % 16));
    }
    return c;
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

void bin_triangles(const SetupResult& setup, uint32_t tiles_x, uint32_t tiles_y, TileBins& b) {
    b.tiles_x = tiles_x;
    b.tiles_y = tiles_y;
    b.entries = b.full_entries = b.rejected = 0;
    b.bins.resize(static_cast<size_t>(tiles_x) * tiles_y);
    for (auto& v : b.bins) v.clear();
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
}

TileBins bin_triangles(const SetupResult& setup, uint32_t tiles_x, uint32_t tiles_y) {
    TileBins b;
    bin_triangles(setup, tiles_x, tiles_y, b);
    return b;
}

}  // namespace ttr
