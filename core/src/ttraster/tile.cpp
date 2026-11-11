// SPDX-License-Identifier: Apache-2.0

#include "ttraster/tile.h"

#include <algorithm>

#include "ttraster/parallel.h"

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

TilePlane make_tile_plane(const Plane& pl, int tx, int ty) {
    const double tcx = static_cast<double>(tile_centre(tx)) / kSubpixelScale;
    const double tcy = static_cast<double>(tile_centre(ty)) / kSubpixelScale;
    TilePlane r;
    r.c = static_cast<float>(pl.c + pl.gx * tcx + pl.gy * tcy);
    r.gx = static_cast<float>(pl.gx / kSubpixelScale);  // per subpixel unit: exact scaling
    r.gy = static_cast<float>(pl.gy / kSubpixelScale);
    return r;
}

TileParams make_tile_params(const SetupTriangle& t, int tx, int ty) {
    TileParams p{};
    p.coverage = classify_tile(t, tx, ty);
    for (int i = 0; i < 3; i++) {
        p.edge[i] = make_tile_edge(t.edge[i], tx, ty);
    }

    auto fill = [&](const Plane& pl, float& c, float& gx, float& gy) {
        const TilePlane tp = make_tile_plane(pl, tx, ty);
        c = tp.c;
        gx = tp.gx;
        gy = tp.gy;
    };
    fill(t.depth, p.depth_c, p.depth_gx, p.depth_gy);
    for (int ch = 0; ch < 4; ch++) {
        fill(t.color[ch], p.color_c[ch], p.color_gx[ch], p.color_gy[ch]);
    }
    p.perspective = t.perspective;
    if (t.perspective) {
        fill(t.inv_w, p.w_c, p.w_gx, p.w_gy);
    }
    return p;
}

namespace {

struct BinCounts {
    uint64_t entries = 0, full = 0, rejected = 0;
};

// Bin the triangles that touch tile rows [row_lo, row_hi] into those rows. Triangles are visited in API order, so the
// bins of those rows come out exactly as in a pass over all rows. `rows` (optional) holds tile_y0 | tile_y1 << 16 of
// every triangle: a band skips the triangles outside its rows by reading 4 bytes instead of a whole SetupTriangle.
void bin_rows(const std::vector<SetupTriangle>& tris, const uint32_t* rows, uint32_t tiles_x, int row_lo, int row_hi,
              std::vector<std::vector<BinEntry>>& bins, BinCounts& n) {
    for (uint32_t i = 0; i < tris.size(); i++) {
        if (rows) {
            const int y0 = static_cast<int>(rows[i] & 0xFFFFu), y1 = static_cast<int>(rows[i] >> 16);
            if (y1 < row_lo || y0 > row_hi) continue;
        }
        const SetupTriangle& t = tris[i];
        const int y0 = std::max(t.tile_y0, row_lo), y1 = std::min(t.tile_y1, row_hi);
        for (int ty = y0; ty <= y1; ty++) {
            for (int tx = t.tile_x0; tx <= t.tile_x1; tx++) {
                const TileCoverage c = classify_tile(t, tx, ty);
                if (c == TileCoverage::Reject) {
                    n.rejected++;
                    continue;
                }
                bins[static_cast<size_t>(ty) * tiles_x + tx].push_back({i, c});
                n.entries++;
                n.full += c == TileCoverage::Full;
            }
        }
    }
}

}  // namespace

void bin_triangles(const SetupResult& setup, uint32_t tiles_x, uint32_t tiles_y, TileBins& b) {
    b.tiles_x = tiles_x;
    b.tiles_y = tiles_y;
    b.entries = b.full_entries = b.rejected = 0;
    b.bins.resize(static_cast<size_t>(tiles_x) * tiles_y);
    for (auto& v : b.bins) v.clear();

    // Tile rows are independent: a bin belongs to exactly one row, and one thread bins a band of rows walking the
    // triangles in API order. Small frames stay on the calling thread.
    constexpr size_t kParallelMinTris = 2048;
    constexpr uint64_t kParallelMinPairs = 16384;    // (tile, triangle) candidate pairs worth waking the pool for
    constexpr uint32_t kRowsPerBand = 1;
    const std::vector<SetupTriangle>& tris = setup.tris;
    uint64_t pairs = 0;
    for (size_t i = 0; i < tris.size() && pairs < kParallelMinPairs; i++) {
        pairs += static_cast<uint64_t>(tris[i].tile_x1 - tris[i].tile_x0 + 1) *
                 static_cast<uint64_t>(tris[i].tile_y1 - tris[i].tile_y0 + 1);
    }
    if (tris.size() < kParallelMinTris || pairs < kParallelMinPairs || tiles_y < 2 || tiles_y > 0xFFFFu) {
        BinCounts n;
        bin_rows(tris, nullptr, tiles_x, 0, static_cast<int>(tiles_y) - 1, b.bins, n);
        b.entries = n.entries;
        b.full_entries = n.full;
        b.rejected = n.rejected;
        return;
    }

    std::vector<uint32_t> rows(tris.size());
    constexpr size_t kChunk = 4096;
    parallel_for((tris.size() + kChunk - 1) / kChunk, 0, [&](size_t c) {
        const size_t end = std::min(tris.size(), (c + 1) * kChunk);
        for (size_t i = c * kChunk; i < end; i++) {
            rows[i] = static_cast<uint32_t>(tris[i].tile_y0) | static_cast<uint32_t>(tris[i].tile_y1) << 16;
        }
    });
    const size_t bands = (tiles_y + kRowsPerBand - 1) / kRowsPerBand;
    std::vector<BinCounts> counts(bands);
    parallel_for(bands, 0, [&](size_t band) {
        const int lo = static_cast<int>(band * kRowsPerBand);
        const int hi = std::min<int>(lo + kRowsPerBand, static_cast<int>(tiles_y)) - 1;
        BinCounts local;   // counted in a local: neighbouring entries of `counts` share a cache line
        bin_rows(tris, rows.data(), tiles_x, lo, hi, b.bins, local);
        counts[band] = local;
    });
    for (const BinCounts& n : counts) {
        b.entries += n.entries;
        b.full_entries += n.full;
        b.rejected += n.rejected;
    }
}

TileBins bin_triangles(const SetupResult& setup, uint32_t tiles_x, uint32_t tiles_y) {
    TileBins b;
    bin_triangles(setup, tiles_x, tiles_y, b);
    return b;
}

}  // namespace ttr
