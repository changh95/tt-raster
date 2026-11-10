// SPDX-License-Identifier: Apache-2.0
//
// Tile-relative evaluation and binning (docs/PLAN.md, Phases 1.2 and 1.3).
//
// A tile is 32x32 pixels. Every per-triangle quantity is expressed relative to the tile centre,
// so that the numbers a device would use are small:
//
//   pixel i (0..31) of a tile has its centre at offset  tile_offset(i) = 16*i - 248
//   subpixel units from the tile centre (exact in FP32).
//
// Edge value at a pixel:   E = fma(a, ddx, fma(b, ddy, c))      (two multiply-adds, c innermost)
//   a, b are the integer edge coefficients (|a|, |b| <= 2^15), c is the edge value at the tile
//   centre minus the top-left bias, computed exactly in int64 and rounded once to FP32.
//   Pixel inside the triangle  <=>  E >= 0 for all three edges.
//
//   Why this is exact in sign: the sign of E only matters when |E| is small, and then every
//   partial sum is bounded by the size of the term still to be added (<= 248 * 2^15 < 2^24), so
//   nothing is rounded. When |E| is large, a rounding error of a few units cannot change the
//   sign. The one real requirement is that c is representable when the edge crosses the tile:
//   |c| <= 248 * (|a| + |b|) < 2^24, which is where the 2^15 coefficient bound (hence the 2048
//   pixel target and guard band) comes from. tests/test_edge_exactness.cpp checks 40 million
//   pixels against int64, including 190,000 where E is exactly zero, and shows where each
//   evaluation order starts to fail beyond the bound. A device multiply-add that is only
//   partially fused (V10) gives the same signs because no intermediate needs rounding.
//
// Plane value at a pixel:  v = fma(gx, ddx, fma(gy, ddy, c))     (depth and each colour channel)
//   c is the plane value at the tile centre, gx and gy are gradients per subpixel unit.

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "ttraster/setup.h"

namespace ttr {

enum class TileCoverage : uint8_t {
    Reject,   // no pixel centre of the tile is inside the triangle
    Partial,  // some pixels may be inside: run the edge tests
    Full,     // every pixel of the tile is inside: skip the edge tests
};

struct TileEdge {
    float a, b, c;
};

struct TileParams {
    TileEdge edge[3];
    float depth_c, depth_gx, depth_gy;
    float color_c[4], color_gx[4], color_gy[4];
    TileCoverage coverage;
};

inline float tile_offset(int i) { return static_cast<float>(kSubpixelScale * i - (kTileSize * kSubpixelScale / 2 - kSubpixelScale / 2)); }

// Exact classification of tile (tx, ty) against a triangle using integer arithmetic.
TileCoverage classify_tile(const SetupTriangle& t, int tx, int ty);

TileParams make_tile_params(const SetupTriangle& t, int tx, int ty);

// One edge of make_tile_params, exposed so tests can drive the evaluation contract directly.
TileEdge make_tile_edge(const Edge& e, int tx, int ty);

// The per-pixel evaluation contract described above: two chained multiply-adds.
inline float edge_value(const TileEdge& e, float ddx, float ddy) {
    return std::fmaf(e.a, ddx, std::fmaf(e.b, ddy, e.c));
}
inline bool edge_inside(const TileEdge& e, float ddx, float ddy) { return edge_value(e, ddx, ddy) >= 0.0f; }

struct BinEntry {
    uint32_t tri;  // index into SetupResult::tris
    TileCoverage coverage;
};

struct TileBins {
    uint32_t tiles_x = 0, tiles_y = 0;
    std::vector<std::vector<BinEntry>> bins;  // tiles_x * tiles_y, row-major, API order inside a bin
    uint64_t entries = 0;        // (tile, triangle) pairs kept
    uint64_t full_entries = 0;   // of which fully covered tiles
    uint64_t rejected = 0;       // pairs inside a bounding box but rejected exactly

    const std::vector<BinEntry>& tile(uint32_t tx, uint32_t ty) const { return bins[ty * tiles_x + tx]; }
};

TileBins bin_triangles(const SetupResult& setup, uint32_t tiles_x, uint32_t tiles_y);

}  // namespace ttr
