// SPDX-License-Identifier: Apache-2.0
//
// Triangle setup (docs/PLAN.md, Phase 1.2). This is the shared module: the CPU reference, the
// oracle and (from Phase 2) the host side of the Tensix renderer all start from its output.
//
// Pipeline per input triangle:
//   1. validate and snap vertices to the 1/16 pixel grid (round to nearest even);
//   2. reject degenerate triangles (zero area after snapping) and cull by winding;
//   3. orient so the signed area is positive, build three integer edge functions with a
//      top-left fill rule folded into the edge constant;
//   4. clip to a guard band if any vertex lies outside it, so every edge coefficient stays
//      within 2^15 (the FP32 exactness bound), and fan the clipped polygon into triangles;
//   5. fit screen-linear planes for depth and colour from the original snapped vertices;
//   6. compute the tile bounding box.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "ttraster/frame.h"

namespace ttr {

enum class SetupStatus : uint8_t { Ok, NonFinite, OutOfRange, Degenerate, Culled, OffScreen };

// First stage on its own: validation, snapping and area. Shared by setup and the oracle.
struct SnappedTriangle {
    int32_t x[3];     // subpixel (1/16 px) coordinates, in input order
    int32_t y[3];
    int64_t area2;    // signed_area2 in subpixel^2 units, in input order
};
SetupStatus snap_triangle(const Triangle& tri, CullMode cull, SnappedTriangle& out);

// Edge function E(P) = a*Px + b*Py + c over integer subpixel coordinates P, positive inside.
// With top_left false the edge itself is excluded: a pixel is inside iff E - 1 >= 0 (E is an
// integer at pixel centres), so every edge is tested with the same ">= 0".
struct Edge {
    int64_t a, b, c;
    bool top_left;
};

// value(px, py) = c + gx*px + gy*py with px, py in pixels (double precision, setup time only).
struct Plane {
    double c = 0.0, gx = 0.0, gy = 0.0;
};

struct SetupTriangle {
    uint32_t source = 0;   // index into Frame::triangles
    uint32_t piece = 0;    // fan piece within the source triangle (0 unless clipped)
    bool clipped = false;
    bool flat = false;
    int32_t x[3] = {}, y[3] = {};  // snapped, oriented so signed area is positive
    Edge edge[3] = {};             // edge i runs from vertex i to vertex (i+1)%3
    Plane depth;
    Plane color[4];
    int32_t tile_x0 = 0, tile_y0 = 0, tile_x1 = 0, tile_y1 = 0;  // inclusive, inside the target
};

struct SetupStats {
    uint64_t input = 0;
    uint64_t accepted = 0;       // input triangles that produced at least one piece
    uint64_t pieces = 0;         // SetupTriangles produced
    uint64_t clipped_sources = 0;
    uint64_t non_finite = 0, out_of_range = 0, degenerate = 0, culled = 0, off_screen = 0;
};

struct SetupResult {
    std::vector<SetupTriangle> tris;
    SetupStats stats;
};

// Axis-aligned clip rectangle in subpixel units (inclusive bounds).
struct ClipRect {
    int64_t x0, y0, x1, y1;
};
// Viewport plus a guard band, sized so the rectangle is exactly kMaxTargetSize pixels wide and
// high (or the padded target itself if that is already larger).
ClipRect guard_rect(uint32_t padded_width, uint32_t padded_height);

// Clip a convex polygon (subpixel integer vertices) against `rect`. Intersections depend only
// on the two endpoints of the clipped edge, not on their order, so two triangles sharing an
// edge clip it to the same points and stay watertight.
std::vector<std::array<int64_t, 2>> clip_polygon(std::vector<std::array<int64_t, 2>> poly, const ClipRect& rect);

// Whole-frame setup. Throws std::invalid_argument if the padded target exceeds kMaxTargetSize.
SetupResult setup_frame(const Frame& frame);

}  // namespace ttr
