// SPDX-License-Identifier: Apache-2.0

#include "ttraster/setup.h"

#include "ttraster/parallel.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace ttr {

namespace {

using Pt = std::array<int64_t, 2>;

int64_t floor_div(int64_t a, int64_t b) {  // b > 0
    int64_t q = a / b;
    if ((a % b) != 0 && a < 0) {
        --q;
    }
    return q;
}

int64_t ceil_div(int64_t a, int64_t b) { return -floor_div(-a, b); }

int64_t cross2(int64_t x0, int64_t y0, int64_t x1, int64_t y1, int64_t x2, int64_t y2) {
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
}

// Intersection of segment p-q with the vertical line x = xb (or horizontal y = yb when `vertical`
// is false). The two endpoints are put in lexicographic order first, so the result is the same
// whichever way the edge is walked.
Pt intersect(Pt p, Pt q, int64_t bound, bool vertical) {
    if (q < p) {
        std::swap(p, q);
    }
    const int u = vertical ? 0 : 1;  // coordinate that equals `bound` on the line
    const int v = 1 - u;
    const double t = static_cast<double>(bound - p[u]) / static_cast<double>(q[u] - p[u]);
    const double other = static_cast<double>(p[v]) + t * static_cast<double>(q[v] - p[v]);
    Pt r{};
    r[u] = bound;
    r[v] = std::llround(other);
    return r;
}

void make_edges(const int32_t x[3], const int32_t y[3], Edge e[3]) {
    for (int i = 0; i < 3; i++) {
        const int j = (i + 1) % 3;
        e[i].a = static_cast<int64_t>(y[i]) - y[j];
        e[i].b = static_cast<int64_t>(x[j]) - x[i];
        e[i].c = -e[i].a * x[i] - e[i].b * y[i];
        e[i].top_left = e[i].a > 0 || (e[i].a == 0 && e[i].b > 0);
    }
}

// Fit value(px, py) = c + gx*px + gy*py through three snapped vertices (pixel units, exact).
Plane fit_plane(const SnappedTriangle& s, double a0, double a1, double a2) {
    const double x0 = s.x[0] / 16.0, y0 = s.y[0] / 16.0;
    const double dx1 = s.x[1] / 16.0 - x0, dy1 = s.y[1] / 16.0 - y0;
    const double dx2 = s.x[2] / 16.0 - x0, dy2 = s.y[2] / 16.0 - y0;
    const double det = dx1 * dy2 - dy1 * dx2;  // = area2 / 256, exact
    const double da1 = a1 - a0, da2 = a2 - a0;
    Plane p;
    p.gx = (da1 * dy2 - da2 * dy1) / det;
    p.gy = (dx1 * da2 - dx2 * da1) / det;
    p.c = a0 - p.gx * x0 - p.gy * y0;
    return p;
}

// Tile bounding box of the pixels whose centre can lie inside the vertex bounding box.
// Returns false if no pixel centre does.
bool tile_bbox(const int32_t x[3], const int32_t y[3], int64_t lo_x, int64_t lo_y, int64_t hi_x, int64_t hi_y, SetupTriangle& t) {
    const int64_t min_x = std::min({x[0], x[1], x[2]}), max_x = std::max({x[0], x[1], x[2]});
    const int64_t min_y = std::min({y[0], y[1], y[2]}), max_y = std::max({y[0], y[1], y[2]});
    // Pixel i has its centre at 16*i + 8 in subpixel units.
    int64_t px0 = ceil_div(min_x - 8, 16), px1 = floor_div(max_x - 8, 16);
    int64_t py0 = ceil_div(min_y - 8, 16), py1 = floor_div(max_y - 8, 16);
    px0 = std::max<int64_t>(px0, lo_x);
    py0 = std::max<int64_t>(py0, lo_y);
    px1 = std::min<int64_t>(px1, hi_x);
    py1 = std::min<int64_t>(py1, hi_y);
    if (px0 > px1 || py0 > py1) {
        return false;
    }
    t.tile_x0 = static_cast<int32_t>(px0 / kTileSize);
    t.tile_x1 = static_cast<int32_t>(px1 / kTileSize);
    t.tile_y0 = static_cast<int32_t>(py0 / kTileSize);
    t.tile_y1 = static_cast<int32_t>(py1 / kTileSize);
    return true;
}

}  // namespace

SetupStatus snap_triangle(const Triangle& tri, CullMode cull, SnappedTriangle& out) {
    for (const Vertex& v : tri.v) {
        if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
            return SetupStatus::NonFinite;
        }
        if (std::fabs(v.x) > kMaxInputCoord || std::fabs(v.y) > kMaxInputCoord) {
            return SetupStatus::OutOfRange;
        }
    }
    for (int i = 0; i < 3; i++) {
        // x * 16 is exact in FP32, nearbyint rounds ties to even in the default rounding mode.
        out.x[i] = static_cast<int32_t>(std::nearbyint(tri.v[i].x * static_cast<float>(kSubpixelScale)));
        out.y[i] = static_cast<int32_t>(std::nearbyint(tri.v[i].y * static_cast<float>(kSubpixelScale)));
    }
    out.area2 = cross2(out.x[0], out.y[0], out.x[1], out.y[1], out.x[2], out.y[2]);
    if (out.area2 == 0) {
        return SetupStatus::Degenerate;
    }
    if ((cull == CullMode::Positive && out.area2 > 0) || (cull == CullMode::Negative && out.area2 < 0)) {
        return SetupStatus::Culled;
    }
    return SetupStatus::Ok;
}

ClipRect guard_rect(uint32_t padded_width, uint32_t padded_height) {
    const int64_t gx = std::max<int64_t>(0, (kMaxTargetSize - static_cast<int64_t>(padded_width)) / 2);
    const int64_t gy = std::max<int64_t>(0, (kMaxTargetSize - static_cast<int64_t>(padded_height)) / 2);
    return ClipRect{
        -gx * kSubpixelScale,
        -gy * kSubpixelScale,
        (static_cast<int64_t>(padded_width) + gx) * kSubpixelScale,
        (static_cast<int64_t>(padded_height) + gy) * kSubpixelScale};
}

std::vector<Pt> clip_polygon(std::vector<Pt> poly, const ClipRect& r) {
    struct Bound {
        bool vertical;   // line x = value (true) or y = value (false)
        bool keep_ge;    // keep coordinates >= value (true) or <= value (false)
        int64_t value;
    };
    const Bound bounds[4] = {
        {true, true, r.x0}, {true, false, r.x1}, {false, true, r.y0}, {false, false, r.y1}};
    for (const Bound& b : bounds) {
        if (poly.empty()) {
            break;
        }
        const int u = b.vertical ? 0 : 1;
        auto inside = [&](const Pt& p) { return b.keep_ge ? p[u] >= b.value : p[u] <= b.value; };
        std::vector<Pt> out;
        for (size_t i = 0; i < poly.size(); i++) {
            const Pt& cur = poly[i];
            const Pt& nxt = poly[(i + 1) % poly.size()];
            const bool ci = inside(cur), ni = inside(nxt);
            if (ci && ni) {
                out.push_back(nxt);
            } else if (ci && !ni) {
                out.push_back(intersect(cur, nxt, b.value, b.vertical));
            } else if (!ci && ni) {
                out.push_back(intersect(cur, nxt, b.value, b.vertical));
                out.push_back(nxt);
            }
        }
        poly = std::move(out);
    }
    // Drop consecutive duplicates, including across the wrap-around.
    std::vector<Pt> clean;
    for (const Pt& p : poly) {
        if (clean.empty() || clean.back() != p) {
            clean.push_back(p);
        }
    }
    while (clean.size() > 1 && clean.front() == clean.back()) {
        clean.pop_back();
    }
    return clean;
}

void setup_frame(const Frame& frame, SetupResult& res, SetupWorkspace& ws) {
    const uint32_t pw = frame.padded_width(), ph = frame.padded_height();
    const bool has_viewports = !frame.viewports.empty();
    ClipRect frame_guard{};
    if (!has_viewports) {
        if (pw > static_cast<uint32_t>(kMaxTargetSize) || ph > static_cast<uint32_t>(kMaxTargetSize)) {
            throw std::invalid_argument("render target larger than kMaxTargetSize");
        }
        frame_guard = guard_rect(pw, ph);
    }
    for (const Viewport& vp : frame.viewports) {
        if (vp.x < 0 || vp.y < 0 || vp.width <= 0 || vp.height <= 0 || vp.width > kMaxTargetSize ||
            vp.height > kMaxTargetSize || static_cast<uint32_t>(vp.x + vp.width) > pw ||
            static_cast<uint32_t>(vp.y + vp.height) > ph) {
            throw std::invalid_argument("viewport outside the render target or larger than kMaxTargetSize");
        }
    }

    res.stats = SetupStats{};
    res.stats.input = frame.triangles.size();
    auto process = [&](uint32_t src, std::vector<SetupTriangle>& tris, SetupStats& stats) {
        const Triangle& tri = frame.triangles[src];
        SnappedTriangle s;
        switch (snap_triangle(tri, frame.cull, s)) {
            case SetupStatus::Ok: break;
            case SetupStatus::NonFinite: stats.non_finite++; return;
            case SetupStatus::OutOfRange: stats.out_of_range++; return;
            case SetupStatus::Degenerate: stats.degenerate++; return;
            case SetupStatus::Culled: stats.culled++; return;
            case SetupStatus::OffScreen: stats.off_screen++; return;
        }

        // Planes come from the original (unclipped) snapped vertices and are shared by all pieces.
        Plane depth = fit_plane(s, tri.v[0].z, tri.v[1].z, tri.v[2].z);
        Plane color[4];
        for (int ch = 0; ch < 4; ch++) {
            color[ch] = fit_plane(s, tri.v[0].color[ch], tri.v[1].color[ch], tri.v[2].color[ch]);
        }
        if (tri.flat) {
            for (int ch = 0; ch < 4; ch++) {
                color[ch] = Plane{tri.v[0].color[ch], 0.0, 0.0};
            }
        }

        // Orient so the signed area is positive: swap vertices 1 and 2 of the geometry.
        std::array<Pt, 3> g = {Pt{s.x[0], s.y[0]}, Pt{s.x[1], s.y[1]}, Pt{s.x[2], s.y[2]}};
        if (s.area2 < 0) {
            std::swap(g[1], g[2]);
        }

        // Clip rectangle and pixel range of this triangle's viewport (or of the whole target).
        ClipRect guard = frame_guard;
        int64_t lo_x = 0, lo_y = 0, hi_x = static_cast<int64_t>(pw) - 1, hi_y = static_cast<int64_t>(ph) - 1;
        if (has_viewports) {
            if (tri.viewport >= frame.viewports.size()) {
                throw std::invalid_argument("triangle viewport index out of range");
            }
            const Viewport& vp = frame.viewports[tri.viewport];
            guard = ClipRect{
                static_cast<int64_t>(vp.x) * kSubpixelScale, static_cast<int64_t>(vp.y) * kSubpixelScale,
                static_cast<int64_t>(vp.x + vp.width) * kSubpixelScale, static_cast<int64_t>(vp.y + vp.height) * kSubpixelScale};
            lo_x = vp.x;
            lo_y = vp.y;
            hi_x = vp.x + vp.width - 1;
            hi_y = vp.y + vp.height - 1;
        }
        bool outside_guard = false;
        for (const Pt& p : g) {
            outside_guard |= p[0] < guard.x0 || p[0] > guard.x1 || p[1] < guard.y0 || p[1] > guard.y1;
        }

        // Reused per thread: a heap allocation per input triangle would dominate this stage.
        thread_local std::vector<std::array<Pt, 3>> pieces;
        pieces.clear();
        if (!outside_guard) {
            pieces.push_back(g);
        } else {
            std::vector<Pt> poly = clip_polygon({g[0], g[1], g[2]}, guard);
            for (size_t k = 1; k + 1 < poly.size(); k++) {
                pieces.push_back({poly[0], poly[k], poly[k + 1]});
            }
        }

        uint32_t made = 0;
        for (size_t k = 0; k < pieces.size(); k++) {
            SetupTriangle t;
            t.source = src;
            t.piece = static_cast<uint32_t>(k);
            t.clipped = outside_guard;
            t.flat = tri.flat;
            for (int i = 0; i < 3; i++) {
                t.x[i] = static_cast<int32_t>(pieces[k][i][0]);
                t.y[i] = static_cast<int32_t>(pieces[k][i][1]);
            }
            if (cross2(t.x[0], t.y[0], t.x[1], t.y[1], t.x[2], t.y[2]) <= 0) {
                continue;  // sliver left over by rounding the clipped vertices
            }
            if (!tile_bbox(t.x, t.y, lo_x, lo_y, hi_x, hi_y, t)) {
                continue;
            }
            make_edges(t.x, t.y, t.edge);
            t.depth = depth;
            for (int ch = 0; ch < 4; ch++) {
                t.color[ch] = color[ch];
            }
            tris.push_back(t);
            made++;
        }
        if (made == 0) {
            stats.off_screen++;
            return;
        }
        stats.accepted++;
        stats.pieces += made;
        stats.clipped_sources += outside_guard ? 1 : 0;
        };

    // Triangles are independent: set them up in chunks on several threads and concatenate the chunks in
    // order, so the result is identical to the sequential run (API order is preserved).
    const size_t n = frame.triangles.size();
    constexpr size_t kChunk = 1024;
    const size_t n_chunks = (n + kChunk - 1) / kChunk;
    const size_t hw = std::max<size_t>(1, std::thread::hardware_concurrency());
    if (n < 4 * kChunk || hw == 1) {
        res.tris.clear();
        for (uint32_t src = 0; src < n; src++) process(src, res.tris, res.stats);
        return;
    }
    if (ws.parts.size() < n_chunks) ws.parts.resize(n_chunks);
    ws.part_stats.assign(n_chunks, SetupStats{});
    for (size_t c = 0; c < n_chunks; c++) ws.parts[c].clear();
    parallel_for(n_chunks, hw, [&](size_t c) {
        for (size_t src = c * kChunk; src < std::min(n, (c + 1) * kChunk); src++) {
            process(static_cast<uint32_t>(src), ws.parts[c], ws.part_stats[c]);
        }
    });

    // Concatenate in parallel into the persistent output. Elements are only default-constructed when
    // the frame has more triangles than any earlier one, so a steady-state loop does no page faults here.
    ws.offset.assign(n_chunks + 1, 0);
    for (size_t c = 0; c < n_chunks; c++) {
        ws.offset[c + 1] = ws.offset[c] + ws.parts[c].size();
        const SetupStats& st = ws.part_stats[c];
        res.stats.accepted += st.accepted; res.stats.pieces += st.pieces; res.stats.clipped_sources += st.clipped_sources;
        res.stats.non_finite += st.non_finite; res.stats.out_of_range += st.out_of_range; res.stats.degenerate += st.degenerate;
        res.stats.culled += st.culled; res.stats.off_screen += st.off_screen;
    }
    // (No clear() before this resize: the old elements stay and are overwritten by the copy below.)
    res.tris.resize(ws.offset[n_chunks]);
    parallel_for(n_chunks, hw, [&](size_t c) {
        std::copy(ws.parts[c].begin(), ws.parts[c].end(), res.tris.begin() + static_cast<std::ptrdiff_t>(ws.offset[c]));
    });
}

SetupResult setup_frame(const Frame& frame) {
    SetupResult res;
    SetupWorkspace ws;
    setup_frame(frame, res, ws);
    return res;
}

}  // namespace ttr
