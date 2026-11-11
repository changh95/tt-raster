// SPDX-License-Identifier: Apache-2.0

#include "ttraster/oracle.h"

#include <algorithm>

namespace ttr {

namespace {

int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b) != 0 && a < 0) {
        --q;
    }
    return q;
}

// Add one to the coverage of every pixel whose centre is inside the triangle (x, y positively
// oriented, subpixel units). Plain definition: E_i(P) = (y_i - y_j)*(Px - x_i) + (x_j - x_i)*(Py - y_i),
// inside iff E > 0, or E == 0 on a top or left edge.
void cover_triangle(
    const int64_t x[3], const int64_t y[3], uint32_t pw, uint32_t ph, std::vector<uint8_t>& cover, int64_t lo_x = 0,
    int64_t lo_y = 0, int64_t hi_x = -1, int64_t hi_y = -1) {
    if (hi_x < 0) hi_x = static_cast<int64_t>(pw) - 1;
    if (hi_y < 0) hi_y = static_cast<int64_t>(ph) - 1;
    const int64_t min_x = std::min({x[0], x[1], x[2]}), max_x = std::max({x[0], x[1], x[2]});
    const int64_t min_y = std::min({y[0], y[1], y[2]}), max_y = std::max({y[0], y[1], y[2]});
    // Smallest pixel whose centre 16*p+8 is >= min, largest whose centre is <= max.
    int64_t px0 = -floor_div(-(min_x - 8), 16), px1 = floor_div(max_x - 8, 16);
    int64_t py0 = -floor_div(-(min_y - 8), 16), py1 = floor_div(max_y - 8, 16);
    px0 = std::max<int64_t>(px0, lo_x);
    py0 = std::max<int64_t>(py0, lo_y);
    px1 = std::min<int64_t>(px1, hi_x);
    py1 = std::min<int64_t>(py1, hi_y);

    for (int64_t py = py0; py <= py1; py++) {
        const int64_t Py = 16 * py + 8;
        for (int64_t px = px0; px <= px1; px++) {
            const int64_t Px = 16 * px + 8;
            bool inside = true;
            for (int i = 0; i < 3 && inside; i++) {
                const int j = (i + 1) % 3;
                const int64_t a = y[i] - y[j];
                const int64_t b = x[j] - x[i];
                const int64_t e = a * (Px - x[i]) + b * (Py - y[i]);
                const bool top_left = a > 0 || (a == 0 && b > 0);
                inside = e > 0 || (e == 0 && top_left);
            }
            if (inside) {
                uint8_t& c = cover[static_cast<size_t>(py) * pw + px];
                c = static_cast<uint8_t>(std::min(255, c + 1));
            }
        }
    }
}

}  // namespace

std::vector<uint8_t> oracle_cover_from_setup(const SetupResult& setup, uint32_t pw, uint32_t ph) {
    std::vector<uint8_t> cover(static_cast<size_t>(pw) * ph, 0);
    for (const SetupTriangle& t : setup.tris) {
        const int64_t x[3] = {t.x[0], t.x[1], t.x[2]};
        const int64_t y[3] = {t.y[0], t.y[1], t.y[2]};
        cover_triangle(x, y, pw, ph, cover);
    }
    return cover;
}

InputOracle oracle_cover_from_input(const Frame& frame) {
    const uint32_t pw = frame.padded_width(), ph = frame.padded_height();
    const ClipRect frame_guard = frame.viewports.empty() ? guard_rect(pw, ph) : ClipRect{};
    InputOracle out;
    out.cover.assign(static_cast<size_t>(pw) * ph, 0);
    out.clip_mask.assign(static_cast<size_t>(pw) * ph, 0);

    for (const Triangle& tri : frame.triangles) {
        SnappedTriangle s;
        if (snap_triangle(tri, frame.cull, s) != SetupStatus::Ok) {
            continue;
        }
        int64_t x[3] = {s.x[0], s.x[1], s.x[2]};
        int64_t y[3] = {s.y[0], s.y[1], s.y[2]};
        if (s.area2 < 0) {
            std::swap(x[1], x[2]);
            std::swap(y[1], y[2]);
        }
        ClipRect guard = frame_guard;
        int64_t lo_x = 0, lo_y = 0, hi_x = static_cast<int64_t>(pw) - 1, hi_y = static_cast<int64_t>(ph) - 1;
        if (!frame.viewports.empty()) {
            const Viewport& vp = frame.viewports[tri.viewport];
            guard = ClipRect{static_cast<int64_t>(vp.x) * 16, static_cast<int64_t>(vp.y) * 16,
                             static_cast<int64_t>(vp.x + vp.width) * 16, static_cast<int64_t>(vp.y + vp.height) * 16};
            lo_x = vp.x; lo_y = vp.y; hi_x = vp.x + vp.width - 1; hi_y = vp.y + vp.height - 1;
        }
        bool outside = false;
        for (int i = 0; i < 3; i++) {
            outside |= x[i] < guard.x0 || x[i] > guard.x1 || y[i] < guard.y0 || y[i] > guard.y1;
        }
        if (outside) {
            // Flag every pixel of the (clamped) bounding box as possibly affected by clipping.
            const int64_t px0 = std::max<int64_t>(0, floor_div(std::min({x[0], x[1], x[2]}) - 8, 16));
            const int64_t px1 = std::min<int64_t>(pw - 1, floor_div(std::max({x[0], x[1], x[2]}) - 8, 16) + 1);
            const int64_t py0 = std::max<int64_t>(0, floor_div(std::min({y[0], y[1], y[2]}) - 8, 16));
            const int64_t py1 = std::min<int64_t>(ph - 1, floor_div(std::max({y[0], y[1], y[2]}) - 8, 16) + 1);
            for (int64_t py = py0; py <= py1; py++) {
                for (int64_t px = px0; px <= px1; px++) {
                    out.clip_mask[static_cast<size_t>(py) * pw + px] = 1;
                }
            }
        }
        cover_triangle(x, y, pw, ph, out.cover, lo_x, lo_y, hi_x, hi_y);
    }
    return out;
}

}  // namespace ttr
