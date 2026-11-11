// SPDX-License-Identifier: Apache-2.0

#include "ttraster/reference.h"

#include <algorithm>
#include <cmath>

namespace ttr {

Framebuffer render_reference(
    const Frame& frame, const SetupResult& setup, const TileBins& bins, const ReferenceOptions& opt,
    ReferenceStats* stats) {
    Framebuffer fb;
    fb.width = frame.padded_width();
    fb.height = frame.padded_height();
    const size_t n = static_cast<size_t>(fb.width) * fb.height;
    fb.color.assign(n, pack_rgba8(frame.clear_color));
    fb.depth.assign(n, frame.clear_depth);
    fb.id.assign(n, 0);
    fb.cover.assign(n, 0);
    fb.ambiguous.assign(n, 0);
    if (opt.float_planes) {
        for (int c = 0; c < 4; c++) {
            fb.plane[c].assign(n, frame.clear_color[c]);
        }
    }

    ReferenceStats local;

    // Blending reads the colour already in the target. The device keeps unquantized FP32 colour planes while it
    // draws and quantizes at the end, so a frame with blended triangles keeps the same floats here.
    bool any_blend = false;
    for (const SetupTriangle& t : setup.tris) any_blend |= t.blend != BlendMode::None;
    std::vector<float> cf;
    if (any_blend) {
        cf.resize(n * 4);
        for (size_t i = 0; i < n; i++) {
            for (int ch = 0; ch < 4; ch++) cf[4 * i + ch] = frame.clear_color[ch];
        }
    }

    for (uint32_t ty = 0; ty < bins.tiles_y; ty++) {
        for (uint32_t tx = 0; tx < bins.tiles_x; tx++) {
            for (const BinEntry& entry : bins.tile(tx, ty)) {
                const SetupTriangle& tri = setup.tris[entry.tri];
                const TileParams p = make_tile_params(tri, static_cast<int>(tx), static_cast<int>(ty));
                const bool skip_tests = opt.trust_full_tiles && entry.coverage == TileCoverage::Full;
                const DepthFunc df = tri.depth_func;
                local.tile_triangle_pairs++;
                local.full_pairs += entry.coverage == TileCoverage::Full;

                for (int j = 0; j < kTileSize; j++) {
                    const float ddy = tile_offset(j);
                    const size_t row = fb.index(tx * kTileSize, ty * kTileSize + j);
                    for (int i = 0; i < kTileSize; i++) {
                        const float ddx = tile_offset(i);
                        local.pixels_evaluated++;
                        if (!skip_tests && !(edge_inside(p.edge[0], ddx, ddy) && edge_inside(p.edge[1], ddx, ddy) &&
                                             edge_inside(p.edge[2], ddx, ddy))) {
                            continue;
                        }
                        const size_t idx = row + i;
                        fb.cover[idx] = static_cast<uint8_t>(std::min(255, fb.cover[idx] + 1));

                        const float z = std::fmaf(p.depth_gx, ddx, std::fmaf(p.depth_gy, ddy, p.depth_c));
                        const float zo = fb.depth[idx];
                        if (df != DepthFunc::Always) {
                            const float diff = std::fabs(z - zo);
                            if (diff <= kDepthAmbiguity * std::max(std::fabs(z), std::fabs(zo))) {
                                fb.ambiguous[idx] = 1;
                            }
                            if (df == DepthFunc::Less ? !(z < zo) : !(z <= zo)) {
                                continue;
                            }
                        }
                        float rgba[4];
                        for (int ch = 0; ch < 4; ch++) {
                            rgba[ch] = std::fmaf(p.color_gx[ch], ddx, std::fmaf(p.color_gy[ch], ddy, p.color_c[ch]));
                        }
                        if (p.perspective) {
                            // colour * rhw and rhw were interpolated: divide (the device multiplies by a
                            // Newton-refined reciprocal, which can differ in the last float bit)
                            const float q = 1.0f / std::fmaf(p.w_gx, ddx, std::fmaf(p.w_gy, ddy, p.w_c));
                            for (int ch = 0; ch < 4; ch++) {
                                rgba[ch] *= q;
                            }
                        }
                        if (tri.blend != BlendMode::None) {
                            const float sa = clamp01(rgba[3]);
                            float out[4];
                            for (int ch = 0; ch < 4; ch++) {
                                out[ch] = blend_channel(tri.blend, clamp01(rgba[ch]), sa, cf[4 * idx + ch]);
                            }
                            for (int ch = 0; ch < 4; ch++) rgba[ch] = out[ch];
                        }
                        if (any_blend) {
                            for (int ch = 0; ch < 4; ch++) cf[4 * idx + ch] = rgba[ch];
                        }
                        if (tri.depth_write) {
                            fb.depth[idx] = z;
                        }
                        if (opt.float_planes) {
                            for (int ch = 0; ch < 4; ch++) {
                                fb.plane[ch][idx] = rgba[ch];
                            }
                        }
                        fb.color[idx] = pack_rgba8(rgba);
                        fb.id[idx] = tri.source + 1;
                        local.fragments_written++;
                    }
                }
            }
        }
    }
    if (stats != nullptr) {
        *stats = local;
    }
    return fb;
}

Framebuffer render_reference(const Frame& frame, const ReferenceOptions& opt, ReferenceStats* stats) {
    const SetupResult setup = setup_frame(frame);
    const TileBins bins = bin_triangles(setup, frame.tiles_x(), frame.tiles_y());
    return render_reference(frame, setup, bins, opt, stats);
}

}  // namespace ttr
