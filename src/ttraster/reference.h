// SPDX-License-Identifier: Apache-2.0
//
// CPU reference rasterizer (docs/PLAN.md, Phase 1.3) and its framebuffer.
//
// It consumes the same setup and binning output as the device path and evaluates every pixel
// with the same tile-relative FP32 contract (tile.h), so a difference against the device points
// at the device. Coverage (edge tests) is exact; depth and colour use std::fmaf, which differs
// from the device's partially fused multiply-add by at most 1 ulp (docs/FINDINGS.md, V10).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ttraster/frame.h"
#include "ttraster/setup.h"
#include "ttraster/tile.h"

namespace ttr {

// Relative depth difference below which a depth comparison is flagged as ambiguous: two
// surfaces that close can swap order under a 1 ulp change in the evaluation.
inline constexpr float kDepthAmbiguity = 1.0e-6f;

struct Framebuffer {
    uint32_t width = 0, height = 0;     // padded target size
    std::vector<uint32_t> color;        // packed RGBA8 (pack_rgba8)
    std::vector<float> depth;
    std::vector<uint32_t> id;           // 1 + source triangle index of the visible fragment, 0 = clear
    std::vector<uint8_t> cover;         // number of triangles whose edge tests covered the pixel (saturates at 255)
    std::vector<uint8_t> ambiguous;     // 1 if a depth comparison at this pixel was within kDepthAmbiguity

    size_t index(uint32_t x, uint32_t y) const { return static_cast<size_t>(y) * width + x; }
};

struct ReferenceOptions {
    // Skip the edge tests for tiles classified Full. Turning this off evaluates every pixel and
    // must give identical output; the tests use it to check the classification.
    bool trust_full_tiles = true;
};

struct ReferenceStats {
    uint64_t tile_triangle_pairs = 0;
    uint64_t full_pairs = 0;
    uint64_t pixels_evaluated = 0;
    uint64_t fragments_written = 0;
};

Framebuffer render_reference(
    const Frame& frame, const SetupResult& setup, const TileBins& bins, const ReferenceOptions& opt = {},
    ReferenceStats* stats = nullptr);

// Convenience: setup, bin and render.
Framebuffer render_reference(const Frame& frame, const ReferenceOptions& opt = {}, ReferenceStats* stats = nullptr);

}  // namespace ttr
