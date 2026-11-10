// SPDX-License-Identifier: Apache-2.0
//
// Exact int64 coverage oracle (docs/PLAN.md, Phase 1.3). With vertices snapped to 1/16 pixel
// every edge function is an integer at a pixel centre, so coverage is decided without any
// floating point. The oracle shares only the snapping stage with the renderer; it has no tiles,
// no tile-relative FP32 evaluation and no precomputed edge constants.

#pragma once

#include <cstdint>
#include <vector>

#include "ttraster/frame.h"
#include "ttraster/setup.h"

namespace ttr {

// Coverage count per pixel of the padded target, from the post-setup triangles (snapped and, where
// needed, clipped). The reference's `cover` must equal this exactly.
std::vector<uint8_t> oracle_cover_from_setup(const SetupResult& setup, uint32_t padded_width, uint32_t padded_height);

struct InputOracle {
    std::vector<uint8_t> cover;      // coverage count from the snapped input triangles, no clipping
    std::vector<uint8_t> clip_mask;  // 1 where a triangle that needed guard-band clipping may touch the pixel
};

// Coverage straight from the input triangles. Pixels flagged in clip_mask can differ from the
// renderer by the (sub-1/16 pixel) change that clipping makes to an edge; everywhere else the two
// must agree exactly.
InputOracle oracle_cover_from_input(const Frame& frame);

}  // namespace ttr
