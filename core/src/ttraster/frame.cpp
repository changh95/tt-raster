// SPDX-License-Identifier: Apache-2.0

#include "ttraster/frame.h"

#include <cmath>

namespace ttr {

uint8_t quantize_unorm8(float v) {
    if (!(v > 0.0f)) {  // also catches NaN
        return 0;
    }
    if (v >= 1.0f) {
        return 255;
    }
    return static_cast<uint8_t>(std::floor(std::fmaf(v, 255.0f, 0.5f)));
}

uint32_t pack_rgba8(const float rgba[4]) {
    return static_cast<uint32_t>(quantize_unorm8(rgba[0])) | (static_cast<uint32_t>(quantize_unorm8(rgba[1])) << 8) |
           (static_cast<uint32_t>(quantize_unorm8(rgba[2])) << 16) |
           (static_cast<uint32_t>(quantize_unorm8(rgba[3])) << 24);
}

}  // namespace ttr
