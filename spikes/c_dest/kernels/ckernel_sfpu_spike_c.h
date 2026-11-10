// SPDX-License-Identifier: Apache-2.0
//
// Spike C: fill n_tiles Dest tiles with a signature, tile i vector v = (i + 1) * 100 + v, so the
// host can tell whether every tile slot is independent and intact after packing.

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel::sfpu {

inline void spike_c_fill(uint32_t n_tiles) {
    for (uint32_t i = 0; i < n_tiles; i++) {
        for (uint32_t v = 0; v < 32; v++) {
            sfpi::vInt code = static_cast<int>((i + 1) * 100 + v);
            sfpi::dst_reg[i * 32 + v] = sfpi::convert<sfpi::vFloat>(code, sfpi::RoundMode::Nearest);
        }
    }
}

}  // namespace ckernel::sfpu
