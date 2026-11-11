// SPDX-License-Identifier: Apache-2.0
//
// Multiply-add micro-benchmark: MAD_CHAINS (1, 2, 4 or 8) independent dependency chains of
// x = x*a + b, 8 steps per loop iteration. One chain measures MAD latency; many chains measure
// issue throughput. The chains are separate variables because SFPU register arrays cannot be
// spilled to memory.

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel::sfpu {

inline void mad_bench(uint32_t iters) {
    sfpi::vFloat a = 0.999f;
    sfpi::vFloat b = 0.001f;
    sfpi::vFloat x0 = 1.00f;
#if MAD_CHAINS >= 2
    sfpi::vFloat x1 = 1.25f;
#endif
#if MAD_CHAINS >= 4
    sfpi::vFloat x2 = 1.50f, x3 = 1.75f;
#endif
#if MAD_CHAINS >= 8
    sfpi::vFloat x4 = 2.00f, x5 = 2.25f, x6 = 2.50f, x7 = 2.75f;
#endif
    for (uint32_t it = 0; it < iters; it++) {
#pragma GCC unroll 8
        for (int rep = 0; rep < 8; rep++) {
            x0 = x0 * a + b;
#if MAD_CHAINS >= 2
            x1 = x1 * a + b;
#endif
#if MAD_CHAINS >= 4
            x2 = x2 * a + b;
            x3 = x3 * a + b;
#endif
#if MAD_CHAINS >= 8
            x4 = x4 * a + b;
            x5 = x5 * a + b;
            x6 = x6 * a + b;
            x7 = x7 * a + b;
#endif
        }
    }
    sfpi::vFloat sum = x0;
#if MAD_CHAINS >= 2
    sum += x1;
#endif
#if MAD_CHAINS >= 4
    sum += x2 + x3;
#endif
#if MAD_CHAINS >= 8
    sum += x4 + x5 + x6 + x7;
#endif
    sfpi::dst_reg[0] = sum;  // keep the work observable
}

inline void mad_write_timing(uint32_t lo16, uint32_t hi) {
    sfpi::vInt lo = static_cast<int>(lo16);
    sfpi::vInt hi_v = static_cast<int>(hi);
    sfpi::dst_reg[32 + 0] = sfpi::convert<sfpi::vFloat>(lo, sfpi::RoundMode::Nearest);
    sfpi::dst_reg[32 + 1] = sfpi::convert<sfpi::vFloat>(hi_v, sfpi::RoundMode::Nearest);
}

}  // namespace ckernel::sfpu
