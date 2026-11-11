// SPDX-License-Identifier: Apache-2.0
//
// Multiply-add micro-benchmark compute kernel (define MAD_CHAINS). Outputs one timing tile.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "ckernel_sfpu_mad_bench.h"
#endif

void kernel_main() {
    const uint32_t iters = get_arg_val<uint32_t>(0);
    constexpr auto cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_out, cb_out);

    tile_regs_acquire();
    uint64_t t0 = 0, t1 = 0;
    MATH(t0 = ckernel::read_wall_clock();)
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, mad_bench, 0, VectorMode::RC_custom, iters));
    MATH(ckernel::tensix_sync(); t1 = ckernel::read_wall_clock();)
    uint32_t cycles = static_cast<uint32_t>(t1 - t0);
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, mad_write_timing, 0, VectorMode::RC_custom, cycles & 0xFFFFu, cycles >> 16));
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(1, cb_out);  // timing tile
    cb_push_back(cb_out, 1);
    tile_regs_release();
}
