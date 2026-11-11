// SPDX-License-Identifier: Apache-2.0
//
// Spike C compute kernel: hold n_tiles FP32 tiles in Dest at once, then pack all of them.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "ckernel_sfpu_spike_c.h"
#endif

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_out, cb_out);

    tile_regs_acquire();
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, spike_c_fill, 0, VectorMode::RC_custom, n_tiles));
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, n_tiles);
    for (uint32_t i = 0; i < n_tiles; i++) {
        pack_tile(i, cb_out);
    }
    cb_push_back(cb_out, n_tiles);
    tile_regs_release();
}
