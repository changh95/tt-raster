// SPDX-License-Identifier: Apache-2.0
//
// Spike A compute kernel: two FP32 coordinate tiles in, three FP32 tiles out
//   tile 0: checkerboard + gradient f(dx, dy)
//   tile 1: lane probe (which face / vector / lane constant produced each position)
//   tile 2: x * 0.1 + y * 0.3 (FP32 arithmetic check)

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "ckernel_sfpu_spike_a.h"
#endif

void kernel_main() {
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);

    cb_wait_front(cb_in, 2);  // tile 0 = dx, tile 1 = dy

    // Pass 1: checkerboard and lane probe. Dest: 0 = dx, 1 = dy, 2 = f, 3 = probe.
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    copy_tile(cb_in, 1, 1);
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, spike_a_checker_face, 0, 1, 2, VectorMode::RC));
    MATH(ckernel::sfpu::spike_a_face = 0);
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, spike_a_probe_face, 3, VectorMode::RC));
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 2);
    pack_tile(2, cb_out);
    pack_tile(3, cb_out);
    cb_push_back(cb_out, 2);
    tile_regs_release();

    // Pass 2: FP32 arithmetic check. Dest: 0 = dx, 1 = dy, 2 = result.
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    copy_tile(cb_in, 1, 1);
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, spike_a_math_face, 0, 1, 2, VectorMode::RC));
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(2, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();

    cb_pop_front(cb_in, 2);
}
