// SPDX-License-Identifier: Apache-2.0
//
// Spike B compute kernel. See ckernel_sfpu_spike_b.h for the four approaches.
//
// Output tiles: acc (the masked accumulation) and a timing tile whose first two elements hold the
// math thread's cycle count for the timed loop (low 16 bits, high bits).
//
// Defines (from ComputeConfig): SPIKE_APPROACH, SPIKE_VEC, SPIKE_TPC (triangles per SFPU call).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "ckernel_sfpu_spike_b.h"
#endif

constexpr uint32_t kTrianglesPerBatch = 64;
constexpr uint32_t kCoefTiles = 8;  // 8 triangles x 4 scalars = 32 vectors per coefficient tile

void kernel_main() {
    const uint32_t k_batches = get_arg_val<uint32_t>(0);
    const uint32_t nv = get_arg_val<uint32_t>(1);  // vectors updated per triangle (1..32)

    constexpr auto cb_scal = tt::CBIndex::c_1;
    constexpr auto cb_coef = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_coef, cb_out);
    copy_init(cb_coef);

    cb_wait_front(cb_scal, 1);
    cb_wait_front(cb_coef, kCoefTiles);
    const uint32_t scal_addr = get_tile_address(cb_scal, 0);  // L1 byte address, same on all threads

    tile_regs_acquire();
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, spike_b_init_tiles, 0, VectorMode::RC_custom));

    uint64_t t0 = 0, t1 = 0;
    MATH(t0 = ckernel::read_wall_clock();)
    for (uint32_t k = 0; k < k_batches; k++) {
#if SPIKE_APPROACH == 2
        // Coefficient tiles: unpack each group of 8 triangles into Dest slot 3, then run the SFPU on it.
        for (uint32_t g = 0; g < kTrianglesPerBatch / SPIKE_TPC; g++) {
            copy_tile(cb_coef, g, 3);
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                spike_b_batch_sel,
                0,
                VectorMode::RC_custom,
                static_cast<const volatile float*>(nullptr),
                static_cast<uint32_t>(SPIKE_TPC),
                3u,
                nv));
        }
#else
        for (uint32_t g = 0; g < kTrianglesPerBatch / SPIKE_TPC; g++) {
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                spike_b_batch_sel,
                0,
                VectorMode::RC_custom,
                reinterpret_cast<const volatile float*>(scal_addr) + 4 * SPIKE_TPC * g,
                static_cast<uint32_t>(SPIKE_TPC),
                0u,
                nv));
        }
#endif
    }
    MATH(ckernel::tensix_sync(); t1 = ckernel::read_wall_clock();)
    uint32_t cycles = static_cast<uint32_t>(t1 - t0);
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        spike_b_write_timing,
        0,
        VectorMode::RC_custom,
        cycles & 0xFFFFu,
        cycles >> 16));

    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 2);
    pack_tile(2, cb_out);  // acc
    pack_tile(0, cb_out);  // timing
    cb_push_back(cb_out, 2);
    tile_regs_release();

    cb_pop_front(cb_scal, 1);
    cb_pop_front(cb_coef, kCoefTiles);
}
