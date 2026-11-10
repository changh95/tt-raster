// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel: for every tile block, clear the tile in Dest, draw every triangle record of
// the block with the SFPU, quantize colour, pack three planes.
//
// The math thread reads the triangle records straight from the L1 page. A page may only be
// popped (and refilled by the reader) after the math thread's last load of it has returned, and
// the CB does not synchronize those threads, so the math thread posts a release on the
// MATH -> UNPACK mailbox and the unpack thread consumes it before cb_pop_front, the same
// handshake tt-metal's triangle_solve_tile uses.
//
// Runtime args: number of blocks, clear depth, clear R, G, B, A (floats as raw words).
// Defines (set by the host): RASTER_LESS_EQUAL (depth function), RASTER_COVER_COUNT (debug: every
// covering triangle adds 1 to the id plane and the depth test is skipped, so the plane is the
// coverage count of the reference).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "raster_layout.h"

#ifdef TRISC_MATH
#include "ckernel_sfpu_raster.h"
#endif

namespace {
constexpr uint32_t kPageReleased = 0x52414C53;  // 'RALS'

inline float word_to_float(uint32_t w) {
    union { uint32_t u; float f; } x;
    x.u = w;
    return x.f;
}
}  // namespace

void kernel_main() {
    using namespace ttr::layout;
    const uint32_t n_blocks = get_arg_val<uint32_t>(0);
    const float clear_depth = word_to_float(get_arg_val<uint32_t>(1));
    const float clear_r = word_to_float(get_arg_val<uint32_t>(2));
    const float clear_g = word_to_float(get_arg_val<uint32_t>(3));
    const float clear_b = word_to_float(get_arg_val<uint32_t>(4));
    const float clear_a = word_to_float(get_arg_val<uint32_t>(5));

    constexpr auto cb_tri = tt::CBIndex::c_0;
    constexpr auto cb_color = tt::CBIndex::c_16;
    constexpr auto cb_depth = tt::CBIndex::c_17;
    constexpr auto cb_id = tt::CBIndex::c_18;

    compute_kernel_hw_startup(cb_tri, cb_color);

    for (uint32_t blk = 0; blk < n_blocks; blk++) {
        tile_regs_acquire();
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, raster_clear_tile, 0, VectorMode::RC_custom, clear_depth, clear_r, clear_g,
            clear_b, clear_a));

        uint32_t remaining = 0;  // records still to draw in this block
        uint32_t n_pages = 1;
        for (uint32_t p = 0; p < n_pages; p++) {
            cb_wait_front(cb_tri, 1);
            const uint32_t page = get_tile_address(cb_tri, 0);  // L1 byte address, same on all threads
            const volatile uint32_t* words = reinterpret_cast<const volatile uint32_t*>(page);
            uint32_t first = 0;  // first record slot of this page holding a triangle
            if (p == 0) {
                remaining = words[kHdrRecords];
                n_pages = pages_for_records(remaining);
                first = 1;
            }
            for (uint32_t s = first; s < kRecordsPerPage && remaining > 0; s++, remaining--) {
                const volatile float* rec = reinterpret_cast<const volatile float*>(page + s * kRecordBytes);
#if defined(RASTER_COVER_COUNT)
                MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, raster_triangle_cover, 0, VectorMode::RC_custom, rec));
#elif defined(RASTER_LESS_EQUAL)
                MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, raster_triangle_lequal, 0, VectorMode::RC_custom, rec));
#else
                MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, raster_triangle_less, 0, VectorMode::RC_custom, rec));
#endif
            }
            MATH(ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, kPageReleased);)
            UNPACK((void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId);)
            cb_pop_front(cb_tri, 1);
        }

        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, raster_finalize_tile, 0, VectorMode::RC_custom));
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_color, 1);
        cb_reserve_back(cb_depth, 1);
        cb_reserve_back(cb_id, 1);
        pack_tile(1, cb_color);   // slot 1 (R plane) now holds the packed RGBA8 words
        pack_tile(0, cb_depth);
        pack_tile(7, cb_id);
        cb_push_back(cb_color, 1);
        cb_push_back(cb_depth, 1);
        cb_push_back(cb_id, 1);
        tile_regs_release();
    }
}
