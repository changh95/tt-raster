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
// Runtime args: number of blocks, clear depth, clear R, G, B, A (floats as raw words), clear colour packed, program table pages.
// Defines (set by the host): RASTER_PLANE_MASK (bit 0 colour, 1 depth, 2 triangle id, 3 G, 4 B planes to write
// back; without colour nothing is quantized and no colour is computed), RASTER_FLOAT_PLANES (colour,
// G and B are unquantized floats), RASTER_LESS_EQUAL (depth function), RASTER_PACKED_COLOUR (Phase 2.4 variant: colour is packed to
// RGBA8 per pixel as triangles are drawn instead of once per tile), RASTER_COVER_COUNT (debug: every
// covering triangle adds 1 to the id plane and the depth test is skipped, so the plane is the
// coverage count of the reference). RASTER_SHADERS (frames with fragment shaders, implies RASTER_PACKED_COLOUR):
// every triangle goes through the shader interpreter (ckernel_sfpu_shader.h); the program table is in CB c_1,
// read once by the reader.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "raster_layout.h"

#ifdef TRISC_MATH
#ifdef RASTER_SHADERS
#include "ckernel_sfpu_shader.h"
#else
#include "ckernel_sfpu_raster.h"
#endif
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
    const uint32_t clear_packed = get_arg_val<uint32_t>(6);
    const uint32_t table_pages = get_arg_val<uint32_t>(7);   // shader frames: pages of the program table

    constexpr auto cb_tri = tt::CBIndex::c_0;
    constexpr auto cb_color = tt::CBIndex::c_16;
    constexpr auto cb_depth = tt::CBIndex::c_17;
    constexpr auto cb_id = tt::CBIndex::c_18;
    constexpr auto cb_g = tt::CBIndex::c_19;
    constexpr auto cb_b = tt::CBIndex::c_20;

    // The packer is configured from one output buffer; it must be one that exists, i.e. the first
    // requested plane's (all planes share one format).
#if RASTER_PLANE_MASK & 1
    constexpr auto cb_first = cb_color;
#elif RASTER_PLANE_MASK & 2
    constexpr auto cb_first = cb_depth;
#else
    constexpr auto cb_first = cb_id;
#endif
    compute_kernel_hw_startup(cb_tri, cb_first);

#ifdef RASTER_SHADERS
    // The program table: filled once by the reader, never popped. get_tile_address hands the address to the math thread
    // through a mailbox, so that thread starts reading only after the unpack thread has seen the table arrive.
    constexpr auto cb_prog = tt::CBIndex::c_1;
    cb_wait_front(cb_prog, table_pages);
    const volatile uint32_t* prog_table = reinterpret_cast<const volatile uint32_t*>(get_tile_address(cb_prog, 0));
#endif

    for (uint32_t blk = 0; blk < n_blocks; blk++) {
        tile_regs_acquire();
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, raster_clear_tile, 0, VectorMode::RC_custom, clear_depth, clear_r, clear_g,
            clear_b, clear_a, clear_packed));

        uint32_t remaining = 0;  // records still to draw in this block
        uint32_t n_pages = 1;
#ifdef RASTER_SHADERS
        float tile_cx = 0.0f, tile_cy = 0.0f;   // the tile centre in window pixels
#endif
        for (uint32_t p = 0; p < n_pages; p++) {
            cb_wait_front(cb_tri, 1);
            const uint32_t page = get_tile_address(cb_tri, 0);  // L1 byte address, same on all threads
            const volatile uint32_t* words = reinterpret_cast<const volatile uint32_t*>(page);
            uint32_t first = 0;  // first record slot of this page holding a triangle
            if (p == 0) {
                remaining = words[kHdrRecords];
#ifdef RASTER_SHADERS
                n_pages = words[kHdrPages];
                tile_cx = word_to_float(words[kHdrCx]);
                tile_cy = word_to_float(words[kHdrCy]);
#else
                n_pages = pages_for_records(remaining);
#endif
                first = 1;
            }
#ifdef RASTER_SHADERS
            // Records take one slot plus their extension slots and never straddle a page; a page whose tail the next
            // record does not fit in ends with a marker (raster_layout.h).
            for (uint32_t slot = first; slot < kRecordsPerPage && remaining > 0;) {
                const volatile float* rec = reinterpret_cast<const volatile float*>(page + slot * kRecordBytes);
                const uint32_t flags = words[slot * kRecordWords + kFlags];
                if (flags == kPageEnd) {
                    break;
                }
                MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, raster_triangle_shader, 0, VectorMode::RC_custom, rec, prog_table, tile_cx,
                    tile_cy));
                slot += 1 + ((flags >> kExtShift) & 3u);
                remaining--;
            }
#else
            uint32_t n_here = kRecordsPerPage - first;  // records in this page
            if (n_here > remaining) {
                n_here = remaining;
            }
            remaining -= n_here;
#ifdef RASTER_EXP_SKIP_TRIANGLES
            n_here = 0;  // experiment: no drawing, measures clear + finalize + dataflow
#endif
            for (uint32_t k = 0; k < n_here; k++) {
                const volatile float* rec = reinterpret_cast<const volatile float*>(page + (first + k) * kRecordBytes);
                MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, raster_triangle, 0, VectorMode::RC_custom, rec));
            }
#endif
            MATH(ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, kPageReleased);)
            UNPACK((void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId);)
            cb_pop_front(cb_tri, 1);
        }

#if !defined(RASTER_FLOAT_PLANES) && (RASTER_PLANE_MASK & 1)
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, raster_finalize_tile, 0, VectorMode::RC_custom));
#endif
        tile_regs_commit();
        tile_regs_wait();
        // Planes present in RASTER_PLANE_MASK, in this order: colour (Dest slot 1: packed RGBA8, or R as a
        // float), depth (0), triangle id (7), G (2), B (3); each goes to its own circular buffer c_16 + bit.
#if RASTER_PLANE_MASK & 1
        cb_reserve_back(cb_color, 1);
        pack_tile(1, cb_color);
        cb_push_back(cb_color, 1);
#endif
#if RASTER_PLANE_MASK & 2
        cb_reserve_back(cb_depth, 1);
        pack_tile(0, cb_depth);
        cb_push_back(cb_depth, 1);
#endif
#if RASTER_PLANE_MASK & 4
        cb_reserve_back(cb_id, 1);
        pack_tile(7, cb_id);
        cb_push_back(cb_id, 1);
#endif
#if RASTER_PLANE_MASK & 8
        cb_reserve_back(cb_g, 1);
        pack_tile(2, cb_g);
        cb_push_back(cb_g, 1);
#endif
#if RASTER_PLANE_MASK & 16
        cb_reserve_back(cb_b, 1);
        pack_tile(3, cb_b);
        cb_push_back(cb_b, 1);
#endif
        tile_regs_release();
    }
}
