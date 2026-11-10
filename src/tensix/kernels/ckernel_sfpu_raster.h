// SPDX-License-Identifier: Apache-2.0
//
// SFPU rasterizer for one 32x32 tile held in Dest (docs/FINDINGS.md, "Phase 2").
//
// Dest slots (FP32, full sync, 8 tiles). Colour is kept as four FP32 planes while triangles are
// drawn and quantized once per tile at the end, which is much cheaper than packing RGBA8 per
// triangle; the packed layout of the Phase 0 proposal is evaluated in Phase 2.4.
//   0 depth   1 R   2 G   3 B   4 A   5 dx   6 dy   7 triangle id
// dx, dy are the pixel-centre offsets from the tile centre in 1/16 pixel units, 16*i - 248
// (see src/ttraster/tile.h), generated once per tile from the lane ids.
//
// Each function walks all four faces itself and is called with VectorMode::RC_custom.
// Lane map (Spike A): vector v of face f covers rows 4*(v>>1) + lane_row and columns
// 2*(lane&7) + (v&1) inside the face, faces ordered TL, TR, BL, BR.

#pragma once

#include <cstdint>

#include "raster_layout.h"
#include "sfpi.h"

namespace ckernel::sfpu {

constexpr uint32_t kSlotZ = 0, kSlotR = 1, kSlotG = 2, kSlotB = 3, kSlotA = 4, kSlotDx = 5, kSlotDy = 6, kSlotId = 7;
constexpr uint32_t kVectors = 32;

// Clear the tile: depth, colour planes, id; generate dx and dy.
inline void raster_clear_tile(float clear_depth, float cr, float cg, float cb, float ca) {
    sfpi::vInt tid = sfpi::vConstTileId;  // 2 * lane
    sfpi::vInt col_base = tid & 0xE;
    sfpi::vInt row_base = tid >> 4;
#pragma GCC unroll 32
    for (uint32_t v = 0; v < kVectors; v++) {
        const uint32_t face = v / 8, vv = v % 8;
        const int col_off = 16 * (face & 1) + (vv & 1);
        const int row_off = 16 * (face >> 1) + 4 * (vv >> 1);
        sfpi::vFloat col = sfpi::convert<sfpi::vFloat>(col_base + col_off, sfpi::RoundMode::Nearest);
        sfpi::vFloat row = sfpi::convert<sfpi::vFloat>(row_base + row_off, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[kSlotDx * kVectors + v] = col * 16.0f - 248.0f;
        sfpi::dst_reg[kSlotDy * kVectors + v] = row * 16.0f - 248.0f;
        sfpi::dst_reg[kSlotZ * kVectors + v] = clear_depth;
        sfpi::dst_reg[kSlotR * kVectors + v] = cr;
        sfpi::dst_reg[kSlotG * kVectors + v] = cg;
        sfpi::dst_reg[kSlotB * kVectors + v] = cb;
        sfpi::dst_reg[kSlotA * kVectors + v] = ca;
        sfpi::dst_reg[kSlotId * kVectors + v] = 0.0f;
    }
}

// Draw one triangle record (see raster_layout.h) into the tile. `rec` is the L1 address of the
// record. Edge test: E = fma(a, dx, fma(b, dy, c)) >= 0 for all three edges; depth and colour use
// the same two-multiply-add shape. Pixels that pass the depth test write depth, colour and id.
template <bool LESS_EQUAL>
inline void raster_triangle_impl(const volatile float* rec) {
    using namespace ckernel;
    const float a0 = rec[0], b0 = rec[1], c0 = rec[2];
    const float a1 = rec[3], b1 = rec[4], c1 = rec[5];
    const float a2 = rec[6], b2 = rec[7], c2 = rec[8];
    const float zc = rec[9], zx = rec[10], zy = rec[11];
    const float id = rec[24];
    for (uint32_t v = 0; v < kVectors; v++) {
        sfpi::vFloat dx = sfpi::dst_reg[kSlotDx * kVectors + v];
        sfpi::vFloat dy = sfpi::dst_reg[kSlotDy * kVectors + v];
        sfpi::vFloat t0 = b0 * dy + c0;
        sfpi::vFloat e0 = a0 * dx + t0;
        v_if (e0 >= 0.0f) {
            sfpi::vFloat t1 = b1 * dy + c1;
            sfpi::vFloat e1 = a1 * dx + t1;
            v_if (e1 >= 0.0f) {
                sfpi::vFloat t2 = b2 * dy + c2;
                sfpi::vFloat e2 = a2 * dx + t2;
                v_if (e2 >= 0.0f) {
                    sfpi::vFloat tz = zy * dy + zc;
                    sfpi::vFloat z = zx * dx + tz;
                    sfpi::vFloat zo = sfpi::dst_reg[kSlotZ * kVectors + v];
                    if constexpr (LESS_EQUAL) {
                        v_if (z <= zo) {
                            sfpi::dst_reg[kSlotZ * kVectors + v] = z;
                            sfpi::dst_reg[kSlotId * kVectors + v] = id;
                            sfpi::dst_reg[kSlotR * kVectors + v] = rec[13] * dx + (rec[14] * dy + rec[12]);
                            sfpi::dst_reg[kSlotG * kVectors + v] = rec[16] * dx + (rec[17] * dy + rec[15]);
                            sfpi::dst_reg[kSlotB * kVectors + v] = rec[19] * dx + (rec[20] * dy + rec[18]);
                            sfpi::dst_reg[kSlotA * kVectors + v] = rec[22] * dx + (rec[23] * dy + rec[21]);
                        }
                        v_endif;
                    } else {
                        v_if (z < zo) {
                            sfpi::dst_reg[kSlotZ * kVectors + v] = z;
                            sfpi::dst_reg[kSlotId * kVectors + v] = id;
                            sfpi::dst_reg[kSlotR * kVectors + v] = rec[13] * dx + (rec[14] * dy + rec[12]);
                            sfpi::dst_reg[kSlotG * kVectors + v] = rec[16] * dx + (rec[17] * dy + rec[15]);
                            sfpi::dst_reg[kSlotB * kVectors + v] = rec[19] * dx + (rec[20] * dy + rec[18]);
                            sfpi::dst_reg[kSlotA * kVectors + v] = rec[22] * dx + (rec[23] * dy + rec[21]);
                        }
                        v_endif;
                    }
                }
                v_endif;
            }
            v_endif;
        }
        v_endif;
    }
}

// Debug variant: every covering triangle adds 1 to the id plane, no depth test, no colour. Used to
// compare coverage with the reference's coverage count independently of depth.
inline void raster_triangle_cover(const volatile float* rec) {
    const float a0 = rec[0], b0 = rec[1], c0 = rec[2];
    const float a1 = rec[3], b1 = rec[4], c1 = rec[5];
    const float a2 = rec[6], b2 = rec[7], c2 = rec[8];
    for (uint32_t v = 0; v < kVectors; v++) {
        sfpi::vFloat dx = sfpi::dst_reg[kSlotDx * kVectors + v];
        sfpi::vFloat dy = sfpi::dst_reg[kSlotDy * kVectors + v];
        sfpi::vFloat t0 = b0 * dy + c0;
        sfpi::vFloat e0 = a0 * dx + t0;
        v_if (e0 >= 0.0f) {
            sfpi::vFloat t1 = b1 * dy + c1;
            sfpi::vFloat e1 = a1 * dx + t1;
            v_if (e1 >= 0.0f) {
                sfpi::vFloat t2 = b2 * dy + c2;
                sfpi::vFloat e2 = a2 * dx + t2;
                v_if (e2 >= 0.0f) {
                    sfpi::vFloat n = sfpi::dst_reg[kSlotId * kVectors + v];
                    sfpi::dst_reg[kSlotId * kVectors + v] = n + 1.0f;
                }
                v_endif;
            }
            v_endif;
        }
        v_endif;
    }
}

// Quantize one colour channel as src/ttraster/frame.cpp: clamp(floor(v * 255 + 0.5), 0, 255).
inline sfpi::vUInt raster_quantize(sfpi::vFloat v) {
    sfpi::vFloat t = v * 255.0f;
    v_if (t < 0.0f) { t = 0.0f; }
    v_endif;
    v_if (t > 255.0f) { t = 255.0f; }
    v_endif;
    return sfpi::convert<sfpi::vUInt16>(t, sfpi::RoundMode::Nearest);
}

// Pack the four colour planes into RGBA8 words (R in the low byte), stored in the R plane's slot
// as a raw 32-bit integer (SFPSTORE INT32).
inline void raster_finalize_tile() {
#pragma GCC unroll 4
    for (uint32_t v = 0; v < kVectors; v++) {
        sfpi::vUInt r = raster_quantize(sfpi::dst_reg[kSlotR * kVectors + v]);
        sfpi::vUInt g = raster_quantize(sfpi::dst_reg[kSlotG * kVectors + v]);
        sfpi::vUInt b = raster_quantize(sfpi::dst_reg[kSlotB * kVectors + v]);
        sfpi::vUInt a = raster_quantize(sfpi::dst_reg[kSlotA * kVectors + v]);
        sfpi::vUInt packed = r | (g << 8) | (b << 16) | (a << 24);
        sfpi::dst_reg[kSlotR * kVectors + v] = packed;  // raw 32-bit integer store
    }
}

// Non-template entry points for the SFPU_UNARY_CALL macros (template commas break macros).
inline void raster_triangle_less(const volatile float* rec) { raster_triangle_impl<false>(rec); }
inline void raster_triangle_lequal(const volatile float* rec) { raster_triangle_impl<true>(rec); }

}  // namespace ckernel::sfpu
