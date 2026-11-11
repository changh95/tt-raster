// SPDX-License-Identifier: Apache-2.0
//
// Spike B SFPU functions: four ways to get per-triangle scalars into the SFPU.
//
// Every variant evaluates, for one triangle with scalars (a, b, c, d) and for all 1024 pixels of
// a 32x32 tile held in Dest:
//     if (a*dx + b*dy + c >= 0) acc += d;
//
// Dest slots: 0 = dx, 1 = dy (generated from lane ids), 2 = acc, 3 = coefficient tile (approach 2).
// The functions are called once per batch with VectorMode::RC_custom and walk all four faces
// themselves: tile t, vector v (0..31) is dst_reg[t * 32 + v].
//
// Compile-time selection (ComputeConfig defines):
//   SPIKE_APPROACH  0 constants baked into the code (no scalar delivery, timing baseline)
//                   1 A: the math thread reads floats from L1 and issues them as runtime immediates
//                   2 B: a coefficient tile unpacked into Dest, one SFPLOAD per scalar
//                   3 C: programmable SFPU constant registers (a, b, c), d as a runtime immediate
//   SPIKE_VEC       0 triangle-major (scalars once per triangle, dx/dy/acc reloaded per vector)
//                   1 vector-major (dx/dy/acc loaded once per vector, scalars reloaded per triangle)

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel::sfpu {

constexpr uint32_t kSlotDx = 0, kSlotDy = 1, kSlotAcc = 2;

// Constants used by approach 0 (mirrored on the host).
constexpr float kConstA = 2.0f, kConstB = -3.0f, kConstC = 17.0f, kConstD = 1.0f;

// dx, dy from the lane map measured in Spike A; acc = 0.
inline void spike_b_init_tiles() {
    sfpi::vInt tid = sfpi::vConstTileId;  // 2 * lane
    sfpi::vInt col_base = tid & 0xE;
    sfpi::vInt row_base = tid >> 4;
#pragma GCC unroll 32
    for (uint32_t v = 0; v < 32; v++) {
        const uint32_t face = v / 8, vv = v % 8;
        const int col_off = 16 * (face & 1) + (vv & 1);
        const int row_off = 16 * (face >> 1) + 4 * (vv >> 1);
        sfpi::dst_reg[kSlotDx * 32 + v] = sfpi::convert<sfpi::vFloat>(col_base + col_off, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[kSlotDy * 32 + v] = sfpi::convert<sfpi::vFloat>(row_base + row_off, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[kSlotAcc * 32 + v] = 0.0f;
    }
}

// Write the cycle count (low 16 bits, high bits) into vectors 0 and 1 of slot 0.
inline void spike_b_write_timing(uint32_t lo16, uint32_t hi) {
    sfpi::vInt lo = static_cast<int>(lo16);
    sfpi::vInt hi_v = static_cast<int>(hi);
    sfpi::dst_reg[kSlotDx * 32 + 0] = sfpi::convert<sfpi::vFloat>(lo, sfpi::RoundMode::Nearest);
    sfpi::dst_reg[kSlotDx * 32 + 1] = sfpi::convert<sfpi::vFloat>(hi_v, sfpi::RoundMode::Nearest);
}

// One triangle over all 32 vectors, scalars already in registers.
inline void spike_b_tri(sfpi::vFloat a, sfpi::vFloat b, sfpi::vFloat c, sfpi::vFloat d, uint32_t nv) {
    for (uint32_t v = 0; v < nv; v++) {
        sfpi::vFloat dx = sfpi::dst_reg[kSlotDx * 32 + v];
        sfpi::vFloat dy = sfpi::dst_reg[kSlotDy * 32 + v];
        sfpi::vFloat acc = sfpi::dst_reg[kSlotAcc * 32 + v];
        sfpi::vFloat e = a * dx + b * dy + c;
        v_if (e >= 0.0f) { acc += d; }
        v_endif;
        sfpi::dst_reg[kSlotAcc * 32 + v] = acc;
    }
}

// Same, with a, b, c in the programmable constant registers.
inline void spike_b_tri_prgm(sfpi::vFloat d, uint32_t nv) {
    for (uint32_t v = 0; v < nv; v++) {
        sfpi::vFloat dx = sfpi::dst_reg[kSlotDx * 32 + v];
        sfpi::vFloat dy = sfpi::dst_reg[kSlotDy * 32 + v];
        sfpi::vFloat acc = sfpi::dst_reg[kSlotAcc * 32 + v];
        sfpi::vFloat e = sfpi::vConstFloatPrgm0 * dx + sfpi::vConstFloatPrgm1 * dy + sfpi::vConstFloatPrgm2;
        v_if (e >= 0.0f) { acc += d; }
        v_endif;
        sfpi::dst_reg[kSlotAcc * 32 + v] = acc;
    }
}

// n_tri triangles. scal: L1 array of (a, b, c, d) floats (approaches 0, 1, 3).
// coef_slot: Dest slot of the coefficient tile (approach 2): triangle t uses vectors 4t..4t+3.
// nv: how many vectors (of 32) each triangle updates, 1..32. A triangle that covers only part of
// the tile would restrict this loop to the face / row-band it touches.
template <int APPROACH, bool VEC>
inline void spike_b_batch(const volatile float* scal, uint32_t n_tri, uint32_t coef_slot, uint32_t nv) {
    if constexpr (!VEC) {
        for (uint32_t t = 0; t < n_tri; t++) {
            if constexpr (APPROACH == 0) {
                spike_b_tri(kConstA, kConstB, kConstC, kConstD, nv);
            } else if constexpr (APPROACH == 1) {
                const float a = scal[4 * t + 0], b = scal[4 * t + 1], c = scal[4 * t + 2], d = scal[4 * t + 3];
                spike_b_tri(a, b, c, d, nv);
            } else if constexpr (APPROACH == 2) {
                sfpi::vFloat a = sfpi::dst_reg[coef_slot * 32 + 4 * t + 0];
                sfpi::vFloat b = sfpi::dst_reg[coef_slot * 32 + 4 * t + 1];
                sfpi::vFloat c = sfpi::dst_reg[coef_slot * 32 + 4 * t + 2];
                sfpi::vFloat d = sfpi::dst_reg[coef_slot * 32 + 4 * t + 3];
                spike_b_tri(a, b, c, d, nv);
            } else {
                sfpi::vConstFloatPrgm0 = scal[4 * t + 0];
                sfpi::vConstFloatPrgm1 = scal[4 * t + 1];
                sfpi::vConstFloatPrgm2 = scal[4 * t + 2];
                spike_b_tri_prgm(scal[4 * t + 3], nv);
            }
        }
    } else {
        for (uint32_t v = 0; v < nv; v++) {
            sfpi::vFloat dx = sfpi::dst_reg[kSlotDx * 32 + v];
            sfpi::vFloat dy = sfpi::dst_reg[kSlotDy * 32 + v];
            sfpi::vFloat acc = sfpi::dst_reg[kSlotAcc * 32 + v];
            for (uint32_t t = 0; t < n_tri; t++) {
                sfpi::vFloat a, b, c, d;
                if constexpr (APPROACH == 0) {
                    a = kConstA; b = kConstB; c = kConstC; d = kConstD;
                } else if constexpr (APPROACH == 1) {
                    a = scal[4 * t + 0]; b = scal[4 * t + 1]; c = scal[4 * t + 2]; d = scal[4 * t + 3];
                } else {
                    a = sfpi::dst_reg[coef_slot * 32 + 4 * t + 0];
                    b = sfpi::dst_reg[coef_slot * 32 + 4 * t + 1];
                    c = sfpi::dst_reg[coef_slot * 32 + 4 * t + 2];
                    d = sfpi::dst_reg[coef_slot * 32 + 4 * t + 3];
                }
                sfpi::vFloat e = a * dx + b * dy + c;
                v_if (e >= 0.0f) { acc += d; }
                v_endif;
            }
            sfpi::dst_reg[kSlotAcc * 32 + v] = acc;
        }
    }
}

// Non-template entry point for the SFPU_UNARY_CALL macro (template commas break macros).
inline void spike_b_batch_sel(const volatile float* scal, uint32_t n_tri, uint32_t coef_slot, uint32_t nv) {
    spike_b_batch<SPIKE_APPROACH, (SPIKE_VEC != 0)>(scal, n_tri, coef_slot, nv);
}

}  // namespace ckernel::sfpu
