// SPDX-License-Identifier: Apache-2.0
//
// Spike A SFPU functions. Included by compute.cpp under TRISC_MATH only.
//
// Each function is called once per 16x16 face (VectorMode::RC wrapper), and a face
// is 8 SFPU vectors of 32 lanes. Tile t sits at dst_reg[t * 32 + i] for i in 0..7
// inside the current face.

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel::sfpu {

// Set to 0 by the compute kernel before each probe call; counts the faces within one tile call.
inline uint32_t spike_a_face = 0;

// out = checkerboard + gradient of (x, y). x and y hold integer-valued coordinates 0..31 as FP32.
inline void spike_a_checker_face(uint32_t idx_x, uint32_t idx_y, uint32_t idx_out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vFloat x = sfpi::dst_reg[idx_x * 32 + i];
        sfpi::vFloat y = sfpi::dst_reg[idx_y * 32 + i];
        sfpi::vFloat out = x * 0.25f + y * 0.5f;
        // Blackhole converts float to 16-bit integers only (coordinates here are 0..62).
        sfpi::vUInt sum = sfpi::convert<sfpi::vUInt16>(x + y, sfpi::RoundMode::Nearest);
        v_if ((sum & 1) != 0) { out += 100.0f; }
        v_endif;
        sfpi::dst_reg[idx_out * 32 + i] = out;
    }
}

// Lane probe: write (face * 8 + vector) * 64 + vConstTileId into every lane, so the host can
// read back which (face, vector, lane-constant) produced each tile position.
inline void spike_a_probe_face() {
    const uint32_t face = spike_a_face++;
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vInt code = sfpi::vConstTileId + static_cast<int>((face * 8 + i) * 64);
        sfpi::dst_reg[i] = sfpi::convert<sfpi::vFloat>(code, sfpi::RoundMode::Nearest);
    }
}

// FP32 arithmetic check: out = x * 0.1 + y * 0.3, compared on the host against both a
// two-rounding and a fused (single-rounding) CPU evaluation.
inline void spike_a_math_face(uint32_t idx_x, uint32_t idx_y, uint32_t idx_out) {
    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vFloat x = sfpi::dst_reg[idx_x * 32 + i];
        sfpi::vFloat y = sfpi::dst_reg[idx_y * 32 + i];
        sfpi::dst_reg[idx_out * 32 + i] = x * 0.1f + y * 0.3f;
    }
}

}  // namespace ckernel::sfpu
