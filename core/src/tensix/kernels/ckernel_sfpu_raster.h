// SPDX-License-Identifier: Apache-2.0
//
// SFPU rasterizer for one 32x32 tile held in Dest (docs/FINDINGS.md, "Phase 2").
//
// Dest slots (FP32, full sync, 8 tiles). Colour is kept as four FP32 planes while triangles are
// drawn and quantized once per tile at the end, which is much cheaper than packing RGBA8 per
// triangle; the packed layout of the Phase 0 proposal is evaluated in Phase 2.4.
//   0 depth   1 R   2 G   3 B   4 A   5 dx   6 dy   7 triangle id
// dx, dy are the pixel-centre offsets from the tile centre in 1/16 pixel units, 16*i - 248
// (see core/src/ttraster/tile.h), generated once per tile from the lane ids.
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
inline void raster_clear_tile(float clear_depth, float cr, float cg, float cb, float ca, uint32_t clear_packed) {
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
#ifdef RASTER_PACKED_COLOUR
        sfpi::dst_reg[kSlotR * kVectors + v] = sfpi::vUInt(clear_packed);
        (void)cr; (void)cg; (void)cb; (void)ca;
#else
        sfpi::dst_reg[kSlotR * kVectors + v] = cr;
        sfpi::dst_reg[kSlotG * kVectors + v] = cg;
        sfpi::dst_reg[kSlotB * kVectors + v] = cb;
        sfpi::dst_reg[kSlotA * kVectors + v] = ca;
        (void)clear_packed;
#endif
        sfpi::dst_reg[kSlotId * kVectors + v] = 0.0f;
    }
}

// Scalars of one triangle record, read from L1 once per triangle.
struct TriScalars {
    float a0, b0, c0, a1, b1, c1, a2, b2, c2;  // edges
    float zc, zx, zy;                          // depth plane
    float rc, rx, ry, gc, gx, gy, bc, bx, by, ac, ax, ay;  // colour planes (gradients unused when flat)
    float wc, wx, wy;                          // rhw plane (perspective triangles only)
    float id;
    uint32_t depth_func;                       // stateful records: 0 LESS, 1 LESS_EQUAL, 2 ALWAYS
    uint32_t depth_write;
    float ka_c, ka_sa, kb_c, kb_sa, kb_s;      // blend coefficients (raster_layout.h); blend off: all 0 and `blend` false
    bool blend;
    uint32_t packed;  // flat colour packed to RGBA8 (packed-colour variant)
};

template <bool FLAT, bool PERSP>
inline TriScalars load_scalars(const volatile float* rec) {
    TriScalars s;
    s.a0 = rec[0]; s.b0 = rec[1]; s.c0 = rec[2];
    s.a1 = rec[3]; s.b1 = rec[4]; s.c1 = rec[5];
    s.a2 = rec[6]; s.b2 = rec[7]; s.c2 = rec[8];
    s.zc = rec[9]; s.zx = rec[10]; s.zy = rec[11];
    s.rc = rec[12]; s.gc = rec[15]; s.bc = rec[18]; s.ac = rec[21];
    if constexpr (!FLAT) {
        s.rx = rec[13]; s.ry = rec[14];
        s.gx = rec[16]; s.gy = rec[17];
        s.bx = rec[19]; s.by = rec[20];
        s.ax = rec[22]; s.ay = rec[23];
    }
    if constexpr (PERSP) {
        s.wc = rec[ttr::layout::kWPlane]; s.wx = rec[ttr::layout::kWPlane + 1]; s.wy = rec[ttr::layout::kWPlane + 2];
    }
    s.id = rec[24];
    s.depth_func = 0;
    s.depth_write = 1;
    s.blend = false;
    s.packed = reinterpret_cast<const volatile uint32_t*>(rec)[ttr::layout::kFlatPacked];
    return s;
}

// Quantize one colour channel as core/src/ttraster/frame.cpp: clamp(floor(v * 255 + 0.5), 0, 255).
__attribute__((always_inline)) inline sfpi::vUInt raster_quantize(sfpi::vFloat v) {
    sfpi::vFloat t = v * 255.0f;
    v_if (t < 0.0f) { t = 0.0f; }
    v_endif;
    v_if (t > 255.0f) { t = 255.0f; }
    v_endif;
    return sfpi::convert<sfpi::vUInt16>(t, sfpi::RoundMode::Nearest);
}

// 1 / x for x > 0: the hardware estimate refined by two Newton steps (as tt-metal's FP32 reciprocal).
__attribute__((always_inline)) inline sfpi::vFloat raster_reciprocal(sfpi::vFloat x) {
    sfpi::vFloat y = sfpi::approx_recip(x);
    sfpi::vFloat t = x * y - 2.0f;
    y = y * -t;
    t = x * y - 2.0f;
    return y * -t;
}

// Colour planes of a pixel that won the depth test: constants for flat triangles, otherwise
// fma(gx, dx, fma(gy, dy, c)) per channel; perspective triangles divide by the rhw plane.
template <bool FLAT, bool PERSP>
__attribute__((always_inline)) inline void write_colour(const TriScalars& s, sfpi::vFloat dx, sfpi::vFloat dy) {
#if defined(RASTER_EXP_NO_COLOUR) || !(RASTER_PLANE_MASK & 1)  // no colour plane requested (or timing experiment)
    (void)s; (void)dx; (void)dy;
#elif defined(RASTER_PACKED_COLOUR)
    if constexpr (FLAT) {
        sfpi::dst_reg[kSlotR * kVectors] = sfpi::vUInt(s.packed);
    } else {
        // Channel by channel, accumulating in Dest: the SFPU has 8 registers and vectors cannot be spilled.
        sfpi::dst_reg[kSlotR * kVectors] = raster_quantize(s.ax * dx + (s.ay * dy + s.ac)) << 24;
        sfpi::vUInt b = raster_quantize(s.bx * dx + (s.by * dy + s.bc)) << 16;
        sfpi::dst_reg[kSlotR * kVectors] = sfpi::vUInt(sfpi::dst_reg[kSlotR * kVectors]) | b;
        sfpi::vUInt g = raster_quantize(s.gx * dx + (s.gy * dy + s.gc)) << 8;
        sfpi::dst_reg[kSlotR * kVectors] = sfpi::vUInt(sfpi::dst_reg[kSlotR * kVectors]) | g;
        sfpi::vUInt r = raster_quantize(s.rx * dx + (s.ry * dy + s.rc));
        sfpi::dst_reg[kSlotR * kVectors] = sfpi::vUInt(sfpi::dst_reg[kSlotR * kVectors]) | r;
    }
#else
    if constexpr (FLAT) {
        sfpi::dst_reg[kSlotR * kVectors] = s.rc;
        sfpi::dst_reg[kSlotG * kVectors] = s.gc;
        sfpi::dst_reg[kSlotB * kVectors] = s.bc;
        sfpi::dst_reg[kSlotA * kVectors] = s.ac;
    } else if constexpr (PERSP) {
        sfpi::vFloat q = raster_reciprocal(s.wx * dx + (s.wy * dy + s.wc));
        sfpi::dst_reg[kSlotR * kVectors] = (s.rx * dx + (s.ry * dy + s.rc)) * q;
        sfpi::dst_reg[kSlotG * kVectors] = (s.gx * dx + (s.gy * dy + s.gc)) * q;
        sfpi::dst_reg[kSlotB * kVectors] = (s.bx * dx + (s.by * dy + s.bc)) * q;
        sfpi::dst_reg[kSlotA * kVectors] = (s.ax * dx + (s.ay * dy + s.ac)) * q;
    } else {
        sfpi::dst_reg[kSlotR * kVectors] = s.rx * dx + (s.ry * dy + s.rc);
        sfpi::dst_reg[kSlotG * kVectors] = s.gx * dx + (s.gy * dy + s.gc);
        sfpi::dst_reg[kSlotB * kVectors] = s.bx * dx + (s.by * dy + s.bc);
        sfpi::dst_reg[kSlotA * kVectors] = s.ax * dx + (s.ay * dy + s.ac);
    }
#endif
}

// What a pixel does once the edge tests have passed. COVER (debug): count coverage, nothing else.
template <bool FLAT, bool PERSP, bool LESS_EQUAL, bool COVER>
__attribute__((always_inline)) inline void shade_vector(const TriScalars& s, sfpi::vFloat dx, sfpi::vFloat dy) {
    if constexpr (COVER) {
        sfpi::vFloat n = sfpi::dst_reg[kSlotId * kVectors];
        sfpi::dst_reg[kSlotId * kVectors] = n + 1.0f;
    } else {
        sfpi::vFloat tz = s.zy * dy + s.zc;
        sfpi::vFloat z = s.zx * dx + tz;
        sfpi::vFloat zo = sfpi::dst_reg[kSlotZ * kVectors];
        if constexpr (LESS_EQUAL) {
            v_if (z <= zo) {
                sfpi::dst_reg[kSlotZ * kVectors] = z;
                sfpi::dst_reg[kSlotId * kVectors] = s.id;
                write_colour<FLAT, PERSP>(s, dx, dy);
            }
            v_endif;
        } else {
            v_if (z < zo) {
                sfpi::dst_reg[kSlotZ * kVectors] = z;
                sfpi::dst_reg[kSlotId * kVectors] = s.id;
                write_colour<FLAT, PERSP>(s, dx, dy);
            }
            v_endif;
        }
    }
}

// One vector at the current destination counter. FULL: every pixel is inside, no edge tests.
// Edge test: E = fma(a, dx, fma(b, dy, c)) >= 0 for all three edges (core/src/ttraster/tile.h).
template <bool FULL, bool FLAT, bool PERSP, bool LESS_EQUAL, bool COVER>
__attribute__((always_inline)) inline void draw_vector(const TriScalars& s) {
    sfpi::vFloat dx = sfpi::dst_reg[kSlotDx * kVectors];
    sfpi::vFloat dy = sfpi::dst_reg[kSlotDy * kVectors];
    if constexpr (FULL) {
        shade_vector<FLAT, PERSP, LESS_EQUAL, COVER>(s, dx, dy);
    } else {
        sfpi::vFloat t0 = s.b0 * dy + s.c0;
        sfpi::vFloat e0 = s.a0 * dx + t0;
        v_if (e0 >= 0.0f) {
            sfpi::vFloat t1 = s.b1 * dy + s.c1;
            sfpi::vFloat e1 = s.a1 * dx + t1;
            v_if (e1 >= 0.0f) {
                sfpi::vFloat t2 = s.b2 * dy + s.c2;
                sfpi::vFloat e2 = s.a2 * dx + t2;
                v_if (e2 >= 0.0f) { shade_vector<FLAT, PERSP, LESS_EQUAL, COVER>(s, dx, dy); }
                v_endif;
            }
            v_endif;
        }
        v_endif;
    }
}

// Draw one triangle record (see raster_layout.h) into the tile. The destination counter advances
// one vector per iteration (dst_reg++), so every SFPU instruction word is a compile-time constant
// (no runtime index arithmetic on the RISC-V core, which is the bottleneck). The record's
// per-vector classes skip vectors that cannot be touched (0) and drop the edge tests for vectors
// that are entirely inside (2).
template <bool FLAT, bool PERSP, bool LESS_EQUAL, bool COVER>
inline void raster_triangle_impl(const volatile float* rec) {
    const TriScalars s = load_scalars<FLAT, PERSP>(rec);
    const volatile uint32_t* classes = reinterpret_cast<const volatile uint32_t*>(rec) + ttr::layout::kVecClass;
    for (uint32_t half = 0; half < 2; half++) {
        uint32_t bits = classes[half];
        for (uint32_t i = 0; i < 16; i++) {
            uint32_t cls = bits & 3u;
            bits >>= 2;
#ifdef RASTER_EXP_FORCE_CLASS  // timing experiment only (output is wrong): every non-rejected vector gets this class
            cls = cls != 0u ? RASTER_EXP_FORCE_CLASS : 0u;
#endif
            if (cls == 1u) {
                draw_vector<false, FLAT, PERSP, LESS_EQUAL, COVER>(s);
            } else if (cls == 2u) {
                draw_vector<true, FLAT, PERSP, LESS_EQUAL, COVER>(s);
            }
            sfpi::dst_reg++;
        }
    }
}

// ---- General path: records with their own depth function, depth write and blend state -----------------------------
// (kFlagStateful). The specialized path above stays untouched for the common case.

inline void load_state(TriScalars& s, uint32_t flags) {
    s.depth_func = (flags >> ttr::layout::kDepthFuncShift) & 3u;
    s.depth_write = (flags & ttr::layout::kFlagDepthWrite) != 0u;
    const uint32_t mode = (flags >> ttr::layout::kBlendShift) & 7u;
    s.blend = mode != 0u;
    s.ka_c = ttr::layout::kBlendCoef[mode][0];
    s.ka_sa = ttr::layout::kBlendCoef[mode][1];
    s.kb_c = ttr::layout::kBlendCoef[mode][2];
    s.kb_sa = ttr::layout::kBlendCoef[mode][3];
    s.kb_s = ttr::layout::kBlendCoef[mode][4];
}

__attribute__((always_inline)) inline sfpi::vFloat raster_clamp01(sfpi::vFloat v) {
    v_if (v < 0.0f) { v = 0.0f; }
    v_endif;
    v_if (v > 1.0f) { v = 1.0f; }
    v_endif;
    return v;
}

// The fragment colour of a pixel that passed the tests, blended with the colour in Dest. Evaluation order as
// ttr::blend_channel (core/src/ttraster/tile.h): every channel is clamped, then r = s * A + d * B, clamped again.
template <bool FLAT, bool PERSP>
__attribute__((always_inline)) inline void write_colour_blend(const TriScalars& s, sfpi::vFloat dx, sfpi::vFloat dy) {
#if defined(RASTER_EXP_NO_COLOUR) || !(RASTER_PLANE_MASK & 1) || defined(RASTER_PACKED_COLOUR)
    (void)s; (void)dx; (void)dy;   // no colour plane requested: nothing to blend
#else
    sfpi::vFloat q = 1.0f;
    if constexpr (PERSP) {
        q = raster_reciprocal(s.wx * dx + (s.wy * dy + s.wc));
    }
    sfpi::vFloat sa;
    if constexpr (FLAT) {
        sa = s.ac;
    } else if constexpr (PERSP) {
        sa = (s.ax * dx + (s.ay * dy + s.ac)) * q;
    } else {
        sa = s.ax * dx + (s.ay * dy + s.ac);
    }
    sa = raster_clamp01(sa);
    sfpi::vFloat a = sa * s.ka_sa + s.ka_c;
    sfpi::vFloat bp = sa * s.kb_sa + s.kb_c;
    // Channels one after the other: only a few registers are live at a time.
#define RASTER_BLEND_CHANNEL(SLOT, CC, CX, CY)                                                      \
    {                                                                                                \
        sfpi::vFloat c;                                                                              \
        if constexpr (FLAT) {                                                                        \
            c = s.CC;                                                                                \
        } else if constexpr (PERSP) {                                                                \
            c = (s.CX * dx + (s.CY * dy + s.CC)) * q;                                                \
        } else {                                                                                     \
            c = s.CX * dx + (s.CY * dy + s.CC);                                                      \
        }                                                                                            \
        c = raster_clamp01(c);                                                                       \
        sfpi::vFloat d = sfpi::dst_reg[(SLOT) * kVectors];                                           \
        sfpi::vFloat b = c * s.kb_s + bp;                                                            \
        sfpi::vFloat r = c * a + d * b;                                                              \
        sfpi::dst_reg[(SLOT) * kVectors] = raster_clamp01(r);                                        \
    }
    RASTER_BLEND_CHANNEL(kSlotR, rc, rx, ry)
    RASTER_BLEND_CHANNEL(kSlotG, gc, gx, gy)
    RASTER_BLEND_CHANNEL(kSlotB, bc, bx, by)
    RASTER_BLEND_CHANNEL(kSlotA, ac, ax, ay)
#undef RASTER_BLEND_CHANNEL
#endif
}

template <bool FLAT, bool PERSP, bool LEQ>
__attribute__((always_inline)) inline void shade_vector_state(const TriScalars& s, sfpi::vFloat dx, sfpi::vFloat dy) {
    sfpi::vFloat z = s.zx * dx + (s.zy * dy + s.zc);
    sfpi::vFloat zo = sfpi::dst_reg[kSlotZ * kVectors];
    if (s.depth_func == 2u) {
        zo = 3.0e38f;   // depth test off: every fragment passes
    }
    if constexpr (LEQ) {
        v_if (z <= zo) {
            if (s.depth_write) { sfpi::dst_reg[kSlotZ * kVectors] = z; }
            sfpi::dst_reg[kSlotId * kVectors] = s.id;
            if (s.blend) { write_colour_blend<FLAT, PERSP>(s, dx, dy); } else { write_colour<FLAT, PERSP>(s, dx, dy); }
        }
        v_endif;
    } else {
        v_if (z < zo) {
            if (s.depth_write) { sfpi::dst_reg[kSlotZ * kVectors] = z; }
            sfpi::dst_reg[kSlotId * kVectors] = s.id;
            if (s.blend) { write_colour_blend<FLAT, PERSP>(s, dx, dy); } else { write_colour<FLAT, PERSP>(s, dx, dy); }
        }
        v_endif;
    }
}

template <bool FULL, bool FLAT, bool PERSP, bool LEQ>
__attribute__((always_inline)) inline void draw_vector_state(const TriScalars& s) {
    sfpi::vFloat dx = sfpi::dst_reg[kSlotDx * kVectors];
    sfpi::vFloat dy = sfpi::dst_reg[kSlotDy * kVectors];
    if constexpr (FULL) {
        shade_vector_state<FLAT, PERSP, LEQ>(s, dx, dy);
    } else {
        sfpi::vFloat t0 = s.b0 * dy + s.c0;
        sfpi::vFloat e0 = s.a0 * dx + t0;
        v_if (e0 >= 0.0f) {
            sfpi::vFloat t1 = s.b1 * dy + s.c1;
            sfpi::vFloat e1 = s.a1 * dx + t1;
            v_if (e1 >= 0.0f) {
                sfpi::vFloat t2 = s.b2 * dy + s.c2;
                sfpi::vFloat e2 = s.a2 * dx + t2;
                v_if (e2 >= 0.0f) { shade_vector_state<FLAT, PERSP, LEQ>(s, dx, dy); }
                v_endif;
            }
            v_endif;
        }
        v_endif;
    }
}

template <bool FLAT, bool PERSP, bool LEQ>
inline void raster_triangle_state_impl(const volatile float* rec, uint32_t flags) {
    TriScalars s = load_scalars<FLAT, PERSP>(rec);
    load_state(s, flags);
    const volatile uint32_t* classes = reinterpret_cast<const volatile uint32_t*>(rec) + ttr::layout::kVecClass;
    for (uint32_t half = 0; half < 2; half++) {
        uint32_t bits = classes[half];
        for (uint32_t i = 0; i < 16; i++) {
            const uint32_t cls = bits & 3u;
            bits >>= 2;
            if (cls == 1u) {
                draw_vector_state<false, FLAT, PERSP, LEQ>(s);
            } else if (cls == 2u) {
                draw_vector_state<true, FLAT, PERSP, LEQ>(s);
            }
            sfpi::dst_reg++;
        }
    }
}

#if defined(RASTER_LESS_EQUAL)
constexpr bool kLessEqual = true;
#else
constexpr bool kLessEqual = false;
#endif
#if defined(RASTER_COVER_COUNT)
constexpr bool kCover = true;
#else
constexpr bool kCover = false;
#endif

// Entry point for the SFPU_UNARY_CALL macro: draws one record, choosing the flat or smooth colour
// path. (Batching a page of records into one call was measured and does not help; see
// docs/PERF.md.)
inline void raster_triangle(const volatile float* rec) {
    const uint32_t flags = reinterpret_cast<const volatile uint32_t*>(rec)[ttr::layout::kFlags];
    if (!kCover && (flags & ttr::layout::kFlagStateful)) {   // the coverage-count debug mode ignores state
        const bool leq = ((flags >> ttr::layout::kDepthFuncShift) & 3u) == 1u;   // ALWAYS is LESS against +infinity
        if (flags & ttr::layout::kFlagFlat) {
            if (leq) raster_triangle_state_impl<true, false, true>(rec, flags); else raster_triangle_state_impl<true, false, false>(rec, flags);
        } else if (flags & ttr::layout::kFlagPersp) {
            if (leq) raster_triangle_state_impl<false, true, true>(rec, flags); else raster_triangle_state_impl<false, true, false>(rec, flags);
        } else {
            if (leq) raster_triangle_state_impl<false, false, true>(rec, flags); else raster_triangle_state_impl<false, false, false>(rec, flags);
        }
        return;
    }
    if (flags & ttr::layout::kFlagFlat) {
        raster_triangle_impl<true, false, kLessEqual, kCover>(rec);
    } else if (flags & ttr::layout::kFlagPersp) {
        raster_triangle_impl<false, true, kLessEqual, kCover>(rec);
    } else {
        raster_triangle_impl<false, false, kLessEqual, kCover>(rec);
    }
}

// Pack the four colour planes into RGBA8 words (R in the low byte), stored in the R plane's slot
// as a raw 32-bit integer (SFPSTORE INT32).
inline void raster_finalize_tile() {
#ifdef RASTER_PACKED_COLOUR
    return;  // colour was packed per pixel while drawing
#endif
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

}  // namespace ckernel::sfpu
