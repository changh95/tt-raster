// SPDX-License-Identifier: Apache-2.0
//
// Fragment shader interpreter for the SFPU (docs/FINDINGS.md, "Fragment shaders"). Frames with shader triangles use
// this path for every triangle: a fixed-function triangle is a program with four inputs, no instructions and its
// colour in registers 0..3.
//
// Dest layout (FP32, full sync): as ckernel_sfpu_raster.h, but colour is one packed RGBA8 word per pixel in the R slot
// and the G, B and A slots are the shader's scratch registers. A tile slot has 32 vector rows and the destination counter
// advances one vector per iteration, so register r of vector v is row 64 + r + v: registers of different vectors overlap,
// which is harmless because a register is always written before it is read (validate_shader guarantees it). The highest
// row is 64 + 64 + 31 = 159, inside the three scratch slots, so at most kMaxRegs + 3 rows of registers exist.
//   0 depth   1 packed colour   2..4 shader registers   5 dx   6 dy   7 triangle id
//
// The interpreter runs per vector (32 lanes): it fetches an instruction from the program table in L1, reads its operands
// with runtime-indexed dst_reg, and issues the SFPU operations. Conditionals are lane predication (v_if), as everywhere
// else in the kernels.

#pragma once

#include <cstdint>

#include "ckernel_sfpu_raster.h"
#include "raster_layout.h"
#include "shader_isa.h"

namespace ckernel::sfpu {

constexpr uint32_t kRegBase = kSlotG * kVectors;                     // first scratch row (Dest slot 2)
// Device-internal registers above the program's: the "active" flag of the lanes (>= 0: the pixel is inside the triangle,
// passed the depth test and has not been killed; -1 otherwise) and two blend coefficients kept in Dest.
constexpr uint32_t kKillReg = ttr::shader::kMaxRegs;
constexpr uint32_t kTmpA = kKillReg + 1, kTmpB = kKillReg + 2;

// ---- Scalar state of one triangle, loaded from its record and program once ---------------------------------------
struct ShTri {
    float a0, b0, c0, a1, b1, c1, a2, b2, c2;       // edges
    float zc, zx, zy;                               // depth plane
    float wc, wx, wy;                               // rhw plane
    float id, cx, cy;                               // triangle id, tile centre in window pixels
    uint32_t depth_func, depth_write;
    float ka_c, ka_sa, kb_c, kb_sa, kb_s;
    bool blend, persp;
    const volatile float* rec;                      // the record: input planes at words 12.. (first four) and 32.. (the rest)
    const volatile uint32_t* code;
    const volatile float* consts;
    uint32_t n_instr, n_in, out_reg, uses_kill;
    int32_t fc_reg;
};

inline void load_shader_tri(ShTri& s, const volatile float* rec, const volatile uint32_t* table, float cx, float cy) {
    using namespace ttr::layout;
    const volatile uint32_t* u = reinterpret_cast<const volatile uint32_t*>(rec);
    s.a0 = rec[0]; s.b0 = rec[1]; s.c0 = rec[2];
    s.a1 = rec[3]; s.b1 = rec[4]; s.c1 = rec[5];
    s.a2 = rec[6]; s.b2 = rec[7]; s.c2 = rec[8];
    s.zc = rec[9]; s.zx = rec[10]; s.zy = rec[11];
    s.id = rec[kId];
    s.cx = cx;
    s.cy = cy;
    const uint32_t flags = u[kFlags];
    s.persp = (flags & kFlagPersp) != 0u;
    s.depth_func = (flags >> kDepthFuncShift) & 3u;
    s.depth_write = (flags & kFlagDepthWrite) != 0u;
    const uint32_t mode = (flags >> kBlendShift) & 7u;
    s.blend = mode != 0u;
    s.ka_c = kBlendCoef[mode][0];
    s.ka_sa = kBlendCoef[mode][1];
    s.kb_c = kBlendCoef[mode][2];
    s.kb_sa = kBlendCoef[mode][3];
    s.kb_s = kBlendCoef[mode][4];
    s.rec = rec;
    s.wc = rec[kWPlane]; s.wx = rec[kWPlane + 1]; s.wy = rec[kWPlane + 2];
    if (flags & kFlagShader) {
        const volatile uint32_t* dir = table + kTabDirectory + kTabEntryWords * u[kProgramId];
        s.code = table + dir[0];
        s.n_instr = dir[1];
        s.consts = reinterpret_cast<const volatile float*>(table + dir[2]);
        s.n_in = dir[3];
        s.fc_reg = static_cast<int32_t>(dir[4]) - 1;
        s.out_reg = dir[5];
        s.uses_kill = dir[6];
    } else {
        s.code = nullptr;
        s.consts = nullptr;
        s.n_instr = 0;
        s.n_in = 4;            // the colour planes are the inputs, the colour is the output
        s.fc_reg = -1;
        s.out_reg = 0;
        s.uses_kill = 0;
    }
}

// ---- Arithmetic helpers (documented tolerances in docs/FINDINGS.md) ----------------------------------------------

// Round to nearest integer value, |x| < 2^22 (the magic constant 1.5 * 2^23 puts the sum in [2^23, 2^24), ulp 1).
__attribute__((always_inline)) inline sfpi::vFloat sh_round(sfpi::vFloat x) {
    sfpi::vFloat t = x + 12582912.0f;
    return t - 12582912.0f;
}

__attribute__((always_inline)) inline sfpi::vFloat sh_floor(sfpi::vFloat x) {
    sfpi::vFloat r = sh_round(x);
    v_if (r > x) { r = r - 1.0f; }
    v_endif;
    return r;
}

// 1/x for any non-zero x: the sign is handled here, the magnitude by the Newton-refined reciprocal estimate.
__attribute__((always_inline)) inline sfpi::vFloat sh_rcp(sfpi::vFloat x) {
    sfpi::vFloat ax = sfpi::abs(x);
    sfpi::vFloat y = raster_reciprocal(ax);
    v_if (x < 0.0f) { y = -y; }
    v_endif;
    v_if (ax == 0.0f) { y = 3.0e38f * 10.0f; }   // +inf
    v_endif;
    return y;
}

// 1/sqrt(x) for x > 0: bit-pattern estimate refined by three Newton steps.
__attribute__((always_inline)) inline sfpi::vFloat sh_rsq(sfpi::vFloat x) {
    sfpi::vInt i = sfpi::as<sfpi::vInt>(x);
    i = sfpi::vInt(0x5f3759df) - (i >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(i);
    sfpi::vFloat hx = x * 0.5f;
    y = y * (1.5f - hx * y * y);
    y = y * (1.5f - hx * y * y);
    y = y * (1.5f - hx * y * y);
    return y;
}

// 2^x: x clamped to [-126, 127], integer part into the exponent, fraction by a degree 5 polynomial (relative error
// 1.2e-7, one ulp).
__attribute__((always_inline)) inline sfpi::vFloat sh_ex2(sfpi::vFloat x) {
    v_if (x < -126.0f) { x = -126.0f; }
    v_endif;
    v_if (x > 127.0f) { x = 127.0f; }
    v_endif;
    sfpi::vFloat n = sh_floor(x);
    sfpi::vFloat f = x - n;
    sfpi::vFloat p = f * 0.0018951073f + 0.0089462148f;
    p = p * f + 0.055863284f;
    p = p * f + 0.24014077f;
    p = p * f + 0.69315463f;
    p = p * f + 0.99999988f;
    sfpi::vUInt e = sfpi::convert<sfpi::vUInt16>(n + 127.0f, sfpi::RoundMode::Nearest);
    sfpi::vFloat scale = sfpi::setexp(sfpi::vFloat(1.0f), sfpi::as<sfpi::vInt>(e));
    return p * scale;
}

// log2(|x|), x != 0: exponent plus an odd series in (m - 1) / (m + 1) for the mantissa m in [sqrt(1/2), sqrt(2)).
__attribute__((always_inline)) inline sfpi::vFloat sh_lg2(sfpi::vFloat x) {
    x = sfpi::abs(x);
    sfpi::vInt ei = sfpi::exexp(x);
    sfpi::vFloat e = sfpi::convert<sfpi::vFloat>(ei, sfpi::RoundMode::Nearest);
    sfpi::vFloat m = sfpi::setexp(x, 127);
    v_if (m > 1.41421356f) {
        m = m * 0.5f;
        e = e + 1.0f;
    }
    v_endif;
    sfpi::vFloat t = (m - 1.0f) * raster_reciprocal(m + 1.0f);
    sfpi::vFloat t2 = t * t;
    sfpi::vFloat s = t2 * 0.142857143f + 0.2f;
    s = s * t2 + 0.333333333f;
    s = s * t2 + 1.0f;
    return e + t * s * 2.88539008f;   // 2 / ln 2
}

// sin(x): reduce to [-pi, pi], fold to [-pi/2, pi/2], odd Taylor polynomial of degree 11 (error 6e-8 at pi/2).
__attribute__((always_inline)) inline sfpi::vFloat sh_sin(sfpi::vFloat x) {
    sfpi::vFloat k = sh_round(x * 0.159154943f);
    sfpi::vFloat r = x - k * 6.28318548f;
    r = r + k * 1.74845553e-7f;                // 2 pi = 6.28318548 - 1.7484555e-7 (second part of the constant)
    v_if (r > 1.57079633f) { r = 3.14159265f - r; }
    v_endif;
    v_if (r < -1.57079633f) { r = -3.14159265f - r; }
    v_endif;
    sfpi::vFloat r2 = r * r;
    sfpi::vFloat p = r2 * -2.50521084e-8f + 2.75573192e-6f;
    p = p * r2 + -1.98412698e-4f;
    p = p * r2 + 8.33333333e-3f;
    p = p * r2 + -1.66666667e-1f;
    p = p * r2 + 1.0f;
    return r * p;
}

__attribute__((always_inline)) inline sfpi::vFloat sh_sat(sfpi::vFloat v) {
    v_if (v < 0.0f) { v = 0.0f; }
    v_endif;
    v_if (v > 1.0f) { v = 1.0f; }
    v_endif;
    return v;
}

// Operand: register or constant, with the absolute value and negate modifiers.
__attribute__((always_inline)) inline sfpi::vFloat sh_src(uint32_t w, const volatile float* consts) {
    using namespace ttr::shader;
    sfpi::vFloat v;
    const uint32_t idx = w & kSrcIndexMask;
    if (w & kSrcConst) {
        v = consts[idx];
    } else {
        v = sfpi::dst_reg[kRegBase + idx];
    }
    if (w & kSrcAbs) { v = sfpi::abs(v); }
    if (w & kSrcNeg) { v = -v; }
    return v;
}

// Run the program for the current vector on all lanes (results of lanes that are not active are never used); Kill clears the
// active flag. Inlined at its single call site: a call from inside predicated code does not keep the lane predicate.
__attribute__((always_inline)) inline void sh_run(const ShTri& s) {
    using namespace ttr::shader;
    for (uint32_t n = 0; n < s.n_instr; n++) {
        const volatile uint32_t* in = s.code + n * kInstrWords;
        const uint32_t w0 = in[0];
        const uint32_t op = instr_op(w0);
        sfpi::vFloat a = sh_src(in[1], s.consts);
        sfpi::vFloat b = sh_src(in[2], s.consts);   // unused operands are zero: they read register 0, harmlessly
        sfpi::vFloat c = sh_src(in[3], s.consts);
        sfpi::vFloat d = a;
        switch (op) {
            case kOpMov: break;
            case kOpAdd: d = a + b; break;
            case kOpMul: d = a * b; break;
            case kOpMad: d = a * b + c; break;
            case kOpMin: v_if (b < a) { d = b; } v_endif; break;
            case kOpMax: v_if (b > a) { d = b; } v_endif; break;
            case kOpFrc: d = a - sh_floor(a); break;
            case kOpFlr: d = sh_floor(a); break;
            case kOpRcp: d = sh_rcp(a); break;
            case kOpRsq: d = sh_rsq(sfpi::abs(a)); break;
            case kOpSqrt: {
                sfpi::vFloat x = sfpi::abs(a);
                d = x * sh_rsq(x);
                v_if (x == 0.0f) { d = 0.0f; }
                v_endif;
                break;
            }
            case kOpEx2: d = sh_ex2(a); break;
            case kOpLg2: d = sh_lg2(a); break;
            case kOpSin: d = sh_sin(a); break;
            case kOpCos: d = sh_sin(a + 1.57079633f); break;
            case kOpSlt: d = 0.0f; v_if (a < b) { d = 1.0f; } v_endif; break;
            case kOpSge: d = 0.0f; v_if (a >= b) { d = 1.0f; } v_endif; break;
            case kOpSeq: d = 0.0f; v_if (a == b) { d = 1.0f; } v_endif; break;
            case kOpSne: d = 0.0f; v_if (a != b) { d = 1.0f; } v_endif; break;
            case kOpCmp: d = c; v_if (a < 0.0f) { d = b; } v_endif; break;
            case kOpSel: d = c; v_if (a != 0.0f) { d = b; } v_endif; break;
            case kOpKill:
                v_if (a < 0.0f) { sfpi::dst_reg[kRegBase + kKillReg] = -1.0f; }
                v_endif;
                continue;
            default: break;
        }
        if (instr_flags(w0) & kFlagSat) { d = sh_sat(d); }
        sfpi::dst_reg[kRegBase + instr_dst(w0)] = d;
    }
}

// Quantize like src/ttraster/frame.cpp (see raster_quantize) and pack R in the low byte.
__attribute__((always_inline)) inline sfpi::vUInt sh_pack(sfpi::vFloat r, sfpi::vFloat g, sfpi::vFloat b, sfpi::vFloat a) {
    sfpi::vUInt qr = raster_quantize(r);
    sfpi::vUInt qg = raster_quantize(g);
    sfpi::vUInt qb = raster_quantize(b);
    sfpi::vUInt qa = raster_quantize(a);
    return qr | (qg << 8) | (qb << 16) | (qa << 24);
}

// Unpack channel ch (0..3) of a packed RGBA8 word to [0, 1]: byte * (1/255), as the reference.
__attribute__((always_inline)) inline sfpi::vFloat sh_unpack(sfpi::vUInt packed, uint32_t ch) {
    sfpi::vUInt byte = (packed >> (8 * ch)) & 0xFFu;
    return sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vInt>(byte), sfpi::RoundMode::Nearest) * (1.0f / 255.0f);
}

// The fragment colour in registers out..out+3, written to the packed colour, blended with it if the triangle blends
// (ttr::blend_channel, src/ttraster/tile.h).
__attribute__((always_inline)) inline void sh_write_colour(const ShTri& s) {
    const uint32_t o = kRegBase + s.out_reg;
    if (!s.blend) {
        sfpi::vFloat r = sfpi::dst_reg[o], g = sfpi::dst_reg[o + 1], b = sfpi::dst_reg[o + 2], a = sfpi::dst_reg[o + 3];
        sfpi::dst_reg[kSlotR * kVectors] = sh_pack(r, g, b, a);
        return;
    }
    sfpi::vFloat sa = sh_sat(sfpi::dst_reg[o + 3]);
    sfpi::dst_reg[kRegBase + kTmpA] = sa * s.ka_sa + s.ka_c;
    sfpi::dst_reg[kRegBase + kTmpB] = sa * s.kb_sa + s.kb_c;
    sfpi::vUInt acc = 0u;
#pragma GCC unroll 4
    for (uint32_t ch = 0; ch < 4; ch++) {
        sfpi::vFloat c = sh_sat(sfpi::dst_reg[o + ch]);
        sfpi::vFloat d = sh_unpack(sfpi::vUInt(sfpi::dst_reg[kSlotR * kVectors]), ch);
        sfpi::vFloat bb = c * s.kb_s + sfpi::dst_reg[kRegBase + kTmpB];
        sfpi::vFloat r = c * sfpi::dst_reg[kRegBase + kTmpA] + d * bb;
        acc = acc | (raster_quantize(sh_sat(r)) << (8 * ch));
    }
    sfpi::dst_reg[kSlotR * kVectors] = acc;
}

// Phase 1 of a vector: the active flag. Lanes inside the triangle (edge tests, unless the whole vector is inside) that pass the
// depth test get 0, all others -1.
__attribute__((always_inline)) inline void sh_depth_pass(const ShTri& s, sfpi::vFloat dx, sfpi::vFloat dy) {
    sfpi::vFloat z = s.zx * dx + (s.zy * dy + s.zc);
    sfpi::vFloat zo = sfpi::dst_reg[kSlotZ * kVectors];
    if (s.depth_func == 2u) { zo = 3.0e38f; }   // depth test off
    if (s.depth_func == 1u) {
        v_if (z <= zo) { sfpi::dst_reg[kRegBase + kKillReg] = 0.0f; }
        v_endif;
    } else {
        v_if (z < zo) { sfpi::dst_reg[kRegBase + kKillReg] = 0.0f; }
        v_endif;
    }
}

__attribute__((always_inline)) inline void sh_active_partial(const ShTri& s) {
    sfpi::dst_reg[kRegBase + kKillReg] = -1.0f;
    sfpi::vFloat dx = sfpi::dst_reg[kSlotDx * kVectors];
    sfpi::vFloat dy = sfpi::dst_reg[kSlotDy * kVectors];
    sfpi::vFloat e0 = s.a0 * dx + (s.b0 * dy + s.c0);
    v_if (e0 >= 0.0f) {
        sfpi::vFloat e1 = s.a1 * dx + (s.b1 * dy + s.c1);
        v_if (e1 >= 0.0f) {
            sfpi::vFloat e2 = s.a2 * dx + (s.b2 * dy + s.c2);
            v_if (e2 >= 0.0f) { sh_depth_pass(s, dx, dy); }
            v_endif;
        }
        v_endif;
    }
    v_endif;
}

__attribute__((always_inline)) inline void sh_active_full(const ShTri& s) {
    sfpi::dst_reg[kRegBase + kKillReg] = -1.0f;
    sfpi::vFloat dx = sfpi::dst_reg[kSlotDx * kVectors];
    sfpi::vFloat dy = sfpi::dst_reg[kSlotDy * kVectors];
    sh_depth_pass(s, dx, dy);
}

// Phases 2 and 3: inputs, program, then the writes of the lanes that are still active.
__attribute__((always_inline)) inline void sh_shade(const ShTri& s) {
    using namespace ttr::shader;
    sfpi::vFloat dx = sfpi::dst_reg[kSlotDx * kVectors];
    sfpi::vFloat dy = sfpi::dst_reg[kSlotDy * kVectors];
    sfpi::vFloat z = s.zx * dx + (s.zy * dy + s.zc);
    sfpi::vFloat wv = s.wx * dx + (s.wy * dy + s.wc);   // interpolated rhw
    sfpi::vFloat q = 1.0f;
    if (s.persp) {
        q = raster_reciprocal(wv);
    }
    for (uint32_t k = 0; k < s.n_in; k++) {
        const volatile float* p = k < 4 ? s.rec + ttr::layout::kColor + 3 * k : s.rec + 32 + 3 * (k - 4);
        sfpi::vFloat v = p[1] * dx + (p[2] * dy + p[0]);
        if (s.persp) { v = v * q; }
        sfpi::dst_reg[kRegBase + k] = v;
    }
    if (s.fc_reg >= 0) {
        const uint32_t f = kRegBase + static_cast<uint32_t>(s.fc_reg);
        sfpi::dst_reg[f] = dx * 0.0625f + s.cx;
        sfpi::dst_reg[f + 1] = dy * 0.0625f + s.cy;
        sfpi::dst_reg[f + 2] = z;
        sfpi::dst_reg[f + 3] = wv;
    }
    sh_run(s);
    sfpi::vFloat active = sfpi::dst_reg[kRegBase + kKillReg];
    v_if (active >= 0.0f) {
        if (s.depth_write) { sfpi::dst_reg[kSlotZ * kVectors] = z; }
        sfpi::dst_reg[kSlotId * kVectors] = s.id;
        sh_write_colour(s);
    }
    v_endif;
}

// Entry point for the SFPU call macro: one triangle record of a shader frame.
inline void raster_triangle_shader(const volatile float* rec, const volatile uint32_t* table, float cx, float cy) {
    ShTri s;
    load_shader_tri(s, rec, table, cx, cy);
    const volatile uint32_t* classes = reinterpret_cast<const volatile uint32_t*>(rec) + ttr::layout::kVecClass;
    for (uint32_t half = 0; half < 2; half++) {
        uint32_t bits = classes[half];
        for (uint32_t i = 0; i < 16; i++) {
            const uint32_t cls = bits & 3u;
            bits >>= 2;
            if (cls != 0u) {
                if (cls == 1u) {
                    sh_active_partial(s);
                } else {
                    sh_active_full(s);
                }
                sh_shade(s);   // the only call: one copy of the interpreter
            }
            sfpi::dst_reg++;
        }
    }
}

}  // namespace ckernel::sfpu
