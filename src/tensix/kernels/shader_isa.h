// SPDX-License-Identifier: Apache-2.0
//
// Fragment shader instruction set (docs/FINDINGS.md, "Fragment shaders"). No includes beyond <cstdint>: this header is
// compiled by the host (the CPU interpreter, the Mesa driver) and by the Tensix compute kernel.
//
// A shader is a straight-line program over vector registers: one float per pixel in every register, so an instruction
// is one SFPU operation per 32-lane vector on the device and one scalar operation per pixel in the CPU reference. There
// are no branches; conditional code is written with the compare and select instructions (if-conversion is the
// producer's job). The program reads
//   * registers r[0 .. num_inputs): the interpolated varying components of the pixel,
//   * registers r[fragcoord .. +4): x, y, z, 1/w of the pixel (if the program asks for them),
//   * constants: uniforms and literals, a table of scalars shared by all pixels,
// and leaves the fragment colour (R, G, B, A) in r[out .. out+4). Reading a register that nothing has written is
// rejected by validate_shader(), so a program never depends on stale storage.
//
// Instruction: 4 words.
//   word 0:  bits 0..7 opcode, bits 8..15 flags, bits 16..31 destination register
//   word 1..3: operands a, b, c (unused operands are zero):
//       bits 0..11 index, bit 12 constant (index into the constant table, else a register),
//       bit 13 negate, bit 14 absolute value (applied first, then the negate)
// Flags: kSat clamps the result to [0, 1] after the operation (NaN gives 0).
//
// Semantics, per lane, in IEEE binary32 with round to nearest. Operands are read after their modifiers.
//   Mov  d = a                        Add  d = a + b                  Mul  d = a * b
//   Mad  d = fma(a, b, c)  (single rounding)
//   Min  d = b < a ? b : a            Max  d = b > a ? b : a
//   Frc  d = a - floor(a)             Flr  d = floor(a)
//   Rcp  d = 1 / a                    Rsq  d = 1 / sqrt(|a|)          Sqrt d = sqrt(|a|)
//   Ex2  d = 2^a                      Lg2  d = log2(|a|)
//   Sin  d = sin(a)                   Cos  d = cos(a)                 (radians)
//   Slt  d = a <  b ? 1 : 0           Sge  d = a >= b ? 1 : 0
//   Seq  d = a == b ? 1 : 0           Sne  d = a != b ? 1 : 0
//   Cmp  d = a < 0 ? b : c            Sel  d = a != 0 ? b : c
//   Kill the fragment is discarded if a < 0 in this lane (no destination)
// The transcendental functions (Rcp, Rsq, Sqrt, Ex2, Lg2, Sin, Cos) agree between host and device to a few ulps, not
// bit for bit; everything else is exact.

#pragma once

#include <cstdint>

namespace ttr::shader {

constexpr uint32_t kMaxRegs = 96;          // vector registers: three Dest tiles of 32 vectors on the device
constexpr uint32_t kMaxConsts = 1024;      // scalars in the constant table
constexpr uint32_t kMaxInstructions = 1024;
constexpr uint32_t kMaxInputs = 32;        // varying components (eight vec4)
constexpr uint32_t kInstrWords = 4;

enum Op : uint32_t {
    kOpNop = 0,
    kOpMov,
    kOpAdd,
    kOpMul,
    kOpMad,
    kOpMin,
    kOpMax,
    kOpFrc,
    kOpFlr,
    kOpRcp,
    kOpRsq,
    kOpSqrt,
    kOpEx2,
    kOpLg2,
    kOpSin,
    kOpCos,
    kOpSlt,
    kOpSge,
    kOpSeq,
    kOpSne,
    kOpCmp,
    kOpSel,
    kOpKill,
    kOpCount,
};

constexpr uint32_t kFlagSat = 1u;

constexpr uint32_t kSrcIndexMask = 0xFFFu;
constexpr uint32_t kSrcConst = 1u << 12;
constexpr uint32_t kSrcNeg = 1u << 13;
constexpr uint32_t kSrcAbs = 1u << 14;

// Number of source operands an opcode reads (the rest must be zero), and whether it writes a register.
constexpr uint32_t op_sources(uint32_t op) {
    switch (op) {
        case kOpMov: case kOpFrc: case kOpFlr: case kOpRcp: case kOpRsq: case kOpSqrt: case kOpEx2: case kOpLg2:
        case kOpSin: case kOpCos: case kOpKill: return 1;
        case kOpAdd: case kOpMul: case kOpMin: case kOpMax: case kOpSlt: case kOpSge: case kOpSeq: case kOpSne: return 2;
        case kOpMad: case kOpCmp: case kOpSel: return 3;
        default: return 0;
    }
}
constexpr bool op_writes(uint32_t op) { return op != kOpNop && op != kOpKill; }

constexpr uint32_t make_word0(uint32_t op, uint32_t flags, uint32_t dst) { return op | (flags << 8) | (dst << 16); }
constexpr uint32_t instr_op(uint32_t w0) { return w0 & 0xFFu; }
constexpr uint32_t instr_flags(uint32_t w0) { return (w0 >> 8) & 0xFFu; }
constexpr uint32_t instr_dst(uint32_t w0) { return w0 >> 16; }

}  // namespace ttr::shader
