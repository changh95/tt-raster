/* SPDX-License-Identifier: Apache-2.0
 *
 * Fragment shader instruction set, C view (the C++ master is core/src/tensix/kernels/shader_isa.h; core/tests/test_shader.cpp checks
 * that the two agree). A program is an array of instructions of four 32-bit words:
 *   word 0:   opcode in bits 0..7, flags in bits 8..15, destination register in bits 16..31
 *   word 1..3 operands a, b, c: bits 0..11 index, bit 12 constant (index into the constant table, else a register),
 *             bit 13 negate, bit 14 absolute value
 * Semantics per lane (see the C++ header for the full description): one float per pixel in every register; the program
 * reads registers 0..num_inputs-1 (the interpolated varying components), optionally four registers holding the pixel's
 * x, y, z and 1/w, and constants; it leaves the fragment colour R, G, B, A in four consecutive registers.
 */
#ifndef TTRASTER_ISA_H
#define TTRASTER_ISA_H

#include <stdint.h>

enum {
    TTR_OP_NOP = 0,
    TTR_OP_MOV,   /* d = a */
    TTR_OP_ADD,   /* d = a + b */
    TTR_OP_MUL,   /* d = a * b */
    TTR_OP_MAD,   /* d = a * b + c, fused (the device: within one ulp of max(|a*b|, |c|)) */
    TTR_OP_MIN,   /* d = b < a ? b : a */
    TTR_OP_MAX,   /* d = b > a ? b : a */
    TTR_OP_FRC,   /* d = a - floor(a) */
    TTR_OP_FLR,   /* d = floor(a) */
    TTR_OP_RCP,   /* d = 1 / a */
    TTR_OP_RSQ,   /* d = 1 / sqrt(|a|) */
    TTR_OP_SQRT,  /* d = sqrt(|a|) */
    TTR_OP_EX2,   /* d = 2^a */
    TTR_OP_LG2,   /* d = log2(|a|) */
    TTR_OP_SIN,   /* d = sin(a) */
    TTR_OP_COS,   /* d = cos(a) */
    TTR_OP_SLT,   /* d = a <  b ? 1 : 0 */
    TTR_OP_SGE,   /* d = a >= b ? 1 : 0 */
    TTR_OP_SEQ,   /* d = a == b ? 1 : 0 */
    TTR_OP_SNE,   /* d = a != b ? 1 : 0 */
    TTR_OP_CMP,   /* d = a < 0 ? b : c */
    TTR_OP_SEL,   /* d = a != 0 ? b : c */
    TTR_OP_KILL,  /* the fragment is discarded if a < 0 (no destination) */
    TTR_OP_COUNT
};

#define TTR_FLAG_SAT 1u        /* clamp the result to [0, 1] (NaN gives 0) */
#define TTR_SRC_INDEX_MASK 0xFFFu
#define TTR_SRC_CONST (1u << 12)
#define TTR_SRC_NEG (1u << 13)
#define TTR_SRC_ABS (1u << 14)

#define TTR_MAX_REGS 61
#define TTR_MAX_CONSTS 1024
#define TTR_MAX_INSTRUCTIONS 1024
#define TTR_MAX_INPUTS 32
#define TTR_INSTR_WORDS 4

#endif
