/*
 * Copyright 2026 tt-raster contributors
 * SPDX-License-Identifier: MIT
 *
 * TGSI fragment shader to libttraster shader program. See sp_tensix_shader.h.
 *
 * Pipeline: (1) parse the TGSI and expand every instruction per component into virtual-register instructions (a vreg is
 * one float per pixel); inputs, gl_FragCoord, outputs, temporaries and scratch values each have a range of vreg numbers.
 * Control flow is if-converted with lane masks. (2) Allocate the vregs to at most 57 physical registers by a linear scan
 * (the code is straight-line, so a register is live from its first definition to its last use); the four colour outputs
 * get the four registers above everything else. (3) Emit the instruction words: operands are registers or entries of
 * the constant table (TGSI immediates and the uniform floats the program reads).
 */

#include "sp_tensix_shader.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pipe/p_shader_tokens.h"
#include "tgsi/tgsi_info.h"
#include "tgsi/tgsi_parse.h"
#include "util/u_memory.h"

#include "ttraster_isa.h"

#define MAX_IN_DECL 32
#define MAX_TEMPS 128
#define MAX_TIMM 128
#define MAX_PHYS 57          /* registers for everything but the four colour outputs */
#define MAX_IMM_SLOTS 640    /* of TTR_MAX_CONSTS: the rest is for uniforms */
#define MAX_VI 4096

/* vreg numbering */
enum { V_IN = 0, V_FC = 32, V_OUT = 36, V_TEMP = 40 };
#define V_SCRATCH (V_TEMP + 4 * MAX_TEMPS)
#define MAX_VREGS (V_SCRATCH + 4096)

enum { K_NONE = 0, K_REG, K_IMM, K_UNI };

/* A uniform slot holding the framebuffer height (minus half a pixel for an integer pixel centre), filled per draw. */
#define HEIGHT_MARK 0xFFFFFFFFu

struct opnd {
   uint8_t kind, neg, abs;
   int32_t id; /* vreg, immediate index or uniform index */
};

struct vi {
   uint8_t op, sat;
   int32_t dst; /* vreg or -1 */
   struct opnd s[3];
};

struct ifctx {
   int mask, cond, parent;
   int if_mask;   /* on the ELSE side: the mask of the IF side, whose comparison results join the ELSE side's (store_lane); else -1 */
};

struct xl {
   struct vi *v;
   unsigned n;
   uint32_t imm[MAX_IMM_SLOTS];
   unsigned n_imm;
   uint32_t uni[TTR_MAX_CONSTS];
   unsigned n_uni;
   uint8_t *defined;
   /* A vreg that holds a comparison result (1.0 or 0.0): written by a compare, AND/OR/XOR/NOT of such, or a move of one. A write
    * under a lane mask only makes the register a comparison result for the lanes of that mask, so reads inside the region (the
    * mask is on the stack) see one, reads after the ENDIF do not unless the value it overwrote was one too (bool_scope: the mask
    * the property is limited to, -1 for all lanes). */
   uint8_t *is_bool;
   int32_t *bool_scope;
   int next_scratch;
   int in_slot[MAX_IN_DECL][4];
   unsigned n_in;
   struct ttr_prog_input map[32];
   int semantic[MAX_IN_DECL];
   int position_input;
   bool fc_used;
   uint32_t timm[MAX_TIMM][4];
   unsigned n_timm;
   struct ifctx stack[16];
   int depth;
   bool out_written[4];
   bool uses_kill;
   bool origin_lower_left, pixel_center_integer;   /* gl_FragCoord conventions of the shader (TGSI properties) */
   int fcx, fcy;                                    /* vregs of the converted fragment coordinate, -1 until needed */
   const char *why;
};

static bool
fail(struct xl *x, const char *why)
{
   if (!x->why)
      x->why = why;
   return false;
}

static struct opnd
none(void)
{
   struct opnd o = {K_NONE, 0, 0, 0};
   return o;
}

static struct opnd
reg(int id)
{
   struct opnd o = {K_REG, 0, 0, id};
   return o;
}

static struct opnd
imm_bits(struct xl *x, uint32_t bits)
{
   struct opnd o = {K_IMM, 0, 0, 0};
   for (unsigned i = 0; i < x->n_imm; i++) {
      if (x->imm[i] == bits) {
         o.id = i;
         return o;
      }
   }
   if (x->n_imm >= MAX_IMM_SLOTS) {
      fail(x, "too many immediates");
      return o;
   }
   x->imm[x->n_imm] = bits;
   o.id = x->n_imm++;
   return o;
}

static struct opnd
imm_f(struct xl *x, float f)
{
   uint32_t bits;
   memcpy(&bits, &f, 4);
   return imm_bits(x, bits);
}

static struct opnd
uni_op(struct xl *x, uint32_t float_index)
{
   struct opnd o = {K_UNI, 0, 0, 0};
   for (unsigned i = 0; i < x->n_uni; i++) {
      if (x->uni[i] == float_index) {
         o.id = i;
         return o;
      }
   }
   if (x->n_uni >= TTR_MAX_CONSTS) {
      fail(x, "too many uniforms");
      return o;
   }
   x->uni[x->n_uni] = float_index;
   o.id = x->n_uni++;
   return o;
}

static int
new_scratch(struct xl *x)
{
   if (x->next_scratch >= MAX_VREGS) {
      fail(x, "shader too large");
      return V_SCRATCH;
   }
   return x->next_scratch++;
}

static void
emit(struct xl *x, uint8_t op, int dst, struct opnd a, struct opnd b, struct opnd c, bool sat)
{
   if (x->n >= MAX_VI) {
      fail(x, "shader too long");
      return;
   }
   struct vi *i = &x->v[x->n++];
   i->op = op;
   i->sat = sat;
   i->dst = dst;
   i->s[0] = a;
   i->s[1] = b;
   i->s[2] = c;
   if (dst >= 0) {
      x->defined[dst] = 1;
      x->is_bool[dst] = 0;   /* the instruction translators that produce a comparison result say so afterwards */
      x->bool_scope[dst] = -1;
   }
}

/* A register that was never written reads as 0 (TGSI leaves it undefined; the device must not read stale rows). */
static void
ensure_defined(struct xl *x, int id)
{
   if (!x->defined[id])
      emit(x, TTR_OP_MOV, id, imm_f(x, 0.0f), none(), none(), false);
}

static int
cur_mask(const struct xl *x)
{
   return x->depth ? x->stack[x->depth - 1].mask : -1;
}

/* Does register `v` hold a comparison result in the lanes that are active here? */
static bool
bool_now(const struct xl *x, int v)
{
   if (!x->is_bool[v])
      return false;
   if (x->bool_scope[v] < 0)
      return true;
   for (int i = 0; i < x->depth; i++)
      if (x->stack[i].mask == x->bool_scope[v])
         return true;
   return false;
}

/* The fragment coordinate as softpipe gives it to TGSI (sp_setup.c, setup_fragcoord_coeff): x + 0.5 or x for an integer pixel
 * centre, y counted from the bottom (framebuffer height - y) for a lower-left origin. The rasterizer's own coordinate is the
 * pixel centre from the top. */
static int
fc_reg(struct xl *x, unsigned c)
{
   if (c == 0 && x->pixel_center_integer) {
      if (x->fcx < 0) {
         x->fcx = new_scratch(x);
         emit(x, TTR_OP_ADD, x->fcx, reg(V_FC), imm_f(x, -0.5f), none(), false);
      }
      return x->fcx;
   }
   if (c == 1 && (x->origin_lower_left || x->pixel_center_integer)) {
      if (x->fcy < 0) {
         x->fcy = new_scratch(x);
         if (x->origin_lower_left)
            emit(x, TTR_OP_MAD, x->fcy, reg(V_FC + 1), imm_f(x, -1.0f), uni_op(x, HEIGHT_MARK), false);
         else
            emit(x, TTR_OP_ADD, x->fcy, reg(V_FC + 1), imm_f(x, -0.5f), none(), false);
      }
      return x->fcy;
   }
   return V_FC + c;
}

/* ---- Operands and destinations ----------------------------------------------------------------------------------- */

static bool
get_src(struct xl *x, const struct tgsi_full_src_register *s, unsigned lane, struct opnd *o)
{
   const unsigned swz[4] = {s->Register.SwizzleX, s->Register.SwizzleY, s->Register.SwizzleZ, s->Register.SwizzleW};
   const unsigned c = swz[lane];
   if (s->Register.Indirect)
      return fail(x, "indirect addressing");
   switch (s->Register.File) {
   case TGSI_FILE_INPUT: {
      const int j = s->Register.Index;
      if (j < 0 || j >= MAX_IN_DECL)
         return fail(x, "fragment input index");
      const int sem = x->semantic[j];
      if (sem == TGSI_SEMANTIC_POSITION) {
         x->position_input = j;
         x->fc_used = true;
         *o = reg(fc_reg(x, c));
      } else if (sem == TGSI_SEMANTIC_GENERIC || sem == TGSI_SEMANTIC_COLOR || sem == TGSI_SEMANTIC_TEXCOORD ||
                 sem == TGSI_SEMANTIC_FOG) {
         int slot = x->in_slot[j][c];
         if (slot < 0) {
            if (x->n_in >= TTR_MAX_INPUTS)
               return fail(x, "too many varying components");
            slot = x->n_in++;
            x->in_slot[j][c] = slot;
            x->map[slot].input = j;
            x->map[slot].comp = c;
            x->defined[V_IN + slot] = 1;
         }
         *o = reg(V_IN + slot);
      } else {
         return fail(x, "unsupported fragment shader input (front face, primitive id, ...)");
      }
      break;
   }
   case TGSI_FILE_TEMPORARY: {
      if (s->Register.Index < 0 || s->Register.Index >= MAX_TEMPS)
         return fail(x, "too many temporaries");
      *o = reg(V_TEMP + 4 * s->Register.Index + c);
      ensure_defined(x, o->id);
      break;
   }
   case TGSI_FILE_OUTPUT:
      if (s->Register.Index != 0)
         return fail(x, "reading an output other than colour 0");
      *o = reg(V_OUT + c);
      ensure_defined(x, o->id);
      break;
   case TGSI_FILE_IMMEDIATE:
      if (s->Register.Index < 0 || (unsigned)s->Register.Index >= x->n_timm)
         return fail(x, "immediate index");
      {
         /* "True" of an integer mask is all ones, which as a float is a NaN: comparison results are 1.0 in this instruction set
          * (see f_and), so the constant is 1.0 here. NIR writes it for the taken side of a short-circuit || or &&. */
         const uint32_t bits = x->timm[s->Register.Index][c];
         *o = imm_bits(x, bits == 0xffffffffu ? 0x3f800000u : bits);
      }
      break;
   case TGSI_FILE_CONSTANT:
      if (s->Register.Dimension && (s->Dimension.Index != 0 || s->Dimension.Indirect))
         return fail(x, "uniform buffer object");
      if (s->Register.Index < 0)
         return fail(x, "constant index");
      *o = uni_op(x, (uint32_t)s->Register.Index * 4 + c);
      break;
   default:
      return fail(x, "unsupported source register file");
   }
   o->neg = s->Register.Negate;
   o->abs = s->Register.Absolute;
   return !x->why;
}

static bool
get_dst(struct xl *x, const struct tgsi_full_dst_register *d, int ids[4])
{
   if (d->Register.Indirect || d->Register.Dimension)
      return fail(x, "indirect destination");
   for (int c = 0; c < 4; c++) {
      if (!(d->Register.WriteMask & (1u << c)))
         continue;
      if (d->Register.File == TGSI_FILE_TEMPORARY) {
         if (d->Register.Index < 0 || d->Register.Index >= MAX_TEMPS)
            return fail(x, "too many temporaries");
         ids[c] = V_TEMP + 4 * d->Register.Index + c;
      } else if (d->Register.File == TGSI_FILE_OUTPUT) {
         if (d->Register.Index != 0)
            return fail(x, "output other than colour 0");
         ids[c] = V_OUT + c;
         x->out_written[c] = true;
      } else {
         return fail(x, "unsupported destination register file");
      }
   }
   return true;
}

static bool
same_opnd(const struct opnd *a, const struct opnd *b)
{
   return a->kind == b->kind && a->id == b->id && a->neg == b->neg && a->abs == b->abs;
}

/* ---- Per-lane expansions ----------------------------------------------------------------------------------------- */

typedef bool (*lane_fn)(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat);

#define SRC(si, lane, var)                                          \
   struct opnd var;                                                 \
   if (!get_src(x, &inst->Src[si], lane, &var))                     \
      return false;

#define LANE_OP1(NAME, OP)                                                                                        \
   static bool NAME(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat)    \
   {                                                                                                              \
      SRC(0, lane, a)                                                                                             \
      emit(x, OP, dst, a, none(), none(), sat);                                                                   \
      return true;                                                                                                \
   }
#define LANE_OP2(NAME, OP)                                                                                        \
   static bool NAME(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat)    \
   {                                                                                                              \
      SRC(0, lane, a)                                                                                             \
      SRC(1, lane, b)                                                                                             \
      emit(x, OP, dst, a, b, none(), sat);                                                                        \
      return true;                                                                                                \
   }
#define LANE_OP2_SWAPPED(NAME, OP)                                                                                \
   static bool NAME(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat)    \
   {                                                                                                              \
      SRC(0, lane, a)                                                                                             \
      SRC(1, lane, b)                                                                                             \
      emit(x, OP, dst, b, a, none(), sat);                                                                        \
      return true;                                                                                                \
   }
/* Comparisons: 1.0 or 0.0, which is also what the integer masks of FSLT & co. are in this instruction set (see f_and). */
#define LANE_CMP(NAME, OP, SWAP)                                                                                  \
   static bool NAME(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat)    \
   {                                                                                                              \
      SRC(0, lane, a)                                                                                             \
      SRC(1, lane, b)                                                                                             \
      emit(x, OP, dst, SWAP ? b : a, SWAP ? a : b, none(), sat);                                                  \
      x->is_bool[dst] = !sat;                                                                                     \
      return true;                                                                                                \
   }
#define LANE_OP3(NAME, OP)                                                                                        \
   static bool NAME(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat)    \
   {                                                                                                              \
      SRC(0, lane, a)                                                                                             \
      SRC(1, lane, b)                                                                                             \
      SRC(2, lane, c)                                                                                             \
      emit(x, OP, dst, a, b, c, sat);                                                                             \
      return true;                                                                                                \
   }

/* A move keeps the comparison-result property of its source. */
static bool
f_mov(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat)
{
   SRC(0, lane, a)
   emit(x, TTR_OP_MOV, dst, a, none(), none(), sat);
   if (!a.neg && !a.abs && !sat &&
       ((a.kind == K_REG && bool_now(x, a.id)) || (a.kind == K_IMM && (x->imm[a.id] == 0u || x->imm[a.id] == 0x3f800000u))))
      x->is_bool[dst] = 1;
   return true;
}
LANE_OP1(f_frc, TTR_OP_FRC)
LANE_OP1(f_flr, TTR_OP_FLR)
LANE_OP2(f_add, TTR_OP_ADD)
LANE_OP2(f_mul, TTR_OP_MUL)
LANE_OP2(f_min, TTR_OP_MIN)
LANE_OP2(f_max, TTR_OP_MAX)
LANE_CMP(f_slt, TTR_OP_SLT, false)
LANE_CMP(f_sge, TTR_OP_SGE, false)
LANE_CMP(f_seq, TTR_OP_SEQ, false)
LANE_CMP(f_sne, TTR_OP_SNE, false)
LANE_CMP(f_sgt, TTR_OP_SLT, true) /* a > b is b < a */
LANE_CMP(f_sle, TTR_OP_SGE, true) /* a <= b is b >= a */
LANE_OP3(f_mad, TTR_OP_MAD)
LANE_OP3(f_cmp, TTR_OP_CMP)
LANE_OP3(f_ucmp, TTR_OP_SEL)

static bool
f_ceil(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat)
{
   SRC(0, lane, a)
   struct opnd na = a;
   na.neg = !na.neg;
   const int t = new_scratch(x);
   emit(x, TTR_OP_FLR, t, na, none(), none(), false); /* floor(-a) */
   struct opnd r = reg(t);
   r.neg = 1;
   emit(x, TTR_OP_MOV, dst, r, none(), none(), sat); /* ceil(a) = -floor(-a) */
   return true;
}

static bool
f_trunc(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat)
{
   SRC(0, lane, a)
   struct opnd aa = a;
   aa.abs = 1;
   aa.neg = 0;
   const int t = new_scratch(x);
   emit(x, TTR_OP_FLR, t, aa, none(), none(), false); /* floor(|a|) */
   struct opnd pos = reg(t), negt = reg(t);
   negt.neg = 1;
   emit(x, TTR_OP_CMP, dst, a, negt, pos, sat); /* a < 0 ? -floor(|a|) : floor(|a|) */
   return true;
}

static bool
f_ssg(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat)
{
   SRC(0, lane, a)
   const int p = new_scratch(x), n = new_scratch(x);
   emit(x, TTR_OP_SLT, p, imm_f(x, 0.0f), a, none(), false); /* a > 0 */
   emit(x, TTR_OP_SLT, n, a, imm_f(x, 0.0f), none(), false); /* a < 0 */
   struct opnd rn = reg(n);
   rn.neg = 1;
   emit(x, TTR_OP_ADD, dst, reg(p), rn, none(), sat);
   return true;
}

static bool
f_lrp(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat)
{
   SRC(0, lane, a)
   SRC(1, lane, b)
   SRC(2, lane, c)
   struct opnd nc = c;
   nc.neg = !nc.neg;
   const int t = new_scratch(x);
   emit(x, TTR_OP_ADD, t, b, nc, none(), false);   /* b - c */
   emit(x, TTR_OP_MAD, dst, a, reg(t), c, sat);    /* a * (b - c) + c */
   return true;
}

static bool
f_div(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat)
{
   SRC(0, lane, a)
   SRC(1, lane, b)
   const int t = new_scratch(x);
   emit(x, TTR_OP_RCP, t, b, none(), none(), false);
   emit(x, TTR_OP_MUL, dst, a, reg(t), none(), sat);
   return true;
}

/* A comparison result as an operand of a bitwise instruction: a register that holds one, or the constants of "true" (all
 * ones) and "false" (zero), which become 1.0 and 0.0. Anything else is an integer the instruction set cannot represent. */
static bool
bool_operand(struct xl *x, struct opnd *o)
{
   if (o->neg || o->abs)
      return false;
   if (o->kind == K_REG)
      return bool_now(x, o->id);
   /* The constants of "false" (0) and "true" (all ones, read as 1.0, see get_src). */
   return o->kind == K_IMM && (x->imm[o->id] == 0u || x->imm[o->id] == 0x3f800000u);
}

/* AND: either the boolean to float conversion (a comparison result AND the bits of 1.0f, which in this instruction set is a move,
 * since comparisons produce 1.0 or 0.0), or the conjunction of two comparison results (a product). */
static bool
f_and(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat)
{
   SRC(0, lane, a)
   SRC(1, lane, b)
   const bool a_one = a.kind == K_IMM && x->imm[a.id] == 0x3f800000u;
   const bool b_one = b.kind == K_IMM && x->imm[b.id] == 0x3f800000u;
   if (a_one || b_one) {
      emit(x, TTR_OP_MOV, dst, b_one ? a : b, none(), none(), sat);
      return true;
   }
   if (!sat && bool_operand(x, &a) && bool_operand(x, &b)) {
      emit(x, TTR_OP_MUL, dst, a, b, none(), false);
      x->is_bool[dst] = 1;
      return true;
   }
   return fail(x, "integer or bitwise operation");
}

static bool
f_or(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat)
{
   SRC(0, lane, a)
   SRC(1, lane, b)
   if (sat || !bool_operand(x, &a) || !bool_operand(x, &b))
      return fail(x, "integer or bitwise operation");
   emit(x, TTR_OP_MAX, dst, a, b, none(), false);
   x->is_bool[dst] = 1;
   return true;
}

static bool
f_xor(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat)
{
   SRC(0, lane, a)
   SRC(1, lane, b)
   if (sat || !bool_operand(x, &a) || !bool_operand(x, &b))
      return fail(x, "integer or bitwise operation");
   emit(x, TTR_OP_SNE, dst, a, b, none(), false);
   x->is_bool[dst] = 1;
   return true;
}

static bool
f_not(struct xl *x, const struct tgsi_full_instruction *inst, unsigned lane, int dst, bool sat)
{
   SRC(0, lane, a)
   if (sat || !bool_operand(x, &a))
      return fail(x, "integer or bitwise operation");
   emit(x, TTR_OP_SEQ, dst, a, imm_f(x, 0.0f), none(), false);   /* NOT: true where the comparison result is 0 */
   x->is_bool[dst] = 1;
   return true;
}

static bool
has_hazard(struct xl *x, const struct tgsi_full_instruction *inst, unsigned mask, const int ids[4])
{
   int written[4], nw = 0;
   for (unsigned lane = 0; lane < 4; lane++) {
      if (!(mask & (1u << lane)))
         continue;
      for (unsigned si = 0; si < inst->Instruction.NumSrcRegs; si++) {
         struct opnd o;
         if (!get_src(x, &inst->Src[si], lane, &o))
            return true;
         if (o.kind != K_REG)
            continue;
         for (int k = 0; k < nw; k++)
            if (written[k] == o.id)
               return true;
      }
      written[nw++] = ids[lane];
   }
   return false;
}

static void
store_lane(struct xl *x, int dst, int value)
{
   const int m = cur_mask(x);
   const bool value_bool = x->is_bool[value];
   if (m >= 0) {
      const bool old_bool = !x->defined[dst] || bool_now(x, dst);   /* an unwritten register reads as 0 */
      int scope = x->defined[dst] && x->is_bool[dst] ? x->bool_scope[dst] : -1;
      if (!old_bool) {
         scope = m;
         /* Both sides of one IF/ELSE wrote a comparison result: it is one wherever the whole construct is active. */
         const struct ifctx *c = &x->stack[x->depth - 1];
         if (x->is_bool[dst] && c->if_mask >= 0 && x->bool_scope[dst] == c->if_mask)
            scope = c->parent;
      }
      ensure_defined(x, dst);
      emit(x, TTR_OP_SEL, dst, reg(m), reg(value), reg(dst), false);
      if (value_bool) {
         x->is_bool[dst] = 1;
         x->bool_scope[dst] = scope;
      }
   } else {
      emit(x, TTR_OP_MOV, dst, reg(value), none(), none(), false);
      x->is_bool[dst] = value_bool;
   }
}

static bool
tr_lanes(struct xl *x, const struct tgsi_full_instruction *inst, lane_fn fn)
{
   const struct tgsi_full_dst_register *d = &inst->Dst[0];
   const unsigned mask = d->Register.WriteMask;
   int ids[4] = {-1, -1, -1, -1};
   if (!get_dst(x, d, ids))
      return false;
   const bool sat = inst->Instruction.Saturate;
   bool direct = cur_mask(x) < 0;
   if (direct && __builtin_popcount(mask) > 1)
      direct = !has_hazard(x, inst, mask, ids);
   if (x->why)
      return false;
   int tgt[4] = {-1, -1, -1, -1};
   for (unsigned lane = 0; lane < 4; lane++) {
      if (!(mask & (1u << lane)))
         continue;
      tgt[lane] = direct ? ids[lane] : new_scratch(x);
      if (!fn(x, inst, lane, tgt[lane], sat))
         return false;
   }
   if (!direct) {
      for (unsigned lane = 0; lane < 4; lane++)
         if (mask & (1u << lane))
            store_lane(x, ids[lane], tgt[lane]);
   }
   return !x->why;
}

/* ---- Scalar instructions: one result written to every enabled component ------------------------------------------- */

typedef bool (*scalar_fn)(struct xl *x, const struct tgsi_full_instruction *inst, int target, bool sat);

#define SCALAR_OP1(NAME, OP)                                                                       \
   static bool NAME(struct xl *x, const struct tgsi_full_instruction *inst, int target, bool sat) \
   {                                                                                               \
      SRC(0, 0, a)                                                                                 \
      emit(x, OP, target, a, none(), none(), sat);                                                 \
      return true;                                                                                 \
   }
SCALAR_OP1(s_rcp, TTR_OP_RCP)
SCALAR_OP1(s_rsq, TTR_OP_RSQ)
SCALAR_OP1(s_sqrt, TTR_OP_SQRT)
SCALAR_OP1(s_ex2, TTR_OP_EX2)
SCALAR_OP1(s_lg2, TTR_OP_LG2)
SCALAR_OP1(s_sin, TTR_OP_SIN)
SCALAR_OP1(s_cos, TTR_OP_COS)

static bool
s_pow(struct xl *x, const struct tgsi_full_instruction *inst, int target, bool sat)
{
   SRC(0, 0, a)
   SRC(1, 0, b)
   const int l = new_scratch(x), m = new_scratch(x);
   emit(x, TTR_OP_LG2, l, a, none(), none(), false);
   emit(x, TTR_OP_MUL, m, reg(l), b, none(), false);
   emit(x, TTR_OP_EX2, target, reg(m), none(), none(), sat);
   return true;
}

static bool
dot_n(struct xl *x, const struct tgsi_full_instruction *inst, int target, bool sat, unsigned n)
{
   const int acc = new_scratch(x);
   for (unsigned i = 0; i < n; i++) {
      SRC(0, i, a)
      SRC(1, i, b)
      if (i == 0)
         emit(x, n == 1 ? TTR_OP_MUL : TTR_OP_MUL, n == 1 ? target : acc, a, b, none(), n == 1 ? sat : false);
      else if (i + 1 == n)
         emit(x, TTR_OP_MAD, target, a, b, reg(acc), sat);
      else
         emit(x, TTR_OP_MAD, acc, a, b, reg(acc), false);
   }
   return true;
}
static bool s_dp2(struct xl *x, const struct tgsi_full_instruction *inst, int target, bool sat) { return dot_n(x, inst, target, sat, 2); }
static bool s_dp3(struct xl *x, const struct tgsi_full_instruction *inst, int target, bool sat) { return dot_n(x, inst, target, sat, 3); }
static bool s_dp4(struct xl *x, const struct tgsi_full_instruction *inst, int target, bool sat) { return dot_n(x, inst, target, sat, 4); }

static bool
tr_scalar(struct xl *x, const struct tgsi_full_instruction *inst, scalar_fn fn)
{
   const struct tgsi_full_dst_register *d = &inst->Dst[0];
   const unsigned mask = d->Register.WriteMask;
   int ids[4] = {-1, -1, -1, -1};
   if (!get_dst(x, d, ids))
      return false;
   const bool sat = inst->Instruction.Saturate;
   const int m = cur_mask(x);
   int first = -1, count = 0;
   for (int lane = 0; lane < 4; lane++) {
      if (mask & (1u << lane)) {
         if (first < 0)
            first = lane;
         count++;
      }
   }
   if (first < 0)
      return true;
   if (m < 0 && count == 1)
      return fn(x, inst, ids[first], sat) && !x->why;
   const int v = new_scratch(x);
   if (!fn(x, inst, v, sat))
      return false;
   for (int lane = 0; lane < 4; lane++)
      if (mask & (1u << lane))
         store_lane(x, ids[lane], v);
   return !x->why;
}

/* ---- Control flow and kills --------------------------------------------------------------------------------------- */

static bool
tr_if(struct xl *x, const struct tgsi_full_instruction *inst)
{
   if (x->depth >= 16)
      return fail(x, "conditionals nested too deeply");
   SRC(0, 0, c)
   const int m = new_scratch(x);
   emit(x, TTR_OP_SNE, m, c, imm_f(x, 0.0f), none(), false);
   const int parent = cur_mask(x);
   int mk = m;
   if (parent >= 0) {
      mk = new_scratch(x);
      emit(x, TTR_OP_MUL, mk, reg(parent), reg(m), none(), false);
   }
   x->stack[x->depth].mask = mk;
   x->stack[x->depth].cond = m;
   x->stack[x->depth].parent = parent;
   x->stack[x->depth].if_mask = -1;
   x->depth++;
   return true;
}

static bool
tr_else(struct xl *x)
{
   if (!x->depth)
      return fail(x, "ELSE without IF");
   struct ifctx *c = &x->stack[x->depth - 1];
   const int inv = new_scratch(x);
   emit(x, TTR_OP_SEQ, inv, reg(c->cond), imm_f(x, 0.0f), none(), false);
   int mk = inv;
   if (c->parent >= 0) {
      mk = new_scratch(x);
      emit(x, TTR_OP_MUL, mk, reg(c->parent), reg(inv), none(), false);
   }
   c->if_mask = c->mask;
   c->mask = mk;
   return true;
}

static void
kill_value(struct xl *x, struct opnd v)
{
   const int m = cur_mask(x);
   x->uses_kill = true;
   if (m >= 0) {
      const int t = new_scratch(x);
      emit(x, TTR_OP_SEL, t, reg(m), v, imm_f(x, 1.0f), false);   /* lanes outside the branch never kill */
      emit(x, TTR_OP_KILL, -1, reg(t), none(), none(), false);
   } else {
      emit(x, TTR_OP_KILL, -1, v, none(), none(), false);
   }
}

static bool
tr_kill_if(struct xl *x, const struct tgsi_full_instruction *inst)
{
   struct opnd prev[4];
   for (unsigned lane = 0; lane < 4; lane++) {
      SRC(0, lane, a)
      bool dup = false;
      for (unsigned k = 0; k < lane; k++)
         dup |= same_opnd(&prev[k], &a);
      prev[lane] = a;
      if (!dup)
         kill_value(x, a);
   }
   return !x->why;
}

/* ---- Instructions ------------------------------------------------------------------------------------------------- */

static bool
translate_inst(struct xl *x, const struct tgsi_full_instruction *inst)
{
   switch (inst->Instruction.Opcode) {
   case TGSI_OPCODE_MOV: return tr_lanes(x, inst, f_mov);
   case TGSI_OPCODE_ADD: return tr_lanes(x, inst, f_add);
   case TGSI_OPCODE_MUL: return tr_lanes(x, inst, f_mul);
   case TGSI_OPCODE_MAD:
   case TGSI_OPCODE_FMA: return tr_lanes(x, inst, f_mad);
   case TGSI_OPCODE_MIN: return tr_lanes(x, inst, f_min);
   case TGSI_OPCODE_MAX: return tr_lanes(x, inst, f_max);
   case TGSI_OPCODE_SLT:
   case TGSI_OPCODE_FSLT: return tr_lanes(x, inst, f_slt);
   case TGSI_OPCODE_SGE:
   case TGSI_OPCODE_FSGE: return tr_lanes(x, inst, f_sge);
   case TGSI_OPCODE_SEQ:
   case TGSI_OPCODE_FSEQ: return tr_lanes(x, inst, f_seq);
   case TGSI_OPCODE_SNE:
   case TGSI_OPCODE_FSNE: return tr_lanes(x, inst, f_sne);
   case TGSI_OPCODE_SGT: return tr_lanes(x, inst, f_sgt);
   case TGSI_OPCODE_SLE: return tr_lanes(x, inst, f_sle);
   case TGSI_OPCODE_CMP: return tr_lanes(x, inst, f_cmp);
   case TGSI_OPCODE_UCMP: return tr_lanes(x, inst, f_ucmp);
   case TGSI_OPCODE_FRC: return tr_lanes(x, inst, f_frc);
   case TGSI_OPCODE_FLR: return tr_lanes(x, inst, f_flr);
   case TGSI_OPCODE_CEIL: return tr_lanes(x, inst, f_ceil);
   case TGSI_OPCODE_TRUNC: return tr_lanes(x, inst, f_trunc);
   case TGSI_OPCODE_SSG: return tr_lanes(x, inst, f_ssg);
   case TGSI_OPCODE_LRP: return tr_lanes(x, inst, f_lrp);
   case TGSI_OPCODE_DIV: return tr_lanes(x, inst, f_div);
   case TGSI_OPCODE_AND: return tr_lanes(x, inst, f_and);
   case TGSI_OPCODE_OR: return tr_lanes(x, inst, f_or);
   case TGSI_OPCODE_XOR: return tr_lanes(x, inst, f_xor);
   case TGSI_OPCODE_NOT: return tr_lanes(x, inst, f_not);
   case TGSI_OPCODE_RCP: return tr_scalar(x, inst, s_rcp);
   case TGSI_OPCODE_RSQ: return tr_scalar(x, inst, s_rsq);
   case TGSI_OPCODE_SQRT: return tr_scalar(x, inst, s_sqrt);
   case TGSI_OPCODE_EX2: return tr_scalar(x, inst, s_ex2);
   case TGSI_OPCODE_LG2: return tr_scalar(x, inst, s_lg2);
   case TGSI_OPCODE_SIN: return tr_scalar(x, inst, s_sin);
   case TGSI_OPCODE_COS: return tr_scalar(x, inst, s_cos);
   case TGSI_OPCODE_POW: return tr_scalar(x, inst, s_pow);
   case TGSI_OPCODE_DP2: return tr_scalar(x, inst, s_dp2);
   case TGSI_OPCODE_DP3: return tr_scalar(x, inst, s_dp3);
   case TGSI_OPCODE_DP4: return tr_scalar(x, inst, s_dp4);
   case TGSI_OPCODE_IF:
   case TGSI_OPCODE_UIF: return tr_if(x, inst);
   case TGSI_OPCODE_ELSE: return tr_else(x);
   case TGSI_OPCODE_ENDIF:
      if (!x->depth)
         return fail(x, "ENDIF without IF");
      x->depth--;
      return true;
   case TGSI_OPCODE_KILL_IF: return tr_kill_if(x, inst);
   case TGSI_OPCODE_KILL:
      kill_value(x, imm_f(x, -1.0f));
      return !x->why;
   case TGSI_OPCODE_NOP:
   case TGSI_OPCODE_END: return true;
   case TGSI_OPCODE_BGNLOOP:
   case TGSI_OPCODE_ENDLOOP:
   case TGSI_OPCODE_BRK:
   case TGSI_OPCODE_CONT: return fail(x, "loop that was not unrolled");
   case TGSI_OPCODE_CAL:
   case TGSI_OPCODE_RET:
   case TGSI_OPCODE_BGNSUB:
   case TGSI_OPCODE_ENDSUB: return fail(x, "subroutine");
   case TGSI_OPCODE_TEX:
   case TGSI_OPCODE_TXB:
   case TGSI_OPCODE_TXP:
   case TGSI_OPCODE_TXD:
   case TGSI_OPCODE_TXL:
   case TGSI_OPCODE_TXF:
   case TGSI_OPCODE_TXQ:
   case TGSI_OPCODE_SAMPLE:
   case TGSI_OPCODE_SAMPLE_B:
   case TGSI_OPCODE_SAMPLE_C:
   case TGSI_OPCODE_SAMPLE_L:
   case TGSI_OPCODE_SAMPLE_D: return fail(x, "texture sampling");
   case TGSI_OPCODE_DDX:
   case TGSI_OPCODE_DDY:
   case TGSI_OPCODE_DDX_FINE:
   case TGSI_OPCODE_DDY_FINE: return fail(x, "derivatives");
   default: {
      static __thread char msg[64];
      snprintf(msg, sizeof msg, "TGSI opcode %s without a translation", tgsi_get_opcode_name(inst->Instruction.Opcode));
      return fail(x, msg);
   }
   }
}

/* ---- Register allocation and emission ------------------------------------------------------------------------------ */

static bool
is_out(int id)
{
   return id >= V_OUT && id < V_OUT + 4;
}

static struct ttr_prog *
finish(struct xl *x, const char **why)
{
   /* Colour components the shader never wrote are 0. */
   for (int c = 0; c < 4; c++) {
      if (!x->out_written[c] && !x->defined[V_OUT + c])
         emit(x, TTR_OP_MOV, V_OUT + c, imm_f(x, 0.0f), none(), none(), false);
   }
   if (x->why) {
      *why = x->why;
      return NULL;
   }
   if (x->n == 0 || x->n > TTR_MAX_INSTRUCTIONS) {
      *why = "program length";
      return NULL;
   }
   if (x->n_imm + x->n_uni > TTR_MAX_CONSTS) {
      *why = "too many constants";
      return NULL;
   }

   /* Live ranges. */
   int *last_use = MALLOC(MAX_VREGS * sizeof(int)), *first_def = MALLOC(MAX_VREGS * sizeof(int));
   int *phys = MALLOC(MAX_VREGS * sizeof(int));
   if (!last_use || !first_def || !phys) {
      FREE(last_use);
      FREE(first_def);
      FREE(phys);
      *why = "out of memory";
      return NULL;
   }
   for (int i = 0; i < MAX_VREGS; i++) {
      last_use[i] = -1;
      first_def[i] = INT_MAX;
      phys[i] = -1;
   }
   for (unsigned k = 0; k < x->n_in; k++)
      first_def[V_IN + k] = -1;
   if (x->fc_used)
      for (int c = 0; c < 4; c++)
         first_def[V_FC + c] = -1;
   for (unsigned n = 0; n < x->n; n++) {
      const struct vi *i = &x->v[n];
      for (int s = 0; s < 3; s++) {
         if (i->s[s].kind == K_REG) {
            const int id = i->s[s].id;
            if (last_use[id] < (int)n)
               last_use[id] = n;
         }
      }
      if (i->dst >= 0) {
         if (first_def[i->dst] == INT_MAX)
            first_def[i->dst] = n;
         if (last_use[i->dst] < (int)n)
            last_use[i->dst] = n;
      }
   }

   /* Linear scan over the non-output vregs. holder[p] = the vreg in physical register p and until when. */
   int holder_end[MAX_PHYS];
   bool busy[MAX_PHYS];
   memset(busy, 0, sizeof busy);
   int top = 0;
   if (x->n_in + (x->fc_used ? 4 : 0) > MAX_PHYS) {
      *why = "too many varying components";
      goto bad;
   }
   for (unsigned k = 0; k < x->n_in; k++) {
      phys[V_IN + k] = k;
      busy[k] = true;
      holder_end[k] = last_use[V_IN + k];
   }
   if (x->fc_used) {
      for (int c = 0; c < 4; c++) {
         const int p = x->n_in + c;
         phys[V_FC + c] = p;
         busy[p] = true;
         holder_end[p] = last_use[V_FC + c];
      }
   }
   top = x->n_in + (x->fc_used ? 4 : 0);
   for (unsigned n = 0; n < x->n; n++) {
      const struct vi *i = &x->v[n];
      /* registers whose value ends at or before this instruction can take its result (operands are read first) */
      for (int p = 0; p < MAX_PHYS; p++)
         if (busy[p] && holder_end[p] <= (int)n)
            busy[p] = false;
      if (i->dst >= 0 && !is_out(i->dst) && phys[i->dst] < 0) {
         int p = 0;
         while (p < MAX_PHYS && busy[p])
            p++;
         if (p == MAX_PHYS) {
            *why = "too many live values for 57 registers";
            goto bad;
         }
         phys[i->dst] = p;
         busy[p] = true;
         holder_end[p] = last_use[i->dst];
         if (p + 1 > top)
            top = p + 1;
      }
   }
   const int out_reg = top;
   if (out_reg + 4 > TTR_MAX_REGS) {
      *why = "too many live values";
      goto bad;
   }
   for (int c = 0; c < 4; c++)
      phys[V_OUT + c] = out_reg + c;

   /* Emit. */
   struct ttr_prog *p = CALLOC_STRUCT(ttr_prog);
   if (!p) {
      *why = "out of memory";
      goto bad;
   }
   p->code = MALLOC(x->n * 4 * sizeof(uint32_t));
   p->n_instr = x->n;
   p->n_imm = x->n_imm;
   p->n_consts = x->n_imm + x->n_uni;
   p->consts = MALLOC(MAX2(p->n_consts, 1u) * sizeof(float));
   p->uniform_index = MALLOC(MAX2(x->n_uni, 1u) * sizeof(uint32_t));
   if (!p->code || !p->consts || !p->uniform_index) {
      ttr_prog_free(p);
      *why = "out of memory";
      goto bad;
   }
   for (unsigned k = 0; k < x->n_imm; k++)
      memcpy(&p->consts[k], &x->imm[k], 4);
   for (unsigned k = 0; k < x->n_uni; k++) {
      p->consts[x->n_imm + k] = 0.0f;
      p->uniform_index[k] = x->uni[k];
   }
   p->num_inputs = x->n_in;
   memcpy(p->map, x->map, sizeof(struct ttr_prog_input) * x->n_in);
   p->num_regs = out_reg + 4;
   p->out_reg = out_reg;
   p->fragcoord_reg = x->fc_used ? (int32_t)x->n_in : -1;
   p->position_input = x->fc_used ? x->position_input : -1;
   p->height_offset = x->pixel_center_integer ? 0.5f : 0.0f;
   p->uses_kill = x->uses_kill;
   for (unsigned n = 0; n < x->n; n++) {
      const struct vi *i = &x->v[n];
      uint32_t *w = &p->code[4 * n];
      w[0] = i->op | ((i->sat ? TTR_FLAG_SAT : 0u) << 8) | ((i->dst >= 0 ? (uint32_t)phys[i->dst] : 0u) << 16);
      for (int s = 0; s < 3; s++) {
         const struct opnd *o = &i->s[s];
         uint32_t v = 0;
         if (o->kind == K_REG)
            v = (uint32_t)phys[o->id];
         else if (o->kind == K_IMM)
            v = TTR_SRC_CONST | (uint32_t)o->id;
         else if (o->kind == K_UNI)
            v = TTR_SRC_CONST | (uint32_t)(x->n_imm + o->id);
         if (o->kind != K_NONE) {
            if (o->abs)
               v |= TTR_SRC_ABS;
            if (o->neg)
               v |= TTR_SRC_NEG;
         }
         w[1 + s] = v;
      }
   }
   FREE(last_use);
   FREE(first_def);
   FREE(phys);
   return p;
bad:
   FREE(last_use);
   FREE(first_def);
   FREE(phys);
   return NULL;
}

struct ttr_prog *
ttr_prog_translate(const struct tgsi_token *tokens, const char **why)
{
   static const char *oom = "out of memory";
   struct xl *x = CALLOC_STRUCT(xl);
   if (!x) {
      *why = oom;
      return NULL;
   }
   x->v = MALLOC(MAX_VI * sizeof(struct vi));
   x->defined = CALLOC(MAX_VREGS, 1);
   x->is_bool = CALLOC(MAX_VREGS, 1);
   x->bool_scope = CALLOC(MAX_VREGS, sizeof(int32_t));
   if (!x->v || !x->defined || !x->is_bool || !x->bool_scope) {
      FREE(x->v);
      FREE(x->defined);
      FREE(x->is_bool);
      FREE(x->bool_scope);
      FREE(x);
      *why = oom;
      return NULL;
   }
   x->next_scratch = V_SCRATCH;
   x->position_input = -1;
   x->fcx = x->fcy = -1;
   for (int j = 0; j < MAX_IN_DECL; j++) {
      x->semantic[j] = -1;
      for (int c = 0; c < 4; c++)
         x->in_slot[j][c] = -1;
   }

   struct ttr_prog *result = NULL;
   struct tgsi_parse_context parse;
   if (tgsi_parse_init(&parse, tokens) != TGSI_PARSE_OK) {
      *why = "cannot parse TGSI";
      goto out;
   }
   while (!tgsi_parse_end_of_tokens(&parse) && !x->why) {
      tgsi_parse_token(&parse);
      switch (parse.FullToken.Token.Type) {
      case TGSI_TOKEN_TYPE_DECLARATION: {
         const struct tgsi_full_declaration *d = &parse.FullToken.FullDeclaration;
         switch (d->Declaration.File) {
         case TGSI_FILE_INPUT:
            for (unsigned j = d->Range.First; j <= d->Range.Last && j < MAX_IN_DECL; j++)
               x->semantic[j] = d->Declaration.Semantic ? (int)d->Semantic.Name : TGSI_SEMANTIC_GENERIC;
            if (d->Range.Last >= MAX_IN_DECL)
               fail(x, "too many fragment inputs");
            break;
         case TGSI_FILE_OUTPUT:
            if (d->Range.Last > 0 || (d->Declaration.Semantic && d->Semantic.Name != TGSI_SEMANTIC_COLOR))
               fail(x, "outputs other than colour 0");
            break;
         case TGSI_FILE_TEMPORARY:
            if (d->Range.Last >= MAX_TEMPS)
               fail(x, "too many temporaries");
            break;
         case TGSI_FILE_CONSTANT:
         case TGSI_FILE_IMMEDIATE:
            break;
         default:
            fail(x, "unsupported register file (samplers, address registers, system values, buffers)");
         }
         break;
      }
      case TGSI_TOKEN_TYPE_IMMEDIATE: {
         const struct tgsi_full_immediate *im = &parse.FullToken.FullImmediate;
         if (x->n_timm >= MAX_TIMM) {
            fail(x, "too many immediates");
            break;
         }
         if (im->Immediate.DataType != TGSI_IMM_FLOAT32 && im->Immediate.DataType != TGSI_IMM_UINT32 &&
             im->Immediate.DataType != TGSI_IMM_INT32) {
            fail(x, "64-bit immediate");
            break;
         }
         for (int c = 0; c < 4; c++)
            x->timm[x->n_timm][c] = im->u[c].Uint;
         x->n_timm++;
         break;
      }
      case TGSI_TOKEN_TYPE_PROPERTY: {
         const struct tgsi_full_property *pr = &parse.FullToken.FullProperty;
         if (pr->Property.PropertyName == TGSI_PROPERTY_FS_COORD_ORIGIN)
            x->origin_lower_left = pr->u[0].Data == TGSI_FS_COORD_ORIGIN_LOWER_LEFT;
         else if (pr->Property.PropertyName == TGSI_PROPERTY_FS_COORD_PIXEL_CENTER)
            x->pixel_center_integer = pr->u[0].Data == TGSI_FS_COORD_PIXEL_CENTER_INTEGER;
         break;
      }
      case TGSI_TOKEN_TYPE_INSTRUCTION:
         translate_inst(x, &parse.FullToken.FullInstruction);
         break;
      default:
         break;
      }
   }
   tgsi_parse_free(&parse);
   if (!x->why && x->depth)
      fail(x, "IF without ENDIF");
   result = finish(x, why);
out:
   FREE(x->v);
   FREE(x->defined);
   FREE(x->is_bool);
   FREE(x->bool_scope);
   FREE(x);
   return result;
}

void
ttr_prog_free(struct ttr_prog *p)
{
   if (!p)
      return;
   FREE(p->code);
   FREE(p->consts);
   FREE(p->uniform_index);
   FREE(p);
}

bool
ttr_prog_constants(const struct ttr_prog *p, const float *cbuf, unsigned cbuf_bytes, unsigned fb_height, float *out)
{
   memcpy(out, p->consts, p->n_imm * sizeof(float));
   for (unsigned k = 0; k < p->n_consts - p->n_imm; k++) {
      const uint32_t idx = p->uniform_index[k];
      if (idx == HEIGHT_MARK) {
         out[p->n_imm + k] = (float)fb_height - p->height_offset;
         continue;
      }
      if ((size_t)(idx + 1) * sizeof(float) > cbuf_bytes || !cbuf)
         return false;
      out[p->n_imm + k] = cbuf[idx];
   }
   return true;
}
