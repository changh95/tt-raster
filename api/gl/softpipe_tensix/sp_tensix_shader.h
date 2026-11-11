/*
 * Copyright 2026 tt-raster contributors
 * SPDX-License-Identifier: MIT
 *
 * Translation of TGSI fragment shaders to the straight-line shader instruction set of libttraster
 * (core/include/ttraster/ttraster_isa.h), for the Tensix fast path of softpipe. See api/gl/README.md.
 *
 * TGSI vec4 instructions are expanded per component. Conditionals (IF/ELSE/ENDIF) are if-converted: both sides run and
 * assignments are selected by a lane mask. Not handled (the translator refuses and softpipe draws): texture and memory
 * instructions, derivatives, loops and subroutines, indirect addressing, integer and bit operations other than the
 * boolean-to-float conversion and the logic of comparison results (&&, ||, !), outputs other than colour 0, system values and
 * the front-face flag.
 */

#ifndef SP_TENSIX_SHADER_H
#define SP_TENSIX_SHADER_H

#include <stdbool.h>
#include <stdint.h>

struct tgsi_token;

/* One fragment shader input component used by the program: component `comp` of fragment shader input `input`. */
struct ttr_prog_input {
   uint8_t input;
   uint8_t comp;
};

struct ttr_prog {
   uint32_t *code;          /* n_instr instructions of 4 words */
   uint32_t n_instr;
   float *consts;           /* immediates first (n_imm), then one slot per uniform component the program reads */
   uint32_t n_imm, n_consts;
   uint32_t *uniform_index; /* for const slot n_imm + k: (constant buffer << 24) | float index in it (HEIGHT_MARK: the framebuffer height) */
   uint32_t num_inputs;     /* varying components: program input register k is (map[k].input, map[k].comp) */
   struct ttr_prog_input map[32];
   uint32_t num_regs, out_reg;
   int32_t fragcoord_reg;   /* >= 0: the program reads gl_FragCoord (x, y, z, 1/w) in 4 registers from here */
   int position_input;      /* the fragment shader input with the POSITION semantic, or -1 */
   float height_offset;     /* gl_FragCoord with a lower-left origin uses the framebuffer height: slot value = height - offset */
   bool uses_kill;
};

/* Translate a fragment shader. Returns NULL and sets *why to a static string if the shader is out of reach. */
struct ttr_prog *ttr_prog_translate(const struct tgsi_token *tokens, const char **why);
/* The same; origin_in_shader: the shader flips gl_FragCoord.y itself (llvmpipe's NIR does, with a constant of the driver state), so the
 * FS_COORD_ORIGIN property must not flip it a second time. */
struct ttr_prog *ttr_prog_translate_ex(const struct tgsi_token *tokens, const char **why, bool origin_in_shader);
void ttr_prog_free(struct ttr_prog *p);

/* Constant buffers a program may read (llvmpipe's PIPE_MAX_CONSTANT_BUFFERS). */
#define TTR_MAX_CBUFS 16
struct ttr_prog_cbuf {
   const float *data;
   unsigned bytes;
};

/* The constant table of the program for the current constants: immediates, then uniforms read from `cbuf` (constant
 * buffer 0, `cbuf_bytes` long), and the framebuffer height where the program needs it. `out` has p->n_consts floats. Returns
 * false if a uniform lies outside the buffer. */
bool ttr_prog_constants(const struct ttr_prog *p, const float *cbuf, unsigned cbuf_bytes, unsigned fb_height, float *out);
/* The same with all constant buffers (`bufs` has TTR_MAX_CBUFS entries; a program's uniform slot names its buffer and float index). */
bool ttr_prog_constants_multi(const struct ttr_prog *p, const struct ttr_prog_cbuf *bufs, unsigned fb_height, float *out);

#endif
