/*
 * Copyright 2026 tt-raster contributors
 * SPDX-License-Identifier: MIT
 *
 * Tensix fast path for softpipe. See sp_tensix.h.
 *
 * Life of a frame: a full colour+depth clear opens a frame (the clear values are remembered, the actual
 * rasterizer frame starts at the first accepted draw, when the depth function is known). Every draw is checked
 * against what libttraster can do; accepted draws hand their post-transform triangles (window coordinates from the
 * draw module) to libttraster instead of softpipe's setup stage. The frame is rendered once, when something needs
 * the pixels (swap, flush, read-back, a different framebuffer) or when a draw the fast path cannot do arrives; the
 * result is written into the colour and depth tile caches, and softpipe carries on with that content. After such a
 * fallback the rest of the frame runs on softpipe, so output is never wrong, only slower.
 *
 * Environment: TTR_GL = "tensix" | "reference" selects the backend and enables the path (GALLIUM_DRIVER=tensix is an
 * alias that sets "tensix"); TTR_GL_CHIPS = number of chips; TTRASTER_LIB = path of libttraster.so;
 * TTR_GL_DEBUG = 1 logs decisions.
 */

#include "sp_tensix.h"
#include "sp_tensix_shader.h"

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sp_context.h"
#include "sp_setup.h"
#include "sp_state.h"
#include "sp_tile_cache.h"
#include "tgsi/tgsi_dump.h"
#include "tgsi/tgsi_parse.h"
#include "util/format/u_format.h"
#include "util/u_memory.h"
#include "util/u_prim.h"

/* ---- libttraster ABI (include/ttraster/ttraster.h), loaded with dlopen --------------------------------------- */

typedef struct {
   float x, y, z, rhw;
   float color[4];
} ttr_vertex;

static unsigned g_min_pixels;   /* TTR_GL_MIN_PIXELS, see framebuffer_ok */

enum { TTR_DEPTH_LESS = 0, TTR_DEPTH_LEQUAL = 1, TTR_DEPTH_ALWAYS = 2 };
enum { TTR_BLEND_NONE = 0, TTR_BLEND_ALPHA = 1, TTR_BLEND_ADD = 2, TTR_BLEND_PREMULTIPLIED = 3, TTR_BLEND_MULTIPLY = 4, TTR_BLEND_ALPHA_ADD = 5 };
typedef struct {
   int depth_func, depth_write, blend;
} ttr_state;
enum { TTR_SHADE_LINEAR = 0, TTR_SHADE_FLAT = 1, TTR_SHADE_PERSPECTIVE = 2 };
enum { TTR_OUT_SWAP_RB = 1, TTR_OUT_OPAQUE = 2 };

typedef struct {
   const uint32_t *code;
   uint32_t n_instr;
   const float *consts;
   uint32_t n_consts;
   uint32_t num_inputs, num_regs, out_reg;
   int32_t fragcoord_reg;
} ttr_shader_desc;

struct ttr_api {
   void *(*context_create)(const char *backend, uint32_t n_chips);
   void (*context_destroy)(void *ctx);
   int (*frame_begin)(void *ctx, uint32_t w, uint32_t h, const float clear_color[4], float clear_depth, int depth_func);
   int (*frame_triangles)(void *ctx, const ttr_vertex *verts, uint32_t n, int shading, int cull, const ttr_state *state);
   uint32_t (*frame_shader)(void *ctx, const ttr_shader_desc *desc);
   int (*frame_triangles_shaded)(void *ctx, const ttr_vertex *verts, uint32_t n, uint32_t shader, const float *attrs,
                                 int perspective, int cull, const ttr_state *state);
   int (*frame_end_ex)(void *ctx, uint8_t *rgba, uint32_t stride, uint32_t color_flags, float *depth, int flip_y);
   int (*frame_depth)(void *ctx, float *depth, int flip_y);
   const char *(*last_error)(void);
   const char *(*context_report)(void *ctx);
};

enum frame_state {
   FRAME_NONE,     /* nothing pending; draws go to softpipe (or wait for a clear) */
   FRAME_CLEARED,  /* a full clear was seen; the rasterizer frame has not begun */
   FRAME_OPEN,     /* the rasterizer frame is accumulating triangles */
};

struct sp_tensix {
   struct softpipe_context *sp;
   void *lib;
   struct ttr_api api;
   void *ctx;
   bool debug;
   bool debug_tgsi;                        /* TTR_GL_DEBUG=2: print the TGSI of every fragment shader the first time it is seen */
   const struct tgsi_token *dumped_tokens;

   enum frame_state state;
   float clear_color[4];
   float clear_depth;
   uint32_t width, height;  /* of the open frame */

   /* Translated fragment shaders, by TGSI token pointer (checked against a hash of the tokens: a deleted shader's address
    * can be reused). */
   struct {
      const struct tgsi_token *tokens;
      uint64_t hash;
      unsigned ntokens;
      struct ttr_prog *prog;   /* NULL: refused */
      const char *why;
   } shaders[32];
   unsigned n_shaders, next_shader;

   /* The draw in progress. */
   bool shaded;             /* the fragment shader is a translated program (otherwise a colour pass-through) */
   const struct ttr_prog *prog;
   uint32_t shader_id;      /* its id in the open frame */
   int input_slot[32];      /* per program input component: the vertex slot of the fragment input, and its interpolation mode */
   int input_interp[32];
   float *attrs;            /* varyings of the batch: num_inputs floats per vertex */
   size_t attrs_cap;
   bool tri_active;
   int color_slot;          /* vertex slot of the colour varying */
   bool flat;
   bool perspective;        /* the colour varying is perspective-interpolated */
   ttr_state draw_state;    /* depth and blend state of the current draw */
   bool own_state;          /* it differs from the frame default (LESS, depth writes on, no blending) */
   ttr_vertex *verts;
   size_t nverts, cap;

   /* The rendered depth, when something asked for it. */
   float *depth;
   size_t depth_px;

   /* The depth buffer of the last rendered frame is not in softpipe's depth surface yet. Normally nothing reads it before
    * the next full clear drops it, so a frame is ended without its depth plane (no read-back, no copy) and the plane is
    * rendered afterwards only when it is wanted (`depth_cached`: it is already in `depth`). */
   bool depth_pending;
   bool depth_cached;
   /* An application that reads the depth buffer every frame would render every frame twice. Once a read had to render the frame
    * again, frames end with their depth plane until it has gone unused (`depth_unused` frames in a row). */
   bool depth_wanted;
   unsigned depth_unused;
   unsigned depth_rerenders;
   unsigned frames_fast, draws_fast, draws_slow, frames_fallback;
   double ms_render, ms_write, ms_depth;   /* accumulated time ending frames, in writing depth surfaces (ms_write; ms_depth is the same) */
   unsigned depth_writes;
   bool logged_unsupported;
};

#define DBG(t, ...)                       \
   do {                                   \
      if ((t)->debug) {                   \
         fprintf(stderr, "tensix: " __VA_ARGS__); \
         fputc('\n', stderr);             \
      }                                   \
   } while (0)

static double
now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* ---- setup ------------------------------------------------------------------------------------------------- */

static bool
load_api(struct sp_tensix *t, const char *path)
{
   t->lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
   if (!t->lib) {
      fprintf(stderr, "tensix: cannot load %s: %s\n", path, dlerror());
      return false;
   }
#define LOAD(field, name)                                                    \
   t->api.field = (__typeof__(t->api.field))dlsym(t->lib, name);            \
   if (!t->api.field) {                                                      \
      fprintf(stderr, "tensix: %s not found in %s\n", name, path);          \
      return false;                                                          \
   }
   LOAD(context_create, "ttr_context_create")
   LOAD(context_destroy, "ttr_context_destroy")
   LOAD(frame_begin, "ttr_frame_begin")
   LOAD(frame_triangles, "ttr_frame_triangles")
   LOAD(frame_shader, "ttr_frame_shader")
   LOAD(frame_triangles_shaded, "ttr_frame_triangles_shaded")
   LOAD(frame_end_ex, "ttr_frame_end_ex")
   LOAD(frame_depth, "ttr_frame_depth")
   LOAD(last_error, "ttr_last_error")
   LOAD(context_report, "ttr_context_report")
#undef LOAD
   return true;
}

struct sp_tensix *
sp_tensix_create(struct softpipe_context *sp)
{
   const char *mode = getenv("TTR_GL");
   if (!mode || !*mode || !strcmp(mode, "0"))
      return NULL;
   const char *lib = getenv("TTRASTER_LIB");
   struct sp_tensix *t = CALLOC_STRUCT(sp_tensix);
   if (!t)
      return NULL;
   t->sp = sp;
   t->debug = getenv("TTR_GL_DEBUG") != NULL;
   g_min_pixels = getenv("TTR_GL_MIN_PIXELS") ? (unsigned)strtoul(getenv("TTR_GL_MIN_PIXELS"), NULL, 10) : 0;
   t->debug_tgsi = t->debug && atoi(getenv("TTR_GL_DEBUG")) >= 2;
   if (!load_api(t, lib && *lib ? lib : "libttraster.so")) {
      FREE(t);
      return NULL;
   }
   const char *chips = getenv("TTR_GL_CHIPS");
   t->ctx = t->api.context_create(!strcmp(mode, "reference") ? "reference" : "tensix", chips ? (uint32_t)atoi(chips) : 1);
   if (!t->ctx) {
      fprintf(stderr, "tensix: cannot create the rasterizer context: %s\n", t->api.last_error());
      dlclose(t->lib);
      FREE(t);
      return NULL;
   }
   fprintf(stderr, "tensix: softpipe fast path enabled (backend %s)\n", !strcmp(mode, "reference") ? "reference" : "tensix");
   return t;
}

void
sp_tensix_destroy(struct sp_tensix *t)
{
   if (!t)
      return;
   if (t->debug)
      fprintf(stderr, "tensix: %u frames on the fast path, %u draws accepted, %u draws refused, %u mid-frame fallbacks, %u depth planes rendered afterwards\n",
              t->frames_fast, t->draws_fast, t->draws_slow, t->frames_fallback, t->depth_rerenders);
   t->api.context_destroy(t->ctx);
   FREE(t->verts);
   FREE(t->attrs);
   for (unsigned i = 0; i < t->n_shaders; i++)
      ttr_prog_free(t->shaders[i].prog);
   FREE(t->depth);
   dlclose(t->lib);
   FREE(t);
}

/* ---- what the fast path can do ------------------------------------------------------------------------------- */

static bool
color_format_ok(enum pipe_format f)
{
   const struct util_format_description *d = util_format_description(f);
   return d && d->layout == UTIL_FORMAT_LAYOUT_PLAIN && d->colorspace == UTIL_FORMAT_COLORSPACE_RGB &&
          util_format_is_unorm8(d) && d->block.bits == 32;
}

static bool
depth_format_ok(enum pipe_format f)
{
   switch (f) {
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
   case PIPE_FORMAT_Z24X8_UNORM:
   case PIPE_FORMAT_S8_UINT_Z24_UNORM:
   case PIPE_FORMAT_X8Z24_UNORM:
   case PIPE_FORMAT_Z32_UNORM:
   case PIPE_FORMAT_Z32_FLOAT:
   case PIPE_FORMAT_Z16_UNORM:
      return true;
   default:
      return false;
   }
}

/* Windows with fewer pixels than this stay on softpipe: a frame costs the chip about 1.5 ms (launch, upload, read-back) however
 * small it is, and softpipe is faster below roughly 300,000 pixels (glxgears: 560 against 330 FPS at 300 x 300, 275 against 245 at
 * 500 x 500, 150 against 235 at 700 x 700). TTR_GL_MIN_PIXELS overrides it; the default of the library is 0 (always), scripts/run_gl.sh sets 300000. */
static bool
framebuffer_ok(const struct softpipe_context *sp)
{
   const struct pipe_framebuffer_state *fb = &sp->framebuffer;
   if ((unsigned long)fb->width * fb->height < g_min_pixels)
      return false;
   if (fb->nr_cbufs != 1 || !fb->cbufs[0].texture || !fb->zsbuf.texture)
      return false;
   if (fb->width == 0 || fb->height == 0 || fb->width > 2048 || fb->height > 2048)
      return false;
   if (fb->cbufs[0].first_layer != fb->cbufs[0].last_layer || fb->zsbuf.first_layer != fb->zsbuf.last_layer)
      return false;
   if (fb->cbufs[0].texture->nr_samples > 1 || fb->zsbuf.texture->nr_samples > 1)
      return false;
   return color_format_ok(fb->cbufs[0].format) && depth_format_ok(fb->zsbuf.format);
}

/* The fragment shader must be `MOV OUT[0], IN[k]` and nothing else. Returns the fragment input index k, or -1. */
static int
passthrough_input(const struct tgsi_token *tokens)
{
   struct tgsi_parse_context parse;
   int input = -1, count = 0;
   bool ok = true;
   if (tgsi_parse_init(&parse, tokens) != TGSI_PARSE_OK)
      return -1;
   while (!tgsi_parse_end_of_tokens(&parse)) {
      tgsi_parse_token(&parse);
      if (parse.FullToken.Token.Type != TGSI_TOKEN_TYPE_INSTRUCTION)
         continue;
      const struct tgsi_full_instruction *inst = &parse.FullToken.FullInstruction;
      if (inst->Instruction.Opcode == TGSI_OPCODE_END)
         continue;
      count++;
      const struct tgsi_full_src_register *s = &inst->Src[0];
      const struct tgsi_full_dst_register *d = &inst->Dst[0];
      if (inst->Instruction.Opcode != TGSI_OPCODE_MOV || inst->Instruction.Saturate ||
          inst->Instruction.NumDstRegs != 1 || inst->Instruction.NumSrcRegs != 1 ||
          d->Register.File != TGSI_FILE_OUTPUT || d->Register.Index != 0 || d->Register.WriteMask != TGSI_WRITEMASK_XYZW ||
          d->Register.Indirect || d->Register.Dimension || s->Register.File != TGSI_FILE_INPUT || s->Register.Indirect ||
          s->Register.Dimension || s->Register.Absolute || s->Register.Negate || s->Register.SwizzleX != TGSI_SWIZZLE_X ||
          s->Register.SwizzleY != TGSI_SWIZZLE_Y || s->Register.SwizzleZ != TGSI_SWIZZLE_Z ||
          s->Register.SwizzleW != TGSI_SWIZZLE_W) {
         ok = false;
      } else {
         input = s->Register.Index;
      }
   }
   tgsi_parse_free(&parse);
   return ok && count == 1 ? input : -1;
}

/* The translated program of a fragment shader variant (cached; a refusal is cached too, with its reason). */
static const struct ttr_prog *
get_program(struct sp_tensix *t, const struct sp_fragment_shader_variant *variant, const char **why)
{
   const struct tgsi_token *tok = variant->tokens;
   const unsigned nt = tgsi_num_tokens(tok);
   uint64_t h = 1469598103934665603ull;
   for (unsigned i = 0; i < nt * sizeof(struct tgsi_token); i++)
      h = (h ^ ((const uint8_t *)tok)[i]) * 1099511628211ull;
   for (unsigned i = 0; i < t->n_shaders; i++) {
      if (t->shaders[i].tokens == tok && t->shaders[i].hash == h && t->shaders[i].ntokens == nt) {
         *why = t->shaders[i].why;
         return t->shaders[i].prog;
      }
   }
   const char *reason = NULL;
   struct ttr_prog *prog = ttr_prog_translate(tok, &reason);
   if (t->debug) {
      if (prog) {
         fprintf(stderr, "tensix: fragment shader translated: %u instructions, %u inputs, %u constants, %u registers\n", prog->n_instr,
                 prog->num_inputs, prog->n_consts, prog->num_regs);
         if (t->debug_tgsi) {
            for (unsigned i = 0; i < prog->n_instr; i++) {
               const uint32_t *w = &prog->code[4 * i];
               fprintf(stderr, "    %3u: op %2u%s r%u", i, w[0] & 0xff, (w[0] >> 8) & 1 ? ".sat" : "", w[0] >> 16);
               for (int k = 1; k < 4; k++) {
                  if (w[k] == 0 && k > 1)
                     continue;
                  fprintf(stderr, "  %s%s%s%u", (w[k] & (1u << 13)) ? "-" : "", (w[k] & (1u << 14)) ? "|" : "", (w[k] & (1u << 12)) ? "c" : "r",
                          w[k] & 0xfff);
               }
               fputc('\n', stderr);
            }
            fprintf(stderr, "    consts:");
            for (unsigned i = 0; i < prog->n_imm; i++)
               fprintf(stderr, " %g", prog->consts[i]);
            fprintf(stderr, "  | out r%u fragcoord r%d\n", prog->out_reg, prog->fragcoord_reg);
         }
      }
      else
         fprintf(stderr, "tensix: fragment shader not translated: %s\n", reason);
   }
   const unsigned slot = t->n_shaders < 32 ? t->n_shaders++ : (t->next_shader++ % 32);
   ttr_prog_free(t->shaders[slot].prog);
   t->shaders[slot].tokens = tok;
   t->shaders[slot].hash = h;
   t->shaders[slot].ntokens = nt;
   t->shaders[slot].prog = prog;
   t->shaders[slot].why = reason;
   *why = reason;
   return prog;
}

/* Why the current draw cannot use the fast path, or NULL. Sets t->color_slot / t->flat when it can. */
static const char *
draw_refused(struct sp_tensix *t, const struct pipe_draw_info *info)
{
   struct softpipe_context *sp = t->sp;
   if (info->instance_count != 1)
      return "instanced draw";
   if (info->mode != MESA_PRIM_TRIANGLES && info->mode != MESA_PRIM_TRIANGLE_STRIP && info->mode != MESA_PRIM_TRIANGLE_FAN &&
       info->mode != MESA_PRIM_QUADS && info->mode != MESA_PRIM_QUAD_STRIP && info->mode != MESA_PRIM_POLYGON)
      return "not a triangle primitive";
   if (sp->reduced_api_prim != MESA_PRIM_TRIANGLES || sp->gs || sp->num_so_targets > 0)
      return "geometry shader or stream output";
   if (!framebuffer_ok(sp))
      return (unsigned long)sp->framebuffer.width * sp->framebuffer.height < g_min_pixels
                ? "window smaller than TTR_GL_MIN_PIXELS pixels (softpipe is faster there)"
                : "framebuffer format or size";
   const struct pipe_rasterizer_state *r = sp->rasterizer;
   if (r->fill_front != PIPE_POLYGON_MODE_FILL || r->fill_back != PIPE_POLYGON_MODE_FILL)
      return "polygon mode is not fill";
   if (r->scissor || r->poly_stipple_enable || r->poly_smooth || r->multisample || r->light_twoside || r->rasterizer_discard)
      return "rasterizer state (scissor, stipple, smoothing, multisample, two-sided colour, discard)";
   if (sp->active_query_count || sp->active_statistics_queries)
      return "active query";
   const struct pipe_blend_state *b = sp->blend;
   if (b->logicop_enable || b->alpha_to_coverage || b->alpha_to_one || b->rt[0].colormask != 0xf)
      return "logic op, alpha to coverage or colour write mask";
   const struct pipe_depth_stencil_alpha_state *ds = sp->depth_stencil;
   if (ds->stencil[0].enabled || ds->stencil[1].enabled || ds->alpha_enabled || ds->depth_bounds_test)
      return "stencil, alpha or depth bounds test";
   ttr_state st = {TTR_DEPTH_ALWAYS, 0, TTR_BLEND_NONE};   /* depth test off: nothing is tested or written */
   if (ds->depth_enabled) {
      st.depth_write = ds->depth_writemask;
      switch (ds->depth_func) {
      case PIPE_FUNC_LESS: st.depth_func = TTR_DEPTH_LESS; break;
      case PIPE_FUNC_LEQUAL: st.depth_func = TTR_DEPTH_LEQUAL; break;
      case PIPE_FUNC_ALWAYS: st.depth_func = TTR_DEPTH_ALWAYS; break;
      default: return "depth function";
      }
   }
   const struct pipe_rt_blend_state *rt = &b->rt[0];
   if (rt->blend_enable) {
      if (rt->rgb_func != PIPE_BLEND_ADD || rt->alpha_func != PIPE_BLEND_ADD || rt->rgb_src_factor != rt->alpha_src_factor ||
          rt->rgb_dst_factor != rt->alpha_dst_factor)
         return "blend equation or separate alpha factors";
      const unsigned sf = rt->rgb_src_factor, df = rt->rgb_dst_factor;
      if (sf == PIPE_BLENDFACTOR_SRC_ALPHA && df == PIPE_BLENDFACTOR_INV_SRC_ALPHA)
         st.blend = TTR_BLEND_ALPHA;
      else if (sf == PIPE_BLENDFACTOR_ONE && df == PIPE_BLENDFACTOR_ONE)
         st.blend = TTR_BLEND_ADD;
      else if (sf == PIPE_BLENDFACTOR_ONE && df == PIPE_BLENDFACTOR_INV_SRC_ALPHA)
         st.blend = TTR_BLEND_PREMULTIPLIED;
      else if (sf == PIPE_BLENDFACTOR_DST_COLOR && df == PIPE_BLENDFACTOR_ZERO)
         st.blend = TTR_BLEND_MULTIPLY;
      else if (sf == PIPE_BLENDFACTOR_SRC_ALPHA && df == PIPE_BLENDFACTOR_ONE)
         st.blend = TTR_BLEND_ALPHA_ADD;
      else
         return "blend factors";
   }
   t->draw_state = st;
   t->own_state = !(st.depth_func == TTR_DEPTH_LESS && st.depth_write && st.blend == TTR_BLEND_NONE);
   if (!sp->fs_variant)
      return "no fragment shader";
   /* softpipe derives the vertex layout and the setup info lazily, when the draw module asks for it. */
   softpipe_get_vbuf_vertex_info(sp);
   const struct tgsi_shader_info *fi = &sp->fs_variant->info;
   if (t->debug_tgsi && t->dumped_tokens != sp->fs_variant->tokens) {
      t->dumped_tokens = sp->fs_variant->tokens;
      tgsi_dump(sp->fs_variant->tokens, 0);
   }
   if (fi->writes_z || fi->writes_stencil || fi->num_outputs != 1)
      return "fragment shader writes depth or stencil, or several outputs";
   t->shaded = false;
   t->prog = NULL;
   const int k = fi->uses_kill ? -1 : passthrough_input(sp->fs_variant->tokens);
   if (k < 0) {
      /* Arithmetic fragment shader: translated to a program of the rasterizer. */
      const char *why = NULL;
      const struct ttr_prog *prog = get_program(t, sp->fs_variant, &why);
      if (!prog)
         return why;
      bool any_persp = false, any_linear = false;
      for (unsigned i = 0; i < prog->num_inputs; i++) {
         const unsigned j = prog->map[i].input;
         if (j >= fi->num_inputs)
            return "fragment shader input out of range";
         const enum sp_interp_mode mode = sp->setup_info.attrib[j].interp;
         t->input_slot[i] = sp->setup_info.attrib[j].src_index;
         t->input_interp[i] = mode;
         any_persp |= mode == SP_INTERP_PERSPECTIVE;
         any_linear |= mode == SP_INTERP_LINEAR;
         if (mode != SP_INTERP_PERSPECTIVE && mode != SP_INTERP_LINEAR && mode != SP_INTERP_CONSTANT)
            return "fragment input interpolation mode";
      }
      if (any_persp && any_linear)
         return "mixed perspective and screen-linear varyings";
      t->shaded = true;
      t->prog = prog;
      t->flat = false;
      t->perspective = any_persp;
      return NULL;
   }
   if (t->debug) {
      for (unsigned i = 0; i < fi->num_inputs; i++)
         fprintf(stderr, "tensix: fs input %u: semantic %d index %d interpolate %d -> setup interp %d src slot %d%s\n", i,
                 (int)fi->input_semantic_name[i], (int)fi->input_semantic_index[i], (int)fi->input_interpolate[i],
                 (int)sp->setup_info.attrib[i].interp, (int)sp->setup_info.attrib[i].src_index, (int)i == k ? " <- colour" : "");
   }
   const enum sp_interp_mode interp = sp->setup_info.attrib[k].interp;
   if (interp != SP_INTERP_LINEAR && interp != SP_INTERP_CONSTANT && interp != SP_INTERP_PERSPECTIVE) {
      if (t->debug)
         fprintf(stderr, "tensix: fragment input %d has interpolation mode %d, semantic %d\n", k, (int)interp,
                 (int)fi->input_semantic_name[k]);
      return "colour varying interpolation mode";
   }
   t->color_slot = sp->setup_info.attrib[k].src_index;
   t->flat = interp == SP_INTERP_CONSTANT;
   t->perspective = interp == SP_INTERP_PERSPECTIVE;
   return NULL;
}

/* ---- frame handling ------------------------------------------------------------------------------------------ */

/* Forget what a tile cache holds (nothing in it is wanted: the whole surface is about to be overwritten) and a
 * pending lazy clear with it. */
static void
discard_cache(struct softpipe_tile_cache *tc)
{
   for (unsigned pos = 0; pos < ARRAY_SIZE(tc->tile_addrs); pos++)
      tc->tile_addrs[pos].bits.invalid = 1;
   if (tc->clear_flags)
      memset(tc->clear_flags, 0, tc->clear_flags_size);
   tc->last_tile_addr.bits.invalid = 1;
}

/* Layout of a 32-bit colour format in the words of the surface: bit position of R and B (G is at 8, A at 24). */
static bool
color_shifts(enum pipe_format f, unsigned *r_shift, unsigned *b_shift, bool *has_alpha)
{
   switch (f) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_A8B8G8R8_UNORM:
      *r_shift = 0, *b_shift = 16, *has_alpha = true;
      return true;
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_X8B8G8R8_UNORM:
      *r_shift = 0, *b_shift = 16, *has_alpha = false;
      return true;
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_A8R8G8B8_UNORM:
      *r_shift = 16, *b_shift = 0, *has_alpha = true;
      return true;
   case PIPE_FORMAT_B8G8R8X8_UNORM:
   case PIPE_FORMAT_X8R8G8B8_UNORM:
      *r_shift = 16, *b_shift = 0, *has_alpha = false;
      return true;
   default:
      return false;
   }
}

/* The rendered depth into the depth buffer (stencil bits are kept for combined formats). */
static void
write_depth(struct sp_tensix *t)
{
   struct softpipe_context *sp = t->sp;
   struct softpipe_tile_cache *tc = sp->zsbuf_cache;
   const enum pipe_format zf = sp->framebuffer.zsbuf.format;
   const uint32_t w = t->width, h = t->height;
   if (!tc->num_maps)
      return;
   if (t->debug) {
      unsigned ones = 0;
      for (size_t i = 0; i < (size_t)w * h; i++)
         ones += t->depth[i] >= 0.9999f;
      fprintf(stderr, "tensix: writing depth (%ux%u, format %d, %u pixels at 1.0)\n", w, h, (int)zf, ones);
   }
   discard_cache(tc);
   uint8_t *dst = tc->transfer_map[0];
   const unsigned stride = tc->transfer[0]->stride;
   for (uint32_t y = 0; y < h; y++) {
      const float *src = t->depth + (size_t)y * w;
      uint32_t *d32 = (uint32_t *)(dst + (size_t)y * stride);
      uint16_t *d16 = (uint16_t *)(dst + (size_t)y * stride);
      for (uint32_t x = 0; x < w; x++) {
         const double zd = src[x] < 0.0f ? 0.0 : src[x] > 1.0f ? 1.0 : (double)src[x];   /* double: 1.0 * 16777215 + 0.5 must not round up to 2^24 */
         switch (zf) {
         case PIPE_FORMAT_Z24_UNORM_S8_UINT:   /* depth in the low 24 bits, stencil above */
            d32[x] = (d32[x] & 0xff000000u) | (uint32_t)(zd * 16777215.0 + 0.5);
            break;
         case PIPE_FORMAT_Z24X8_UNORM:
            d32[x] = (uint32_t)(zd * 16777215.0 + 0.5);
            break;
         case PIPE_FORMAT_S8_UINT_Z24_UNORM:   /* stencil in the low byte, depth above */
            d32[x] = (d32[x] & 0xffu) | ((uint32_t)(zd * 16777215.0 + 0.5) << 8);
            break;
         case PIPE_FORMAT_X8Z24_UNORM:
            d32[x] = (uint32_t)(zd * 16777215.0 + 0.5) << 8;
            break;
         case PIPE_FORMAT_Z32_UNORM:
            d32[x] = (uint32_t)(zd * 4294967295.0 + 0.5);
            break;
         case PIPE_FORMAT_Z32_FLOAT:
            memcpy(&d32[x], &src[x], 4);
            break;
         case PIPE_FORMAT_Z16_UNORM:
            d16[x] = (uint16_t)(zd * 65535.0 + 0.5);
            break;
         default:
            break;
         }
      }
   }
   sp->dirty_render_cache = true;
}

/* Make sure `depth` holds the depth of the last rendered frame (render the frame again if it was ended without). */
static bool
fetch_depth(struct sp_tensix *t)
{
   if (t->depth_cached)
      return true;
   t->depth_wanted = true;
   t->depth_rerenders++;
   const size_t px = (size_t)t->width * t->height;
   if (t->depth_px < px) {
      FREE(t->depth);
      t->depth = MALLOC(px * sizeof(float));
      t->depth_px = t->depth ? px : 0;
      if (!t->depth_px) {
         fprintf(stderr, "tensix: out of memory\n");
         return false;
      }
   }
   if (t->api.frame_depth(t->ctx, t->depth, 0) != 0) {
      fprintf(stderr, "tensix: depth of the frame failed: %s\n", t->api.last_error());
      return false;
   }
   t->depth_cached = true;
   return true;
}

/* Write a deferred depth buffer now (someone is about to read or draw into it). */
static void
materialize_depth(struct sp_tensix *t)
{
   if (!t->depth_pending)
      return;
   t->depth_pending = false;
   t->depth_unused = 0;
   const double t0 = now_ms();
   if (fetch_depth(t))
      write_depth(t);
   t->ms_depth += now_ms() - t0;
   t->depth_writes++;
   t->ms_write += now_ms() - t0;
}

/* End the frame in progress: the rendered colours go straight into the colour surface, converted to its layout on the way.
 * with_depth: read the depth plane back now (the caller is about to use it); otherwise it is rendered later if needed. */
static void
flush_frame(struct sp_tensix *t, bool with_depth)
{
   if (t->state != FRAME_OPEN)
      return;
   t->state = FRAME_NONE;
   struct softpipe_tile_cache *tc = t->sp->cbuf_cache[0];
   unsigned rs, bs;
   bool has_alpha;
   if (!tc->num_maps || !color_shifts(t->sp->framebuffer.cbufs[0].format, &rs, &bs, &has_alpha)) {
      fprintf(stderr, "tensix: the colour surface is not writable; frame dropped\n");
      return;
   }
   if (with_depth) {
      const size_t px = (size_t)t->width * t->height;
      if (t->depth_px < px) {
         FREE(t->depth);
         t->depth = MALLOC(px * sizeof(float));
         t->depth_px = t->depth ? px : 0;
         if (!t->depth_px) {
            fprintf(stderr, "tensix: out of memory\n");
            return;
         }
      }
   }
   discard_cache(tc);   /* nothing in the cache is wanted: the rendering owns every pixel */
   t->depth_pending = false;
   t->depth_cached = false;
   const double t0 = now_ms();
   const uint32_t flags = (rs != 0 ? TTR_OUT_SWAP_RB : 0) | (has_alpha ? 0 : TTR_OUT_OPAQUE);
   if (t->api.frame_end_ex(t->ctx, tc->transfer_map[0], tc->transfer[0]->stride, flags, with_depth ? t->depth : NULL, 0) != 0) {
      fprintf(stderr, "tensix: frame failed: %s\n", t->api.last_error());
      return;
   }
   t->sp->dirty_render_cache = true;
   t->frames_fast++;
   /* The depth buffer is only needed if something reads it before the next full clear overwrites it. In the usual
    * loop (draw, swap, clear) nothing does, so the write is deferred until a draw, a framebuffer change, a partial
    * clear or a CPU access to the depth resource asks for it. */
   t->depth_pending = true;
   t->depth_cached = with_depth;
   t->ms_render += now_ms() - t0;
   if (t->debug && t->frames_fast % 300 == 0)
      fprintf(stderr, "tensix: %u frames (%u depth writes): %.2f ms per frame in the rasterizer and the copy into softpipe's colour surface (%s), %.2f ms per frame writing depth (%u times)\n",
              t->frames_fast, t->depth_writes, t->ms_render / t->frames_fast, t->api.context_report(t->ctx),
              t->ms_write / t->frames_fast, t->depth_writes);
}

/* softpipe's flush (glFlush, a swap, a read-back): render what the frame holds. */
void
sp_tensix_flush(struct sp_tensix *t)
{
   if (t)
      flush_frame(t, t->depth_wanted);
}

void
sp_tensix_resource_access(struct sp_tensix *t, const struct pipe_resource *resource)
{
   if (t && t->depth_pending && resource == t->sp->framebuffer.zsbuf.texture)
      materialize_depth(t);
}

void
sp_tensix_framebuffer_changing(struct sp_tensix *t)
{
   if (!t)
      return;
   flush_frame(t, true);
   materialize_depth(t);
   t->state = FRAME_NONE;
}

void
sp_tensix_clear(struct sp_tensix *t, unsigned buffers, const struct pipe_scissor_state *scissor,
                const union pipe_color_union *color, double depth)
{
   if (!t)
      return;
   const bool full = !scissor && (buffers & PIPE_CLEAR_COLOR0) && (buffers & PIPE_CLEAR_DEPTH);
   if (!full || !framebuffer_ok(t->sp)) {
      /* A partial clear or a clear of other buffers touches pixels the pending frame owns. */
      flush_frame(t, true);
      materialize_depth(t);
      t->state = FRAME_NONE;
      return;
   }
   /* Everything drawn before this clear is overwritten by it: the pending frame is dropped, not rendered, and a
    * deferred depth buffer is never written. */
   if (t->depth_pending && t->depth_cached && ++t->depth_unused >= 4)
      t->depth_wanted = false;   /* the depth plane came with the frame and nobody wanted it, four times in a row */
   t->depth_pending = false;
   t->state = FRAME_CLEARED;
   for (int i = 0; i < 4; i++)
      t->clear_color[i] = color->f[i];
   t->clear_depth = (float)depth;
   t->width = t->sp->framebuffer.width;
   t->height = t->sp->framebuffer.height;
}

/* The draw goes to softpipe: render what the frame holds so far, the rest of the frame is softpipe's. */
static void
refuse_draw(struct sp_tensix *t, const char *why)
{
   if (!t->logged_unsupported || t->debug) {
      fprintf(stderr, "tensix: draw handled by softpipe: %s\n", why);
      t->logged_unsupported = true;
   }
   if (t->state == FRAME_OPEN) {
      t->frames_fallback++;
      flush_frame(t, true);
   }
   materialize_depth(t);
   t->state = FRAME_NONE;
   t->draws_slow++;
}

/* Hand the translated shader of the draw (with the uniforms it reads, now) to the open frame. NULL on success. */
static const char *
register_shader(struct sp_tensix *t)
{
   const struct ttr_prog *p = t->prog;
   float consts[1024];
   const struct tgsi_exec_consts_info *cb = &t->sp->mapped_constants[MESA_SHADER_FRAGMENT][0];
   if (!ttr_prog_constants(p, (const float *)cb->ptr, cb->size, t->sp->framebuffer.height, consts))
      return "uniform outside the constant buffer";
   const ttr_shader_desc desc = {p->code, p->n_instr, consts, p->n_consts, p->num_inputs, p->num_regs, p->out_reg, p->fragcoord_reg};
   t->shader_id = t->api.frame_shader(t->ctx, &desc);
   if (!t->shader_id) {
      if (t->debug)
         fprintf(stderr, "tensix: shader refused: %s\n", t->api.last_error());
      return "fragment shader program does not fit the frame";
   }
   return NULL;
}

void
sp_tensix_begin_draw(struct sp_tensix *t, const struct pipe_draw_info *info)
{
   if (!t)
      return;
   t->tri_active = false;
   if (t->state == FRAME_NONE) {
      materialize_depth(t);   /* softpipe is about to draw into the depth buffer */
      t->draws_slow++;
      return;
   }
   const char *why = draw_refused(t, info);
   if (why) {
      refuse_draw(t, why);
      return;
   }
   if (t->state == FRAME_CLEARED) {
      if (t->api.frame_begin(t->ctx, t->width, t->height, t->clear_color, t->clear_depth, TTR_DEPTH_LESS) != 0) {
         fprintf(stderr, "tensix: frame_begin failed: %s\n", t->api.last_error());
         t->state = FRAME_NONE;
         t->draws_slow++;
         return;
      }
      t->state = FRAME_OPEN;
   }
   if (t->shaded && (why = register_shader(t)) != NULL) {
      refuse_draw(t, why);
      return;
   }
   t->nverts = 0;
   t->tri_active = true;
   t->draws_fast++;
}

bool
sp_tensix_tri_active(const struct sp_tensix *t)
{
   return t && t->tri_active;
}

static void
submit_batch(struct sp_tensix *t)
{
   if (t->nverts && t->shaded) {
      if (t->api.frame_triangles_shaded(t->ctx, t->verts, (uint32_t)(t->nverts / 3), t->shader_id, t->attrs, t->perspective, 0,
                                        t->own_state ? &t->draw_state : NULL) != 0)
         fprintf(stderr, "tensix: frame_triangles_shaded failed: %s\n", t->api.last_error());
   } else if (t->nverts &&
       t->api.frame_triangles(t->ctx, t->verts, (uint32_t)(t->nverts / 3),
                              t->flat ? TTR_SHADE_FLAT : t->perspective ? TTR_SHADE_PERSPECTIVE : TTR_SHADE_LINEAR, 0,
                              t->own_state ? &t->draw_state : NULL) != 0) {
      fprintf(stderr, "tensix: frame_triangles failed: %s\n", t->api.last_error());
   }
   t->nverts = 0;
}

/* Returns true if the triangle was taken. False: the fast path stopped (the frame so far was rendered into the
 * buffers) and the caller must rasterize this triangle with softpipe. */
bool
sp_tensix_tri(struct sp_tensix *t, const float (*v0)[4], const float (*v1)[4], const float (*v2)[4],
              const float (*vprovoke)[4])
{
   const float (*in[3])[4] = {v0, v1, v2};
   if (t->perspective && !(in[0][0][3] > 0.0f && in[1][0][3] > 0.0f && in[2][0][3] > 0.0f)) {
      /* a vertex at or behind the eye plane (1/w <= 0): not expected after clipping; let softpipe decide */
      submit_batch(t);
      t->tri_active = false;
      t->frames_fallback++;
      flush_frame(t, true);
      t->state = FRAME_NONE;
      return false;
   }
   if (t->nverts + 3 > t->cap) {
      t->cap = t->cap ? t->cap * 2 : 3 * 1024;
      t->verts = REALLOC(t->verts, 0, t->cap * sizeof(ttr_vertex));
   }
   if (t->shaded) {
      const size_t need = (t->nverts + 3) * (size_t)t->prog->num_inputs;
      if (need > t->attrs_cap) {
         t->attrs_cap = MAX2(need * 2, 3 * 1024 * 8);
         t->attrs = REALLOC(t->attrs, 0, t->attrs_cap * sizeof(float));
      }
   }
   for (int k = 0; k < 3; k++) {
      ttr_vertex *o = &t->verts[t->nverts];
      const float *p = in[k][0];   /* slot 0: window position, w holds 1/w */
      o->x = p[0];
      o->y = p[1];
      o->z = p[2];
      o->rhw = p[3];
      if (t->shaded) {
         /* the varyings of this vertex: constant ones take the provoking vertex's value */
         float *a = &t->attrs[t->nverts * (size_t)t->prog->num_inputs];
         for (unsigned i = 0; i < t->prog->num_inputs; i++) {
            const float (*src)[4] = t->input_interp[i] == SP_INTERP_CONSTANT ? vprovoke : in[k];
            a[i] = src[t->input_slot[i]][t->prog->map[i].comp];
         }
         memset(o->color, 0, sizeof o->color);
      } else {
         memcpy(o->color, t->flat ? vprovoke[t->color_slot] : in[k][t->color_slot], 4 * sizeof(float));
      }
      t->nverts++;
   }
   return true;
}

void
sp_tensix_end_draw(struct sp_tensix *t)
{
   if (!t || !t->tri_active)
      return;
   t->tri_active = false;
   submit_batch(t);
}
