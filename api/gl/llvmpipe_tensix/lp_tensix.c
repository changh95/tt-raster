/*
 * Copyright 2026 tt-raster contributors
 * SPDX-License-Identifier: MIT
 *
 * Tensix offload for llvmpipe. See lp_tensix.h.
 *
 * Life of a frame (the same as sp_tensix.c for softpipe): a full colour+depth clear opens a frame (the clear is not given to
 * llvmpipe: the values are remembered, and the rasterizer frame starts at the first accepted draw, when the depth function is
 * known). Triangles of accepted draws arrive from llvmpipe's set-up stage, after clipping, viewport transform and culling, and go
 * to libttraster instead of llvmpipe's binner. The frame is rendered when llvmpipe flushes (a swap, a read-back, a different
 * framebuffer) or when a draw the offload cannot do arrives; the result is written straight into the colour and depth surfaces and
 * llvmpipe carries on from that content, so output is never wrong, only slower.
 *
 * Environment: TTR_LP = "tensix" | "reference" selects the backend and enables the offload; TTR_LP_CHIPS = number of chips;
 * TTRASTER_LIB = path of libttraster.so; TTR_LP_DEBUG = 1 logs decisions, 2 also dumps the TGSI of every fragment shader.
 */

#include "lp_tensix.h"

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lp_bld_interp.h"
#include "lp_context.h"
#include "lp_flush.h"
#include "lp_fence.h"
#include "lp_rast.h"
#include "lp_screen.h"
#include "lp_setup.h"
#include "lp_setup_context.h"
#include "lp_state_fs.h"
#include "lp_texture.h"
#include "lp_tensix_shader.h"
#include "nir.h"
#include "nir/nir_to_tgsi.h"
#include "tgsi/tgsi_dump.h"
#include "tgsi/tgsi_parse.h"
#include "tgsi/tgsi_scan.h"
#include "util/format/u_format.h"
#include "util/u_memory.h"

/* ---- libttraster ABI (core/include/ttraster/ttraster.h), loaded with dlopen --------------------------------------- */

typedef struct {
   float x, y, z, rhw;
   float color[4];
} ttr_vertex;

enum { TTR_DEPTH_LESS = 0, TTR_DEPTH_LEQUAL = 1, TTR_DEPTH_ALWAYS = 2 };
enum { TTR_BLEND_NONE = 0, TTR_BLEND_ALPHA = 1, TTR_BLEND_ADD = 2, TTR_BLEND_PREMULTIPLIED = 3, TTR_BLEND_MULTIPLY = 4, TTR_BLEND_ALPHA_ADD = 5 };
typedef struct {
   int depth_func, depth_write, blend;
} ttr_state;
enum { TTR_SHADE_LINEAR = 0, TTR_SHADE_FLAT = 1, TTR_SHADE_PERSPECTIVE = 2 };
enum { TTR_OUT_SWAP_RB = 1, TTR_OUT_OPAQUE = 2 };

enum { TTR_DEPTH_F32, TTR_DEPTH_U32, TTR_DEPTH_U16, TTR_DEPTH_U24_LOW, TTR_DEPTH_U24_HIGH, TTR_DEPTH_U24_LOW_X, TTR_DEPTH_U24_HIGH_X };
typedef struct {
   uint8_t *data;
   uint32_t stride_bytes;
   int format;
} ttr_depth_target;

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
   int (*frame_end_depth)(void *ctx, uint8_t *rgba, uint32_t stride, uint32_t color_flags, const ttr_depth_target *depth, int flip_y);
   const char *(*last_error)(void);
   const char *(*context_report)(void *ctx);
};

enum frame_state {
   FRAME_NONE,     /* nothing pending; draws go to llvmpipe (or wait for a clear) */
   FRAME_CLEARED,  /* a full clear was seen; the rasterizer frame has not begun */
   FRAME_OPEN,     /* the rasterizer frame is accumulating triangles */
};

struct lp_tensix {
   struct lp_setup_context *setup;
   struct llvmpipe_context *lp;
   void *lib;
   struct ttr_api api;
   void *ctx;
   bool debug, debug_tgsi;
   bool busy;               /* inside lp_tensix_flush (llvmpipe may call back into the flush) */

   enum frame_state state;
   float clear_color[4];
   float clear_depth;
   uint32_t width, height;                  /* of the open frame */
   struct pipe_resource *cbuf, *zbuf;       /* the surfaces the open frame writes */
   unsigned clevel, zlevel, cfirst, zfirst;
   enum pipe_format cformat, zformat;
   bool loaded;             /* the frame was opened on the surfaces' uniform content (a load, not a clear) */
   bool load_blocked;       /* a load was wasted (the draw was refused): no more until llvmpipe flushes */
   unsigned load_attempts;  /* loads tried since llvmpipe last flushed */

   /* Translated fragment shaders, by llvmpipe's shader number (and a hash of the tokens). */
   struct {
      unsigned no;
      struct lp_ttr_prog *prog;   /* NULL: refused */
      const char *why;
   } shaders[32];
   unsigned n_shaders, next_shader;

   /* The draw in progress. */
   bool state_ok;           /* the pipeline state of this draw is acceptable */
   bool prim_ok;            /* the primitive type of this draw is a triangle family */
   const char *why;         /* otherwise why not */
   bool shaded;             /* the fragment shader is a translated program (otherwise a colour pass-through) */
   const struct lp_ttr_prog *prog;
   uint32_t shader_id;      /* its id in the open frame */
   int input_slot[32];      /* per program input component: the vertex slot of the fragment input, and its interpolation mode */
   int input_interp[32];
   float *attrs;            /* varyings of the batch: num_inputs floats per vertex */
   size_t attrs_cap;
   int color_slot;          /* vertex slot of the colour varying */
   bool flat;
   bool perspective;
   ttr_state draw_state;
   bool own_state;
   bool batch_open;         /* triangles of the current draw are in `verts`; the batch keeps the state it was opened with */
   ttr_state batch_state;
   bool batch_own_state, batch_shaded, batch_flat, batch_perspective;
   uint32_t batch_shader_id;
   ttr_vertex *verts;
   size_t nverts, cap;

   unsigned frames, draws_fast, draws_slow, fallbacks, clears_seen, clears_not_taken, tris_outside, loads, loads_not_uniform;
   bool logged_unsupported;
};

#define DBG(t, ...)                       \
   do {                                   \
      if ((t)->debug) {                   \
         fprintf(stderr, "tensix: " __VA_ARGS__); \
         fputc('\n', stderr);             \
      }                                   \
   } while (0)

/* ---- setup --------------------------------------------------------------------------------------------------- */

static bool
load_api(struct lp_tensix *t, const char *path)
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
   LOAD(frame_end_depth, "ttr_frame_end_depth")
   LOAD(last_error, "ttr_last_error")
   LOAD(context_report, "ttr_context_report")
#undef LOAD
   return true;
}

bool
lp_tensix_enabled(void)
{
   const char *mode = getenv("TTR_LP");
   return mode && *mode && strcmp(mode, "0") != 0;
}

struct lp_tensix *
lp_tensix_create(struct lp_setup_context *setup)
{
   const char *mode = getenv("TTR_LP");
   if (!lp_tensix_enabled())
      return NULL;
   const char *lib = getenv("TTRASTER_LIB");
   struct lp_tensix *t = CALLOC_STRUCT(lp_tensix);
   if (!t)
      return NULL;
   t->setup = setup;
   t->lp = llvmpipe_context(setup->pipe);
   t->debug = getenv("TTR_LP_DEBUG") != NULL;
   t->debug_tgsi = t->debug && atoi(getenv("TTR_LP_DEBUG")) >= 2;
   if (!load_api(t, lib && *lib ? lib : "libttraster.so")) {
      FREE(t);
      return NULL;
   }
   const char *chips = getenv("TTR_LP_CHIPS");
   t->ctx = t->api.context_create(!strcmp(mode, "reference") ? "reference" : "tensix", chips ? (uint32_t)atoi(chips) : 1);
   if (!t->ctx) {
      fprintf(stderr, "tensix: cannot create the rasterizer context: %s\n", t->api.last_error());
      dlclose(t->lib);
      FREE(t);
      return NULL;
   }
   fprintf(stderr, "tensix: llvmpipe offload enabled (backend %s)\n", !strcmp(mode, "reference") ? "reference" : "tensix");
   return t;
}

void
lp_tensix_destroy(struct lp_tensix *t)
{
   if (!t)
      return;
   if (t->debug)
      fprintf(stderr, "tensix: %u frames offloaded, %u draws accepted, %u draws refused, %u mid-frame fallbacks; %u clears seen, %u not taken; %u frames opened on loaded content (%u surfaces not uniform); %u triangles outside a frame\n",
              t->frames, t->draws_fast, t->draws_slow, t->fallbacks, t->clears_seen, t->clears_not_taken, t->loads, t->loads_not_uniform,
              t->tris_outside);
   t->api.context_destroy(t->ctx);
   FREE(t->verts);
   FREE(t->attrs);
   for (unsigned i = 0; i < t->n_shaders; i++)
      lp_ttr_prog_free(t->shaders[i].prog);
   dlclose(t->lib);
   FREE(t);
}

/* ---- what the offload can do --------------------------------------------------------------------------------- */

static bool
color_format_ok(enum pipe_format f, unsigned *r_shift, bool *has_alpha)
{
   switch (f) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_A8B8G8R8_UNORM:
      *r_shift = 0, *has_alpha = true;
      return true;
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_X8B8G8R8_UNORM:
      *r_shift = 0, *has_alpha = false;
      return true;
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_A8R8G8B8_UNORM:
      *r_shift = 16, *has_alpha = true;
      return true;
   case PIPE_FORMAT_B8G8R8X8_UNORM:
   case PIPE_FORMAT_X8R8G8B8_UNORM:
      *r_shift = 16, *has_alpha = false;
      return true;
   default:
      return false;
   }
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

static bool
format_has_stencil(enum pipe_format f)
{
   return f == PIPE_FORMAT_Z24_UNORM_S8_UINT || f == PIPE_FORMAT_S8_UINT_Z24_UNORM;
}

static bool
framebuffer_ok(const struct llvmpipe_context *lp)
{
   const struct pipe_framebuffer_state *fb = &lp->framebuffer;
   unsigned rs;
   bool ha;
   if (fb->nr_cbufs != 1 || !fb->cbufs[0].texture)
      return false;   /* (a framebuffer without a depth buffer is fine: no depth test, no depth output) */
   if (fb->width == 0 || fb->height == 0 || fb->width > 2048 || fb->height > 2048)
      return false;
   if (fb->cbufs[0].first_layer != fb->cbufs[0].last_layer || (fb->zsbuf.texture && fb->zsbuf.first_layer != fb->zsbuf.last_layer))
      return false;
   if (fb->cbufs[0].texture->nr_samples > 1 || (fb->zsbuf.texture && fb->zsbuf.texture->nr_samples > 1) || fb->cbufs[0].nr_samples > 1)
      return false;
   if (fb->cbufs[0].texture->width0 != fb->width || fb->cbufs[0].texture->height0 != fb->height)
      return false;   /* a render area smaller than the image: not worth the care */
   return color_format_ok(fb->cbufs[0].format, &rs, &ha) && (!fb->zsbuf.texture || depth_format_ok(fb->zsbuf.format));
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
      if (inst->Instruction.Opcode != TGSI_OPCODE_MOV || inst->Instruction.Saturate || inst->Instruction.NumDstRegs != 1 ||
          inst->Instruction.NumSrcRegs != 1 || d->Register.File != TGSI_FILE_OUTPUT || d->Register.Index != 0 ||
          d->Register.WriteMask != TGSI_WRITEMASK_XYZW || d->Register.Indirect || d->Register.Dimension ||
          s->Register.File != TGSI_FILE_INPUT || s->Register.Indirect || s->Register.Dimension || s->Register.Absolute ||
          s->Register.Negate || s->Register.SwizzleX != TGSI_SWIZZLE_X || s->Register.SwizzleY != TGSI_SWIZZLE_Y ||
          s->Register.SwizzleZ != TGSI_SWIZZLE_Z || s->Register.SwizzleW != TGSI_SWIZZLE_W) {
         ok = false;
      } else {
         input = s->Register.Index;
      }
   }
   tgsi_parse_free(&parse);
   return ok && count == 1 ? input : -1;
}

/* The translated program of a fragment shader (cached by llvmpipe's shader number; a refusal is cached too, with its reason). llvmpipe
 * keeps NIR only, so the shader goes through nir_to_tgsi first, as it does for softpipe, and then through the same translator. A shader
 * that only passes a colour through needs no program: *why is then "passthrough". */
static const struct lp_ttr_prog *
get_program(struct lp_tensix *t, const struct lp_fragment_shader *fs, const char **why)
{
   for (unsigned i = 0; i < t->n_shaders; i++) {
      if (t->shaders[i].no == fs->no) {
         *why = t->shaders[i].why;
         return t->shaders[i].prog;
      }
   }
   const char *reason = NULL;
   struct lp_ttr_prog *prog = NULL;
   if (!fs->base.ir.nir) {
      reason = "fragment shader without NIR";
   } else {
      nir_shader *clone = nir_shader_clone(NULL, fs->base.ir.nir);
      /* llvmpipe lowered gl_FragColor to eight colour outputs, all written; only the first is a render target here. */
      nir_foreach_shader_out_variable(var, clone) {
         if (var->data.location != FRAG_RESULT_DATA0 && var->data.location != FRAG_RESULT_COLOR)
            var->data.mode = nir_var_shader_temp;
      }
      NIR_PASS(_, clone, nir_lower_global_vars_to_local);
      NIR_PASS(_, clone, nir_lower_vars_to_ssa);
      NIR_PASS(_, clone, nir_opt_dce);
      if (t->debug_tgsi)
         nir_print_shader(clone, stderr);
      const struct tgsi_token *tok = nir_to_tgsi(clone, t->lp->pipe.screen);
      if (!tok) {
         reason = "nir_to_tgsi failed";
      } else {
         if (t->debug_tgsi)
            tgsi_dump(tok, 0);
         if (passthrough_input(tok) >= 0) {
            reason = "passthrough";
         } else {
            prog = lp_ttr_prog_translate_ex(tok, &reason, true);
            DBG(t, "fragment shader %u %s%s", fs->no, prog ? "translated" : "not translated: ", prog ? "" : reason);
         }
         FREE((void *)tok);
      }
   }
   const unsigned slot = t->n_shaders < 32 ? t->n_shaders++ : (t->next_shader++ % 32);
   lp_ttr_prog_free(t->shaders[slot].prog);
   t->shaders[slot].no = fs->no;
   t->shaders[slot].prog = prog;
   t->shaders[slot].why = reason;
   *why = reason;
   return prog;
}

/* nir_to_tgsi does not survive every intrinsic llvmpipe's shaders can contain (bindless image loads of input attachments, global and
 * shared memory, atomics, barriers): screen the shader before converting it. Returns why it is refused, or NULL. */
static const char *
shader_nir_refused(struct lp_tensix *t, const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex)
               return "textures, images or storage buffers";
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            const char *name = nir_intrinsic_infos[nir_instr_as_intrinsic(instr)->intrinsic].name;
            if (strstr(name, "image") || strstr(name, "bindless") || strstr(name, "ssbo") || strstr(name, "fbfetch") ||
                strstr(name, "const_buf_base_addr"))
               return "textures, images or storage buffers";
            if (strstr(name, "global") || strstr(name, "shared") || strstr(name, "atomic") || strstr(name, "barrier") ||
                strstr(name, "scratch") || strstr(name, "sample_") || strstr(name, "ballot") || strstr(name, "subgroup") ||
                strstr(name, "vote") || strstr(name, "shuffle") || strstr(name, "quad")) {
               DBG(t, "shader refused for intrinsic %s", name);
               return "memory, subgroup or sample intrinsics";
            }
         }
      }
   }
   return NULL;
}

/* Why the current state cannot be offloaded, or NULL (and fills in the draw's description). */
static const char *
state_refused(struct lp_tensix *t)
{
   struct llvmpipe_context *lp = t->lp;
   struct lp_setup_context *setup = t->setup;
   if (!framebuffer_ok(lp))
      return "framebuffer format or size";
   if (lp->gs || lp->tes || lp->tcs || lp->num_so_targets > 0)
      return "geometry, tessellation or stream output";
   const struct pipe_rasterizer_state *r = lp->rasterizer;
   if (r->fill_front != PIPE_POLYGON_MODE_FILL || r->fill_back != PIPE_POLYGON_MODE_FILL)
      return "polygon mode is not fill";
   if (r->scissor) {
      /* Vulkan always has the scissor test on; a scissor that covers the whole framebuffer changes nothing. */
      const struct pipe_scissor_state *sc = &lp->scissors[0];
      if (sc->minx > 0 || sc->miny > 0 || sc->maxx < lp->framebuffer.width || sc->maxy < lp->framebuffer.height)
         return "scissor smaller than the framebuffer";
   }
   if (r->poly_stipple_enable)
      return "polygon stipple";
   if (r->poly_smooth || r->multisample || r->rasterizer_discard)
      return "polygon smoothing, multisample or rasterizer discard";
   if (r->light_twoside)
      return "two-sided colour";
   if (r->offset_tri && (r->offset_units != 0.0f || r->offset_scale != 0.0f))
      return "polygon offset";   /* (Vulkan applications enable depth bias with zero factors: nothing is added) */
   if (!r->half_pixel_center)
      return "pixel centres at integer coordinates";
   if (!r->depth_clip_near || !r->depth_clip_far)
      return "depth clamp";
   if (setup->viewport_index_slot >= 0 || setup->layer_slot >= 0 || setup->face_slot >= 0)
      return "layer, viewport index or facing varyings";
   if (lp->active_occlusion_queries || lp->active_statistics_queries || lp->active_primgen_queries || setup->active_binned_queries)
      return "active query";
   const struct pipe_blend_state *b = lp->blend;
   if (b->logicop_enable)
      return "logic op";
   if (b->alpha_to_coverage || b->alpha_to_one)
      return "alpha to coverage or alpha to one";
   if (b->rt[0].colormask != 0xf)
      return "colour write mask";   /* (one colour attachment: independent_blend_enable makes no difference) */
   const struct pipe_depth_stencil_alpha_state *ds = lp->depth_stencil;
   if (ds->stencil[0].enabled || ds->stencil[1].enabled || ds->alpha_enabled || ds->depth_bounds_test)
      return "stencil, alpha or depth bounds test";
   ttr_state st = {TTR_DEPTH_ALWAYS, 0, TTR_BLEND_NONE};
   if (ds->depth_enabled) {
      st.depth_write = ds->depth_writemask;
      switch (ds->depth_func) {
      case PIPE_FUNC_LESS: st.depth_func = TTR_DEPTH_LESS; break;
      case PIPE_FUNC_LEQUAL: st.depth_func = TTR_DEPTH_LEQUAL; break;
      case PIPE_FUNC_ALWAYS: st.depth_func = TTR_DEPTH_ALWAYS; break;
      default: return "depth function";
      }
   }
   if (!lp->framebuffer.zsbuf.texture)
      st.depth_func = TTR_DEPTH_ALWAYS, st.depth_write = 0;   /* nothing to test against or write */
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

   const struct lp_fragment_shader *fs = lp->fs;
   if (!fs)
      return "no fragment shader";
   /* What the shader uses decides, not what is bound (lavapipe binds null descriptors whatever the shader does). */
   const nir_shader *nir = fs->base.ir.nir;
   if (!nir || BITSET_LAST_BIT(nir->info.textures_used) || BITSET_LAST_BIT(nir->info.images_used) || nir->info.num_ssbos ||
       nir->info.num_textures || nir->info.num_images || nir->info.fs.uses_fbfetch_output || nir->info.writes_memory)
      return "textures, images or storage buffers";
   if (nir->info.outputs_written & (BITFIELD64_BIT(FRAG_RESULT_DEPTH) | BITFIELD64_BIT(FRAG_RESULT_STENCIL) | BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK)))
      return "shader writes depth, stencil or sample mask";
   const char *nir_why = shader_nir_refused(t, nir);
   if (nir_why)
      return nir_why;
   const char *why = NULL;
   const struct lp_ttr_prog *prog = get_program(t, fs, &why);
   t->shaded = false;
   t->prog = NULL;
   if (!prog) {
      if (!why || strcmp(why, "passthrough") != 0)
         return why ? why : "fragment shader";
      /* Pass-through: the colour is the single fragment input. */
      int k = -1;
      for (unsigned i = 0; i < fs->info.base.num_inputs; i++) {
         if (fs->info.base.input_semantic_name[i] == TGSI_SEMANTIC_COLOR || fs->info.base.input_semantic_name[i] == TGSI_SEMANTIC_GENERIC)
            k = (int)i;
      }
      if (k < 0 || fs->info.base.num_inputs != 1)
         return "pass-through shader with an unexpected input";
      const struct lp_shader_input *in = &fs->inputs[0];
      t->color_slot = in->src_index;
      t->flat = in->interp == LP_INTERP_CONSTANT || (in->interp == LP_INTERP_COLOR && r->flatshade);
      t->perspective = in->interp == LP_INTERP_PERSPECTIVE || (in->interp == LP_INTERP_COLOR && !r->flatshade);
      if (in->interp != LP_INTERP_CONSTANT && in->interp != LP_INTERP_COLOR && in->interp != LP_INTERP_LINEAR &&
          in->interp != LP_INTERP_PERSPECTIVE)
         return "colour varying interpolation mode";
      return NULL;
   }
   bool any_persp = false, any_linear = false;
   for (unsigned i = 0; i < prog->num_inputs; i++) {
      const unsigned j = prog->map[i].input;
      if (j >= fs->info.base.num_inputs)
         return "fragment shader input out of range";
      unsigned mode = fs->inputs[j].interp;
      if (mode == LP_INTERP_COLOR)
         mode = r->flatshade ? LP_INTERP_CONSTANT : LP_INTERP_PERSPECTIVE;
      t->input_slot[i] = fs->inputs[j].src_index;
      t->input_interp[i] = mode;
      any_persp |= mode == LP_INTERP_PERSPECTIVE;
      any_linear |= mode == LP_INTERP_LINEAR;
      if (mode != LP_INTERP_PERSPECTIVE && mode != LP_INTERP_LINEAR && mode != LP_INTERP_CONSTANT)
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

/* ---- surfaces ------------------------------------------------------------------------------------------------ */

/* How the library writes the depth surface of this format, or -1. */
static int
depth_target_format(enum pipe_format f)
{
   switch (f) {
   case PIPE_FORMAT_Z24_UNORM_S8_UINT: return TTR_DEPTH_U24_LOW;
   case PIPE_FORMAT_S8_UINT_Z24_UNORM: return TTR_DEPTH_U24_HIGH;
   case PIPE_FORMAT_Z24X8_UNORM: return TTR_DEPTH_U24_LOW_X;
   case PIPE_FORMAT_X8Z24_UNORM: return TTR_DEPTH_U24_HIGH_X;
   case PIPE_FORMAT_Z32_UNORM: return TTR_DEPTH_U32;
   case PIPE_FORMAT_Z32_FLOAT: return TTR_DEPTH_F32;
   case PIPE_FORMAT_Z16_UNORM: return TTR_DEPTH_U16;
   default: return -1;
   }
}

/* Wait until the rasterizer threads of llvmpipe are done with everything that was submitted: the frame is about to write into surfaces
 * they may still be drawing into. */
static void
wait_rasterizer(struct lp_tensix *t)
{
   struct llvmpipe_screen *screen = llvmpipe_screen(t->lp->pipe.screen);
   struct lp_fence *fence = NULL;
   mtx_lock(&screen->rast_mutex);
   lp_rast_fence(screen->rast, &fence);
   mtx_unlock(&screen->rast_mutex);
   if (fence) {
      lp_fence_wait(fence);
      lp_fence_reference(&fence, NULL);
   }
}

static void
submit_batch(struct lp_tensix *t)
{
   if (t->nverts && t->batch_shaded) {
      if (t->api.frame_triangles_shaded(t->ctx, t->verts, (uint32_t)(t->nverts / 3), t->batch_shader_id, t->attrs, t->batch_perspective, 0,
                                        t->batch_own_state ? &t->batch_state : NULL) != 0)
         fprintf(stderr, "tensix: frame_triangles_shaded failed: %s\n", t->api.last_error());
   } else if (t->nverts &&
              t->api.frame_triangles(t->ctx, t->verts, (uint32_t)(t->nverts / 3),
                                     t->batch_flat ? TTR_SHADE_FLAT : t->batch_perspective ? TTR_SHADE_PERSPECTIVE : TTR_SHADE_LINEAR, 0,
                                     t->batch_own_state ? &t->batch_state : NULL) != 0) {
      fprintf(stderr, "tensix: frame_triangles failed: %s\n", t->api.last_error());
   }
   t->nverts = 0;
   t->batch_open = false;
}

/* The frame was opened by a clear that llvmpipe did not do, and nothing was drawn into it (the next draw is refused, or the frame is
 * flushed): do the clear now. */
static void
apply_clear(struct lp_tensix *t)
{
   if (t->state != FRAME_CLEARED)
      return;
   t->state = FRAME_NONE;
   if (t->loaded)
      return;   /* the surfaces already hold this content */
   union pipe_color_union c;
   for (int i = 0; i < 4; i++)
      c.f[i] = t->clear_color[i];
   const bool was_busy = t->busy;
   t->busy = true;   /* the clear passes through our hook */
   lp_setup_clear(t->setup, &c, t->clear_depth, 0, PIPE_CLEAR_COLOR0 | (t->zbuf ? PIPE_CLEAR_DEPTH : 0));
   t->busy = was_busy;
}

/* End the frame in progress: colours (converted to the surface's layout) and depth (converted to its format) go straight into the surfaces. */
static void
flush_frame(struct lp_tensix *t)
{
   if (t->state == FRAME_CLEARED) {
      apply_clear(t);
      return;
   }
   if (t->state != FRAME_OPEN)
      return;
   t->state = FRAME_NONE;
   if (t->batch_open)
      submit_batch(t);
   unsigned rs;
   bool has_alpha;
   const int dfmt = t->zbuf ? depth_target_format(t->zformat) : -1;
   if (!color_format_ok(t->cformat, &rs, &has_alpha) || (t->zbuf && dfmt < 0)) {
      fprintf(stderr, "tensix: the surface format is not supported; frame dropped\n");
      return;
   }
   wait_rasterizer(t);
   uint8_t *dst = llvmpipe_resource_map(t->cbuf, t->clevel, t->cfirst, LP_TEX_USAGE_WRITE_ALL);
   uint8_t *zdst = t->zbuf ? llvmpipe_resource_map(t->zbuf, t->zlevel, t->zfirst, LP_TEX_USAGE_READ_WRITE) : NULL;
   if (dst && (zdst || !t->zbuf)) {
      const uint32_t flags = (rs != 0 ? TTR_OUT_SWAP_RB : 0) | (has_alpha ? 0 : TTR_OUT_OPAQUE);
      const ttr_depth_target zt = {zdst, t->zbuf ? llvmpipe_resource_stride(t->zbuf, t->zlevel) : 0, dfmt};
      if (t->api.frame_end_depth(t->ctx, dst, llvmpipe_resource_stride(t->cbuf, t->clevel), flags, zdst ? &zt : NULL, 0) != 0)
         fprintf(stderr, "tensix: frame failed: %s\n", t->api.last_error());
      else {
         t->frames++;
         DBG(t, "frame rendered (%s)", t->api.context_report(t->ctx));
      }
   }
   if (zdst)
      llvmpipe_resource_unmap(t->zbuf, t->zlevel, t->zfirst);
   if (dst)
      llvmpipe_resource_unmap(t->cbuf, t->clevel, t->cfirst);
}

/* ---- the hooks ----------------------------------------------------------------------------------------------- */

/* The frame the next triangles go to starts with this content (`loaded`: the surfaces already hold it; otherwise a clear is pending). */
static void
begin_frame_state(struct lp_tensix *t, const float color[4], float depth, bool loaded)
{
   struct llvmpipe_context *lp = t->lp;
   t->state = FRAME_CLEARED;
   t->loaded = loaded;
   t->batch_open = false;
   t->nverts = 0;
   for (int i = 0; i < 4; i++)
      t->clear_color[i] = color[i];
   t->clear_depth = depth;
   t->width = lp->framebuffer.width;
   t->height = lp->framebuffer.height;
   t->cbuf = lp->framebuffer.cbufs[0].texture;
   t->zbuf = lp->framebuffer.zsbuf.texture;
   t->cformat = lp->framebuffer.cbufs[0].format;
   t->zformat = lp->framebuffer.zsbuf.format;
   t->clevel = lp->framebuffer.cbufs[0].level;
   t->zlevel = lp->framebuffer.zsbuf.level;
   t->cfirst = lp->framebuffer.cbufs[0].first_layer;
   t->zfirst = lp->framebuffer.zsbuf.first_layer;
}

unsigned
lp_tensix_clear(struct lp_tensix *t, const union pipe_color_union *color, double depth, unsigned stencil, unsigned flags)
{
   (void)stencil;
   if (!t || t->busy)
      return flags;
   t->clears_seen++;
   t->load_blocked = false;
   t->load_attempts = 0;
   struct llvmpipe_context *lp = t->lp;
   const unsigned both = PIPE_CLEAR_COLOR0 | PIPE_CLEAR_DEPTH;
   const bool has_depth = lp->framebuffer.zsbuf.texture != NULL;
   /* A clear of the colour buffer and, if there is a depth buffer, of the depth buffer (its stencil bits are not touched). */
   const bool full = (flags & PIPE_CLEAR_COLOR0) && (!has_depth || (flags & PIPE_CLEAR_DEPTH)) && !(flags & PIPE_CLEAR_STENCIL) &&
                     !(flags & ~(both | PIPE_CLEAR_STENCIL));
   if (!full || !framebuffer_ok(lp)) {
      t->clears_not_taken++;
      DBG(t, "clear not taken (flags 0x%x%s; framebuffer %ux%u, %u colour buffers, depth buffer %s)", flags, full ? ", framebuffer not supported" : "",
          lp->framebuffer.width, lp->framebuffer.height, lp->framebuffer.nr_cbufs, lp->framebuffer.zsbuf.texture ? "yes" : "no");
      lp_tensix_flush(t);
      return flags;
   }
   DBG(t, "clear taken: %ux%u, colour format %d, depth format %d", lp->framebuffer.width, lp->framebuffer.height,
       (int)lp->framebuffer.cbufs[0].format, (int)lp->framebuffer.zsbuf.format);
   /* Everything drawn before this clear is overwritten by it: an open frame is dropped, not rendered. */
   begin_frame_state(t, color->f, has_depth ? (float)depth : 1.0f, false);
   return 0;
}

/* ---- frames that start from what the surfaces hold ----------------------------------------------------------- */

/* True if every pixel's `(word & mask)` is the same (`bytes` is 2 or 4); *value is that word. */
static bool
uniform_surface(const uint8_t *base, size_t stride, unsigned width, unsigned height, unsigned bytes, uint32_t mask, uint32_t *value)
{
   uint32_t first = 0;
   for (unsigned y = 0; y < height; y++) {
      const uint8_t *row = base + (size_t)y * stride;
      for (unsigned x = 0; x < width; x++) {
         uint32_t w;
         if (bytes == 4)
            memcpy(&w, row + (size_t)x * 4, 4);
         else {
            uint16_t h;
            memcpy(&h, row + (size_t)x * 2, 2);
            w = h;
         }
         w &= mask;
         if (x == 0 && y == 0)
            first = w;
         else if (w != first)
            return false;
      }
   }
   *value = first;
   return true;
}

/* The depth (0..1) a surface word of this format means, if converting it back at the end of the frame gives the same word (so that
 * pixels no triangle touches keep their value exactly). */
static bool
depth_from_word(int dfmt, uint32_t w, float *z)
{
   double max;
   uint32_t v;
   switch (dfmt) {
   case TTR_DEPTH_F32: {
      float f;
      memcpy(&f, &w, 4);
      if (!(f >= 0.0f && f <= 1.0f))
         return false;
      *z = f;
      return true;
   }
   case TTR_DEPTH_U32: v = w, max = 4294967295.0; break;
   case TTR_DEPTH_U16: v = w & 0xffffu, max = 65535.0; break;
   case TTR_DEPTH_U24_LOW:
   case TTR_DEPTH_U24_LOW_X: v = w & 0xffffffu, max = 16777215.0; break;
   case TTR_DEPTH_U24_HIGH:
   case TTR_DEPTH_U24_HIGH_X: v = w >> 8, max = 16777215.0; break;
   default: return false;
   }
   const float f = (float)((double)v / max);
   *z = f;
   return (uint32_t)((double)f * max + 0.5) == v;
}

/* The draw about to be set up goes to a frame of its own that starts from what the colour (and depth) surfaces hold, if that is one value
 * everywhere (what a clear leaves; an image clear of Vulkan with loadOp LOAD afterwards does not pass through lp_setup_clear). llvmpipe's
 * own pending work on the surfaces is finished first. */
static void
open_loaded_frame(struct lp_tensix *t)
{
   struct llvmpipe_context *lp = t->lp;
   if (t->load_blocked || t->load_attempts >= 4)
      return;
   t->why = state_refused(t);
   t->state_ok = t->why == NULL;
   if (!t->state_ok) {
      DBG(t, "no frame opened: %s", t->why);
      return;
   }
   unsigned rs;
   bool has_alpha;
   const struct pipe_framebuffer_state *fb = &lp->framebuffer;
   if (!color_format_ok(fb->cbufs[0].format, &rs, &has_alpha))
      return;
   const int dfmt = fb->zsbuf.texture ? depth_target_format(fb->zsbuf.format) : -1;
   if (fb->zsbuf.texture && dfmt < 0)
      return;
   t->load_attempts++;

   const bool was_busy = t->busy;
   t->busy = true;   /* llvmpipe calls back into our flush */
   lp_setup_flush(t->setup, "tensix load");
   wait_rasterizer(t);
   t->busy = was_busy;

   bool ok = false;
   float color[4] = {0, 0, 0, 1}, depth = 1.0f;
   const uint8_t *c = llvmpipe_resource_map(fb->cbufs[0].texture, fb->cbufs[0].level, fb->cbufs[0].first_layer, LP_TEX_USAGE_READ);
   if (c) {
      uint32_t w;
      if (uniform_surface(c, llvmpipe_resource_stride(fb->cbufs[0].texture, fb->cbufs[0].level), fb->width, fb->height, 4,
                          has_alpha ? 0xffffffffu : 0x00ffffffu, &w)) {
         const uint32_t b[4] = {w & 0xff, (w >> 8) & 0xff, (w >> 16) & 0xff, (w >> 24) & 0xff};
         color[0] = (float)b[rs == 0 ? 0 : 2] / 255.0f;
         color[1] = (float)b[1] / 255.0f;
         color[2] = (float)b[rs == 0 ? 2 : 0] / 255.0f;
         color[3] = has_alpha ? (float)b[3] / 255.0f : 1.0f;
         ok = true;
      }
      llvmpipe_resource_unmap(fb->cbufs[0].texture, fb->cbufs[0].level, fb->cbufs[0].first_layer);
   }
   if (ok && fb->zsbuf.texture) {
      ok = false;
      const uint8_t *z = llvmpipe_resource_map(fb->zsbuf.texture, fb->zsbuf.level, fb->zsbuf.first_layer, LP_TEX_USAGE_READ);
      if (z) {
         const unsigned bytes = dfmt == TTR_DEPTH_U16 ? 2 : 4;
         const uint32_t mask = dfmt == TTR_DEPTH_U16 ? 0xffffu
                               : dfmt == TTR_DEPTH_U24_LOW || dfmt == TTR_DEPTH_U24_LOW_X ? 0x00ffffffu
                               : dfmt == TTR_DEPTH_U24_HIGH || dfmt == TTR_DEPTH_U24_HIGH_X ? 0xffffff00u
                                                                                            : 0xffffffffu;
         uint32_t w;
         ok = uniform_surface(z, llvmpipe_resource_stride(fb->zsbuf.texture, fb->zsbuf.level), fb->width, fb->height, bytes, mask, &w) &&
              depth_from_word(dfmt, w, &depth);
         llvmpipe_resource_unmap(fb->zsbuf.texture, fb->zsbuf.level, fb->zsbuf.first_layer);
      }
   }
   if (!ok) {
      t->loads_not_uniform++;
      DBG(t, "surfaces are not one value everywhere: llvmpipe draws");
      return;
   }
   DBG(t, "frame opened on loaded content: colour %g %g %g %g, depth %g", color[0], color[1], color[2], color[3], depth);
   begin_frame_state(t, color, depth, true);
   t->loads++;
}

static void refuse(struct lp_tensix *t, const char *why);

void
lp_tensix_set_prim(struct lp_tensix *t, enum mesa_prim prim)
{
   if (!t)
      return;
   t->prim_ok = prim == MESA_PRIM_TRIANGLES || prim == MESA_PRIM_TRIANGLE_STRIP || prim == MESA_PRIM_TRIANGLE_FAN ||
                prim == MESA_PRIM_QUADS || prim == MESA_PRIM_QUAD_STRIP || prim == MESA_PRIM_POLYGON;
   /* Lines and points do not pass through the triangle hook: what the frame holds must be in the surfaces before they are drawn. */
   if (!t->prim_ok && !t->busy && t->state != FRAME_NONE)
      refuse(t, "not a triangle primitive");
}

void
lp_tensix_update(struct lp_tensix *t)
{
   if (!t || t->busy)
      return;
   if (t->state == FRAME_NONE) {
      open_loaded_frame(t);
      return;
   }
   /* A new draw (or the next chunk of vertices of one): the batch of the previous one goes to the frame with the state it was drawn with. */
   if (t->batch_open)
      submit_batch(t);
   t->why = state_refused(t);
   t->state_ok = t->why == NULL;
   DBG(t, "update: %s", t->why ? t->why : "draw acceptable");
}

/* The draw cannot be offloaded: what the frame holds so far goes into the surfaces first, llvmpipe draws the rest. */
static void
refuse(struct lp_tensix *t, const char *why)
{
   if (!t->logged_unsupported || t->debug) {
      fprintf(stderr, "tensix: draw handled by llvmpipe: %s\n", why);
      t->logged_unsupported = true;
   }
   if (t->state == FRAME_OPEN)
      t->fallbacks++;
   else if (t->state == FRAME_CLEARED && t->loaded)
      t->load_blocked = true;   /* the load was for nothing: not again until llvmpipe flushes */
   flush_frame(t);   /* renders an open frame, or does the clear that nothing was drawn into */
   t->state = FRAME_NONE;
   t->draws_slow++;
}

static const char *
register_shader(struct lp_tensix *t)
{
   const struct lp_ttr_prog *p = t->prog;
   float consts[1024];
   /* Every constant buffer the program reads: buffer 0 holds uniforms, push constants and driver state, the others are uniform buffers
    * (lavapipe binds descriptor-set UBOs to the constant buffer slots). */
   struct lp_ttr_prog_cbuf bufs[TTR_MAX_CBUFS];
   memset(bufs, 0, sizeof bufs);
   for (unsigned b = 0; b < TTR_MAX_CBUFS && b < PIPE_MAX_CONSTANT_BUFFERS; b++) {
      const struct pipe_constant_buffer *cb = &t->lp->constants[MESA_SHADER_FRAGMENT][b];
      if (cb->user_buffer) {
         bufs[b].data = cb->user_buffer;
         bufs[b].bytes = cb->buffer_size;
      } else if (cb->buffer) {
         bufs[b].data = (const float *)((const uint8_t *)llvmpipe_resource_data(cb->buffer) + cb->buffer_offset);
         bufs[b].bytes = cb->buffer_size;
      }
   }
   if (!lp_ttr_prog_constants_multi(p, bufs, t->lp->framebuffer.height, consts))
      return "uniform outside the constant buffer";
   if (t->debug_tgsi) {
      fprintf(stderr, "tensix: constant buffer 0 %u bytes at %p, program constants:", bufs[0].bytes, (const void *)bufs[0].data);
      for (unsigned i = 0; i < p->n_consts && i < 16; i++)
         fprintf(stderr, " %g", consts[i]);
      fprintf(stderr, " | buffer floats:");
      for (unsigned i = 0; bufs[0].data && i < bufs[0].bytes / 4 && i < 8; i++)
         fprintf(stderr, " %g", bufs[0].data[i]);
      fprintf(stderr, " | fragcoord_reg %d\n", p->fragcoord_reg);
   }
   const ttr_shader_desc desc = {p->code, p->n_instr, consts, p->n_consts, p->num_inputs, p->num_regs, p->out_reg, p->fragcoord_reg};
   t->shader_id = t->api.frame_shader(t->ctx, &desc);
   if (!t->shader_id)
      return "fragment shader program does not fit the frame";
   return NULL;
}

bool
lp_tensix_tri(struct lp_tensix *t, const float (*v0)[4], const float (*v1)[4], const float (*v2)[4])
{
   if (!t || t->busy)
      return false;
   if (t->state == FRAME_NONE) {
      t->tris_outside++;
      return false;
   }
   if (t->debug && !t->batch_open)
      DBG(t, "triangle: prim_ok %d state_ok %d (%s)", t->prim_ok, t->state_ok, t->why ? t->why : "ok");
   if (!t->prim_ok || !t->state_ok) {
      refuse(t, !t->prim_ok ? "not a triangle primitive" : t->why ? t->why : "state");
      return false;
   }
   if (t->state == FRAME_CLEARED) {
      if (t->api.frame_begin(t->ctx, t->width, t->height, t->clear_color, t->clear_depth, TTR_DEPTH_LESS) != 0) {
         fprintf(stderr, "tensix: frame_begin failed: %s\n", t->api.last_error());
         t->state = FRAME_NONE;
         return false;
      }
      t->state = FRAME_OPEN;
   }
   /* The first triangle of a draw (or of a changed fragment shader or state): the batch of the previous one goes to the frame. */
   if (!t->batch_open) {
      const char *why = t->shaded ? register_shader(t) : NULL;
      if (why) {
         refuse(t, why);
         return false;
      }
      t->batch_open = true;
      t->batch_state = t->draw_state;
      t->batch_own_state = t->own_state;
      t->batch_shaded = t->shaded;
      t->batch_flat = t->flat;
      t->batch_perspective = t->perspective;
      t->batch_shader_id = t->shader_id;
      t->draws_fast++;
   }
   const float (*in[3])[4] = {v0, v1, v2};
   /* llvmpipe's rotation keeps the provoking vertex of flat colours first or last. */
   const float (*vprovoke)[4] = t->setup->flatshade_first ? v0 : v2;
   if (t->perspective && !(in[0][0][3] > 0.0f && in[1][0][3] > 0.0f && in[2][0][3] > 0.0f)) {
      refuse(t, "vertex at or behind the eye plane");
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
         float *a = &t->attrs[t->nverts * (size_t)t->prog->num_inputs];
         for (unsigned i = 0; i < t->prog->num_inputs; i++) {
            const float (*src)[4] = t->input_interp[i] == LP_INTERP_CONSTANT ? vprovoke : in[k];
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
lp_tensix_flush(struct lp_tensix *t)
{
   if (!t || t->busy)
      return;
   t->load_blocked = false;
   t->load_attempts = 0;
   if (t->state != FRAME_NONE)
      DBG(t, "flush with the frame %s", t->state == FRAME_CLEARED ? "only cleared" : "open");
   t->busy = true;
   flush_frame(t);
   t->busy = false;
}

void
lp_tensix_bind_framebuffer(struct lp_tensix *t, const struct pipe_framebuffer_state *fb)
{
   if (!t || t->busy)
      return;
   if (t->state == FRAME_NONE) {
      t->load_blocked = false;
      t->load_attempts = 0;
      return;
   }
   if (fb->nr_cbufs == 1 && fb->cbufs[0].texture == t->cbuf && fb->zsbuf.texture == t->zbuf && fb->width == t->width &&
       fb->height == t->height && fb->cbufs[0].format == t->cformat && fb->zsbuf.format == t->zformat && fb->cbufs[0].level == t->clevel &&
       fb->zsbuf.level == t->zlevel && fb->cbufs[0].first_layer == t->cfirst && fb->zsbuf.first_layer == t->zfirst)
      return;   /* the same surfaces again: the frame goes on */
   lp_tensix_flush(t);
}

bool
lp_tensix_references(const struct lp_tensix *t, const struct pipe_resource *resource)
{
   return t && t->state != FRAME_NONE && (resource == t->cbuf || resource == t->zbuf);
}
