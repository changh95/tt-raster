/*
 * Copyright 2026 tt-raster contributors
 * SPDX-License-Identifier: MIT
 *
 * Tensix offload for llvmpipe (and with it for lavapipe, which is built on llvmpipe): whole frames of triangles with
 * depth test, simple blending and straight-line fragment shaders are collected after clipping and set-up and rasterized by
 * libttraster (Tenstorrent Tensix cores, or its CPU reference backend). Everything else runs on llvmpipe unchanged. The
 * counterpart of sp_tensix.h for softpipe; see api/vulkan/README.md and docs/EVALUATION_GL_VULKAN.md.
 */

#ifndef LP_TENSIX_H
#define LP_TENSIX_H

#include "pipe/p_state.h"
#include "util/macros.h"
#include "util/u_prim.h"
#include "util/u_inlines.h"

struct lp_setup_context;
struct lp_tensix;

/** NULL when the offload is disabled (environment variable TTR_LP unset) or libttraster cannot be loaded. */
struct lp_tensix *lp_tensix_create(struct lp_setup_context *setup);
void lp_tensix_destroy(struct lp_tensix *t);

/** Called at the top of lp_setup_clear. Returns the clear flags llvmpipe still has to do: 0 if the clear was taken (a full colour+depth
 *  clear that opens a frame), otherwise `flags` unchanged, after any frame in progress has been rendered into the surfaces. */
unsigned lp_tensix_clear(struct lp_tensix *t, const union pipe_color_union *color, double depth, unsigned stencil, unsigned flags);

/** Called when the draw module announces the primitive type of the next vertices. */
void lp_tensix_set_prim(struct lp_tensix *t, enum mesa_prim prim);

/** Called when the setup state is brought up to date before a draw's vertices arrive: decides where this draw's triangles go. */
void lp_tensix_update(struct lp_tensix *t);

/** Called with every triangle that survived culling (vertices in counter-clockwise order, v0 or v2 the provoking vertex).
 *  Returns true if the offload took it. */
bool lp_tensix_tri(struct lp_tensix *t, const float (*v0)[4], const float (*v1)[4], const float (*v2)[4]);

/** Render the pending frame into the surfaces (llvmpipe has no scene using them that is still being rasterized: the caller waits).
 *  Called whenever llvmpipe flushes. */
void lp_tensix_flush(struct lp_tensix *t);

/** The framebuffer binding is about to change (lavapipe also rebinds the same one before a draw): render the pending frame into the
 *  old framebuffer unless the new one is the same. */
void lp_tensix_bind_framebuffer(struct lp_tensix *t, const struct pipe_framebuffer_state *fb);

/** True if the open frame will write this resource (so that llvmpipe flushes before anyone reads it). */
bool lp_tensix_references(const struct lp_tensix *t, const struct pipe_resource *resource);

#endif
