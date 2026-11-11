/*
 * Copyright 2026 tt-raster contributors
 * SPDX-License-Identifier: MIT
 *
 * Tensix fast path for softpipe: untextured, unblended triangles with a pass-through colour fragment shader are
 * collected for a whole frame and rasterized by libttraster (Tenstorrent Tensix cores, or its CPU reference
 * backend). Everything else runs on softpipe unchanged. See mesa/README.md in the tt-raster repository.
 */

#ifndef SP_TENSIX_H
#define SP_TENSIX_H

#include "pipe/p_state.h"
#include "util/u_inlines.h"

struct softpipe_context;
struct sp_tensix;

/** NULL when the fast path is disabled (environment variable TTR_GL unset) or libttraster cannot be loaded. */
struct sp_tensix *sp_tensix_create(struct softpipe_context *sp);
void sp_tensix_destroy(struct sp_tensix *t);

/** Called from softpipe_clear after the normal clear. Starts a Tensix frame on a full colour+depth clear. */
void sp_tensix_clear(struct sp_tensix *t, unsigned buffers, const struct pipe_scissor_state *scissor,
                     const union pipe_color_union *color, double depth);

/** Called at the start of a draw after derived state is up to date. Decides where this draw's triangles go. */
void sp_tensix_begin_draw(struct sp_tensix *t, const struct pipe_draw_info *info);
/** True if the current draw's triangles must go to sp_tensix_tri() instead of the software rasterizer. */
bool sp_tensix_tri_active(const struct sp_tensix *t);
/** Takes one post-transform triangle. Returns false if the fast path stopped at this triangle (the frame so far has
 *  been rendered into the buffers): the caller must then rasterize it with softpipe. */
bool sp_tensix_tri(struct sp_tensix *t, const float (*v0)[4], const float (*v1)[4], const float (*v2)[4],
                   const float (*vprovoke)[4]);
void sp_tensix_end_draw(struct sp_tensix *t);

/** Render the pending frame and write it into the colour and depth buffers. Call before anything reads them. */
void sp_tensix_flush(struct sp_tensix *t);
/** The CPU is about to map this resource: write a deferred depth buffer first if it is the one. */
void sp_tensix_resource_access(struct sp_tensix *t, const struct pipe_resource *resource);
/** The framebuffer binding changes: render what is pending into the old one first. */
void sp_tensix_framebuffer_changing(struct sp_tensix *t);

#endif
