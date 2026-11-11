// SPDX-License-Identifier: Apache-2.0
//
// The C interface of the renderer core: contexts and frames (core/include/ttraster/ttraster.h).

#include "tensix/kernels/raster_layout.h"
#include "ttraster/ttraster.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "ttraster/backend.h"
#include "ttraster/frame.h"
#include "ttraster/parallel.h"
#ifdef TT_RASTER_WITH_TENSIX
#include "tensix/tensix_backend.h"
#endif

namespace {
thread_local std::string g_error;
}

extern "C" {

const char* ttr_last_error(void) { return g_error.c_str(); }

// ---- Raster API ----------------------------------------------------------------------------------------

struct ttr_context {
    std::unique_ptr<ttr::Backend> backend;
    ttr::Frame frame;
    bool open = false;
    bool ended = false;       // the last frame was rendered; its triangles are still in `frame` (ttr_frame_depth renders it again)
    std::string report;
    size_t table_words = 0;   // size of the device's program table for the frame's shaders (raster_layout.h)
};

ttr_context* ttr_context_create(const char* backend, uint32_t n_chips) {
    try {
        auto c = std::make_unique<ttr_context>();
        const std::string b = backend ? backend : "reference";
        if (b == "reference") {
            c->backend = std::make_unique<ttr::ReferenceBackend>();
        } else if (b == "tensix") {
#ifdef TT_RASTER_WITH_TENSIX
            c->backend = ttr::make_tensix_backend(n_chips == 0 ? 1 : n_chips);
#else
            g_error = "this libttraster was built without Tensix support";
            return nullptr;
#endif
        } else {
            g_error = "unknown backend '" + b + "'";
            return nullptr;
        }
        return c.release();
    } catch (const std::exception& e) {
        g_error = e.what();
        return nullptr;
    }
}

void ttr_context_destroy(ttr_context* ctx) { delete ctx; }
const char* ttr_context_report(ttr_context* ctx) { return ctx->report.c_str(); }

int ttr_frame_begin(
    ttr_context* ctx, uint32_t width, uint32_t height, const float clear_color[4], float clear_depth, int depth_func) {
    if (width == 0 || height == 0 || width > static_cast<uint32_t>(ttr::kMaxTargetSize) || height > static_cast<uint32_t>(ttr::kMaxTargetSize)) {
        g_error = "frame size must be 1 to 2048 pixels in each direction";
        return -1;
    }
    ctx->frame.width = width;
    ctx->frame.height = height;
    std::memcpy(ctx->frame.clear_color, clear_color, 4 * sizeof(float));
    ctx->frame.clear_depth = clear_depth;
    ctx->frame.depth_func = depth_func == TTR_DEPTH_LESS      ? ttr::DepthFunc::Less
                            : depth_func == TTR_DEPTH_ALWAYS ? ttr::DepthFunc::Always
                                                             : ttr::DepthFunc::LessEqual;
    ctx->frame.cull = ttr::CullMode::None;
    ctx->frame.triangles.clear();   // keeps the capacity
    ctx->frame.viewports.clear();
    ctx->frame.shaders.clear();
    ctx->frame.attrs.clear();
    ctx->table_words = 0;
    ctx->open = true;
    ctx->ended = false;
    return 0;
}

namespace {

// Append triangles to the open frame. shader == 0: fixed function (vertex colours); otherwise `attrs` holds the varyings.
int append_triangles(
    ttr_context* ctx, const ttr_vertex* verts, uint32_t n_triangles, int shading, int cull, const ttr_state* state,
    uint32_t shader, const float* attrs) {
    if (!ctx->open) {
        g_error = "triangles outside a frame";
        return -1;
    }
    // Culling is per frame in the rasterizer: draws with different cull modes cannot share a frame, so the
    // caller must end the frame (or the driver must cull before submitting). Mixed modes are refused.
    const ttr::CullMode mode = cull == TTR_CULL_POSITIVE_AREA ? ttr::CullMode::Positive
                               : cull == TTR_CULL_NEGATIVE_AREA ? ttr::CullMode::Negative : ttr::CullMode::None;
    if (!ctx->frame.triangles.empty() && ctx->frame.cull != mode) {
        g_error = "all triangles of a frame must use the same cull mode";
        return -1;
    }
    uint32_t n_in = 0;
    if (shader != 0) {
        if (shader > ctx->frame.shaders.size()) {
            g_error = "unknown shader id";
            return -1;
        }
        n_in = ctx->frame.shaders[shader - 1].num_inputs;
        if (n_in != 0 && attrs == nullptr) {
            g_error = "shader triangles need their varyings";
            return -1;
        }
    }
    ctx->frame.cull = mode;
    const size_t base = ctx->frame.triangles.size();
    ctx->frame.triangles.resize(base + n_triangles);
    for (uint32_t t = 0; t < n_triangles; t++) {
        ttr::Triangle& tri = ctx->frame.triangles[base + t];
        tri.viewport = 0;
        tri.flat = shading == TTR_SHADE_FLAT && shader == 0;
        tri.perspective = shading == TTR_SHADE_PERSPECTIVE;
        if (state) {
            tri.own_state = true;
            tri.depth_func = state->depth_func == TTR_DEPTH_LESS      ? ttr::DepthFunc::Less
                             : state->depth_func == TTR_DEPTH_ALWAYS ? ttr::DepthFunc::Always
                                                                     : ttr::DepthFunc::LessEqual;
            tri.depth_write = state->depth_write != 0;
            tri.blend = static_cast<ttr::BlendMode>(state->blend >= 0 && state->blend <= TTR_BLEND_ALPHA_ADD ? state->blend : 0);
        }
        for (int k = 0; k < 3; k++) {
            const ttr_vertex& v = verts[3 * t + k];
            tri.v[k].x = v.x;
            tri.v[k].y = v.y;
            tri.v[k].z = v.z;
            tri.v[k].rhw = v.rhw;
            std::memcpy(tri.v[k].color, v.color, sizeof v.color);
        }
        if (shader != 0) {
            tri.shader = static_cast<uint16_t>(shader);
            tri.attr_offset = static_cast<uint32_t>(ctx->frame.attrs.size());
            ctx->frame.attrs.insert(ctx->frame.attrs.end(), attrs + static_cast<size_t>(t) * 3 * n_in, attrs + static_cast<size_t>(t + 1) * 3 * n_in);
        }
    }
    return 0;
}

bool same_program(const ttr::ShaderProgram& a, const ttr::ShaderProgram& b) {
    if (a.num_inputs != b.num_inputs || a.num_regs != b.num_regs || a.out_reg != b.out_reg || a.fragcoord_reg != b.fragcoord_reg ||
        a.code.size() != b.code.size() || a.consts.size() != b.consts.size()) {
        return false;
    }
    for (size_t i = 0; i < a.code.size(); i++) {
        if (std::memcmp(a.code[i].w, b.code[i].w, sizeof a.code[i].w) != 0) return false;
    }
    return a.consts.empty() || std::memcmp(a.consts.data(), b.consts.data(), a.consts.size() * sizeof(float)) == 0;
}

}  // namespace

int ttr_frame_triangles(
    ttr_context* ctx, const ttr_vertex* verts, uint32_t n_triangles, int shading, int cull, const ttr_state* state) {
    return append_triangles(ctx, verts, n_triangles, shading, cull, state, 0, nullptr);
}

int ttr_frame_triangles_shaded(
    ttr_context* ctx, const ttr_vertex* verts, uint32_t n_triangles, uint32_t shader, const float* attrs, int perspective,
    int cull, const ttr_state* state) {
    return append_triangles(ctx, verts, n_triangles, perspective ? TTR_SHADE_PERSPECTIVE : TTR_SHADE_LINEAR, cull, state, shader, attrs);
}

uint32_t ttr_frame_shader(ttr_context* ctx, const ttr_shader_desc* desc) {
    try {
        if (!ctx->open) {
            g_error = "ttr_frame_shader outside a frame";
            return 0;
        }
        ttr::ShaderProgram p;
        p.code.resize(desc->n_instr);
        for (uint32_t i = 0; i < desc->n_instr; i++) {
            std::memcpy(p.code[i].w, desc->code + static_cast<size_t>(i) * TTR_INSTR_WORDS, sizeof p.code[i].w);
        }
        p.consts.assign(desc->consts, desc->consts + desc->n_consts);
        p.num_inputs = desc->num_inputs;
        p.num_regs = desc->num_regs;
        p.out_reg = desc->out_reg;
        p.fragcoord_reg = desc->fragcoord_reg;
        const std::string err = ttr::validate_shader(p);
        if (!err.empty()) {
            g_error = "invalid shader: " + err;
            return 0;
        }
        for (size_t k = 0; k < ctx->frame.shaders.size(); k++) {
            if (same_program(ctx->frame.shaders[k], p)) return static_cast<uint32_t>(k + 1);
        }
        // The device's program table (raster_layout.h): a header word, a directory entry per program, code and constants.
        const size_t words = ctx->table_words == 0 ? ttr::layout::kTabDirectory : ctx->table_words;
        const size_t add = ttr::layout::kTabEntryWords + p.code.size() * TTR_INSTR_WORDS + p.consts.size();
        if (words + add > static_cast<size_t>(ttr::layout::kMaxTablePages) * ttr::layout::kTileWords || ctx->frame.shaders.size() >= 65535) {
            g_error = "the shader programs of the frame do not fit the program table";
            return 0;
        }
        ctx->table_words = words + add;
        ctx->frame.shaders.push_back(std::move(p));
        return static_cast<uint32_t>(ctx->frame.shaders.size());
    } catch (const std::exception& e) {
        g_error = e.what();
        return 0;
    }
}

namespace {

// Copy rows of a rendered plane into the caller's buffer on the worker pool. `color` fixes up the packed RGBA8 words
// (R in the low byte) for the destination format on the way; the depth plane is copied as it is.
void copy_planes(
    const ttr::Framebuffer& fb, uint32_t width, uint32_t height, uint8_t* rgba, uint32_t pitch, uint32_t color_flags, float* depth,
    int flip_y) {
    constexpr uint32_t kRowsPerItem = 32;
    const size_t items = (height + kRowsPerItem - 1) / kRowsPerItem;
    ttr::parallel_for(items, 0, [&](size_t item) {
        const uint32_t y0 = static_cast<uint32_t>(item) * kRowsPerItem, y1 = std::min(height, y0 + kRowsPerItem);
        for (uint32_t y = y0; y < y1; y++) {
            const uint32_t src_y = flip_y ? height - 1 - y : y;
            if (rgba) {
                const uint32_t* src = &fb.color[fb.index(0, src_y)];
                uint8_t* dst = rgba + static_cast<size_t>(y) * pitch;
                if (color_flags == 0) {
                    std::memcpy(dst, src, static_cast<size_t>(width) * 4);
                } else {
                    const uint32_t opaque = (color_flags & TTR_OUT_OPAQUE) ? 0xff000000u : 0u;
                    uint32_t* out = reinterpret_cast<uint32_t*>(dst);
                    if (color_flags & TTR_OUT_SWAP_RB) {
                        for (uint32_t x = 0; x < width; x++) {
                            const uint32_t c = src[x];
                            out[x] = ((c & 0xffu) << 16) | (c & 0xff00u) | ((c >> 16) & 0xffu) | (c & 0xff000000u) | opaque;
                        }
                    } else {
                        for (uint32_t x = 0; x < width; x++) out[x] = src[x] | opaque;
                    }
                }
            }
            if (depth) {
                std::memcpy(depth + static_cast<size_t>(y) * width, &fb.depth[fb.index(0, src_y)], static_cast<size_t>(width) * 4);
            }
        }
    });
}

}  // namespace

int ttr_frame_end_ex(ttr_context* ctx, uint8_t* rgba, uint32_t stride_bytes, uint32_t color_flags, float* depth, int flip_y) {
    try {
        if (!ctx->open) {
            g_error = "ttr_frame_end outside a frame";
            return -1;
        }
        ctx->open = false;
        const ttr::Frame& f = ctx->frame;
        const uint32_t planes = (rgba || !depth ? ttr::kRenderColor : 0u) | (depth ? ttr::kRenderDepth : 0u);
        const auto t0 = std::chrono::steady_clock::now();
        const ttr::Framebuffer& fb = ctx->backend->render(f, planes);
        const auto t1 = std::chrono::steady_clock::now();
        ctx->report = ctx->backend->report();
        ctx->ended = true;
        copy_planes(fb, f.width, f.height, rgba, stride_bytes != 0 ? stride_bytes : f.width * 4, color_flags, depth, flip_y);
        const auto t2 = std::chrono::steady_clock::now();
        char buf[96];
        std::snprintf(buf, sizeof buf, " | render %.1f copy-out %.1f ms", std::chrono::duration<double, std::milli>(t1 - t0).count(),
                      std::chrono::duration<double, std::milli>(t2 - t1).count());
        ctx->report += buf;
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int ttr_frame_end(ttr_context* ctx, uint8_t* rgba, uint32_t stride_bytes, float* depth, int flip_y) {
    return ttr_frame_end_ex(ctx, rgba, stride_bytes, 0, depth, flip_y);
}

int ttr_frame_depth(ttr_context* ctx, float* depth, int flip_y) {
    try {
        if (ctx->open || !ctx->ended || !depth) {
            g_error = "ttr_frame_depth needs a rendered frame and a buffer";
            return -1;
        }
        const ttr::Frame& f = ctx->frame;
        // Frames with fragment shaders always have the colour plane on the device.
        const ttr::Framebuffer& fb = ctx->backend->render(f, ttr::kRenderDepth | (f.shaders.empty() ? 0u : ttr::kRenderColor));
        copy_planes(fb, f.width, f.height, nullptr, 0, 0, depth, flip_y);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

}  // extern "C"
