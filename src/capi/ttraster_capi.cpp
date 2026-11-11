// SPDX-License-Identifier: Apache-2.0

#include "tensix/kernels/raster_layout.h"
#include "ttraster/ttraster.h"

#include <cstring>
#include <future>
#include <memory>
#include <string>

#include "scene/scene.h"
#include "ttraster/frame.h"
#ifdef TT_RASTER_WITH_TENSIX
#include "tensix/scene_backend.h"
#endif

namespace {
thread_local std::string g_error;
}

struct ttr_scene {
    ttr::scene::Scene scene;
    std::unique_ptr<ttr::scene::Backend> backend;
    std::string report;
    // Pipelined frames: ticket t runs in slot t % kRenderSlots until it is collected.
    struct Job {
        std::future<ttr::scene::RenderStats> result;
        int ticket = -1;
        bool busy = false;
    };
    Job jobs[ttr::kRenderSlots];
    int next_ticket = 0;
    explicit ttr_scene(uint32_t worlds) : scene(worlds) {}
    bool in_flight() const {
        for (const Job& j : jobs) {
            if (j.busy) return true;
        }
        return false;
    }
    ~ttr_scene() {
        for (Job& j : jobs) {
            if (j.busy) j.result.wait();   // never destroy the scene under a running frame
        }
    }
};

namespace {
void fill_stats(const ttr::scene::RenderStats& st, ttr_stats* stats) {
    if (!stats) return;
    stats->vertex_ms = st.vertex_ms;
    stats->raster_ms = st.raster_ms;
    stats->assemble_ms = st.assemble_ms;
    stats->build_ms = st.build_ms;
    stats->total_ms = st.total_ms;
    stats->triangles = st.triangles;
    stats->passes = st.passes;
}

// The arguments of a render call, copied out of the caller's structs.
struct RenderArgs {
    ttr::scene::Camera cam;
    std::vector<ttr::scene::Pose> poses;
    ttr::scene::Light light;
    ttr::scene::Images img;
};

RenderArgs make_args(ttr_scene* s, const ttr_camera* camera, const float* camera_poses, const float* light_direction, const ttr_images* out) {
    RenderArgs a;
    a.cam.width = camera->width;
    a.cam.height = camera->height;
    a.cam.focal_length = camera->focal_length;
    a.cam.horizontal_aperture = camera->horizontal_aperture;
    a.cam.vertical_aperture = camera->vertical_aperture;
    a.cam.near_clip = camera->near_clip;
    a.cam.far_clip = camera->far_clip;
    a.cam.background = camera->background;
    a.poses.resize(s->scene.worlds());
    for (size_t w = 0; w < a.poses.size(); w++) {
        std::memcpy(a.poses[w].position, camera_poses + 7 * w, 3 * sizeof(float));
        std::memcpy(a.poses[w].rotation, camera_poses + 7 * w + 3, 4 * sizeof(float));
    }
    if (light_direction) std::memcpy(a.light.direction, light_direction, sizeof a.light.direction);
    a.img.depth = out->depth;
    a.img.forward_depth = out->forward_depth;
    a.img.shape = out->shape;
    a.img.albedo = out->albedo;
    a.img.rgb = out->rgb;
    a.img.normal = out->normal;
    return a;
}
}  // namespace

extern "C" {

const char* ttr_last_error(void) { return g_error.c_str(); }
const char* ttr_backend_report(ttr_scene* s) { return s->report.c_str(); }

ttr_scene* ttr_scene_create(uint32_t num_worlds, const char* backend, uint32_t n_chips) {
    try {
        auto s = std::make_unique<ttr_scene>(num_worlds);
        const std::string b = backend ? backend : "reference";
        if (b == "reference") {
            s->backend = std::make_unique<ttr::scene::ReferenceBackend>();
        } else if (b == "tensix") {
#ifdef TT_RASTER_WITH_TENSIX
            s->backend = ttr::make_tensix_backend(n_chips == 0 ? 1 : n_chips);
#else
            g_error = "this libttraster was built without Tensix support";
            return nullptr;
#endif
        } else {
            g_error = "unknown backend '" + b + "'";
            return nullptr;
        }
        (void)n_chips;
        return s.release();
    } catch (const std::exception& e) {
        g_error = e.what();
        return nullptr;
    }
}

void ttr_scene_destroy(ttr_scene* scene) { delete scene; }

#define TTR_TRY(expr)                 \
    try {                             \
        return (expr);                \
    } catch (const std::exception& e) { \
        g_error = e.what();           \
        return -1;                    \
    }

int32_t ttr_add_box(ttr_scene* s, float hx, float hy, float hz) { TTR_TRY(static_cast<int32_t>(s->scene.add_mesh(ttr::scene::make_box(hx, hy, hz)))) }
int32_t ttr_add_sphere(ttr_scene* s, float r, int rings, int segs) { TTR_TRY(static_cast<int32_t>(s->scene.add_mesh(ttr::scene::make_sphere(r, rings, segs)))) }
int32_t ttr_add_capsule(ttr_scene* s, float r, float hh, int rings, int segs) { TTR_TRY(static_cast<int32_t>(s->scene.add_mesh(ttr::scene::make_capsule(r, hh, rings, segs)))) }
int32_t ttr_add_ground(ttr_scene* s, float e) { TTR_TRY(static_cast<int32_t>(s->scene.add_mesh(ttr::scene::make_ground(e)))) }

int32_t ttr_add_mesh(
    ttr_scene* s, const float* positions, const float* normals, uint32_t nv, const uint32_t* indices, uint32_t nt) {
    try {
        ttr::scene::Mesh m;
        m.position.assign(positions, positions + 3ull * nv);
        m.normal.assign(normals, normals + 3ull * nv);
        m.index.assign(indices, indices + 3ull * nt);
        for (uint32_t i : m.index) {
            if (i >= nv) {
                g_error = "mesh index out of range";
                return -1;
            }
        }
        return static_cast<int32_t>(s->scene.add_mesh(std::move(m)));
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int32_t ttr_add_instance(
    ttr_scene* s, uint32_t mesh, uint32_t world, uint32_t shape_id, const float albedo[3], const float pose[7]) {
    try {
        ttr::scene::Instance i;
        i.mesh = mesh;
        i.world = world;
        i.shape_id = shape_id;
        std::memcpy(i.albedo, albedo, sizeof i.albedo);
        std::memcpy(i.position, pose, 3 * sizeof(float));
        std::memcpy(i.rotation, pose + 3, 4 * sizeof(float));
        return static_cast<int32_t>(s->scene.add_instance(i));
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int ttr_set_instance_poses(ttr_scene* s, const float* poses) {
    for (size_t i = 0; i < s->scene.instances(); i++) {
        s->scene.set_pose(static_cast<uint32_t>(i), poses + 7 * i, poses + 7 * i + 3);
    }
    return 0;
}

int ttr_render(
    ttr_scene* s, const ttr_camera* camera, const float* camera_poses, const float* light_direction, const ttr_images* out,
    ttr_stats* stats) {
    try {
        if (s->in_flight()) {
            g_error = "frames are in flight: ttr_collect() them before a synchronous ttr_render()";
            return -1;
        }
        const RenderArgs a = make_args(s, camera, camera_poses, light_direction, out);
        ttr::scene::RenderStats st;
        s->scene.render(a.cam, a.poses, a.light, a.img, *s->backend, &st);
        s->report = st.backend_report;
        fill_stats(st, stats);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int ttr_submit(
    ttr_scene* s, const ttr_camera* camera, const float* camera_poses, const float* light_direction, const ttr_images* out) {
    try {
        const int ticket = s->next_ticket;
        ttr_scene::Job& job = s->jobs[ticket % ttr::kRenderSlots];
        if (job.busy) {
            g_error = "no free slot: ttr_collect() ticket " + std::to_string(job.ticket) + " first";
            return -1;
        }
        auto args = std::make_shared<RenderArgs>(make_args(s, camera, camera_poses, light_direction, out));
        auto snap = std::make_shared<ttr::scene::Scene::Snapshot>(s->scene.snapshot());   // poses as of now
        const int slot = ticket % ttr::kRenderSlots;
        job.result = std::async(std::launch::async, [s, args, snap, slot]() {
            ttr::scene::RenderStats st;
            s->scene.render_snapshot(*snap, args->cam, args->poses, args->light, args->img, *s->backend, slot, &st);
            return st;
        });
        job.ticket = ticket;
        job.busy = true;
        s->next_ticket++;
        return ticket;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int ttr_collect(ttr_scene* s, int ticket, ttr_stats* stats) {
    ttr_scene::Job& job = s->jobs[((ticket % ttr::kRenderSlots) + ttr::kRenderSlots) % ttr::kRenderSlots];
    if (!job.busy || job.ticket != ticket) {
        g_error = "ttr_collect: ticket " + std::to_string(ticket) + " is not in flight";
        return -1;
    }
    job.busy = false;   // the slot is free again whatever happens below
    try {
        const ttr::scene::RenderStats st = job.result.get();
        s->report = st.backend_report;
        fill_stats(st, stats);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

// ---- Raster API ----------------------------------------------------------------------------------------

struct ttr_context {
    std::unique_ptr<ttr::scene::Backend> backend;
    ttr::Frame frame;
    bool open = false;
    std::string report;
    size_t table_words = 0;   // size of the device's program table for the frame's shaders (raster_layout.h)
};

ttr_context* ttr_context_create(const char* backend, uint32_t n_chips) {
    try {
        auto c = std::make_unique<ttr_context>();
        const std::string b = backend ? backend : "reference";
        if (b == "reference") {
            c->backend = std::make_unique<ttr::scene::ReferenceBackend>();
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

int ttr_frame_end(ttr_context* ctx, uint8_t* rgba, uint32_t stride_bytes, float* depth, int flip_y) {
    try {
        if (!ctx->open) {
            g_error = "ttr_frame_end outside a frame";
            return -1;
        }
        ctx->open = false;
        const ttr::Frame& f = ctx->frame;
        const ttr::Framebuffer& fb = ctx->backend->render(
            f, ttr::scene::kPlaneColor | (depth ? ttr::scene::kPlaneDepth : 0u));
        ctx->report = ctx->backend->report();
        const uint32_t pitch = stride_bytes != 0 ? stride_bytes : f.width * 4;
        for (uint32_t y = 0; y < f.height; y++) {
            const uint32_t src_y = flip_y ? f.height - 1 - y : y;
            if (rgba) {
                std::memcpy(rgba + static_cast<size_t>(y) * pitch, &fb.color[fb.index(0, src_y)], static_cast<size_t>(f.width) * 4);
            }
            if (depth) {
                std::memcpy(depth + static_cast<size_t>(y) * f.width, &fb.depth[fb.index(0, src_y)], static_cast<size_t>(f.width) * 4);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

}  // extern "C"
