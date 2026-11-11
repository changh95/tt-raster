// SPDX-License-Identifier: Apache-2.0

#include "scene/scene.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace ttr::scene {

using Clock = std::chrono::steady_clock;
namespace {

constexpr float kPi = 3.14159265358979323846f;

struct V3 {
    double x = 0, y = 0, z = 0;
};
V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
V3 normalize(V3 a) {
    const double l = std::sqrt(dot(a, a));
    return l > 0 ? a * (1.0 / l) : V3{0, 0, 1};
}

// Rotate v by the unit quaternion q (xyzw).
V3 rotate(const double q[4], V3 v) {
    const V3 u{q[0], q[1], q[2]};
    const V3 t = cross(u, v) * 2.0;
    return v + t * q[3] + cross(u, t);
}
void conjugate(const double q[4], double out[4]) {
    out[0] = -q[0]; out[1] = -q[1]; out[2] = -q[2]; out[3] = q[3];
}

void push_vertex(Mesh& m, V3 p, V3 n) {
    m.position.insert(m.position.end(), {static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z)});
    m.normal.insert(m.normal.end(), {static_cast<float>(n.x), static_cast<float>(n.y), static_cast<float>(n.z)});
}

// A latitude-longitude surface: `rings` + 1 rows of `segments` + 1 vertices, outward counter-clockwise.
// pos(theta, phi) and normal come from the callbacks.
template <typename PosFn, typename NrmFn>
Mesh make_lat_long(int rings, int segments, PosFn pos, NrmFn nrm) {
    Mesh m;
    for (int r = 0; r <= rings; r++) {
        for (int s = 0; s <= segments; s++) {
            const double theta = static_cast<double>(r) / rings;      // 0 at +z pole, 1 at -z pole
            const double phi = 2.0 * kPi * static_cast<double>(s) / segments;
            push_vertex(m, pos(theta, phi), nrm(theta, phi));
        }
    }
    for (int r = 0; r < rings; r++) {
        for (int s = 0; s < segments; s++) {
            const uint32_t a = static_cast<uint32_t>(r * (segments + 1) + s), b = a + 1;
            const uint32_t c = a + static_cast<uint32_t>(segments + 1), d = c + 1;
            // Rows run from +z to -z and phi increases counter-clockwise seen from +z, so (a, c, b) faces outward.
            m.index.insert(m.index.end(), {a, c, b, b, c, d});
        }
    }
    return m;
}

}  // namespace

float srgb_to_linear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }
float linear_to_srgb(float l) {
    l = std::min(1.0f, std::max(0.0f, l));
    return l <= 0.0031308f ? l * 12.92f : 1.055f * std::pow(l, 1.0f / 2.4f) - 0.055f;
}

Mesh make_box(float hx, float hy, float hz) {
    Mesh m;
    // Six faces, four vertices each, with the face normal.
    struct Face { V3 n, u, v; };  // u x v = n
    const Face faces[6] = {
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
        {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}, {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
        {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}}, {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}}};
    const V3 h{hx, hy, hz};
    for (const Face& f : faces) {
        const uint32_t base = static_cast<uint32_t>(m.position.size() / 3);
        const double su[4] = {-1, 1, 1, -1}, sv[4] = {-1, -1, 1, 1};
        for (int i = 0; i < 4; i++) {
            const V3 p{f.n.x * h.x + f.u.x * su[i] * h.x + f.v.x * sv[i] * h.x,
                       f.n.y * h.y + f.u.y * su[i] * h.y + f.v.y * sv[i] * h.y,
                       f.n.z * h.z + f.u.z * su[i] * h.z + f.v.z * sv[i] * h.z};
            push_vertex(m, p, f.n);
        }
        m.index.insert(m.index.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    }
    return m;
}

Mesh make_sphere(float radius, int rings, int segments) {
    auto dir = [](double theta, double phi) {
        const double a = theta * kPi;
        return V3{std::sin(a) * std::cos(phi), std::sin(a) * std::sin(phi), std::cos(a)};
    };
    return make_lat_long(
        rings, segments, [&](double t, double p) { return dir(t, p) * radius; }, [&](double t, double p) { return dir(t, p); });
}

Mesh make_capsule(float radius, float half_height, int rings, int segments) {
    // `rings` rows per hemisphere; the cylinder is the seam between the two halves.
    const int total = 2 * rings;
    auto pos = [&](double t, double p) {
        const double a = t * kPi;  // 0 .. pi over the whole capsule, remapped per hemisphere below
        const bool top = t < 0.5;
        const double ha = top ? t * 2.0 * kPi / 2.0 : (t - 0.5) * 2.0 * kPi / 2.0 + kPi / 2.0;
        (void)a;
        const V3 d{std::sin(ha) * std::cos(p), std::sin(ha) * std::sin(p), std::cos(ha)};
        return d * radius + V3{0, 0, top ? half_height : -half_height};
    };
    auto nrm = [&](double t, double p) {
        const bool top = t < 0.5;
        const double ha = top ? t * kPi : (t - 0.5) * kPi + kPi / 2.0;
        return V3{std::sin(ha) * std::cos(p), std::sin(ha) * std::sin(p), std::cos(ha)};
    };
    return make_lat_long(total + 1, segments, pos, nrm);
}

Mesh make_ground(float e) {
    Mesh m;
    const V3 n{0, 0, 1};
    push_vertex(m, {-e, -e, 0}, n);
    push_vertex(m, {e, -e, 0}, n);
    push_vertex(m, {e, e, 0}, n);
    push_vertex(m, {-e, e, 0}, n);
    m.index = {0, 1, 2, 0, 2, 3};
    return m;
}

const Framebuffer& ReferenceBackend::render(const Frame& frame, bool float_planes) {
    ReferenceOptions opt;
    opt.float_planes = float_planes;
    fb_ = render_reference(frame, opt);
    return fb_;
}

uint32_t Scene::add_mesh(Mesh m) {
    meshes_.push_back(std::move(m));
    return static_cast<uint32_t>(meshes_.size() - 1);
}

uint32_t Scene::add_instance(const Instance& i) {
    if (i.mesh >= meshes_.size() || i.world >= worlds_) {
        throw std::invalid_argument("instance refers to an unknown mesh or world");
    }
    instances_.push_back(i);
    return static_cast<uint32_t>(instances_.size() - 1);
}

void Scene::set_pose(uint32_t instance, const float position[3], const float rotation[4]) {
    Instance& i = instances_.at(instance);
    std::memcpy(i.position, position, sizeof i.position);
    std::memcpy(i.rotation, rotation, sizeof i.rotation);
}

namespace {

// A vertex in camera space with its payloads, for clipping.
struct CV {
    V3 pc;           // camera space position (the camera looks down -z)
    float lit[3];    // shaded colour (sRGB)
    float n[3];      // world normal
};

CV lerp(const CV& a, const CV& b, double t) {
    CV r;
    r.pc = a.pc + (b.pc - a.pc) * t;
    for (int i = 0; i < 3; i++) {
        r.lit[i] = static_cast<float>(a.lit[i] + (b.lit[i] - a.lit[i]) * t);
        r.n[i] = static_cast<float>(a.n[i] + (b.n[i] - a.n[i]) * t);
    }
    return r;
}

// Clip a convex polygon against near <= -z <= far. Intersections depend only on the edge's two end
// points, taken in a fixed order, so neighbouring triangles agree on shared cut points.
std::vector<CV> clip_depth(std::vector<CV> poly, double near_d, double far_d) {
    for (int pass = 0; pass < 2 && !poly.empty(); pass++) {
        const double bound = pass == 0 ? near_d : far_d;
        auto depth = [](const CV& v) { return -v.pc.z; };
        auto inside = [&](const CV& v) { return pass == 0 ? depth(v) >= bound : depth(v) <= bound; };
        std::vector<CV> out;
        for (size_t i = 0; i < poly.size(); i++) {
            const CV &a = poly[i], &b = poly[(i + 1) % poly.size()];
            const bool ia = inside(a), ib = inside(b);
            auto cut = [&]() {
                // Order the end points by (x, y, z) so that a shared edge is cut identically from either side.
                const bool swap = std::tie(a.pc.x, a.pc.y, a.pc.z) > std::tie(b.pc.x, b.pc.y, b.pc.z);
                const CV &p = swap ? b : a, &q = swap ? a : b;
                const double t = (bound - depth(p)) / (depth(q) - depth(p));
                return lerp(p, q, t);
            };
            if (ia && ib) out.push_back(b);
            else if (ia && !ib) out.push_back(cut());
            else if (!ia && ib) { out.push_back(cut()); out.push_back(b); }
        }
        poly = std::move(out);
    }
    return poly;
}

// One triangle ready for rasterization, in screen space of its camera slot.
struct ScreenTri {
    float x[3], y[3], z[3];
    float albedo[3];      // constant per triangle (instance)
    float lit[3][3];      // per vertex
    float normal[3][3];   // per vertex
    uint32_t shape;
};

}  // namespace

void Scene::render(
    const Camera& cam, const std::vector<Pose>& poses, const Light& light, const Images& out, Backend& backend,
    RenderStats* stats, size_t threads) const {
    if (poses.size() != worlds_) {
        throw std::invalid_argument("one camera pose per world is required");
    }
    const uint32_t mask = out.mask();
    RenderStats local;
    const auto t0 = Clock::now();

    // ---- Atlas layout: one slot of whole tiles per world -----------------------------------------
    const uint32_t slot_w = (cam.width + kTileSize - 1) / kTileSize * kTileSize;
    const uint32_t slot_h = (cam.height + kTileSize - 1) / kTileSize * kTileSize;
    const uint32_t cols = std::max<uint32_t>(1, std::min<uint32_t>(worlds_, 4096 / slot_w));
    const uint32_t rows = (worlds_ + cols - 1) / cols;
    Frame base;
    base.width = cols * slot_w;
    base.height = rows * slot_h;
    base.clear_color[0] = base.clear_color[1] = base.clear_color[2] = 0.0f;
    base.clear_color[3] = 1.0f;
    for (uint32_t w = 0; w < worlds_; w++) {
        base.viewports.push_back(
            {static_cast<int32_t>((w % cols) * slot_w), static_cast<int32_t>((w / cols) * slot_h),
             static_cast<int32_t>(cam.width), static_cast<int32_t>(cam.height)});
    }

    // ---- Vertex stage, per world in parallel -----------------------------------------------------
    std::vector<std::vector<uint32_t>> by_world(worlds_);
    for (uint32_t i = 0; i < instances_.size(); i++) {
        by_world[instances_[i].world].push_back(i);
    }
    const double near_d = cam.near_clip, far_d = cam.far_clip;
    const double fx = cam.focal_length / cam.horizontal_aperture * cam.width;   // pixels per unit of x / depth
    const double fy = cam.focal_length / cam.vertical_aperture * cam.height;
    const V3 L = normalize(V3{-light.direction[0], -light.direction[1], -light.direction[2]});

    std::vector<std::vector<ScreenTri>> per_world(worlds_);
    auto do_world = [&](uint32_t w) {
        const Pose& pose = poses[w];
        const double cq[4] = {pose.rotation[0], pose.rotation[1], pose.rotation[2], pose.rotation[3]};
        double cqi[4];
        conjugate(cq, cqi);
        const V3 cpos{pose.position[0], pose.position[1], pose.position[2]};
        const Viewport& vp = base.viewports[w];
        std::vector<ScreenTri>& tris = per_world[w];
        for (uint32_t ii : by_world[w]) {
            const Instance& inst = instances_[ii];
            const Mesh& mesh = meshes_[inst.mesh];
            const double iq[4] = {inst.rotation[0], inst.rotation[1], inst.rotation[2], inst.rotation[3]};
            const V3 ipos{inst.position[0], inst.position[1], inst.position[2]};
            float alb_lin[3];
            for (int c = 0; c < 3; c++) alb_lin[c] = srgb_to_linear(inst.albedo[c]);

            const size_t nv = mesh.position.size() / 3;
            std::vector<CV> cv(nv);
            for (size_t v = 0; v < nv; v++) {
                const V3 pw = rotate(iq, V3{mesh.position[3 * v], mesh.position[3 * v + 1], mesh.position[3 * v + 2]}) + ipos;
                const V3 nw = normalize(rotate(iq, V3{mesh.normal[3 * v], mesh.normal[3 * v + 1], mesh.normal[3 * v + 2]}));
                cv[v].pc = rotate(cqi, pw - cpos);
                cv[v].n[0] = static_cast<float>(nw.x); cv[v].n[1] = static_cast<float>(nw.y); cv[v].n[2] = static_cast<float>(nw.z);
                // Per-vertex shading (reference: Newton Warp): albedo * (ambient + n.L), linear, then sRGB.
                const double h = 0.5 * (nw.z + 1.0);
                const double amb[3] = {(0.4 * h + 0.1 * (1 - h)) * 0.5, (0.4 * h + 0.1 * (1 - h)) * 0.5, (0.45 * h + 0.12 * (1 - h)) * 0.5};
                const double ndl = std::max(0.0, dot(nw, L));
                for (int c = 0; c < 3; c++) cv[v].lit[c] = linear_to_srgb(static_cast<float>(alb_lin[c] * (amb[c] + ndl)));
            }
            for (size_t t = 0; t + 2 < mesh.index.size(); t += 3) {
                std::vector<CV> poly = {cv[mesh.index[t]], cv[mesh.index[t + 1]], cv[mesh.index[t + 2]]};
                bool all_in = true;
                for (const CV& v : poly) all_in &= -v.pc.z >= near_d && -v.pc.z <= far_d;
                if (!all_in) poly = clip_depth(std::move(poly), near_d, far_d);
                if (poly.size() < 3) continue;
                // Project, then fan.
                struct P { float x, y, z; const CV* v; };
                std::vector<P> pr(poly.size());
                for (size_t k = 0; k < poly.size(); k++) {
                    const double d = -poly[k].pc.z;
                    pr[k].x = static_cast<float>(vp.x + vp.width * 0.5 + poly[k].pc.x / d * fx);
                    pr[k].y = static_cast<float>(vp.y + vp.height * 0.5 - poly[k].pc.y / d * fy);
                    pr[k].z = static_cast<float>(far_d * (d - near_d) / (d * (far_d - near_d)));
                    pr[k].v = &poly[k];
                }
                for (size_t k = 1; k + 1 < pr.size(); k++) {
                    const P* tri[3] = {&pr[0], &pr[k], &pr[k + 1]};
                    ScreenTri st;
                    for (int c = 0; c < 3; c++) {
                        st.x[c] = tri[c]->x; st.y[c] = tri[c]->y; st.z[c] = tri[c]->z;
                        for (int a = 0; a < 3; a++) {
                            st.lit[c][a] = tri[c]->v->lit[a];
                            st.normal[c][a] = tri[c]->v->n[a];
                        }
                    }
                    for (int a = 0; a < 3; a++) st.albedo[a] = inst.albedo[a];
                    st.shape = inst.shape_id;
                    tris.push_back(st);
                }
            }
        }
    };
    {
        const size_t hw = threads != 0 ? threads : std::max<size_t>(1, std::thread::hardware_concurrency());
        std::atomic<uint32_t> next{0};
        std::vector<std::thread> pool;
        for (size_t t = 0; t < std::min<size_t>(hw, worlds_); t++) {
            pool.emplace_back([&] { for (uint32_t w = next++; w < worlds_; w = next++) do_world(w); });
        }
        for (auto& th : pool) th.join();
    }
    local.vertex_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    // ---- Passes ------------------------------------------------------------------------------------
    // Every pass draws the same triangles with a different colour payload; depth and triangle id come
    // with each. Pass 1 is the first one requested.
    enum Payload { kPAlbedo, kPLit, kPNormal };
    std::vector<uint32_t> shape_of;   // per triangle of the frame: shape id (same for all passes)
    auto build = [&](Payload payload) {
        Frame f = base;
        f.triangles.clear();
        const bool fill_shape = shape_of.empty();
        for (uint32_t w = 0; w < worlds_; w++) {
            for (const ScreenTri& st : per_world[w]) {
                Triangle t;
                t.viewport = w;
                t.flat = payload == kPAlbedo;
                for (int c = 0; c < 3; c++) {
                    Vertex& v = t.v[c];
                    v.x = st.x[c]; v.y = st.y[c]; v.z = st.z[c]; v.rhw = 1.0f;
                    for (int a = 0; a < 3; a++) {
                        v.color[a] = payload == kPAlbedo ? st.albedo[a] : payload == kPLit ? st.lit[c][a] : st.normal[c][a];
                    }
                    v.color[3] = 1.0f;
                }
                f.triangles.push_back(t);
                if (fill_shape) shape_of.push_back(st.shape);
            }
        }
        return f;
    };

    auto timed_render = [&](const Frame& f, bool float_planes) -> const Framebuffer& {
        const auto r0 = Clock::now();
        const Framebuffer& fb = backend.render(f, float_planes);
        local.raster_ms += std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
        local.passes++;
        local.triangles = f.triangles.size();
        return fb;
    };

    const double inv_fx = 1.0 / fx, inv_fy = 1.0 / fy;
    bool have_geometry = false;
    auto slot_index = [&](const Framebuffer& fb, uint32_t w, uint32_t x, uint32_t y) {
        const Viewport& vp = base.viewports[w];
        return fb.index(static_cast<uint32_t>(vp.x) + x, static_cast<uint32_t>(vp.y) + y);
    };
    const size_t img = static_cast<size_t>(cam.width) * cam.height;
    auto geometry_from = [&](const Framebuffer& fb) {
        if (have_geometry || !(mask & (kDepth | kForwardDepth | kShape))) return;
        have_geometry = true;
        const auto a0 = Clock::now();
        for (uint32_t w = 0; w < worlds_; w++) {
            for (uint32_t y = 0; y < cam.height; y++) {
                for (uint32_t x = 0; x < cam.width; x++) {
                    const size_t i = slot_index(fb, w, x, y), o = w * img + static_cast<size_t>(y) * cam.width + x;
                    const uint32_t id = fb.id[i];
                    if (id == 0) {
                        if (out.depth) out.depth[o] = 0.0f;
                        if (out.forward_depth) out.forward_depth[o] = 0.0f;
                        if (out.shape) out.shape[o] = 0xFFFFFFFFu;
                        continue;
                    }
                    const double z = fb.depth[i];
                    const double d = far_d * near_d / (far_d - z * (far_d - near_d));   // distance along the axis
                    if (out.forward_depth) out.forward_depth[o] = static_cast<float>(d);
                    if (out.depth) {
                        const double xn = (x + 0.5 - cam.width * 0.5) * inv_fx, yn = (cam.height * 0.5 - y - 0.5) * inv_fy;
                        out.depth[o] = static_cast<float>(d * std::sqrt(1.0 + xn * xn + yn * yn));
                    }
                    if (out.shape) out.shape[o] = shape_of[id - 1];
                }
            }
        }
        local.assemble_ms += std::chrono::duration<double, std::milli>(Clock::now() - a0).count();
    };
    auto colour_from = [&](const Framebuffer& fb, uint8_t* dst) {
        const auto a0 = Clock::now();
        for (uint32_t w = 0; w < worlds_; w++)
            for (uint32_t y = 0; y < cam.height; y++)
                for (uint32_t x = 0; x < cam.width; x++) {
                    const uint32_t c = fb.color[slot_index(fb, w, x, y)];
                    uint8_t* p = dst + (w * img + static_cast<size_t>(y) * cam.width + x) * 4;
                    p[0] = static_cast<uint8_t>(c); p[1] = static_cast<uint8_t>(c >> 8);
                    p[2] = static_cast<uint8_t>(c >> 16); p[3] = static_cast<uint8_t>(c >> 24);
                }
        local.assemble_ms += std::chrono::duration<double, std::milli>(Clock::now() - a0).count();
    };

    if (mask & kAlbedo) {
        const Frame f = build(kPAlbedo);
        const Framebuffer& fb = timed_render(f, false);
        colour_from(fb, out.albedo);
        geometry_from(fb);
    }
    if (mask & kRgb) {
        const Frame f = build(kPLit);
        const Framebuffer& fb = timed_render(f, false);
        colour_from(fb, out.rgb);
        geometry_from(fb);
    }
    if (mask & kNormal) {
        const Frame f = build(kPNormal);
        const Framebuffer& fb = timed_render(f, true);
        const auto a0 = Clock::now();
        for (uint32_t w = 0; w < worlds_; w++)
            for (uint32_t y = 0; y < cam.height; y++)
                for (uint32_t x = 0; x < cam.width; x++) {
                    const size_t i = slot_index(fb, w, x, y), o = (w * img + static_cast<size_t>(y) * cam.width + x) * 3;
                    if (fb.id[i] == 0) { out.normal[o] = out.normal[o + 1] = out.normal[o + 2] = 0.0f; continue; }
                    const V3 n = normalize(V3{fb.plane[0][i], fb.plane[1][i], fb.plane[2][i]});
                    out.normal[o] = static_cast<float>(n.x); out.normal[o + 1] = static_cast<float>(n.y); out.normal[o + 2] = static_cast<float>(n.z);
                }
        local.assemble_ms += std::chrono::duration<double, std::milli>(Clock::now() - a0).count();
        geometry_from(fb);
    }
    if (!have_geometry && (mask & (kDepth | kForwardDepth | kShape))) {
        const Frame f = build(kPAlbedo);
        geometry_from(timed_render(f, false));
    }
    if (stats) *stats = local;
}

}  // namespace ttr::scene
