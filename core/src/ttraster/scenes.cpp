// SPDX-License-Identifier: Apache-2.0

#include "ttraster/scenes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ttr {

namespace {

// splitmix64: small, fast, identical everywhere.
class Rng {
public:
    explicit Rng(uint64_t seed) : s_(seed) {}
    uint64_t next() {
        s_ += 0x9E3779B97F4A7C15ull;
        uint64_t z = s_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uniform() { return static_cast<float>(next() >> 40) * (1.0f / 16777216.0f); }  // [0, 1)
    float range(float a, float b) { return a + (b - a) * uniform(); }

private:
    uint64_t s_;
};

Vertex vtx(float x, float y, float z, float r, float g, float b) {
    Vertex v{};
    v.x = x;
    v.y = y;
    v.z = z;
    v.rhw = 1.0f;
    v.color[0] = r;
    v.color[1] = g;
    v.color[2] = b;
    v.color[3] = 1.0f;
    return v;
}

Triangle tri(const Vertex& a, const Vertex& b, const Vertex& c, bool flat) {
    Triangle t;
    t.v[0] = a;
    t.v[1] = b;
    t.v[2] = c;
    t.flat = flat;
    return t;
}

// Draw order matters: C++ leaves the evaluation order of function arguments unspecified, so every
// random draw below is its own statement and the scenes come out identical under any compiler.
Vertex random_color_vertex(Rng& rng, float x, float y, float z) {
    const float r = rng.range(0.05f, 1.0f);
    const float g = rng.range(0.05f, 1.0f);
    const float b = rng.range(0.05f, 1.0f);
    return vtx(x, y, z, r, g, b);
}

void base_frame(Frame& f, uint32_t w, uint32_t h) {
    f = Frame{};
    f.width = w;
    f.height = h;
    f.clear_color[0] = 0.08f;
    f.clear_color[1] = 0.09f;
    f.clear_color[2] = 0.12f;
    f.clear_color[3] = 1.0f;
}

void scene_one_triangle(Frame& f, uint32_t w, uint32_t h) {
    const float W = static_cast<float>(w), H = static_cast<float>(h);
    f.triangles.push_back(tri(
        vtx(0.1563f * W, 0.1107f * H, 0.2f, 1.0f, 0.1f, 0.1f),
        vtx(0.7812f * W, 0.3693f * H, 0.5f, 0.1f, 1.0f, 0.1f),
        vtx(0.3647f * W, 0.9229f * H, 0.8f, 0.1f, 0.1f, 1.0f),
        false));
}

// A polygon fan around one centre, smooth shaded: every triangle shares edges with its two
// neighbours and with the centre.
void scene_fan(Frame& f, uint32_t w, uint32_t h) {
    const double W = w, H = h;
    const double cx = 0.5 * W + 0.37, cy = 0.5 * H - 0.21;
    constexpr int kN = 96;
    const double kPi = 3.14159265358979323846;
    std::vector<Vertex> rim;
    for (int k = 0; k < kN; k++) {
        const double a = 2.0 * kPi * k / kN;
        const double r = 1.0 + 0.18 * std::sin(5.0 * a);
        const float x = static_cast<float>(cx + 0.42 * W * r * std::cos(a));
        const float y = static_cast<float>(cy + 0.42 * H * r * std::sin(a));
        const float hue = static_cast<float>(k) / kN;
        const float rr = 0.5f + 0.5f * static_cast<float>(std::cos(2.0 * kPi * hue));
        const float gg = 0.5f + 0.5f * static_cast<float>(std::cos(2.0 * kPi * (hue - 1.0 / 3.0)));
        const float bb = 0.5f + 0.5f * static_cast<float>(std::cos(2.0 * kPi * (hue - 2.0 / 3.0)));
        rim.push_back(vtx(x, y, 0.7f, rr, gg, bb));
    }
    const Vertex centre = vtx(static_cast<float>(cx), static_cast<float>(cy), 0.3f, 1.0f, 1.0f, 1.0f);
    for (int k = 0; k < kN; k++) {
        f.triangles.push_back(tri(centre, rim[k], rim[(k + 1) % kN], false));
    }
}

// A jittered grid of shared vertices covering the whole viewport; every cell is two triangles
// with the diagonal alternating. Boundary vertices stay on the viewport edge, so the mesh covers
// exactly the viewport and every pixel must be hit once.
void scene_grid(Frame& f, uint32_t w, uint32_t h) {
    const int cells_x = static_cast<int>(w / 32 > 0 ? w / 32 : 1);
    const int cells_y = static_cast<int>(h / 32 > 0 ? h / 32 : 1);
    const float cw = static_cast<float>(w) / static_cast<float>(cells_x);
    const float ch = static_cast<float>(h) / static_cast<float>(cells_y);
    Rng rng(0x6772696Dull);
    std::vector<Vertex> grid(static_cast<size_t>(cells_x + 1) * (cells_y + 1));
    for (int j = 0; j <= cells_y; j++) {
        for (int i = 0; i <= cells_x; i++) {
            float x = static_cast<float>(i) * cw;
            float y = static_cast<float>(j) * ch;
            if (i == cells_x) x = static_cast<float>(w);
            if (j == cells_y) y = static_cast<float>(h);
            if (i > 0 && i < cells_x) x += rng.range(-0.3f, 0.3f) * cw;
            if (j > 0 && j < cells_y) y += rng.range(-0.3f, 0.3f) * ch;
            grid[static_cast<size_t>(j) * (cells_x + 1) + i] = vtx(x, y, 0.5f, 0.5f, 0.5f, 0.5f);
        }
    }
    auto at = [&](int i, int j) -> const Vertex& { return grid[static_cast<size_t>(j) * (cells_x + 1) + i]; };
    for (int j = 0; j < cells_y; j++) {
        for (int i = 0; i < cells_x; i++) {
            const Vertex &a = at(i, j), &b = at(i + 1, j), &c = at(i + 1, j + 1), &d = at(i, j + 1);
            Triangle t0, t1;
            if (((i + j) & 1) == 0) {
                t0 = tri(a, b, c, true);
                t1 = tri(a, c, d, true);
            } else {
                t0 = tri(a, b, d, true);
                t1 = tri(b, c, d, true);
            }
            for (Triangle* t : {&t0, &t1}) {
                const float r = rng.range(0.1f, 1.0f), g = rng.range(0.1f, 1.0f), bl = rng.range(0.1f, 1.0f);
                for (Vertex& v : t->v) {
                    v.color[0] = r;
                    v.color[1] = g;
                    v.color[2] = bl;
                }
                f.triangles.push_back(*t);
            }
        }
    }
}

// Large triangles with random depths so that they cut through each other.
void scene_interpenetrating(Frame& f, uint32_t w, uint32_t h) {
    Rng rng(0x696E7465ull);
    const float W = static_cast<float>(w), H = static_cast<float>(h);
    for (int k = 0; k < 24; k++) {
        const float cx = rng.range(0.25f, 0.75f) * W;
        const float cy = rng.range(0.25f, 0.75f) * H;
        Triangle t;
        t.flat = true;
        const float cr = rng.range(0.15f, 1.0f);
        const float cg = rng.range(0.15f, 1.0f);
        const float cb = rng.range(0.15f, 1.0f);
        for (int i = 0; i < 3; i++) {
            const float dx = rng.range(-0.38f, 0.38f) * W;
            const float dy = rng.range(-0.38f, 0.38f) * H;
            const float z = rng.range(0.05f, 0.95f);
            t.v[i] = vtx(cx + dx, cy + dy, z, cr, cg, cb);
        }
        f.triangles.push_back(t);
    }
}

// A receding floor and 20 large smooth triangles with strongly varying 1/w, colours interpolated
// perspective-correctly (Triangle::perspective).
void scene_perspective(Frame& f, uint32_t w, uint32_t h) {
    Rng rng(0x70657273ull);
    const float W = static_cast<float>(w), H = static_cast<float>(h);
    auto pv = [&](float x, float y, float z, float depth_w, float r, float g, float b) {
        Vertex v = vtx(x, y, z, r, g, b);
        v.rhw = 1.0f / depth_w;
        return v;
    };
    // The floor: near edge at the bottom (w = 1), far edge at the top (w = 16).
    Triangle a, b;
    a.perspective = b.perspective = true;
    a.v[0] = pv(0.05f * W, 0.95f * H, 0.1f, 1.0f, 1.0f, 0.1f, 0.1f);
    a.v[1] = pv(0.95f * W, 0.95f * H, 0.1f, 1.0f, 0.1f, 1.0f, 0.1f);
    a.v[2] = pv(0.65f * W, 0.35f * H, 0.9f, 16.0f, 0.1f, 0.1f, 1.0f);
    b.v[0] = a.v[0];
    b.v[1] = a.v[2];
    b.v[2] = pv(0.35f * W, 0.35f * H, 0.9f, 16.0f, 1.0f, 1.0f, 0.1f);
    f.triangles.push_back(a);
    f.triangles.push_back(b);
    for (int k = 0; k < 20; k++) {
        const float cx = rng.range(0.25f, 0.75f) * W;
        const float cy = rng.range(0.25f, 0.75f) * H;
        Triangle t;
        t.perspective = true;
        for (int i = 0; i < 3; i++) {
            const float dx = rng.range(-0.4f, 0.4f) * W;
            const float dy = rng.range(-0.4f, 0.4f) * H;
            const float z = rng.range(0.05f, 0.95f);
            const float depth_w = rng.range(1.0f, 12.0f);
            const float cr = rng.range(0.0f, 1.0f);
            const float cg = rng.range(0.0f, 1.0f);
            const float cb = rng.range(0.0f, 1.0f);
            t.v[i] = pv(cx + dx, cy + dy, z, depth_w, cr, cg, cb);
        }
        f.triangles.push_back(t);
    }
}

// Fragment shaders (Triangle::shader): a procedural background from the pixel position, a receding checkerboard floor with
// perspective-correct texture coordinates, a Phong-lit sphere (normalised normals, reciprocal square root, a specular
// power from log2 and exp2), a striped screen-door quad that discards fragments, and a translucent quad whose alpha comes
// from a sine of the varying. The frame therefore has an RGBA8 target.
void scene_shader(Frame& f, uint32_t w, uint32_t h) {
    const float W = static_cast<float>(w), H = static_cast<float>(h);
    const double kPi = 3.14159265358979323846;
    using shader::kOpAdd; using shader::kOpMul; using shader::kOpMad; using shader::kOpMov; using shader::kOpMax;
    using shader::kOpFrc; using shader::kOpSlt; using shader::kOpRsq; using shader::kOpLg2; using shader::kOpEx2;
    using shader::kOpSin; using shader::kOpSqrt; using shader::kOpSeq;

    // 1: background from the fragment position: a radial vignette over a diagonal gradient.
    ShaderBuilder bg(0);
    {
        const uint32_t fc = bg.fragcoord();
        const uint32_t out = bg.alloc4();
        const uint32_t nx = bg.emit(kOpMad, bg.reg(fc), bg.constant(2.0f / W), bg.constant(-1.0f));          // x in [-1, 1]
        const uint32_t ny = bg.emit(kOpMad, bg.reg(fc + 1), bg.constant(2.0f / H), bg.constant(-1.0f));
        const uint32_t r2 = bg.emit(kOpMul, bg.reg(nx), bg.reg(nx));
        const uint32_t r2b = bg.emit(kOpMad, bg.reg(ny), bg.reg(ny), bg.reg(r2));
        const uint32_t r = bg.emit(kOpSqrt, bg.reg(r2b));
        const uint32_t v = bg.emit(kOpMad, bg.reg(r), bg.constant(-0.35f), bg.constant(0.55f));              // darker toward the corners
        bg.emit_to(kOpMul, out, bg.reg(v), bg.constant(0.55f), {0}, true);
        bg.emit_to(kOpMul, out + 1, bg.reg(v), bg.constant(0.75f), {0}, true);
        bg.emit_to(kOpMov, out + 2, bg.reg(v), {0}, {0}, true);
        bg.emit_to(kOpMov, out + 3, bg.constant(1.0f));
        f.shaders.push_back(bg.finish(out));                                                                  // shader 1
    }
    // 2: checkerboard of 12 x 12 cells from (u, v), lit by a constant: two colours.
    ShaderBuilder chk(2);
    {
        const uint32_t out = chk.alloc4();
        const uint32_t su = chk.emit(kOpMul, chk.in(0), chk.constant(12.0f));
        const uint32_t sv = chk.emit(kOpMul, chk.in(1), chk.constant(12.0f));
        const uint32_t fu = chk.emit(kOpFrc, chk.reg(su));
        const uint32_t fv = chk.emit(kOpFrc, chk.reg(sv));
        const uint32_t hu = chk.emit(kOpSlt, chk.reg(fu), chk.constant(0.5f));
        const uint32_t hv = chk.emit(kOpSlt, chk.reg(fv), chk.constant(0.5f));
        const uint32_t sum = chk.emit(kOpAdd, chk.reg(hu), chk.reg(hv));
        const uint32_t par = chk.emit(kOpSeq, chk.reg(sum), chk.constant(1.0f));
        chk.emit_to(kOpMad, out, chk.reg(par), chk.constant(0.80f), chk.constant(0.12f));
        chk.emit_to(kOpMad, out + 1, chk.reg(par), chk.constant(0.70f), chk.constant(0.10f));
        chk.emit_to(kOpMad, out + 2, chk.reg(par), chk.constant(0.30f), chk.constant(0.14f));
        chk.emit_to(kOpMov, out + 3, chk.constant(1.0f));
        f.shaders.push_back(chk.finish(out));                                                                 // shader 2
    }
    // 3: Phong lighting from an interpolated normal, light from the upper left, specular exponent 32.
    ShaderBuilder ph(3);
    {
        const float L[3] = {-0.5f, -0.6f, 0.62f};
        const float len = std::sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]);
        const float lx = L[0] / len, ly = L[1] / len, lz = L[2] / len;
        const float hx = lx, hy = ly, hz = lz + 1.0f;                       // half vector with the viewer on +z
        const float hl = std::sqrt(hx * hx + hy * hy + hz * hz);
        const uint32_t out = ph.alloc4();
        const uint32_t n2 = ph.emit(kOpMul, ph.in(0), ph.in(0));
        const uint32_t n2b = ph.emit(kOpMad, ph.in(1), ph.in(1), ph.reg(n2));
        const uint32_t n2c = ph.emit(kOpMad, ph.in(2), ph.in(2), ph.reg(n2b));
        const uint32_t inv = ph.emit(kOpRsq, ph.reg(n2c));
        auto dot = [&](float ax, float ay, float az) {
            const uint32_t t0 = ph.emit(kOpMul, ph.in(0), ph.constant(ax));
            const uint32_t t1 = ph.emit(kOpMad, ph.in(1), ph.constant(ay), ph.reg(t0));
            const uint32_t t2 = ph.emit(kOpMad, ph.in(2), ph.constant(az), ph.reg(t1));
            return ph.emit(kOpMul, ph.reg(t2), ph.reg(inv));
        };
        const uint32_t ndl = dot(lx, ly, lz);
        const uint32_t diff = ph.emit(kOpMax, ph.reg(ndl), ph.constant(0.0f));
        const uint32_t ndh = dot(hx / hl, hy / hl, hz / hl);
        const uint32_t ndhc = ph.emit(kOpMax, ph.reg(ndh), ph.constant(1.0e-4f));
        const uint32_t lg = ph.emit(kOpLg2, ph.reg(ndhc));
        const uint32_t lg32 = ph.emit(kOpMul, ph.reg(lg), ph.constant(32.0f));
        const uint32_t spec = ph.emit(kOpEx2, ph.reg(lg32));
        const uint32_t lit = ph.emit(kOpMad, ph.reg(diff), ph.constant(0.85f), ph.constant(0.15f));
        const uint32_t sp = ph.emit(kOpMul, ph.reg(spec), ph.constant(0.6f));
        ph.emit_to(kOpMad, out, ph.reg(lit), ph.constant(0.85f), ph.reg(sp), true);
        ph.emit_to(kOpMad, out + 1, ph.reg(lit), ph.constant(0.30f), ph.reg(sp), true);
        ph.emit_to(kOpMad, out + 2, ph.reg(lit), ph.constant(0.25f), ph.reg(sp), true);
        ph.emit_to(kOpMov, out + 3, ph.constant(1.0f));
        f.shaders.push_back(ph.finish(out));                                                                  // shader 3
    }
    // 4: screen door: discard the fragments whose fract(8 u) is below 0.4.
    ShaderBuilder door(1);
    {
        const uint32_t out = door.alloc4();
        const uint32_t su = door.emit(kOpMul, door.in(0), door.constant(8.0f));
        const uint32_t fu = door.emit(kOpFrc, door.reg(su));
        door.kill(ShaderBuilder::Src{door.emit(kOpAdd, door.reg(fu), door.constant(-0.4f))});
        door.emit_to(kOpMad, out, door.in(0), door.constant(-0.7f), door.constant(1.0f));
        door.emit_to(kOpMov, out + 1, door.constant(0.75f));
        door.emit_to(kOpMad, out + 2, door.in(0), door.constant(0.8f), door.constant(0.1f));
        door.emit_to(kOpMov, out + 3, door.constant(1.0f));
        f.shaders.push_back(door.finish(out));                                                                // shader 4
    }
    // 5: translucent colour whose alpha follows 0.5 + 0.4 sin(20 u).
    ShaderBuilder tr(1);
    {
        const uint32_t out = tr.alloc4();
        const uint32_t arg = tr.emit(kOpMul, tr.in(0), tr.constant(20.0f));
        const uint32_t s = tr.emit(kOpSin, tr.reg(arg));
        tr.emit_to(kOpMov, out, tr.constant(0.95f));
        tr.emit_to(kOpMov, out + 1, tr.constant(0.85f));
        tr.emit_to(kOpMov, out + 2, tr.constant(0.15f));
        tr.emit_to(kOpMad, out + 3, tr.reg(s), tr.constant(0.4f), tr.constant(0.5f));
        f.shaders.push_back(tr.finish(out));                                                                  // shader 5
    }

    auto add_tri = [&](uint16_t shader, const float xy[3][2], const float z[3], const float rhw[3], const float* attr, uint32_t n,
                       bool persp, bool own = false, BlendMode blend = BlendMode::None, bool dw = true) {
        Triangle t;
        t.shader = shader;
        t.perspective = persp;
        t.own_state = own;
        t.blend = blend;
        t.depth_write = dw;
        t.attr_offset = static_cast<uint32_t>(f.attrs.size());
        for (int k = 0; k < 3; k++) {
            t.v[k] = Vertex{};
            t.v[k].x = xy[k][0];
            t.v[k].y = xy[k][1];
            t.v[k].z = z[k];
            t.v[k].rhw = rhw[k];
            for (uint32_t c = 0; c < n; c++) f.attrs.push_back(attr[k * n + c]);
        }
        f.triangles.push_back(t);
    };
    auto add_quad = [&](uint16_t shader, float x0, float y0, float x1, float y1, float z, const float* corner_attr, uint32_t n,
                        bool own = false, BlendMode blend = BlendMode::None, bool dw = true) {
        const float xy[4][2] = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
        const float zz[3] = {z, z, z}, one[3] = {1, 1, 1};
        for (const auto& idx : {std::array<int, 3>{0, 1, 2}, std::array<int, 3>{0, 2, 3}}) {
            const float p[3][2] = {{xy[idx[0]][0], xy[idx[0]][1]}, {xy[idx[1]][0], xy[idx[1]][1]}, {xy[idx[2]][0], xy[idx[2]][1]}};
            std::vector<float> a;
            for (int k = 0; k < 3; k++) for (uint32_t c = 0; c < n; c++) a.push_back(corner_attr[idx[k] * n + c]);
            add_tri(shader, p, zz, one, a.data(), n, false, own, blend, dw);
        }
    };

    // Background (no varyings), far plane.
    add_quad(1, 0, 0, W, H, 0.95f, nullptr, 0);

    // The floor: near edge at the bottom (w = 1), far edge at the top (w = 8); u, v map the floor plane.
    {
        const float yn = 0.97f * H, yf = 0.52f * H, xn0 = 0.02f * W, xn1 = 0.98f * W, xf0 = 0.36f * W, xf1 = 0.64f * W;
        const float xy[4][2] = {{xn0, yn}, {xn1, yn}, {xf1, yf}, {xf0, yf}};
        const float rhw[4] = {1.0f, 1.0f, 1.0f / 8.0f, 1.0f / 8.0f};
        const float zv[4] = {0.2f, 0.2f, 0.7f, 0.7f};
        const float uv[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        for (const auto& idx : {std::array<int, 3>{0, 1, 2}, std::array<int, 3>{0, 2, 3}}) {
            const float p[3][2] = {{xy[idx[0]][0], xy[idx[0]][1]}, {xy[idx[1]][0], xy[idx[1]][1]}, {xy[idx[2]][0], xy[idx[2]][1]}};
            const float z3[3] = {zv[idx[0]], zv[idx[1]], zv[idx[2]]}, r3[3] = {rhw[idx[0]], rhw[idx[1]], rhw[idx[2]]};
            const float a[6] = {uv[idx[0]][0], uv[idx[0]][1], uv[idx[1]][0], uv[idx[1]][1], uv[idx[2]][0], uv[idx[2]][1]};
            add_tri(2, p, z3, r3, a, 2, true);
        }
    }

    // The sphere: 40 x 28 quads, orthographic, normal = position / radius as varyings.
    {
        const int nu = 40, nv = 28;
        const float cx = 0.40f * W, cy = 0.42f * H, R = 0.28f * std::min(W, H);
        auto pos = [&](int i, int j, float p[3]) {
            const double phi = kPi * static_cast<double>(j) / nv, th = 2.0 * kPi * static_cast<double>(i) / nu;
            p[0] = static_cast<float>(std::sin(phi) * std::cos(th));
            p[1] = static_cast<float>(-std::cos(phi));
            p[2] = static_cast<float>(std::sin(phi) * std::sin(th));   // towards the viewer when positive
        };
        for (int j = 0; j < nv; j++) {
            for (int i = 0; i < nu; i++) {
                float p[4][3];
                pos(i, j, p[0]); pos(i + 1, j, p[1]); pos(i + 1, j + 1, p[2]); pos(i, j + 1, p[3]);
                for (const auto& idx : {std::array<int, 3>{0, 1, 2}, std::array<int, 3>{0, 2, 3}}) {
                    float a[9], xy[3][2], z3[3], r3[3] = {1, 1, 1};
                    float facing = 0.0f;
                    for (int k = 0; k < 3; k++) {
                        const float* q = p[idx[k]];
                        xy[k][0] = cx + R * q[0];
                        xy[k][1] = cy + R * q[1];
                        z3[k] = 0.5f - 0.3f * q[2];
                        for (int c = 0; c < 3; c++) a[3 * k + c] = q[c];
                        facing += q[2];
                    }
                    if (facing <= 0.0f) continue;   // back half: not drawn (no culling stage needed)
                    add_tri(3, xy, z3, r3, a, 3, true);
                }
            }
        }
    }

    // Screen door in front of the sphere's right edge, discarding stripes.
    {
        const float corner[4] = {0, 1, 1, 0};
        add_quad(4, 0.52f * W, 0.14f * H, 0.82f * W, 0.60f * H, 0.3f, corner, 1);
    }
    // A translucent layer over everything on the right, blended with alpha, depth writes off.
    {
        const float corner[4] = {0, 1, 1, 0};
        add_quad(5, 0.60f * W, 0.30f * H, 0.97f * W, 0.88f * H, 0.1f, corner, 1, true, BlendMode::Alpha, false);
    }
}

// 60 triangles with their own depth and blend state in API order: opaque and LESS_EQUAL layers, and alpha, additive,
// premultiplied and multiplicative blended layers with depth writes off or the depth test off.
void scene_state(Frame& f, uint32_t w, uint32_t h) {
    Rng rng(0x73746174ull);
    const float W = static_cast<float>(w), H = static_cast<float>(h);
    for (int k = 0; k < 60; k++) {
        const float cx = rng.range(0.2f, 0.8f) * W;
        const float cy = rng.range(0.2f, 0.8f) * H;
        Triangle t;
        const float z0 = rng.range(0.05f, 0.95f);
        const float alpha = rng.range(0.2f, 0.9f);
        for (int i = 0; i < 3; i++) {
            const float dx = rng.range(-0.3f, 0.3f) * W;
            const float dy = rng.range(-0.3f, 0.3f) * H;
            Vertex v = vtx(cx + dx, cy + dy, k % 6 == 5 ? z0 : z0 + rng.range(-0.05f, 0.05f), rng.range(0.1f, 1.0f),
                           rng.range(0.1f, 1.0f), rng.range(0.1f, 1.0f));
            v.color[3] = (k % 6 == 0 || k % 6 == 5) ? 1.0f : alpha;
            t.v[i] = v;
        }
        switch (k % 6) {
            case 0: break;   // default state
            case 1: t.own_state = true; t.depth_func = DepthFunc::Always; t.depth_write = false; t.blend = BlendMode::Alpha; break;
            case 2: t.own_state = true; t.depth_func = DepthFunc::Less; t.depth_write = false; t.blend = BlendMode::Add; break;
            case 3: t.own_state = true; t.depth_func = DepthFunc::Always; t.depth_write = false; t.blend = (k / 6) % 2 ? BlendMode::AlphaAdd : BlendMode::Premultiplied; break;
            case 4: t.own_state = true; t.depth_func = DepthFunc::LessEqual; t.depth_write = false; t.blend = BlendMode::Multiply; break;
            case 5: t.own_state = true; t.depth_func = DepthFunc::LessEqual; t.depth_write = true; break;
        }
        f.triangles.push_back(t);
    }
}

// One right triangle whose legs run far past the viewport, so that its edges need guard-band
// clipping and the hypotenuse still lies beyond the far corner of the viewport.
void scene_fullscreen(Frame& f, uint32_t w, uint32_t h) {
    const float W = static_cast<float>(w), H = static_cast<float>(h);
    const float ax = -8.0f, ay = -8.0f;
    const float bx = ax + 2.2f * (W + 8.0f), by = ay;
    const float cx = ax, cy = ay + 2.2f * (H + 8.0f);
    f.triangles.push_back(tri(
        vtx(ax, ay, 0.25f, 0.9f, 0.2f, 0.1f), vtx(bx, by, 0.50f, 0.1f, 0.8f, 0.3f), vtx(cx, cy, 0.75f, 0.2f, 0.3f, 0.9f),
        false));
}

// Overlapping flat triangles at exactly the same depth under LESS_EQUAL: the later triangle wins
// every tie. Two of them sit slightly behind, to check that LESS_EQUAL still rejects farther ones.
void scene_coplanar(Frame& f, uint32_t w, uint32_t h) {
    f.depth_func = DepthFunc::LessEqual;
    Rng rng(0x636F706Cull);
    const float W = static_cast<float>(w), H = static_cast<float>(h);
    for (int k = 0; k < 10; k++) {
        const float cx = rng.range(0.3f, 0.7f) * W;
        const float cy = rng.range(0.3f, 0.7f) * H;
        Triangle t;
        t.flat = true;
        const float r = rng.range(0.15f, 1.0f);
        const float g = rng.range(0.15f, 1.0f);
        const float b = rng.range(0.15f, 1.0f);
        const float z = (k == 3 || k == 6) ? 0.6f : 0.5f;
        for (int i = 0; i < 3; i++) {
            const float dx = rng.range(-0.3f, 0.3f) * W;
            const float dy = rng.range(-0.3f, 0.3f) * H;
            t.v[i] = vtx(cx + dx, cy + dy, z, r, g, b);
        }
        f.triangles.push_back(t);
    }
}

void scene_tiny(Frame& f, uint32_t w, uint32_t h) {
    Rng rng(0x74696E79ull);
    const float W = static_cast<float>(w), H = static_cast<float>(h);
    for (int k = 0; k < 10000; k++) {
        const float cx = rng.range(0.0f, W);
        const float cy = rng.range(0.0f, H);
        const float z = rng.range(0.0f, 1.0f);
        Triangle t;
        t.flat = true;
        const float r = rng.range(0.1f, 1.0f);
        const float g = rng.range(0.1f, 1.0f);
        const float b = rng.range(0.1f, 1.0f);
        const float size = rng.range(0.5f, 3.0f);
        for (int i = 0; i < 3; i++) {
            const float dx = rng.range(-size, size);
            const float dy = rng.range(-size, size);
            t.v[i] = vtx(cx + dx, cy + dy, z, r, g, b);
        }
        f.triangles.push_back(t);
    }
}

void scene_random(Frame& f, uint32_t w, uint32_t h) {
    Rng rng(0x72616E64ull);
    const float W = static_cast<float>(w), H = static_cast<float>(h);
    const float min_size = 3.0f, max_size = 0.35f * (W < H ? W : H);
    const float log_min = std::log(min_size), log_max = std::log(max_size);
    for (int k = 0; k < 10000; k++) {
        const float cx = rng.range(-0.05f, 1.05f) * W;
        const float cy = rng.range(-0.05f, 1.05f) * H;
        const float size = std::exp(rng.range(log_min, log_max));
        Triangle t;
        t.flat = rng.uniform() < 0.3f;
        for (int i = 0; i < 3; i++) {
            const float dx = rng.range(-1.0f, 1.0f) * size;
            const float dy = rng.range(-1.0f, 1.0f) * size;
            const float z = rng.range(0.0f, 1.0f);
            t.v[i] = random_color_vertex(rng, cx + dx, cy + dy, z);
        }
        f.triangles.push_back(t);
    }
}

}  // namespace

const std::vector<SceneInfo>& scene_list() {
    static const std::vector<SceneInfo> scenes = {
        {"one_triangle", "a single smooth-shaded triangle", false},
        {"fan", "96-triangle smooth-shaded fan with shared centre and rim edges", false},
        {"grid", "jittered shared-vertex grid mesh covering the viewport, flat shaded", true},
        {"interpenetrating", "24 large flat triangles with random depths that cut through each other", false},
        {"fullscreen", "one right triangle covering the viewport, edges beyond the guard band", false},
        {"coplanar_lequal", "overlapping coplanar flat triangles under LESS_EQUAL: the last drawn wins ties", false},
        {"perspective", "a receding floor and 20 large triangles with perspective-correct colours", false},
        {"state", "60 triangles with their own depth function, depth write and blend state (alpha, add, premultiplied, multiply)", false},
        {"shader", "fragment shaders: background, checkerboard floor, Phong sphere, discard stripes, translucent layer (RGBA8 target)", false},
        {"tiny_10k", "10,000 triangles 1 to 6 pixels across", false},
        {"random_10k", "10,000 random triangles, 3 px to a third of the screen, partly off screen", false},
    };
    return scenes;
}

bool make_scene(const std::string& name, uint32_t width, uint32_t height, Frame& out) {
    base_frame(out, width, height);
    if (name == "one_triangle") scene_one_triangle(out, width, height);
    else if (name == "fan") scene_fan(out, width, height);
    else if (name == "grid") scene_grid(out, width, height);
    else if (name == "interpenetrating") scene_interpenetrating(out, width, height);
    else if (name == "fullscreen") scene_fullscreen(out, width, height);
    else if (name == "coplanar_lequal") scene_coplanar(out, width, height);
    else if (name == "perspective") scene_perspective(out, width, height);
    else if (name == "state") scene_state(out, width, height);
    else if (name == "shader") scene_shader(out, width, height);
    else if (name == "tiny_10k") scene_tiny(out, width, height);
    else if (name == "random_10k") scene_random(out, width, height);
    else return false;
    return true;
}

}  // namespace ttr
