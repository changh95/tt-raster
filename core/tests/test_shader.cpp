// SPDX-License-Identifier: Apache-2.0
//
// Fragment shaders on the CPU reference: every instruction, operand modifiers, validation, and shader triangles in
// frames (varyings, perspective, fragment coordinates, discard, blending, API order with fixed-function triangles).

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "check.h"
#include "ttraster/reference.h"
#include "ttraster/setup.h"
#include "ttraster/ttraster_isa.h"   // the C view of the instruction set, used by the Mesa translator

using namespace ttr;
using namespace ttr::shader;

// The C header (core/include/ttraster/ttraster_isa.h) and the C++ one the kernels and the interpreter use must agree.
static_assert(TTR_OP_NOP == kOpNop && TTR_OP_MOV == kOpMov && TTR_OP_ADD == kOpAdd && TTR_OP_MUL == kOpMul && TTR_OP_MAD == kOpMad &&
              TTR_OP_MIN == kOpMin && TTR_OP_MAX == kOpMax && TTR_OP_FRC == kOpFrc && TTR_OP_FLR == kOpFlr && TTR_OP_RCP == kOpRcp &&
              TTR_OP_RSQ == kOpRsq && TTR_OP_SQRT == kOpSqrt && TTR_OP_EX2 == kOpEx2 && TTR_OP_LG2 == kOpLg2 && TTR_OP_SIN == kOpSin &&
              TTR_OP_COS == kOpCos && TTR_OP_SLT == kOpSlt && TTR_OP_SGE == kOpSge && TTR_OP_SEQ == kOpSeq && TTR_OP_SNE == kOpSne &&
              TTR_OP_CMP == kOpCmp && TTR_OP_SEL == kOpSel && TTR_OP_KILL == kOpKill && TTR_OP_COUNT == kOpCount);
static_assert(TTR_FLAG_SAT == kFlagSat && TTR_SRC_INDEX_MASK == kSrcIndexMask && TTR_SRC_CONST == kSrcConst && TTR_SRC_NEG == kSrcNeg &&
              TTR_SRC_ABS == kSrcAbs && TTR_MAX_REGS == kMaxRegs && TTR_MAX_CONSTS == kMaxConsts &&
              TTR_MAX_INSTRUCTIONS == kMaxInstructions && TTR_MAX_INPUTS == kMaxInputs && TTR_INSTR_WORDS == kInstrWords);

namespace {

bool near_f(float a, float b, float tol = 1e-6f) { return std::fabs(a - b) <= tol * std::max(1.0f, std::fabs(b)); }

// Result of a one-operation program: r = op(a, b, c) with a, b, c the inputs 0..2, returned in the first channel.
float run_op(uint32_t op, float a, float b = 0.0f, float c = 0.0f, bool sat = false) {
    ShaderBuilder sb(3);
    const uint32_t out = sb.alloc4();
    const uint32_t ns = op_sources(op);   // unused operands must be zero
    sb.emit_to(op, out, ns > 0 ? sb.in(0) : ShaderBuilder::Src{0}, ns > 1 ? sb.in(1) : ShaderBuilder::Src{0},
            ns > 2 ? sb.in(2) : ShaderBuilder::Src{0}, sat);
    sb.emit_to(kOpMov, out + 1, sb.constant(0.0f));
    sb.emit_to(kOpMov, out + 2, sb.constant(0.0f));
    sb.emit_to(kOpMov, out + 3, sb.constant(1.0f));
    const ShaderProgram p = sb.finish(out);
    CHECK(validate_shader(p).empty());
    const float in[3] = {a, b, c};
    float regs[kMaxRegs];
    return run_shader(p, in, nullptr, regs).rgba[0];
}

// A shaded frame under construction: 64x64 target, shader instances and their varyings.
struct Scene {
    Frame f;
    Scene() {
        f.width = f.height = 64;
        f.clear_color[0] = f.clear_color[1] = f.clear_color[2] = 0.0f;
        f.clear_color[3] = 1.0f;
    }
    // A rectangle [x0, x1) x [y0, y1) as two triangles with the shader's varyings: `uv` gives the varyings of the four
    // corners (TL, TR, BR, BL), n floats each, and `rhw` the 1/w of each corner (all 1 for a linear triangle).
    void rect(uint16_t shader, float x0, float y0, float x1, float y1, float z, const std::vector<float>& corners, uint32_t n,
              const float rhw[4] = nullptr, bool persp = false, bool own = false, BlendMode blend = BlendMode::None,
              DepthFunc df = DepthFunc::Less, bool dw = true) {
        const float px[4] = {x0, x1, x1, x0}, py[4] = {y0, y0, y1, y1};
        const int tris[2][3] = {{0, 1, 2}, {0, 2, 3}};
        for (const auto& t : tris) {
            Triangle tri;
            tri.shader = shader;
            tri.perspective = persp;
            tri.own_state = own;
            tri.blend = blend;
            tri.depth_func = df;
            tri.depth_write = dw;
            tri.attr_offset = static_cast<uint32_t>(f.attrs.size());
            for (int k = 0; k < 3; k++) {
                Vertex& v = tri.v[k];
                v = Vertex{};
                v.x = px[t[k]];
                v.y = py[t[k]];
                v.z = z;
                v.rhw = rhw != nullptr ? rhw[t[k]] : 1.0f;
                for (uint32_t c = 0; c < n; c++) f.attrs.push_back(corners[t[k] * n + c]);
            }
            f.triangles.push_back(tri);
        }
    }
    void flat_rect(float x0, float y0, float x1, float y1, float z, const float c[4]) {
        Triangle t[2];
        const float px[4] = {x0, x1, x1, x0}, py[4] = {y0, y0, y1, y1};
        const int idx[2][3] = {{0, 1, 2}, {0, 2, 3}};
        for (int i = 0; i < 2; i++) {
            t[i].flat = true;
            for (int k = 0; k < 3; k++) {
                t[i].v[k] = Vertex{};
                t[i].v[k].x = px[idx[i][k]];
                t[i].v[k].y = py[idx[i][k]];
                t[i].v[k].z = z;
                t[i].v[k].rhw = 1.0f;
                for (int ch = 0; ch < 4; ch++) t[i].v[k].color[ch] = c[ch];
            }
            f.triangles.push_back(t[i]);
        }
    }
};

uint32_t pack(float r, float g, float b, float a) {
    const float c[4] = {r, g, b, a};
    return pack_rgba8(c);
}

// out = (in0, in1, in2, in3)
ShaderProgram passthrough4() {
    ShaderBuilder sb(4);
    const uint32_t out = sb.alloc4();
    for (uint32_t k = 0; k < 4; k++) sb.emit_to(kOpMov, out + k, sb.in(k));
    return sb.finish(out);
}

}  // namespace

int main() {
    const float inf = std::numeric_limits<float>::infinity();

    // ---- every instruction --------------------------------------------------------------------------------------
    CHECK(run_op(kOpMov, 1.5f) == 1.5f);
    CHECK(run_op(kOpAdd, 1.5f, 2.25f) == 3.75f);
    CHECK(run_op(kOpMul, 1.5f, -4.0f) == -6.0f);
    CHECK(run_op(kOpMad, 3.0f, 4.0f, 5.0f) == 17.0f);
    {   // fused: the product is not rounded before the addition
        const float a = 1.0f + 0x1p-12f, c = -(1.0f + 0x1p-11f);
        CHECK(run_op(kOpMad, a, a, c) == 0x1p-24f);   // a * a = 1 + 2^-11 + 2^-24 exactly; rounded first it would be 0
        CHECK(a * a + c == 0.0f);
    }
    CHECK(run_op(kOpMin, 2.0f, 3.0f) == 2.0f && run_op(kOpMin, 3.0f, 2.0f) == 2.0f);
    CHECK(run_op(kOpMax, 2.0f, 3.0f) == 3.0f && run_op(kOpMax, 3.0f, 2.0f) == 3.0f);
    CHECK(run_op(kOpFrc, 2.75f) == 0.75f && run_op(kOpFrc, -0.25f) == 0.75f && run_op(kOpFrc, 3.0f) == 0.0f);
    CHECK(run_op(kOpFlr, 2.75f) == 2.0f && run_op(kOpFlr, -0.25f) == -1.0f);
    CHECK(run_op(kOpRcp, 4.0f) == 0.25f && run_op(kOpRcp, 0.0f) == inf);
    CHECK(near_f(run_op(kOpRsq, 4.0f), 0.5f) && near_f(run_op(kOpRsq, -4.0f), 0.5f));
    CHECK(near_f(run_op(kOpSqrt, 9.0f), 3.0f) && near_f(run_op(kOpSqrt, -9.0f), 3.0f));
    CHECK(near_f(run_op(kOpEx2, 3.0f), 8.0f) && near_f(run_op(kOpEx2, -1.0f), 0.5f));
    CHECK(near_f(run_op(kOpLg2, 8.0f), 3.0f) && near_f(run_op(kOpLg2, 0.5f), -1.0f));
    CHECK(near_f(run_op(kOpSin, 0.5f), std::sin(0.5f)) && near_f(run_op(kOpCos, 0.5f), std::cos(0.5f)));
    CHECK(run_op(kOpSlt, 1.0f, 2.0f) == 1.0f && run_op(kOpSlt, 2.0f, 2.0f) == 0.0f);
    CHECK(run_op(kOpSge, 2.0f, 2.0f) == 1.0f && run_op(kOpSge, 1.0f, 2.0f) == 0.0f);
    CHECK(run_op(kOpSeq, 2.0f, 2.0f) == 1.0f && run_op(kOpSeq, 1.0f, 2.0f) == 0.0f);
    CHECK(run_op(kOpSne, 2.0f, 2.0f) == 0.0f && run_op(kOpSne, 1.0f, 2.0f) == 1.0f);
    CHECK(run_op(kOpCmp, -1.0f, 10.0f, 20.0f) == 10.0f && run_op(kOpCmp, 0.0f, 10.0f, 20.0f) == 20.0f);
    CHECK(run_op(kOpSel, 0.0f, 10.0f, 20.0f) == 20.0f && run_op(kOpSel, 0.25f, 10.0f, 20.0f) == 10.0f);
    // saturate: clamp to [0, 1], NaN to 0
    CHECK(run_op(kOpMov, 1.5f, 0, 0, true) == 1.0f && run_op(kOpMov, -0.5f, 0, 0, true) == 0.0f &&
          run_op(kOpMov, 0.25f, 0, 0, true) == 0.25f);
    CHECK(run_op(kOpMov, std::numeric_limits<float>::quiet_NaN(), 0, 0, true) == 0.0f);

    // ---- operand modifiers and constants ------------------------------------------------------------------------
    {
        ShaderBuilder sb(2);
        const uint32_t out = sb.alloc4();
        sb.emit_to(kOpAdd, out, -sb.in(0), sb.in(1));                     // -a + b
        sb.emit_to(kOpMul, out + 1, sb.in(0).abs(), sb.in(1).abs());      // |a| * |b|
        sb.emit_to(kOpMov, out + 2, -sb.in(0).abs());                     // -|a|
        sb.emit_to(kOpMad, out + 3, sb.in(0), sb.constant(2.0f), sb.constant(0.5f));
        const ShaderProgram p = sb.finish(out);
        CHECK(validate_shader(p).empty());
        const float in[2] = {-3.0f, 5.0f};
        float regs[kMaxRegs];
        const ShaderResult r = run_shader(p, in, nullptr, regs);
        CHECK(r.rgba[0] == 8.0f && r.rgba[1] == 15.0f && r.rgba[2] == -3.0f && r.rgba[3] == -5.5f);
        CHECK(!r.killed);
    }
    {   // constants are deduplicated, so a program with the same literal twice has one table entry
        ShaderBuilder sb(1);
        const auto a = sb.constant(0.5f), b = sb.constant(0.5f), c = sb.constant(0.25f);
        CHECK(a.bits == b.bits && a.bits != c.bits);
    }

    // ---- kill ---------------------------------------------------------------------------------------------------
    {
        ShaderBuilder sb(1);
        const uint32_t out = sb.alloc4();
        sb.kill(sb.in(0));
        for (uint32_t k = 0; k < 3; k++) sb.emit_to(kOpMov, out + k, sb.constant(0.5f));
        sb.emit_to(kOpMov, out + 3, sb.constant(1.0f));
        const ShaderProgram p = sb.finish(out);
        CHECK(validate_shader(p).empty() && p.uses_kill());
        float regs[kMaxRegs];
        float in = -0.1f;
        CHECK(run_shader(p, &in, nullptr, regs).killed);
        in = 0.0f;
        CHECK(!run_shader(p, &in, nullptr, regs).killed);
    }

    // ---- validation ---------------------------------------------------------------------------------------------
    {
        ShaderBuilder sb(1);
        const uint32_t out = sb.alloc4();
        const uint32_t tmp = sb.alloc();
        sb.emit_to(kOpAdd, out, sb.reg(tmp), sb.in(0));   // tmp was never written
        CHECK(validate_shader(sb.finish(out)).find("read before") != std::string::npos);
    }
    {
        ShaderBuilder sb(1);
        const uint32_t out = sb.alloc4();
        sb.emit_to(kOpMov, out, sb.in(0));                // only r[out] is written
        CHECK(validate_shader(sb.finish(out)).find("never written") != std::string::npos);
    }
    {
        ShaderProgram p = passthrough4();
        CHECK(validate_shader(p).empty());
        ShaderProgram q = p;
        q.code[0].w[2] = 1;                            // Mov has one source: operand b must be zero
        CHECK(!validate_shader(q).empty());
        q = p;
        q.code[0].w[1] = kSrcConst | 7;                // constant out of range
        CHECK(validate_shader(q).find("constant") != std::string::npos);
        q = p;
        q.code[0].w[0] = make_word0(kOpMov, 0, 200);   // destination out of range
        CHECK(validate_shader(q).find("destination") != std::string::npos);
        q = p;
        q.code[0].w[0] = make_word0(kOpCount, 0, 4);   // opcode out of range
        CHECK(!validate_shader(q).empty());
        q = p;
        q.num_regs = kMaxRegs + 1;
        CHECK(!validate_shader(q).empty());
        q = p;
        q.code.clear();
        CHECK(!validate_shader(q).empty());
    }

    // ---- a pass-through shader is the fixed-function pipeline ---------------------------------------------------
    for (int persp = 0; persp < 2; persp++) {
        const float rhw[4] = {1.0f, 0.25f, 0.125f, 0.5f};
        const float colours[16] = {1, 0, 0, 1, 0, 1, 0, 0.75f, 0, 0, 1, 0.5f, 1, 1, 0, 1};
        Scene fixed, shaded;
        shaded.f.shaders.push_back(passthrough4());
        // The same quad twice: with the vertex colours, and with the same values as shader varyings.
        for (int pass = 0; pass < 2; pass++) {
            Scene& s = pass == 0 ? fixed : shaded;
            const float px[4] = {4, 60, 60, 4}, py[4] = {4, 4, 60, 60};
            const int idx[2][3] = {{0, 1, 2}, {0, 2, 3}};
            for (const auto& t : idx) {
                Triangle tri;
                tri.perspective = persp != 0;
                tri.shader = pass == 1 ? 1 : 0;
                tri.attr_offset = static_cast<uint32_t>(s.f.attrs.size());
                for (int k = 0; k < 3; k++) {
                    tri.v[k] = Vertex{};
                    tri.v[k].x = px[t[k]];
                    tri.v[k].y = py[t[k]];
                    tri.v[k].z = 0.25f + 0.1f * static_cast<float>(t[k]);
                    tri.v[k].rhw = rhw[t[k]];
                    for (int ch = 0; ch < 4; ch++) {
                        tri.v[k].color[ch] = colours[4 * t[k] + ch];
                        if (pass == 1) s.f.attrs.push_back(colours[4 * t[k] + ch]);
                    }
                }
                s.f.triangles.push_back(tri);
            }
        }
        const Framebuffer a = render_reference(fixed.f), b = render_reference(shaded.f);
        CHECK(a.color == b.color && a.depth == b.depth && a.id == b.id);
        CHECK(a.color[a.index(32, 32)] != a.color[a.index(2, 2)]);   // the quad is not empty
    }

    // ---- procedural: a checkerboard from the varyings -----------------------------------------------------------
    {
        Scene s;
        ShaderBuilder sb(2);
        const uint32_t out = sb.alloc4();
        const uint32_t su = sb.emit(kOpMul, sb.in(0), sb.constant(4.0f));
        const uint32_t sv = sb.emit(kOpMul, sb.in(1), sb.constant(4.0f));
        const uint32_t fu = sb.emit(kOpFrc, sb.reg(su)), fv = sb.emit(kOpFrc, sb.reg(sv));
        const uint32_t hu = sb.emit(kOpSlt, sb.reg(fu), sb.constant(0.5f)), hv = sb.emit(kOpSlt, sb.reg(fv), sb.constant(0.5f));
        const uint32_t x = sb.emit(kOpAdd, sb.reg(hu), sb.reg(hv));          // 0, 1 or 2
        const uint32_t par = sb.emit(kOpSeq, sb.reg(x), sb.constant(1.0f));  // 1 where exactly one half is set
        sb.emit_to(kOpMov, out, sb.reg(par));
        sb.emit_to(kOpMov, out + 1, sb.reg(par));
        sb.emit_to(kOpMov, out + 2, sb.reg(par));
        sb.emit_to(kOpMov, out + 3, sb.constant(1.0f));
        s.f.shaders.push_back(sb.finish(out));
        s.rect(1, 0, 0, 64, 64, 0.5f, {0, 0, 1, 0, 1, 1, 0, 1}, 2);
        const Framebuffer fb = render_reference(s.f);
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                const bool white = ((x % 16) < 8) != ((y % 16) < 8);   // the pixel centre is in the first half of its cell
                CHECK(fb.color[fb.index(x, y)] == (white ? pack(1, 1, 1, 1) : pack(0, 0, 0, 1)));
            }
        }
    }

    // ---- fragment coordinates -----------------------------------------------------------------------------------
    {
        Scene s;
        ShaderBuilder sb2(0);
        const uint32_t fc = sb2.fragcoord();
        const uint32_t out = sb2.alloc4();
        sb2.emit_to(kOpMul, out, sb2.reg(fc), sb2.constant(1.0f / 64.0f));        // x / 64
        sb2.emit_to(kOpMul, out + 1, sb2.reg(fc + 1), sb2.constant(1.0f / 64.0f));  // y / 64
        sb2.emit_to(kOpMov, out + 2, sb2.reg(fc + 2));                              // z
        sb2.emit_to(kOpMov, out + 3, sb2.reg(fc + 3));                              // rhw
        s.f.shaders.push_back(sb2.finish(out));
        s.rect(1, 0, 0, 64, 64, 0.5f, {}, 0);
        const Framebuffer fb = render_reference(s.f);
        for (const auto& pt : {std::pair<int, int>{0, 0}, {10, 20}, {63, 63}, {31, 5}}) {
            const float fx = (static_cast<float>(pt.first) + 0.5f) / 64.0f, fy = (static_cast<float>(pt.second) + 0.5f) / 64.0f;
            CHECK(fb.color[fb.index(pt.first, pt.second)] == pack(fx, fy, 0.5f, 1.0f));
        }
    }

    // ---- discard: killed fragments leave colour, depth and id alone --------------------------------------------
    {
        Scene s;
        ShaderBuilder sb(1);
        const uint32_t out = sb.alloc4();
        sb.kill(sb.in(0));                                                    // kill where u < 0
        sb.emit_to(kOpMov, out, sb.constant(1.0f));
        sb.emit_to(kOpMov, out + 1, sb.constant(0.0f));
        sb.emit_to(kOpMov, out + 2, sb.constant(0.0f));
        sb.emit_to(kOpMov, out + 3, sb.constant(1.0f));
        s.f.shaders.push_back(sb.finish(out));
        const float green[4] = {0, 1, 0, 1};
        s.flat_rect(0, 0, 64, 64, 0.9f, green);                               // behind
        s.rect(1, 0, 0, 64, 64, 0.1f, {-1, 1, 1, -1}, 1);                     // u runs from -1 to 1 across the target
        const Framebuffer fb = render_reference(s.f);
        const size_t left = fb.index(10, 30), right = fb.index(50, 30);
        CHECK(fb.color[left] == pack(0, 1, 0, 1) && fb.depth[left] == 0.9f && fb.id[left] == 2);   // the second flat triangle
        CHECK(fb.color[right] == pack(1, 0, 0, 1) && near_f(fb.depth[right], 0.1f) && fb.id[right] == 3);
    }

    // ---- API order with fixed-function triangles, depth test, per-triangle state -------------------------------
    {
        Scene s;
        s.f.shaders.push_back(passthrough4());
        const float blue[4] = {0, 0, 1, 1};
        s.rect(1, 8, 8, 40, 40, 0.5f, {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1}, 4);   // red shader quad
        s.flat_rect(24, 24, 56, 56, 0.2f, blue);                                                // nearer blue quad
        s.flat_rect(4, 40, 30, 60, 0.8f, blue);                                                 // behind: loses where red is
        const Framebuffer fb = render_reference(s.f);
        CHECK(fb.color[fb.index(12, 12)] == pack(1, 0, 0, 1));
        CHECK(fb.color[fb.index(30, 30)] == pack(0, 0, 1, 1));   // blue is nearer
        CHECK(fb.color[fb.index(10, 50)] == pack(0, 0, 1, 1));
    }

    // ---- blending reads the quantized target -------------------------------------------------------------------
    {
        Scene s;
        s.f.clear_color[0] = s.f.clear_color[1] = s.f.clear_color[2] = 0.5f;   // 128 / 255 once quantized
        s.f.shaders.push_back(passthrough4());
        const std::vector<float> white_quarter = {1, 1, 1, 0.25f, 1, 1, 1, 0.25f, 1, 1, 1, 0.25f, 1, 1, 1, 0.25f};
        s.rect(1, 0, 0, 64, 64, 0.5f, white_quarter, 4, nullptr, false, true, BlendMode::Alpha);
        const Framebuffer fb = render_reference(s.f);
        const float d = 128.0f / 255.0f;
        const float r = blend_channel(BlendMode::Alpha, 1.0f, 0.25f, d);
        const float a = blend_channel(BlendMode::Alpha, 0.25f, 0.25f, 1.0f);
        CHECK(fb.color[fb.index(20, 20)] == pack(r, r, r, a));
        // Blending against the unquantized 0.5 would give another byte (159 against 160): the target is RGBA8.
        CHECK(pack(blend_channel(BlendMode::Alpha, 1.0f, 0.25f, 0.5f), 0, 0, 0) != pack(r, 0, 0, 0));
        CHECK((fb.color[fb.index(20, 20)] & 0xFFu) == 160u);
    }

    // ---- two instances, offsets into attrs, trust_full_tiles --------------------------------------------------
    {
        Scene s;
        s.f.shaders.push_back(passthrough4());
        ShaderBuilder sb(1);                                                  // grey from one varying
        const uint32_t out = sb.alloc4();
        sb.emit_to(kOpMov, out, sb.in(0));
        sb.emit_to(kOpMov, out + 1, sb.in(0));
        sb.emit_to(kOpMov, out + 2, sb.in(0));
        sb.emit_to(kOpMov, out + 3, sb.constant(1.0f));
        s.f.shaders.push_back(sb.finish(out));
        s.rect(2, 0, 0, 64, 32, 0.5f, {0, 1, 1, 0}, 1);
        s.rect(1, 0, 32, 64, 64, 0.5f, {0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1}, 4);
        ReferenceOptions all;
        all.trust_full_tiles = false;
        const Framebuffer a = render_reference(s.f), b = render_reference(s.f, all);
        CHECK(a.color == b.color && a.depth == b.depth && a.id == b.id);
        CHECK(a.color[a.index(0, 10)] != a.color[a.index(63, 10)]);
        CHECK(a.color[a.index(2, 50)] != a.color[a.index(60, 50)]);
    }

    // ---- bad frames are rejected --------------------------------------------------------------------------------
    {
        Scene s;
        s.f.shaders.push_back(passthrough4());
        s.rect(2, 0, 0, 64, 64, 0.5f, std::vector<float>(16, 0.0f), 4);       // shader 2 does not exist
        bool threw = false;
        try { setup_frame(s.f); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        Scene t;
        t.f.shaders.push_back(passthrough4());
        t.rect(1, 0, 0, 64, 64, 0.5f, std::vector<float>(16, 0.0f), 4);
        t.f.attrs.resize(t.f.attrs.size() - 3);                                // varyings run past the array
        threw = false;
        try { setup_frame(t.f); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        Scene u;
        ShaderProgram bad = passthrough4();
        bad.code[0].w[1] = 9;                                                  // reads a register nothing wrote
        u.f.shaders.push_back(bad);
        u.rect(1, 0, 0, 64, 64, 0.5f, std::vector<float>(16, 0.0f), 4);
        threw = false;
        try { setup_frame(u.f); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }

    return ttr_test::finish();
}
