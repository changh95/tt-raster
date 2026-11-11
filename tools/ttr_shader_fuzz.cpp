// SPDX-License-Identifier: Apache-2.0
//
// Random-program test of the shader interpreter on the Tensix device (docs/FINDINGS.md, "Fragment shaders").
//
//   scripts/run.sh <node> build/tools/ttr_shader_fuzz [--mode exact|smooth] [--iters N] [--seed S] [--tris M] [--blend] [--verbose]
//
// Each iteration builds one frame of random triangles (large, small, slivers, partly off screen, perspective and linear
// varyings, random depth so that triangles overlap, optional own depth and blend state, a few fixed-function triangles
// mixed in; on some frames viewports that clip every triangle against a rectangle, and face culling) that use up to a few dozen random shader programs, renders it on the device and with the CPU reference and
// compares colour, depth and triangle id of every pixel.
//
// Two modes, because the device evaluates the interpolated planes with a multiply-add that is not exactly fused (an input can
// differ from the CPU's by one ulp), and the discontinuous instructions (Frc, Flr, the compares, Cmp, Sel, Kill) turn one ulp
// into a whole colour:
//   exact   (default)  All 13 instructions that are exact by definition (Mov, Add, Mul, Min, Max, Frc, Flr, the compares, Cmp,
//                      Sel, Kill) with random operands, constants (including -0), negate / absolute value modifiers,
//                      saturation, destinations anywhere in the register file (inputs included), up to 61 registers, 32
//                      varyings, and up to several hundred instructions. The varyings are constant over a triangle and
//                      linear (a constant plane evaluates exactly) and the programs read only x and y of fragcoord (pixel
//                      centres, exact), so the device must agree with the CPU bit for bit: any difference is a bug in the
//                      interpreter (operand decode, register file, lane predicate, constants, program table, extension
//                      records), not rounding.
//   smooth             Varyings that really vary (linear and perspective), fragcoord z and 1/w readable, but only the continuous
//                      instructions (Mov, Add, Mul, Mad, Min, Max), so that one ulp of input stays one ulp; a pixel may differ
//                      by one level (counted), never more.
// Programs that produce a NaN, an infinity or a subnormal on sampled inputs are discarded, because there the device may
// legitimately differ (shader.h, run_shader). Pixels whose depth test is ambiguous in the reference (docs/FINDINGS.md, V10)
// are skipped; depth must agree within 4 ulp or 2e-6 plus a quarter of the rounding of a term of 16 pixels times the plane's gradient (the same multiply-add). Blending uses multiply-adds that are not exactly fused,
// so with --blend one level of difference in a channel is accepted (and counted) in both modes.
//
// Exit code 0 only if every pixel of every frame agrees. A failure prints the seed: --seed S --iters 1 repeats it.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tensix/tensix_renderer.h"
#include "ttraster/reference.h"

using namespace ttr;
using namespace ttr::shader;

namespace {

struct Rng {
    uint64_t s;
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
    double range(double a, double b) { return a + (b - a) * uniform(); }
    uint32_t below(uint32_t n) { return static_cast<uint32_t>(next() % n); }
    bool chance(double p) { return uniform() < p; }
};

constexpr uint32_t kExactOps[] = {kOpMov, kOpAdd, kOpMul, kOpMin, kOpMax, kOpFrc, kOpFlr, kOpSlt, kOpSge, kOpSeq, kOpSne, kOpCmp, kOpSel};
// Relative weights of kExactOps: arithmetic and the bounded functions (Frc, Min, Max) dominate so that values stay finite.
constexpr uint32_t kOpWeights[] = {6, 14, 14, 6, 6, 8, 3, 3, 3, 2, 2, 6, 6};
// The continuous subset used by the smooth mode: Mov, Add, Mul, Mad, Min, Max.
constexpr uint32_t kSmoothOps[] = {kOpMov, kOpAdd, kOpMul, kOpMad, kOpMin, kOpMax};
constexpr uint32_t kSmoothWeights[] = {6, 14, 14, 10, 6, 6};

float random_constant(Rng& rng) {
    static const float special[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.25f, 2.0f, 3.0f, -3.0f, 0.1f, 0.7f, 1.5f, 7.5f, 100.0f, 0.001f, 0.33333334f};
    if (rng.chance(0.45)) return special[rng.below(sizeof special / sizeof special[0])];
    return static_cast<float>(rng.chance(0.5) ? rng.range(-4.0, 4.0) : rng.range(0.0, 1.0));
}

struct ProgramBuilder {
    Rng& rng;
    const bool smooth;
    ShaderProgram p;
    std::vector<uint8_t> defined;   // registers that may be read: written, or an input the mode allows
    std::vector<uint32_t> recent;   // registers written lately, picked more often as operands

    ProgramBuilder(Rng& r, bool smooth_mode) : rng(r), smooth(smooth_mode) {}

    uint32_t add_const(float v) {
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        for (size_t i = 0; i < p.consts.size(); i++) {
            uint32_t b;
            std::memcpy(&b, &p.consts[i], 4);
            if (b == bits) return static_cast<uint32_t>(i);
        }
        p.consts.push_back(v);
        return static_cast<uint32_t>(p.consts.size() - 1);
    }

    uint32_t operand() {
        uint32_t bits;
        if (rng.chance(0.28)) {
            // Reuse an existing constant often (small table), sometimes a new one.
            bits = (p.consts.empty() || rng.chance(0.4) ? add_const(random_constant(rng)) : rng.below(static_cast<uint32_t>(p.consts.size()))) | kSrcConst;
        } else if (!recent.empty() && rng.chance(0.5)) {
            bits = recent[rng.below(static_cast<uint32_t>(recent.size()))];
        } else {
            uint32_t r;
            do r = rng.below(p.num_regs); while (!defined[r]);
            bits = r;
        }
        if (rng.chance(0.15)) bits |= kSrcNeg;
        if (rng.chance(0.10)) { bits |= kSrcAbs; }
        return bits;
    }

    void emit(uint32_t op, uint32_t dst, bool sat) {
        ShaderInstr in;
        in.w[0] = make_word0(op, sat ? kFlagSat : 0u, op_writes(op) ? dst : 0u);
        const uint32_t ns = op_sources(op);
        for (uint32_t s = 0; s < ns; s++) in.w[1 + s] = operand();
        p.code.push_back(in);
        if (op_writes(op)) {
            defined[dst] = 1;
            recent.push_back(dst);
            if (recent.size() > 8) recent.erase(recent.begin());
        }
    }

    ShaderProgram build(uint32_t length) {
        const uint32_t n_in = rng.chance(0.1) ? 1 + rng.below(kMaxInputs) : 1 + rng.below(12);
        const bool fc = rng.chance(0.3);
        const uint32_t fixed = n_in + (fc ? 4 : 0);
        const uint32_t max_regs = std::min<uint32_t>(kMaxRegs, fixed + 4 + 40);
        p.num_inputs = n_in;
        p.num_regs = rng.chance(0.25) ? kMaxRegs : std::min<uint32_t>(kMaxRegs, fixed + 4 + rng.below(max_regs - fixed - 3));
        p.fragcoord_reg = fc ? static_cast<int32_t>(n_in) : -1;
        p.out_reg = rng.below(p.num_regs - 3);
        defined.assign(p.num_regs, 0);
        for (uint32_t i = 0; i < fixed; i++) defined[i] = 1;
        if (fc && !smooth) defined[n_in + 2] = defined[n_in + 3] = 0;   // z and 1/w are interpolated planes: not exact
        const uint32_t kill_at = !smooth && rng.chance(0.3) ? rng.below(length) : ~0u;
        const uint32_t kill_at2 = !smooth && rng.chance(0.1) ? rng.below(length) : ~0u;
        for (uint32_t n = 0; n < length; n++) {
            if (n == kill_at || n == kill_at2) {
                emit(kOpKill, 0, false);
                continue;
            }
            const uint32_t* ops = smooth ? kSmoothOps : kExactOps;
            const uint32_t* weights = smooth ? kSmoothWeights : kOpWeights;
            const uint32_t count = smooth ? sizeof kSmoothOps / sizeof kSmoothOps[0] : sizeof kExactOps / sizeof kExactOps[0];
            uint32_t total = 0;
            for (uint32_t i = 0; i < count; i++) total += weights[i];
            uint32_t pick = rng.below(total), k = 0;
            while (pick >= weights[k]) pick -= weights[k++];
            emit(ops[k], rng.below(p.num_regs), rng.chance(0.1));
        }
        // The colour: every channel from some register, mostly through Frc so that it is not just 0 or 1.
        for (uint32_t k = 0; k < 4; k++) {
            ShaderInstr in;
            const bool frc = !smooth && rng.chance(0.7);
            in.w[0] = make_word0(frc ? kOpFrc : kOpMov, frc ? 0u : kFlagSat, p.out_reg + k);
            if (k == 3 && rng.chance(0.4)) {
                in.w[0] = make_word0(kOpMov, 0, p.out_reg + k);
                in.w[1] = add_const(1.0f) | kSrcConst;   // opaque
            } else {
                uint32_t r;
                do r = rng.below(p.num_regs); while (!defined[r]);
                in.w[1] = r | (rng.chance(0.1) ? kSrcNeg : 0u);
            }
            p.code.push_back(in);
            defined[p.out_reg + k] = 1;
            if (smooth && instr_op(in.w[0]) == kOpMov && instr_flags(in.w[0]) == kFlagSat) {
                // 0.5 + 0.2 * value, saturated: keeps the colour inside [0, 1] more often than clamping the raw value.
                ShaderInstr m, a;
                m.w[0] = make_word0(kOpMul, 0, p.out_reg + k);
                m.w[1] = p.out_reg + k;
                m.w[2] = add_const(0.2f) | kSrcConst;
                a.w[0] = make_word0(kOpAdd, kFlagSat, p.out_reg + k);
                a.w[1] = p.out_reg + k;
                a.w[2] = add_const(0.5f) | kSrcConst;
                p.code.push_back(m);
                p.code.push_back(a);
            }
        }
        return p;
    }
};

// True if the program produces a NaN, infinity or subnormal for a sample of the inputs it can see.
bool hazardous(const ShaderProgram& p, Rng& rng, uint32_t width, uint32_t height) {
    std::vector<float> r(p.num_regs), in(p.num_inputs);
    for (int s = 0; s < 400; s++) {
        for (float& v : in) v = static_cast<float>(rng.range(-2.0, 2.0));
        const float fc[4] = {static_cast<float>(rng.range(0, width)), static_cast<float>(rng.range(0, height)),
                             static_cast<float>(rng.uniform()), static_cast<float>(rng.range(0.4, 1.0))};
        bool h = false;
        run_shader(p, in.data(), fc, r.data(), &h);
        if (h) return true;
    }
    return false;
}

const char* op_name(uint32_t op) {
    static const char* names[] = {"nop", "mov", "add", "mul", "mad", "min", "max", "frc", "flr", "rcp", "rsq", "sqrt", "ex2", "lg2", "sin", "cos", "slt", "sge", "seq", "sne", "cmp", "sel", "kill"};
    return op < sizeof names / sizeof names[0] ? names[op] : "?";
}

std::string operand_text(const ShaderProgram& p, uint32_t w) {
    char buf[64];
    if (w & kSrcConst) std::snprintf(buf, sizeof buf, "%s%s#%.9g%s", (w & kSrcNeg) ? "-" : "", (w & kSrcAbs) ? "|" : "", p.consts[w & kSrcIndexMask], (w & kSrcAbs) ? "|" : "");
    else std::snprintf(buf, sizeof buf, "%s%sr%u%s", (w & kSrcNeg) ? "-" : "", (w & kSrcAbs) ? "|" : "", w & kSrcIndexMask, (w & kSrcAbs) ? "|" : "");
    return buf;
}

// The program with the CPU interpreter's value after every instruction, for one pixel (a copy of the exact-instruction part of
// run_shader: it only prints, the comparison is done by the real interpreter).
void trace_program(const ShaderProgram& p, const float* inputs, const float* fragcoord) {
    std::vector<float> r(p.num_regs, 0.0f);
    for (uint32_t k = 0; k < p.num_inputs; k++) r[k] = inputs[k];
    if (p.fragcoord_reg >= 0) for (int k = 0; k < 4; k++) r[p.fragcoord_reg + k] = fragcoord[k];
    std::printf("   program: %u inputs, %u regs, fragcoord r%d, colour r%u..r%u, %zu instructions, %zu constants\n", p.num_inputs, p.num_regs, p.fragcoord_reg, p.out_reg, p.out_reg + 3, p.code.size(), p.consts.size());
    for (uint32_t k = 0; k < p.num_inputs; k++) std::printf("     in r%u = %.9g\n", k, inputs[k]);
    for (size_t n = 0; n < p.code.size(); n++) {
        const ShaderInstr& in = p.code[n];
        const uint32_t op = instr_op(in.w[0]);
        float v[3] = {0, 0, 0};
        for (uint32_t k = 0; k < op_sources(op); k++) {
            const uint32_t w = in.w[1 + k];
            float x = (w & kSrcConst) ? p.consts[w & kSrcIndexMask] : r[w & kSrcIndexMask];
            if (w & kSrcAbs) x = std::fabs(x);
            if (w & kSrcNeg) x = -x;
            v[k] = x;
        }
        float d = 0;
        switch (op) {
            case kOpMov: d = v[0]; break;
            case kOpAdd: d = v[0] + v[1]; break;
            case kOpMul: d = v[0] * v[1]; break;
            case kOpMin: d = v[1] < v[0] ? v[1] : v[0]; break;
            case kOpMax: d = v[1] > v[0] ? v[1] : v[0]; break;
            case kOpFrc: d = v[0] - std::floor(v[0]); break;
            case kOpFlr: d = std::floor(v[0]); break;
            case kOpSlt: d = v[0] < v[1] ? 1.0f : 0.0f; break;
            case kOpSge: d = v[0] >= v[1] ? 1.0f : 0.0f; break;
            case kOpSeq: d = v[0] == v[1] ? 1.0f : 0.0f; break;
            case kOpSne: d = v[0] != v[1] ? 1.0f : 0.0f; break;
            case kOpCmp: d = v[0] < 0.0f ? v[1] : v[2]; break;
            case kOpSel: d = v[0] != 0.0f ? v[1] : v[2]; break;
            default: break;
        }
        const bool sat = instr_flags(in.w[0]) & kFlagSat;
        if (sat) d = !(d > 0.0f) ? 0.0f : (d > 1.0f ? 1.0f : d);
        std::printf("     %3zu: %-4s", n, op_name(op));
        if (op_writes(op)) std::printf(" r%-2u", instr_dst(in.w[0]));
        else std::printf("     ");
        for (uint32_t k = 0; k < op_sources(op); k++) std::printf(" %s(%.9g)", operand_text(p, in.w[1 + k]).c_str(), v[k]);
        if (op_writes(op)) { std::printf("%s -> %.9g", sat ? " sat" : "", d); r[instr_dst(in.w[0])] = d; }
        else std::printf(" -> %s", v[0] < 0.0f ? "KILL" : "keep");
        std::printf("\n");
    }
}

struct Stats {
    uint64_t pixels = 0, ambiguous = 0, exact = 0, off_by_one = 0, bad = 0, depth_bad = 0, id_bad = 0;
    float max_depth_err = 0.0f;
};

}  // namespace

int main(int argc, char** argv) {
    uint32_t iters = 40, tris = 220;
    uint64_t seed0 = 1;
    bool blend = false, verbose = false, smooth = false, dump = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--mode" && i + 1 < argc) {
            const std::string m = argv[++i];
            if (m != "exact" && m != "smooth") { std::fprintf(stderr, "--mode is exact or smooth\n"); return 2; }
            smooth = m == "smooth";
            continue;
        }
        if (a == "--iters" && i + 1 < argc) iters = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (a == "--seed" && i + 1 < argc) seed0 = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--tris" && i + 1 < argc) tris = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (a == "--blend") blend = true;
        else if (a == "--verbose") verbose = true;
        else if (a == "--dump") dump = true;
        else {
            std::fprintf(stderr, "usage: ttr_shader_fuzz [--mode exact|smooth] [--iters N] [--seed S] [--tris M] [--blend] [--verbose]\n");
            return 2;
        }
    }
    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    TensixRenderer renderer(mesh);
    const auto cores = renderer.worker_cores();

    Stats total;
    uint32_t failed_frames = 0;
    std::printf("%-6s %9s %7s %9s %9s %9s %9s %8s %8s %6s\n", "seed", "programs", "tris", "pixels", "ambig", "exact", "off by 1", "bad", "depth", "id");
    for (uint32_t it = 0; it < iters; it++) {
        const uint64_t seed = seed0 + it;
        Rng rng{seed * 0x1234567ull + 99};
        Frame f;
        f.width = 32 * (4 + rng.below(9));    // 128 .. 384
        f.height = 32 * (3 + rng.below(6));   // 96 .. 224
        f.clear_color[0] = static_cast<float>(rng.uniform());
        f.clear_color[1] = static_cast<float>(rng.uniform());
        f.clear_color[2] = static_cast<float>(rng.uniform());
        f.clear_color[3] = 1.0f;
        f.depth_func = DepthFunc::Less;
        if (rng.chance(0.2)) f.cull = rng.chance(0.5) ? CullMode::Positive : CullMode::Negative;
        if (rng.chance(0.3)) {   // viewports: every triangle is confined to (clipped against) one rectangle
            const uint32_t n = 2 + rng.below(5);
            for (uint32_t k = 0; k < n; k++) {
                Viewport vp;
                vp.x = static_cast<int32_t>(rng.below(f.width - 8));
                vp.y = static_cast<int32_t>(rng.below(f.height - 8));
                vp.width = static_cast<int32_t>(8 + rng.below(f.width - vp.x - 7));
                vp.height = static_cast<int32_t>(8 + rng.below(f.height - vp.y - 7));
                f.viewports.push_back(vp);
            }
        }

        // Programs, up to the size of the program table (16 pages of 1024 words, the directory and one more page of margin).
        const uint32_t want = 1 + rng.below(rng.chance(0.2) ? 40 : 12);
        uint64_t table_words = 1;
        while (f.shaders.size() < want) {
            const uint32_t length = rng.chance(0.05) ? 400 + rng.below(560) : 2 + rng.below(rng.chance(0.5) ? 40 : 200);
            ShaderProgram p;
            bool ok = false;
            for (int attempt = 0; attempt < 30 && !ok; attempt++) {
                ProgramBuilder b(rng, smooth);
                p = b.build(length);
                ok = validate_shader(p).empty() && !hazardous(p, rng, f.width, f.height);
            }
            if (!ok) continue;
            const uint64_t words = 8 + 4 * p.code.size() + p.consts.size();
            if (table_words + words > 14 * 1024) break;
            table_words += words;
            f.shaders.push_back(std::move(p));
        }
        if (f.shaders.empty()) continue;

        for (uint32_t t = 0; t < tris; t++) {
            Triangle tri;
            const double kind = rng.uniform();
            const double cx = rng.range(-20, f.width + 20), cy = rng.range(-20, f.height + 20);
            const double radius = kind < 0.55 ? rng.range(20, 120) : kind < 0.8 ? rng.range(3, 20) : kind < 0.9 ? rng.range(150, 500) : rng.range(30, 200);
            const bool sliver = kind >= 0.9;
            const bool perspective = rng.chance(0.4);   // the colour planes of fixed-function triangles; shader triangles below
            for (int k = 0; k < 3; k++) {
                const double ang = rng.range(0, 6.2831853);
                Vertex& v = tri.v[k];
                v = Vertex{};
                v.x = static_cast<float>(cx + radius * std::cos(ang));
                v.y = static_cast<float>(cy + radius * std::sin(ang) * (sliver ? 0.05 : 1.0));
                v.z = static_cast<float>(rng.uniform());
                v.rhw = perspective ? static_cast<float>(rng.range(0.4, 1.0)) : 1.0f;
                for (float& c : v.color) c = static_cast<float>(rng.uniform());
            }
            tri.perspective = perspective;
            if (rng.chance(0.1)) {
                tri.shader = 0;   // fixed function mixed with shader triangles
                tri.flat = !smooth;   // a flat colour is exact; interpolated colours are within a level
            } else {
                if (!smooth) {
                    tri.perspective = false;
                    for (Vertex& v : tri.v) v.rhw = 1.0f;
                }
                tri.shader = static_cast<uint16_t>(1 + rng.below(static_cast<uint32_t>(f.shaders.size())));
                const uint32_t n = f.shaders[tri.shader - 1].num_inputs;
                tri.attr_offset = static_cast<uint32_t>(f.attrs.size());
                std::vector<float> a(3 * n);
                for (uint32_t c = 0; c < n; c++) {
                    const bool flat = !smooth || rng.chance(0.2);
                    const float v0 = static_cast<float>(rng.range(-2, 2));
                    for (int k = 0; k < 3; k++) a[k * n + c] = flat ? v0 : static_cast<float>(rng.range(-2, 2));
                }
                f.attrs.insert(f.attrs.end(), a.begin(), a.end());
            }
            if (rng.chance(0.25)) {
                tri.own_state = true;
                tri.depth_func = static_cast<DepthFunc>(rng.below(3));
                tri.depth_write = rng.chance(0.7);
                tri.blend = blend && rng.chance(0.6) ? static_cast<BlendMode>(1 + rng.below(5)) : BlendMode::None;
            }
            if (!f.viewports.empty()) tri.viewport = rng.below(static_cast<uint32_t>(f.viewports.size()));
            f.triangles.push_back(tri);
        }

        const SetupResult setup = setup_frame(f);
        const TileBins bins = bin_triangles(setup, f.tiles_x(), f.tiles_y());
        const Framebuffer cpu = render_reference(f, setup, bins);
        const auto plan = plan_balanced(cores, all_tiles(f.tiles_x(), f.tiles_y()), bins);
        DeviceOptions opt;
        opt.debug_planes = true;
        const DeviceFrame dev = renderer.render(f, setup, bins, plan, opt);

        // The depth plane is evaluated relative to the tile centre with a multiply-add that is not exactly fused: the error is about an
        // ulp of the largest term, 16 pixels times the gradient, so a steep sliver (gradient 10 per pixel) is allowed more than 2e-6.
        std::vector<double> steepness(f.triangles.size(), 0.0);
        for (const SetupTriangle& st : setup.tris) steepness[st.source] = std::max(steepness[st.source], std::fabs(st.depth.gx) + std::fabs(st.depth.gy));
        Stats s;
        uint32_t first_bad_x = 0, first_bad_y = 0;
        bool have_bad = false;
        for (uint32_t y = 0; y < f.height; y++) {
            for (uint32_t x = 0; x < f.width; x++) {
                const size_t i = cpu.index(x, y);
                s.pixels++;
                if (cpu.ambiguous[i]) {
                    s.ambiguous++;
                    continue;
                }
                int worst = 0;
                for (int ch = 0; ch < 4; ch++) {
                    const int a = static_cast<int>((cpu.color[i] >> (8 * ch)) & 255u), b = static_cast<int>((dev.fb.color[i] >> (8 * ch)) & 255u);
                    worst = std::max(worst, std::abs(a - b));
                }
                if (worst == 0) s.exact++;
                else if (worst == 1 && (blend || smooth)) s.off_by_one++;
                else {
                    s.bad++;
                    if (!have_bad) { have_bad = true; first_bad_x = x; first_bad_y = y; }
                }
                if (dev.fb.id[i] != cpu.id[i]) s.id_bad++;
                if (dev.fb.id[i] == cpu.id[i] && dev.fb.id[i] != 0) {
                    uint32_t da, db;
                    std::memcpy(&da, &cpu.depth[i], 4);
                    std::memcpy(&db, &dev.fb.depth[i], 4);
                    // The plane is evaluated with a multiply-add that is not exactly fused: an error of an ulp of the largest term
                    // (about 1.2e-7 for depth in [0, 1], several times that on the steep planes of slivers), which is many ulps of a depth close to 0. Accept 4 ulp or 2e-6 plus a share of the rounding of a term that large (the largest difference seen: 3.6e-6 on a plane with a gradient of 11 per pixel).
                    const bool close_ulp = (da > db ? da - db : db - da) <= 4;
                    s.max_depth_err = std::max(s.max_depth_err, std::fabs(cpu.depth[i] - dev.fb.depth[i]));
                    const double tolerance = 2e-6 + 0.25 * 1.19e-7 * 16.0 * steepness[dev.fb.id[i] - 1];
                    if (!close_ulp && std::fabs(cpu.depth[i] - dev.fb.depth[i]) > tolerance) {
                        if (!s.depth_bad++) {
                            double gx = 0, gy = 0;
                            for (const SetupTriangle& st : setup.tris)
                                if (cpu.id[i] && st.source == cpu.id[i] - 1) { gx = st.depth.gx; gy = st.depth.gy; break; }
                            std::printf("   first depth difference at (%u, %u): cpu %.9g device %.9g, triangle %u, depth gradient %.3g, %.3g per pixel\n", x, y,
                                        static_cast<double>(cpu.depth[i]), static_cast<double>(dev.fb.depth[i]), cpu.id[i], gx, gy);
                        }
                    }
                }
            }
        }
        std::printf("%-6llu %9zu %7zu %9llu %9llu %9llu %9llu %8llu %8llu %6llu\n", static_cast<unsigned long long>(seed), f.shaders.size(), f.triangles.size(),
                    static_cast<unsigned long long>(s.pixels), static_cast<unsigned long long>(s.ambiguous), static_cast<unsigned long long>(s.exact),
                    static_cast<unsigned long long>(s.off_by_one), static_cast<unsigned long long>(s.bad), static_cast<unsigned long long>(s.depth_bad),
                    static_cast<unsigned long long>(s.id_bad));
        if (have_bad || s.depth_bad || s.id_bad) {
            failed_frames++;
            if (have_bad) {
                const size_t i = cpu.index(first_bad_x, first_bad_y);
                const uint32_t tri_id = cpu.id[i];
                std::printf("   first bad pixel (%u, %u): cpu %08x device %08x, cpu id %u device id %u", first_bad_x, first_bad_y, cpu.color[i],
                            dev.fb.color[i], tri_id, dev.fb.id[i]);
                if (tri_id != 0 && tri_id <= f.triangles.size()) std::printf(" (program %u of %zu)", f.triangles[tri_id - 1].shader, f.shaders.size());
                std::printf("\n");
                if (dump && tri_id != 0 && f.triangles[tri_id - 1].shader != 0) {
                    const Triangle& tr = f.triangles[tri_id - 1];
                    const ShaderProgram& pg = f.shaders[tr.shader - 1];
                    // Constant varyings (exact mode): vertex 0's values are the pixel's. Smooth mode: the dump is only indicative.
                    const float fcv[4] = {first_bad_x + 0.5f, first_bad_y + 0.5f, 0.0f, 1.0f};
                    trace_program(pg, f.attrs.data() + tr.attr_offset, fcv);
                }
            }
            std::printf("   repeat with: --mode %s --seed %llu --iters 1%s\n", smooth ? "smooth" : "exact", static_cast<unsigned long long>(seed), blend ? " --blend" : "");
        }
        total.pixels += s.pixels;
        total.ambiguous += s.ambiguous;
        total.exact += s.exact;
        total.off_by_one += s.off_by_one;
        total.bad += s.bad;
        total.depth_bad += s.depth_bad;
        total.id_bad += s.id_bad;
        total.max_depth_err = std::max(total.max_depth_err, s.max_depth_err);
        if (verbose) std::fflush(stdout);
    }
    mesh->close();
    std::printf("total: %llu pixels, %llu ambiguous skipped, %llu exact, %llu off by one level, %llu bad colour, %llu depth, %llu id; %u of %u frames failed\n",
                static_cast<unsigned long long>(total.pixels), static_cast<unsigned long long>(total.ambiguous), static_cast<unsigned long long>(total.exact),
                static_cast<unsigned long long>(total.off_by_one), static_cast<unsigned long long>(total.bad), static_cast<unsigned long long>(total.depth_bad),
                static_cast<unsigned long long>(total.id_bad), failed_frames, iters);
    std::printf("largest depth difference %.3g\n", static_cast<double>(total.max_depth_err));
    std::printf("%s\n", failed_frames == 0 ? "OK" : "FAILED");
    return failed_frames == 0 ? 0 : 1;
}
