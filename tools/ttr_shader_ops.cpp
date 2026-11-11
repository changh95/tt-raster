// SPDX-License-Identifier: Apache-2.0
//
// Accuracy of every shader instruction on the Tensix device (docs/FINDINGS.md, "Fragment shaders").
//
//   scripts/run.sh <node> build/tools/ttr_shader_ops [op|all] [--tests N]
//
// One frame per instruction. Test i is one pixel: a quad of two triangles whose five varyings are constant over the pixel:
// the operands a, b, c, a reference value computed on the host in double precision, and a scale. The shader computes
// d = op(a, b, c) and writes 0.5 + (d - ref) * scale to the colour, with scale chosen so that the 8-bit colour spans
// +-64 units in the last place (u = 2^-23 * max(|ref|, 1)): the error of the device is read back at a resolution of about
// u / 2, for every pixel independently. Instructions that are exact by definition must come back at 0.
// Reported per instruction: the largest error in u, how many results were out of the +-64 u range, and the device against the
// CPU interpreter (which uses libm): largest difference in 8-bit levels of the amplified error image.
// Exit code 0 only if every instruction is within its limit.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
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
    // sign * 10^(uniform in [lo, hi]), the sign random
    double log_signed(double lo, double hi) { return (uniform() < 0.5 ? -1.0 : 1.0) * std::pow(10.0, range(lo, hi)); }
    double log_pos(double lo, double hi) { return std::pow(10.0, range(lo, hi)); }
};

struct OpTest {
    const char* name;
    uint32_t op;
    double limit_u;   // largest accepted error in u
    std::function<void(Rng&, double[3])> gen;
    std::function<double(const double[3])> ref;
    // Magnitude that sets the unit u (default max(|ref|, 1)); Mad: the multiply-add of the SFPU keeps the product to about
    // 2^-23 of its own size, so a cancelling sum is wrong relative to |a * b|, not to the result.
    std::function<double(const double[3], double)> mag = nullptr;
};

float f32(double x) { return static_cast<float>(x); }

std::vector<OpTest> make_tests() {
    auto any = [](Rng& r, double v[3]) { for (int k = 0; k < 3; k++) v[k] = r.log_signed(-3, 3); };
    auto first = [](Rng& r, double v[3]) { v[0] = r.log_signed(-3, 3); v[1] = v[2] = 0; };
    auto pos = [](Rng& r, double v[3]) { v[0] = r.log_pos(-4, 4); v[1] = v[2] = 0; };
    auto cmpgen = [](Rng& r, double v[3]) {
        v[0] = r.log_signed(-2, 2);
        v[1] = r.uniform() < 0.3 ? v[0] : r.log_signed(-2, 2);   // equal operands now and then
        v[2] = r.log_signed(-2, 2);
    };
    auto sel = [](Rng& r, double v[3]) {
        v[0] = r.uniform() < 0.3 ? 0.0 : r.log_signed(-2, 2);
        v[1] = r.log_signed(-2, 2);
        v[2] = r.log_signed(-2, 2);
    };
    return {
        {"Mov", kOpMov, 0.0, first, [](const double* v) { return v[0]; }},
        {"Add", kOpAdd, 0.0, any, [](const double* v) { return static_cast<double>(f32(v[0] + v[1])); }},
        {"Mul", kOpMul, 0.0, any, [](const double* v) { return static_cast<double>(f32(v[0] * v[1])); }},
        {"Mad", kOpMad, 1.5, any, [](const double* v) { return static_cast<double>(std::fmaf(f32(v[0]), f32(v[1]), f32(v[2]))); },
         [](const double* v, double ref) { return std::max({std::fabs(v[0] * v[1]), std::fabs(v[2]), std::fabs(ref), 1.0}); }},
        {"Min", kOpMin, 0.0, any, [](const double* v) { return std::min(v[0], v[1]); }},
        {"Max", kOpMax, 0.0, any, [](const double* v) { return std::max(v[0], v[1]); }},
        {"Frc", kOpFrc, 0.0, [](Rng& r, double v[3]) { v[0] = r.range(-100, 100); v[1] = v[2] = 0; },
         [](const double* v) { return static_cast<double>(f32(v[0]) - std::floor(f32(v[0]))); }},
        {"Flr", kOpFlr, 0.0, [](Rng& r, double v[3]) { v[0] = r.range(-1000, 1000); v[1] = v[2] = 0; },
         [](const double* v) { return std::floor(static_cast<double>(f32(v[0]))); }},
        {"Rcp", kOpRcp, 4.0, first, [](const double* v) { return 1.0 / f32(v[0]); }},
        {"Rsq", kOpRsq, 4.0, pos, [](const double* v) { return 1.0 / std::sqrt(f32(v[0])); }},
        {"Sqrt", kOpSqrt, 4.0, pos, [](const double* v) { return std::sqrt(f32(v[0])); }},
        {"Ex2", kOpEx2, 4.0, [](Rng& r, double v[3]) { v[0] = r.range(-30, 30); v[1] = v[2] = 0; },
         [](const double* v) { return std::exp2(static_cast<double>(f32(v[0]))); }},
        {"Lg2", kOpLg2, 3.0, pos, [](const double* v) { return std::log2(static_cast<double>(f32(v[0]))); }},
        {"Sin", kOpSin, 4.0, [](Rng& r, double v[3]) { v[0] = r.range(-50, 50); v[1] = v[2] = 0; },
         [](const double* v) { return std::sin(static_cast<double>(f32(v[0]))); }},
        {"Cos", kOpCos, 4.0, [](Rng& r, double v[3]) { v[0] = r.range(-50, 50); v[1] = v[2] = 0; },
         [](const double* v) { return std::cos(static_cast<double>(f32(v[0]))); }},
        {"Slt", kOpSlt, 0.0, cmpgen, [](const double* v) { return f32(v[0]) < f32(v[1]) ? 1.0 : 0.0; }},
        {"Sge", kOpSge, 0.0, cmpgen, [](const double* v) { return f32(v[0]) >= f32(v[1]) ? 1.0 : 0.0; }},
        {"Seq", kOpSeq, 0.0, cmpgen, [](const double* v) { return f32(v[0]) == f32(v[1]) ? 1.0 : 0.0; }},
        {"Sne", kOpSne, 0.0, cmpgen, [](const double* v) { return f32(v[0]) != f32(v[1]) ? 1.0 : 0.0; }},
        {"Cmp", kOpCmp, 0.0, any, [](const double* v) { return f32(v[0]) < 0.0f ? v[1] : v[2]; }},
        {"Sel", kOpSel, 0.0, sel, [](const double* v) { return f32(v[0]) != 0.0f ? v[1] : v[2]; }},
    };
}

ShaderProgram make_program(uint32_t op) {
    ShaderBuilder sb(5);   // a, b, c, ref, scale
    const uint32_t out = sb.alloc4();
    const uint32_t ns = op_sources(op);
    const ShaderBuilder::Src z{0};
    const uint32_t d = sb.emit(op, ns > 0 ? sb.in(0) : z, ns > 1 ? sb.in(1) : z, ns > 2 ? sb.in(2) : z);
    const uint32_t e = sb.emit(kOpAdd, sb.reg(d), -sb.in(3));
    const uint32_t f = sb.emit(kOpMul, sb.reg(e), sb.in(4));
    const uint32_t o = sb.emit(kOpAdd, sb.reg(f), sb.constant(0.5f), z, false);
    for (uint32_t k = 0; k < 3; k++) sb.emit_to(kOpMov, out + k, sb.reg(o), z, z, true);
    sb.emit_to(kOpMov, out + 3, sb.constant(1.0f));
    return sb.finish(out);
}

}  // namespace

int main(int argc, char** argv) {
    std::string which = "all";
    uint32_t n_tests = 8192;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--tests" && i + 1 < argc) n_tests = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (a[0] != '-') which = a;
        else {
            std::fprintf(stderr, "usage: ttr_shader_ops [op|all] [--tests N]\n");
            return 2;
        }
    }
    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    TensixRenderer renderer(mesh);
    const auto cores = renderer.worker_cores();
    bool all_ok = true;
    std::printf("%-5s %10s %12s %10s %14s   (error in u = 2^-23 * max(|ref|, 1), range +-64 u; Mad: max(|a*b|, |c|, |ref|, 1))\n", "op", "max err u", "out of range", "limit u",
                "vs CPU levels");

    for (const OpTest& t : make_tests()) {
        if (which != "all" && which != t.name) continue;
        const uint32_t W = 128, H = (n_tests + W - 1) / W;
        Frame f;
        f.width = W;
        f.height = (H + 31) / 32 * 32;
        f.clear_color[0] = f.clear_color[1] = f.clear_color[2] = 0.0f;
        f.shaders.push_back(make_program(t.op));
        Rng rng{0x5eed0000ull + t.op};
        std::vector<double> refs(n_tests), scales(n_tests);
        std::vector<float> inputs(3 * static_cast<size_t>(n_tests));
        for (uint32_t i = 0; i < n_tests; i++) {
            double v[3];
            t.gen(rng, v);
            for (int k = 0; k < 3; k++) {
                v[k] = static_cast<double>(f32(v[k]));
                inputs[3 * i + k] = static_cast<float>(v[k]);
            }
            refs[i] = t.ref(v);
            const double magnitude = t.mag ? t.mag(v, refs[i]) : std::max(std::fabs(refs[i]), 1.0);
            scales[i] = static_cast<double>(f32(65536.0 / magnitude));
            const float x0 = static_cast<float>(i % W), y0 = static_cast<float>(i / W);
            const float px[4] = {x0, x0 + 1, x0 + 1, x0}, py[4] = {y0, y0, y0 + 1, y0 + 1};
            const float attr[5] = {inputs[3 * i], inputs[3 * i + 1], inputs[3 * i + 2], f32(refs[i]), static_cast<float>(scales[i])};
            for (const auto& idx : {std::array<int, 3>{0, 1, 2}, std::array<int, 3>{0, 2, 3}}) {
                Triangle tri;
                tri.shader = 1;
                tri.attr_offset = static_cast<uint32_t>(f.attrs.size());
                for (int k = 0; k < 3; k++) {
                    tri.v[k] = Vertex{};
                    tri.v[k].x = px[idx[k]];
                    tri.v[k].y = py[idx[k]];
                    tri.v[k].z = 0.5f;
                    tri.v[k].rhw = 1.0f;
                    for (float a : attr) f.attrs.push_back(a);
                }
                f.triangles.push_back(tri);
            }
        }
        const SetupResult setup = setup_frame(f);
        const TileBins bins = bin_triangles(setup, f.tiles_x(), f.tiles_y());
        const Framebuffer cpu = render_reference(f, setup, bins);
        const auto plan = plan_balanced(cores, all_tiles(f.tiles_x(), f.tiles_y()), bins);
        DeviceOptions opt;
        opt.debug_planes = false;
        const DeviceFrame dev = renderer.render(f, setup, bins, plan, opt);

        double max_err = 0.0;
        uint32_t out_of_range = 0, worst_i = 0;
        int max_level_diff = 0;
        for (uint32_t i = 0; i < n_tests; i++) {
            const size_t idx = dev.fb.index(i % W, i / W);
            const int byte = static_cast<int>(dev.fb.color[idx] & 255u), cpu_byte = static_cast<int>(cpu.color[idx] & 255u);
            max_level_diff = std::max(max_level_diff, std::abs(byte - cpu_byte));
            if (byte == 0 || byte == 255) {
                out_of_range++;
                continue;
            }
            const double err = (byte - 128) / 255.0 * 128.0;   // in u
            if (std::fabs(err) > max_err) {
                max_err = std::fabs(err);
                worst_i = i;
            }
        }
        const bool ok = max_err <= t.limit_u + 0.5 && out_of_range == 0;
        all_ok &= ok;
        std::printf("%-5s %10.2f %12u %10.1f %14d  %s", t.name, max_err, out_of_range, t.limit_u, max_level_diff, ok ? "ok" : "FAILED");
        if (!ok) std::printf("   worst input (%g, %g, %g) ref %.9g", inputs[3 * worst_i], inputs[3 * worst_i + 1], inputs[3 * worst_i + 2], refs[worst_i]);
        std::printf("\n");
    }
    mesh->close();
    std::printf("%s\n", all_ok ? "OK" : "FAILED");
    return all_ok ? 0 : 1;
}
