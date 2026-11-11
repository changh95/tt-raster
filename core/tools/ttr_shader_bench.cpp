// SPDX-License-Identifier: Apache-2.0
//
// Cost of the shader interpreter on the device (docs/PERF.md): a full-screen quad at 1920 x 1088 with a straight-line program
// of K instructions of one kind, K = 0, 8, 32, 128; reports the device time of the frame and what that is per instruction and
// pixel vector. Instruction kinds: Mad (the common one), Mul, Rcp, Ex2, Sin, Slt, and Frc (the number of the opcode: a Mad/Frc chain).
// --compiled: for the Mad and Mad/Frc chains, run the same program as straight SFPI built into the kernel instead of the interpreter
// (a timing experiment: what a shader compiler could gain, docs/EVALUATION_GL_VULKAN.md).
//
//   scripts/run.sh <node> build/tools/ttr_shader_bench [--runs N]

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "tensix/tensix_renderer.h"
#include "ttraster/reference.h"

using namespace ttr;
using namespace ttr::shader;

namespace {

bool g_regs_only = false;     // --regs: the independent Mad has register operands only (no constant: SFPLOADI with a runtime immediate)
bool g_compiled = false;      // --compiled: the chain is compiled into the kernel (RASTER_EXP_SH_COMPILED_K), not interpreted
bool g_independent = false;   // --indep: instruction i reads only the inputs and writes one of 8 registers (no dependency chain)

ShaderProgram make_chain(uint32_t op, int k) {
    ShaderBuilder sb(2);   // u, v
    const uint32_t out = sb.alloc4();
    const uint32_t acc = sb.alloc();
    uint32_t extra[8];
    for (auto& e : extra) e = sb.alloc();
    sb.emit_to(kOpMov, acc, sb.in(0));
    for (auto e : extra) sb.emit_to(kOpMov, e, sb.in(1));
    const ShaderBuilder::Src z{0};
    for (int i = 0; i < k; i++) {
        if (g_independent && op == kOpMad) {
            const uint32_t d = extra[i % 8];
            if (g_regs_only) sb.emit_to(kOpMad, d, sb.in(0), sb.in(1), sb.in(1));
            else sb.emit_to(kOpMad, d, sb.in(0), sb.constant(0.9f), sb.in(1));
            continue;
        }
        switch (op) {
            case kOpMad: sb.emit_to(kOpMad, acc, sb.reg(acc), sb.constant(0.9f), sb.in(1)); break;
            case kOpMul: sb.emit_to(kOpMul, acc, sb.reg(acc), sb.constant(0.99f)); break;
            case kOpRcp: sb.emit_to(kOpRcp, acc, sb.reg(acc)); break;
            case kOpEx2: sb.emit_to(kOpEx2, acc, sb.reg(acc)); break;
            case kOpSin: sb.emit_to(kOpSin, acc, sb.reg(acc)); break;
            case kOpSlt: sb.emit_to(kOpSlt, acc, sb.reg(acc), sb.in(1)); break;
            case kOpFrc:   // Mad, Frc, Mad, Frc ... (the chain of the OpenGL benchmark, api/gl/tests/gl_bench.c): k counts instructions
                if (i % 2 == 0) sb.emit_to(kOpMad, acc, sb.reg(acc), sb.constant(1.37f), sb.in(1));
                else sb.emit_to(kOpFrc, acc, sb.reg(acc));
                break;
            default: break;
        }
    }
    sb.emit_to(kOpMov, out, sb.reg(acc), z, z, true);
    sb.emit_to(kOpMov, out + 1, sb.in(1));
    sb.emit_to(kOpMov, out + 2, sb.constant(0.2f));
    sb.emit_to(kOpMov, out + 3, sb.constant(1.0f));
    return sb.finish(out);
}

}  // namespace

int main(int argc, char** argv) {
    int runs = 20;
    std::string only;
    std::map<std::string, std::string> defines;   // extra kernel defines (timing experiments)
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--runs" && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (a == "--op" && i + 1 < argc) only = argv[++i];
        else if (a == "--indep") g_independent = true;
        else if (a == "--compiled") g_compiled = true;
        else if (a == "--regs") g_independent = g_regs_only = true;
        else if (a == "--define" && i + 1 < argc) {
            const std::string kv = argv[++i];
            const size_t eq = kv.find('=');
            defines[kv.substr(0, eq)] = eq == std::string::npos ? "1" : kv.substr(eq + 1);
        }
    }
    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    TensixRenderer renderer(mesh);
    const auto cores = renderer.worker_cores();
    const uint32_t W = 1920, H = 1088;
    std::printf("%-5s %5s %12s %12s %20s\n", "op", "K", "device ms", "ms vs K=0", "ns per instr per vector (busiest core)");
    for (uint32_t op : {kOpMad, kOpMul, kOpRcp, kOpEx2, kOpSin, kOpSlt, kOpFrc}) {
        if (!only.empty() && std::to_string(op) != only) continue;
        double base = 0.0;
        for (int k : {0, 8, 32, 128}) {
            Frame f;
            f.width = W;
            f.height = H;
            f.shaders.push_back(make_chain(op, k));
            const float x[4] = {0, static_cast<float>(W), static_cast<float>(W), 0}, y[4] = {0, 0, static_cast<float>(H), static_cast<float>(H)};
            const float u[4][2] = {{0.1f, 0.5f}, {0.9f, 0.5f}, {0.9f, 0.9f}, {0.1f, 0.9f}};
            for (const auto& idx : {std::array<int, 3>{0, 1, 2}, std::array<int, 3>{0, 2, 3}}) {
                Triangle tri;
                tri.shader = 1;
                tri.attr_offset = static_cast<uint32_t>(f.attrs.size());
                for (int j = 0; j < 3; j++) {
                    tri.v[j] = Vertex{};
                    tri.v[j].x = x[idx[j]];
                    tri.v[j].y = y[idx[j]];
                    tri.v[j].z = 0.5f;
                    tri.v[j].rhw = 1.0f;
                    f.attrs.push_back(u[idx[j]][0]);
                    f.attrs.push_back(u[idx[j]][1]);
                }
                f.triangles.push_back(tri);
            }
            const SetupResult setup = setup_frame(f);
            const TileBins bins = bin_triangles(setup, f.tiles_x(), f.tiles_y());
            const auto plan = plan_row_major(cores, f.tiles_x(), f.tiles_y());
            if (k == 0) std::printf("(%zu tiles, %llu triangle-tile records)\n", static_cast<size_t>(f.tiles_x()) * f.tiles_y(), static_cast<unsigned long long>(bins.entries));
            DeviceOptions opt;
            opt.debug_planes = false;
            opt.defines = defines;
            if (g_compiled) {
                opt.defines["RASTER_EXP_SH_COMPILED_K"] = std::to_string(k);
                opt.defines["RASTER_EXP_SH_COMPILED_KIND"] = op == kOpFrc ? "1" : "0";
            }
            renderer.render(f, setup, bins, plan, opt);   // warm-up (kernel build)
            std::vector<double> t;
            for (int r = 0; r < runs; r++) t.push_back(renderer.render(f, setup, bins, plan, opt).times.device_ms);
            std::sort(t.begin(), t.end());
            const double med = t[t.size() / 2];
            if (k == 0) base = med;
            // 2040 tiles over 120 cores: the busiest core draws 17 tiles, one of them on the diagonal of the quad (two records, one per
            // triangle; every other tile has one record, 2132 records for 2040 tiles, printed above), and a record is 32 vectors, all full.
            // (Until 2026-11-11 this assumed two records per tile and so reported times 1.9 times too small.)
            const double records_per_core = 18.0;
            const double per = k > 0 ? (med - base) * 1e6 / (records_per_core * 32 * k) : 0.0;
            std::printf("%-5u %5d %12.3f %12.3f %20.1f\n", op, k, med, med - base, per);
        }
    }
    mesh->close();
    return 0;
}
