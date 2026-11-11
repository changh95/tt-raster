// SPDX-License-Identifier: Apache-2.0
//
// Cost of the shader interpreter on the device (docs/PERF.md): a full-screen quad at 1920 x 1088 with a straight-line program
// of K instructions of one kind, K = 0, 8, 32, 128; reports the device time of the frame and what that is per instruction and
// pixel vector. Instruction kinds: Mad (the common one), Mul, Rcp, Ex2, Sin, Slt.
//
//   scripts/run.sh <node> build/tools/ttr_shader_bench [--runs N]

#include <algorithm>
#include <array>
#include <chrono>
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

ShaderProgram make_chain(uint32_t op, int k) {
    ShaderBuilder sb(2);   // u, v
    const uint32_t out = sb.alloc4();
    const uint32_t acc = sb.alloc();
    sb.emit_to(kOpMov, acc, sb.in(0));
    const ShaderBuilder::Src z{0};
    for (int i = 0; i < k; i++) {
        switch (op) {
            case kOpMad: sb.emit_to(kOpMad, acc, sb.reg(acc), sb.constant(0.9f), sb.in(1)); break;
            case kOpMul: sb.emit_to(kOpMul, acc, sb.reg(acc), sb.constant(0.99f)); break;
            case kOpRcp: sb.emit_to(kOpRcp, acc, sb.reg(acc)); break;
            case kOpEx2: sb.emit_to(kOpEx2, acc, sb.reg(acc)); break;
            case kOpSin: sb.emit_to(kOpSin, acc, sb.reg(acc)); break;
            case kOpSlt: sb.emit_to(kOpSlt, acc, sb.reg(acc), sb.in(1)); break;
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
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--runs" && i + 1 < argc) runs = std::atoi(argv[++i]);
    }
    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    TensixRenderer renderer(mesh);
    const auto cores = renderer.worker_cores();
    const uint32_t W = 1920, H = 1088;
    std::printf("%-5s %5s %12s %12s %20s\n", "op", "K", "device ms", "ms vs K=0", "ns per instr per vector (busiest core)");
    for (uint32_t op : {kOpMad, kOpMul, kOpRcp, kOpEx2, kOpSin, kOpSlt}) {
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
            DeviceOptions opt;
            opt.debug_planes = false;
            renderer.render(f, setup, bins, plan, opt);   // warm-up (kernel build)
            std::vector<double> t;
            for (int r = 0; r < runs; r++) t.push_back(renderer.render(f, setup, bins, plan, opt).times.device_ms);
            std::sort(t.begin(), t.end());
            const double med = t[t.size() / 2];
            if (k == 0) base = med;
            // 2040 tiles over 120 cores: the busiest core draws 17 tiles; every tile has one record per triangle (2), and a record is
            // 32 vectors, all full.
            const double tiles_per_core = 17.0;
            const double per = k > 0 ? (med - base) * 1e6 / (tiles_per_core * 2 * 32 * k) : 0.0;
            std::printf("%-5u %5d %12.3f %12.3f %20.1f\n", op, k, med, med - base, per);
        }
    }
    mesh->close();
    return 0;
}
