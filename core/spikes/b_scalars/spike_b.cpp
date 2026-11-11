// SPDX-License-Identifier: Apache-2.0
//
// Spike B: runtime scalars into the SFPU (docs/PLAN.md, Phase 0.4).
//
// For one 32x32 tile in Dest, apply T triangles' masked update
//     if (a*dx + b*dy + c >= 0) acc += d
// K times, with the triangle scalars delivered by four different mechanisms, and report cycles
// measured on the math thread plus a bit-exact check against an exact int64 CPU reference.
//
//   TT_VISIBLE_DEVICES=<node> spike_b [K]
//
// Exit code 0 only if every variant matches the CPU reference exactly.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t kTile = 32;
constexpr uint32_t kElems = kTile * kTile;
constexpr uint32_t kTileBytes = kElems * sizeof(float);
constexpr uint32_t kTriangles = 64;  // per batch, must match kTrianglesPerBatch in compute.cpp
constexpr uint32_t kCoefTiles = 8;

struct Tri {
    int32_t a, b, c, d;
};

// Three classes, to also test integer exactness of the SFPU multiply-add near 2^24:
//   0: small values
//   1: large a (|a| up to 2^19, so a*dx reaches 16.2M, just under 2^24), small b, c
//   2: both a and b up to 2^17, large c up to 2^22 (all partial sums stay below 2^24)
std::vector<Tri> make_triangles(uint32_t seed) {
    std::mt19937 rng(seed);
    auto uni = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
    std::vector<Tri> t(kTriangles);
    for (uint32_t i = 0; i < kTriangles; i++) {
        switch (i % 3) {
            case 0: t[i] = {uni(-16, 16), uni(-16, 16), uni(-256, 256), uni(-8, 8)}; break;
            case 1: t[i] = {uni(-(1 << 19), 1 << 19), uni(-16, 16), uni(-4096, 4096), uni(-8, 8)}; break;
            default: t[i] = {uni(-(1 << 17), 1 << 17), uni(-(1 << 17), 1 << 17), uni(-(1 << 22), 1 << 22), uni(-8, 8)};
        }
    }
    return t;
}

// 32x32 row-major matrix whose element (r, c) holds the value of the SFPU vector that covers it,
// using the lane map from Spike A: face = (r/16)*2 + c/16, v = 2*((r%16)/4) + (c%16)&1, k = face*8 + v.
std::vector<float> coef_tile(const std::vector<Tri>& tris, uint32_t group) {
    std::vector<float> m(kElems);
    for (uint32_t r = 0; r < kTile; r++) {
        for (uint32_t c = 0; c < kTile; c++) {
            const uint32_t face = (r / 16) * 2 + c / 16;
            const uint32_t v = 2 * ((r % 16) / 4) + ((c % 16) & 1);
            const uint32_t k = face * 8 + v;
            const Tri& t = tris[group * 8 + k / 4];
            const int32_t vals[4] = {t.a, t.b, t.c, t.d};
            m[r * kTile + c] = static_cast<float>(vals[k % 4]);
        }
    }
    return tilize_nfaces(m, kTile, kTile);
}

struct Variant {
    const char* name;
    int approach;
    int tpc;  // triangles per SFPU call
    int vec;
};

struct Result {
    bool ok = false;
    uint32_t mismatches = 0;
    uint32_t cycles = 0;
    double host_ms = 0;
};

}  // namespace

int main(int argc, char** argv) {
    const uint32_t k_default = argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 64;

    const std::vector<Tri> tris = make_triangles(12345);

    // Scalar tile: raw floats (a, b, c, d) per triangle, read by the math thread straight from L1.
    std::vector<float> scal(kElems, 0.0f);
    for (uint32_t i = 0; i < kTriangles; i++) {
        scal[4 * i + 0] = static_cast<float>(tris[i].a);
        scal[4 * i + 1] = static_cast<float>(tris[i].b);
        scal[4 * i + 2] = static_cast<float>(tris[i].c);
        scal[4 * i + 3] = static_cast<float>(tris[i].d);
    }
    std::vector<float> coef;
    for (uint32_t g = 0; g < kCoefTiles; g++) {
        const auto t = coef_tile(tris, g);
        coef.insert(coef.end(), t.begin(), t.end());
    }

    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh->mesh_command_queue();

    distributed::DeviceLocalBufferConfig dram_cfg{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    auto make_buf = [&](uint32_t tiles) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = tiles * kTileBytes}, dram_cfg, mesh.get());
    };
    auto scal_buf = make_buf(1), coef_buf = make_buf(kCoefTiles), out_buf = make_buf(2);
    distributed::EnqueueWriteMeshBuffer(cq, scal_buf, scal, false);
    distributed::EnqueueWriteMeshBuffer(cq, coef_buf, coef, false);

    const std::string kdir = std::string(TT_RASTER_SPIKE_B_KERNELS);

    auto run = [&](const Variant& var, uint32_t k, uint32_t nv = 32) {
        Result res;
        Program program = CreateProgram();
        constexpr CoreCoord core = {0, 0};
        auto make_cb = [&](tt::CBIndex idx, uint32_t tiles) {
            CreateCircularBuffer(
                program,
                core,
                CircularBufferConfig(tiles * kTileBytes, {{idx, tt::DataFormat::Float32}})
                    .set_page_size(idx, kTileBytes));
        };
        make_cb(tt::CBIndex::c_1, 1);
        make_cb(tt::CBIndex::c_2, kCoefTiles);
        make_cb(tt::CBIndex::c_16, 4);

        std::vector<uint32_t> reader_args;
        TensorAccessorArgs(*scal_buf->get_backing_buffer()).append_to(reader_args);
        TensorAccessorArgs(*coef_buf->get_backing_buffer()).append_to(reader_args);
        auto reader = CreateKernel(
            program,
            kdir + "reader.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_args});
        std::vector<uint32_t> writer_args;
        TensorAccessorArgs(*out_buf->get_backing_buffer()).append_to(writer_args);
        auto writer = CreateKernel(
            program,
            kdir + "writer.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_args});

        std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
        unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_2)] = UnpackToDestMode::UnpackToDestFp32;
        auto compute = CreateKernel(
            program,
            kdir + "compute.cpp",
            core,
            ComputeConfig{
                .fp32_dest_acc_en = true,
                .unpack_to_dest_mode = unpack_modes,
                .defines = {
                    {"SPIKE_APPROACH", std::to_string(var.approach)},
                    {"SPIKE_VEC", std::to_string(var.vec)},
                    {"SPIKE_TPC", std::to_string(var.tpc)}}});

        SetRuntimeArgs(
            program,
            reader,
            core,
            {static_cast<uint32_t>(scal_buf->address()), static_cast<uint32_t>(coef_buf->address()), kCoefTiles});
        SetRuntimeArgs(program, writer, core, {static_cast<uint32_t>(out_buf->address()), 2u});
        SetRuntimeArgs(program, compute, core, {k, nv});

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
        const auto h0 = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        const auto h1 = std::chrono::steady_clock::now();
        res.host_ms = std::chrono::duration<double, std::milli>(h1 - h0).count();

        std::vector<float> raw;
        distributed::EnqueueReadMeshBuffer(cq, raw, out_buf, true);
        auto untile = [&](uint32_t t) {
            std::vector<float> one(raw.begin() + t * kElems, raw.begin() + (t + 1) * kElems);
            return untilize_nfaces(one, kTile, kTile);
        };
        const std::vector<float> acc = untile(0), timing = untile(1);
        res.cycles = static_cast<uint32_t>(timing[0]) + (static_cast<uint32_t>(timing[1]) << 16);

        // Exact reference: every quantity is an integer, evaluated in int64.
        for (uint32_t r = 0; r < kTile; r++) {
            for (uint32_t c = 0; c < kTile; c++) {
                int64_t sum = 0;
                // Only the first nv of the 32 SFPU vectors are updated (same lane map as Spike A).
                const uint32_t face = (r / 16) * 2 + c / 16;
                const uint32_t vec = face * 8 + 2 * ((r % 16) / 4) + ((c % 16) & 1);
                for (uint32_t i = 0; i < (vec < nv ? kTriangles : 0u); i++) {
                    const int64_t a = var.approach == 0 ? 2 : tris[i].a, b = var.approach == 0 ? -3 : tris[i].b,
                                  cc = var.approach == 0 ? 17 : tris[i].c, d = var.approach == 0 ? 1 : tris[i].d;
                    const int64_t e = a * static_cast<int64_t>(c) + b * static_cast<int64_t>(r) + cc;
                    if (e >= 0) {
                        sum += d;
                    }
                }
                const float expect = static_cast<float>(sum * static_cast<int64_t>(k));
                if (acc[r * kTile + c] != expect) {
                    if (res.mismatches < 3) {
                        std::printf(
                            "    mismatch (%s) at r=%u c=%u: device %.9g, expected %.9g\n", var.name, r, c,
                            acc[r * kTile + c], expect);
                    }
                    res.mismatches++;
                }
            }
        }
        res.ok = res.mismatches == 0;
        return res;
    };

    const Variant variants[] = {
        {"const  tri-major, 64/call", 0, 64, 0},
        {"A: L1->imm tri-major, 64/call", 1, 64, 0},
        {"A: L1->imm tri-major,  8/call", 1, 8, 0},
        {"B: coef tile tri-major, 8/call", 2, 8, 0},
        {"C: prgm const tri-major, 64/call", 3, 64, 0},
        {"A: L1->imm vec-major, 64/call", 1, 64, 1},
        {"B: coef tile vec-major, 8/call", 2, 8, 1},
    };

    std::printf("K = %u batches of %u triangles = %u triangle-tile updates per variant\n\n", k_default, kTriangles,
                k_default * kTriangles);
    std::printf("%-36s %-6s %12s %14s %12s\n", "variant", "exact", "cycles", "cycles/tri", "host ms");
    bool all_ok = true;
    for (const Variant& v : variants) {
        const Result r = run(v, k_default);
        all_ok &= r.ok;
        std::printf(
            "%-36s %-6s %12u %14.1f %12.3f   (%u mismatches)\n", v.name, r.ok ? "yes" : "NO", r.cycles,
            static_cast<double>(r.cycles) / (k_default * kTriangles), r.host_ms, r.mismatches);
    }

    // Vector-count sweep: cycles per triangle = fixed + per_vector * nv. The fixed part is what a
    // tiny triangle costs even if it touches one vector; the slope is the per-vector SFPU cost.
    std::printf("\nvector-count sweep (cycles per triangle-tile update):\n%-34s", "variant \\ vectors touched");
    const uint32_t nvs[] = {1, 2, 4, 8, 16, 32};
    for (uint32_t n : nvs) {
        std::printf(" %8u", n);
    }
    std::printf("   fixed  /vector\n");
    for (int vi : {0, 1, 3, 4}) {
        std::printf("%-34s", variants[vi].name);
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (uint32_t n : nvs) {
            const Result r = run(variants[vi], k_default, n);
            all_ok &= r.ok;
            const double y = static_cast<double>(r.cycles) / (k_default * kTriangles);
            std::printf(" %8.1f%s", y, r.ok ? " " : "!");
            sx += n; sy += y; sxx += static_cast<double>(n) * n; sxy += n * y;
        }
        const double m = 6.0, slope = (m * sxy - sx * sy) / (m * sxx - sx * sx), icpt = (sy - slope * sx) / m;
        std::printf("  %6.1f  %6.2f\n", icpt, slope);
    }

    // Wall-clock tick rate: two runs of the baseline with different K; the slope removes launch overhead.
    const Result s1 = run(variants[0], 64), s2 = run(variants[0], 2048);
    const double d_ticks = static_cast<double>(s2.cycles) - s1.cycles;
    const double d_ms = s2.host_ms - s1.host_ms;
    std::printf("\ntick calibration: %.0f ticks in %.3f ms of host time -> %.1f MHz\n", d_ticks, d_ms,
                d_ticks / d_ms / 1000.0);

    mesh->close();
    return all_ok ? 0 : 1;
}
