// SPDX-License-Identifier: Apache-2.0
//
// Spike F: texel fetch and filter rate of a Tensix core's data-movement RISC-V, for the texture-unit gate of docs/FULL_COVERAGE_PLAN.md
// (workstream 2: "Phase 3 texel-rate numbers"). One core and all cores, texture in L1 (nearest, bilinear) and in DRAM (NoC reads).
//
//   TT_VISIBLE_DEVICES=<node> spike_f

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {
constexpr double kClockGhz = 1.35;

struct Cfg {
    const char* name;
    uint32_t mode, bilinear, tex_log2, batch, pages;
    bool all_cores;
    uint32_t npix;
};
}  // namespace

int main() {
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh->mesh_command_queue();
    const CoreCoord grid = mesh->compute_with_storage_grid_size();
    const uint32_t ncores_all = grid.x * grid.y;
    std::printf("worker grid %ux%u = %u cores\n", static_cast<unsigned>(grid.x), static_cast<unsigned>(grid.y), ncores_all);

    const std::string kdir = std::string(TT_RASTER_SPIKE_F_KERNELS);
    constexpr uint32_t kTexLogDram = 11;  // 2048 x 2048 RGBA8 = 16 MB in DRAM
    distributed::DeviceLocalBufferConfig dram_cfg{.page_size = 64, .buffer_type = BufferType::DRAM};
    auto out_buf = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = ncores_all * 64}, dram_cfg, mesh.get());
    const uint64_t tex_bytes = (1ull << kTexLogDram) * (1ull << kTexLogDram) * 4;
    auto tex_buf = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = tex_bytes}, dram_cfg, mesh.get());
    {
        std::vector<uint32_t> fill(tex_bytes / 4);
        for (size_t i = 0; i < fill.size(); i++) fill[i] = static_cast<uint32_t>((i + 1) * 2654435761u);
        distributed::EnqueueWriteMeshBuffer(cq, tex_buf, fill, true);
    }

    const std::vector<Cfg> cfgs = {
        {"L1 nearest, 1 core", 0, 0, 8, 0, 0, false, 1u << 21},
        {"L1 bilinear, 1 core", 0, 1, 8, 0, 0, false, 1u << 21},
        {"L1 gather of 4 texels + weights, 1 core", 0, 2, 8, 0, 0, false, 1u << 21},
        {"DRAM 1 page per pixel, 1 core, 32 in flight", 1, 0, kTexLogDram, 32, 1, false, 1u << 18},
        {"DRAM 4 pages per pixel, 1 core, 32 in flight", 1, 1, kTexLogDram, 32, 4, false, 1u << 18},
        {"L1 bilinear, all cores", 0, 1, 8, 0, 0, true, 1u << 21},
        {"L1 gather of 4 texels + weights, all cores", 0, 2, 8, 0, 0, true, 1u << 21},
        {"DRAM 1 page per pixel, all cores", 1, 0, kTexLogDram, 32, 1, true, 1u << 18},
        {"DRAM 4 pages per pixel, all cores", 1, 1, kTexLogDram, 32, 4, true, 1u << 18},
    };

    std::printf("%-48s %6s %12s %12s %14s\n", "configuration", "cores", "cyc/pixel", "Mpix/s/core", "Mpix/s total");
    for (const Cfg& c : cfgs) {
        const CoreRange range = c.all_cores ? CoreRange({0, 0}, {grid.x - 1, grid.y - 1}) : CoreRange({0, 0}, {0, 0});
        const uint32_t ncores = static_cast<uint32_t>(range.size());
        Program program = CreateProgram();
        const uint32_t cb0 = c.mode == 0 ? (1u << (2 * c.tex_log2)) * 4 : c.batch * c.pages * 64;
        CreateCircularBuffer(program, range, CircularBufferConfig(cb0, {{tt::CBIndex::c_0, tt::DataFormat::UInt32}}).set_page_size(tt::CBIndex::c_0, cb0));
        CreateCircularBuffer(program, range, CircularBufferConfig(64, {{tt::CBIndex::c_1, tt::DataFormat::UInt32}}).set_page_size(tt::CBIndex::c_1, 64));
        CreateCircularBuffer(program, range, CircularBufferConfig(8192, {{tt::CBIndex::c_2, tt::DataFormat::UInt32}}).set_page_size(tt::CBIndex::c_2, 8192));
        std::vector<uint32_t> cargs = {c.mode, c.bilinear, c.tex_log2, c.batch, c.pages};
        TensorAccessorArgs(*out_buf->get_backing_buffer()).append_to(cargs);
        TensorAccessorArgs(*tex_buf->get_backing_buffer()).append_to(cargs);   // (the kernel names them in both modes)
        auto k = CreateKernel(program, kdir + "texel.cpp", range,
                              DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = cargs});
        const uint32_t out_addr = static_cast<uint32_t>(out_buf->address()), tex_addr = static_cast<uint32_t>(tex_buf->address());
        for (uint32_t y = 0; y < range.end_coord.y + 1; y++)
            for (uint32_t x = 0; x < range.end_coord.x + 1; x++) {
                const uint32_t idx = y * grid.x + x;
                // one texel per pixel along a direction that is not axis-aligned, a different start per core
                SetRuntimeArgs(program, k, CoreCoord{x, y},
                               {c.npix, (idx * 7919u) << 16, (idx * 104729u) << 16, 52429u, 39321u, out_addr, tex_addr, idx});
            }
        std::vector<uint32_t> zeros(ncores_all * 16, 0);
        distributed::EnqueueWriteMeshBuffer(cq, out_buf, zeros, true);
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        std::vector<uint32_t> raw;
        distributed::EnqueueReadMeshBuffer(cq, raw, out_buf, true);
        uint32_t slowest = 0;
        for (uint32_t i = 0; i < ncores; i++) slowest = std::max(slowest, raw[i * 16]);
        const double cyc = static_cast<double>(slowest) / c.npix;
        const double per_core = c.npix / (slowest / (kClockGhz * 1e9)) / 1e6;
        std::printf("%-48s %6u %12.1f %12.1f %14.0f\n", c.name, ncores, cyc, per_core, per_core * ncores);
    }
    mesh->close();
    return 0;
}
