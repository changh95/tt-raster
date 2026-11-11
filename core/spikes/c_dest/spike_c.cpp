// SPDX-License-Identifier: Apache-2.0
//
// Spike C: Dest capacity and residency (docs/PLAN.md, Phase 0.5).
//
// For each FP32 sync mode, hold N tiles in Dest at once, with N at the documented capacity and one
// above it, pack them all, and check that every tile holds its own signature.
//
//   TT_VISIBLE_DEVICES=<node> spike_c
//
// Capacity from the installed source (ckernel.h get_dest_max_tiles_rt): Dest is 1024 rows; a 32x32
// tile takes 64 rows in 16-bit mode and 128 in FP32 mode; half-sync uses half of Dest.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    constexpr uint32_t kElems = 32 * 32;
    constexpr uint32_t kTileBytes = kElems * sizeof(float);
    constexpr uint32_t kMaxTiles = 10;

    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh->mesh_command_queue();
    distributed::DeviceLocalBufferConfig dram_cfg{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    auto out_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kMaxTiles * kTileBytes}, dram_cfg, mesh.get());
    const std::string kdir = std::string(TT_RASTER_SPIKE_C_KERNELS);

    auto run = [&](bool full_sync, uint32_t n) {
        Program program = CreateProgram();
        constexpr CoreCoord core = {0, 0};
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(n * kTileBytes, {{tt::CBIndex::c_16, tt::DataFormat::Float32}})
                .set_page_size(tt::CBIndex::c_16, kTileBytes));
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
        auto compute = CreateKernel(
            program,
            kdir + "compute.cpp",
            core,
            ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = full_sync});
        SetRuntimeArgs(program, writer, core, {static_cast<uint32_t>(out_buf->address()), n});
        SetRuntimeArgs(program, compute, core, {n});
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        std::vector<float> raw;
        distributed::EnqueueReadMeshBuffer(cq, raw, out_buf, true);
        uint32_t tiles_ok = 0;
        std::string detail;
        for (uint32_t t = 0; t < n; t++) {
            std::vector<float> one(raw.begin() + t * kElems, raw.begin() + (t + 1) * kElems);
            const std::vector<float> m = untilize_nfaces(one, 32, 32);
            uint32_t bad = 0;
            for (uint32_t r = 0; r < 32; r++) {
                for (uint32_t c = 0; c < 32; c++) {
                    const uint32_t face = (r / 16) * 2 + c / 16;
                    const uint32_t v = face * 8 + 2 * ((r % 16) / 4) + ((c % 16) & 1);
                    bad += (m[r * 32 + c] != static_cast<float>((t + 1) * 100 + v));
                }
            }
            tiles_ok += (bad == 0);
            if (bad != 0) {
                detail += " tile" + std::to_string(t) + ":" + std::to_string(bad) + "bad";
            }
        }
        std::printf(
            "%-10s N=%u  %u of %u tiles intact%s\n", full_sync ? "full-sync" : "half-sync", n, tiles_ok, n,
            detail.empty() ? "" : (" (" + detail + " )").c_str());
        return tiles_ok == n;
    };

    // FP32 capacity: half-sync 4 tiles, full-sync 8 tiles. Test at the capacity and one above.
    bool at_cap_ok = true;
    at_cap_ok &= run(false, 1);
    at_cap_ok &= run(false, 4);
    run(false, 5);
    at_cap_ok &= run(true, 8);
    run(true, 9);
    mesh->close();
    std::printf("%s\n", at_cap_ok ? "all runs at or below the documented capacity are intact" : "UNEXPECTED FAILURE at or below capacity");
    return at_cap_ok ? 0 : 1;
}
