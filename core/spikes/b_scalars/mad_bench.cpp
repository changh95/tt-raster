// SPDX-License-Identifier: Apache-2.0
//
// SFPU multiply-add issue rate (V6): 1, 2, 4 and 8 independent chains of x = x*a + b.
// Cycles per multiply-add come from the slope between two run lengths, which removes the
// loop and start-up overhead.
//
//   TT_VISIBLE_DEVICES=<node> mad_bench

#include <chrono>
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
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh->mesh_command_queue();
    distributed::DeviceLocalBufferConfig dram_cfg{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    auto out_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kTileBytes}, dram_cfg, mesh.get());
    const std::string kdir = std::string(TT_RASTER_SPIKE_B_KERNELS);

    auto run = [&](int chains, uint32_t iters) {
        Program program = CreateProgram();
        constexpr CoreCoord core = {0, 0};
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(2 * kTileBytes, {{tt::CBIndex::c_16, tt::DataFormat::Float32}})
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
            kdir + "compute_mad.cpp",
            core,
            ComputeConfig{.fp32_dest_acc_en = true, .defines = {{"MAD_CHAINS", std::to_string(chains)}}});
        SetRuntimeArgs(program, writer, core, {static_cast<uint32_t>(out_buf->address()), 1u});
        SetRuntimeArgs(program, compute, core, {iters});
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        std::vector<float> raw;
        distributed::EnqueueReadMeshBuffer(cq, raw, out_buf, true);
        const std::vector<float> t = untilize_nfaces(raw, 32, 32);
        return static_cast<uint32_t>(t[0]) + (static_cast<uint32_t>(t[1]) << 16);
    };

    std::printf("%-8s %12s %12s %16s\n", "chains", "cycles(1k)", "cycles(3k)", "cycles/MAD");
    for (int chains : {1, 2, 4, 8}) {
        const uint32_t c1 = run(chains, 1000), c3 = run(chains, 3000);
        const double per_mad = static_cast<double>(c3 - c1) / (2000.0 * 8 * chains);
        std::printf("%-8d %12u %12u %16.3f\n", chains, c1, c3, per_mad);
    }
    mesh->close();
    return 0;
}
