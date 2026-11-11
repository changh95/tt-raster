// SPDX-License-Identifier: Apache-2.0
//
// Spike E: single chip and full tray (docs/PLAN.md, Phase 0.7).
//
// One binary, two modes chosen at run time:
//   TT_VISIBLE_DEVICES=<node>      spike_e unit    one chip as a unit mesh
//   TT_VISIBLE_DEVICES=0,...,7     spike_e mesh    the visible chips as one mesh
// In both modes: run a trivial program on every device and verify its output, then measure
// host-to-chip (write) and chip-to-host (read) bandwidth per device, and for all devices at once.
//
// Exit code 0 only if every device ran the program and every transferred buffer verified.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
using Clock = std::chrono::steady_clock;

namespace {

double seconds(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "unit";
    if (mode != "unit" && mode != "mesh") {
        std::fprintf(stderr, "usage: spike_e [unit|mesh] [size_MiB]\n");
        return 2;
    }
    const uint64_t size_mib = argc > 2 ? static_cast<uint64_t>(std::atoi(argv[2])) : 256;
    const uint64_t bytes = size_mib << 20;
    constexpr int kReps = 3;

    std::shared_ptr<distributed::MeshDevice> mesh =
        mode == "unit" ? distributed::MeshDevice::create_unit_mesh(0)
                       : distributed::MeshDevice::create(
                             distributed::MeshDeviceConfig(distributed::SystemMesh::instance().shape()));
    auto& cq = mesh->mesh_command_queue();
    const distributed::MeshCoordinateRange all(mesh->shape());
    const size_t n_dev = mesh->num_devices();
    std::ostringstream shape_str;
    shape_str << mesh->shape();
    std::printf("mode %s: mesh %s, %zu device(s), buffer %llu MiB\n", mode.c_str(), shape_str.str().c_str(), n_dev,
                static_cast<unsigned long long>(size_mib));

    bool ok = true;

    // ---- 1. Trivial program on every device ----------------------------------------------------
    {
        distributed::DeviceLocalBufferConfig cfg{.page_size = 64, .buffer_type = BufferType::DRAM};
        auto out = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = 64}, cfg, mesh.get());
        Program program = CreateProgram();
        constexpr CoreCoord core = {0, 0};
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(64, {{tt::CBIndex::c_0, tt::DataFormat::UInt32}}).set_page_size(tt::CBIndex::c_0, 64));
        std::vector<uint32_t> args;
        TensorAccessorArgs(*out->get_backing_buffer()).append_to(args);
        auto k = CreateKernel(
            program,
            std::string(TT_RASTER_SPIKE_E_KERNELS) + "write_const.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = args});
        SetRuntimeArgs(program, k, core, {static_cast<uint32_t>(out->address())});
        distributed::MeshWorkload workload;
        workload.add_program(all, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        uint32_t good = 0;
        for (const auto& coord : all) {
            std::vector<uint32_t> got;
            distributed::ReadShard(cq, got, out, coord, true);
            bool match = got.size() >= 16;
            for (uint32_t i = 0; match && i < 16; i++) {
                match = (got[i] == 0xC0FFEE00u + i);
            }
            good += match;
        }
        std::printf("trivial program: %u of %zu devices wrote the expected words\n", good, n_dev);
        ok &= (good == n_dev);
    }

    // ---- 2. Host <-> chip bandwidth ------------------------------------------------------------
    distributed::DeviceLocalBufferConfig big_cfg{.page_size = 1u << 16, .buffer_type = BufferType::DRAM};
    auto buf = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = bytes}, big_cfg, mesh.get());
    std::vector<uint32_t> src(bytes / 4);
    for (size_t i = 0; i < src.size(); i++) {
        src[i] = static_cast<uint32_t>(i * 2654435761u);
    }

    std::printf("\n%-8s %-10s %14s %14s\n", "coord", "device id", "write GB/s", "read GB/s");
    for (const auto& coord : all) {
        std::vector<double> w, r;
        std::vector<uint32_t> dst;
        for (int rep = 0; rep < kReps; rep++) {
            auto t0 = Clock::now();
            distributed::WriteShard(cq, buf, src, coord, true);
            auto t1 = Clock::now();
            distributed::ReadShard(cq, dst, buf, coord, true);
            auto t2 = Clock::now();
            w.push_back(bytes / seconds(t0, t1) / 1e9);
            r.push_back(bytes / seconds(t1, t2) / 1e9);
        }
        const bool same = (dst == src);
        ok &= same;
        std::ostringstream cs;
        cs << coord;
        std::printf("%-8s %-10d %14.2f %14.2f %s\n", cs.str().c_str(), mesh->get_device(coord)->id(), median(w),
                    median(r), same ? "" : " MISMATCH");
    }

    if (n_dev > 1) {
        // All devices at once: one replicated write, then non-blocking reads of every shard.
        std::vector<double> w, r;
        std::vector<std::vector<uint32_t>> dsts(n_dev);
        for (int rep = 0; rep < kReps; rep++) {
            auto t0 = Clock::now();
            distributed::EnqueueWriteMeshBuffer(cq, buf, src, true);
            auto t1 = Clock::now();
            size_t i = 0;
            for (const auto& coord : all) {
                distributed::ReadShard(cq, dsts[i++], buf, coord, false);
            }
            distributed::Finish(cq);
            auto t2 = Clock::now();
            w.push_back(static_cast<double>(bytes) * n_dev / seconds(t0, t1) / 1e9);
            r.push_back(static_cast<double>(bytes) * n_dev / seconds(t1, t2) / 1e9);
        }
        bool same = true;
        for (const auto& d : dsts) {
            same &= (d == src);
        }
        ok &= same;
        std::printf("%-8s %-10s %14.2f %14.2f %s\n", "all", "(aggregate)", median(w), median(r),
                    same ? "" : " MISMATCH");
    }

    mesh->close();
    std::printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
