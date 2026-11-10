// SPDX-License-Identifier: Apache-2.0
//
// Spike D: NoC multicast on one chip (docs/PLAN.md, Phase 0.6).
//
// One sender core multicasts a 4 KB buffer to a rectangle of worker cores with a semaphore handshake;
// every receiver checksums the buffer and writes a record to DRAM. Rectangles include grid edges,
// full rows and columns, and the full grid (sender inside the rectangle, loopback multicast).
//
//   TT_VISIBLE_DEVICES=<node> spike_d
//
// Exit code 0 only if every receiver of every rectangle verified the data.

#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {

struct Test {
    const char* name;
    CoreCoord sender;
    CoreRange rect;
};

constexpr uint32_t kBytes = 4096;
constexpr uint32_t kRecordBytes = 64;
constexpr uint32_t kPages = 128;
constexpr uint32_t kSenderPage = kPages - 1;
constexpr uint32_t kRepeats = 256;
constexpr double kClockGhz = 1.35;

bool contains(const CoreRange& r, const CoreCoord& c) {
    return c.x >= r.start_coord.x && c.x <= r.end_coord.x && c.y >= r.start_coord.y && c.y <= r.end_coord.y;
}

}  // namespace

int main() {
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh->mesh_command_queue();
    const CoreCoord grid = mesh->compute_with_storage_grid_size();
    const uint32_t W = grid.x, H = grid.y;
    std::printf("worker grid %ux%u\n", W, H);

    // Logical -> NoC coordinate map: shows whether the worker columns and rows are contiguous.
    std::printf("logical x -> NoC x:");
    for (uint32_t x = 0; x < W; x++) {
        std::printf(" %u", static_cast<unsigned>(mesh->worker_core_from_logical_core({x, 0}).x));
    }
    std::printf("\nlogical y -> NoC y:");
    for (uint32_t y = 0; y < H; y++) {
        std::printf(" %u", static_cast<unsigned>(mesh->worker_core_from_logical_core({0, y}).y));
    }
    std::printf("\n\n");

    distributed::DeviceLocalBufferConfig dram_cfg{.page_size = kRecordBytes, .buffer_type = BufferType::DRAM};
    auto out_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kPages * kRecordBytes}, dram_cfg, mesh.get());

    const std::string kdir = std::string(TT_RASTER_SPIKE_D_KERNELS);
    const uint32_t X = W - 1, Y = H - 1;  // last logical column and row

    const std::vector<Test> tests = {
        {"single receiver", {0, 0}, CoreRange({1, 0}, {1, 0})},
        {"top row", {0, 0}, CoreRange({1, 0}, {X, 0})},
        {"left column", {0, 0}, CoreRange({0, 1}, {0, Y})},
        {"interior block", {0, 0}, CoreRange({3, 3}, {8, 6})},
        {"right edge column", {0, 0}, CoreRange({X, 0}, {X, Y})},
        {"bottom row (edge)", {0, 0}, CoreRange({0, Y}, {X, Y})},
        {"all rows below the sender", {0, 0}, CoreRange({0, 1}, {X, Y})},
        {"sender at far corner", {X, Y}, CoreRange({0, 0}, {X - 1, Y - 1})},
        {"sender mid-grid, outside", {5, 0}, CoreRange({0, 2}, {X, Y})},
        {"full grid, sender inside (loopback)", {0, 0}, CoreRange({0, 0}, {X, Y})},
        {"full grid, sender mid-grid (loopback)", {5, 4}, CoreRange({0, 0}, {X, Y})},
    };

    bool all_ok = true;
    std::printf("%-40s %6s %5s %10s %10s %12s %12s\n", "rectangle", "cores", "ok", "lat cyc", "lat ns", "inject GB/s",
                "delivered GB/s");
    for (const Test& t : tests) {
        const bool loopback = contains(t.rect, t.sender);
        const uint32_t num_dests = static_cast<uint32_t>(t.rect.size());
        const uint32_t n_receivers = num_dests - (loopback ? 1 : 0);

        Program program = CreateProgram();
        std::vector<CoreRange> ranges = {t.rect};
        if (!loopback) {
            ranges.push_back(CoreRange(t.sender, t.sender));
        }
        const CoreRangeSet all_cores(ranges);

        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(kBytes, {{tt::CBIndex::c_0, tt::DataFormat::UInt32}})
                .set_page_size(tt::CBIndex::c_0, kBytes));
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(kRecordBytes, {{tt::CBIndex::c_1, tt::DataFormat::UInt32}})
                .set_page_size(tt::CBIndex::c_1, kRecordBytes));
        const uint32_t ready_sem = CreateSemaphore(program, all_cores, 0);
        const uint32_t flag_sem = CreateSemaphore(program, all_cores, 0);
        const uint32_t ack_sem = CreateSemaphore(program, all_cores, 0);

        std::vector<uint32_t> out_args;
        TensorAccessorArgs(*out_buf->get_backing_buffer()).append_to(out_args);
        auto sender = CreateKernel(
            program,
            kdir + "sender.cpp",
            t.sender,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = out_args});
        auto receiver = CreateKernel(
            program,
            kdir + "receiver.cpp",
            t.rect,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = out_args});

        const CoreCoord s_phys = mesh->worker_core_from_logical_core(t.sender);
        const CoreCoord r_start = mesh->worker_core_from_logical_core(t.rect.start_coord);
        const CoreCoord r_end = mesh->worker_core_from_logical_core(t.rect.end_coord);
        const uint32_t out_addr = static_cast<uint32_t>(out_buf->address());

        SetRuntimeArgs(
            program,
            sender,
            t.sender,
            {static_cast<uint32_t>(r_start.x), static_cast<uint32_t>(r_start.y), static_cast<uint32_t>(r_end.x),
             static_cast<uint32_t>(r_end.y), num_dests, n_receivers, ready_sem, flag_sem, ack_sem,
             loopback ? 1u : 0u, kRepeats, out_addr, kBytes, kSenderPage});
        for (uint32_t y = t.rect.start_coord.y; y <= t.rect.end_coord.y; y++) {
            for (uint32_t x = t.rect.start_coord.x; x <= t.rect.end_coord.x; x++) {
                const bool is_sender = (x == t.sender.x && y == t.sender.y);
                SetRuntimeArgs(
                    program,
                    receiver,
                    CoreCoord{x, y},
                    {is_sender ? 1u : 0u, static_cast<uint32_t>(s_phys.x), static_cast<uint32_t>(s_phys.y), ready_sem,
                     flag_sem, ack_sem, out_addr, y * W + x, kBytes, x, y});
            }
        }

        // Clear the result pages so a missing write cannot pass.
        std::vector<uint32_t> zeros(kPages * kRecordBytes / 4, 0);
        distributed::EnqueueWriteMeshBuffer(cq, out_buf, zeros, true);

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        std::vector<uint32_t> raw;
        distributed::EnqueueReadMeshBuffer(cq, raw, out_buf, true);
        uint32_t ok = 0;
        for (uint32_t y = t.rect.start_coord.y; y <= t.rect.end_coord.y; y++) {
            for (uint32_t x = t.rect.start_coord.x; x <= t.rect.end_coord.x; x++) {
                if (x == t.sender.x && y == t.sender.y) {
                    continue;
                }
                const uint32_t* rec = &raw[(y * W + x) * (kRecordBytes / 4)];
                if (rec[0] == 1 && rec[2] == x && rec[3] == y) {
                    ok++;
                } else if (n_receivers - ok <= 3) {
                    std::printf("    receiver (%u,%u): ok=%u bad=%u first bad idx %u value 0x%08x\n", x, y, rec[0],
                                rec[1], rec[4], rec[5]);
                }
            }
        }
        const uint32_t* s = &raw[kSenderPage * (kRecordBytes / 4)];
        const double lat_cyc = s[0];
        const double train_cyc = s[1];
        const double inject = (static_cast<double>(kBytes) * kRepeats) / (train_cyc / kClockGhz);
        std::printf("%-40s %6u %2u/%-2u %10.0f %10.0f %12.2f %12.2f\n", t.name, n_receivers, ok, n_receivers, lat_cyc,
                    lat_cyc / kClockGhz, inject, inject * num_dests);
        all_ok &= (ok == n_receivers);
    }
    mesh->close();
    std::printf("%s\n", all_ok ? "all rectangles verified" : "FAILURES");
    return all_ok ? 0 : 1;
}
