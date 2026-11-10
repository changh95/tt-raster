// SPDX-License-Identifier: Apache-2.0
//
// Spike A: lane map and tile layout (docs/PLAN.md, Phase 0.3).
//
// The host builds dx and dy coordinate tiles (0..31, row-major), tilizes them, and uploads
// them. The compute kernel unpacks them into Dest (FP32), the SFPU computes a checkerboard
// plus gradient, a lane probe, and an FP32 arithmetic check, and the results come back.
//
//   TT_VISIBLE_DEVICES=<node> spike_a
//
// Exit code 0 only if the checkerboard output equals the CPU result bit for bit.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
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

uint32_t bits_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

// Distance in FP32 ulps between two finite floats.
int64_t ulp_distance(float a, float b) {
    auto ordered = [](float f) {
        const int32_t i = static_cast<int32_t>(bits_of(f));
        return i < 0 ? static_cast<int64_t>(INT32_MIN) - i : static_cast<int64_t>(i);
    };
    const int64_t d = ordered(a) - ordered(b);
    return d < 0 ? -d : d;
}

}  // namespace

int main() {
    // Inputs: dx[r][c] = c, dy[r][c] = r.
    std::vector<float> dx(kElems), dy(kElems);
    for (uint32_t r = 0; r < kTile; r++) {
        for (uint32_t c = 0; c < kTile; c++) {
            dx[r * kTile + c] = static_cast<float>(c);
            dy[r * kTile + c] = static_cast<float>(r);
        }
    }
    std::vector<float> in_tiles = tilize_nfaces(dx, kTile, kTile);
    {
        const std::vector<float> ty = tilize_nfaces(dy, kTile, kTile);
        in_tiles.insert(in_tiles.end(), ty.begin(), ty.end());
    }

    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh->mesh_command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t n_in = 2, n_out = 3;
    distributed::DeviceLocalBufferConfig dram_cfg{.page_size = kTileBytes, .buffer_type = BufferType::DRAM};
    auto src = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = n_in * kTileBytes}, dram_cfg, mesh.get());
    auto dst = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = n_out * kTileBytes}, dram_cfg, mesh.get());
    distributed::EnqueueWriteMeshBuffer(cq, src, in_tiles, false);

    // FP32 circular buffers, tile-sized pages.
    auto make_cb = [&](tt::CBIndex idx, uint32_t tiles) {
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tiles * kTileBytes, {{idx, tt::DataFormat::Float32}}).set_page_size(idx, kTileBytes));
    };
    make_cb(tt::CBIndex::c_0, 2);
    make_cb(tt::CBIndex::c_16, 4);

    const std::string kdir = std::string(TT_RASTER_SPIKE_A_KERNELS);

    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(*src->get_backing_buffer()).append_to(reader_args);
    auto reader = CreateKernel(
        program,
        kdir + "reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_args});
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(*dst->get_backing_buffer()).append_to(writer_args);
    auto writer = CreateKernel(
        program,
        kdir + "writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_args});

    std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_modes[static_cast<uint32_t>(tt::CBIndex::c_0)] = UnpackToDestMode::UnpackToDestFp32;
    auto compute = CreateKernel(
        program,
        kdir + "compute.cpp",
        core,
        ComputeConfig{.fp32_dest_acc_en = true, .unpack_to_dest_mode = unpack_modes});

    SetRuntimeArgs(program, reader, core, {static_cast<uint32_t>(src->address()), n_in});
    SetRuntimeArgs(program, writer, core, {static_cast<uint32_t>(dst->address()), n_out});

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<float> raw;
    distributed::EnqueueReadMeshBuffer(cq, raw, dst, true);
    mesh->close();
    if (raw.size() != n_out * kElems) {
        std::cerr << "unexpected result size " << raw.size() << "\n";
        return 2;
    }

    auto untile = [&](uint32_t t) {
        std::vector<float> one(raw.begin() + t * kElems, raw.begin() + (t + 1) * kElems);
        return untilize_nfaces(one, kTile, kTile);
    };
    const std::vector<float> f = untile(0), probe = untile(1), math = untile(2);

    // 1. Checkerboard + gradient: must be bit-exact.
    uint32_t f_bad = 0;
    for (uint32_t r = 0; r < kTile; r++) {
        for (uint32_t c = 0; c < kTile; c++) {
            float expect = static_cast<float>(c) * 0.25f + static_cast<float>(r) * 0.5f;
            if (((r + c) & 1) != 0) {
                expect += 100.0f;
            }
            if (bits_of(f[r * kTile + c]) != bits_of(expect)) {
                if (f_bad < 5) {
                    std::printf("  f mismatch at (r=%u,c=%u): device %.9g, cpu %.9g\n", r, c, f[r * kTile + c], expect);
                }
                f_bad++;
            }
        }
    }
    std::printf("checkerboard+gradient: %u of %u elements differ from the CPU\n", f_bad, kElems);

    // 2. FP32 arithmetic: x*0.1 + y*0.3 against two CPU evaluation orders.
    uint32_t unfused_eq = 0, fused_eq = 0, either_eq = 0;
    int64_t max_ulp_unfused = 0, max_ulp_fused = 0;
    for (uint32_t r = 0; r < kTile; r++) {
        for (uint32_t c = 0; c < kTile; c++) {
            const float x = static_cast<float>(c), y = static_cast<float>(r);
            volatile float p0 = x * 0.1f;  // volatile: keep the CPU from fusing the reference
            volatile float p1 = y * 0.3f;
            const float unfused = p0 + p1;
            const float fused = std::fmaf(y, 0.3f, p0);  // MAD(y, 0.3, round(x*0.1))
            const float dev = math[r * kTile + c];
            const bool eu = bits_of(dev) == bits_of(unfused), ef = bits_of(dev) == bits_of(fused);
            unfused_eq += eu;
            fused_eq += ef;
            either_eq += (eu || ef);
            max_ulp_unfused = std::max(max_ulp_unfused, ulp_distance(dev, unfused));
            max_ulp_fused = std::max(max_ulp_fused, ulp_distance(dev, fused));
        }
    }
    std::printf(
        "x*0.1f+y*0.3f: bit-equal to two-rounding CPU %u/%u (max %lld ulp), to fused CPU %u/%u (max %lld ulp), "
        "to either %u/%u\n",
        unfused_eq, kElems, static_cast<long long>(max_ulp_unfused), fused_eq, kElems,
        static_cast<long long>(max_ulp_fused), either_eq, kElems);

    // 3. Lane map. probe value = (face*8 + vector) * 64 + vConstTileId, where face and vector
    //    are the SFPU's own loop counters and the lane constant is vConstTileId.
    //    For every tile position (r, c) report which (face, vector, tileid) wrote it.
    std::map<int, int> tid_seen;
    uint32_t face_consistent = 0;
    for (uint32_t r = 0; r < kTile; r++) {
        for (uint32_t c = 0; c < kTile; c++) {
            const int code = static_cast<int>(probe[r * kTile + c]);
            const int face = code / 512, tid = code % 64;
            tid_seen[tid]++;
            face_consistent += (face == static_cast<int>((r / 16) * 2 + (c / 16)));
        }
    }
    std::printf("lane probe: SFPU face index equals row-major face (r/16)*2+(c/16) at %u of %u positions\n",
                face_consistent, kElems);
    std::printf("lane probe: %zu distinct vConstTileId values seen\n", tid_seen.size());

    std::printf("face 0 (rows 0-15, cols 0-15): SFPU vector index (0..7) | vConstTileId\n");
    for (uint32_t r = 0; r < 16; r++) {
        for (uint32_t c = 0; c < 16; c++) {
            std::printf("%d ", (static_cast<int>(probe[r * kTile + c]) / 64) % 8);
        }
        std::printf("  | ");
        for (uint32_t c = 0; c < 16; c++) {
            std::printf("%2d ", static_cast<int>(probe[r * kTile + c]) % 64);
        }
        std::printf("\n");
    }
    return f_bad == 0 ? 0 : 1;
}
