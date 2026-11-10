// SPDX-License-Identifier: Apache-2.0

#include "tensix/tensix_renderer.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include <tt-metalium/tensor_accessor_args.hpp>

#include "tensix/kernels/raster_layout.h"

using namespace tt;
using namespace tt::tt_metal;
using Clock = std::chrono::steady_clock;

namespace ttr {

namespace {

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

uint32_t float_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

void put_float(std::vector<uint32_t>& w, size_t at, float f) { w[at] = float_bits(f); }

// Encode one (tile, triangle) record at word offset `at` of `w`.
void encode_record(std::vector<uint32_t>& w, size_t at, const SetupTriangle& tri, const TileParams& p, TileCoverage cov) {
    using namespace layout;
    for (int e = 0; e < 3; e++) {
        put_float(w, at + kEdge + 3 * e + 0, p.edge[e].a);
        put_float(w, at + kEdge + 3 * e + 1, p.edge[e].b);
        put_float(w, at + kEdge + 3 * e + 2, p.edge[e].c);
    }
    put_float(w, at + kDepth + 0, p.depth_c);
    put_float(w, at + kDepth + 1, p.depth_gx);
    put_float(w, at + kDepth + 2, p.depth_gy);
    for (int ch = 0; ch < 4; ch++) {
        put_float(w, at + kColor + 3 * ch + 0, p.color_c[ch]);
        put_float(w, at + kColor + 3 * ch + 1, p.color_gx[ch]);
        put_float(w, at + kColor + 3 * ch + 2, p.color_gy[ch]);
    }
    put_float(w, at + kId, static_cast<float>(tri.source + 1));
    w[at + kFlags] = (tri.flat ? kFlagFlat : 0u) | (cov == TileCoverage::Full ? kFlagFull : 0u);
}

// Untile one 32x32 tile of 32-bit words (faces TL, TR, BL, BR, each 16x16 row-major) into row-major.
void untile_words(const uint32_t* tile, uint32_t* out /* 32x32 */) {
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = 0; c < 32; c++) {
            const uint32_t face = (r / 16) * 2 + c / 16;
            out[r * 32 + c] = tile[face * 256 + (r % 16) * 16 + (c % 16)];
        }
    }
}

}  // namespace

std::vector<CoreCoord> TensixRenderer::worker_cores() const {
    const CoreCoord grid = mesh_->compute_with_storage_grid_size();
    std::vector<CoreCoord> cores;
    for (size_t y = 0; y < grid.y; y++) {
        for (size_t x = 0; x < grid.x; x++) {
            cores.push_back(CoreCoord(x, y));
        }
    }
    return cores;
}

std::vector<CorePlan> plan_row_major(const std::vector<CoreCoord>& cores, uint32_t tiles_x, uint32_t tiles_y) {
    const size_t n_tiles = static_cast<size_t>(tiles_x) * tiles_y;
    const size_t n_cores = std::min(cores.size(), n_tiles);
    std::vector<CorePlan> plan(n_cores);
    for (size_t i = 0; i < n_cores; i++) {
        plan[i].core = cores[i];
        const size_t lo = n_tiles * i / n_cores, hi = n_tiles * (i + 1) / n_cores;
        for (size_t t = lo; t < hi; t++) {
            plan[i].tiles.emplace_back(static_cast<int>(t % tiles_x), static_cast<int>(t / tiles_x));
        }
    }
    return plan;
}

DeviceFrame TensixRenderer::render(
    const Frame& frame, const SetupResult& setup, const TileBins& bins, const std::vector<CorePlan>& plan,
    const DeviceOptions& opt) {
    using namespace layout;
    DeviceFrame out;
    const uint32_t pw = frame.padded_width(), ph = frame.padded_height();
    out.fb.width = pw;
    out.fb.height = ph;
    const size_t n_px = static_cast<size_t>(pw) * ph;
    out.fb.color.assign(n_px, pack_rgba8(frame.clear_color));
    out.fb.depth.assign(n_px, frame.clear_depth);
    out.fb.id.assign(n_px, 0);
    out.fb.cover.assign(n_px, 0);
    out.fb.ambiguous.assign(n_px, 0);
    out.tile_rendered.assign(static_cast<size_t>(frame.tiles_x()) * frame.tiles_y(), 0);

    // ---- 1. Work lists ------------------------------------------------------------------------
    auto t0 = Clock::now();
    std::vector<uint32_t> words;                  // all cores' pages, concatenated
    std::vector<uint32_t> first_page(plan.size()), n_pages(plan.size()), first_block(plan.size());
    uint32_t blocks_total = 0;
    for (size_t c = 0; c < plan.size(); c++) {
        first_page[c] = static_cast<uint32_t>(words.size() / (kPageBytes / 4));
        first_block[c] = blocks_total;
        for (const auto& [tx, ty] : plan[c].tiles) {
            const auto& entries = bins.tile(static_cast<uint32_t>(tx), static_cast<uint32_t>(ty));
            const uint32_t pages = pages_for_records(static_cast<uint32_t>(entries.size()));
            const size_t base = words.size();
            words.resize(base + static_cast<size_t>(pages) * (kPageBytes / 4), 0u);
            words[base + kHdrRecords] = static_cast<uint32_t>(entries.size());
            words[base + kHdrTileX] = static_cast<uint32_t>(tx);
            words[base + kHdrTileY] = static_cast<uint32_t>(ty);
            for (size_t i = 0; i < entries.size(); i++) {
                const SetupTriangle& tri = setup.tris[entries[i].tri];
                const TileParams p = make_tile_params(tri, tx, ty);
                encode_record(words, base + (i + 1) * kRecordWords, tri, p, entries[i].coverage);
            }
            out.times.records += entries.size();
            out.times.blocks++;
            blocks_total++;
            out.tile_rendered[static_cast<size_t>(ty) * frame.tiles_x() + tx] = 1;
        }
        n_pages[c] = static_cast<uint32_t>(words.size() / (kPageBytes / 4)) - first_page[c];
    }
    if (words.empty()) {
        return out;
    }
    out.times.build_ms = ms_since(t0);

    // ---- 2. Buffers, program --------------------------------------------------------------------
    t0 = Clock::now();
    auto& cq = mesh_->mesh_command_queue();
    const uint32_t total_pages = static_cast<uint32_t>(words.size() / (kPageBytes / 4));
    distributed::DeviceLocalBufferConfig cfg{.page_size = kPageBytes, .buffer_type = BufferType::DRAM};
    auto in_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = static_cast<uint64_t>(total_pages) * kPageBytes}, cfg, mesh_.get());
    auto out_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = static_cast<uint64_t>(blocks_total) * kOutPlanes * kTileBytes}, cfg,
        mesh_.get());
    distributed::EnqueueWriteMeshBuffer(cq, in_buf, words, false);

    Program program = CreateProgram();
    std::set<CoreRange> ranges;
    for (const CorePlan& cp : plan) {
        ranges.insert(CoreRange(cp.core));
    }
    const CoreRangeSet core_set(ranges);

    auto make_cb = [&](tt::CBIndex idx, uint32_t pages) {
        CreateCircularBuffer(
            program, core_set,
            CircularBufferConfig(pages * kPageBytes, {{idx, tt::DataFormat::Float32}}).set_page_size(idx, kPageBytes));
    };
    make_cb(tt::CBIndex::c_0, 4);
    make_cb(tt::CBIndex::c_16, 2);
    make_cb(tt::CBIndex::c_17, 2);
    make_cb(tt::CBIndex::c_18, 2);

    const std::string kdir = TT_RASTER_TENSIX_KERNELS;
    std::vector<uint32_t> reader_args, writer_args;
    TensorAccessorArgs(*in_buf->get_backing_buffer()).append_to(reader_args);
    TensorAccessorArgs(*out_buf->get_backing_buffer()).append_to(writer_args);
    auto reader = CreateKernel(
        program, kdir + "reader.cpp", core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_args});
    auto writer = CreateKernel(
        program, kdir + "writer.cpp", core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_args});

    std::map<std::string, std::string> defines;
    if (frame.depth_func == DepthFunc::LessEqual) {
        defines["RASTER_LESS_EQUAL"] = "1";
    }
    if (opt.cover_count) {
        defines["RASTER_COVER_COUNT"] = "1";
    }
    auto compute = CreateKernel(
        program, kdir + "compute.cpp", core_set,
        ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .defines = defines});

    for (size_t c = 0; c < plan.size(); c++) {
        const CoreCoord core = plan[c].core;
        SetRuntimeArgs(program, reader, core, {static_cast<uint32_t>(in_buf->address()), first_page[c], n_pages[c]});
        SetRuntimeArgs(
            program, writer, core,
            {static_cast<uint32_t>(out_buf->address()), first_block[c], static_cast<uint32_t>(plan[c].tiles.size())});
        SetRuntimeArgs(
            program, compute, core,
            {static_cast<uint32_t>(plan[c].tiles.size()), float_bits(frame.clear_depth), float_bits(frame.clear_color[0]),
             float_bits(frame.clear_color[1]), float_bits(frame.clear_color[2]), float_bits(frame.clear_color[3])});
    }
    distributed::Finish(cq);  // the upload
    out.times.upload_ms = ms_since(t0);

    // ---- 3. Run ---------------------------------------------------------------------------------
    t0 = Clock::now();
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    out.times.device_ms = ms_since(t0);

    // ---- 4. Read back ---------------------------------------------------------------------------
    t0 = Clock::now();
    std::vector<uint32_t> raw;
    distributed::EnqueueReadMeshBuffer(cq, raw, out_buf, true);
    uint32_t block = 0;
    std::vector<uint32_t> rm[kOutPlanes];
    for (auto& v : rm) {
        v.resize(kTileWords);
    }
    for (size_t c = 0; c < plan.size(); c++) {
        for (const auto& [tx, ty] : plan[c].tiles) {
            for (uint32_t p = 0; p < kOutPlanes; p++) {
                untile_words(&raw[(static_cast<size_t>(block) * kOutPlanes + p) * kTileWords], rm[p].data());
            }
            for (uint32_t r = 0; r < 32; r++) {
                for (uint32_t col = 0; col < 32; col++) {
                    const size_t idx = out.fb.index(tx * 32 + col, ty * 32 + r);
                    out.fb.color[idx] = rm[kPlaneColor][r * 32 + col];
                    std::memcpy(&out.fb.depth[idx], &rm[kPlaneDepth][r * 32 + col], 4);
                    float id;
                    std::memcpy(&id, &rm[kPlaneId][r * 32 + col], 4);
                    if (opt.cover_count) {
                        out.fb.cover[idx] = static_cast<uint8_t>(std::min(255.0f, id));
                    } else {
                        out.fb.id[idx] = static_cast<uint32_t>(id);
                    }
                }
            }
            block++;
        }
    }
    out.times.readback_ms = ms_since(t0);
    return out;
}

}  // namespace ttr
