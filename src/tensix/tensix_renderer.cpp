// SPDX-License-Identifier: Apache-2.0

#include "tensix/tensix_renderer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>

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
void encode_record(
    std::vector<uint32_t>& w, size_t at, const SetupTriangle& tri, const TileParams& p, TileCoverage cov, int tx, int ty,
    uint64_t* vec_counts = nullptr) {
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
    w[at + kFlatPacked] = pack_rgba8(p.color_c);  // meaningful for flat triangles (gradients are zero)
    w[at + kFlags] = (tri.flat ? kFlagFlat : 0u) | (cov == TileCoverage::Full ? kFlagFull : 0u);
    const VectorClasses vc = classify_vectors(tri, tx, ty, cov);
    w[at + kVecClass + 0] = vc.word[0];
    w[at + kVecClass + 1] = vc.word[1];
    if (vec_counts != nullptr) {
        for (int v = 0; v < kVectorsPerTile; v++) {
            const TileCoverage k = vector_class(vc, v);
            vec_counts[0] += k == TileCoverage::Partial;
            vec_counts[1] += k == TileCoverage::Full;
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

std::vector<std::pair<int, int>> all_tiles(uint32_t tiles_x, uint32_t tiles_y) {
    std::vector<std::pair<int, int>> t;
    t.reserve(static_cast<size_t>(tiles_x) * tiles_y);
    for (uint32_t y = 0; y < tiles_y; y++) {
        for (uint32_t x = 0; x < tiles_x; x++) {
            t.emplace_back(static_cast<int>(x), static_cast<int>(y));
        }
    }
    return t;
}

double tile_weight(const TileBins& bins, int tx, int ty) {
    // Measured on the device (docs/PERF.md): about 5 us per tile of fixed work and about 3.5 us per
    // record, so a tile costs as much as roughly 1.5 records before the first one.
    return 1.5 + static_cast<double>(bins.tile(static_cast<uint32_t>(tx), static_cast<uint32_t>(ty)).size());
}

std::vector<std::vector<std::pair<int, int>>> split_balanced(
    const std::vector<std::pair<int, int>>& tiles, const TileBins& bins, size_t parts) {
    std::vector<std::vector<std::pair<int, int>>> out(parts);
    double total = 0;
    for (const auto& [tx, ty] : tiles) {
        total += tile_weight(bins, tx, ty);
    }
    // Part k ends where the running weight first reaches (k + 1) / parts of the total.
    double run = 0;
    size_t part = 0;
    for (const auto& t : tiles) {
        const double w = tile_weight(bins, t.first, t.second);
        // Put the tile in the next part if that is closer to the ideal boundary.
        while (part + 1 < parts && run + 0.5 * w > total * static_cast<double>(part + 1) / static_cast<double>(parts)) {
            part++;
        }
        out[part].push_back(t);
        run += w;
    }
    return out;
}

std::vector<CorePlan> plan_balanced(
    const std::vector<CoreCoord>& cores, const std::vector<std::pair<int, int>>& tiles, const TileBins& bins) {
    const size_t parts = std::max<size_t>(1, std::min(cores.size(), tiles.size()));
    auto chunks = split_balanced(tiles, bins, parts);
    std::vector<CorePlan> plan;
    for (size_t i = 0; i < parts; i++) {
        if (chunks[i].empty()) {
            continue;
        }
        CorePlan cp;
        cp.core = cores[i];
        cp.tiles = std::move(chunks[i]);
        plan.push_back(std::move(cp));
    }
    return plan;
}

DeviceFrame TensixRenderer::render(
    const Frame& frame, const SetupResult& setup, const TileBins& bins, const std::vector<CorePlan>& plan,
    const DeviceOptions& opt, Framebuffer* target) {
    using namespace layout;
    DeviceFrame out;
    const uint32_t pw = frame.padded_width(), ph = frame.padded_height();
    if (target == nullptr) {
        out.fb.width = pw;
        out.fb.height = ph;
        const size_t n_px = static_cast<size_t>(pw) * ph;
        out.fb.color.assign(n_px, pack_rgba8(frame.clear_color));
        if (opt.debug_planes) {  // production (colour-only) frames do not carry the debug planes
            out.fb.depth.assign(n_px, frame.clear_depth);
            out.fb.id.assign(n_px, 0);
            out.fb.cover.assign(n_px, 0);
            out.fb.ambiguous.assign(n_px, 0);
        }
        target = &out.fb;
    }
    Framebuffer& fb = *target;
    out.tile_rendered.assign(static_cast<size_t>(frame.tiles_x()) * frame.tiles_y(), 0);

    // ---- 1. Work lists ------------------------------------------------------------------------
    // Page counts are known from the bins alone, so every core's slice of the final buffer has a
    // fixed offset and the cores are encoded in parallel straight into it.
    auto t0 = Clock::now();
    std::vector<uint32_t> first_page(plan.size()), n_pages(plan.size()), first_block(plan.size());
    uint32_t blocks_total = 0, pages_total = 0;
    for (size_t c = 0; c < plan.size(); c++) {
        first_page[c] = pages_total;
        first_block[c] = blocks_total;
        for (const auto& [tx, ty] : plan[c].tiles) {
            pages_total += pages_for_records(static_cast<uint32_t>(bins.tile(tx, ty).size()));
            out.tile_rendered[static_cast<size_t>(ty) * frame.tiles_x() + tx] = 1;
        }
        n_pages[c] = pages_total - first_page[c];
        blocks_total += static_cast<uint32_t>(plan[c].tiles.size());
        out.times.blocks += plan[c].tiles.size();
    }
    // Only headers and the records' words are written; padding is never read by the kernels, so the
    // buffer is not cleared between frames.
    std::vector<uint32_t>& words = words_;
    words.resize(static_cast<size_t>(pages_total) * (kPageBytes / 4));
    struct CoreCounts {
        uint64_t records = 0, vec_partial = 0, vec_full = 0;
    };
    std::vector<CoreCounts> counts(plan.size());
    auto encode_core = [&](size_t c) {
        CoreCounts& cc = counts[c];
        size_t base = static_cast<size_t>(first_page[c]) * (kPageBytes / 4);
        for (const auto& [tx, ty] : plan[c].tiles) {
            const auto& entries = bins.tile(static_cast<uint32_t>(tx), static_cast<uint32_t>(ty));
            words[base + kHdrRecords] = static_cast<uint32_t>(entries.size());
            words[base + kHdrTileX] = static_cast<uint32_t>(tx);
            words[base + kHdrTileY] = static_cast<uint32_t>(ty);
            for (size_t i = 0; i < entries.size(); i++) {
                const SetupTriangle& tri = setup.tris[entries[i].tri];
                const TileParams p = make_tile_params(tri, tx, ty);
                uint64_t vc[2] = {0, 0};
                encode_record(words, base + (i + 1) * kRecordWords, tri, p, entries[i].coverage, tx, ty, vc);
                cc.vec_partial += vc[0];
                cc.vec_full += vc[1];
            }
            cc.records += entries.size();
            base += static_cast<size_t>(pages_for_records(static_cast<uint32_t>(entries.size()))) * (kPageBytes / 4);
        }
    };
    {
        const size_t hw = opt.host_threads != 0 ? opt.host_threads : std::thread::hardware_concurrency();
        const size_t n_threads = std::max<size_t>(1, std::min<size_t>(plan.size(), hw));
        std::atomic<size_t> next{0};
        std::vector<std::thread> threads;
        for (size_t t = 0; t < n_threads; t++) {
            threads.emplace_back([&] {
                for (size_t c = next++; c < plan.size(); c = next++) {
                    encode_core(c);
                }
            });
        }
        for (auto& th : threads) {
            th.join();
        }
    }
    for (const CoreCounts& cc : counts) {
        out.times.records += cc.records;
        out.times.vec_partial += cc.vec_partial;
        out.times.vec_full += cc.vec_full;
    }
    if (words.empty()) {
        return out;
    }
    out.times.build_ms = ms_since(t0);

    // ---- 2. Buffers, program --------------------------------------------------------------------
    t0 = Clock::now();
    auto& cq = mesh_->mesh_command_queue();
    const uint32_t total_pages = static_cast<uint32_t>(words.size() / (kPageBytes / 4));
    const uint32_t planes = opt.debug_planes ? kOutPlanes : 1u;
    distributed::DeviceLocalBufferConfig cfg{.page_size = kPageBytes, .buffer_type = BufferType::DRAM};
    auto in_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = static_cast<uint64_t>(total_pages) * kPageBytes}, cfg, mesh_.get());
    auto out_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = static_cast<uint64_t>(blocks_total) * planes * kTileBytes}, cfg,
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
    if (planes > 1) {
        make_cb(tt::CBIndex::c_17, 2);
        make_cb(tt::CBIndex::c_18, 2);
    }

    // TT_RASTER_KERNEL_DIR overrides the kernel directory (A/B runs of older kernel versions).
    const char* kdir_env = std::getenv("TT_RASTER_KERNEL_DIR");
    const std::string kdir = kdir_env != nullptr ? std::string(kdir_env) + "/" : std::string(TT_RASTER_TENSIX_KERNELS);
    std::vector<uint32_t> reader_args, writer_args;
    TensorAccessorArgs(*in_buf->get_backing_buffer()).append_to(reader_args);
    TensorAccessorArgs(*out_buf->get_backing_buffer()).append_to(writer_args);
    auto reader = CreateKernel(
        program, kdir + "reader.cpp", core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_args});
    std::map<std::string, std::string> defines = opt.defines;
    defines["RASTER_OUT_PLANES"] = std::to_string(planes);
    auto writer = CreateKernel(
        program, kdir + "writer.cpp", core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_args,
            .defines = defines});

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
             float_bits(frame.clear_color[1]), float_bits(frame.clear_color[2]), float_bits(frame.clear_color[3]),
             pack_rgba8(frame.clear_color)});
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
    std::vector<uint32_t>& raw = raw_;
    distributed::EnqueueReadMeshBuffer(cq, raw, out_buf, true);
    // Untile: a tile holds four 16x16 faces (TL, TR, BL, BR), each row-major, so every face row is a
    // 16-word copy into the frame. Colour is a plain copy; depth is a plain copy; the id plane is
    // stored as floats and converted.
    uint32_t block = 0;
    for (size_t c = 0; c < plan.size(); c++) {
        for (const auto& [tx, ty] : plan[c].tiles) {
            for (uint32_t p = 0; p < planes; p++) {
                const uint32_t* tile = &raw[(static_cast<size_t>(block) * planes + p) * kTileWords];
                for (uint32_t face = 0; face < 4; face++) {
                    for (uint32_t r = 0; r < 16; r++) {
                        const uint32_t* src = tile + face * 256 + r * 16;
                        const size_t idx = fb.index(tx * 32 + (face & 1) * 16, ty * 32 + (face >> 1) * 16 + r);
                        if (p == kPlaneColor) {
                            std::memcpy(&fb.color[idx], src, 64);
                        } else if (p == kPlaneDepth) {
                            std::memcpy(&fb.depth[idx], src, 64);
                        } else {
                            for (uint32_t i = 0; i < 16; i++) {
                                float id;
                                std::memcpy(&id, &src[i], 4);
                                if (opt.cover_count) {
                                    fb.cover[idx + i] = static_cast<uint8_t>(std::min(255.0f, id));
                                } else {
                                    fb.id[idx + i] = static_cast<uint32_t>(id);
                                }
                            }
                        }
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
