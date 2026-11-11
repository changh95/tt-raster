// SPDX-License-Identifier: Apache-2.0

#include "tensix/tensix_renderer.h"
#include "tensix/shader_codegen.h"

#include "ttraster/parallel.h"
#include "ttraster/trace.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>

#include <unistd.h>

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

// Slots a (tile, triangle) record takes in a page: one, plus the extension slots of a shader triangle's varyings.
uint32_t record_slots(const Frame& frame, const SetupTriangle& tri) {
    return tri.shader != 0 ? 1 + layout::ext_slots(frame.shaders[tri.shader - 1].num_inputs) : 1;
}

// Where the records of a tile go in a shader frame: slot positions (page * kRecordsPerPage + slot, the header taking slot 0
// of page 0), padding so that no record straddles a page. Returns the pages of the block.
uint32_t layout_shader_block(const Frame& frame, const SetupResult& setup, const std::vector<BinEntry>& entries,
                             std::vector<uint32_t>* position) {
    using namespace layout;
    uint32_t page = 0, slot = 1;
    if (position != nullptr) position->resize(entries.size());
    for (size_t i = 0; i < entries.size(); i++) {
        const uint32_t n = record_slots(frame, setup.tris[entries[i].tri]);
        if (slot + n > kRecordsPerPage) {
            page++;
            slot = 0;
        }
        if (position != nullptr) (*position)[i] = page * kRecordsPerPage + slot;
        slot += n;
    }
    return page + 1;
}

// The program table of a frame (raster_layout.h): directory, then code and constants of every program.
std::vector<uint32_t> build_program_table(const Frame& frame) {
    using namespace layout;
    const size_t n = frame.shaders.size();
    std::vector<uint32_t> t(kTabDirectory + n * kTabEntryWords, 0);
    t[0] = static_cast<uint32_t>(n);
    for (size_t k = 0; k < n; k++) {
        const ShaderProgram& p = frame.shaders[k];
        const size_t dir = kTabDirectory + k * kTabEntryWords;
        t[dir + 0] = static_cast<uint32_t>(t.size());
        t[dir + 1] = static_cast<uint32_t>(p.code.size());
        for (const ShaderInstr& in : p.code) {
            for (uint32_t w : in.w) t.push_back(w);
        }
        t[dir + 2] = static_cast<uint32_t>(t.size());
        for (float c : p.consts) t.push_back(float_bits(c));
        t[dir + 3] = p.num_inputs;
        t[dir + 4] = static_cast<uint32_t>(p.fragcoord_reg + 1);
        t[dir + 5] = p.out_reg;
        t[dir + 6] = p.uses_kill() ? 1u : 0u;
    }
    return t;
}

// Encode one (tile, triangle) record at word offset `at` of `w`. `shaded`: the frame has shader triangles, so every record
// carries its depth and blend state in the flags and the kernel's shader path draws it.
void encode_record(
    std::vector<uint32_t>& w, size_t at, const Frame& frame, const SetupResult& setup, const SetupTriangle& tri,
    const TileParams& p, TileCoverage cov, int tx, int ty, DepthFunc frame_depth, bool shaded, uint64_t* vec_counts = nullptr) {
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
    if (p.perspective) {
        put_float(w, at + kWPlane + 0, p.w_c);
        put_float(w, at + kWPlane + 1, p.w_gx);
        put_float(w, at + kWPlane + 2, p.w_gy);
    }
    put_float(w, at + kId, static_cast<float>(tri.source + 1));
    w[at + kFlatPacked] = pack_rgba8(p.color_c);  // meaningful for flat triangles (gradients are zero)
    uint32_t flags = (tri.flat ? kFlagFlat : 0u) | (cov == TileCoverage::Full ? kFlagFull : 0u) | (p.perspective ? kFlagPersp : 0u);
    if (tri.shader != 0) {
        // Varyings: the first four planes where the colour planes are, the rest in the extension slots; the rhw plane
        // (always: gl_FragCoord.w and the perspective divisor) in its usual place.
        const ShaderProgram& prog = frame.shaders[tri.shader - 1];
        for (uint32_t k = 0; k < prog.num_inputs; k++) {
            const TilePlane tp = make_tile_plane(setup.attr_planes[tri.plane_base + k], tx, ty);
            const size_t o = k < kInputsInRecord ? at + kColor + 3 * k : at + kRecordWords + 3 * (k - kInputsInRecord);
            put_float(w, o, tp.c);
            put_float(w, o + 1, tp.gx);
            put_float(w, o + 2, tp.gy);
        }
        const TilePlane wp = make_tile_plane(tri.inv_w, tx, ty);
        put_float(w, at + kWPlane + 0, wp.c);
        put_float(w, at + kWPlane + 1, wp.gx);
        put_float(w, at + kWPlane + 2, wp.gy);
        w[at + kProgramId] = tri.shader - 1;
        flags |= kFlagShader | (ext_slots(prog.num_inputs) << kExtShift);
    }
    if (shaded || tri.depth_func != frame_depth || !tri.depth_write || tri.blend != BlendMode::None) {
        // Own depth/blend state: the kernel's general path decodes it (the common case keeps the specialized path).
        flags |= kFlagStateful | (static_cast<uint32_t>(tri.depth_func) << kDepthFuncShift) |
                 (tri.depth_write ? kFlagDepthWrite : 0u) | (static_cast<uint32_t>(tri.blend) << kBlendShift);
    }
    w[at + kFlags] = flags;
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

namespace {

bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// A key over the programs of a frame (code, register counts, interface), for remembering which program sets did not build.
uint64_t shader_set_key(const std::vector<ShaderProgram>& programs) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    for (const ShaderProgram& p : programs) {
        mix(p.num_inputs); mix(p.num_regs); mix(p.out_reg); mix(static_cast<uint64_t>(p.fragcoord_reg) + 7);
        for (const ShaderInstr& in : p.code) for (uint32_t w : in.w) mix(w);
        mix(0xfeedu);
    }
    return h & ~3ull;   // the low two bits carry the level
}

// A copy of the kernel directory plus the generated shader header, in a cache directory named after the contents (tt-metal caches the
// kernel binaries by their source, so a program set that was built before costs nothing). TT_RASTER_SHADER_CACHE overrides the
// location (default ~/.cache/tt-raster/shader-kernels).
std::string prepare_kernel_dir(const std::string& kernel_dir, const std::string& header) {
    namespace fs = std::filesystem;
    uint64_t h = 1469598103934665603ull;   // FNV-1a over the header and the kernel directory name
    for (char c : header + kernel_dir) h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ull;
    // The kernel sources are part of the key too: an edit of a kernel must not reuse an old copy.
    for (const auto& e : fs::directory_iterator(kernel_dir)) {
        if (!e.is_regular_file()) continue;
        std::ifstream f(e.path(), std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        for (char c : text) h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ull;
    }
    const char* root_env = std::getenv("TT_RASTER_SHADER_CACHE");
    const fs::path root = root_env != nullptr ? fs::path(root_env)
                                              : fs::path(std::getenv("HOME") != nullptr ? std::getenv("HOME") : "/tmp") / ".cache" / "tt-raster" / "shader-kernels";
    char name[32];
    std::snprintf(name, sizeof name, "%016llx", static_cast<unsigned long long>(h));
    const fs::path dir = root / name;
    if (!fs::exists(dir / "shader_compiled.h")) {
        const fs::path tmp = root / (std::string(name) + ".tmp" + std::to_string(::getpid()));
        fs::create_directories(tmp);
        for (const auto& e : fs::directory_iterator(kernel_dir)) {
            if (e.is_regular_file()) fs::copy_file(e.path(), tmp / e.path().filename(), fs::copy_options::overwrite_existing);
        }
        std::ofstream(tmp / "shader_compiled.h") << header;
        std::error_code ec;
        fs::rename(tmp, dir, ec);   // another process may have won the race: then its copy is as good as ours
        if (ec) fs::remove_all(tmp);
    }
    return dir.string() + "/";
}

}  // namespace

namespace {

struct RasterKernels {
    tt::tt_metal::KernelHandle reader, writer, compute;
};

// The three kernels of a frame (reader, writer, compute) with their compile-time arguments and defines; with compiled shaders the compute
// kernel comes from the generated copy of the kernel directory. 
RasterKernels create_raster_kernels(
    Program& program, const CoreRangeSet& core_set, const TensorAccessorArgs& in_args, const TensorAccessorArgs& out_args, const Frame& frame,
    uint32_t mask, bool shaded, const DeviceOptions& opt) {
    using namespace tt::tt_metal;
    // TT_RASTER_KERNEL_DIR overrides the kernel directory (A/B runs of older kernel versions).
    const char* kdir_env = std::getenv("TT_RASTER_KERNEL_DIR");
    const std::string kdir = kdir_env != nullptr ? std::string(kdir_env) + "/" : std::string(TT_RASTER_TENSIX_KERNELS);
    std::vector<uint32_t> reader_args, writer_args;
    in_args.append_to(reader_args);
    out_args.append_to(writer_args);
    std::map<std::string, std::string> reader_defines;
    if (shaded) reader_defines["RASTER_SHADERS"] = "1";
    auto reader = CreateKernel(
        program, kdir + "reader.cpp", core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_args,
            .defines = reader_defines});
    std::map<std::string, std::string> defines = opt.defines;
    if (shaded) {
        defines["RASTER_SHADERS"] = "1";
        defines["RASTER_PACKED_COLOUR"] = "1";   // colour in one Dest slot frees three tiles for shader registers
    }
    defines["RASTER_PLANE_MASK"] = std::to_string(mask);
    if (opt.float_planes) {
        defines["RASTER_FLOAT_PLANES"] = "1";
    }
    auto writer = CreateKernel(
        program, kdir + "writer.cpp", core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_args,
            .defines = defines});

    if (frame.depth_func == DepthFunc::LessEqual) {   // the specialized path's comparison; Always is per-triangle state
        defines["RASTER_LESS_EQUAL"] = "1";
    }
    if (opt.cover_count) {
        defines["RASTER_COVER_COUNT"] = "1";
    }
    std::string compute_dir = kdir;
    if (shaded && !opt.no_compile && (opt.compile_shaders || env_flag("TTR_SHADER_COMPILE"))) {
        uint32_t n_compiled = 0;
        CodegenOptions copt;
        copt.level = opt.compile_level;
        if (const char* v = std::getenv("TTR_SHADER_COMPILE_BYTES")) copt.max_code_bytes = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));   // tests
        if (const char* v = std::getenv("TTR_SHADER_COMPILE_MIN")) copt.min_instructions = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
        if (const char* v = std::getenv("TTR_SHADER_COMPILE_PROGRAMS")) copt.max_programs = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
        const std::string header = generate_shader_header(frame.shaders, copt, &n_compiled);
        if (n_compiled > 0) {
            compute_dir = prepare_kernel_dir(kdir, header);
            defines["RASTER_COMPILED_SHADERS"] = "1";
            if (const char* v = std::getenv("TTR_SHADER_COMPILE"); v != nullptr && v[0] == '2') {
                std::fprintf(stderr, "tensix: compiled %u of %zu shader programs, kernels in %s\n", n_compiled, frame.shaders.size(), compute_dir.c_str());
            }
        }
    }
    auto compute = CreateKernel(
        program, compute_dir + "compute.cpp", core_set,
        ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .defines = defines});

    return {reader, writer, compute};
}

}  // namespace

DeviceFrame TensixRenderer::render(
    const Frame& frame, const SetupResult& setup, const TileBins& bins, const std::vector<CorePlan>& plan,
    const DeviceOptions& opt, Framebuffer* target) {
    const bool compiling = !frame.shaders.empty() && !opt.no_compile && (opt.compile_shaders || env_flag("TTR_SHADER_COMPILE"));
    if (!compiling) return render_impl(frame, setup, bins, plan, opt, target);
    // Compiled kernels may not build (the code does not fit the core's kernel text, or the compiler runs out of local registers):
    // try level 2 (values kept in local registers), then level 1 (everything through Dest), then the interpreter, and remember what failed.
    const uint64_t key = shader_set_key(frame.shaders);
    for (uint32_t level = std::min<uint32_t>(opt.compile_level, 2); level >= 1; level--) {
        {
            std::lock_guard<std::mutex> lock(failed_mutex_);
            if (failed_.count(key | level) != 0) continue;
        }
        try {
            DeviceOptions o = opt;
            o.compile_level = level;
            return render_impl(frame, setup, bins, plan, o, target);
        } catch (const std::runtime_error& e) {
            std::string why = e.what();
            why = why.substr(0, why.find('\n'));
            std::fprintf(stderr, "tensix: kernel with compiled shaders (level %u) failed (%s); trying the next level\n", level, why.c_str());
            std::lock_guard<std::mutex> lock(failed_mutex_);
            failed_.insert(key | level);
        }
    }
    DeviceOptions plain = opt;
    plain.no_compile = true;
    return render_impl(frame, setup, bins, plan, plain, target);
}

DeviceFrame TensixRenderer::render_impl(
    const Frame& frame, const SetupResult& setup, const TileBins& bins, const std::vector<CorePlan>& plan,
    const DeviceOptions& opt, Framebuffer* target) {
    using namespace layout;
    const bool shaded = !frame.shaders.empty();
    if (shaded) {
        const uint32_t m = effective_plane_mask(opt);
        if (opt.float_planes || !(m & 1) || opt.cover_count) {
            throw std::runtime_error("frames with fragment shaders need the colour plane, no float planes and no coverage count");
        }
        if (frame.shaders.size() > 65535) throw std::runtime_error("too many shader programs");
    }
    DeviceFrame out;
    const uint32_t pw = frame.padded_width(), ph = frame.padded_height();
    if (target == nullptr) {
        out.fb.width = pw;
        out.fb.height = ph;
        const size_t n_px = static_cast<size_t>(pw) * ph;
        const uint32_t m = effective_plane_mask(opt);
        if ((m & 1) && !opt.float_planes) {
            out.fb.color.assign(n_px, pack_rgba8(frame.clear_color));
        }
        if (opt.float_planes) {
            for (int c = 0; c < 3; c++) {
                out.fb.plane[c].assign(n_px, frame.clear_color[c]);
            }
        }
        if (m & 2) {
            out.fb.depth.assign(n_px, frame.clear_depth);
        }
        if (m & 4) {
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
    std::vector<uint32_t> table;   // shader frames: the program table goes first in the buffer, cores' pages after it
    uint32_t table_pages = 0;
    if (shaded) {
        table = build_program_table(frame);
        table_pages = static_cast<uint32_t>((table.size() + kTileWords - 1) / kTileWords);
        if (table_pages > kMaxTablePages) {
            throw std::runtime_error("the shader programs of the frame do not fit the program table (" + std::to_string(kMaxTablePages) + " pages)");
        }
    }
    std::vector<uint32_t> first_page(plan.size()), n_pages(plan.size()), first_block(plan.size());
    uint32_t blocks_total = 0, pages_total = table_pages;
    for (size_t c = 0; c < plan.size(); c++) {
        first_page[c] = pages_total;
        first_block[c] = blocks_total;
        for (const auto& [tx, ty] : plan[c].tiles) {
            const auto& entries = bins.tile(tx, ty);
            pages_total += shaded ? layout_shader_block(frame, setup, entries, nullptr)
                                  : pages_for_records(static_cast<uint32_t>(entries.size()));
            out.tile_rendered[static_cast<size_t>(ty) * frame.tiles_x() + tx] = 1;
        }
        n_pages[c] = pages_total - first_page[c];
        blocks_total += static_cast<uint32_t>(plan[c].tiles.size());
        out.times.blocks += plan[c].tiles.size();
    }
    // Only headers and the records' words are written; padding is never read by the kernels, so the
    // buffer is not cleared between frames.
    std::vector<uint32_t>& words = words_[opt.slot];
    words.resize(static_cast<size_t>(pages_total) * (kPageBytes / 4));
    std::copy(table.begin(), table.end(), words.begin());
    // The specialized kernel path compares with LESS, or LESS_EQUAL when the frame says so (RASTER_LESS_EQUAL);
    // a triangle with any other depth function takes the general path.
    const DepthFunc kernel_depth = frame.depth_func == DepthFunc::LessEqual ? DepthFunc::LessEqual : DepthFunc::Less;
    struct CoreCounts {
        uint64_t records = 0, vec_partial = 0, vec_full = 0;
    };
    std::vector<CoreCounts> counts(plan.size());
    auto encode_core = [&](size_t c) {
        CoreCounts& cc = counts[c];
        size_t base = static_cast<size_t>(first_page[c]) * (kPageBytes / 4);
        std::vector<uint32_t> position;
        for (const auto& [tx, ty] : plan[c].tiles) {
            const auto& entries = bins.tile(static_cast<uint32_t>(tx), static_cast<uint32_t>(ty));
            const uint32_t block_pages = shaded ? layout_shader_block(frame, setup, entries, &position)
                                                : pages_for_records(static_cast<uint32_t>(entries.size()));
            words[base + kHdrRecords] = static_cast<uint32_t>(entries.size());
            words[base + kHdrTileX] = static_cast<uint32_t>(tx);
            words[base + kHdrTileY] = static_cast<uint32_t>(ty);
            words[base + kHdrPages] = block_pages;
            put_float(words, base + kHdrCx, static_cast<float>(tx * kTileSize + kTileSize / 2));
            put_float(words, base + kHdrCy, static_cast<float>(ty * kTileSize + kTileSize / 2));
            for (size_t i = 0; i < entries.size(); i++) {
                const SetupTriangle& tri = setup.tris[entries[i].tri];
                const TileParams p = make_tile_params(tri, tx, ty);
                uint64_t vc[2] = {0, 0};
                const size_t at = shaded ? base + static_cast<size_t>(position[i]) * kRecordWords : base + (i + 1) * kRecordWords;
                encode_record(words, at, frame, setup, tri, p, entries[i].coverage, tx, ty, kernel_depth, shaded, vc);
                if (shaded) {
                    // The next record starts on the next page: mark the unused tail of this one.
                    const uint32_t slot_end = position[i] % kRecordsPerPage + record_slots(frame, tri);
                    if (i + 1 < entries.size() && position[i + 1] / kRecordsPerPage != position[i] / kRecordsPerPage &&
                        slot_end < kRecordsPerPage) {
                        words[base + (position[i] - position[i] % kRecordsPerPage + slot_end) * kRecordWords + kFlags] = kPageEnd;
                    }
                }
                cc.vec_partial += vc[0];
                cc.vec_full += vc[1];
            }
            cc.records += entries.size();
            base += static_cast<size_t>(block_pages) * (kPageBytes / 4);
        }
    };
    parallel_for(plan.size(), opt.host_threads, [&](size_t c) { encode_core(c); });
    for (const CoreCounts& cc : counts) {
        out.times.records += cc.records;
        out.times.vec_partial += cc.vec_partial;
        out.times.vec_full += cc.vec_full;
    }
    if (words.empty()) {
        return out;
    }
    out.times.build_ms = ms_since(t0);

    const uint32_t mask = effective_plane_mask(opt);
    const uint32_t planes = static_cast<uint32_t>(__builtin_popcount(mask));
    std::vector<uint32_t>& raw = raw_[opt.slot];
    {   // The chip serves one frame at a time; the buffers and the program are destroyed inside the lock.
    trace("  device lock wait", opt.slot);
    std::lock_guard<std::mutex> device_lock(device_mutex_);
    trace("  device lock held", opt.slot);
    // ---- 2. Buffers, program --------------------------------------------------------------------
    t0 = Clock::now();
    auto& cq = mesh_->mesh_command_queue();
    const uint32_t total_pages = static_cast<uint32_t>(words.size() / (kPageBytes / 4));
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
    if (shaded) make_cb(tt::CBIndex::c_1, kMaxTablePages);   // the program table; a fixed size keeps the kernels the same for every frame
    for (uint32_t bit = 0; bit < 5; bit++) {   // one output buffer per requested plane: c_16 + bit
        if (mask & (1u << bit)) {
            make_cb(static_cast<tt::CBIndex>(static_cast<uint32_t>(tt::CBIndex::c_16) + bit), 2);
        }
    }

    const RasterKernels kernels = create_raster_kernels(
        program, core_set, TensorAccessorArgs(*in_buf->get_backing_buffer()), TensorAccessorArgs(*out_buf->get_backing_buffer()), frame, mask,
        shaded, opt);
    const auto reader = kernels.reader, writer = kernels.writer, compute = kernels.compute;

    for (size_t c = 0; c < plan.size(); c++) {
        const CoreCoord core = plan[c].core;
        SetRuntimeArgs(program, reader, core, {static_cast<uint32_t>(in_buf->address()), first_page[c], n_pages[c], table_pages});
        SetRuntimeArgs(
            program, writer, core,
            {static_cast<uint32_t>(out_buf->address()), first_block[c], static_cast<uint32_t>(plan[c].tiles.size())});
        SetRuntimeArgs(
            program, compute, core,
            {static_cast<uint32_t>(plan[c].tiles.size()), float_bits(frame.clear_depth), float_bits(frame.clear_color[0]),
             float_bits(frame.clear_color[1]), float_bits(frame.clear_color[2]), float_bits(frame.clear_color[3]),
             pack_rgba8(frame.clear_color), table_pages});
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
    distributed::EnqueueReadMeshBuffer(cq, raw, out_buf, true);
    out.times.read_ms = ms_since(t0);
    trace("  device section done", opt.slot);
    }   // device_lock released, buffers and program destroyed
    // Untile: a tile holds four 16x16 faces (TL, TR, BL, BR), each row-major, so every face row is a
    // 16-word copy into the frame. Colour is a plain copy; depth is a plain copy; the id plane is
    // stored as floats and converted.
    struct Job {
        uint32_t block;
        int tx, ty;
    };
    std::vector<Job> jobs;
    jobs.reserve(blocks_total);
    {
        uint32_t block = 0;
        for (size_t c = 0; c < plan.size(); c++) {
            for (const auto& [tx, ty] : plan[c].tiles) {
                jobs.push_back({block++, tx, ty});
            }
        }
    }
    auto decode = [&](const Job& job) {
        const uint32_t block = job.block;
        const int tx = job.tx, ty = job.ty;
        uint32_t rank = 0;
        for (uint32_t bit = 0; bit < 5; bit++) {
            if (!(mask & (1u << bit))) {
                continue;
            }
            const uint32_t* tile = &raw[(static_cast<size_t>(block) * planes + rank++) * kTileWords];
            for (uint32_t face = 0; face < 4; face++) {
                for (uint32_t r = 0; r < 16; r++) {
                    const uint32_t* src = tile + face * 256 + r * 16;
                    const size_t idx = fb.index(tx * 32 + (face & 1) * 16, ty * 32 + (face >> 1) * 16 + r);
                    if (bit == 0) {
                        std::memcpy(opt.float_planes ? static_cast<void*>(&fb.plane[0][idx]) : static_cast<void*>(&fb.color[idx]), src, 64);
                    } else if (bit == 3 || bit == 4) {
                        std::memcpy(&fb.plane[bit - 2][idx], src, 64);   // G -> plane[1], B -> plane[2]
                    } else if (bit == 1) {
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
    };
    {   // Blocks are independent tiles of the frame: untile them in groups on the host threads of this chip.
        constexpr size_t kGroup = 16;
        parallel_for((jobs.size() + kGroup - 1) / kGroup, opt.host_threads, [&](size_t g) {
            for (size_t i = g * kGroup; i < std::min(jobs.size(), (g + 1) * kGroup); i++) {
                decode(jobs[i]);
            }
        });
    }
    out.times.readback_ms = ms_since(t0);
    return out;
}

}  // namespace ttr
