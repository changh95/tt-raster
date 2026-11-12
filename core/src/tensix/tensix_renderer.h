// SPDX-License-Identifier: Apache-2.0
//
// Host side of the Tensix rasterizer (docs/PLAN.md, Phase 2): turns the shared setup and binning
// output into per-core DRAM work lists, runs the kernels, and reads the planes back into a
// Framebuffer that can be compared with the CPU reference.

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <thread>
#include <set>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

#include "tensix/shader_codegen.h"
#include "tensix/shader_warm.h"
#include "ttraster/reference.h"
#include "ttraster/setup.h"
#include "ttraster/tile.h"

namespace ttr {

// The tiles one core draws, in order.
struct CorePlan {
    tt::tt_metal::CoreCoord core;                      // logical worker core
    std::vector<std::pair<int, int>> tiles;            // (tx, ty)
};

struct DeviceOptions {
    // Debug: skip the depth test; the id plane becomes the per-pixel coverage count.
    bool cover_count = false;
    // Also write back depth and triangle id (three planes instead of one). Off for production
    // runs: colour is a third of the read-back traffic.
    bool debug_planes = true;
    // Five planes without quantization: R, depth, id, G, B as floats (Framebuffer::plane[0..2]),
    // for outputs that need more than 8 bits, such as normals. Implies debug_planes.
    bool float_planes = false;
    // Which planes to read back (bits: 1 colour, 2 depth, 4 triangle id, 8 G, 16 B; see the compute
    // kernel). 0 = derive from debug_planes / float_planes. Planes not requested are not computed or
    // transferred: a depth-and-id request skips colour entirely.
    uint32_t plane_mask = 0;
    // Compile the fragment shader programs of the frame to straight SFPI code instead of interpreting them (shader_codegen.h). The
    // kernel is built for the set of programs (about a minute the first time, then cached by tt-metal); also switched on by the
    // environment variable TTR_SHADER_COMPILE=1.
    bool compile_shaders = false;
    // Code generation level (shader_codegen.h): 2 keeps values in local registers across instructions, 1 does not. render() lowers
    // it when a kernel fails to build.
    uint32_t compile_level = 2;
    // Never compile (set when a kernel with compiled shaders failed to build and the frame is run again with the interpreter).
    bool no_compile = false;
    // Which programs are compiled and how much code is allowed; unset = the defaults, adjusted by TTR_SHADER_COMPILE_* (tests). A
    // replay of a recorded program set (shader_warm.h) sets it to what the recording used.
    std::optional<CodegenOptions> codegen;
    // Extra compile-time defines for the kernels, for experiments (name -> value).
    std::map<std::string, std::string> defines;
    // Host threads used to encode the work lists; 0 = one per hardware thread.
    size_t host_threads = 0;
    // Which of the renderer's kRenderSlots sets of per-frame buffers this frame uses. Frames in flight at
    // the same time (different threads) need different slots.
    int slot = 0;
};

inline uint32_t effective_plane_mask(const DeviceOptions& o) {
    return o.plane_mask != 0 ? o.plane_mask : (o.float_planes ? 0x1Fu : (o.debug_planes ? 0x07u : 0x01u));
}

struct DeviceTimes {
    double build_ms = 0;     // host: encode records, build the work lists
    double upload_ms = 0;    // host to device DRAM
    double device_ms = 0;    // enqueue, run, finish (includes kernel launch; JIT cached after the first run)
    double readback_ms = 0;  // device DRAM to host, untilize into the framebuffer
    double read_ms = 0;      // of which: the device-to-host transfer alone (the rest is untilizing)
    uint64_t records = 0;
    uint64_t blocks = 0;
    uint64_t vec_partial = 0, vec_full = 0;  // SFPU vectors the kernel will process (rejected ones are skipped)
};

struct DeviceFrame {
    Framebuffer fb;                          // only the planned tiles are filled; the rest holds clear values
    std::vector<uint8_t> tile_rendered;      // per tile of the target: 1 if a plan covered it
    DeviceTimes times;
};

class TensixRenderer {
public:
    explicit TensixRenderer(std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh) : mesh_(std::move(mesh)) {}

    // All worker cores of the device as logical coordinates, row-major.
    std::vector<tt::tt_metal::CoreCoord> worker_cores() const;

    // Draw the planned tiles. With `target` (a Framebuffer the caller has sized to the padded target)
    // the pixels of the planned tiles are written there and nothing is allocated per frame;
    // otherwise a framebuffer pre-filled with clear values is returned in DeviceFrame::fb.
    DeviceFrame render(
        const Frame& frame, const SetupResult& setup, const TileBins& bins, const std::vector<CorePlan>& plan,
        const DeviceOptions& opt = {}, Framebuffer* target = nullptr);

    // Build (or find in tt-metal's cache) the kernel of a recorded program set by drawing one small frame with it (shader_warm.h).
    // Returns the host time of that frame in seconds; the set may end up on a lower code generation level or on the interpreter,
    // as it would in a real frame.
    double prewarm(const ShaderSetRecord& record);

private:
    DeviceFrame render_impl(
        const Frame& frame, const SetupResult& setup, const TileBins& bins, const std::vector<CorePlan>& plan, const DeviceOptions& opt,
        Framebuffer* target);
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_;
    // Scratch reused from frame to frame: fresh multi-megabyte allocations cost page faults, which
    // serialize badly when several chips are driven from one process.
    std::vector<uint32_t> words_[kRenderSlots], raw_[kRenderSlots];
    // One frame at a time uses the chip (buffers, program, upload, run, read-back); encoding and untiling of
    // frames in other slots overlap with it.
    std::mutex device_mutex_;
    // Program sets (a key over the programs and a code generation level) whose compiled kernel did not build: not tried again.
    std::mutex failed_mutex_;
    std::set<uint64_t> failed_;
    std::set<uint64_t> failed_checked_;   // sets whose entry in the cache directory's list was looked up (once per process)
};

// Plans: every tile of the target, split over `cores` in contiguous row-major chunks.
std::vector<CorePlan> plan_row_major(
    const std::vector<tt::tt_metal::CoreCoord>& cores, uint32_t tiles_x, uint32_t tiles_y);

// Tiles in row-major order, as (tx, ty).
std::vector<std::pair<int, int>> all_tiles(uint32_t tiles_x, uint32_t tiles_y);

// Estimated cost of drawing a tile, in arbitrary units: a fixed part per tile plus one per triangle
// record. The busiest core sets the frame time, so plans balance this, not the tile count.
double tile_weight(const TileBins& bins, int tx, int ty);

// Split `tiles` into `parts` contiguous chunks (order preserved) whose summed weights are as equal
// as a contiguous split allows.
std::vector<std::vector<std::pair<int, int>>> split_balanced(
    const std::vector<std::pair<int, int>>& tiles, const TileBins& bins, size_t parts);

// Balanced plan of `tiles` over `cores`.
std::vector<CorePlan> plan_balanced(
    const std::vector<tt::tt_metal::CoreCoord>& cores, const std::vector<std::pair<int, int>>& tiles,
    const TileBins& bins);

}  // namespace ttr
