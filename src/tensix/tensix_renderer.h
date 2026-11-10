// SPDX-License-Identifier: Apache-2.0
//
// Host side of the Tensix rasterizer (docs/PLAN.md, Phase 2): turns the shared setup and binning
// output into per-core DRAM work lists, runs the kernels, and reads the planes back into a
// Framebuffer that can be compared with the CPU reference.

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

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
};

struct DeviceTimes {
    double build_ms = 0;     // host: encode records, build the work lists
    double upload_ms = 0;    // host to device DRAM
    double device_ms = 0;    // enqueue, run, finish (includes kernel launch; JIT cached after the first run)
    double readback_ms = 0;  // device DRAM to host, untilize into the framebuffer
    uint64_t records = 0;
    uint64_t blocks = 0;
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

    DeviceFrame render(
        const Frame& frame, const SetupResult& setup, const TileBins& bins, const std::vector<CorePlan>& plan,
        const DeviceOptions& opt = {});

private:
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_;
};

// Plans: every tile of the target, split over `cores` in contiguous row-major chunks.
std::vector<CorePlan> plan_row_major(
    const std::vector<tt::tt_metal::CoreCoord>& cores, uint32_t tiles_x, uint32_t tiles_y);

}  // namespace ttr
