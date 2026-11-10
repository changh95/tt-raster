// SPDX-License-Identifier: Apache-2.0
//
// Several chips, one frame (docs/PLAN.md, Phase 2.6): the target's tiles are split into
// cost-balanced contiguous bands, one per chip; every chip draws its band exactly as a single chip
// would (src/tensix/tensix_renderer.h) and the host assembles the frame. Chips are opened as
// independent unit meshes and driven from one host thread each; no chip-to-chip traffic is used.

#pragma once

#include <memory>
#include <vector>

#include "tensix/tensix_renderer.h"

namespace ttr {

struct MultiChipTimes {
    double wall_ms = 0;                 // whole frame: partition, per-chip render in parallel, merge
    double partition_ms = 0;
    double merge_ms = 0;
    std::vector<DeviceTimes> chip;      // per chip, same fields as a single-chip frame
    std::vector<uint64_t> chip_tiles;
    std::vector<uint64_t> chip_records;
};

// The frame lives in the renderer and is overwritten by the next render() call.
struct MultiChipFrame {
    const Framebuffer* fb = nullptr;
    MultiChipTimes times;
};

class MultiChipRenderer {
public:
    // Opens unit meshes for device ids 0 .. n_chips - 1 (as numbered inside this process, i.e. after
    // TT_VISIBLE_DEVICES).
    explicit MultiChipRenderer(size_t n_chips);
    ~MultiChipRenderer();

    size_t chips() const { return active_; }              // chips used by render()
    size_t opened() const { return renderers_.size(); }
    // Use only the first n opened chips (for scaling measurements without reopening devices).
    void set_active(size_t n) { active_ = n < renderers_.size() ? n : renderers_.size(); }
    TensixRenderer& chip(size_t i) { return *renderers_[i]; }

    // Chips write their tiles straight into one shared framebuffer (disjoint tiles), so a frame needs
    // no per-chip allocation and no merge copy.
    MultiChipFrame render(const Frame& frame, const SetupResult& setup, const TileBins& bins, const DeviceOptions& opt = {});

private:
    size_t active_ = 0;
    Framebuffer fb_;
    std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> meshes_;
    std::vector<std::unique_ptr<TensixRenderer>> renderers_;
};

}  // namespace ttr
