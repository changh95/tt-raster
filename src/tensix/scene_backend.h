// SPDX-License-Identifier: Apache-2.0
//
// Tensix backend for the scene layer (src/scene/scene.h): rasterizes the atlas frames on one or more
// chips and hands the framebuffer back.

#pragma once

#include <memory>
#include <string>

#include "scene/scene.h"
#include "tensix/multichip.h"

namespace ttr {

class TensixSceneBackend : public scene::Backend {
public:
    explicit TensixSceneBackend(size_t n_chips) : mc_(n_chips) {}
    const Framebuffer& render(const Frame& frame, uint32_t planes) override;
    std::string report() const override;

    // Phase timings of the most recent render() call.
    struct Timings {
        double setup_ms = 0, bin_ms = 0, device_wall_ms = 0;
        MultiChipTimes chips;
        uint64_t records = 0;
    };
    const Timings& last() const { return last_; }
    MultiChipRenderer& chips() { return mc_; }

private:
    MultiChipRenderer mc_;
    Timings last_;
    // Kept between frames so a render loop reuses the (tens of MB) triangle and bin storage.
    SetupResult setup_;
    SetupWorkspace setup_ws_;
    TileBins bins_;
};

std::unique_ptr<scene::Backend> make_tensix_backend(size_t n_chips);

}  // namespace ttr
