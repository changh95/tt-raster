// SPDX-License-Identifier: Apache-2.0
//
// Tensix backend (ttraster/backend.h): rasterizes frames on one or more chips and hands the framebuffer back.

#pragma once

#include <memory>
#include <string>

#include "ttraster/backend.h"
#include "tensix/chip_lock.h"
#include "tensix/multichip.h"

namespace ttr {

class TensixBackend : public Backend {
public:
    explicit TensixBackend(size_t n_chips) : lock_(n_chips), mc_(n_chips) {}
    const Framebuffer& render(const Frame& frame, uint32_t planes, int slot = 0) override;
    std::string report(int slot = 0) const override;

    // Phase timings of the most recent render() call (of the given slot).
    struct Timings {
        double setup_ms = 0, bin_ms = 0, device_wall_ms = 0;
        MultiChipTimes chips;
        uint64_t records = 0;
    };
    const Timings& last(int slot = 0) const { return slots_[slot].last; }
    MultiChipRenderer& chips() { return mc_; }

private:
    ChipLock lock_;   // before mc_: refuse (throw) before the devices are opened
    MultiChipRenderer mc_;
    // Per in-flight frame: kept between frames so a render loop reuses the (tens of MB) triangle and bin
    // storage, and so that two frames can be prepared at the same time.
    struct SlotState {
        Timings last;
        SetupResult setup;
        SetupWorkspace setup_ws;
        TileBins bins;
    };
    SlotState slots_[kRenderSlots];
};

std::unique_ptr<Backend> make_tensix_backend(size_t n_chips);

}  // namespace ttr
