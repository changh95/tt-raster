// SPDX-License-Identifier: Apache-2.0

#include "tensix/scene_backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace ttr {

using Clock = std::chrono::steady_clock;

const Framebuffer& TensixSceneBackend::render(const Frame& frame, uint32_t planes) {
    auto t0 = Clock::now();
    const SetupResult setup = setup_frame(frame);
    last_.setup_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    t0 = Clock::now();
    const TileBins bins = bin_triangles(setup, frame.tiles_x(), frame.tiles_y());
    last_.bin_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    last_.records = bins.entries;

    DeviceOptions opt;
    // Map the requested planes to the device's: colour 1, depth 2, id 4; float colour adds G and B.
    opt.float_planes = (planes & scene::kPlaneFloat) != 0;
    opt.plane_mask = ((planes & scene::kPlaneColor) ? 1u : 0u) | ((planes & scene::kPlaneDepth) ? 2u : 0u) |
                     ((planes & scene::kPlaneId) ? 4u : 0u) | (opt.float_planes ? 8u | 16u : 0u);
    t0 = Clock::now();
    const MultiChipFrame f = mc_.render(frame, setup, bins, opt);
    last_.device_wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    last_.chips = f.times;
    return *f.fb;
}

std::string TensixSceneBackend::report() const {
    double b = 0, u = 0, d = 0, r = 0;
    for (const DeviceTimes& t : last_.chips.chip) {
        b = std::max(b, t.build_ms);
        u = std::max(u, t.upload_ms);
        d = std::max(d, t.device_ms);
        r = std::max(r, t.readback_ms);
    }
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "setup %.1f bin %.1f | device wall %.1f = slowest chip: encode %.1f upload %.1f device %.1f readback %.1f ms | %llu records",
                  last_.setup_ms, last_.bin_ms, last_.device_wall_ms, b, u, d, r, static_cast<unsigned long long>(last_.records));
    return buf;
}

std::unique_ptr<scene::Backend> make_tensix_backend(size_t n_chips) { return std::make_unique<TensixSceneBackend>(n_chips); }

}  // namespace ttr
