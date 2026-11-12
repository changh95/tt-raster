// SPDX-License-Identifier: Apache-2.0

#include "tensix/tensix_backend.h"

#include "ttraster/trace.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

namespace ttr {

using Clock = std::chrono::steady_clock;

TensixBackend::TensixBackend(size_t n_chips) : lock_(n_chips), mc_(n_chips) {
    const char* manifest = std::getenv("TTR_SHADER_WARM");
    if (manifest == nullptr || manifest[0] == '\0') return;
    // The kernel cache does not depend on the chip, so chip 0 builds for all of them.
    std::string error;
    const std::vector<ShaderSetRecord> sets = read_shader_manifest(manifest, &error);
    if (!error.empty()) std::fprintf(stderr, "tensix: TTR_SHADER_WARM: %s\n", error.c_str());
    const auto t0 = Clock::now();
    for (const ShaderSetRecord& s : sets) {
        try {
            mc_.chip(0).prewarm(s);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "tensix: TTR_SHADER_WARM: a program set did not build (%s)\n", e.what());
        }
    }
    if (!sets.empty()) {
        std::fprintf(stderr, "tensix: warmed %zu compiled-shader program sets in %.1f s\n", sets.size(), std::chrono::duration<double>(Clock::now() - t0).count());
    }
}

const Framebuffer& TensixBackend::render(const Frame& frame, uint32_t planes, int slot) {
    SlotState& st = slots_[slot];
    Timings& last_ = st.last;
    auto t0 = Clock::now();
    setup_frame(frame, st.setup, st.setup_ws);
    last_.setup_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    trace("  setup done", slot);
    t0 = Clock::now();
    bin_triangles(st.setup, frame.tiles_x(), frame.tiles_y(), st.bins);
    last_.bin_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    last_.records = st.bins.entries;
    trace("  bins done", slot);

    DeviceOptions opt;
    opt.slot = slot;
    // Map the requested planes to the device's: colour 1, depth 2, id 4; float colour adds G and B.
    opt.float_planes = (planes & kRenderFloat) != 0;
    opt.plane_mask = ((planes & kRenderColor) ? 1u : 0u) | ((planes & kRenderDepth) ? 2u : 0u) |
                     ((planes & kRenderId) ? 4u : 0u) | (opt.float_planes ? 8u | 16u : 0u);
    t0 = Clock::now();
    const MultiChipFrame f = mc_.render(frame, st.setup, st.bins, opt);
    last_.device_wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    last_.chips = f.times;
    return *f.fb;
}

std::string TensixBackend::report(int slot) const {
    const Timings& last_ = slots_[slot].last;
    double b = 0, u = 0, d = 0, r = 0, rd = 0;
    for (const DeviceTimes& t : last_.chips.chip) {
        b = std::max(b, t.build_ms);
        u = std::max(u, t.upload_ms);
        d = std::max(d, t.device_ms);
        r = std::max(r, t.readback_ms);
        rd = std::max(rd, t.read_ms);
    }
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "setup %.1f bin %.1f | device wall %.1f = slowest chip: encode %.1f upload %.1f device %.1f readback %.1f (transfer %.1f) ms | %llu records",
                  last_.setup_ms, last_.bin_ms, last_.device_wall_ms, b, u, d, r, rd, static_cast<unsigned long long>(last_.records));
    return buf;
}

std::unique_ptr<Backend> make_tensix_backend(size_t n_chips) { return std::make_unique<TensixBackend>(n_chips); }

}  // namespace ttr
