// SPDX-License-Identifier: Apache-2.0

#include "tensix/multichip.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <thread>

using namespace tt::tt_metal;
using Clock = std::chrono::steady_clock;

namespace ttr {

namespace {
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
}  // namespace

MultiChipRenderer::MultiChipRenderer(size_t n_chips) {
    // Devices must be opened together: opening unit meshes one after the other in a process leaves
    // the first chip's first program hanging (see docs/FINDINGS.md, "Multi-chip").
    std::vector<int> ids;
    for (size_t i = 0; i < n_chips; i++) {
        ids.push_back(static_cast<int>(i));
    }
    auto opened = distributed::MeshDevice::create_unit_meshes(ids);
    for (int id : ids) {
        meshes_.push_back(opened.at(id));
        renderers_.push_back(std::make_unique<TensixRenderer>(meshes_.back()));
    }
    active_ = renderers_.size();
}

MultiChipRenderer::~MultiChipRenderer() {
    for (auto& m : meshes_) {
        m->close();
    }
}

MultiChipFrame MultiChipRenderer::render(
    const Frame& frame, const SetupResult& setup, const TileBins& bins, const DeviceOptions& opt) {
    MultiChipFrame out;
    const auto t_start = Clock::now();

    auto t0 = Clock::now();
    const auto bands = split_balanced(all_tiles(frame.tiles_x(), frame.tiles_y()), bins, chips());
    out.times.partition_ms = ms_since(t0);
    out.times.chip.resize(chips());
    out.times.chip_tiles.resize(chips());
    out.times.chip_records.resize(chips());

    // Every tile belongs to exactly one band, so the chips can write the shared framebuffer
    // concurrently. Size it once; later frames overwrite it.
    Framebuffer& fb_ = fb_slots_[opt.slot];
    const uint32_t pw = frame.padded_width(), ph = frame.padded_height();
    const size_t n_px = static_cast<size_t>(pw) * ph;
    const uint32_t m = effective_plane_mask(opt);
    const bool want_color = (m & 1) && !opt.float_planes, want_depth = m & 2, want_id = m & 4, want_float = opt.float_planes;
    if (fb_.width != pw || fb_.height != ph || (want_color && fb_.color.size() != n_px) || (want_depth && fb_.depth.size() != n_px) ||
        (want_id && fb_.id.size() != n_px) || (want_float && fb_.plane[0].size() != n_px)) {
        fb_.width = pw;
        fb_.height = ph;
        if (want_color) fb_.color.assign(n_px, 0);
        if (want_float) for (int c = 0; c < 3; c++) fb_.plane[c].assign(n_px, 0.0f);
        if (want_depth) fb_.depth.assign(n_px, 0.0f);
        if (want_id) {
            fb_.id.assign(n_px, 0);
            fb_.cover.assign(n_px, 0);
            fb_.ambiguous.assign(n_px, 0);
        }
    }

    DeviceOptions chip_opt = opt;
    if (chip_opt.host_threads == 0) {
        chip_opt.host_threads = std::max<size_t>(1, std::thread::hardware_concurrency() / chips());
    }
    std::vector<DeviceFrame> parts(chips());
    std::vector<std::exception_ptr> errors(chips());
    std::vector<std::thread> threads;
    for (size_t c = 0; c < chips(); c++) {
        threads.emplace_back([&, c] {
            try {
                const auto plan = plan_balanced(renderers_[c]->worker_cores(), bands[c], bins);
                parts[c] = renderers_[c]->render(frame, setup, bins, plan, chip_opt, &fb_);
            } catch (...) {
                errors[c] = std::current_exception();
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    for (auto& e : errors) {
        if (e) {
            std::rethrow_exception(e);
        }
    }
    for (size_t c = 0; c < chips(); c++) {
        out.times.chip[c] = parts[c].times;
        out.times.chip_tiles[c] = bands[c].size();
        for (const auto& [tx, ty] : bands[c]) {
            out.times.chip_records[c] += bins.tile(static_cast<uint32_t>(tx), static_cast<uint32_t>(ty)).size();
        }
    }
    out.fb = &fb_;
    out.times.wall_ms = ms_since(t_start);
    return out;
}

}  // namespace ttr
