// SPDX-License-Identifier: Apache-2.0

#include "tensix/multichip.h"

#include "ttraster/parallel.h"

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

    // Every tile belongs to exactly one band, so the chips can write the shared framebuffer
    // concurrently. Size it once; later frames overwrite it.
    Framebuffer& fb_ = fb_slots_[opt.slot];
    SlotFill& fill = fill_[opt.slot];
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
        fill.valid = false;   // contents unknown: every empty tile must be written once
    }

    // Tiles without a single record show only the clear values. They are not sent to the chip (no encoding, no
    // draw, no read-back); the host writes the clear values into them, and only when they do not hold them
    // already (`fill.clean`, reset whenever the clear values, the planes or the target size change).
    auto t0 = Clock::now();
    const size_t tiles_x = frame.tiles_x(), tiles_y = frame.tiles_y();
    uint32_t clear_bits[2];
    std::memcpy(&clear_bits[0], &frame.clear_depth, 4);
    clear_bits[1] = pack_rgba8(frame.clear_color);
    uint32_t key[SlotFill::kKeyWords] = {clear_bits[0], clear_bits[1], m, opt.float_planes ? 1u : 0u, pw, ph};
    for (int c = 0; c < 3; c++) {
        std::memcpy(&key[6 + c], &frame.clear_color[c], 4);   // the unquantized clear colour matters for float planes
    }
    if (!fill.valid || std::memcmp(fill.key, key, sizeof key) != 0 || fill.clean.size() != tiles_x * tiles_y) {
        fill.clean.assign(tiles_x * tiles_y, 0);
        std::memcpy(fill.key, key, sizeof key);
        fill.valid = true;
    }
    std::vector<std::pair<int, int>> active;
    std::vector<size_t> to_fill;
    active.reserve(tiles_x * tiles_y);
    for (size_t ty = 0; ty < tiles_y; ty++) {
        for (size_t tx = 0; tx < tiles_x; tx++) {
            const size_t i = ty * tiles_x + tx;
            if (bins.tile(static_cast<uint32_t>(tx), static_cast<uint32_t>(ty)).empty()) {
                if (!fill.clean[i]) to_fill.push_back(i);
                fill.clean[i] = 1;
            } else {
                active.emplace_back(static_cast<int>(tx), static_cast<int>(ty));
                fill.clean[i] = 0;
            }
        }
    }
    parallel_for(to_fill.size(), 0, [&](size_t k) {
        const size_t tx = to_fill[k] % tiles_x, ty = to_fill[k] / tiles_x;
        for (int j = 0; j < kTileSize; j++) {
            const size_t row = fb_.index(static_cast<uint32_t>(tx * kTileSize), static_cast<uint32_t>(ty * kTileSize + j));
            if (want_color) std::fill_n(fb_.color.begin() + static_cast<std::ptrdiff_t>(row), kTileSize, clear_bits[1]);
            if (want_float) for (int c = 0; c < 3; c++) std::fill_n(fb_.plane[c].begin() + static_cast<std::ptrdiff_t>(row), kTileSize, frame.clear_color[c]);
            if (want_depth) std::fill_n(fb_.depth.begin() + static_cast<std::ptrdiff_t>(row), kTileSize, frame.clear_depth);
            if (want_id) {
                std::fill_n(fb_.id.begin() + static_cast<std::ptrdiff_t>(row), kTileSize, 0u);
                std::fill_n(fb_.cover.begin() + static_cast<std::ptrdiff_t>(row), kTileSize, static_cast<uint8_t>(0));
            }
        }
    });
    const auto bands = split_balanced(active, bins, chips());
    out.times.partition_ms = ms_since(t0);
    out.times.chip.resize(chips());
    out.times.chip_tiles.resize(chips());
    out.times.chip_records.resize(chips());

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
