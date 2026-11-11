// SPDX-License-Identifier: Apache-2.0
//
// Render test scenes on Tensix and compare with the CPU reference (docs/PLAN.md, Phase 2).
//
//   scripts/run.sh <nodes> build/tools/ttr_device [scene|all] [--size WxH] [--mode crops|grid]
//                  [--tile tx,ty] [--cores N] [--png DIR] [--bench R] [--define NAME=VALUE]...
//                  [--colour-only]
//
//   crops   one core draws a set of 32x32 tiles of the scene in turn (Phase 2.1 and 2.2)
//   grid    every tile of the target, split over N worker cores (Phase 2.3, default all cores)
//   --bench R   after the checks, run the frame R more times and report device time statistics
//   --define    extra kernel compile-time define, for experiments (affects the checks too)
//   --colour-only  read back only the colour plane (production read-back; checks image only)
//   --golden    (grid mode, 1920x1088) also compare the colour plane with tests/golden/<scene>_1920x1088.png
//   --strict    also compare pixels the reference flags as depth-ambiguous (exact ties are expected to agree)
//
// Checks, per scene, on the drawn tiles:
//   coverage  a debug run counts covering triangles per pixel; must equal the reference exactly
//   image     visible triangle id, depth and colour of the normal run: id exact, colour within
//             1 LSB, depth within 1e-6 relative, skipping pixels the reference flags as depth-ambiguous
// Exit code 0 only if everything matches.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "tensix/tensix_renderer.h"
#include "ttraster/image.h"
#include "ttraster/oracle.h"
#include "ttraster/reference.h"
#include "ttraster/scenes.h"

using namespace ttr;
using Clock = std::chrono::steady_clock;

namespace {

struct Compare {
    uint64_t pixels = 0, cover_bad = 0, id_bad = 0, depth_bad = 0, color_exact = 0, color_1lsb = 0, color_bad = 0, skipped = 0;
    uint32_t max_color_err = 0;
    double max_depth_err = 0;
};

Compare compare(
    const Frame& frame, const Framebuffer& ref, const DeviceFrame& dev, const DeviceFrame& dev_cover, bool strict) {
    Compare c;
    const uint32_t w = frame.width, h = frame.height;
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            if (!dev.tile_rendered[(y / 32) * frame.tiles_x() + x / 32]) {
                continue;
            }
            const size_t i = ref.index(x, y);
            c.pixels++;
            c.cover_bad += dev_cover.fb.cover[i] != ref.cover[i];
            if (ref.ambiguous[i] && !strict) {
                c.skipped++;
                continue;
            }
            c.id_bad += dev.fb.id[i] != ref.id[i];
            const double dz = std::fabs(static_cast<double>(dev.fb.depth[i]) - ref.depth[i]);
            c.max_depth_err = std::max(c.max_depth_err, dz);
            c.depth_bad += dz > 1e-6 * std::max(1.0f, std::fabs(ref.depth[i]));
            uint32_t worst = 0;
            for (int ch = 0; ch < 4; ch++) {
                const int a = (dev.fb.color[i] >> (8 * ch)) & 255, b = (ref.color[i] >> (8 * ch)) & 255;
                worst = std::max<uint32_t>(worst, static_cast<uint32_t>(std::abs(a - b)));
            }
            c.max_color_err = std::max(c.max_color_err, worst);
            c.color_exact += worst == 0;
            c.color_1lsb += worst == 1;
            c.color_bad += worst > 1;
        }
    }
    return c;
}

Compare compare_colour(const Frame& frame, const Framebuffer& ref, const DeviceFrame& dev) {
    Compare c;
    for (uint32_t y = 0; y < frame.height; y++) {
        for (uint32_t x = 0; x < frame.width; x++) {
            if (!dev.tile_rendered[(y / 32) * frame.tiles_x() + x / 32]) continue;
            const size_t i = ref.index(x, y);
            c.pixels++;
            if (ref.ambiguous[i]) { c.skipped++; continue; }
            uint32_t worst = 0;
            for (int ch = 0; ch < 4; ch++) {
                const int a = (dev.fb.color[i] >> (8 * ch)) & 255, b = (ref.color[i] >> (8 * ch)) & 255;
                worst = std::max<uint32_t>(worst, static_cast<uint32_t>(std::abs(a - b)));
            }
            c.max_color_err = std::max(c.max_color_err, worst);
            c.color_exact += worst == 0;
            c.color_1lsb += worst == 1;
            c.color_bad += worst > 1;
        }
    }
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    std::string which = "all", mode = "crops", png_dir;
    uint32_t w = 1920, h = 1088, n_cores = 0;
    int tile_x = -1, tile_y = -1, bench = 0;
    bool colour_only = false, strict = false, golden = false;
    std::map<std::string, std::string> defines;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--size" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%ux%u", &w, &h) != 2) return 2;
        } else if (a == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (a == "--tile" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%d,%d", &tile_x, &tile_y) != 2) return 2;
        } else if (a == "--cores" && i + 1 < argc) {
            n_cores = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (a == "--png" && i + 1 < argc) {
            png_dir = argv[++i];
        } else if (a == "--bench" && i + 1 < argc) {
            bench = std::atoi(argv[++i]);
        } else if (a == "--golden") {
            golden = true;
        } else if (a == "--strict") {
            strict = true;
        } else if (a == "--colour-only") {
            colour_only = true;
        } else if (a == "--define" && i + 1 < argc) {
            const std::string kv = argv[++i];
            const size_t eq = kv.find('=');
            defines[kv.substr(0, eq)] = eq == std::string::npos ? "1" : kv.substr(eq + 1);
        } else if (a[0] != '-') {
            which = a;
        } else {
            std::fprintf(stderr, "usage: ttr_device [scene|all] [--size WxH] [--mode crops|grid] [--tile tx,ty] [--cores N] [--png DIR]\n");
            return 2;
        }
    }
    std::vector<std::string> names;
    for (const SceneInfo& s : scene_list()) {
        if (which == "all" || which == s.name) names.push_back(s.name);
    }
    if (names.empty()) {
        std::fprintf(stderr, "unknown scene '%s'\n", which.c_str());
        return 2;
    }

    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    TensixRenderer renderer(mesh);
    std::vector<tt::tt_metal::CoreCoord> cores = renderer.worker_cores();
    if (n_cores > 0 && n_cores < cores.size()) cores.resize(n_cores);

    bool all_ok = true;
    for (const std::string& name : names) {
        Frame frame;
        make_scene(name, w, h, frame);
        const SetupResult setup = setup_frame(frame);
        const TileBins bins = bin_triangles(setup, frame.tiles_x(), frame.tiles_y());
        const Framebuffer ref = render_reference(frame, setup, bins);

        std::vector<CorePlan> plan;
        if (mode == "grid") {
            plan = plan_row_major(cores, frame.tiles_x(), frame.tiles_y());
        } else {
            CorePlan cp;
            cp.core = cores[0];
            if (tile_x >= 0) {
                cp.tiles.emplace_back(tile_x, tile_y);
            } else {
                // A deterministic sample: the busiest tiles, a few evenly spaced non-empty ones, an empty one.
                std::vector<std::pair<size_t, int>> by_load;
                for (uint32_t ty = 0; ty < frame.tiles_y(); ty++)
                    for (uint32_t tx = 0; tx < frame.tiles_x(); tx++)
                        by_load.emplace_back(bins.tile(tx, ty).size(), static_cast<int>(ty * frame.tiles_x() + tx));
                std::vector<int> chosen;
                std::vector<std::pair<size_t, int>> sorted = by_load;
                std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
                for (size_t k = 0; k < 4 && k < sorted.size() && sorted[k].first > 0; k++) chosen.push_back(sorted[k].second);
                std::vector<int> nonempty;
                for (auto& [n, t] : by_load) if (n > 0) nonempty.push_back(t);
                for (size_t k = 0; k < 8 && !nonempty.empty(); k++) chosen.push_back(nonempty[nonempty.size() * k / 8]);
                for (auto& [n, t] : by_load) if (n == 0) { chosen.push_back(t); break; }
                std::sort(chosen.begin(), chosen.end());
                chosen.erase(std::unique(chosen.begin(), chosen.end()), chosen.end());
                for (int t : chosen) cp.tiles.emplace_back(t % frame.tiles_x(), t / frame.tiles_x());
            }
            plan.push_back(cp);
        }

        DeviceOptions cover_opt;
        cover_opt.cover_count = true;
        cover_opt.defines = defines;
        DeviceOptions opt;
        opt.defines = defines;
        opt.debug_planes = !colour_only;
        // Coverage check needs the id plane; in colour-only mode it is skipped.
        const DeviceFrame dev_cover = colour_only ? DeviceFrame{} : renderer.render(frame, setup, bins, plan, cover_opt);
        const DeviceFrame dev = renderer.render(frame, setup, bins, plan, opt);
        const Compare c = colour_only ? compare_colour(frame, ref, dev) : compare(frame, ref, dev, dev_cover, strict);
        const bool ok = c.cover_bad == 0 && c.id_bad == 0 && c.depth_bad == 0 && c.color_bad == 0 && c.pixels > 0;
        all_ok &= ok;
        std::printf(
            "%-17s %s  tiles %zu on %zu core(s)  pixels %llu  coverage mismatches %llu  id %llu  depth %llu (max %.2e)  "
            "colour exact %llu / 1 LSB %llu / worse %llu  skipped(ambiguous) %llu\n",
            name.c_str(), ok ? "OK    " : "FAILED", dev.times.blocks, plan.size(), static_cast<unsigned long long>(c.pixels),
            static_cast<unsigned long long>(c.cover_bad), static_cast<unsigned long long>(c.id_bad),
            static_cast<unsigned long long>(c.depth_bad), c.max_depth_err, static_cast<unsigned long long>(c.color_exact),
            static_cast<unsigned long long>(c.color_1lsb), static_cast<unsigned long long>(c.color_bad),
            static_cast<unsigned long long>(c.skipped));
        std::printf(
            "    records %llu (vectors: %.1f partial + %.1f full per record)  build %.1f ms  upload %.1f ms  device %.2f ms  readback %.1f ms\n",
            static_cast<unsigned long long>(dev.times.records),
            dev.times.records ? double(dev.times.vec_partial) / double(dev.times.records) : 0.0,
            dev.times.records ? double(dev.times.vec_full) / double(dev.times.records) : 0.0, dev.times.build_ms, dev.times.upload_ms, dev.times.device_ms,
            dev.times.readback_ms);
        if (golden) {
            Image g;
            const std::string path = std::string(TT_RASTER_GOLDEN_DIR) + name + "_" + std::to_string(w) + "x" + std::to_string(h) + ".png";
            if (!read_png(path, g) || g.width != w || g.height != h) {
                std::printf("    golden: cannot read %s\n", path.c_str());
                all_ok = false;
            } else {
                const Image d = image_from_framebuffer(dev.fb, w, h);
                // Compare everywhere except pixels the reference flags as depth-ambiguous (the golden is the reference's own output).
                // Perspective colours (a Newton-refined reciprocal instead of IEEE division) and blend chains (the SFPU's
                // partially fused multiply-add) can differ from the reference in the last float bit: one colour level is then
                // allowed (and counted).
                bool perspective = false;
                for (const Triangle& t : frame.triangles) perspective |= (t.perspective && !t.flat) || t.blend != BlendMode::None;
                uint64_t bad = 0, off_by_one = 0;
                for (uint32_t y = 0; y < h; y++)
                    for (uint32_t x = 0; x < w; x++) {
                        if (ref.ambiguous[ref.index(x, y)]) continue;
                        const uint8_t* a = d.pixel(x, y);
                        const uint8_t* b = g.pixel(x, y);
                        if (std::memcmp(a, b, 4) == 0) continue;
                        int worst = 0;
                        for (int ch = 0; ch < 4; ch++) worst = std::max(worst, std::abs(static_cast<int>(a[ch]) - static_cast<int>(b[ch])));
                        if (perspective && worst <= 1) off_by_one++; else bad++;
                    }
                std::printf("    golden image: %llu mismatching pixels", static_cast<unsigned long long>(bad));
                if (perspective) std::printf(" (+ %llu within one level, allowed for perspective colours and blending)", static_cast<unsigned long long>(off_by_one));
                std::printf("\n");
                all_ok &= bad == 0;
            }
        }
        if (bench > 0) {
            std::vector<double> t_dev, t_all;
            DeviceTimes last;
            size_t max_rec = 0;
            for (const CorePlan& cp : plan) {
                size_t r = 0;
                for (auto& [tx, ty] : cp.tiles) r += bins.tile(tx, ty).size();
                max_rec = std::max(max_rec, r);
            }
            for (int r = 0; r < bench; r++) {
                const auto t0 = Clock::now();
                const DeviceFrame d = renderer.render(frame, setup, bins, plan, opt);
                t_all.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                t_dev.push_back(d.times.device_ms);
                last = d.times;
            }
            std::sort(t_dev.begin(), t_dev.end());
            std::sort(t_all.begin(), t_all.end());
            std::printf(
                "    bench x%d: device min %.3f / median %.3f ms, whole render() median %.2f ms, busiest core %zu records "
                "(%.2f us per record at min)\n",
                bench, t_dev.front(), t_dev[t_dev.size() / 2], t_all[t_all.size() / 2], max_rec,
                max_rec ? t_dev.front() * 1000.0 / static_cast<double>(max_rec) : 0.0);
            std::printf(
                "    warm frame: build %.2f ms  upload %.2f ms  device %.2f ms  readback %.2f ms\n", last.build_ms, last.upload_ms,
                last.device_ms, last.readback_ms);
        }
        if (!png_dir.empty()) {
            write_png(png_dir + "/" + name + "_device.png", image_from_framebuffer(dev.fb, w, h));
            write_png(png_dir + "/" + name + "_reference.png", image_from_framebuffer(ref, w, h));
        }
    }
    mesh->close();
    std::printf("%s\n", all_ok ? "OK" : "FAILED");
    return all_ok ? 0 : 1;
}
