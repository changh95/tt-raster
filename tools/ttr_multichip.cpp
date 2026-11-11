// SPDX-License-Identifier: Apache-2.0
//
// Render test scenes on 1, 2, 4 and 8 chips and compare (docs/PLAN.md, Phase 2.6).
//
//   scripts/run.sh 0,1,2,3,4,5,6,7 build/tools/ttr_multichip [scene|all] [--chips 1,2,4,8] [--bench R]
//                                                            [--size WxH] [--png DIR]
//
// Per scene and chip count: the merged frame must equal the single-chip frame bit for bit, and match
// the CPU reference (colour exact except on pixels the reference flags as depth-ambiguous). Then the
// frame is rendered R more times and the wall time of the whole frame (host partition and encode,
// upload, device, read-back, merge) is reported with the slowest chip's phases.
// Exit code 0 only if every check passes.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tensix/multichip.h"
#include "ttraster/image.h"
#include "ttraster/reference.h"
#include "ttraster/scenes.h"

using namespace ttr;

int main(int argc, char** argv) {
    std::string which = "all", png_dir;
    std::vector<size_t> chip_counts = {1, 2, 4, 8};
    uint32_t w = 1920, h = 1088;
    int bench = 10;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--size" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%ux%u", &w, &h) != 2) return 2;
        } else if (a == "--bench" && i + 1 < argc) {
            bench = std::atoi(argv[++i]);
        } else if (a == "--png" && i + 1 < argc) {
            png_dir = argv[++i];
        } else if (a == "--chips" && i + 1 < argc) {
            chip_counts.clear();
            const std::string list = argv[++i];
            size_t pos = 0;
            while (pos < list.size()) {
                const size_t comma = list.find(',', pos);
                chip_counts.push_back(static_cast<size_t>(std::atoi(list.substr(pos, comma - pos).c_str())));
                pos = comma == std::string::npos ? list.size() : comma + 1;
            }
        } else if (a[0] != '-') {
            which = a;
        } else {
            std::fprintf(stderr, "usage: ttr_multichip [scene|all] [--chips 1,2,4,8] [--bench R] [--size WxH] [--png DIR]\n");
            return 2;
        }
    }
    const size_t max_chips = *std::max_element(chip_counts.begin(), chip_counts.end());
    MultiChipRenderer mc(max_chips);
    std::printf("%zu chip(s) opened\n", mc.opened());

    DeviceOptions opt;
    opt.debug_planes = false;  // colour only: the production read-back

    bool all_ok = true;
    for (const SceneInfo& info : scene_list()) {
        if (which != "all" && which != info.name) continue;
        Frame frame;
        make_scene(info.name, w, h, frame);
        const SetupResult setup = setup_frame(frame);
        const TileBins bins = bin_triangles(setup, frame.tiles_x(), frame.tiles_y());
        const Framebuffer ref = render_reference(frame, setup, bins);

        std::vector<uint32_t> single;  // the 1-chip colour plane
        std::printf("\n%s\n  chips   identical to 1 chip   vs reference (mismatch / ambiguous)   frame ms (median)   slowest chip: build  upload  device  readback   tiles/chip (min..max)  records/chip (min..max)\n", info.name);
        for (size_t n : chip_counts) {
            if (n > mc.opened()) continue;
            mc.set_active(n);
            const MultiChipFrame f = mc.render(frame, setup, bins, opt);
            if (n == chip_counts.front()) single = f.fb->color;
            // Crop to the viewport for the comparisons.
            // Perspective colours divide by a Newton-refined reciprocal on the device: one level of difference from the
            // reference's IEEE division is allowed for such scenes (the chips still have to agree with each other exactly).
            bool perspective = false;
            for (const Triangle& t : frame.triangles) perspective |= t.perspective && !t.flat;
            auto off_by_more = [&](uint32_t a, uint32_t b) {
                if (a == b) return false;
                if (!perspective) return true;
                for (int ch = 0; ch < 4; ch++) {
                    if (std::abs(static_cast<int>((a >> (8 * ch)) & 255) - static_cast<int>((b >> (8 * ch)) & 255)) > 1) return true;
                }
                return false;
            };
            uint64_t diff_single = 0, diff_ref = 0, amb = 0;
            for (uint32_t y = 0; y < h; y++) {
                for (uint32_t x = 0; x < w; x++) {
                    const size_t i = f.fb->index(x, y);
                    diff_single += f.fb->color[i] != single[i];
                    if (ref.ambiguous[i]) { amb++; continue; }
                    diff_ref += off_by_more(f.fb->color[i], ref.color[i]);
                }
            }
            const bool ok = diff_single == 0 && diff_ref == 0;
            all_ok &= ok;

            std::vector<double> walls;
            MultiChipTimes last;
            for (int r = 0; r < bench; r++) {
                MultiChipFrame g = mc.render(frame, setup, bins, opt);
                walls.push_back(g.times.wall_ms);
                last = g.times;
            }
            std::sort(walls.begin(), walls.end());
            double b = 0, u = 0, d = 0, rb = 0;
            uint64_t tmin = ~0ull, tmax = 0, rmin = ~0ull, rmax = 0;
            for (size_t c = 0; c < n; c++) {
                b = std::max(b, last.chip[c].build_ms);
                u = std::max(u, last.chip[c].upload_ms);
                d = std::max(d, last.chip[c].device_ms);
                rb = std::max(rb, last.chip[c].readback_ms);
                tmin = std::min(tmin, last.chip_tiles[c]);
                tmax = std::max(tmax, last.chip_tiles[c]);
                rmin = std::min(rmin, last.chip_records[c]);
                rmax = std::max(rmax, last.chip_records[c]);
            }
            std::printf("  %5zu   %-20s %8llu / %-6llu %*s %14.2f %27.2f %7.2f %7.2f %9.2f %12llu..%-6llu %10llu..%llu\n", n,
                        diff_single == 0 ? "yes" : "NO", static_cast<unsigned long long>(diff_ref),
                        static_cast<unsigned long long>(amb), 18, ok ? "" : "FAILED", walls[walls.size() / 2], b, u, d, rb,
                        static_cast<unsigned long long>(tmin), static_cast<unsigned long long>(tmax),
                        static_cast<unsigned long long>(rmin), static_cast<unsigned long long>(rmax));
            if (!png_dir.empty() && n == max_chips) {
                write_png(png_dir + "/" + info.name + "_" + std::to_string(n) + "chips.png", image_from_framebuffer(*f.fb, w, h));
            }
        }
    }
    std::printf("\n%s\n", all_ok ? "OK" : "FAILED");
    return all_ok ? 0 : 1;
}
