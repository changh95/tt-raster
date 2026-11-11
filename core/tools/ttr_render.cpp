// SPDX-License-Identifier: Apache-2.0
//
// Render test scenes with the CPU reference and write images and a golden manifest.
//   ttr_render [scene|all] [--size WxH] [--out DIR] [--ppm] [--manifest FILE]
// Without --out nothing is written; the per-scene summary is always printed.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ttraster/image.h"
#include "ttraster/oracle.h"
#include "ttraster/reference.h"
#include "ttraster/scenes.h"

using namespace ttr;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    std::string which = "all", out_dir, manifest;
    uint32_t w = 1920, h = 1088;
    bool ppm = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--size" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%ux%u", &w, &h) != 2) {
                std::fprintf(stderr, "bad --size\n");
                return 2;
            }
        } else if (a == "--out" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (a == "--manifest" && i + 1 < argc) {
            manifest = argv[++i];
        } else if (a == "--ppm") {
            ppm = true;
        } else if (a[0] != '-') {
            which = a;
        } else {
            std::fprintf(stderr, "usage: ttr_render [scene|all] [--size WxH] [--out DIR] [--ppm] [--manifest FILE]\n");
            return 2;
        }
    }

    std::vector<std::string> names;
    for (const SceneInfo& s : scene_list()) {
        if (which == "all" || which == s.name) {
            names.push_back(s.name);
        }
    }
    if (names.empty()) {
        std::fprintf(stderr, "unknown scene '%s'\n", which.c_str());
        return 2;
    }

    std::string manifest_text;
    int rc = 0;
    for (const std::string& name : names) {
        Frame frame;
        make_scene(name, w, h, frame);
        const auto t0 = Clock::now();
        const SetupResult setup = setup_frame(frame);
        const auto t1 = Clock::now();
        const TileBins bins = bin_triangles(setup, frame.tiles_x(), frame.tiles_y());
        const auto t2 = Clock::now();
        ReferenceStats rs;
        const Framebuffer fb = render_reference(frame, setup, bins, {}, &rs);
        const auto t3 = Clock::now();
        const std::vector<uint8_t> oracle = oracle_cover_from_setup(setup, fb.width, fb.height);
        const bool oracle_ok = oracle == fb.cover;
        rc |= oracle_ok ? 0 : 1;

        const FrameHashes hh = hash_framebuffer(fb, w, h);
        auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::printf(
            "%-17s %ux%u  tris %zu  pieces %llu  tile pairs %llu (full %llu)  setup %.1f ms  bin %.1f ms  render %.1f ms  oracle %s\n",
            name.c_str(), w, h, frame.triangles.size(), static_cast<unsigned long long>(setup.stats.pieces),
            static_cast<unsigned long long>(bins.entries), static_cast<unsigned long long>(bins.full_entries), ms(t0, t1),
            ms(t1, t2), ms(t2, t3), oracle_ok ? "match" : "MISMATCH");

        char line[256];
        std::snprintf(
            line, sizeof line, "%s %u %u %016llx %016llx %016llx %016llx\n", name.c_str(), w, h,
            static_cast<unsigned long long>(hh.color), static_cast<unsigned long long>(hh.depth),
            static_cast<unsigned long long>(hh.id), static_cast<unsigned long long>(hh.cover));
        manifest_text += line;

        if (!out_dir.empty()) {
            const Image img = image_from_framebuffer(fb, w, h);
            const std::string base = out_dir + "/" + name + "_" + std::to_string(w) + "x" + std::to_string(h);
            if (!(ppm ? write_ppm(base + ".ppm", img) : write_png(base + ".png", img))) {
                std::fprintf(stderr, "cannot write %s\n", base.c_str());
                return 1;
            }
        }
    }
    if (!manifest.empty()) {
        FILE* f = std::fopen(manifest.c_str(), "w");
        if (f == nullptr) {
            std::fprintf(stderr, "cannot write %s\n", manifest.c_str());
            return 1;
        }
        std::fputs("# scene width height color_hash depth_hash id_hash cover_hash (FNV-1a 64, viewport region)\n", f);
        std::fputs(manifest_text.c_str(), f);
        std::fclose(f);
    }
    return rc;
}
