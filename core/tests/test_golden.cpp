// SPDX-License-Identifier: Apache-2.0
//
// Phase 1 definition of done: every scene at 1920 x 1088 passes the watertightness check against
// the int64 oracle, and matches the committed golden image and manifest.
//
//   test_golden            check against core/tests/golden
//   test_golden --update   regenerate core/tests/golden (PNG per scene plus manifest.txt)

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "check.h"
#include "ttraster/image.h"
#include "ttraster/oracle.h"
#include "ttraster/reference.h"
#include "ttraster/scenes.h"

using namespace ttr;

int main(int argc, char** argv) {
    const bool update = argc > 1 && std::strcmp(argv[1], "--update") == 0;
    const uint32_t w = 1920, h = 1088;
    const std::string dir = TT_RASTER_GOLDEN_DIR;
    const std::string manifest_path = dir + "manifest.txt";

    std::map<std::string, FrameHashes> golden;
    {
        std::ifstream in(manifest_path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream ls(line);
            std::string name;
            unsigned gw = 0, gh = 0;
            FrameHashes hh;
            ls >> name >> gw >> gh >> std::hex >> hh.color >> hh.depth >> hh.id >> hh.cover;
            if (ls && gw == w && gh == h) {
                golden[name] = hh;
            }
        }
    }

    std::string manifest_out = "# scene width height color_hash depth_hash id_hash cover_hash (FNV-1a 64, viewport region)\n";
    for (const SceneInfo& info : scene_list()) {
        const std::string name = info.name;
        Frame f;
        CHECK(make_scene(name, w, h, f));
        const SetupResult setup = setup_frame(f);
        const TileBins bins = bin_triangles(setup, f.tiles_x(), f.tiles_y());
        const Framebuffer fb = render_reference(f, setup, bins);

        // Coverage against the exact oracle, over the whole frame.
        const std::vector<uint8_t> oracle = oracle_cover_from_setup(setup, fb.width, fb.height);
        uint64_t mismatches = 0, once = 0, more = 0;
        for (size_t i = 0; i < oracle.size(); i++) {
            mismatches += oracle[i] != fb.cover[i];
        }
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                const uint8_t c = fb.cover[fb.index(x, y)];
                once += c == 1;
                more += c > 1;
            }
        }
        CHECK_MSG(mismatches == 0, "%s: %llu pixels differ from the int64 oracle", name.c_str(),
                  static_cast<unsigned long long>(mismatches));
        if (info.watertight_mesh) {
            CHECK_MSG(once == static_cast<uint64_t>(w) * h && more == 0,
                      "%s: %llu pixels hit once, %llu hit more than once (want %u, 0)", name.c_str(),
                      static_cast<unsigned long long>(once), static_cast<unsigned long long>(more), w * h);
        }
        if (name == "fan") {
            CHECK_MSG(more == 0, "fan: %llu pixels hit more than once", static_cast<unsigned long long>(more));
        }

        // Golden image and hashes.
        const Image img = image_from_framebuffer(fb, w, h);
        const FrameHashes hh = hash_framebuffer(fb, w, h);
        char line[256];
        std::snprintf(line, sizeof line, "%s %u %u %016llx %016llx %016llx %016llx\n", name.c_str(), w, h,
                      static_cast<unsigned long long>(hh.color), static_cast<unsigned long long>(hh.depth),
                      static_cast<unsigned long long>(hh.id), static_cast<unsigned long long>(hh.cover));
        manifest_out += line;

        const std::string png = dir + name + "_" + std::to_string(w) + "x" + std::to_string(h) + ".png";
        if (update) {
            CHECK_MSG(write_png(png, img), "cannot write %s", png.c_str());
        } else {
            Image ref;
            CHECK_MSG(read_png(png, ref), "%s: missing golden image %s", name.c_str(), png.c_str());
            if (ref.width == w && ref.height == h) {
                const DiffStats d = diff_images(img, ref);
                CHECK_MSG(d.mismatched == 0, "%s: %llu pixels differ from the golden image", name.c_str(),
                          static_cast<unsigned long long>(d.mismatched));
            }
            auto it = golden.find(name);
            CHECK_MSG(it != golden.end(), "%s: no manifest entry", name.c_str());
            if (it != golden.end()) {
                CHECK_MSG(it->second == hh, "%s: depth, id or coverage hash differs from the manifest", name.c_str());
            }
        }
        std::printf("%-17s oracle ok=%d  hit once %llu  more %llu\n", name.c_str(), mismatches == 0,
                    static_cast<unsigned long long>(once), static_cast<unsigned long long>(more));
    }
    if (update) {
        std::ofstream out(manifest_path);
        out << manifest_out;
        std::printf("golden files written to %s\n", dir.c_str());
    }
    return ttr_test::finish();
}
