// SPDX-License-Identifier: Apache-2.0
//
// Every scene at a small, non-tile-aligned size (250 x 150, padded to 256 x 160): the reference
// against the int64 oracle, tile classification against full evaluation, determinism, and the
// structural properties of the mesh scenes.

#include <cstdio>
#include <string>

#include "check.h"
#include "ttraster/image.h"
#include "ttraster/oracle.h"
#include "ttraster/reference.h"
#include "ttraster/scenes.h"

using namespace ttr;

int main() {
    const uint32_t w = 250, h = 150;
    for (const SceneInfo& info : scene_list()) {
        const std::string name = info.name;
        Frame f, f2;
        CHECK(make_scene(name, w, h, f));
        CHECK(make_scene(name, w, h, f2));
        CHECK(f.triangles.size() == f2.triangles.size());

        const SetupResult setup = setup_frame(f);
        const TileBins bins = bin_triangles(setup, f.tiles_x(), f.tiles_y());
        ReferenceStats rs;
        const Framebuffer fb = render_reference(f, setup, bins, {}, &rs);
        CHECK(fb.width == 256 && fb.height == 160);

        // Coverage must equal the exact integer coverage of the same snapped (and clipped) triangles.
        const std::vector<uint8_t> oracle = oracle_cover_from_setup(setup, fb.width, fb.height);
        CHECK_MSG(oracle == fb.cover, "%s: coverage differs from the int64 oracle", name.c_str());

        // Skipping the edge tests for Full tiles must not change anything.
        ReferenceOptions slow;
        slow.trust_full_tiles = false;
        const Framebuffer fb2 = render_reference(f, setup, bins, slow);
        CHECK_MSG(fb2.color == fb.color && fb2.depth == fb.depth && fb2.cover == fb.cover && fb2.id == fb.id,
                  "%s: Full-tile shortcut changes the result", name.c_str());

        // Against unclipped input coverage, everywhere clipping cannot have touched.
        const InputOracle in = oracle_cover_from_input(f);
        uint64_t bad = 0;
        for (size_t i = 0; i < in.cover.size(); i++) {
            bad += !in.clip_mask[i] && in.cover[i] != fb.cover[i];
        }
        CHECK_MSG(bad == 0, "%s: %llu pixels differ from the unclipped oracle", name.c_str(), static_cast<unsigned long long>(bad));

        // Deterministic: a second render is bit-identical.
        const Framebuffer fb3 = render_reference(f2);
        CHECK(hash_framebuffer(fb, w, h) == hash_framebuffer(fb3, w, h));

        // id buffer implies coverage; written fragments imply a visible id.
        uint64_t visible = 0;
        for (size_t i = 0; i < fb.id.size(); i++) {
            CHECK(fb.id[i] == 0 || fb.cover[i] >= 1);
            visible += fb.id[i] != 0;
        }
        CHECK(visible > 0 || name == "tiny_10k");

        uint64_t max_cover = 0, covered_in_view = 0;
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                const uint8_t c = fb.cover[fb.index(x, y)];
                max_cover = c > max_cover ? c : max_cover;
                covered_in_view += c == 1;
            }
        }
        if (info.watertight_mesh) {
            CHECK_MSG(covered_in_view == static_cast<uint64_t>(w) * h && max_cover == 1,
                      "%s: mesh not watertight (%llu of %u pixels hit exactly once, max %llu)", name.c_str(),
                      static_cast<unsigned long long>(covered_in_view), w * h, static_cast<unsigned long long>(max_cover));
        }
        if (name == "fan") {
            CHECK_MSG(max_cover == 1, "fan: a pixel was covered %llu times", static_cast<unsigned long long>(max_cover));
        }
        if (name == "fullscreen") {
            CHECK(covered_in_view == static_cast<uint64_t>(w) * h);
        }
        std::printf("%-17s tris %5zu  pieces %5llu  pairs %7llu  full %6llu  max cover %3llu  visible %llu\n", name.c_str(),
                    f.triangles.size(), static_cast<unsigned long long>(setup.stats.pieces),
                    static_cast<unsigned long long>(bins.entries), static_cast<unsigned long long>(bins.full_entries),
                    static_cast<unsigned long long>(max_cover), static_cast<unsigned long long>(visible));
    }
    return ttr_test::finish();
}
