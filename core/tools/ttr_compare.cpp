// SPDX-License-Identifier: Apache-2.0
//
// Side-by-side comparison image: reference | device | difference heatmap, each downscaled.
//   ttr_compare REFERENCE.png DEVICE.png OUT.png [--scale N]
// Prints the pixel mismatch count; exit 0 when the two images are identical.

#include <cstdio>
#include <cstdlib>
#include <string>

#include "ttraster/image.h"

using namespace ttr;

int main(int argc, char** argv) {
    std::string pa, pb, out;
    uint32_t scale = 4;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--scale" && i + 1 < argc) scale = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (pa.empty()) pa = a;
        else if (pb.empty()) pb = a;
        else if (out.empty()) out = a;
    }
    Image a, b;
    if (out.empty() || !read_png(pa, a) || !read_png(pb, b) || a.width != b.width || a.height != b.height) {
        std::fprintf(stderr, "usage: ttr_compare REFERENCE.png DEVICE.png OUT.png [--scale N]\n");
        return 2;
    }
    const DiffStats d = diff_images(a, b);
    const uint8_t gap[4] = {255, 255, 255, 255};
    const Image heat = make_heatmap(a, b);
    // Differences would vanish when averaged down; max-pool the heatmap so a single pixel stays visible.
    Image heat_small;
    heat_small.width = heat.width / scale;
    heat_small.height = heat.height / scale;
    heat_small.rgba.assign(static_cast<size_t>(heat_small.width) * heat_small.height * 4, 255);
    for (uint32_t y = 0; y < heat_small.height; y++)
        for (uint32_t x = 0; x < heat_small.width; x++) {
            uint8_t* o = &heat_small.rgba[(static_cast<size_t>(y) * heat_small.width + x) * 4];
            o[0] = o[1] = o[2] = 0;
            for (uint32_t j = 0; j < scale; j++)
                for (uint32_t i = 0; i < scale; i++) {
                    const uint8_t* p = heat.pixel(x * scale + i, y * scale + j);
                    if (p[0] > o[0]) { o[0] = p[0]; o[1] = p[1]; o[2] = p[2]; }
                }
        }
    const Image m = hstack({downscale(a, scale), downscale(b, scale), heat_small}, 6, gap);
    if (!write_png(out, m)) return 2;
    std::printf("%s: %llu of %llu pixels differ, max error %u\n", out.c_str(), static_cast<unsigned long long>(d.mismatched),
                static_cast<unsigned long long>(d.pixels), d.max_abs_error);
    return d.mismatched == 0 ? 0 : 1;
}
