// SPDX-License-Identifier: Apache-2.0
//
// Image tools: PNG round trip, diff statistics, heatmap, crop, hashing.

#include <cmath>
#include <cstdio>
#include <string>

#include "check.h"
#include "ttraster/image.h"

using namespace ttr;

namespace {

Image noise(uint32_t w, uint32_t h, uint32_t seed) {
    Image img;
    img.width = w;
    img.height = h;
    img.rgba.resize(static_cast<size_t>(w) * h * 4);
    uint32_t s = seed;
    for (uint8_t& b : img.rgba) {
        s = s * 1664525u + 1013904223u;
        b = static_cast<uint8_t>(s >> 24);
    }
    return img;
}

}  // namespace

int main() {
    // ctest runs each test in its own build directory, so a relative path is safe.
    const std::string path = "test_image_tmp.png";

    // PNG round trip, awkward sizes included (1-pixel rows and columns, widths not a multiple of 4).
    for (auto [w, h] : {std::pair<uint32_t, uint32_t>{1, 1}, {7, 5}, {64, 33}, {1, 40}, {97, 1}}) {
        const Image a = noise(w, h, w * 31 + h);
        CHECK(write_png(path, a));
        Image b;
        CHECK(read_png(path, b));
        CHECK(b.width == w && b.height == h && b.rgba == a.rgba);
    }
    {
        Image b;
        CHECK(!read_png("does_not_exist.png", b));
    }

    // Diff statistics.
    {
        Image a = noise(16, 16, 1), b = a;
        DiffStats d = diff_images(a, b);
        CHECK(d.mismatched == 0 && d.max_abs_error == 0 && std::isinf(d.psnr_db) && d.first_x == -1);
        b.rgba[(3 * 16 + 5) * 4 + 2] = static_cast<uint8_t>(a.rgba[(3 * 16 + 5) * 4 + 2] ^ 1);  // 1 LSB, one pixel
        d = diff_images(a, b);
        CHECK(d.mismatched == 1 && d.max_abs_error == 1 && d.first_x == 5 && d.first_y == 3);
        CHECK(diff_images(a, b, 1).mismatched == 0);  // within tolerance 1
        const double mse = 1.0 / (256.0 * 4.0);
        CHECK(std::fabs(d.psnr_db - 10.0 * std::log10(255.0 * 255.0 / mse)) < 1e-9);
        b.rgba[0] = static_cast<uint8_t>(a.rgba[0] + 40);
        d = diff_images(a, b);
        CHECK(d.mismatched == 2 && d.max_abs_error == 40);
        const Image heat = make_heatmap(a, b);
        CHECK(heat.width == 16 && heat.rgba[0] == 255 && heat.rgba[3] == 255);                // large error: bright
        CHECK(heat.rgba[(3 * 16 + 5) * 4] == 120 && heat.rgba[(3 * 16 + 5) * 4 + 1] == 0);   // 1 LSB: dark red
        CHECK(heat.rgba[4 * 4] == 0 && heat.rgba[4 * 4 + 3] == 255);                          // identical: black
    }

    // Crop and PPM.
    {
        const Image a = noise(40, 30, 9);
        const Image c = crop_image(a, 8, 4, 32, 20);
        CHECK(c.width == 32 && c.height == 20);
        CHECK(std::equal(c.pixel(0, 0), c.pixel(0, 0) + 4, a.pixel(8, 4)));
        CHECK(std::equal(c.pixel(31, 19), c.pixel(31, 19) + 4, a.pixel(39, 23)));
        CHECK(write_ppm("test_image_tmp.ppm", a));
    }

    // FNV-1a 64 known answers.
    CHECK(fnv1a64("", 0) == 0xcbf29ce484222325ull);
    CHECK(fnv1a64("a", 1) == 0xaf63dc4c8601ec8cull);
    CHECK(fnv1a64("foobar", 6) == 0x85944171f73967e8ull);

    std::remove(path.c_str());
    std::remove("test_image_tmp.ppm");
    return ttr_test::finish();
}
