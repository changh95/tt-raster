// SPDX-License-Identifier: Apache-2.0
//
// Image tools (docs/PLAN.md, Phase 1.5): PNG and PPM writers, a PNG reader for the files this
// library writes, a diff with mismatch count, maximum error and PSNR, and a heatmap.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ttraster/reference.h"

namespace ttr {

struct Image {
    uint32_t width = 0, height = 0;
    std::vector<uint8_t> rgba;  // width * height * 4, row-major, top row first

    const uint8_t* pixel(uint32_t x, uint32_t y) const { return &rgba[(static_cast<size_t>(y) * width + x) * 4]; }
};

// Crop the padded framebuffer to the viewport and unpack the colour to RGBA8.
Image image_from_framebuffer(const Framebuffer& fb, uint32_t width, uint32_t height);
// Crop an RGBA image to the rectangle [x, x+w) x [y, y+h) (e.g. one 32x32 tile).
Image crop_image(const Image& img, uint32_t x, uint32_t y, uint32_t w, uint32_t h);

// 8-bit RGBA PNG, Sub-filtered and deflated with zlib. The reader accepts 8-bit RGB and RGBA,
// non-interlaced, all five filter types.
bool write_png(const std::string& path, const Image& img);
bool read_png(const std::string& path, Image& img);
bool write_ppm(const std::string& path, const Image& img);  // P6, alpha dropped

struct DiffStats {
    uint64_t pixels = 0;
    uint64_t mismatched = 0;     // pixels with any channel differing by more than `tolerance`
    uint32_t max_abs_error = 0;  // over all channels
    double psnr_db = 0.0;        // over RGBA; infinity when identical
    int64_t first_x = -1, first_y = -1;  // first mismatched pixel, -1 if none
};

// Both images must have the same size (checked by the caller).
DiffStats diff_images(const Image& a, const Image& b, int tolerance = 0);
// Black where identical, dark red to yellow by increasing maximum channel error.
Image make_heatmap(const Image& a, const Image& b);

// Box-filter downscale by an integer factor (edge pixels that do not fill a block are dropped).
Image downscale(const Image& img, uint32_t factor);
// Place images side by side (same height), separated by `gap` pixels of `gap_rgba`.
Image hstack(const std::vector<Image>& panels, uint32_t gap, const uint8_t gap_rgba[4]);

// FNV-1a 64-bit, used for golden manifests of buffers that have no image form.
uint64_t fnv1a64(const void* data, size_t bytes, uint64_t seed = 0xcbf29ce484222325ull);

// Hashes of the viewport region of every framebuffer plane, row by row (padding excluded).
struct FrameHashes {
    uint64_t color = 0, depth = 0, id = 0, cover = 0;
    bool operator==(const FrameHashes&) const = default;
};
FrameHashes hash_framebuffer(const Framebuffer& fb, uint32_t width, uint32_t height);

}  // namespace ttr
