// SPDX-License-Identifier: Apache-2.0

#include "ttraster/image.h"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace ttr {

namespace {

void put_be32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(static_cast<uint8_t>(x >> 24));
    v.push_back(static_cast<uint8_t>(x >> 16));
    v.push_back(static_cast<uint8_t>(x >> 8));
    v.push_back(static_cast<uint8_t>(x));
}

void put_chunk(std::vector<uint8_t>& out, const char type[4], const std::vector<uint8_t>& data) {
    put_be32(out, static_cast<uint32_t>(data.size()));
    const size_t start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    const uLong crc = crc32(0, out.data() + start, static_cast<uInt>(out.size() - start));
    put_be32(out, static_cast<uint32_t>(crc));
}

uint32_t get_be32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | p[3];
}

bool read_file(const std::string& path, std::vector<uint8_t>& data) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    data.resize(static_cast<size_t>(n));
    const bool ok = n == 0 || std::fread(data.data(), 1, data.size(), f) == data.size();
    std::fclose(f);
    return ok;
}

bool write_file(const std::string& path, const std::vector<uint8_t>& data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    const bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
    return std::fclose(f) == 0 && ok;
}

int paeth(int a, int b, int c) {
    const int p = a + b - c;
    const int pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
    if (pa <= pb && pa <= pc) {
        return a;
    }
    return pb <= pc ? b : c;
}

}  // namespace

Image image_from_framebuffer(const Framebuffer& fb, uint32_t width, uint32_t height) {
    Image img;
    img.width = width;
    img.height = height;
    img.rgba.resize(static_cast<size_t>(width) * height * 4);
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            const uint32_t c = fb.color[fb.index(x, y)];
            uint8_t* p = &img.rgba[(static_cast<size_t>(y) * width + x) * 4];
            p[0] = static_cast<uint8_t>(c);
            p[1] = static_cast<uint8_t>(c >> 8);
            p[2] = static_cast<uint8_t>(c >> 16);
            p[3] = static_cast<uint8_t>(c >> 24);
        }
    }
    return img;
}

Image crop_image(const Image& img, uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    Image out;
    out.width = w;
    out.height = h;
    out.rgba.resize(static_cast<size_t>(w) * h * 4);
    for (uint32_t j = 0; j < h; j++) {
        std::memcpy(&out.rgba[static_cast<size_t>(j) * w * 4], img.pixel(x, y + j), static_cast<size_t>(w) * 4);
    }
    return out;
}

bool write_png(const std::string& path, const Image& img) {
    const size_t stride = static_cast<size_t>(img.width) * 4;
    std::vector<uint8_t> raw((stride + 1) * img.height);
    for (uint32_t y = 0; y < img.height; y++) {
        const uint8_t* src = &img.rgba[y * stride];
        uint8_t* dst = &raw[y * (stride + 1)];
        dst[0] = 1;  // Sub filter
        for (size_t i = 0; i < stride; i++) {
            dst[1 + i] = static_cast<uint8_t>(src[i] - (i >= 4 ? src[i - 4] : 0));
        }
    }
    uLongf bound = compressBound(static_cast<uLong>(raw.size()));
    std::vector<uint8_t> z(bound);
    if (compress2(z.data(), &bound, raw.data(), static_cast<uLong>(raw.size()), 9) != Z_OK) {
        return false;
    }
    z.resize(bound);

    std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    std::vector<uint8_t> ihdr;
    put_be32(ihdr, img.width);
    put_be32(ihdr, img.height);
    ihdr.insert(ihdr.end(), {8, 6, 0, 0, 0});  // 8 bit, RGBA, deflate, adaptive filtering, no interlace
    put_chunk(out, "IHDR", ihdr);
    put_chunk(out, "IDAT", z);
    put_chunk(out, "IEND", {});
    return write_file(path, out);
}

bool read_png(const std::string& path, Image& img) {
    std::vector<uint8_t> f;
    if (!read_file(path, f) || f.size() < 8 || std::memcmp(f.data(), "\x89PNG\r\n\x1a\n", 8) != 0) {
        return false;
    }
    uint32_t w = 0, h = 0;
    int channels = 0;
    std::vector<uint8_t> idat;
    for (size_t pos = 8; pos + 12 <= f.size();) {
        const uint32_t len = get_be32(&f[pos]);
        const char* type = reinterpret_cast<const char*>(&f[pos + 4]);
        if (pos + 12 + len > f.size()) {
            return false;
        }
        const uint8_t* data = &f[pos + 8];
        if (std::memcmp(type, "IHDR", 4) == 0 && len >= 13) {
            w = get_be32(data);
            h = get_be32(data + 4);
            const int depth = data[8], color = data[9], interlace = data[12];
            if (depth != 8 || interlace != 0 || (color != 6 && color != 2)) {
                return false;
            }
            channels = color == 6 ? 4 : 3;
        } else if (std::memcmp(type, "IDAT", 4) == 0) {
            idat.insert(idat.end(), data, data + len);
        }
        pos += 12 + len;
    }
    if (w == 0 || h == 0 || channels == 0) {
        return false;
    }
    const size_t stride = static_cast<size_t>(w) * channels;
    std::vector<uint8_t> raw((stride + 1) * h);
    uLongf raw_len = static_cast<uLongf>(raw.size());
    if (uncompress(raw.data(), &raw_len, idat.data(), static_cast<uLong>(idat.size())) != Z_OK ||
        raw_len != raw.size()) {
        return false;
    }
    // Undo the row filters in place.
    for (uint32_t y = 0; y < h; y++) {
        uint8_t* row = &raw[y * (stride + 1)];
        const uint8_t* prev = y > 0 ? &raw[(y - 1) * (stride + 1) + 1] : nullptr;
        const int type = row[0];
        uint8_t* px = row + 1;
        for (size_t i = 0; i < stride; i++) {
            const int a = i >= static_cast<size_t>(channels) ? px[i - channels] : 0;
            const int b = prev != nullptr ? prev[i] : 0;
            const int c = (prev != nullptr && i >= static_cast<size_t>(channels)) ? prev[i - channels] : 0;
            int add = 0;
            switch (type) {
                case 0: add = 0; break;
                case 1: add = a; break;
                case 2: add = b; break;
                case 3: add = (a + b) / 2; break;
                case 4: add = paeth(a, b, c); break;
                default: return false;
            }
            px[i] = static_cast<uint8_t>(px[i] + add);
        }
    }
    img.width = w;
    img.height = h;
    img.rgba.resize(static_cast<size_t>(w) * h * 4);
    for (uint32_t y = 0; y < h; y++) {
        const uint8_t* src = &raw[y * (stride + 1) + 1];
        for (uint32_t x = 0; x < w; x++) {
            uint8_t* dst = &img.rgba[(static_cast<size_t>(y) * w + x) * 4];
            dst[0] = src[x * channels];
            dst[1] = src[x * channels + 1];
            dst[2] = src[x * channels + 2];
            dst[3] = channels == 4 ? src[x * channels + 3] : 255;
        }
    }
    return true;
}

bool write_ppm(const std::string& path, const Image& img) {
    const std::string header = "P6\n" + std::to_string(img.width) + " " + std::to_string(img.height) + "\n255\n";
    std::vector<uint8_t> out(header.begin(), header.end());
    out.reserve(out.size() + static_cast<size_t>(img.width) * img.height * 3);
    for (size_t i = 0; i < static_cast<size_t>(img.width) * img.height; i++) {
        out.push_back(img.rgba[i * 4]);
        out.push_back(img.rgba[i * 4 + 1]);
        out.push_back(img.rgba[i * 4 + 2]);
    }
    return write_file(path, out);
}

DiffStats diff_images(const Image& a, const Image& b, int tolerance) {
    DiffStats d;
    d.pixels = static_cast<uint64_t>(a.width) * a.height;
    double sq = 0.0;
    for (uint32_t y = 0; y < a.height; y++) {
        for (uint32_t x = 0; x < a.width; x++) {
            const uint8_t* pa = a.pixel(x, y);
            const uint8_t* pb = b.pixel(x, y);
            int worst = 0;
            for (int c = 0; c < 4; c++) {
                const int e = std::abs(static_cast<int>(pa[c]) - static_cast<int>(pb[c]));
                worst = std::max(worst, e);
                sq += static_cast<double>(e) * e;
            }
            d.max_abs_error = std::max<uint32_t>(d.max_abs_error, static_cast<uint32_t>(worst));
            if (worst > tolerance) {
                if (d.mismatched == 0) {
                    d.first_x = x;
                    d.first_y = y;
                }
                d.mismatched++;
            }
        }
    }
    const double mse = sq / (static_cast<double>(d.pixels) * 4.0);
    d.psnr_db = mse == 0.0 ? std::numeric_limits<double>::infinity() : 10.0 * std::log10(255.0 * 255.0 / mse);
    return d;
}

Image make_heatmap(const Image& a, const Image& b) {
    Image h;
    h.width = a.width;
    h.height = a.height;
    h.rgba.assign(static_cast<size_t>(a.width) * a.height * 4, 0);
    for (size_t i = 0; i < static_cast<size_t>(a.width) * a.height; i++) {
        int worst = 0;
        for (int c = 0; c < 4; c++) {
            worst = std::max(worst, std::abs(static_cast<int>(a.rgba[i * 4 + c]) - static_cast<int>(b.rgba[i * 4 + c])));
        }
        uint8_t* p = &h.rgba[i * 4];
        p[3] = 255;
        if (worst > 0) {
            const float t = std::min(1.0f, static_cast<float>(worst) / 64.0f);
            p[0] = 255;
            p[1] = static_cast<uint8_t>(t * 255.0f);
            p[2] = 0;
            if (worst <= 1) {  // a 1 LSB difference is dark red so it stays visible but quiet
                p[0] = 120;
                p[1] = 0;
            }
        }
    }
    return h;
}

uint64_t fnv1a64(const void* data, size_t bytes, uint64_t seed) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t h = seed;
    for (size_t i = 0; i < bytes; i++) {
        h = (h ^ p[i]) * 0x100000001b3ull;
    }
    return h;
}

FrameHashes hash_framebuffer(const Framebuffer& fb, uint32_t width, uint32_t height) {
    FrameHashes h;
    h.color = h.depth = h.id = h.cover = 0xcbf29ce484222325ull;
    for (uint32_t y = 0; y < height; y++) {
        const size_t row = fb.index(0, y);
        h.color = fnv1a64(&fb.color[row], width * sizeof(uint32_t), h.color);
        h.depth = fnv1a64(&fb.depth[row], width * sizeof(float), h.depth);
        h.id = fnv1a64(&fb.id[row], width * sizeof(uint32_t), h.id);
        h.cover = fnv1a64(&fb.cover[row], width, h.cover);
    }
    return h;
}

}  // namespace ttr
