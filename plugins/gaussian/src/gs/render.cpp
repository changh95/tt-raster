// SPDX-License-Identifier: Apache-2.0

#include "gs/render.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "ttraster/frame.h"
#include "ttraster/parallel.h"

namespace ttr::gs {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Front to back: nearer first, scene order among equal depths.
std::vector<uint32_t> depth_order(const std::vector<Splat>& s) {
    std::vector<uint32_t> order(s.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return s[a].depth != s[b].depth ? s[a].depth < s[b].depth : s[a].index < s[b].index; });
    return order;
}
}  // namespace

FloatImage render_reference(const std::vector<Splat>& splats, const Camera& cam, const float bg[3], RenderStats* stats, size_t threads) {
    constexpr uint32_t kTile = 32;
    const uint32_t W = cam.width, H = cam.height;
    const uint32_t tx_n = (W + kTile - 1) / kTile, ty_n = (H + kTile - 1) / kTile;
    auto t0 = Clock::now();
    const std::vector<uint32_t> order = depth_order(splats);

    // Two passes over the splats in depth order: count the tiles' entries, then fill the lists, which are then in depth order by construction.
    auto tile_range = [&](const Splat& s, int* x0, int* x1, int* y0, int* y1) {
        *x0 = std::max(0, static_cast<int>(std::floor((s.mx - s.radius) / kTile)));
        *x1 = std::min(static_cast<int>(tx_n) - 1, static_cast<int>(std::floor((s.mx + s.radius) / kTile)));
        *y0 = std::max(0, static_cast<int>(std::floor((s.my - s.radius) / kTile)));
        *y1 = std::min(static_cast<int>(ty_n) - 1, static_cast<int>(std::floor((s.my + s.radius) / kTile)));
    };
    std::vector<uint32_t> start(static_cast<size_t>(tx_n) * ty_n + 1, 0);
    for (uint32_t id : order) {
        int x0, x1, y0, y1;
        tile_range(splats[id], &x0, &x1, &y0, &y1);
        for (int ty = y0; ty <= y1; ty++) for (int tx = x0; tx <= x1; tx++) start[static_cast<size_t>(ty) * tx_n + tx + 1]++;
    }
    for (size_t i = 1; i < start.size(); i++) start[i] += start[i - 1];
    std::vector<uint32_t> lists(start.back());
    std::vector<uint32_t> fill(start.begin(), start.end() - 1);
    for (uint32_t id : order) {
        int x0, x1, y0, y1;
        tile_range(splats[id], &x0, &x1, &y0, &y1);
        for (int ty = y0; ty <= y1; ty++) for (int tx = x0; tx <= x1; tx++) lists[fill[static_cast<size_t>(ty) * tx_n + tx]++] = id;
    }
    const double bin_ms = ms_since(t0);

    t0 = Clock::now();
    FloatImage img;
    img.width = W;
    img.height = H;
    img.rgb.assign(static_cast<size_t>(W) * H * 3, 0.0f);
    parallel_for(static_cast<size_t>(tx_n) * ty_n, threads, [&](size_t tile) {
        const uint32_t tx = static_cast<uint32_t>(tile % tx_n), ty = static_cast<uint32_t>(tile / tx_n);
        const uint32_t px0 = tx * kTile, py0 = ty * kTile, px1 = std::min(W, px0 + kTile), py1 = std::min(H, py0 + kTile);
        const uint32_t* list = lists.data() + start[tile];
        const uint32_t count = start[tile + 1] - start[tile];
        for (uint32_t y = py0; y < py1; y++) {
            for (uint32_t x = px0; x < px1; x++) {
                const float fx = static_cast<float>(x) + 0.5f, fy = static_cast<float>(y) + 0.5f;
                float T = 1.0f, cr = 0.0f, cg = 0.0f, cb = 0.0f;
                for (uint32_t k = 0; k < count; k++) {
                    const Splat& s = splats[list[k]];
                    const float dx = fx - s.mx, dy = fy - s.my;
                    const float power = -0.5f * (s.conic[0] * dx * dx + s.conic[2] * dy * dy) - s.conic[1] * dx * dy;
                    if (power > 0.0f) continue;
                    const float alpha = std::min(0.99f, s.opacity * std::exp(power));
                    if (alpha < 1.0f / 255.0f) continue;
                    const float test_T = T * (1.0f - alpha);
                    if (test_T < 0.0001f) break;
                    cr += s.rgb[0] * alpha * T; cg += s.rgb[1] * alpha * T; cb += s.rgb[2] * alpha * T;
                    T = test_T;
                }
                float* o = &img.rgb[(static_cast<size_t>(y) * W + x) * 3];
                o[0] = cr + T * bg[0]; o[1] = cg + T * bg[1]; o[2] = cb + T * bg[2];
            }
        }
    });
    if (stats != nullptr) {
        stats->splats = splats.size();
        stats->tile_records = lists.size();
        stats->bin_ms = bin_ms;
        stats->raster_ms = ms_since(t0);
    }
    return img;
}

Image to_image(const FloatImage& f) {
    Image img;
    img.width = f.width;
    img.height = f.height;
    img.rgba.resize(static_cast<size_t>(f.width) * f.height * 4);
    for (size_t i = 0; i < static_cast<size_t>(f.width) * f.height; i++) {
        for (int c = 0; c < 3; c++) img.rgba[4 * i + c] = static_cast<uint8_t>(std::lround(std::clamp(f.rgb[3 * i + c], 0.0f, 1.0f) * 255.0f));
        img.rgba[4 * i + 3] = 255;
    }
    return img;
}

ShaderProgram splat_shader() {
    // Inputs (the same value at the three vertices): mx, my, conic a, 2 conic b, conic c, opacity, r, g, b.
    ShaderBuilder sb(9);
    const uint32_t fc = sb.fragcoord();
    const auto in = [&](uint32_t k) { return sb.in(k); };
    const auto reg = [&](uint32_t r) { return sb.reg(r); };
    const uint32_t dx = sb.emit(shader::kOpAdd, reg(fc), -in(0));
    const uint32_t dy = sb.emit(shader::kOpAdd, reg(fc + 1), -in(1));
    const uint32_t e1 = sb.emit(shader::kOpMul, in(2), reg(dx));                   // a dx
    const uint32_t u = sb.emit(shader::kOpMad, in(3), reg(dy), reg(e1));           // a dx + 2 b dy
    const uint32_t e3 = sb.emit(shader::kOpMul, in(4), reg(dy));                   // c dy
    const uint32_t e4 = sb.emit(shader::kOpMul, reg(e3), reg(dy));                 // c dy^2
    const uint32_t e5 = sb.emit(shader::kOpMad, reg(dx), reg(u), reg(e4));         // a dx^2 + 2 b dx dy + c dy^2 = -2 power
    sb.kill(reg(e5));                                                               // power > 0: skipped, as in the reference
    const uint32_t x = sb.emit(shader::kOpMul, reg(e5), sb.constant(-0.5f * 1.4426950408889634f));   // power * log2(e)
    const uint32_t ex = sb.emit(shader::kOpEx2, reg(x));
    const uint32_t a0 = sb.emit(shader::kOpMul, reg(ex), in(5));
    const uint32_t alpha = sb.emit(shader::kOpMin, reg(a0), sb.constant(0.99f));
    const uint32_t t = sb.emit(shader::kOpAdd, reg(alpha), sb.constant(-1.0f / 255.0f));
    sb.kill(reg(t));                                                                // alpha < 1/255
    const uint32_t out = sb.alloc4();
    sb.emit_to(shader::kOpMov, out, in(6));
    sb.emit_to(shader::kOpMov, out + 1, in(7));
    sb.emit_to(shader::kOpMov, out + 2, in(8));
    sb.emit_to(shader::kOpMov, out + 3, reg(alpha));
    return sb.finish(out);
}

Image render_backend(const std::vector<Splat>& splats, const Camera& cam, const float bg[3], Backend& backend, RenderStats* stats) {
    if (cam.width == 0 || cam.height == 0 || cam.width > static_cast<uint32_t>(kMaxTargetSize) || cam.height > static_cast<uint32_t>(kMaxTargetSize))
        throw std::runtime_error("the target must be 1 to 2048 pixels in each direction");
    auto t0 = Clock::now();
    const std::vector<uint32_t> order = depth_order(splats);   // front to back
    Frame f;
    f.width = cam.width;
    f.height = cam.height;
    f.clear_color[0] = bg[0]; f.clear_color[1] = bg[1]; f.clear_color[2] = bg[2]; f.clear_color[3] = 1.0f;
    f.depth_func = DepthFunc::Always;
    f.shaders.push_back(splat_shader());
    f.triangles.reserve(2 * splats.size());
    f.attrs.reserve(27 * splats.size());
    for (size_t k = order.size(); k-- > 0;) {   // back to front
        const Splat& s = splats[order[k]];
        const uint32_t attr = static_cast<uint32_t>(f.attrs.size());
        const float v[9] = {s.mx, s.my, s.conic[0], 2.0f * s.conic[1], s.conic[2], s.opacity, s.rgb[0], s.rgb[1], s.rgb[2]};
        for (int vert = 0; vert < 3; vert++) f.attrs.insert(f.attrs.end(), v, v + 9);
        const float x0 = s.mx - s.radius, x1 = s.mx + s.radius, y0 = s.my - s.radius, y1 = s.my + s.radius;
        const float cx[4] = {x0, x1, x1, x0}, cy[4] = {y0, y0, y1, y1};
        static const int kQuad[2][3] = {{0, 1, 2}, {0, 2, 3}};
        for (const auto& idx : kQuad) {
            Triangle t;
            for (int j = 0; j < 3; j++) {
                t.v[j] = Vertex{};
                t.v[j].x = cx[idx[j]];
                t.v[j].y = cy[idx[j]];
                t.v[j].z = 0.5f;
                t.v[j].rhw = 1.0f;
            }
            t.own_state = true;
            t.depth_func = DepthFunc::Always;
            t.depth_write = false;
            t.blend = BlendMode::Alpha;
            t.shader = 1;
            t.attr_offset = attr;
            f.triangles.push_back(t);
        }
    }
    const double build_ms = ms_since(t0);
    t0 = Clock::now();
    const Framebuffer& fb = backend.render(f, kRenderColor);
    if (stats != nullptr) {
        stats->splats = splats.size();
        stats->build_ms = build_ms;
        stats->render_ms = ms_since(t0);
    }
    return image_from_framebuffer(fb, cam.width, cam.height);
}

double psnr_rgb(const Image& a, const Image& b, uint32_t* max_error) {
    double se = 0;
    uint32_t mx = 0;
    const size_t n = static_cast<size_t>(a.width) * a.height;
    for (size_t i = 0; i < n; i++) {
        for (int c = 0; c < 3; c++) {
            const int d = static_cast<int>(a.rgba[4 * i + c]) - static_cast<int>(b.rgba[4 * i + c]);
            se += static_cast<double>(d) * d;
            mx = std::max<uint32_t>(mx, static_cast<uint32_t>(std::abs(d)));
        }
    }
    if (max_error != nullptr) *max_error = mx;
    if (se == 0) return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(255.0 * 255.0 / (se / (3.0 * static_cast<double>(n))));
}

}  // namespace ttr::gs
