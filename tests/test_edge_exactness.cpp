// SPDX-License-Identifier: Apache-2.0
//
// The FP32 edge evaluation contract (src/ttraster/tile.h) against exact int64 arithmetic.
//
// Phase 1 risk: "cracks or double hits on shared edges" when the tile-relative FP32 evaluation
// rounds. The contract E = fma(a, ddx, fma(b, ddy, c)) is exact in sign for |a|, |b| <= 2^15.
// This test drives it with coefficients at the limit, edge constants chosen so that E is exactly
// 0 at pixel centres, and far-away tiles whose constant c does not fit in 24 bits, and compares
// every pixel with the integer result. It also counts what other evaluation orders get wrong
// at and beyond the limit.

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "check.h"
#include "ttraster/oracle.h"
#include "ttraster/reference.h"
#include "ttraster/setup.h"
#include "ttraster/tile.h"

using namespace ttr;

namespace {

class Rng {
public:
    explicit Rng(uint64_t s) : s_(s) {}
    uint64_t next() {
        s_ += 0x9E3779B97F4A7C15ull;
        uint64_t z = s_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    int64_t range(int64_t lo, int64_t hi) { return lo + static_cast<int64_t>(next() % static_cast<uint64_t>(hi - lo + 1)); }

private:
    uint64_t s_;
};

struct Counts {
    uint64_t samples = 0;
    uint64_t zero_samples = 0;     // pixels where the exact value is 0 (the top-left rule decides)
    uint64_t wrong_contract = 0;      // fma(a, ddx, fma(b, ddy, c)): the contract, c innermost
    uint64_t wrong_products_first = 0;   // (a*ddx + b*ddy) + c, unfused
    uint64_t wrong_products_fma = 0;     // fma(a, ddx, b*ddy) + c
    uint64_t wrong_c_first = 0;          // (c + a*ddx) + b*ddy, unfused
};

void run_edge(Rng& rng, Counts& n, int64_t amax, bool through_pixel_centres) {
    Edge e{};
    e.a = rng.range(-amax, amax);
    e.b = rng.range(-amax, amax);
    if (rng.range(0, 7) == 0) {
        e.a = (rng.range(0, 1) != 0 ? 1 : -1) * amax;  // exactly at the limit
    }
    if (rng.range(0, 7) == 0) {
        e.b = (rng.range(0, 1) != 0 ? 1 : -1) * amax;
    }
    if (rng.range(0, 15) == 0) {
        e.a = 0;  // horizontal or vertical edges
    }
    if (rng.range(0, 15) == 0) {
        e.b = 0;
    }
    e.top_left = e.a > 0 || (e.a == 0 && e.b > 0);

    // A random tile of a 2048 x 2048 target and a pixel q in it that the edge passes through.
    const int tx = static_cast<int>(rng.range(0, 63)), ty = static_cast<int>(rng.range(0, 63));
    const int64_t qx = 16 * (tx * 32 + rng.range(0, 31)) + 8;
    const int64_t qy = 16 * (ty * 32 + rng.range(0, 31)) + 8;
    e.c = -e.a * qx - e.b * qy;
    if (!through_pixel_centres) {
        // Push the edge off the pixel grid by a small integer, or far away from the tile.
        e.c += rng.range(0, 3) == 0 ? rng.range(-(1ll << 30), 1ll << 30) : rng.range(-40, 40);
    }

    const TileEdge te = make_tile_edge(e, tx, ty);
    const float c = te.c;
    for (int j = 0; j < kTileSize; j++) {
        for (int i = 0; i < kTileSize; i++) {
            const int64_t px = 16 * (tx * 32 + i) + 8, py = 16 * (ty * 32 + j) + 8;
            const int64_t exact = e.a * px + e.b * py + e.c - (e.top_left ? 0 : 1);
            const bool want = exact >= 0;
            const float ddx = tile_offset(i), ddy = tile_offset(j);
            n.samples++;
            n.zero_samples += exact + (e.top_left ? 0 : 1) == 0;

            n.wrong_contract += edge_inside(te, ddx, ddy) != want;

            const float p = te.a * ddx, q = te.b * ddy;
            n.wrong_products_first += ((p + q) + c >= 0.0f) != want;
            n.wrong_products_fma += (std::fmaf(te.a, ddx, q) + c >= 0.0f) != want;
            n.wrong_c_first += (((c + p) + q) >= 0.0f) != want;
        }
    }
}

}  // namespace

int main() {
    // 1 and 2. The contract, edge by edge, with coefficients up to and beyond the 2^15 limit.
    for (int bits : {15, 16, 17, 18}) {
        Rng rng(static_cast<uint64_t>(bits));
        Counts n;
        const int iters = bits == 15 ? 40000 : 20000;
        for (int k = 0; k < iters; k++) {
            run_edge(rng, n, 1ll << bits, (k & 1) == 0);
        }
        std::printf(
            "|a|,|b| <= 2^%d: %llu pixels (%llu exactly on the edge); wrong signs:\n"
            "    fma(a,ddx, fma(b,ddy,c))  %llu   (contract)\n"
            "    (a*ddx + b*ddy) + c       %llu\n"
            "    fma(a,ddx, b*ddy) + c     %llu\n"
            "    (c + a*ddx) + b*ddy       %llu\n",
            bits, static_cast<unsigned long long>(n.samples), static_cast<unsigned long long>(n.zero_samples),
            static_cast<unsigned long long>(n.wrong_contract), static_cast<unsigned long long>(n.wrong_products_first),
            static_cast<unsigned long long>(n.wrong_products_fma), static_cast<unsigned long long>(n.wrong_c_first));
        if (bits == 15) {
            CHECK(n.zero_samples > 100000);
            CHECK(n.wrong_contract == 0 && n.wrong_products_first == 0 && n.wrong_products_fma == 0 &&
                  n.wrong_c_first == 0);
        } else if (bits == 18) {
            CHECK_MSG(n.wrong_contract > 0, "the bound is no longer tight");
        }
    }

    // 3. Whole triangles: random triangles with vertices spread over the whole guard band (so the
    //    edges have large coefficients), half of them on the half-pixel grid so that many pixel
    //    centres lie exactly on an edge. Reference coverage must equal the int64 oracle exactly.
    {
        Rng rng(3);
        for (uint32_t size : {256u, 1920u}) {
            Frame f;
            f.width = size;
            f.height = size == 256 ? 160 : 1088;
            const ClipRect g = guard_rect(f.padded_width(), f.padded_height());
            const int n_tri = size == 256 ? 4000 : 600;
            for (int k = 0; k < n_tri; k++) {
                Triangle t;
                t.flat = true;
                const bool half_grid = (k & 1) != 0;
                for (Vertex& v : t.v) {
                    int64_t x = rng.range(g.x0, g.x1), y = rng.range(g.y0, g.y1);
                    if (half_grid) {
                        x = (x / 8) * 8;  // multiples of 1/2 pixel: pixel centres can sit on edges
                        y = (y / 8) * 8;
                    }
                    v.x = static_cast<float>(x) / 16.0f;
                    v.y = static_cast<float>(y) / 16.0f;
                    v.z = 0.5f;
                    v.rhw = 1.0f;
                    v.color[0] = v.color[1] = v.color[2] = v.color[3] = 1.0f;
                }
                f.triangles.push_back(t);
            }
            const SetupResult setup = setup_frame(f);
            const TileBins bins = bin_triangles(setup, f.tiles_x(), f.tiles_y());
            const Framebuffer fb = render_reference(f, setup, bins);
            const std::vector<uint8_t> oracle = oracle_cover_from_setup(setup, fb.width, fb.height);
            uint64_t bad = 0;
            for (size_t i = 0; i < oracle.size(); i++) {
                bad += oracle[i] != fb.cover[i];
            }
            std::printf("random triangles %ux%u: %zu pieces, %llu coverage mismatches\n", f.width, f.height,
                        setup.tris.size(), static_cast<unsigned long long>(bad));
            CHECK(bad == 0);
        }
    }

    return ttr_test::finish();
}
