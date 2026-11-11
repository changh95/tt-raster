// SPDX-License-Identifier: Apache-2.0
//
// Compare two PNG images (docs/PLAN.md, Phase 1.5).
//   ttr_diff A.png B.png [--tol N] [--heatmap OUT.png]
// Prints mismatched pixels, maximum absolute error and PSNR. Exit status 0 when no pixel differs
// by more than the tolerance (default 0), 1 when some do, 2 on a usage or size error.

#include <cstdio>
#include <cstdlib>
#include <string>

#include "ttraster/image.h"

using namespace ttr;

int main(int argc, char** argv) {
    std::string pa, pb, heat;
    int tol = 0;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--tol" && i + 1 < argc) {
            tol = std::atoi(argv[++i]);
        } else if (a == "--heatmap" && i + 1 < argc) {
            heat = argv[++i];
        } else if (a[0] != '-' && pa.empty()) {
            pa = a;
        } else if (a[0] != '-' && pb.empty()) {
            pb = a;
        } else {
            pa.clear();
            break;
        }
    }
    if (pa.empty() || pb.empty()) {
        std::fprintf(stderr, "usage: ttr_diff A.png B.png [--tol N] [--heatmap OUT.png]\n");
        return 2;
    }
    Image a, b;
    if (!read_png(pa, a) || !read_png(pb, b)) {
        std::fprintf(stderr, "cannot read PNG input\n");
        return 2;
    }
    if (a.width != b.width || a.height != b.height) {
        std::fprintf(stderr, "size mismatch: %ux%u vs %ux%u\n", a.width, a.height, b.width, b.height);
        return 2;
    }
    const DiffStats d = diff_images(a, b, tol);
    std::printf(
        "pixels %llu  mismatched %llu  max_abs_error %u  psnr %.2f dB", static_cast<unsigned long long>(d.pixels),
        static_cast<unsigned long long>(d.mismatched), d.max_abs_error, d.psnr_db);
    if (d.mismatched > 0) {
        std::printf("  first (%lld, %lld)", static_cast<long long>(d.first_x), static_cast<long long>(d.first_y));
    }
    std::printf("\n");
    if (!heat.empty() && !write_png(heat, make_heatmap(a, b))) {
        std::fprintf(stderr, "cannot write %s\n", heat.c_str());
        return 2;
    }
    return d.mismatched == 0 ? 0 : 1;
}
