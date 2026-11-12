// SPDX-License-Identifier: Apache-2.0
//
// Fills tt-metal's kernel cache with the compiled-shader kernels of a recorded run, so that the first frame of the next run does not
// stall on a kernel build (src/tensix/shader_warm.h).
//
//   TTR_SHADER_COMPILE=1 TTR_SHADER_RECORD=sets.manifest <run the application once>
//   scripts/run.sh 5 build/core/tools/ttr_prewarm sets.manifest [--check]
//
// --check builds nothing new: it times each set a second time and fails (exit status 1) if any set took more than --limit seconds
// (default 5 s) the second time, i.e. if the cache did not hold it. The kernels are cached per kernel source and defines, not per chip,
// so warming one chip serves every chip of the same kind.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "tensix/tensix_renderer.h"

using namespace ttr;

int main(int argc, char** argv) {
    std::string path;
    bool check = false;
    double limit = 5.0;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--check") check = true;
        else if (a == "--limit" && i + 1 < argc) limit = std::atof(argv[++i]);
        else if (a[0] != '-' && path.empty()) path = a;
        else {
            std::fprintf(stderr, "usage: ttr_prewarm <manifest> [--check] [--limit SECONDS]\n");
            return 2;
        }
    }
    if (path.empty()) {
        std::fprintf(stderr, "usage: ttr_prewarm <manifest> [--check] [--limit SECONDS]\n");
        return 2;
    }
    std::string error;
    const std::vector<ShaderSetRecord> sets = read_shader_manifest(path, &error);
    if (!error.empty()) {
        std::fprintf(stderr, "ttr_prewarm: %s\n", error.c_str());
        return 2;
    }
    if (sets.empty()) {
        std::fprintf(stderr, "ttr_prewarm: %s holds no program sets\n", path.c_str());
        return 2;
    }
    // The kernels with compiled shaders are only built when asked for.
    setenv("TTR_SHADER_COMPILE", "1", 0);
    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    TensixRenderer renderer(mesh);
    int slow = 0;
    for (size_t i = 0; i < sets.size(); i++) {
        const ShaderSetRecord& s = sets[i];
        size_t instrs = 0;
        for (const ShaderProgram& p : s.programs) instrs += p.code.size();
        const double first = renderer.prewarm(s);
        std::printf("set %zu/%zu: %zu programs, %zu instructions, level %u, planes 0x%x: %.1f s", i + 1, sets.size(), s.programs.size(), instrs, s.level,
                    s.plane_mask, first);
        if (check) {
            const double second = renderer.prewarm(s);
            std::printf(", again %.2f s%s", second, second > limit ? "  <-- not cached" : "");
            if (second > limit) slow++;
        }
        std::printf("\n");
    }
    mesh->close();
    return slow == 0 ? 0 : 1;
}
