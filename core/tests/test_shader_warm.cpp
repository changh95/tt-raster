// SPDX-License-Identifier: Apache-2.0
//
// The shader manifest (src/tensix/shader_warm.h): recording through TTR_SHADER_RECORD, reading back, duplicates, malformed files, and the
// list of program sets that did not build. No device is opened.

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "check.h"
#include "tensix/shader_warm.h"

using namespace ttr;

namespace {

ShaderProgram make_program(uint32_t n, uint32_t seed) {
    ShaderProgram p;
    p.num_inputs = 2;
    p.num_regs = 12;
    p.out_reg = 4;
    p.fragcoord_reg = seed % 2 == 0 ? -1 : 8;
    p.consts.assign(3, 0.25f);
    for (uint32_t i = 0; i < n; i++) {
        ShaderInstr in;
        for (uint32_t w = 0; w < shader::kInstrWords; w++) in.w[w] = 0x10000u * seed + 0x101u * i + w;
        p.code.push_back(in);
    }
    return p;
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("ttr-test-warm-" + std::to_string(::getpid()));
    fs::create_directories(dir);
    const std::string manifest = (dir / "sets.manifest").string();
    ::setenv("TTR_SHADER_RECORD", manifest.c_str(), 1);

    ShaderSetRecord a;
    a.programs = {make_program(5, 1), make_program(9, 2)};
    a.plane_mask = 3;
    a.less_equal = true;
    a.level = 2;
    a.codegen.max_programs = 4;
    ShaderSetRecord b = a;
    b.level = 1;   // another level is another kernel

    record_shader_set(a.programs, a.plane_mask, a.less_equal, a.level, a.codegen);
    record_shader_set(a.programs, a.plane_mask, a.less_equal, a.level, a.codegen);   // the same set again: recorded once
    record_shader_set(b.programs, b.plane_mask, b.less_equal, b.level, b.codegen);

    std::string error;
    std::vector<ShaderSetRecord> sets = read_shader_manifest(manifest, &error);
    CHECK_MSG(error.empty(), "error: %s", error.c_str());
    CHECK(sets.size() == 2);
    if (sets.size() == 2) {
        CHECK(shader_set_hash(sets[0]) == shader_set_hash(a));
        CHECK(shader_set_hash(sets[1]) == shader_set_hash(b));
        CHECK(sets[0].programs.size() == 2 && sets[0].programs[1].code.size() == 9);
        CHECK(sets[0].programs[1].code[3].w[2] == a.programs[1].code[3].w[2]);
        CHECK(sets[0].plane_mask == 3 && sets[0].less_equal && sets[0].level == 2 && sets[0].codegen.max_programs == 4);
        CHECK(sets[0].programs[0].fragcoord_reg == 8 && sets[0].programs[1].fragcoord_reg == -1);
        CHECK(sets[0].programs[0].consts.size() == 3);   // the table's size is kept (the values are not part of a kernel)
    }
    CHECK(shader_set_hash(a) != shader_set_hash(b));
    ShaderSetRecord c = a;
    c.programs[0].code[0].w[0] ^= 1;
    CHECK(shader_set_hash(a) != shader_set_hash(c));
    ShaderSetRecord d = a;
    d.programs[0].consts[0] = 99.0f;
    CHECK(shader_set_hash(a) == shader_set_hash(d));   // constants do not change the kernel

    // Two files appended to by the same recorder end up as one manifest with one magic line.
    {
        std::ifstream in(manifest);
        std::string line;
        int magic = 0;
        while (std::getline(in, line)) magic += line.rfind("ttr-shader-manifest", 0) == 0;
        CHECK(magic == 1);
    }

    // Malformed manifests are refused with a message, not read as empty.
    const std::string bad = (dir / "bad.manifest").string();
    std::ofstream(bad) << "not a manifest\n";
    sets = read_shader_manifest(bad, &error);
    CHECK(sets.empty() && !error.empty());
    error.clear();
    std::ofstream(bad) << "ttr-shader-manifest 1\nset mask 1 le 0 level 2 min 4 max 8 bytes 36000\nprog in 2 regs 8 out 4 fc -1 consts 0 n 2\n00000001 0 0 0\n";
    sets = read_shader_manifest(bad, &error);
    CHECK(sets.empty() && !error.empty());   // truncated program
    error.clear();
    sets = read_shader_manifest((dir / "missing").string(), &error);
    CHECK(sets.empty() && !error.empty());

    // Sets that did not build are remembered in the cache directory.
    ::setenv("TT_RASTER_SHADER_CACHE", (dir / "cache").c_str(), 1);
    CHECK(!shader_set_known_failed(0x1234abcdull));
    remember_shader_set_failed(0x1234abcdull);
    CHECK(shader_set_known_failed(0x1234abcdull));
    CHECK(!shader_set_known_failed(0x1234abceull));

    std::error_code ec;
    fs::remove_all(dir, ec);
    return ttr_test::finish();
}
