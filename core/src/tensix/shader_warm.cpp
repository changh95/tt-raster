// SPDX-License-Identifier: Apache-2.0

#include "tensix/shader_warm.h"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <cinttypes>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>

namespace ttr {

namespace {
constexpr const char* kMagic = "ttr-shader-manifest 1";
}

uint64_t shader_set_hash(const std::vector<ShaderProgram>& programs, uint32_t plane_mask, bool less_equal, uint32_t level, const CodegenOptions& codegen) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    mix(plane_mask); mix(less_equal); mix(level);
    mix(codegen.min_instructions); mix(codegen.max_programs); mix(codegen.max_code_bytes);
    for (const ShaderProgram& p : programs) {
        mix(p.num_inputs); mix(p.num_regs); mix(p.out_reg); mix(static_cast<uint64_t>(p.fragcoord_reg) + 7); mix(p.consts.size());
        for (const ShaderInstr& in : p.code) for (uint32_t w : in.w) mix(w);
        mix(0xfeedu);
    }
    return h;
}

void write_shader_set(std::FILE* out, const ShaderSetRecord& r) {
    std::fprintf(out, "set mask %u le %d level %u min %u max %u bytes %u\n", r.plane_mask, r.less_equal ? 1 : 0, r.level,
                 r.codegen.min_instructions, r.codegen.max_programs, r.codegen.max_code_bytes);
    for (const ShaderProgram& p : r.programs) {
        std::fprintf(out, "prog in %u regs %u out %u fc %d consts %zu n %zu\n", p.num_inputs, p.num_regs, p.out_reg, p.fragcoord_reg,
                     p.consts.size(), p.code.size());
        for (const ShaderInstr& in : p.code) {
            for (size_t i = 0; i < shader::kInstrWords; i++) std::fprintf(out, i == 0 ? "%08x" : " %08x", in.w[i]);
            std::fputc('\n', out);
        }
    }
    std::fprintf(out, "endset\n");
}

std::vector<ShaderSetRecord> read_shader_manifest(const std::string& path, std::string* error) {
    std::vector<ShaderSetRecord> sets;
    std::set<uint64_t> seen;
    auto fail = [&](const std::string& why) {
        if (error != nullptr) *error = why;
        return sets;
    };
    std::ifstream in(path);
    if (!in) return fail("cannot open " + path);
    std::string line;
    if (!std::getline(in, line) || line != kMagic) return fail(path + " is not a shader manifest (first line must be '" + std::string(kMagic) + "')");
    ShaderSetRecord cur;
    bool open = false;
    size_t lineno = 1;
    while (std::getline(in, line)) {
        lineno++;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string word;
        ls >> word;
        if (word == "set") {
            std::string k1, k2, k3, k4, k5, k6;
            int le = 0;
            cur = ShaderSetRecord{};
            ls >> k1 >> cur.plane_mask >> k2 >> le >> k3 >> cur.level >> k4 >> cur.codegen.min_instructions >> k5 >> cur.codegen.max_programs >> k6
               >> cur.codegen.max_code_bytes;
            if (!ls || k1 != "mask" || k2 != "le" || k3 != "level" || k4 != "min" || k5 != "max" || k6 != "bytes")
                return fail(path + ":" + std::to_string(lineno) + ": bad set line");
            cur.less_equal = le != 0;
            open = true;
        } else if (word == "prog" && open) {
            ShaderProgram p;
            std::string k[6];
            size_t nconsts = 0, n = 0;
            ls >> k[0] >> p.num_inputs >> k[1] >> p.num_regs >> k[2] >> p.out_reg >> k[3] >> p.fragcoord_reg >> k[4] >> nconsts >> k[5] >> n;
            if (!ls || k[0] != "in" || k[1] != "regs" || k[2] != "out" || k[3] != "fc" || k[4] != "consts" || k[5] != "n" || n > 100000 || nconsts > 100000)
                return fail(path + ":" + std::to_string(lineno) + ": bad prog line");
            p.consts.assign(nconsts, 0.0f);
            for (size_t i = 0; i < n; i++) {
                if (!std::getline(in, line)) return fail(path + ": truncated program");
                lineno++;
                std::istringstream ws(line);
                ShaderInstr ins;
                for (size_t w = 0; w < shader::kInstrWords; w++) ws >> std::hex >> ins.w[w];
                if (!ws) return fail(path + ":" + std::to_string(lineno) + ": bad instruction");
                p.code.push_back(ins);
            }
            cur.programs.push_back(std::move(p));
        } else if (word == "endset" && open) {
            open = false;
            if (!cur.programs.empty() && seen.insert(shader_set_hash(cur)).second) sets.push_back(cur);
        } else {
            return fail(path + ":" + std::to_string(lineno) + ": unexpected '" + word + "'");
        }
    }
    if (open) return fail(path + ": truncated set");
    return sets;
}

std::string shader_cache_root() {
    const char* root_env = std::getenv("TT_RASTER_SHADER_CACHE");
    if (root_env != nullptr && root_env[0] != '\0') return root_env;
    return std::string(std::getenv("HOME") != nullptr ? std::getenv("HOME") : "/tmp") + "/.cache/tt-raster/shader-kernels";
}

bool shader_set_known_failed(uint64_t set_hash) {
    std::ifstream in(shader_cache_root() + "/failed-sets.txt");
    std::string line;
    char want[24];
    std::snprintf(want, sizeof want, "%016" PRIx64, set_hash);
    while (std::getline(in, line)) {
        if (line == want) return true;
    }
    return false;
}

void remember_shader_set_failed(uint64_t set_hash) {
    std::error_code ec;
    std::filesystem::create_directories(shader_cache_root(), ec);
    const int fd = ::open((shader_cache_root() + "/failed-sets.txt").c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) return;
    char line[24];
    const int n = std::snprintf(line, sizeof line, "%016" PRIx64 "\n", set_hash);
    if (::write(fd, line, static_cast<size_t>(n)) < 0) { /* the list is an optimisation */ }
    ::close(fd);
}

void record_shader_set(const std::vector<ShaderProgram>& programs, uint32_t plane_mask, bool less_equal, uint32_t level, const CodegenOptions& codegen) {
    const char* path = std::getenv("TTR_SHADER_RECORD");
    if (path == nullptr || path[0] == '\0') return;
    static std::mutex mutex;
    static std::set<uint64_t> done;
    std::lock_guard<std::mutex> lock(mutex);
    if (!done.insert(shader_set_hash(programs, plane_mask, less_equal, level, codegen)).second) return;
    ShaderSetRecord record;
    record.programs = programs;
    record.plane_mask = plane_mask;
    record.less_equal = less_equal;
    record.level = level;
    record.codegen = codegen;
    const int fd = ::open(path, O_RDWR | O_CREAT | O_APPEND, 0644);
    if (fd < 0) {
        std::fprintf(stderr, "tensix: cannot record shader sets in %s\n", path);
        return;
    }
    ::flock(fd, LOCK_EX);
    // A fresh file starts with the magic line. The sets are written with one write() call, so that readers never see half a set.
    std::string text;
    if (::lseek(fd, 0, SEEK_END) == 0) text = std::string(kMagic) + "\n";
    std::FILE* mem = ::tmpfile();
    if (mem != nullptr) {
        write_shader_set(mem, record);
        const long n = std::ftell(mem);
        std::rewind(mem);
        std::string body(static_cast<size_t>(n), '\0');
        if (std::fread(body.data(), 1, body.size(), mem) == body.size()) text += body;
        std::fclose(mem);
    }
    if (::write(fd, text.data(), text.size()) != static_cast<ssize_t>(text.size())) std::fprintf(stderr, "tensix: short write to %s\n", path);
    ::flock(fd, LOCK_UN);
    ::close(fd);
}

}  // namespace ttr
