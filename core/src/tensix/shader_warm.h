// SPDX-License-Identifier: Apache-2.0
//
// Pre-warming the compiled-shader kernel cache. A frame whose fragment shaders are compiled (TTR_SHADER_COMPILE=1) runs a compute kernel
// built for the set of programs the frame uses; the first time a set appears the build takes seconds (docs/EVALUATION_GL_VULKAN.md),
// after that tt-metal finds the binary in its cache. This file records the sets a run needed (TTR_SHADER_RECORD=<manifest>), and replays
// a manifest on a device (tools/ttr_prewarm, or TTR_SHADER_WARM=<manifest> at start-up of the Tensix backend), so that a later run
// starts with a warm cache and its first frame does not stall. A manifest is a text file; the programs' constants are not part of it
// (the kernel does not depend on them).

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "tensix/shader_codegen.h"
#include "ttraster/shader.h"

namespace ttr {

// Everything that makes a compiled kernel what it is: the programs, the planes and depth comparison the kernel is specialised for, the
// code generation level that finally built, and the options that chose which programs were compiled.
struct ShaderSetRecord {
    std::vector<ShaderProgram> programs;
    uint32_t plane_mask = 1;
    bool less_equal = false;
    uint32_t level = 2;
    CodegenOptions codegen;
};

// Identity of a program set as a kernel sees it (the same set recorded twice is one).
uint64_t shader_set_hash(const std::vector<ShaderProgram>& programs, uint32_t plane_mask, bool less_equal, uint32_t level, const CodegenOptions& codegen);
inline uint64_t shader_set_hash(const ShaderSetRecord& r) { return shader_set_hash(r.programs, r.plane_mask, r.less_equal, r.level, r.codegen); }

// Where compiled-shader kernel directories and the list of program sets that did not build live (TT_RASTER_SHADER_CACHE, default
// ~/.cache/tt-raster/shader-kernels).
std::string shader_cache_root();

// Program sets whose compiled kernel did not build (identified by a hash that includes the kernel sources), kept in the cache directory so
// that the next process does not repeat a failing build: a failed build costs as much time as a good one and is not cached by tt-metal.
bool shader_set_known_failed(uint64_t set_hash);
void remember_shader_set_failed(uint64_t set_hash);

void write_shader_set(std::FILE* out, const ShaderSetRecord& record);

// All sets of a manifest (distinct ones, in file order). On a malformed file returns what was read before it and sets *error.
std::vector<ShaderSetRecord> read_shader_manifest(const std::string& path, std::string* error);

// Append the record to the manifest named by TTR_SHADER_RECORD (nothing if it is unset), once per distinct set and process; the file is
// locked while it is written, so several processes can record into one manifest.
// Only copies the programs when recording is on.
void record_shader_set(const std::vector<ShaderProgram>& programs, uint32_t plane_mask, bool less_equal, uint32_t level, const CodegenOptions& codegen);

}  // namespace ttr
