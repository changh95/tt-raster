// SPDX-License-Identifier: Apache-2.0
//
// A compiler from shader programs (ttraster/shader.h) to straight SFPI source for the Tensix compute kernel. The interpreter
// (kernels/ckernel_sfpu_shader.h) spends 60 to 115 cycles per instruction on fetching and decoding it and on issuing the SFPU
// operations through runtime-indexed registers; the code generated here has every register and constant index known at compile
// time, so an instruction costs the SFPU operations it needs (docs/EVALUATION_GL_VULKAN.md: about 40 times less on a Mad chain,
// 6 times on a Mad/Frc chain). The semantics are the interpreter's, case by case, and the same helper functions (sh_floor_any, sh_rcp,
// ...) are used, so compiled and interpreted programs agree bit for bit.
//
// The output is a header, included by the kernel, defining sh_dispatch(const ShTri&) which runs the compiled program with the
// program index of the triangle record and falls back to the interpreter for programs that were not compiled. Constants stay a
// run-time table (uniforms change between draws), so the code of a program depends only on its instructions.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ttraster/shader.h"

namespace ttr {

struct CodegenOptions {
    // Programs shorter than this are left to the interpreter (the dispatch of a tiny program is not worth a kernel build).
    uint32_t min_instructions = 4;
    // At most this many programs of a frame are compiled (the kernel build time grows with the code), in order.
    uint32_t max_programs = 8;
    // The compute kernel's code must fit the 70,656 bytes of kernel text of a core together with the interpreter (the build
    // fails otherwise); a program is compiled only while the estimated size of everything chosen so far stays below this.
    uint32_t max_code_bytes = 36000;
    // 2: values stay in local registers across instructions while the SFPU's 8 local registers allow it (a lifetime analysis,
    //    estimates of what each helper needs; the kernel may still fail to build). 1: every result goes to its Dest row and
    //    is loaded again (always builds). The renderer tries 2, then 1, then the interpreter.
    uint32_t level = 2;
};

// The header for `programs` (program i is the one with index i in the frame, as the record's program index says).
// `compiled` (optional) receives the number of programs that were compiled.
// Estimated bytes of SFPI code of a compiled program (calibrated against riscv-tt-elf-size of the kernels).
uint32_t estimate_compiled_bytes(const ShaderProgram& program);

std::string generate_shader_header(const std::vector<ShaderProgram>& programs, const CodegenOptions& options = {}, uint32_t* compiled = nullptr);

}  // namespace ttr
