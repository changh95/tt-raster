// SPDX-License-Identifier: Apache-2.0
//
// Fragment shader programs (instruction set: tensix/kernels/shader_isa.h), their validation, the CPU interpreter
// that defines their meaning, and a small builder for programs written by hand (tests, tools).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tensix/kernels/shader_isa.h"

namespace ttr {

struct ShaderInstr {
    uint32_t w[shader::kInstrWords] = {};
};

// A program together with its uniforms. A draw that changes a uniform uses another instance with another constant
// table; the code can be identical.
struct ShaderProgram {
    std::vector<ShaderInstr> code;
    std::vector<float> consts;
    uint32_t num_inputs = 0;      // varying components, interpolated into r[0 .. num_inputs)
    uint32_t num_regs = 0;        // registers used, at least num_inputs
    uint32_t out_reg = 0;         // fragment colour R, G, B, A in r[out_reg .. out_reg + 4)
    int32_t fragcoord_reg = -1;   // >= 0: x, y, z, 1/w of the pixel in r[fragcoord_reg .. +4); x, y are window pixel
                                  // coordinates of the pixel centre (y down), z the interpolated depth, 1/w the
                                  // interpolated rhw
    bool uses_kill() const;
};

// Empty string if the program is well formed, otherwise what is wrong: register and constant indices in range, only
// registers that were written (or are inputs) are read, operand fields of unused operands are zero, limits respected.
std::string validate_shader(const ShaderProgram& p);

struct ShaderResult {
    float rgba[4];
    bool killed;
};

// Run `p` for one pixel. `inputs` holds num_inputs interpolated components, `fragcoord` x, y, z, 1/w (ignored unless the
// program uses it). `scratch` must hold p.num_regs floats. The program must have passed validate_shader().
ShaderResult run_shader(const ShaderProgram& p, const float* inputs, const float* fragcoord, float* scratch);

// Register-allocation-free builder: every instruction result goes to a register chosen by the caller.
class ShaderBuilder {
public:
    // Operand constructors.
    struct Src {
        uint32_t bits;
        Src operator-() const { return {bits ^ shader::kSrcNeg}; }
        Src abs() const { return {(bits | shader::kSrcAbs) & ~shader::kSrcNeg}; }
    };
    explicit ShaderBuilder(uint32_t num_inputs) { p_.num_inputs = num_inputs; p_.num_regs = num_inputs; }

    Src in(uint32_t k) const { return {k}; }                   // input component k (register k)
    Src reg(uint32_t r) const { return {r}; }
    Src constant(float v);                                      // literal in the constant table (deduplicated)
    uint32_t alloc() { return p_.num_regs++; }                  // a fresh register
    uint32_t alloc4() { uint32_t r = p_.num_regs; p_.num_regs += 4; return r; }
    uint32_t fragcoord() { p_.fragcoord_reg = static_cast<int32_t>(alloc4()); return static_cast<uint32_t>(p_.fragcoord_reg); }   // returns the first of its four registers

    // Emit `dst = op(a, b, c)` into a fresh register (emit) or into the given one (emit_to); both return dst. Operands
    // are Src values (in, reg, constant): a bare register number does not convert. sat clamps the result to [0, 1].
    uint32_t emit_to(uint32_t op, uint32_t dst, Src a = {0}, Src b = {0}, Src c = {0}, bool sat = false);
    uint32_t emit(uint32_t op, Src a = {0}, Src b = {0}, Src c = {0}, bool sat = false) {
        return emit_to(op, alloc(), a, b, c, sat);
    }
    void kill(Src a);
    // The fragment colour is r[out .. out + 4); finish() returns the program.
    ShaderProgram finish(uint32_t out_reg);

private:
    ShaderProgram p_;
};

}  // namespace ttr
