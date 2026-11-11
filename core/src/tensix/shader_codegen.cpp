// SPDX-License-Identifier: Apache-2.0

#include "tensix/shader_codegen.h"

#include <cstdarg>
#include <cstdio>

namespace ttr {

using namespace shader;

namespace {

struct Writer {
    std::string out;
    void line(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
        char buf[512];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof buf, fmt, ap);
        va_end(ap);
        out += buf;
        out += '\n';
    }
};

// `sfpi::vFloat <name> = operand;` with the modifiers (absolute value first, then negate). A register is read from its Dest row
// (compile-time address), except that the result of the previous instruction is still in a local variable (`carry`, the name of that
// variable) when this instruction can take the extra live value: the SFPU has 8 local registers and the compiler cannot spill a vector
// ("too few lregs to hold live values"), so values are not kept across more than one instruction.
void operand(Writer& w, const char* name, uint32_t word, int32_t carry_reg, const std::string& carry) {
    const uint32_t idx = word & kSrcIndexMask;
    std::string base;
    if (word & kSrcConst) {
        base = "s.consts[" + std::to_string(idx) + "]";
    } else if (carry_reg >= 0 && idx == static_cast<uint32_t>(carry_reg)) {
        base = carry;
    } else {
        base = "sfpi::dst_reg[kRegBase + " + std::to_string(idx) + "]";
    }
    w.line("        sfpi::vFloat %s = %s;", name, base.c_str());
    if (word & kSrcAbs) w.line("        %s = sfpi::abs(%s);", name, name);
    if (word & kSrcNeg) w.line("        %s = -%s;", name, name);
}

// Instructions whose own temporaries are few enough to hold one more live value from the previous instruction.
bool light(uint32_t op) {
    switch (op) {
        case kOpMov: case kOpAdd: case kOpMul: case kOpMad: case kOpMin: case kOpMax:
        case kOpSlt: case kOpSge: case kOpSeq: case kOpSne: case kOpCmp: case kOpSel: case kOpKill: return true;
        default: return false;
    }
}

// The instruction's computation on a, b, c into d: the interpreter's case bodies (ckernel_sfpu_shader.h, sh_run).
const char* body(uint32_t op) {
    switch (op) {
        case kOpMov: return "d = a;";
        case kOpAdd: return "d = a + b;";
        case kOpMul: return "d = a * b;";
        case kOpMad: return "d = a * b + c;";
        case kOpMin: return "d = a; v_if (b < a) { d = b; } v_endif;";
        case kOpMax: return "d = a; v_if (b > a) { d = b; } v_endif;";
        case kOpFrc: return "d = a - sh_floor_any(a);";
        case kOpFlr: return "d = sh_floor_any(a);";
        case kOpRcp: return "d = sh_rcp(a);";
        case kOpRsq: return "d = sh_rsq(sfpi::abs(a));";
        case kOpSqrt: return "sfpi::vFloat x = sfpi::abs(a); d = x * sh_rsq(x); v_if (x == 0.0f) { d = 0.0f; } v_endif;";
        case kOpEx2: return "d = sh_ex2(a);";
        case kOpLg2: return "d = sh_lg2(a);";
        case kOpSin: return "d = sh_sin_folded(sh_reduce(a));";
        case kOpCos: return "d = sh_sin_folded(sh_reduce(a) + 1.57079633f);";
        // Compares normalize -0 first (add +0), see the interpreter.
        case kOpSlt: return "sfpi::vFloat x = a + 0.0f, y = b + 0.0f; v_if (x < y) { d = 1.0f; } v_endif;";
        case kOpSge: return "sfpi::vFloat x = a + 0.0f, y = b + 0.0f; v_if (x >= y) { d = 1.0f; } v_endif;";
        case kOpSeq: return "sfpi::vFloat x = a + 0.0f, y = b + 0.0f; v_if (x == y) { d = 1.0f; } v_endif;";
        case kOpSne: return "sfpi::vFloat x = a + 0.0f, y = b + 0.0f; v_if (x != y) { d = 1.0f; } v_endif;";
        case kOpCmp: return "sfpi::vFloat x = a + 0.0f; d = c; v_if (x < 0.0f) { d = b; } v_endif;";
        case kOpSel: return "sfpi::vFloat x = a + 0.0f; d = c; v_if (x != 0.0f) { d = b; } v_endif;";
        case kOpKill: return "sfpi::vFloat x = a + 0.0f; v_if (x < 0.0f) { sfpi::dst_reg[kRegBase + kKillReg] = -1.0f; } v_endif;";
        default: return "";
    }
}

void program(Writer& w, const ShaderProgram& p, uint32_t index) {
    w.line("// Program %u: %zu instructions, %u registers.", index, p.code.size(), p.num_regs);
    w.line("__attribute__((always_inline)) inline void sh_prog_%u(const ShTri& s) {", index);
    w.line("    using namespace ttr::shader;");
    int32_t carry_reg = -1;      // the register whose value the previous instruction left in a local variable
    std::string carry;           // ... and the name of that variable
    for (size_t n = 0; n < p.code.size(); n++) {
        const uint32_t* in = p.code[n].w;
        const uint32_t op = instr_op(in[0]);
        const uint32_t sources = op_sources(op);
        const bool take = light(op);
        w.line("    // %zu", n);
        const std::string dname = "d" + std::to_string(n);
        if (op_writes(op)) w.line("    sfpi::vFloat %s = 0.0f;", dname.c_str());
        w.line("    {");
        if (op_writes(op)) w.line("        sfpi::vFloat& d = %s;", dname.c_str());
        if (sources >= 1) operand(w, "a", in[1], take ? carry_reg : -1, carry);
        if (sources >= 2) operand(w, "b", in[2], take ? carry_reg : -1, carry);
        if (sources >= 3) operand(w, "c", in[3], take ? carry_reg : -1, carry);
        w.line("        %s", body(op));
        if (op_writes(op) && (instr_flags(in[0]) & kFlagSat)) w.line("        d = sh_sat(d);");
        w.line("    }");
        if (op_writes(op)) {
            w.line("    sfpi::dst_reg[kRegBase + %u] = %s;", instr_dst(in[0]), dname.c_str());
            carry_reg = static_cast<int32_t>(instr_dst(in[0]));
            carry = dname;
        } else {
            carry_reg = -1;
        }
    }
    w.line("}");
    w.line("");
}

}  // namespace

uint32_t estimate_compiled_bytes(const ShaderProgram& p) {
    uint32_t bytes = 64;
    for (const ShaderInstr& in : p.code) {
        const uint32_t op = instr_op(in.w[0]);
        uint32_t b = 48;
        switch (op) {
            case kOpMin: case kOpMax: case kOpKill: b = 80; break;
            case kOpFrc: case kOpFlr: b = 220; break;
            case kOpRcp: b = 200; break;
            case kOpRsq: b = 140; break;
            case kOpSqrt: b = 180; break;
            case kOpEx2: b = 400; break;
            case kOpLg2: b = 450; break;
            case kOpSin: case kOpCos: b = 500; break;
            default: break;
        }
        if (instr_flags(in.w[0]) & kFlagSat) b += 60;
        bytes += b;
    }
    return bytes;
}

std::string generate_shader_header(const std::vector<ShaderProgram>& programs, const CodegenOptions& options, uint32_t* compiled) {
    Writer w;
    w.line("// Generated by ttr::generate_shader_header (core/src/tensix/shader_codegen.cpp). Included inside namespace ckernel::sfpu.");
    std::vector<uint32_t> chosen;
    uint32_t bytes = 0;
    for (uint32_t i = 0; i < programs.size() && chosen.size() < options.max_programs; i++) {
        if (programs[i].code.size() < options.min_instructions) continue;
        const uint32_t est = estimate_compiled_bytes(programs[i]);
        if (bytes + est > options.max_code_bytes) continue;   // does not fit: the interpreter runs it
        bytes += est;
        chosen.push_back(i);
    }
    for (uint32_t i : chosen) program(w, programs[i], i);
    w.line("__attribute__((always_inline)) inline void sh_dispatch(const ShTri& s) {");
    w.line("    switch (s.prog) {");
    for (uint32_t i : chosen) w.line("        case %u: sh_prog_%u(s); break;", i, i);
    w.line("        default: sh_run(s); break;");
    w.line("    }");
    w.line("}");
    if (compiled) *compiled = static_cast<uint32_t>(chosen.size());
    return w.out;
}

}  // namespace ttr
