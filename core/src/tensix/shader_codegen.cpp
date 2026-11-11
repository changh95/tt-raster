// SPDX-License-Identifier: Apache-2.0

#include "tensix/shader_codegen.h"

#include <algorithm>
#include <array>
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

// `sfpi::vFloat <name> = operand;` with the modifiers (absolute value first, then negate). `local` is the name of the local variable
// that holds the register's value, or empty: then it is read from its Dest row (compile-time address).
void operand(Writer& w, const char* name, uint32_t word, const std::string& local) {
    const uint32_t idx = word & kSrcIndexMask;
    std::string base;
    if (word & kSrcConst) {
        base = "s.consts[" + std::to_string(idx) + "]";
    } else if (!local.empty()) {
        base = local;
    } else {
        base = "sfpi::dst_reg[kRegBase + " + std::to_string(idx) + "]";
    }
    w.line("        sfpi::vFloat %s = %s;", name, base.c_str());
    if (word & kSrcAbs) w.line("        %s = sfpi::abs(%s);", name, name);
    if (word & kSrcNeg) w.line("        %s = -%s;", name, name);
}

// Local registers (LREGs) an instruction needs while it runs, besides the values live across it: its operands, its result and the
// temporaries of the helper it expands to (peak live values in ckernel_sfpu_shader.h, with one spare). The SFPU has 8 and the compiler
// cannot spill a vector ("too few lregs to hold live values"), so the allocator below keeps a value in a local variable across
// instructions only while all of these fit. The numbers are estimates; a kernel that still fails to build is retried with
// level 1 (no values kept across instructions) and then with the interpreter.
uint32_t temporaries(uint32_t op) {
    switch (op) {
        case kOpMin: case kOpMax: case kOpCmp: case kOpSel: case kOpKill: return 2;   // a predicated block copies values
        case kOpSlt: case kOpSge: case kOpSeq: case kOpSne: return 3;
        case kOpFrc: case kOpFlr: case kOpRcp: case kOpRsq: case kOpSin: case kOpCos: return 4;
        case kOpSqrt: case kOpEx2: return 5;
        case kOpLg2: return 6;
        default: return 0;
    }
}

constexpr uint32_t kLocalRegisters = 7;   // of the 8, one is kept free for the compiler

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

void program(Writer& w, const ShaderProgram& p, uint32_t index, uint32_t level) {
    const size_t n_instr = p.code.size();
    w.line("// Program %u: %zu instructions, %u registers, level %u.", index, n_instr, p.num_regs, level);
    w.line("__attribute__((always_inline)) inline void sh_prog_%u(const ShTri& s) {", index);
    w.line("    using namespace ttr::shader;");

    // ---- Which values stay in local variables (level 2) ----------------------------------------------------------------
    // reaching[n][k]: the instruction whose result source k of instruction n reads (-1: an input or an unwritten register).
    // last_use[d]: the last instruction that reads the result of d. live_out[d]: d is the final value of an output register.
    std::vector<std::array<int32_t, 3>> reaching(n_instr, {-1, -1, -1});
    std::vector<int32_t> last_use(n_instr, -1), def_of(p.num_regs, -1);
    std::vector<bool> live_out(n_instr, false);
    // A product that is added to something in a later instruction must not stay in a local variable: the kernels are built with
    // -ffast-math, which lets the compiler fuse `x * y` and the following `+ z` into one multiply-add with a single rounding. The CPU
    // reference rounds the product first (that is what the instruction means), and the interpreter cannot fuse because the product goes
    // through Dest in between. Through Dest it stays here too.
    std::vector<bool> fusible(n_instr, false);
    for (size_t n = 0; n < n_instr; n++) {
        const uint32_t* in = p.code[n].w;
        const uint32_t op = instr_op(in[0]);
        for (uint32_t k = 0; k < op_sources(op); k++) {
            if (in[1 + k] & kSrcConst) continue;
            const uint32_t r = in[1 + k] & kSrcIndexMask;
            reaching[n][k] = r < def_of.size() ? def_of[r] : -1;
            if (reaching[n][k] >= 0) last_use[static_cast<size_t>(reaching[n][k])] = static_cast<int32_t>(n);
            if (reaching[n][k] >= 0 && (op == kOpAdd || (op == kOpMad && k == 2))) fusible[static_cast<size_t>(reaching[n][k])] = true;
        }
        if (op_writes(op) && instr_dst(in[0]) < def_of.size()) def_of[instr_dst(in[0])] = static_cast<int32_t>(n);
    }
    for (uint32_t c = 0; c < 4; c++) {
        if (p.out_reg + c < def_of.size() && def_of[p.out_reg + c] >= 0) live_out[static_cast<size_t>(def_of[p.out_reg + c])] = true;
    }
    std::vector<bool> local(n_instr, false);
    if (level >= 2) {
        // Registers free for values that are live across instruction n: what is left after its own needs.
        std::vector<int32_t> room(n_instr), across(n_instr, 0);
        for (size_t n = 0; n < n_instr; n++) {
            const uint32_t op = instr_op(p.code[n].w[0]);
            const uint32_t need = op_sources(op) + temporaries(op) + (op_writes(op) ? 1 : 0);
            room[n] = static_cast<int32_t>(kLocalRegisters) - static_cast<int32_t>(need);
        }
        std::vector<size_t> order;
        for (size_t n = 0; n < n_instr; n++) {
            const uint32_t op = instr_op(p.code[n].w[0]);
            if (op_writes(op) && last_use[n] > static_cast<int32_t>(n) && !(op == kOpMul && fusible[n])) order.push_back(n);
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return last_use[x] - static_cast<int32_t>(x) < last_use[y] - static_cast<int32_t>(y); });
        for (size_t d : order) {
            bool fits = true;
            for (int32_t n = static_cast<int32_t>(d) + 1; n < last_use[d] && fits; n++) fits = across[n] + 1 <= room[n];
            if (!fits) continue;
            local[d] = true;
            for (int32_t n = static_cast<int32_t>(d) + 1; n < last_use[d]; n++) across[n]++;
        }
    }

    // ---- Emit ---------------------------------------------------------------------------------------------------------
    for (size_t n = 0; n < n_instr; n++) {
        const uint32_t* in = p.code[n].w;
        const uint32_t op = instr_op(in[0]);
        const uint32_t sources = op_sources(op);
        w.line("    // %zu", n);
        const std::string dname = "d" + std::to_string(n);
        if (op_writes(op)) w.line("    sfpi::vFloat %s = 0.0f;", dname.c_str());
        w.line("    {");
        if (op_writes(op)) w.line("        sfpi::vFloat& d = %s;", dname.c_str());
        const char* names[3] = {"a", "b", "c"};
        for (uint32_t k = 0; k < sources; k++) {
            const int32_t from = reaching[n][k];
            // A result that stayed local is read from its variable, anything else from Dest. An instruction that reads the same
            // register as two operands is fine: the variable is copied.
            operand(w, names[k], in[1 + k], from >= 0 && local[static_cast<size_t>(from)] ? "d" + std::to_string(from) : std::string());
        }
        w.line("        %s", body(op));
        if (op_writes(op) && (instr_flags(in[0]) & kFlagSat)) w.line("        d = sh_sat(d);");
        w.line("    }");
        // Results that are read from Dest later, and the output registers, are stored; a value that only local readers use is not.
        if (op_writes(op) && (!local[n] || live_out[n])) w.line("    sfpi::dst_reg[kRegBase + %u] = %s;", instr_dst(in[0]), dname.c_str());
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
    for (uint32_t i : chosen) program(w, programs[i], i, options.level);
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
