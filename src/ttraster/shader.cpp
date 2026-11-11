// SPDX-License-Identifier: Apache-2.0

#include "ttraster/shader.h"

#include <cmath>
#include <cstring>

namespace ttr {

using namespace shader;

bool ShaderProgram::uses_kill() const {
    for (const ShaderInstr& i : code) {
        if (instr_op(i.w[0]) == kOpKill) return true;
    }
    return false;
}

std::string validate_shader(const ShaderProgram& p) {
    if (p.code.empty() || p.code.size() > kMaxInstructions) return "program length out of range";
    if (p.consts.size() > kMaxConsts) return "too many constants";
    if (p.num_inputs > kMaxInputs) return "too many inputs";
    if (p.num_regs < p.num_inputs || p.num_regs > kMaxRegs) return "register count out of range";
    if (p.out_reg + 4 > p.num_regs) return "output registers out of range";
    std::vector<uint8_t> defined(p.num_regs, 0);
    for (uint32_t i = 0; i < p.num_inputs; i++) defined[i] = 1;
    if (p.fragcoord_reg >= 0) {
        if (static_cast<uint32_t>(p.fragcoord_reg) < p.num_inputs || static_cast<uint32_t>(p.fragcoord_reg) + 4 > p.num_regs)
            return "fragcoord registers out of range";
        for (int k = 0; k < 4; k++) defined[p.fragcoord_reg + k] = 1;
    }
    for (size_t n = 0; n < p.code.size(); n++) {
        const ShaderInstr& in = p.code[n];
        const std::string where = "instruction " + std::to_string(n) + ": ";
        const uint32_t op = instr_op(in.w[0]);
        if (op == kOpNop || op >= kOpCount) return where + "bad opcode";
        if (instr_flags(in.w[0]) & ~kFlagSat) return where + "unknown flag";
        const uint32_t ns = op_sources(op);
        for (uint32_t s = 0; s < 3; s++) {
            const uint32_t w = in.w[1 + s];
            if (s >= ns) {
                if (w != 0) return where + "unused operand is not zero";
                continue;
            }
            if (w & ~(kSrcIndexMask | kSrcConst | kSrcNeg | kSrcAbs)) return where + "reserved operand bits";
            const uint32_t idx = w & kSrcIndexMask;
            if (w & kSrcConst) {
                if (idx >= p.consts.size()) return where + "constant index out of range";
            } else {
                if (idx >= p.num_regs) return where + "register index out of range";
                if (!defined[idx]) return where + "register r" + std::to_string(idx) + " is read before it is written";
            }
        }
        if (op_writes(op)) {
            const uint32_t d = instr_dst(in.w[0]);
            if (d >= p.num_regs) return where + "destination out of range";
            defined[d] = 1;
        } else if (instr_dst(in.w[0]) != 0) {
            return where + "instruction without a destination has one";
        }
    }
    for (uint32_t k = 0; k < 4; k++) {
        if (!defined[p.out_reg + k]) return "output register r" + std::to_string(p.out_reg + k) + " is never written";
    }
    return {};
}

namespace {

inline float apply_sat(float v) { return v > 0.0f ? (v < 1.0f ? v : 1.0f) : 0.0f; }   // NaN -> 0

}  // namespace

ShaderResult run_shader(const ShaderProgram& p, const float* inputs, const float* fragcoord, float* r) {
    for (uint32_t k = 0; k < p.num_inputs; k++) r[k] = inputs[k];
    if (p.fragcoord_reg >= 0) {
        for (int k = 0; k < 4; k++) r[p.fragcoord_reg + k] = fragcoord[k];
    }
    bool killed = false;
    for (const ShaderInstr& in : p.code) {
        const uint32_t op = instr_op(in.w[0]);
        float v[3] = {0.0f, 0.0f, 0.0f};
        const uint32_t ns = op_sources(op);
        for (uint32_t s = 0; s < ns; s++) {
            const uint32_t w = in.w[1 + s];
            float x = (w & kSrcConst) ? p.consts[w & kSrcIndexMask] : r[w & kSrcIndexMask];
            if (w & kSrcAbs) x = std::fabs(x);
            if (w & kSrcNeg) x = -x;
            v[s] = x;
        }
        const float a = v[0], b = v[1], c = v[2];
        float d = 0.0f;
        switch (op) {
            case kOpMov: d = a; break;
            case kOpAdd: d = a + b; break;
            case kOpMul: d = a * b; break;
            case kOpMad: d = std::fmaf(a, b, c); break;
            case kOpMin: d = b < a ? b : a; break;
            case kOpMax: d = b > a ? b : a; break;
            case kOpFrc: d = a - std::floor(a); break;
            case kOpFlr: d = std::floor(a); break;
            case kOpRcp: d = 1.0f / a; break;
            case kOpRsq: d = 1.0f / std::sqrt(std::fabs(a)); break;
            case kOpSqrt: d = std::sqrt(std::fabs(a)); break;
            case kOpEx2: d = std::exp2(a); break;
            case kOpLg2: d = std::log2(std::fabs(a)); break;
            case kOpSin: d = std::sin(a); break;
            case kOpCos: d = std::cos(a); break;
            case kOpSlt: d = a < b ? 1.0f : 0.0f; break;
            case kOpSge: d = a >= b ? 1.0f : 0.0f; break;
            case kOpSeq: d = a == b ? 1.0f : 0.0f; break;
            case kOpSne: d = a != b ? 1.0f : 0.0f; break;
            case kOpCmp: d = a < 0.0f ? b : c; break;
            case kOpSel: d = a != 0.0f ? b : c; break;
            case kOpKill: killed |= a < 0.0f; continue;
            default: break;
        }
        if (instr_flags(in.w[0]) & kFlagSat) d = apply_sat(d);
        r[instr_dst(in.w[0])] = d;
    }
    ShaderResult out;
    for (int k = 0; k < 4; k++) out.rgba[k] = r[p.out_reg + k];
    out.killed = killed;
    return out;
}

ShaderBuilder::Src ShaderBuilder::constant(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    for (size_t i = 0; i < p_.consts.size(); i++) {
        uint32_t b;
        std::memcpy(&b, &p_.consts[i], 4);
        if (b == bits) return {static_cast<uint32_t>(i) | kSrcConst};
    }
    p_.consts.push_back(v);
    return {static_cast<uint32_t>(p_.consts.size() - 1) | kSrcConst};
}

uint32_t ShaderBuilder::emit_to(uint32_t op, uint32_t dst, Src a, Src b, Src c, bool sat) {
    ShaderInstr in;
    in.w[0] = make_word0(op, sat ? kFlagSat : 0u, dst);
    in.w[1] = a.bits;
    in.w[2] = b.bits;
    in.w[3] = c.bits;
    p_.code.push_back(in);
    return dst;
}

void ShaderBuilder::kill(Src a) {
    ShaderInstr in;
    in.w[0] = make_word0(kOpKill, 0, 0);
    in.w[1] = a.bits;
    p_.code.push_back(in);
}

ShaderProgram ShaderBuilder::finish(uint32_t out_reg) {
    p_.out_reg = out_reg;
    return p_;
}

}  // namespace ttr
