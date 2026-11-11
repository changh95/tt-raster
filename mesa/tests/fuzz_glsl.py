# SPDX-License-Identifier: Apache-2.0
"""Random GLSL fragment shaders through Mesa: stock softpipe against the Tensix fast path (mesa/README.md).

    python3 mesa/tests/fuzz_glsl.py [--count N] [--seed S] [--backend reference|tensix] [--node 5] [--size 96]
                                    [--jobs 8] [--keep DIR] [--verbose]

Each shader is generated from the seed (typed expression trees over the varyings, uniforms and gl_FragCoord, locals of
float / vec2 / vec3 / vec4, swizzles and write masks, if / else, constant-trip loops, discard), drawn on a sphere by
tests/gl_scene.c once with stock softpipe, which interprets the TGSI itself, and once with the fast path, whose translator
(softpipe_tensix/sp_tensix_shader.c) turns the same TGSI into the shader ISA. The default backend is the CPU reference
(`TTR_GL=reference`, no chip needed, jobs run in parallel): the translator is what is under test here; the chip against the CPU
is tested by tools/ttr_shader_fuzz. With --backend tensix the shaders run on a chip (one at a time).

A shader the translator refuses is drawn by softpipe and tells nothing (counted, with the share of shaders that really took
the fast path). For the others the images are compared: the share of pixels that differ by more than 4 and by more than 16
levels. Rounding differences are small, but fract / floor / step / compares / discard turn them into flipped pixels now and then,
so the thresholds are per-image shares (default 2 % above 4 levels, 1 % above 16; over 600 shaders the worst were 1.0 % and 0.75 %): a wrong swizzle, mask, register or
conditional shows up as much more.

Exit status 0 if no shader exceeded the thresholds. A failing shader is kept (with both images) in --keep DIR (default: a
temporary directory that is printed).
"""
import argparse
import concurrent.futures
import os
import random
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---- generator ------------------------------------------------------------------------------------------------------------

class Gen:
    """Typed random expressions. Every generated shader is valid GLSL 1.20; values are kept in sane ranges by clamping and by
    adding offsets before sqrt / log / inversesqrt / pow, so that the comparison against softpipe is about the translation, not
    about NaNs."""

    FLOAT_INPUTS = ["c.r", "c.g", "c.b", "c.a", "n.x", "n.y", "n.z", "p.x", "p.y", "p.z", "tint.x", "tint.y", "tint.w", "t",
                    "gl_FragCoord.x * 0.01", "gl_FragCoord.y * 0.01", "gl_FragCoord.z", "gl_FragCoord.w"]
    COMP = "xyzw"

    def __init__(self, rng):
        self.r = rng
        self.fvars, self.v2vars, self.v3vars, self.v4vars = [], [], [], []
        self.lines = []
        self.indent = 1
        self.counter = 0
        self.loop_vars = []

    def lit(self):
        return f"{self.r.choice([-2, -1, -0.5, 0, 0.25, 0.5, 1, 2, 3]) + self.r.choice([0, 0, 0.125, 0.37]):.3f}"

    def fresh(self, kind):
        self.counter += 1
        return f"{kind}{self.counter}"

    # float
    def f(self, d):
        r = self.r
        if d <= 0 or r.random() < 0.2:
            u = r.random()
            if u < 0.25:
                return self.lit()
            if u < 0.55 and self.fvars:
                return r.choice(self.fvars)
            if u < 0.6 and self.loop_vars:
                return f"float({r.choice(self.loop_vars)})"
            if u < 0.75:
                return self.vec_comp(self.v3vars + self.v2vars + self.v4vars)
            return r.choice(self.FLOAT_INPUTS)
        k = r.randrange(26)
        a = lambda: self.f(d - 1)
        if k == 0: return f"({a()} + {a()})"
        if k == 1: return f"({a()} - {a()})"
        if k in (2, 3): return f"({a()} * {a()})"
        if k == 4: return f"min({a()}, {a()})"
        if k == 5: return f"max({a()}, {a()})"
        if k == 6: return f"abs({a()})"
        if k == 7: return f"(-({a()}))"
        if k == 8: return f"floor({a()} * 2.0)"
        if k == 9: return f"fract({a()} * 1.5)"
        if k == 10: return f"sqrt(abs({a()}))"
        if k == 11: return f"sin({a()} * 3.0)"
        if k == 12: return f"cos({a()} * 2.0)"
        if k == 13: return f"mix({a()}, {a()}, clamp({a()}, 0.0, 1.0))"
        if k == 14: return f"clamp({a()}, -1.0, 1.0)"
        if k == 15: return f"smoothstep(0.0, 1.0, {a()})"
        if k == 16: return f"dot({self.v3(d - 1)}, {self.v3(d - 1)})"
        if k == 17: return f"length({self.v3(d - 1)})"
        if k == 18: return f"(({self.cond(d - 1)}) ? {a()} : {a()})"
        if k == 19: return f"step({a()}, {a()})"
        if k == 20: return f"mod({a()}, abs({a()}) + 0.5)"
        if k == 21: return f"exp2(clamp({a()}, -4.0, 4.0))"
        if k == 22: return f"log2(abs({a()}) + 0.25)"
        if k == 23: return f"inversesqrt(abs({a()}) + 0.25)"
        if k == 24: return f"pow(abs({a()}) + 0.25, clamp({a()}, 0.5, 2.0))"
        return f"sign({a()})"

    def vec_comp(self, names):
        if not names:
            return self.r.choice(self.FLOAT_INPUTS)
        name = self.r.choice(names)
        n = 2 if name in self.v2vars else 3 if name in self.v3vars else 4
        return f"{name}.{self.r.choice(self.COMP[:n])}"

    def swz(self, n):
        return "".join(self.r.choice(self.COMP[:n]) for _ in range(n))

    # vec2 / vec3 / vec4
    def v2(self, d):
        r = self.r
        if d <= 0 or r.random() < 0.2:
            if self.v2vars and r.random() < 0.5:
                return r.choice(self.v2vars)
            return r.choice(["p.xy", "n.xy", "c.rg", "tint.zw", "gl_FragCoord.xy * 0.01", "c.ba"])
        k = r.randrange(7)
        if k == 0: return f"vec2({self.f(d - 1)}, {self.f(d - 1)})"
        if k == 1: return f"({self.v2(d - 1)} + {self.v2(d - 1)})"
        if k == 2: return f"({self.v2(d - 1)} * {self.v2(d - 1)})"
        if k == 3: return f"({self.v2(d - 1)} * {self.f(d - 1)})"
        if k == 4: return f"{self.v3(d - 1)}.{self.swz(2)}"
        if k == 5: return f"abs({self.v2(d - 1)})"
        return f"mix({self.v2(d - 1)}, {self.v2(d - 1)}, clamp({self.f(d - 1)}, 0.0, 1.0))"

    def v3(self, d):
        r = self.r
        if d <= 0 or r.random() < 0.2:
            if self.v3vars and r.random() < 0.5:
                return r.choice(self.v3vars)
            return r.choice(["c.rgb", "n", "p", "tint.xyz", "vec3(gl_FragCoord.x * 0.01, gl_FragCoord.y * 0.01, 0.5)", "c.bgr", "n.zxy"])
        k = r.randrange(12)
        if k == 0: return f"vec3({self.f(d - 1)}, {self.f(d - 1)}, {self.f(d - 1)})"
        if k == 1: return f"vec3({self.f(d - 1)})"
        if k == 2: return f"({self.v3(d - 1)} + {self.v3(d - 1)})"
        if k == 3: return f"({self.v3(d - 1)} - {self.v3(d - 1)})"
        if k == 4: return f"({self.v3(d - 1)} * {self.v3(d - 1)})"
        if k == 5: return f"({self.v3(d - 1)} * {self.f(d - 1)})"
        if k == 6: return f"normalize({self.v3(d - 1)} + vec3(0.1, 0.2, 0.3))"
        if k == 7: return f"cross({self.v3(d - 1)}, {self.v3(d - 1)})"
        if k == 8: return f"clamp({self.v3(d - 1)}, 0.0, 1.0)"
        if k == 9: return f"{self.v4(d - 1)}.{self.swz(3)}"
        if k == 10: return f"vec3({self.v2(d - 1)}, {self.f(d - 1)})"
        return f"mix({self.v3(d - 1)}, {self.v3(d - 1)}, clamp({self.f(d - 1)}, 0.0, 1.0))"

    def v4(self, d):
        r = self.r
        if d <= 0 or r.random() < 0.2:
            if self.v4vars and r.random() < 0.5:
                return r.choice(self.v4vars)
            return r.choice(["c", "tint", "vec4(n, 1.0)", "vec4(p, 0.5)", "c.abgr"])
        k = r.randrange(7)
        if k == 0: return f"vec4({self.f(d - 1)}, {self.f(d - 1)}, {self.f(d - 1)}, {self.f(d - 1)})"
        if k == 1: return f"vec4({self.v3(d - 1)}, {self.f(d - 1)})"
        if k == 2: return f"({self.v4(d - 1)} + {self.v4(d - 1)})"
        if k == 3: return f"({self.v4(d - 1)} * {self.v4(d - 1)})"
        if k == 4: return f"({self.v4(d - 1)} * {self.f(d - 1)})"
        if k == 5: return f"vec4({self.v2(d - 1)}, {self.v2(d - 1)})"
        return f"clamp({self.v4(d - 1)}, 0.0, 1.0)"

    def cond(self, d):
        op = self.r.choice(["<", ">", "<=", ">="])
        c = f"{self.f(d)} {op} {self.f(d)}"
        if self.r.random() < 0.1:   # && and || are integer operations in TGSI: the translator refuses them
            c = f"{c} {self.r.choice(['&&', '||'])} {self.f(d)} {op} {self.f(d)}"
        return c

    # statements
    def scope(self):
        return (len(self.fvars), len(self.v2vars), len(self.v3vars), len(self.v4vars))

    def leave(self, saved):
        """A block ends: the variables declared inside go out of scope."""
        del self.fvars[saved[0]:], self.v2vars[saved[1]:], self.v3vars[saved[2]:], self.v4vars[saved[3]:]

    def emit(self, text):
        self.lines.append("  " * self.indent + text)

    def declare(self):
        kind = self.r.choice(["float", "float", "vec2", "vec3", "vec3", "vec4"])
        name = self.fresh({"float": "f", "vec2": "u", "vec3": "v", "vec4": "q"}[kind])
        expr = {"float": self.f, "vec2": self.v2, "vec3": self.v3, "vec4": self.v4}[kind](3)
        self.emit(f"{kind} {name} = {expr};")
        {"float": self.fvars, "vec2": self.v2vars, "vec3": self.v3vars, "vec4": self.v4vars}[kind].append(name)

    def assign(self):
        r = self.r
        pools = [(self.fvars, "float"), (self.v2vars, "vec2"), (self.v3vars, "vec3"), (self.v4vars, "vec4")]
        pools = [p for p in pools if p[0]]
        if not pools:
            return
        names, kind = r.choice(pools)
        name = r.choice(names)
        n = {"float": 1, "vec2": 2, "vec3": 3, "vec4": 4}[kind]
        gen = {"float": self.f, "vec2": self.v2, "vec3": self.v3, "vec4": self.v4}[kind]
        op = r.choice(["=", "=", "=", "+=", "*=", "-="])
        if n > 1 and r.random() < 0.5:
            m = r.randrange(1, n)   # write mask of m distinct components, in increasing order
            idx = sorted(r.sample(range(n), m))
            mask = "".join(self.COMP[i] for i in idx)
            rhs = self.f(2) if m == 1 else {2: self.v2, 3: self.v3}[m](2)
            self.emit(f"{name}.{mask} {op} {rhs};")
        else:
            self.emit(f"{name} {op} {gen(2)};")

    def statement(self, depth):
        r = self.r
        u = r.random()
        if u < 0.25:
            self.declare()
        elif u < 0.6:
            self.assign()
        elif u < 0.8 and depth < 2:
            self.emit(f"if ({self.cond(2)}) {{")
            self.indent += 1
            saved = self.scope()
            for _ in range(r.randrange(1, 4)):
                self.statement(depth + 1)
            self.leave(saved)
            self.indent -= 1
            if r.random() < 0.6:
                self.emit("} else {")
                self.indent += 1
                for _ in range(r.randrange(1, 4)):
                    self.statement(depth + 1)
                self.leave(saved)
                self.indent -= 1
            self.emit("}")
        elif u < 0.9 and depth < 2:
            i = self.fresh("i")
            self.emit(f"for (int {i} = 0; {i} < {r.randrange(2, 5)}; {i}++) {{")
            self.indent += 1
            self.loop_vars.append(i)
            saved = self.scope()
            for _ in range(r.randrange(1, 3)):
                self.statement(depth + 1)
            self.leave(saved)
            self.loop_vars.pop()
            self.indent -= 1
            self.emit("}")
        elif u < 0.95:
            self.emit(f"if ({self.cond(2)}) discard;")
        else:
            self.assign()

    def shader(self):
        for _ in range(self.r.randrange(1, 4)):
            self.declare()
        for _ in range(self.r.randrange(2, 10)):
            self.statement(0)
        rgb = self.v3(3) if self.r.random() < 0.7 else (self.r.choice(self.v3vars) if self.v3vars else "c.rgb")
        a = "1.0" if self.r.random() < 0.6 else f"clamp({self.f(2)}, 0.0, 1.0)"
        self.emit(f"gl_FragColor = vec4(clamp({rgb}, 0.0, 1.0), {a});")
        head = "#version 120\nvarying vec4 c;\nvarying vec3 n;\nvarying vec3 p;\nuniform vec4 tint;\nuniform float t;\nvoid main() {\n"
        return head + "\n".join(self.lines) + "\n}\n"


# ---- driver ---------------------------------------------------------------------------------------------------------------

def read_ppm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P6"
        w, h = map(int, f.readline().split())
        f.readline()
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w, 3)


def run_gl(args, exe, shader_path, backend, out_prefix):
    prefix = os.path.expanduser(args.prefix)
    lib = os.path.join(prefix, "lib", "x86_64-linux-gnu")
    env = dict(os.environ, LD_LIBRARY_PATH=lib, LIBGL_DRIVERS_PATH=os.path.join(lib, "dri"), LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="softpipe",
               TTRASTER_LIB=os.path.join(ROOT, "build", "src", "libttraster.so"), TTR_GL_DEBUG="1", TTR_GL_MIN_PIXELS="0")
    env.pop("TTR_GL", None)
    cmd = [exe, "--fs-file", shader_path, "--sphere", "--size", str(args.size), "--frames", "1", "--dump", out_prefix]
    if backend == "tensix":
        cmd = [os.path.join(ROOT, "scripts", "run_gl.sh"), args.node, "--size", f"{args.size}x{args.size}", "--backend", "tensix"] + cmd
    else:
        if backend == "reference":
            env["TTR_GL"] = "reference"
        cmd = ["xvfb-run", "-a", "-s", f"-screen 0 {args.size}x{args.size}x24"] + cmd
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    return p.returncode, p.stdout + p.stderr


def one(args, exe, work, seed):
    src = Gen(random.Random(seed)).shader()
    path = os.path.join(work, f"s{seed}.frag")
    with open(path, "w") as f:
        f.write(src)
    rc0, log0 = run_gl(args, exe, path, "off", os.path.join(work, f"s{seed}_stock"))
    rc1, log1 = run_gl(args, exe, path, args.backend, os.path.join(work, f"s{seed}_fast"))
    res = dict(seed=seed, src=src, ok=True, note="")
    if rc0 != 0 or "GLSL link failed" in log0:
        errors = [l for l in log0.splitlines() if "error:" in l and "link" not in l]
        res.update(kind="invalid", note=errors[0] if errors else "stock softpipe failed")
        return res
    if rc1 != 0:
        res.update(kind="crash", ok=False, note=(log1.strip().splitlines() or ["?"])[-1])
        return res
    m = re.search(r"(\d+) draws accepted, (\d+) draws refused", log1)
    accepted, refused = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    if refused or not accepted:
        why = re.search(r"fragment shader not translated: ([^\n]*)", log1)
        res.update(kind="refused", note=why.group(1) if why else "")
        return res
    a, b = read_ppm(os.path.join(work, f"s{seed}_stock0.ppm")), read_ppm(os.path.join(work, f"s{seed}_fast0.ppm"))
    d = np.abs(a.astype(int) - b.astype(int)).max(axis=2)
    bad4, bad16 = float(np.count_nonzero(d > 4)) / d.size, float(np.count_nonzero(d > 16)) / d.size
    nonblank = float(np.count_nonzero(np.abs(a.astype(int) - a[0, 0].astype(int)).max(axis=2) > 8)) / d.size   # how much the image shows
    res.update(kind="fast", bad4=bad4, bad16=bad16, max=int(d.max()), shown=nonblank)
    res["ok"] = bad4 <= args.max_bad4 and bad16 <= args.max_bad16
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backend", default="reference", choices=["reference", "tensix"])
    ap.add_argument("--node", default="5")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--keep", default=None)
    ap.add_argument("--prefix", default=os.environ.get("TT_RASTER_MESA_PREFIX", "~/experiments/mesa-install"))
    ap.add_argument("--max-bad4", type=float, default=0.02)
    ap.add_argument("--max-bad16", type=float, default=0.01)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    work = args.keep or tempfile.mkdtemp(prefix="fuzz_glsl_")
    os.makedirs(work, exist_ok=True)
    exe = os.path.join(work, "gl_scene")
    subprocess.run(["gcc", "-O1", "-o", exe, os.path.join(ROOT, "mesa", "tests", "gl_scene.c"), "-lGL", "-lX11", "-lm"], check=True)
    seeds = list(range(args.seed, args.seed + args.count))
    jobs = 1 if args.backend == "tensix" else args.jobs
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for res in pool.map(lambda s: one(args, exe, work, s), seeds):
            results.append(res)
            if args.verbose or not res["ok"]:
                extra = f"bad4 {res.get('bad4', 0):.3%} bad16 {res.get('bad16', 0):.3%} max {res.get('max', 0)}" if res["kind"] == "fast" else res["note"]
                print(f"seed {res['seed']}: {res['kind']} {'ok' if res['ok'] else 'FAILED'} {extra}", flush=True)
    kinds = {}
    for r in results:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    fast = [r for r in results if r["kind"] == "fast"]
    print(f"{len(results)} shaders: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    if fast:
        shown = sum(1 for r in fast if r["shown"] > 0.02)
        print(f"fast path: {len(fast)}; {shown} show something other than the background; "
              f"worst bad4 {max(r['bad4'] for r in fast):.3%}, worst bad16 {max(r['bad16'] for r in fast):.3%}, "
              f"mean bad4 {np.mean([r['bad4'] for r in fast]):.3%}; exact {sum(1 for r in fast if r['max'] == 0)}")
    refused = [r for r in results if r["kind"] == "refused"]
    if refused:
        reasons = {}
        for r in refused:
            reasons[r["note"]] = reasons.get(r["note"], 0) + 1
        print("refused: " + "; ".join(f"{v} x {k or '?'}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])))
        if args.verbose:   # which constructs go with a refusal
            for feat in ["&&", "||", "? ", "sign(", "mod(", "floor(", "step(", "smoothstep(", "pow(", "discard", "for (", "if (", "cross(", "normalize(", "length(", "log2(", "exp2(", "inversesqrt("]:
                have = [r for r in results if r["kind"] in ("fast", "refused") and feat in r["src"]]
                if have:
                    print(f"  {feat:14s} in {len(have):3d} shaders, refused {sum(1 for r in have if r['kind'] == 'refused'):3d}")
    failed = [r for r in results if not r["ok"]]
    for r in failed:
        with open(os.path.join(work, f"FAILED_{r['seed']}.frag"), "w") as f:
            f.write(r["src"])
    if failed:
        print(f"{len(failed)} failed; sources and images in {work}")
        return 1
    print("OK" + ("" if args.keep else f" (work files in {work})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
