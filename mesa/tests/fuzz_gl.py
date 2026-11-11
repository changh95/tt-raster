# SPDX-License-Identifier: Apache-2.0
"""Random fixed-function OpenGL frames through Mesa: stock softpipe against the Tensix fast path (mesa/README.md).

    python3 mesa/tests/fuzz_gl.py [--count N] [--seed S] [--backend reference|tensix] [--node 5] [--size 96] [--frames 2]
                                  [--jobs 8] [--no-refused] [--keep DIR] [--verbose]

tests/gl_fuzz.c draws random frames (depth function and writes, blending, shading model, culling, winding, strips / fans / quads /
polygons, clipped and crossing geometry, orthographic and perspective projections, and now and then state the fast path refuses,
which makes softpipe take over in the middle of the frame); each is drawn by stock softpipe and by the fast path and the colour
and depth images are compared. Default backend: the CPU reference (`TTR_GL=reference`, parallel jobs); with --backend tensix
the frames run on a chip, one at a time.

What is compared: the share of pixels more than 4 levels apart in colour (default limit 4 % per run, 0.5 % for the mean over the runs),
and the share of pixels whose 16-bit
depth differs by more than 4 units of 1/255 (limit 1 %): the two rasterizers differ in edge rounding and in the depth
interpolation, which moves pixels along triangle edges and, where two surfaces meet at a shallow angle, over whole bands (z-fighting:
the fast path and softpipe then pick different surfaces, each one is fine). A wrong depth function, blend factor, cull or winding
rule, flat-shading vertex or primitive decomposition changes much more than that: replacing the Premultiplied blend by Add in
the driver gives a mean difference of 3.8 % and 71 of 150 runs over the limit, taking the flat colour from the first vertex of
each triangle gives 7.7 % and 113 of 150 (against 0.15 % and none for the driver as it is, --tame). --tame keeps the triangles
moderate and the projection orthographic, which softpipe itself renders more accurately than huge perspective triangles.
Exit status 0 if no run exceeded the limits and the mean is within its limit.
"""
import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def read_pnm(path):
    with open(path, "rb") as f:
        magic = f.readline().strip()
        w, h = map(int, f.readline().split())
        f.readline()
        if magic == b"P6":
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w, 3).astype(int)
        g = np.frombuffer(f.read(), dtype=">u2").reshape(h, w).astype(float)
        return (g / 257.0).round().astype(int)[:, :, None]


def run(args, exe, seed, backend, prefix):
    install = os.path.expanduser(args.prefix)
    lib = os.path.join(install, "lib", "x86_64-linux-gnu")
    env = dict(os.environ, LD_LIBRARY_PATH=lib, LIBGL_DRIVERS_PATH=os.path.join(lib, "dri"), LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="softpipe",
               TTRASTER_LIB=os.path.join(ROOT, "build", "src", "libttraster.so"), TTR_GL_DEBUG="1", TTR_GL_MIN_PIXELS="0")
    env.pop("TTR_GL", None)
    cmd = [exe, "--seed", str(seed), "--size", str(args.size), "--frames", str(args.frames), "--dump", prefix] + (["--no-refused"] if args.no_refused else []) + (["--tame"] if args.tame else [])
    if backend == "tensix":
        cmd = [os.path.join(ROOT, "scripts", "run_gl.sh"), args.node, "--size", f"{args.size}x{args.size}", "--backend", "tensix"] + cmd
    else:
        if backend == "reference":
            env["TTR_GL"] = "reference"
        cmd = ["xvfb-run", "-a", "-s", f"-screen 0 {args.size}x{args.size}x24"] + cmd
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    return p.returncode, p.stdout + p.stderr


def one(args, exe, work, seed):
    rc0, log0 = run(args, exe, seed, "off", os.path.join(work, f"s{seed}_stock"))
    rc1, log1 = run(args, exe, seed, args.backend, os.path.join(work, f"s{seed}_fast"))
    res = dict(seed=seed, ok=True)
    if rc0 != 0 or rc1 != 0:
        res.update(ok=False, kind="crash", note=((log0 if rc0 else log1).strip().splitlines() or ["?"])[-1])
        return res
    m = re.search(r"(\d+) frames on the fast path, (\d+) draws accepted, (\d+) draws refused, (\d+) mid-frame fallbacks", log1)
    accepted, refused, fallbacks = (int(m.group(2)), int(m.group(3)), int(m.group(4))) if m else (0, 0, 0)
    bad4 = bad_depth = 0.0
    for f in range(args.frames):
        a, b = read_pnm(os.path.join(work, f"s{seed}_stock{f}.ppm")), read_pnm(os.path.join(work, f"s{seed}_fast{f}.ppm"))
        d = np.abs(a - b).max(axis=2)
        bad4 = max(bad4, float(np.count_nonzero(d > 4)) / d.size)
        za, zb = read_pnm(os.path.join(work, f"s{seed}_stock{f}_depth.pgm")), read_pnm(os.path.join(work, f"s{seed}_fast{f}_depth.pgm"))
        bad_depth = max(bad_depth, float(np.count_nonzero(np.abs(za - zb) > 4)) / za.size)
    res.update(kind="compared", accepted=accepted, refused=refused, fallbacks=fallbacks, bad4=bad4, bad_depth=bad_depth,
               ok=bad4 <= args.max_bad4 and bad_depth <= args.max_bad_depth)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backend", default="reference", choices=["reference", "tensix"])
    ap.add_argument("--node", default="5")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-refused", action="store_true")
    ap.add_argument("--tame", action="store_true", help="moderate triangles, orthographic only")
    ap.add_argument("--keep", default=None)
    ap.add_argument("--prefix", default=os.environ.get("TT_RASTER_MESA_PREFIX", "~/experiments/mesa-install"))
    ap.add_argument("--max-bad4", type=float, default=0.04)
    ap.add_argument("--max-mean4", type=float, default=0.005, help="limit on the mean over all runs of the share of pixels more than 4 levels off")
    ap.add_argument("--max-bad-depth", type=float, default=0.01)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    work = args.keep or tempfile.mkdtemp(prefix="fuzz_gl_")
    os.makedirs(work, exist_ok=True)
    exe = os.path.join(work, "gl_fuzz")
    subprocess.run(["gcc", "-O1", "-o", exe, os.path.join(ROOT, "mesa", "tests", "gl_fuzz.c"), "-lGL", "-lX11", "-lm"], check=True)
    jobs = 1 if args.backend == "tensix" else args.jobs
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for res in pool.map(lambda s: one(args, exe, work, s), range(args.seed, args.seed + args.count)):
            results.append(res)
            if args.verbose or not res["ok"]:
                if res["kind"] == "compared":
                    print(f"seed {res['seed']}: {'ok' if res['ok'] else 'FAILED'} bad4 {res['bad4']:.3%} depth {res['bad_depth']:.3%} "
                          f"(draws on the fast path {res['accepted']}, refused {res['refused']}, mid-frame fallbacks {res['fallbacks']})", flush=True)
                else:
                    print(f"seed {res['seed']}: {res['kind']} {res['note']}", flush=True)
    cmp_ = [r for r in results if r["kind"] == "compared"]
    if cmp_:
        print(f"{len(results)} runs of {args.frames} frames: {sum(r['accepted'] for r in cmp_)} draws on the fast path, {sum(r['refused'] for r in cmp_)} refused, "
              f"{sum(1 for r in cmp_ if r['fallbacks'])} runs with a mid-frame fallback; worst colour {max(r['bad4'] for r in cmp_):.3%}, "
              f"worst depth {max(r['bad_depth'] for r in cmp_):.3%}, mean colour {np.mean([r['bad4'] for r in cmp_]):.3%}")
    failed = [r for r in results if not r["ok"]]
    if cmp_ and np.mean([r["bad4"] for r in cmp_]) > args.max_mean4:
        print(f"the mean colour difference {np.mean([r['bad4'] for r in cmp_]):.3%} exceeds {args.max_mean4:.3%}: a systematic difference, not z-fighting")
        failed.append(dict(seed="mean"))
    if failed:
        print(f"{len(failed)} failed: seeds {[r['seed'] for r in failed]}; images in {work}")
        return 1
    print("OK" + ("" if args.keep else f" (work files in {work})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
