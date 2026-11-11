#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# OpenGL through Mesa with the Tensix fast path (mesa/README.md): a deterministic fixed-function scene is
# rendered by stock softpipe, by the fast path on the CPU reference backend, and (with TT_METAL_HOME set) on a
# Tensix chip; the images are compared. Needs scripts/build_mesa.sh done, the host build (libttraster.so), a C
# compiler with X11/GL headers, xvfb, and a Python with numpy (TT_RASTER_PYTHON, default ~/experiments/venv-newton).
#
#   scripts/test_gl.sh                              stock softpipe against the fast path on the CPU backend
#   TT_METAL_HOME=... TT_RASTER_NODE=5 scripts/test_gl.sh   also the Tensix backend
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="${TT_RASTER_MESA_PREFIX:-$HOME/experiments/mesa-install}"
build="${TT_RASTER_BUILD_DIR:-$root/build}"
py="${TT_RASTER_PYTHON:-$HOME/experiments/venv-newton/bin/python}"
node="${TT_RASTER_NODE:-5}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

gcc -O1 -o "$work/gl_scene" "$root/mesa/tests/gl_scene.c" -lGL -lX11 -lm
lib="$prefix/lib/x86_64-linux-gnu"

# gl_run <backend: off|reference|tensix> <output prefix> <gl_scene args...>
gl_run() {
    local backend="$1" out="$2"
    shift 2
    if [[ "$backend" == "tensix" ]]; then
        "$root/scripts/run_gl.sh" "$node" --size 400x400 --backend tensix "$work/gl_scene" "$@" --dump "$out" >"$out.log" 2>&1
    else
        env LD_LIBRARY_PATH="$lib" LIBGL_DRIVERS_PATH="$lib/dri" LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=softpipe \
            TTRASTER_LIB="$build/src/libttraster.so" TTR_GL_DEBUG=1 ${backend:+$([[ $backend == reference ]] && echo TTR_GL=reference)} \
            xvfb-run -a -s "-screen 0 400x400x24" "$work/gl_scene" "$@" --dump "$out" >"$out.log" 2>&1
    fi
}

fail=0
compare() {  # compare <a> <b> <max-bad>
    if ! "$py" "$root/mesa/tests/compare_ppm.py" "$1" "$2" --max-bad "$3"; then
        fail=1
    fi
}

# name | gl_scene arguments | frames that must run on the fast path without a fallback (1) or may fall back (0)
cases=(
    "ortho|--ortho --depth --continue --frames 3|1"
    "perspective|--depth --continue --frames 3|1"
    "lighting|--lighting --frames 2|1"
    "cull|--cull --frames 2|1"
    "swap|--swap --frames 3|1"
    "glsl|--glsl --depth --continue --frames 3|1"
    "glsl_alu|--glsl-alu --frames 2|1"
    "fs_lambert|--fs lambert --depth --frames 2|1"
    "fs_phong|--fs phong --depth --frames 2|1"
    "fs_stripes|--fs stripes --depth --frames 2|1"
    "fs_discard|--fs discard --depth --frames 2|1"
    "fs_cond|--fs cond --depth --frames 2|1"
    "fs_loop|--fs loop --depth --frames 2|1"
    "fs_uniform|--fs uniform --depth --frames 3|1"
    "fs_smooth|--fs smooth --depth --frames 2|1"
    "fs_mask|--fs mask --depth --frames 2|1"
    "fs_fog|--fs fog --depth --frames 2|1"
    "fs_blendalpha|--fs blendalpha --depth --frames 2|1"
    "fs_dynloop_fallback|--fs dynloop --frames 2|0"
    "blend|--blend --depth --frames 2|1"
    "depthmask|--depthmask --depth --continue --frames 2|1"
    "nodepth|--nodepth --depth --frames 2|1"
    "blend_separate_fallback|--blend-sep --frames 2|0"
)
for c in "${cases[@]}"; do
    IFS='|' read -r name args strict <<<"$c"
    echo "== $name ($args)"
    gl_run off "$work/ref_$name" $args
    gl_run reference "$work/fast_$name" $args
    if [[ "$strict" == 1 ]] && grep -q "handled by softpipe\|mid-frame fallbacks\|frames on the fast path, .* [1-9][0-9]* mid-frame" "$work/fast_$name.log" && ! grep -q " 0 mid-frame fallbacks" "$work/fast_$name.log"; then
        echo "FAIL: the fast path was expected to take every draw"; grep -E "tensix: (draw|[0-9]+ frames)" "$work/fast_$name.log" | head -3; fail=1
    fi
    grep -E "tensix: [0-9]+ frames on the fast path" "$work/fast_$name.log" || true
    for f in "$work"/ref_${name}*.ppm "$work"/ref_${name}*.pgm; do
        [[ -e "$f" ]] || continue
        g="${f/ref_/fast_}"
        compare "$f" "$g" "$([[ $f == *.pgm ]] && echo 0.01 || echo 0.002)"
    done
    if [[ -n "${TT_METAL_HOME:-}" ]]; then
        gl_run tensix "$work/tx_$name" $args
        for f in "$work"/fast_${name}*.ppm "$work"/fast_${name}*.pgm; do
            [[ -e "$f" ]] || continue
            g="${f/fast_/tx_}"
            compare "$f" "$g" "$([[ $f == *.pgm ]] && echo 0.001 || echo 0.0005)"   # perspective colours: one level, a few pixels
        done
    fi
done
# Random fixed-function frames (state, primitives, clipping, mid-frame fallbacks): stock softpipe against the fast path.
echo "== random GL frames"
if ! "$py" "$root/mesa/tests/fuzz_gl.py" --count 100 --seed 1 --tame --no-refused --keep "$work/fuzzgl"; then fail=1; fi
if ! "$py" "$root/mesa/tests/fuzz_gl.py" --count 60 --seed 1000 --tame --keep "$work/fuzzgl2"; then fail=1; fi
if [[ -n "${TT_METAL_HOME:-}" ]]; then
    if ! "$py" "$root/mesa/tests/fuzz_gl.py" --count 12 --seed 3000 --tame --backend tensix --node "$node" --keep "$work/fuzzgl_tx"; then fail=1; fi
fi

# Random GLSL fragment shaders: the translator against stock softpipe (CPU backend), and some of them on the chip.
echo "== random GLSL shaders"
if ! "$py" "$root/mesa/tests/fuzz_glsl.py" --count 150 --seed 1 --keep "$work/fuzz"; then fail=1; fi
if [[ -n "${TT_METAL_HOME:-}" ]]; then
    if ! "$py" "$root/mesa/tests/fuzz_glsl.py" --count 12 --seed 7000 --backend tensix --node "$node" --keep "$work/fuzz_tx"; then fail=1; fi
fi
[[ $fail -eq 0 ]] && echo "GL tests OK" || { echo "GL tests FAILED"; exit 1; }
