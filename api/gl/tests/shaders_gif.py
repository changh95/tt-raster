# SPDX-License-Identifier: Apache-2.0
"""An animated GIF of GLSL fragment shaders drawn on a Tensix chip through the Mesa fast path.

    TT_METAL_HOME=... python3 api/gl/tests/shaders_gif.py OUT.gif [--node 5] [--size 256] [--frames 24]
                                                   [--shaders phong,stripes,discard,loop,cond,smooth]

Runs tests/gl_scene.c once per shader (rotating geometry, the `t` uniform of the "uniform" shader advances with the frame),
with the Tensix backend, and tiles the frames of all shaders into one animation. Needs Pillow and numpy (TT_RASTER_PYTHON
in scripts/test_gl.sh), a C compiler with GL and X11 headers, xvfb.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--node", default="5")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--shaders", default="phong,stripes,discard,loop,cond,smooth")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    names = args.shaders.split(",")
    work = tempfile.mkdtemp(prefix="shaders_gif_")
    exe = os.path.join(work, "gl_scene")
    subprocess.run(["gcc", "-O1", "-o", exe, os.path.join(ROOT, "api", "gl", "tests", "gl_scene.c"), "-lGL", "-lX11", "-lm"], check=True)
    sheets = []
    for name in names:
        prefix = os.path.join(work, name)
        cmd = [os.path.join(ROOT, "scripts", "run_gl.sh"), args.node, "--size", f"{args.size}x{args.size}", "--backend", "tensix", exe,
               "--fs", name, "--sphere", "--size", str(args.size), "--frames", str(args.frames), "--dump", prefix]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ, TTR_GL_MIN_PIXELS="0"))
        sheets.append([Image.open(f"{prefix}{f}.ppm").convert("RGB") for f in range(args.frames)])
    cols = 3
    rows = (len(names) + cols - 1) // cols
    s = args.size
    frames = []
    for f in range(args.frames):
        canvas = Image.new("RGB", (cols * s, rows * s), (24, 24, 24))
        draw = ImageDraw.Draw(canvas)
        for k, name in enumerate(names):
            x, y = (k % cols) * s, (k // cols) * s
            canvas.paste(sheets[k][f], (x, y))
            draw.rectangle([x, y, x + 8 + 7 * len(name), y + 14], fill=(0, 0, 0))
            draw.text((x + 4, y + 2), name, fill=(255, 255, 255))
        frames.append(canvas)
    quant = [im.quantize(colors=160, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in frames]
    quant[0].save(args.out, save_all=True, append_images=quant[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {args.out}: {len(frames)} frames, {os.path.getsize(args.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    sys.exit(main())
