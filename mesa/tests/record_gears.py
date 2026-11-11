# SPDX-License-Identifier: Apache-2.0
"""Record glxgears on stock softpipe and on the Tensix fast path side by side (an animated GIF).

    TT_METAL_HOME=... python3 mesa/tests/record_gears.py OUT.gif [--node 5] [--size 1280x720] [--seconds 6] [--fps 15]

Starts two Xvfb servers with a memory-mapped framebuffer each (`-fbdir`), runs glxgears on both (stock softpipe
on the CPU; the Mesa built by scripts/build_mesa.sh with the fast path on a chip through scripts/run.sh), samples
both framebuffers at --fps and writes the GIF. Needs Pillow and numpy (TT_RASTER_PYTHON in scripts/test_gl.sh).
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def read_xwd(path):
    data = open(path, "rb").read()
    (header_size, _v, _f, _d, width, height, _xo, _bo, _bu, _bbo, _bp, bpp, bpl, _vc, rmask, gmask, bmask, _br, _ce,
     ncolors) = struct.unpack(">20I", data[:80])
    off = header_size + ncolors * 12
    px = np.frombuffer(data[off:off + bpl * height], dtype="<u4").reshape(height, bpl // 4)[:, :width]

    def chan(mask):
        return ((px & mask) >> ((mask & -mask).bit_length() - 1)).astype(np.uint8)

    return np.stack([chan(rmask), chan(gmask), chan(bmask)], axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--node", default="5")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--thumb", default="480x270", help="size of each half of the GIF")
    args = ap.parse_args()
    w, h = map(int, args.size.split("x"))
    tw, th = map(int, args.thumb.split("x"))
    prefix = os.environ.get("TT_RASTER_MESA_PREFIX", os.path.expanduser("~/experiments/mesa-install"))
    lib = os.path.join(prefix, "lib", "x86_64-linux-gnu")
    env = dict(os.environ, LD_LIBRARY_PATH=lib + ":" + os.environ.get("LD_LIBRARY_PATH", ""), LIBGL_DRIVERS_PATH=lib + "/dri",
               LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="softpipe",
               TTRASTER_LIB=os.environ.get("TTRASTER_LIB", os.path.join(ROOT, "build", "src", "libttraster.so")))
    tmp = tempfile.mkdtemp()
    procs, fbdirs = [], []
    try:
        for i, label in enumerate(("stock", "tensix")):
            d = os.path.join(tmp, label)
            os.makedirs(d)
            fbdirs.append(d)
            procs.append(subprocess.Popen(["Xvfb", f":{81 + i}", "-screen", "0", f"{w}x{h}x24", "-fbdir", d],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(1.5)
        logs = []
        for i, label in enumerate(("stock", "tensix")):
            log = open(os.path.join(tmp, label + ".log"), "w")
            logs.append(log.name)
            e = dict(env, DISPLAY=f":{81 + i}")
            if label == "stock":
                cmd = ["stdbuf", "-oL", "glxgears", "-geometry", f"{w}x{h}"]
            else:
                e["TTR_GL"] = "tensix"
                cmd = [os.path.join(ROOT, "scripts", "run.sh"), args.node, "/usr/bin/stdbuf", "-oL", "/usr/bin/glxgears",
                       "-geometry", f"{w}x{h}"]
            procs.append(subprocess.Popen(cmd, env=e, stdout=log, stderr=subprocess.STDOUT))
        time.sleep(6.0)   # the first Tensix frames compile the kernels
        frames, t0 = [], time.time()
        n = int(args.seconds * args.fps)
        for k in range(n):
            target = t0 + k / args.fps
            time.sleep(max(0.0, target - time.time()))
            pair = []
            for d in fbdirs:
                img = Image.fromarray(read_xwd(os.path.join(d, "Xvfb_screen0")))
                pair.append(img.resize((tw, th), Image.LANCZOS))
            frames.append(pair)
        time.sleep(max(0.0, 11.0 - (time.time() - t0)))   # glxgears prints its FPS every 5 seconds
        fps = []
        for name in logs:
            m = re.findall(r"= ([0-9.]+) FPS", open(name).read())
            fps.append(float(m[-1]) if m else float("nan"))
        print(f"glxgears {w}x{h}: stock {fps[0]:.0f} FPS, Tensix {fps[1]:.0f} FPS")
        labels = (f"stock softpipe, CPU: {fps[0]:.0f} FPS", f"Tensix fast path, Blackhole chip: {fps[1]:.0f} FPS")
        out = []
        for a, b in frames:
            canvas = Image.new("RGB", (2 * tw + 4, th + 22), (24, 24, 24))
            canvas.paste(a, (0, 22))
            canvas.paste(b, (tw + 4, 22))
            d = ImageDraw.Draw(canvas)
            d.text((6, 5), labels[0], fill=(255, 255, 255))
            d.text((tw + 10, 5), labels[1], fill=(255, 255, 255))
            out.append(canvas.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
        out[0].save(args.out, save_all=True, append_images=out[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
        print(f"wrote {args.out}: {len(out)} frames, {os.path.getsize(args.out) / 1e6:.2f} MB")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    sys.exit(main())
