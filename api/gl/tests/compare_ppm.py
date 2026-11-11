# SPDX-License-Identifier: Apache-2.0
"""Compare two PPM images (P6): python3 compare_ppm.py A.ppm B.ppm [--max-bad FRACTION] [--png OUT.png].

Prints the share of pixels that differ, the largest channel difference and the share with a difference above 1
and above 4 levels. Exit status 1 if the share of pixels differing by more than 4 levels exceeds --max-bad
(default 0.002)."""
import sys

import numpy as np


def read_ppm(path):
    """P6 (8 bit RGB) as (h, w, 3) or P5 (16 bit grey, big endian) as (h, w, 1) scaled to 0..255."""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        w, h = map(int, f.readline().split())
        f.readline()
        if magic == b"P6":
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w, 3)
        assert magic == b"P5"
        g = np.frombuffer(f.read(), dtype=">u2").reshape(h, w, 1).astype(np.float64)
        return (g / 257.0).round().astype(np.uint8)


def main():
    a, b = read_ppm(sys.argv[1]), read_ppm(sys.argv[2])
    max_bad = float(sys.argv[sys.argv.index("--max-bad") + 1]) if "--max-bad" in sys.argv else 0.002
    d = np.abs(a.astype(int) - b.astype(int)).max(axis=2)
    n = d.size
    print(f"{sys.argv[1]} vs {sys.argv[2]}: differing {np.count_nonzero(d) / n:.4%}, >1 level {np.count_nonzero(d > 1) / n:.4%}, "
          f">4 levels {np.count_nonzero(d > 4) / n:.4%}, max {int(d.max())}")
    if "--png" in sys.argv:
        from PIL import Image
        heat = np.clip(d.astype(float) * 16, 0, 255).astype(np.uint8)
        a3, b3 = (np.repeat(x, 3, axis=2) if x.shape[2] == 1 else x for x in (a, b))
        side = np.concatenate([a3, b3, np.stack([heat] * 3, axis=2)], axis=1)
        Image.fromarray(side).save(sys.argv[sys.argv.index("--png") + 1])
    return 1 if np.count_nonzero(d > 4) / n > max_bad else 0


if __name__ == "__main__":
    sys.exit(main())
