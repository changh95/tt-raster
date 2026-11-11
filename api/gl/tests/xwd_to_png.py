# SPDX-License-Identifier: Apache-2.0
"""Convert the framebuffer file of `Xvfb -fbdir DIR` (XWD format, 32 bit) to a PNG: xwd_to_png.py Xvfb_screen0 OUT.png"""
import struct
import sys

import numpy as np
from PIL import Image


def main():
    data = open(sys.argv[1], "rb").read()
    (header_size, _ver, _fmt, _depth, width, height, _xoff, _byte_order, _unit, _bit_order, _pad, bpp, bytes_per_line,
     _vclass, rmask, gmask, bmask, _bits_per_rgb, _cmap_entries, ncolors) = struct.unpack(">20I", data[:80])
    assert bpp == 32, bpp
    off = header_size + ncolors * 12
    px = np.frombuffer(data[off:off + bytes_per_line * height], dtype="<u4").reshape(height, bytes_per_line // 4)[:, :width]
    img = np.stack([(px & rmask) >> (rmask & -rmask).bit_length() - 1, (px & gmask) >> (gmask & -gmask).bit_length() - 1,
                    (px & bmask) >> (bmask & -bmask).bit_length() - 1], axis=2).astype(np.uint8)
    Image.fromarray(img).save(sys.argv[2])


if __name__ == "__main__":
    main()
