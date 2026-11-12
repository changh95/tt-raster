// SPDX-License-Identifier: Apache-2.0
//
// The two renderers of 3D Gaussian Splatting.
//
// render_reference is the oracle: the forward pass of the reference implementation in float on the CPU. Splats are binned into 32 x 32 pixel
// tiles (the tile size of the Tensix renderer; the original uses 16 x 16, which changes nothing but speed), sorted by depth within the tile,
// and blended front to back per pixel with the original's rules: alpha = min(0.99, opacity exp(power)); a splat whose alpha at the pixel is
// below 1/255 is skipped; a pixel is finished when its transmittance would fall below 1e-4; the background is added with the remaining
// transmittance. Where the original culls a splat to the tiles its 3-sigma circle touches, this version applies it wherever its alpha
// reaches 1/255 (up to 3.3 sigma), so the result does not depend on the tile size.
//
// render_backend draws the same splats through the renderer core: each splat is a square of two triangles carrying a fragment shader that
// evaluates the Gaussian (a 20-instruction program) and kills the fragment where alpha would be below 1/255, with "over" blending, drawn
// back to front in API order. Back to front with the over operator gives the same sum as the front-to-back recursion, up to the early stop
// (T < 1e-4) and the 8-bit quantization of the target after every blend. It runs on any Backend: the CPU reference, a Tensix chip or a device server.

#pragma once

#include <cstdint>
#include <vector>

#include "gs/gaussians.h"
#include "ttraster/backend.h"
#include "ttraster/image.h"

namespace ttr::gs {

struct FloatImage {
    uint32_t width = 0, height = 0;
    std::vector<float> rgb;   // 3 per pixel, row-major, top row first, not clamped
};

struct RenderStats {
    size_t splats = 0;
    size_t tile_records = 0;      // reference: (splat, tile) pairs
    double bin_ms = 0, raster_ms = 0, build_ms = 0, render_ms = 0;
};

FloatImage render_reference(const std::vector<Splat>& splats, const Camera& camera, const float background[3], RenderStats* stats = nullptr,
                            size_t threads = 0);
// Clamp to [0, 1] and round to 8 bits (alpha 255).
Image to_image(const FloatImage& image);

// Throws std::runtime_error if the target is larger than the core's 2048 x 2048 or the backend refuses the frame.
Image render_backend(const std::vector<Splat>& splats, const Camera& camera, const float background[3], Backend& backend, RenderStats* stats = nullptr);

// The fragment program render_backend uses (exposed for tests and for compiling it ahead of time).
ShaderProgram splat_shader();

// PSNR over the RGB channels of two images of the same size (infinity if identical).
double psnr_rgb(const Image& a, const Image& b, uint32_t* max_error = nullptr);

}  // namespace ttr::gs
