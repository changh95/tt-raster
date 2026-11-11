// SPDX-License-Identifier: Apache-2.0
//
// Frame format (docs/PLAN.md, Phase 1.1): what a renderer is asked to draw.
// Everything here is host-side and independent of tt-metal.

#pragma once

#include <cstdint>
#include <vector>

namespace ttr {

inline constexpr int32_t kTileSize = 32;
inline constexpr int32_t kSubpixelBits = 4;
inline constexpr int32_t kSubpixelScale = 1 << kSubpixelBits;  // 1/16 pixel grid
// Largest render target edge. With 4 subpixel bits an edge function coefficient is then at most
// 2^15 and the tile-relative products stay exact in FP32 (see docs/FINDINGS.md, "Edge exactness").
inline constexpr int32_t kMaxTargetSize = 2048;
// Input vertices beyond this many pixels from the origin are rejected before snapping.
inline constexpr float kMaxInputCoord = 1048576.0f;  // 2^20
// Frames that may be in flight at once (pipelined rendering). Backends keep this many independent sets of
// per-frame storage, selected by a slot index.
inline constexpr int kRenderSlots = 2;

struct Vertex {
    float x, y;       // window position in pixels, x right, y down
    float z;          // depth in [0, 1]
    float rhw;        // 1/w, carried for the perspective-correct path (Track G); unused in v1
    float color[4];   // RGBA in [0, 1]
};

// Always: the depth test is off (every fragment passes).
enum class DepthFunc : uint8_t { Less, LessEqual, Always };

// Blending of the fragment colour (clamped to [0, 1]) with the colour already in the target, result clamped to
// [0, 1]; the same factor pair is used for RGB and alpha. With s the fragment, d the target, sa the fragment alpha:
//   Alpha          s * sa + d * (1 - sa)        (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
//   Add            s + d                        (GL_ONE, GL_ONE)
//   Premultiplied  s + d * (1 - sa)             (GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
//   Multiply       s * d                        (GL_DST_COLOR, GL_ZERO)
//   AlphaAdd       s * sa + d                   (GL_SRC_ALPHA, GL_ONE)
// evaluated as fma(s, a, d * b) with (a, b) = (sa, 1 - sa), (1, 1), (1, 1 - sa), (0, s), (sa, 1) respectively.
enum class BlendMode : uint8_t { None, Alpha, Add, Premultiplied, Multiply, AlphaAdd };

// A rectangle of the render target that some triangles are confined to (one camera of a tiled
// camera atlas). Triangles are clipped to it, so nothing bleeds into a neighbour. The rectangle
// must not extend past the padded target and is at most kMaxTargetSize pixels wide and high.
struct Viewport {
    int32_t x = 0, y = 0, width = 0, height = 0;
};

struct Triangle {
    Vertex v[3];
    uint32_t viewport = 0;  // index into Frame::viewports (ignored when that list is empty)
    // Flat shading takes the colour of v[0], the provoking vertex. Smooth shading interpolates
    // linearly in screen space, or perspective-correctly when `perspective` is set: the colour is then
    // (sum of lambda_i * c_i * rhw_i) / (sum of lambda_i * rhw_i) with screen-space weights lambda, and every
    // vertex needs a finite rhw > 0 (triangles with other values are dropped as non-finite).
    bool flat = false;
    bool perspective = false;
    // Per-triangle state. Unless `own_state` is set the triangle uses the frame's depth function, writes depth and does
    // not blend. Triangles with their own state can follow each other in one frame (depth test on and off, blended
    // layers, depth writes off); they are drawn in API order like every other triangle.
    bool own_state = false;
    DepthFunc depth_func = DepthFunc::Less;
    bool depth_write = true;
    BlendMode blend = BlendMode::None;
};

// Face culling by the sign of signed_area2 = (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0) in window
// coordinates (y down). A cull mode names the sign that is removed.
enum class CullMode : uint8_t { None, Positive, Negative };

struct Frame {
    uint32_t width = 0;   // viewport in pixels
    uint32_t height = 0;
    float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float clear_depth = 1.0f;
    DepthFunc depth_func = DepthFunc::Less;
    CullMode cull = CullMode::None;
    std::vector<Triangle> triangles;  // API order: later triangles win depth ties under LessEqual
    // Empty: the whole padded target is one viewport (at most kMaxTargetSize wide and high, the
    // original single-image mode). Otherwise every triangle is confined to its viewport and the
    // target may be larger than kMaxTargetSize.
    std::vector<Viewport> viewports;

    // The render target is the viewport padded up to whole tiles. Pixels in the padding are
    // rasterized like any other and are cropped away on read-back.
    uint32_t padded_width() const { return (width + kTileSize - 1) / kTileSize * kTileSize; }
    uint32_t padded_height() const { return (height + kTileSize - 1) / kTileSize * kTileSize; }
    uint32_t tiles_x() const { return padded_width() / kTileSize; }
    uint32_t tiles_y() const { return padded_height() / kTileSize; }
};

// Colour packing used for the framebuffer and for the device's Dest colour tile:
// R in bits 0..7, G 8..15, B 16..23, A 24..31. A channel value v maps to
// clamp(floor(v * 255 + 0.5), 0, 255); NaN maps to 0.
uint8_t quantize_unorm8(float v);
uint32_t pack_rgba8(const float rgba[4]);

}  // namespace ttr
