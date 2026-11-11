// SPDX-License-Identifier: Apache-2.0
//
// Per-triangle depth and blend state (Triangle::own_state): depth function, depth writes, blending, and that
// the default state is the old behaviour.

#include <cmath>
#include <vector>

#include "check.h"
#include "ttraster/reference.h"
#include "ttraster/setup.h"

using namespace ttr;

namespace {

Vertex V(float x, float y, float z, const float c[4]) {
    Vertex v{};
    v.x = x;
    v.y = y;
    v.z = z;
    v.rhw = 1.0f;
    for (int i = 0; i < 4; i++) v.color[i] = c[i];
    return v;
}

// The rectangle [8, 56) x [8, 56) as two flat triangles.
void rect(Frame& f, float z, const float c[4], bool own = false, DepthFunc df = DepthFunc::Less, bool dw = true,
          BlendMode blend = BlendMode::None) {
    Triangle t[2];
    t[0].v[0] = V(8, 8, z, c);
    t[0].v[1] = V(56, 8, z, c);
    t[0].v[2] = V(56, 56, z, c);
    t[1].v[0] = V(8, 8, z, c);
    t[1].v[1] = V(56, 56, z, c);
    t[1].v[2] = V(8, 56, z, c);
    for (Triangle& tri : t) {
        tri.flat = true;
        tri.own_state = own;
        tri.depth_func = df;
        tri.depth_write = dw;
        tri.blend = blend;
        f.triangles.push_back(tri);
    }
}

Frame make_frame() {
    Frame f;
    f.width = f.height = 64;
    f.clear_color[0] = f.clear_color[1] = f.clear_color[2] = 0.0f;
    f.clear_color[3] = 1.0f;
    return f;
}

uint32_t px(const Framebuffer& fb, int x, int y) { return fb.color[fb.index(x, y)]; }
uint32_t pack(float r, float g, float b, float a) {
    const float c[4] = {r, g, b, a};
    return pack_rgba8(c);
}

}  // namespace

int main() {
    const float red[4] = {1, 0, 0, 1}, green[4] = {0, 1, 0, 1}, blue[4] = {0, 0, 1, 1};
    const float red_half[4] = {1, 0, 0, 0.5f}, grey[4] = {0.7f, 0.7f, 0.7f, 0.7f};

    // Default state = old behaviour: LESS, writes on.
    {
        Frame a = make_frame(), b = make_frame();
        rect(a, 0.3f, red);
        rect(a, 0.6f, green);
        rect(b, 0.3f, red, true, DepthFunc::Less, true, BlendMode::None);
        rect(b, 0.6f, green, true, DepthFunc::Less, true, BlendMode::None);
        const Framebuffer fa = render_reference(a), fb = render_reference(b);
        CHECK(fa.color == fb.color && fa.depth == fb.depth);
        CHECK(px(fa, 30, 30) == pack(1, 0, 0, 1));
    }

    // Depth test off: the later triangle wins whatever its depth; depth is not written (the buffer keeps its value).
    {
        Frame f = make_frame();
        rect(f, 0.3f, red);
        rect(f, 0.8f, green, true, DepthFunc::Always, false);
        const Framebuffer fb = render_reference(f);
        CHECK(px(fb, 30, 30) == pack(0, 1, 0, 1));
        CHECK_MSG(std::fabs(fb.depth[fb.index(30, 30)] - 0.3f) < 1e-6f, "depth %f", fb.depth[fb.index(30, 30)]);
    }

    // Depth writes off: the first triangle draws colour but leaves the depth buffer alone, so a farther triangle
    // that comes later still passes the test against the clear depth.
    {
        Frame f = make_frame();
        rect(f, 0.3f, red, true, DepthFunc::Less, false);
        rect(f, 0.6f, green);
        const Framebuffer fb = render_reference(f);
        CHECK(px(fb, 30, 30) == pack(0, 1, 0, 1));
        CHECK(std::fabs(fb.depth[fb.index(30, 30)] - 0.6f) < 1e-6f);
    }

    // A triangle's own LESS_EQUAL against a frame default of LESS: an equal depth wins.
    {
        Frame f = make_frame();
        rect(f, 0.5f, red);
        rect(f, 0.5f, green);
        CHECK(px(render_reference(f), 30, 30) == pack(1, 0, 0, 1));   // LESS: the first stays
        Frame g = make_frame();
        rect(g, 0.5f, red);
        rect(g, 0.5f, green, true, DepthFunc::LessEqual, true);
        CHECK(px(render_reference(g), 30, 30) == pack(0, 1, 0, 1));
        Frame h = make_frame();   // and the other way round: frame LESS_EQUAL, own LESS
        h.depth_func = DepthFunc::LessEqual;
        rect(h, 0.5f, red);
        rect(h, 0.5f, green, true, DepthFunc::Less, true);
        CHECK(px(render_reference(h), 30, 30) == pack(1, 0, 0, 1));
    }

    // Alpha blending over an opaque layer: the result is the blend chain in floats, then quantized.
    {
        Frame f = make_frame();
        rect(f, 0.6f, green);
        rect(f, 0.3f, red_half, true, DepthFunc::Less, false, BlendMode::Alpha);
        const Framebuffer fb = render_reference(f);
        const float r = blend_channel(BlendMode::Alpha, 1.0f, 0.5f, 0.0f), g = blend_channel(BlendMode::Alpha, 0.0f, 0.5f, 1.0f),
                    b = blend_channel(BlendMode::Alpha, 0.0f, 0.5f, 0.0f), a = blend_channel(BlendMode::Alpha, 0.5f, 0.5f, 1.0f);
        CHECK(px(fb, 30, 30) == pack(r, g, b, a));
        CHECK(((px(fb, 30, 30) >> 0) & 255) == 128 && ((px(fb, 30, 30) >> 8) & 255) == 128);
        CHECK(px(fb, 4, 4) == pack(0, 0, 0, 1));   // outside the rectangles: the clear colour
        // Two layers accumulate in floats (not in quantized levels).
        Frame g2 = make_frame();
        rect(g2, 0.3f, red_half, true, DepthFunc::Always, false, BlendMode::Alpha);
        rect(g2, 0.3f, red_half, true, DepthFunc::Always, false, BlendMode::Alpha);
        float rr = 0.0f, ra = 1.0f;
        for (int k = 0; k < 2; k++) {
            rr = blend_channel(BlendMode::Alpha, 1.0f, 0.5f, rr);
            ra = blend_channel(BlendMode::Alpha, 0.5f, 0.5f, ra);
        }
        CHECK(px(render_reference(g2), 30, 30) == pack(rr, 0, 0, ra));
    }

    // Additive blending clamps at 1; multiply scales by the target.
    {
        Frame f = make_frame();
        rect(f, 0.6f, grey);
        rect(f, 0.3f, grey, true, DepthFunc::Always, false, BlendMode::Add);
        const Framebuffer fb = render_reference(f);
        CHECK(px(fb, 30, 30) == pack(1, 1, 1, 1));
        Frame g = make_frame();
        rect(g, 0.6f, blue);
        rect(g, 0.3f, grey, true, DepthFunc::Always, false, BlendMode::Multiply);
        const Framebuffer fm = render_reference(g);
        CHECK(px(fm, 30, 30) == pack(0, 0, 0.7f * 1.0f, 0.7f * 1.0f));
    }

    // (SRC_ALPHA, ONE): the fragment scaled by its alpha added to the target.
    {
        Frame f = make_frame();
        rect(f, 0.6f, blue);
        rect(f, 0.3f, red_half, true, DepthFunc::Always, false, BlendMode::AlphaAdd);
        CHECK(px(render_reference(f), 30, 30) == pack(0.5f, 0, 1, 1));   // alpha: 0.5 * 0.5 + 1 clamps to 1
    }

    // Blending and ordinary triangles in one frame; an own-state triangle in the middle of a default sequence.
    {
        Frame f = make_frame();
        rect(f, 0.5f, red);
        rect(f, 0.2f, green, true, DepthFunc::Less, true, BlendMode::None);
        Triangle t = f.triangles.back();
        CHECK(t.own_state);
        const Framebuffer fb = render_reference(f);
        CHECK(px(fb, 30, 30) == pack(0, 1, 0, 1));
    }
    return ttr_test::finish();
}
