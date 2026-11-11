// SPDX-License-Identifier: Apache-2.0
//
// Scene layer: projection, near/far clipping, outputs, against analytic expectations.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "check.h"
#include "scene/scene.h"

using namespace ttr::scene;

namespace {

// Camera at the origin looking along world +x (the Isaac Lab "world" convention), z up.
Pose forward_pose(float x = 0, float y = 0, float z = 0) {
    return Pose{{x, y, z}, {0.5f, -0.5f, -0.5f, 0.5f}};  // rotation matrix [[0,0,-1],[-1,0,0],[0,1,0]]
}

}  // namespace

int main() {
    Camera cam;
    cam.width = cam.height = 64;
    cam.focal_length = 20.955f;  // 90 degree field of view: tan(half) = aperture / (2 f) = 0.5 ... = 1 below
    cam.horizontal_aperture = cam.vertical_aperture = 2.0f * 20.955f;  // tan(45 deg) = 1
    cam.near_clip = 0.1f;
    cam.far_clip = 50.0f;

    // 1. A box straight ahead: forward depth at the centre is the distance to its near face.
    {
        Scene s(1);
        const uint32_t box = s.add_mesh(make_box(0.5f, 1.0f, 1.0f));
        Instance i;
        i.mesh = box;
        i.shape_id = 7;
        i.albedo[0] = 1.0f; i.albedo[1] = 0.0f; i.albedo[2] = 0.0f;
        i.position[0] = 6.0f;
        s.add_instance(i);
        std::vector<float> depth(64 * 64), fdepth(64 * 64), normal(64 * 64 * 3);
        std::vector<uint32_t> shape(64 * 64);
        std::vector<uint8_t> albedo(64 * 64 * 4), rgb(64 * 64 * 4);
        Images out;
        out.depth = depth.data(); out.forward_depth = fdepth.data(); out.shape = shape.data();
        out.albedo = albedo.data(); out.rgb = rgb.data(); out.normal = normal.data();
        ReferenceBackend be;
        s.render(cam, {forward_pose()}, Light{}, out, be);
        const size_t c = 32 * 64 + 32;
        CHECK_MSG(std::fabs(fdepth[c] - 5.5f) < 2e-3f, "forward depth %.4f", fdepth[c]);
        CHECK(shape[c] == 7);
        CHECK(std::fabs(normal[3 * c + 0] + 1.0f) < 1e-4f && std::fabs(normal[3 * c + 1]) < 1e-4f);  // faces the camera: -x
        CHECK(albedo[4 * c] == 255 && albedo[4 * c + 1] == 0 && albedo[4 * c + 3] == 255);
        CHECK(shape[0] == 0xFFFFFFFFu && depth[0] == 0.0f && fdepth[0] == 0.0f);                    // corner: background
        // Misses: the shaded image holds the camera's background colour (default opaque black), the albedo
        // image is transparent black, as in Newton.
        CHECK(rgb[0] == 0 && rgb[1] == 0 && rgb[2] == 0 && rgb[3] == 255);
        CHECK(albedo[0] == 0 && albedo[1] == 0 && albedo[2] == 0 && albedo[3] == 0);
        {
            Camera grey = cam;
            grey.background = 0xFFEEEEEEu;  // Isaac Lab's default clear colour
            Camera odd = cam;
            odd.background = 0x80FF7F01u;   // A, B, G, R = 0x80, 0xFF, 0x7F, 0x01: exercises every byte
            ReferenceBackend be2;
            s.render(grey, {forward_pose()}, Light{}, out, be2);
            CHECK(rgb[0] == 0xEE && rgb[1] == 0xEE && rgb[2] == 0xEE && rgb[3] == 0xFF);
            CHECK(rgb[4 * c] > 0 && rgb[4 * c + 1] == 0);  // pixels that are hit do not take the background
            s.render(odd, {forward_pose()}, Light{}, out, be2);
            CHECK_MSG(rgb[0] == 0x01 && rgb[1] == 0x7F && rgb[2] == 0xFF && rgb[3] == 0x80, "%02x %02x %02x %02x", rgb[0], rgb[1], rgb[2], rgb[3]);
            s.render(cam, {forward_pose()}, Light{}, out, be2);
        }
        // The ray distance equals the axis distance at the centre and exceeds it away from it.
        CHECK(std::fabs(depth[c] - fdepth[c]) < 2e-3f);
        const size_t edge = 32 * 64 + 36;  // 4 pixels to the right of centre, still on the box face (it spans about 5.8 px each way)
        CHECK(shape[edge] == 7 && depth[edge] > fdepth[edge] * 1.001f);
        CHECK(rgb[4 * c] > 0 && rgb[4 * c + 1] == 0);  // red albedo stays red when shaded
        // Pixel footprint: the box spans y in [-1, 1] at 5.5 m with a 90 degree field: 64 * (1 / 5.5) / 2 px each way.
        int top = 0;
        for (int y = 0; y < 64; y++) if (shape[y * 64 + 32] == 7) { top = y; break; }
        CHECK_MSG(std::abs(top - static_cast<int>(32 - 32.0 / 5.5)) <= 1, "top row %d", top);
    }

    // 2. Near clipping: a ground quad passing through the camera is clipped, not dropped or corrupted.
    {
        Scene s(1);
        Instance g;
        g.mesh = s.add_mesh(make_ground(100.0f));
        g.shape_id = 0;
        s.add_instance(g);
        std::vector<float> fdepth(64 * 64);
        Images out;
        out.forward_depth = fdepth.data();
        ReferenceBackend be;
        s.render(cam, {forward_pose(0, 0, 1.5f)}, Light{}, out, be);   // 1.5 m above the ground, looking horizontally
        // Row below the horizon sees the ground at depth h / tan(angle): at the bottom row ~ 1.5 / (31.5/32).
        const float bottom = fdepth[63 * 64 + 32];
        CHECK_MSG(std::fabs(bottom - 1.5f / (31.5f / 32.0f)) < 0.02f, "bottom row forward depth %.4f", bottom);
        CHECK(fdepth[0] == 0.0f);                     // the sky
        CHECK(fdepth[40 * 64 + 32] > bottom);         // farther up the image is farther away
    }

    // 3. Two worlds with different poses land in different atlas slots and do not mix.
    {
        Scene s(2);
        const uint32_t box = s.add_mesh(make_box(0.5f, 0.5f, 0.5f));
        for (uint32_t w = 0; w < 2; w++) {
            Instance i;
            i.mesh = box; i.world = w; i.shape_id = w + 1;
            i.position[0] = 4.0f + 2.0f * static_cast<float>(w);
            s.add_instance(i);
        }
        std::vector<float> fdepth(2 * 64 * 64);
        std::vector<uint32_t> shape(2 * 64 * 64);
        Images out;
        out.forward_depth = fdepth.data(); out.shape = shape.data();
        ReferenceBackend be;
        s.render(cam, {forward_pose(), forward_pose()}, Light{}, out, be);
        const size_t c = 32 * 64 + 32;
        CHECK(shape[c] == 1 && shape[64 * 64 + c] == 2);
        CHECK(std::fabs(fdepth[c] - 3.5f) < 2e-3f && std::fabs(fdepth[64 * 64 + c] - 5.5f) < 2e-3f);
    }

    // 4. Primitive meshes are closed and consistently wound: every edge is used once in each direction.
    for (const Mesh& m : {make_box(1, 2, 3), make_sphere(1, 8, 12), make_capsule(0.5f, 1, 4, 12)}) {
        std::vector<std::pair<uint32_t, uint32_t>> edges;
        // Positions can be duplicated per face (box normals), so compare edges by position.
        auto key = [&](uint32_t v) {
            return static_cast<uint32_t>(std::lround(m.position[3 * v] * 1000)) * 73856093u ^
                   static_cast<uint32_t>(std::lround(m.position[3 * v + 1] * 1000)) * 19349663u ^
                   static_cast<uint32_t>(std::lround(m.position[3 * v + 2] * 1000)) * 83492791u;
        };
        std::vector<std::pair<uint32_t, uint32_t>> forward;
        for (size_t t = 0; t + 2 < m.index.size(); t += 3)
            for (int k = 0; k < 3; k++) {
                const uint32_t a = key(m.index[t + k]), b = key(m.index[t + (k + 1) % 3]);
                if (a != b) forward.emplace_back(a, b);   // degenerate pole triangles have a zero-length edge
            }
        size_t unmatched = 0;
        std::sort(forward.begin(), forward.end());
        for (auto& e : forward) {
            if (!std::binary_search(forward.begin(), forward.end(), std::make_pair(e.second, e.first))) unmatched++;
        }
        CHECK_MSG(unmatched == 0, "%zu edges without an opposite twin", unmatched);
    }
    return ttr_test::finish();
}
