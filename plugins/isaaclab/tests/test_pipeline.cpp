// SPDX-License-Identifier: Apache-2.0
//
// Frames in different slots may render at the same time from snapshots of the instance poses; each gives
// exactly the image of a synchronous render of those poses (CPU reference backend).

#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#include "check.h"
#include "scene/scene.h"

using namespace ttr::scene;

namespace {

struct Out {
    std::vector<float> depth, fdepth;
    std::vector<uint32_t> shape;
    std::vector<uint8_t> rgb;
    Images images(size_t n) {
        depth.assign(n, -1.0f);
        fdepth.assign(n, -1.0f);
        shape.assign(n, 1);
        rgb.assign(n * 4, 1);
        Images im;
        im.depth = depth.data();
        im.forward_depth = fdepth.data();
        im.shape = shape.data();
        im.rgb = rgb.data();
        return im;
    }
    bool operator==(const Out& o) const { return depth == o.depth && fdepth == o.fdepth && shape == o.shape && rgb == o.rgb; }
};

}  // namespace

int main() {
    const uint32_t worlds = 3, size = 48;
    Scene scene(worlds);
    const uint32_t box = scene.add_mesh(make_box(0.5f, 0.5f, 0.5f));
    const uint32_t ground = scene.add_mesh(make_ground(50.0f));
    for (uint32_t w = 0; w < worlds; w++) {
        Instance g;
        g.mesh = ground;
        g.world = w;
        g.shape_id = 0;
        scene.add_instance(g);
        Instance b;
        b.mesh = box;
        b.world = w;
        b.shape_id = 1 + w;
        b.albedo[0] = 0.9f;
        b.albedo[1] = 0.2f;
        b.albedo[2] = 0.1f;
        b.position[0] = 4.0f;
        b.position[2] = 0.5f;
        scene.add_instance(b);
    }
    Camera cam;
    cam.width = cam.height = size;
    cam.near_clip = 0.1f;
    cam.far_clip = 30.0f;
    std::vector<Pose> cams(worlds);
    for (uint32_t w = 0; w < worlds; w++) {
        const float q[4] = {0.5f, -0.5f, -0.5f, 0.5f};   // looks along +x, z up
        const float pos[3] = {0.0f, 0.0f, 1.0f};
        std::memcpy(cams[w].position, pos, sizeof pos);
        std::memcpy(cams[w].rotation, q, sizeof q);
    }
    const size_t n = static_cast<size_t>(worlds) * size * size;

    auto move_box = [&](float y) {
        for (uint32_t w = 0; w < worlds; w++) {
            const float pos[3] = {4.0f, y + 0.3f * w, 0.5f};
            const float rot[4] = {0, 0, 0, 1};
            scene.set_pose(2 * w + 1, pos, rot);
        }
    };

    // Reference images for four different poses, rendered synchronously.
    ReferenceBackend be_sync;
    std::vector<Out> expected(4);
    for (int k = 0; k < 4; k++) {
        move_box(-0.8f + 0.5f * k);
        Images im = expected[k].images(n);
        scene.render(cam, cams, Light{}, im, be_sync);
    }
    CHECK(!(expected[0] == expected[3]));

    // The same four, as two pairs of concurrent frames; the poses move on after each snapshot.
    ReferenceBackend be_async;
    std::vector<Out> got(4);
    for (int pair = 0; pair < 2; pair++) {
        std::vector<std::thread> threads;
        std::vector<Scene::Snapshot> snaps;
        std::vector<Images> ims;
        for (int s = 0; s < 2; s++) {
            const int k = 2 * pair + s;
            move_box(-0.8f + 0.5f * k);
            snaps.push_back(scene.snapshot());
            ims.push_back(got[k].images(n));
            move_box(10.0f);   // must not leak into the frames already snapshotted
        }
        for (int s = 0; s < 2; s++) {
            threads.emplace_back([&, s] { scene.render_snapshot(snaps[s], cam, cams, Light{}, ims[s], be_async, s); });
        }
        for (auto& t : threads) t.join();
    }
    for (int k = 0; k < 4; k++) {
        CHECK_MSG(got[k] == expected[k], "frame %d differs from its synchronous render", k);
    }

    bool threw = false;
    try {
        Out o;
        Images im = o.images(n);
        scene.render_snapshot(scene.snapshot(), cam, cams, Light{}, im, be_async, 7);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);   // slot out of range
    return ttr_test::finish();
}
