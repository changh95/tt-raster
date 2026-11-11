// SPDX-License-Identifier: Apache-2.0
//
// Scene layer for tiled-camera rendering (docs/PLAN.md, Track I): meshes, instances with poses,
// pinhole cameras, the vertex stage (transform, near/far clipping, projection, per-vertex shading)
// and the assembly of per-camera output images from the rasterizer's framebuffer.
//
// All cameras of a frame share one render target (an atlas): camera k of W occupies a slot of whole
// tiles, and its triangles are confined to a Viewport, so every camera is rendered like a separate
// image by one rasterizer run. Conventions follow Isaac Lab and Newton's Warp renderer: z-up world,
// cameras look down their -z axis with +x right and +y up, quaternions are (x, y, z, w), pixel
// centres at (p + 0.5) / size, shape colours are sRGB, one directional light plus hemispheric ambient.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "ttraster/backend.h"
#include "ttraster/frame.h"
#include "ttraster/reference.h"

namespace ttr::scene {

struct Mesh {
    std::vector<float> position;   // 3 per vertex
    std::vector<float> normal;     // 3 per vertex, unit length
    std::vector<uint32_t> index;   // 3 per triangle, counter-clockwise seen from outside
};
Mesh make_box(float hx, float hy, float hz);
Mesh make_sphere(float radius, int rings, int segments);
Mesh make_capsule(float radius, float half_height, int rings, int segments);  // axis +z
Mesh make_ground(float half_extent);                                          // quad at z = 0, normal +z

struct Instance {
    uint32_t mesh = 0;
    uint32_t world = 0;
    uint32_t shape_id = 0;              // reported by the segmentation output
    float albedo[3] = {1, 1, 1};        // sRGB
    float position[3] = {0, 0, 0};
    float rotation[4] = {0, 0, 0, 1};   // xyzw
};

struct Camera {
    uint32_t width = 96, height = 96;
    float focal_length = 24.0f;
    float horizontal_aperture = 20.955f;
    float vertical_aperture = 20.955f;
    float near_clip = 0.1f, far_clip = 20.0f;
    // Colour of the pixels where nothing is hit in the shaded image, RGBA8 with R in the low byte (Newton's
    // clear colour). The albedo image is cleared to transparent black, as Newton clears its albedo.
    uint32_t background = 0xFF000000u;
};

struct Pose {                  // camera to world
    float position[3];
    float rotation[4];         // xyzw
};

struct Light {
    float direction[3] = {-0.57735027f, 0.57735027f, -0.57735027f};  // the direction light travels
    // false: the colour is lit at the vertices and interpolated (fast, one fixed-function pass). true: the normals are
    // interpolated perspective-correctly and lit per pixel by a fragment shader (the shader path of the rasterizer;
    // closer to a ray tracer on coarse meshes, but much more device time).
    bool per_pixel = false;
};

enum Output : uint32_t {
    kDepth = 1,          // distance along the pixel's ray, 0 where nothing is hit
    kForwardDepth = 2,   // distance along the camera axis, 0 where nothing is hit
    kShape = 4,          // shape id, 0xFFFFFFFF where nothing is hit
    kAlbedo = 8,         // RGBA8, sRGB
    kRgb = 16,           // RGBA8, shaded, sRGB
    kNormal = 32,        // unit normal in world space, 0 where nothing is hit
};

// Caller-allocated outputs, one image per world: [world][height][width][channels]. Null = not wanted.
struct Images {
    float* depth = nullptr;           // 1 channel
    float* forward_depth = nullptr;   // 1
    uint32_t* shape = nullptr;        // 1
    uint8_t* albedo = nullptr;        // 4
    uint8_t* rgb = nullptr;           // 4
    float* normal = nullptr;          // 3
    uint32_t mask() const {
        return (depth ? kDepth : 0u) | (forward_depth ? kForwardDepth : 0u) | (shape ? kShape : 0u) |
               (albedo ? kAlbedo : 0u) | (rgb ? kRgb : 0u) | (normal ? kNormal : 0u);
    }
};

// The rasterizer seam lives in the core (ttraster/backend.h); the scene layer uses it under its old names.
using ::ttr::Backend;
using ::ttr::ReferenceBackend;
using Plane = ::ttr::RenderPlane;
inline constexpr uint32_t kPlaneColor = ::ttr::kRenderColor;
inline constexpr uint32_t kPlaneDepth = ::ttr::kRenderDepth;
inline constexpr uint32_t kPlaneFloat = ::ttr::kRenderFloat;
inline constexpr uint32_t kPlaneId = ::ttr::kRenderId;

struct RenderStats {
    double vertex_ms = 0;     // transform, clipping, projection, frame construction
    double raster_ms = 0;     // backend render calls (all passes)
    double assemble_ms = 0;   // output images from framebuffers
    double build_ms = 0;      // frame construction for the passes
    double total_ms = 0;      // all of Scene::render()
    uint64_t triangles = 0;   // triangles submitted to the rasterizer (per pass)
    uint32_t passes = 0;
    std::string backend_report;   // Backend::report() after the last pass
};

// One screen-space triangle of a camera slot, ready for the rasterizer (internal to the vertex stage).
namespace detail {
struct ScreenTri {
    float x[3], y[3], z[3];
    float albedo[3];      // constant per triangle (instance), sRGB
    float albedo_lin[3];  // the same, linear (per-pixel lighting)
    float rhw[3];         // 1 / camera-space depth per vertex (per-pixel lighting)
    float lit[3][3];      // per vertex
    float normal[3][3];   // per vertex
    uint32_t shape;
};
}  // namespace detail

// Storage one in-flight frame owns and reuses from frame to frame (fresh allocations of tens of megabytes
// per frame would cost page faults that dominate the frame time).
struct RenderSlot {
    Frame work_frame;
    std::vector<uint32_t> shape_cache;
    std::vector<std::vector<detail::ScreenTri>> per_world;
};

class Scene {
public:
    explicit Scene(uint32_t worlds) : worlds_(worlds) {}
    uint32_t worlds() const { return worlds_; }
    uint32_t add_mesh(Mesh m);
    uint32_t add_instance(const Instance& i);   // returns the instance index
    void set_pose(uint32_t instance, const float position[3], const float rotation[4]);
    size_t instances() const { return instances_.size(); }

    // Render one camera per world (poses.size() == worlds()) with the same intrinsics. Uses slot 0.
    void render(
        const Camera& camera, const std::vector<Pose>& poses, const Light& light, const Images& out, Backend& backend,
        RenderStats* stats = nullptr, size_t threads = 0) const;

    // Pipelined use. snapshot() copies the instance poses; render_snapshot() draws that copy later, possibly
    // on another thread, while the scene's poses move on. Frames in flight at the same time must use
    // different slots (here and in the backend) and different output buffers. Meshes and instances must not
    // be added while a frame is in flight.
    struct Snapshot {
        std::vector<Instance> instances;
    };
    Snapshot snapshot() const { return Snapshot{instances_}; }
    void render_snapshot(
        const Snapshot& snapshot, const Camera& camera, const std::vector<Pose>& poses, const Light& light,
        const Images& out, Backend& backend, int slot, RenderStats* stats = nullptr, size_t threads = 0) const;

private:
    void render_impl(
        const std::vector<Instance>& instances, const Camera& camera, const std::vector<Pose>& poses, const Light& light,
        const Images& out, Backend& backend, int slot, RenderStats* stats, size_t threads) const;

    uint32_t worlds_;
    std::vector<Mesh> meshes_;
    std::vector<Instance> instances_;
    mutable RenderSlot slots_[kRenderSlots];
};

float srgb_to_linear(float c);
float linear_to_srgb(float l);

}  // namespace ttr::scene
