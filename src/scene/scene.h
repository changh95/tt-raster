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
#include <vector>

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
};

struct Pose {                  // camera to world
    float position[3];
    float rotation[4];         // xyzw
};

struct Light {
    float direction[3] = {-0.57735027f, 0.57735027f, -0.57735027f};  // the direction light travels
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

// Something that rasterizes a Frame. The CPU reference is one; the Tensix renderer is another.
// `float_planes`: also return the unquantized colour channels in Framebuffer::plane (needed for normals).
class Backend {
public:
    virtual ~Backend() = default;
    virtual const Framebuffer& render(const Frame& frame, bool float_planes) = 0;
};

class ReferenceBackend : public Backend {
public:
    const Framebuffer& render(const Frame& frame, bool float_planes) override;

private:
    Framebuffer fb_;
};

struct RenderStats {
    double vertex_ms = 0;     // transform, clipping, projection, frame construction
    double raster_ms = 0;     // backend render calls (all passes)
    double assemble_ms = 0;   // output images from framebuffers
    uint64_t triangles = 0;   // triangles submitted to the rasterizer (per pass)
    uint32_t passes = 0;
};

class Scene {
public:
    explicit Scene(uint32_t worlds) : worlds_(worlds) {}
    uint32_t worlds() const { return worlds_; }
    uint32_t add_mesh(Mesh m);
    uint32_t add_instance(const Instance& i);   // returns the instance index
    void set_pose(uint32_t instance, const float position[3], const float rotation[4]);
    size_t instances() const { return instances_.size(); }

    // Render one camera per world (poses.size() == worlds()) with the same intrinsics.
    void render(
        const Camera& camera, const std::vector<Pose>& poses, const Light& light, const Images& out, Backend& backend,
        RenderStats* stats = nullptr, size_t threads = 0) const;

private:
    uint32_t worlds_;
    std::vector<Mesh> meshes_;
    std::vector<Instance> instances_;
};

float srgb_to_linear(float c);
float linear_to_srgb(float l);

}  // namespace ttr::scene
