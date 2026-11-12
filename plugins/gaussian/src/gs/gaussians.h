// SPDX-License-Identifier: Apache-2.0
//
// 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) on top of the renderer core: the scene (what a trained 3DGS model is), the camera,
// the screen-space splats a camera sees, and the renderers. Conventions: world to camera is a 4x4 row-major matrix, the camera looks along
// +z with x to the right and y down (OpenCV), pixel (i, j) has its centre at (i + 0.5, j + 0.5) with the origin at the top left, like
// everywhere else in tt-raster. Opacities are in [0, 1], scales are linear (not log), rotations are unit quaternions (w, x, y, z); the PLY
// loader converts the logit, log and unnormalised forms a trained model is stored in.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ttr::gs {

struct GaussianScene {
    uint32_t sh_degree = 0;          // 0 .. 3
    std::vector<float> xyz;          // 3 per Gaussian: world position
    std::vector<float> scale;        // 3 per Gaussian: standard deviations along the Gaussian's own axes
    std::vector<float> rot;          // 4 per Gaussian: unit quaternion (w, x, y, z)
    std::vector<float> opacity;      // 1 per Gaussian
    std::vector<float> sh;           // (sh_degree + 1)^2 coefficients of 3 channels per Gaussian: ((n * K + k) * 3 + c)

    size_t size() const { return opacity.size(); }
    uint32_t sh_coeffs() const { return (sh_degree + 1) * (sh_degree + 1); }
    // Empty if consistent (array sizes, degree, finite values, unit quaternions within 1e-3, positive scales, opacity in [0, 1]).
    std::string validate() const;
};

struct Camera {
    float view[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};   // world to camera, row-major
    float fx = 0, fy = 0, cx = 0, cy = 0;                                    // pinhole intrinsics in pixels
    uint32_t width = 0, height = 0;
    float znear = 0.2f;                                                      // Gaussians closer than this (camera z) are not drawn

    // Camera at `eye` looking at `target`; horizontal field of view in degrees, square pixels, principal point in the middle.
    static Camera look_at(const float eye[3], const float target[3], const float up[3], float fov_x_degrees, uint32_t width, uint32_t height);
    void position(float out[3]) const;   // the camera centre in the world
};

// One Gaussian as the camera sees it: a 2D Gaussian on the screen with a colour and an opacity.
struct Splat {
    float mx, my;          // centre in pixels
    float depth;           // camera z: the sort key
    float conic[3];        // inverse of the 2D covariance: a, b, c of [[a, b], [b, c]]
    float radius;          // half the side of the square (in pixels) outside of which the splat's alpha is below 1/255
    float rgb[3];          // colour from the spherical harmonics, at least 0
    float opacity;
    uint32_t index;        // the Gaussian's index in the scene
};

struct ProjectStats {
    size_t input = 0, behind = 0, degenerate = 0, faint = 0, off_screen = 0, kept = 0;
};

// Splats of all Gaussians the camera sees, in scene order. EWA projection with the 0.3 pixel dilation of the reference implementation.
std::vector<Splat> project(const GaussianScene& scene, const Camera& camera, ProjectStats* stats = nullptr, size_t threads = 0);

// The spherical-harmonics colour of coefficients `sh` ((degree + 1)^2 x 3, as in GaussianScene) seen along the unit direction `dir`
// (from the camera to the Gaussian), plus 0.5, not yet clamped. Exposed for tests.
void sh_to_rgb(uint32_t degree, const float* sh, const float dir[3], float rgb[3]);

}  // namespace ttr::gs
