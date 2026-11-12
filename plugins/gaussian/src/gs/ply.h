// SPDX-License-Identifier: Apache-2.0
//
// PLY files of trained 3D Gaussian Splatting models (the layout of the reference implementation's `point_cloud.ply`: binary little endian,
// per vertex x y z, nx ny nz, f_dc_0..2, f_rest_0..(3 (K - 1) - 1), opacity (logit), scale_0..2 (log), rot_0..3 (w, x, y, z, not unit)),
// and a generator of synthetic scenes for tests and benchmarks.

#pragma once

#include <string>

#include "gs/gaussians.h"

namespace ttr::gs {

// Reads a 3DGS PLY (any property order and any scalar type, unknown properties are skipped). Converts the stored forms: opacity through the
// sigmoid, scales through exp, quaternions normalised, f_rest from the channel-major layout. Returns false with *error set on failure.
bool read_ply(const std::string& path, GaussianScene* scene, std::string* error);
// Writes the layout above (normals zero), so that read_ply(write_ply(s)) gives s back up to the float round trip of logit and log.
bool write_ply(const std::string& path, const GaussianScene& scene, std::string* error);

// A reproducible scene of `n` Gaussians: a few blobs and a ground disc inside a sphere of radius about 2 around the origin, scales of
// 1 to 6 cm, random orientations, opacities 0.2 to 0.95, smooth colours with view-dependent terms up to `sh_degree`.
GaussianScene make_synthetic_scene(size_t n, uint32_t seed, uint32_t sh_degree = 3);

// A recognisable scene for pictures: a chequered ground, a striped sphere and a torus, each surface covered with flat Gaussians (disc-shaped,
// oriented along the surface normal), colours lit by a fixed light, a little view-dependent shine in the degree-1 terms.
GaussianScene make_demo_scene(size_t n, uint32_t seed);

}  // namespace ttr::gs
