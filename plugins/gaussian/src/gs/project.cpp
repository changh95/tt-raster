// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstring>

#include "gs/gaussians.h"
#include "ttraster/parallel.h"

namespace ttr::gs {

std::string GaussianScene::validate() const {
    const size_t n = opacity.size();
    if (sh_degree > 3) return "sh_degree must be 0 to 3";
    if (xyz.size() != 3 * n || scale.size() != 3 * n || rot.size() != 4 * n || sh.size() != n * sh_coeffs() * 3) return "array sizes do not match the number of Gaussians";
    for (size_t i = 0; i < n; i++) {
        if (!(opacity[i] >= 0.0f && opacity[i] <= 1.0f)) return "opacity outside [0, 1] at Gaussian " + std::to_string(i);
        for (int k = 0; k < 3; k++) {
            if (!std::isfinite(xyz[3 * i + k])) return "non-finite position at Gaussian " + std::to_string(i);
            if (!(scale[3 * i + k] > 0.0f) || !std::isfinite(scale[3 * i + k])) return "scale must be positive and finite at Gaussian " + std::to_string(i);
        }
        const float* q = &rot[4 * i];
        const float len2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        if (!(std::fabs(len2 - 1.0f) < 2e-3f)) return "rotation is not a unit quaternion at Gaussian " + std::to_string(i);
    }
    for (float v : sh) {
        if (!std::isfinite(v)) return "non-finite spherical-harmonics coefficient";
    }
    return {};
}

Camera Camera::look_at(const float eye[3], const float target[3], const float up[3], float fov_x_degrees, uint32_t width, uint32_t height) {
    auto sub = [](const float* a, const float* b, float* o) { for (int i = 0; i < 3; i++) o[i] = a[i] - b[i]; };
    auto cross = [](const float* a, const float* b, float* o) {
        o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0];
    };
    auto norm = [](float* v) {
        const float l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        for (int i = 0; i < 3; i++) v[i] /= l;
    };
    float z[3], x[3], y[3];
    sub(target, eye, z);
    norm(z);
    cross(z, up, x);        // right = forward x up for a y-down camera frame
    norm(x);
    cross(z, x, y);         // down
    Camera c;
    const float* rows[3] = {x, y, z};
    for (int r = 0; r < 3; r++) {
        for (int k = 0; k < 3; k++) c.view[r * 4 + k] = rows[r][k];
        c.view[r * 4 + 3] = -(rows[r][0] * eye[0] + rows[r][1] * eye[1] + rows[r][2] * eye[2]);
    }
    c.view[12] = c.view[13] = c.view[14] = 0.0f;
    c.view[15] = 1.0f;
    c.width = width;
    c.height = height;
    c.fx = 0.5f * static_cast<float>(width) / std::tan(0.5f * fov_x_degrees * 3.14159265358979f / 180.0f);
    c.fy = c.fx;
    c.cx = 0.5f * static_cast<float>(width);
    c.cy = 0.5f * static_cast<float>(height);
    return c;
}

void Camera::position(float out[3]) const {
    // camera centre = -R^T t
    for (int k = 0; k < 3; k++) out[k] = -(view[0 * 4 + k] * view[0 * 4 + 3] + view[1 * 4 + k] * view[1 * 4 + 3] + view[2 * 4 + k] * view[2 * 4 + 3]);
}

namespace {
constexpr float kC0 = 0.28209479177387814f;
constexpr float kC1 = 0.4886025119029199f;
constexpr float kC2[5] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f};
constexpr float kC3[7] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};
}  // namespace

void sh_to_rgb(uint32_t degree, const float* sh, const float dir[3], float rgb[3]) {
    const float x = dir[0], y = dir[1], z = dir[2];
    for (int c = 0; c < 3; c++) {
        float v = kC0 * sh[0 * 3 + c];
        if (degree > 0) {
            v += -kC1 * y * sh[1 * 3 + c] + kC1 * z * sh[2 * 3 + c] - kC1 * x * sh[3 * 3 + c];
            if (degree > 1) {
                const float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
                v += kC2[0] * xy * sh[4 * 3 + c] + kC2[1] * yz * sh[5 * 3 + c] + kC2[2] * (2.0f * zz - xx - yy) * sh[6 * 3 + c] +
                     kC2[3] * xz * sh[7 * 3 + c] + kC2[4] * (xx - yy) * sh[8 * 3 + c];
                if (degree > 2) {
                    v += kC3[0] * y * (3.0f * xx - yy) * sh[9 * 3 + c] + kC3[1] * xy * z * sh[10 * 3 + c] +
                         kC3[2] * y * (4.0f * zz - xx - yy) * sh[11 * 3 + c] + kC3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12 * 3 + c] +
                         kC3[4] * x * (4.0f * zz - xx - yy) * sh[13 * 3 + c] + kC3[5] * z * (xx - yy) * sh[14 * 3 + c] +
                         kC3[6] * x * (xx - 3.0f * yy) * sh[15 * 3 + c];
                }
            }
        }
        rgb[c] = v + 0.5f;
    }
}

std::vector<Splat> project(const GaussianScene& scene, const Camera& cam, ProjectStats* stats, size_t threads) {
    const size_t n = scene.size();
    const uint32_t K = scene.sh_coeffs();
    const float tan_fovx = 0.5f * static_cast<float>(cam.width) / cam.fx, tan_fovy = 0.5f * static_cast<float>(cam.height) / cam.fy;
    const float limx = 1.3f * tan_fovx, limy = 1.3f * tan_fovy;
    float campos[3];
    cam.position(campos);

    // Splats are computed in blocks and concatenated in scene order, so the result does not depend on the thread count.
    constexpr size_t kBlock = 4096;
    const size_t blocks = (n + kBlock - 1) / kBlock;
    std::vector<std::vector<Splat>> parts(blocks);
    std::vector<ProjectStats> part_stats(blocks);
    parallel_for(blocks, threads, [&](size_t b) {
        std::vector<Splat>& out = parts[b];
        ProjectStats& st = part_stats[b];
        out.reserve(kBlock);
        for (size_t i = b * kBlock; i < std::min(n, (b + 1) * kBlock); i++) {
            st.input++;
            const float* p = &scene.xyz[3 * i];
            const float tx = cam.view[0] * p[0] + cam.view[1] * p[1] + cam.view[2] * p[2] + cam.view[3];
            const float ty = cam.view[4] * p[0] + cam.view[5] * p[1] + cam.view[6] * p[2] + cam.view[7];
            const float tz = cam.view[8] * p[0] + cam.view[9] * p[1] + cam.view[10] * p[2] + cam.view[11];
            if (!(tz > cam.znear)) { st.behind++; continue; }
            const float o = scene.opacity[i];
            if (o < 1.0f / 255.0f) { st.faint++; continue; }   // never reaches alpha 1/255

            // 3D covariance R S S^T R^T from the unit quaternion and the scales.
            const float* q = &scene.rot[4 * i];
            const float r = q[0], x = q[1], y = q[2], z = q[3];
            const float R[9] = {1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
                                2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                                2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)};
            const float* s = &scene.scale[3 * i];
            float M[9];   // R * diag(s)
            for (int a = 0; a < 3; a++) for (int c = 0; c < 3; c++) M[a * 3 + c] = R[a * 3 + c] * s[c];
            float cov3[9];
            for (int a = 0; a < 3; a++) for (int c = 0; c < 3; c++) cov3[a * 3 + c] = M[a * 3] * M[c * 3] + M[a * 3 + 1] * M[c * 3 + 1] + M[a * 3 + 2] * M[c * 3 + 2];

            // EWA: the projection's Jacobian at the Gaussian's centre (the direction is clamped to 1.3 times the field of view, as in the
            // reference implementation, so Gaussians near the edge do not get absurd footprints), then T = J W, cov2 = T cov3 T^T.
            const float cxz = std::clamp(tx / tz, -limx, limx) * tz, cyz = std::clamp(ty / tz, -limy, limy) * tz;
            const float J[6] = {cam.fx / tz, 0.0f, -cam.fx * cxz / (tz * tz), 0.0f, cam.fy / tz, -cam.fy * cyz / (tz * tz)};
            float T[6];
            for (int a = 0; a < 2; a++) for (int c = 0; c < 3; c++) T[a * 3 + c] = J[a * 3] * cam.view[0 * 4 + c] + J[a * 3 + 1] * cam.view[1 * 4 + c] + J[a * 3 + 2] * cam.view[2 * 4 + c];
            float TC[6];   // T cov3
            for (int a = 0; a < 2; a++) for (int c = 0; c < 3; c++) TC[a * 3 + c] = T[a * 3] * cov3[0 * 3 + c] + T[a * 3 + 1] * cov3[1 * 3 + c] + T[a * 3 + 2] * cov3[2 * 3 + c];
            float ca = TC[0] * T[0] + TC[1] * T[1] + TC[2] * T[2];
            const float cb = TC[0] * T[3] + TC[1] * T[4] + TC[2] * T[5];
            float cc = TC[3] * T[3] + TC[4] * T[4] + TC[5] * T[5];
            ca += 0.3f;   // the low-pass filter of the reference implementation: every splat covers at least about a pixel
            cc += 0.3f;
            const float det = ca * cc - cb * cb;
            if (!(det > 0.0f) || !std::isfinite(det)) { st.degenerate++; continue; }
            const float mid = 0.5f * (ca + cc);
            const float lambda1 = mid + std::sqrt(std::max(0.1f, mid * mid - det));
            // alpha = opacity * exp(power) falls below 1/255 where power < -ln(255 * opacity), i.e. at a Mahalanobis distance of
            // sqrt(2 ln(255 opacity)); the square that contains that ellipse is what the splat can touch (+1 pixel of margin).
            const float radius = std::sqrt(2.0f * std::log(255.0f * o) * lambda1) + 1.0f;

            const float mx = cam.fx * tx / tz + cam.cx, my = cam.fy * ty / tz + cam.cy;
            if (mx + radius < 0 || mx - radius > static_cast<float>(cam.width) || my + radius < 0 || my - radius > static_cast<float>(cam.height)) {
                st.off_screen++;
                continue;
            }
            Splat sp;
            sp.mx = mx; sp.my = my; sp.depth = tz;
            sp.conic[0] = cc / det; sp.conic[1] = -cb / det; sp.conic[2] = ca / det;
            sp.radius = radius;
            sp.opacity = o;
            sp.index = static_cast<uint32_t>(i);
            float dir[3] = {p[0] - campos[0], p[1] - campos[1], p[2] - campos[2]};
            const float len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
            for (float& d : dir) d = len > 0.0f ? d / len : 0.0f;
            float rgb[3];
            sh_to_rgb(scene.sh_degree, &scene.sh[i * K * 3], dir, rgb);
            for (int c = 0; c < 3; c++) sp.rgb[c] = std::max(rgb[c], 0.0f);
            out.push_back(sp);
            st.kept++;
        }
    });
    std::vector<Splat> all;
    size_t total = 0;
    for (const auto& pt : parts) total += pt.size();
    all.reserve(total);
    ProjectStats sum;
    for (size_t b = 0; b < blocks; b++) {
        all.insert(all.end(), parts[b].begin(), parts[b].end());
        sum.input += part_stats[b].input; sum.behind += part_stats[b].behind; sum.degenerate += part_stats[b].degenerate;
        sum.faint += part_stats[b].faint; sum.off_screen += part_stats[b].off_screen; sum.kept += part_stats[b].kept;
    }
    if (stats != nullptr) *stats = sum;
    return all;
}

}  // namespace ttr::gs
