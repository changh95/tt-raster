// SPDX-License-Identifier: Apache-2.0

#include "gs/ply.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <vector>

namespace ttr::gs {

namespace {

struct Prop {
    std::string name;
    size_t size = 4;
    char kind = 'f';   // f float, i signed int, u unsigned int
};

bool type_info(const std::string& t, size_t* size, char* kind) {
    static const std::map<std::string, std::pair<size_t, char>> types = {
        {"float", {4, 'f'}}, {"float32", {4, 'f'}}, {"double", {8, 'f'}}, {"float64", {8, 'f'}}, {"char", {1, 'i'}}, {"int8", {1, 'i'}},
        {"uchar", {1, 'u'}}, {"uint8", {1, 'u'}}, {"short", {2, 'i'}}, {"int16", {2, 'i'}}, {"ushort", {2, 'u'}}, {"uint16", {2, 'u'}},
        {"int", {4, 'i'}}, {"int32", {4, 'i'}}, {"uint", {4, 'u'}}, {"uint32", {4, 'u'}}};
    auto it = types.find(t);
    if (it == types.end()) return false;
    *size = it->second.first;
    *kind = it->second.second;
    return true;
}

double read_value(const uint8_t* p, const Prop& pr) {
    switch (pr.kind) {
        case 'f': if (pr.size == 4) { float v; std::memcpy(&v, p, 4); return v; } else { double v; std::memcpy(&v, p, 8); return v; }
        case 'i': if (pr.size == 1) return static_cast<int8_t>(*p); if (pr.size == 2) { int16_t v; std::memcpy(&v, p, 2); return v; } else { int32_t v; std::memcpy(&v, p, 4); return v; }
        default: if (pr.size == 1) return *p; if (pr.size == 2) { uint16_t v; std::memcpy(&v, p, 2); return v; } else { uint32_t v; std::memcpy(&v, p, 4); return v; }
    }
}

bool fail(std::string* e, const std::string& why) {
    if (e != nullptr) *e = why;
    return false;
}

}  // namespace

bool read_ply(const std::string& path, GaussianScene* out, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail(error, "cannot open " + path);
    std::string line;
    std::getline(in, line);
    if (line.rfind("ply", 0) != 0) return fail(error, path + " is not a PLY file");
    size_t n = 0;
    bool binary = false, in_vertex = false;
    std::vector<Prop> props;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ls(line);
        std::string w;
        ls >> w;
        if (w == "format") {
            std::string f;
            ls >> f;
            if (f != "binary_little_endian") return fail(error, "only binary_little_endian PLY files are supported (this one is " + f + ")");
            binary = true;
        } else if (w == "element") {
            std::string name;
            size_t count = 0;
            ls >> name >> count;
            in_vertex = name == "vertex";
            if (in_vertex) n = count;
            else if (count != 0) return fail(error, "unexpected element '" + name + "' (a 3DGS file has only vertices)");
        } else if (w == "property" && in_vertex) {
            std::string type, name;
            ls >> type >> name;
            Prop p;
            p.name = name;
            if (!type_info(type, &p.size, &p.kind)) return fail(error, "unsupported property type '" + type + "'");
            props.push_back(p);
        } else if (w == "end_header") {
            break;
        }
    }
    if (!binary) return fail(error, "no format line");
    std::map<std::string, size_t> idx;
    std::vector<size_t> offset(props.size());
    size_t stride = 0;
    for (size_t i = 0; i < props.size(); i++) {
        idx[props[i].name] = i;
        offset[i] = stride;
        stride += props[i].size;
    }
    for (const char* need : {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}) {
        if (idx.count(need) == 0) return fail(error, std::string("missing property ") + need);
    }
    size_t n_rest = 0;
    while (idx.count("f_rest_" + std::to_string(n_rest)) != 0) n_rest++;
    uint32_t degree = 0;
    if (n_rest == 0) degree = 0;
    else if (n_rest == 9) degree = 1;
    else if (n_rest == 24) degree = 2;
    else if (n_rest == 45) degree = 3;
    else return fail(error, "f_rest has " + std::to_string(n_rest) + " components; expected 0, 9, 24 or 45");
    const uint32_t K = (degree + 1) * (degree + 1);

    std::vector<uint8_t> row(stride);
    GaussianScene s;
    s.sh_degree = degree;
    s.xyz.resize(3 * n); s.scale.resize(3 * n); s.rot.resize(4 * n); s.opacity.resize(n); s.sh.assign(n * K * 3, 0.0f);
    auto val = [&](const char* name) { return read_value(row.data() + offset[idx[name]], props[idx[name]]); };
    for (size_t i = 0; i < n; i++) {
        if (!in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(stride))) return fail(error, "file ends inside the vertex data");
        s.xyz[3 * i] = static_cast<float>(val("x")); s.xyz[3 * i + 1] = static_cast<float>(val("y")); s.xyz[3 * i + 2] = static_cast<float>(val("z"));
        s.opacity[i] = static_cast<float>(1.0 / (1.0 + std::exp(-val("opacity"))));
        for (int k = 0; k < 3; k++) s.scale[3 * i + k] = static_cast<float>(std::exp(val(("scale_" + std::to_string(k)).c_str())));
        double q[4], len = 0;
        for (int k = 0; k < 4; k++) { q[k] = val(("rot_" + std::to_string(k)).c_str()); len += q[k] * q[k]; }
        len = std::sqrt(len);
        for (int k = 0; k < 4; k++) s.rot[4 * i + k] = len > 0 ? static_cast<float>(q[k] / len) : (k == 0 ? 1.0f : 0.0f);
        for (int c = 0; c < 3; c++) s.sh[(i * K + 0) * 3 + c] = static_cast<float>(val(("f_dc_" + std::to_string(c)).c_str()));
        for (int c = 0; c < 3; c++) {
            for (uint32_t k = 1; k < K; k++) s.sh[(i * K + k) * 3 + c] = static_cast<float>(val(("f_rest_" + std::to_string(c * (K - 1) + (k - 1))).c_str()));
        }
    }
    *out = std::move(s);
    return true;
}

bool write_ply(const std::string& path, const GaussianScene& s, std::string* error) {
    const std::string why = s.validate();
    if (!why.empty()) return fail(error, why);
    std::ofstream out(path, std::ios::binary);
    if (!out) return fail(error, "cannot create " + path);
    const size_t n = s.size();
    const uint32_t K = s.sh_coeffs();
    out << "ply\nformat binary_little_endian 1.0\nelement vertex " << n << "\n";
    for (const char* p : {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"}) out << "property float " << p << "\n";
    for (uint32_t i = 0; i < 3 * (K - 1); i++) out << "property float f_rest_" << i << "\n";
    for (const char* p : {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}) out << "property float " << p << "\n";
    out << "end_header\n";
    std::vector<float> row;
    for (size_t i = 0; i < n; i++) {
        row.clear();
        for (int k = 0; k < 3; k++) row.push_back(s.xyz[3 * i + k]);
        for (int k = 0; k < 3; k++) row.push_back(0.0f);
        for (int c = 0; c < 3; c++) row.push_back(s.sh[(i * K) * 3 + c]);
        for (int c = 0; c < 3; c++) for (uint32_t k = 1; k < K; k++) row.push_back(s.sh[(i * K + k) * 3 + c]);
        const float o = std::clamp(s.opacity[i], 1e-6f, 1.0f - 1e-6f);
        row.push_back(std::log(o / (1.0f - o)));
        for (int k = 0; k < 3; k++) row.push_back(std::log(s.scale[3 * i + k]));
        for (int k = 0; k < 4; k++) row.push_back(s.rot[4 * i + k]);
        out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
    }
    return out.good() ? true : fail(error, "write failed");
}

GaussianScene make_synthetic_scene(size_t n, uint32_t seed, uint32_t sh_degree) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f), u11(-1.0f, 1.0f);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    GaussianScene s;
    s.sh_degree = sh_degree;
    const uint32_t K = s.sh_coeffs();
    s.xyz.resize(3 * n); s.scale.resize(3 * n); s.rot.resize(4 * n); s.opacity.resize(n); s.sh.assign(n * K * 3, 0.0f);
    // blobs: centre, spread, base colour
    struct Blob { float c[3], spread, col[3]; };
    std::vector<Blob> blobs(5);
    for (Blob& b : blobs) {
        for (float& v : b.c) v = 0.9f * u11(rng);
        b.spread = 0.25f + 0.35f * u01(rng);
        for (float& v : b.col) v = 0.15f + 0.7f * u01(rng);
    }
    for (size_t i = 0; i < n; i++) {
        float col[3];
        if (u01(rng) < 0.25f) {   // ground disc
            const float a = 6.2831853f * u01(rng), r = 2.0f * std::sqrt(u01(rng));
            s.xyz[3 * i] = r * std::cos(a); s.xyz[3 * i + 1] = -0.9f + 0.02f * gauss(rng); s.xyz[3 * i + 2] = r * std::sin(a);
            const float checker = ((static_cast<int>(std::floor(s.xyz[3 * i] * 2.0f)) + static_cast<int>(std::floor(s.xyz[3 * i + 2] * 2.0f))) & 1) ? 0.75f : 0.3f;
            col[0] = col[1] = col[2] = checker;
        } else {
            const Blob& b = blobs[rng() % blobs.size()];
            for (int k = 0; k < 3; k++) s.xyz[3 * i + k] = b.c[k] + b.spread * gauss(rng);
            for (int k = 0; k < 3; k++) col[k] = std::clamp(b.col[k] + 0.1f * gauss(rng), 0.0f, 1.0f);
        }
        const float base = 0.01f + 0.05f * u01(rng);
        for (int k = 0; k < 3; k++) s.scale[3 * i + k] = base * std::exp(0.6f * gauss(rng));
        float q[4], len = 0;
        for (float& v : q) { v = gauss(rng); len += v * v; }
        len = std::sqrt(len);
        for (int k = 0; k < 4; k++) s.rot[4 * i + k] = q[k] / len;
        s.opacity[i] = 0.2f + 0.75f * u01(rng);
        for (int c = 0; c < 3; c++) {
            s.sh[(i * K) * 3 + c] = (col[c] - 0.5f) / 0.28209479177387814f;   // degree-0 term gives exactly this colour
            for (uint32_t k = 1; k < K; k++) s.sh[(i * K + k) * 3 + c] = 0.12f * gauss(rng) / static_cast<float>(1 + (k >= 4) + (k >= 9));
        }
    }
    return s;
}

GaussianScene make_demo_scene(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    GaussianScene s;
    s.sh_degree = 1;
    const uint32_t K = 4;
    s.xyz.resize(3 * n); s.scale.resize(3 * n); s.rot.resize(4 * n); s.opacity.resize(n); s.sh.assign(n * K * 3, 0.0f);
    const float light[3] = {0.5f, 0.8f, 0.33f};
    const float ll = std::sqrt(light[0] * light[0] + light[1] * light[1] + light[2] * light[2]);
    // area-proportional share of the points: ground 4.5, sphere 2.5, torus 3
    for (size_t i = 0; i < n; i++) {
        float p[3], nrm[3], col[3];
        const float pick = u01(rng) * 10.0f;
        if (pick < 4.5f) {
            p[0] = 4.0f * (u01(rng) - 0.5f) * 1.5f; p[1] = -1.0f; p[2] = 4.0f * (u01(rng) - 0.5f) * 1.5f;
            nrm[0] = 0; nrm[1] = 1; nrm[2] = 0;
            const bool dark = ((static_cast<int>(std::floor(p[0] * 1.5f)) + static_cast<int>(std::floor(p[2] * 1.5f))) & 1) != 0;
            col[0] = dark ? 0.18f : 0.8f; col[1] = dark ? 0.2f : 0.78f; col[2] = dark ? 0.26f : 0.7f;
        } else if (pick < 7.0f) {
            const float z = 2.0f * u01(rng) - 1.0f, a = 6.2831853f * u01(rng), r = std::sqrt(1 - z * z);
            nrm[0] = r * std::cos(a); nrm[1] = z; nrm[2] = r * std::sin(a);
            for (int k = 0; k < 3; k++) p[k] = 0.8f * nrm[k];
            p[1] += -0.2f; p[0] -= 0.7f;
            const bool stripe = (static_cast<int>(std::floor(z * 5.0f)) & 1) != 0;
            col[0] = stripe ? 0.85f : 0.9f; col[1] = stripe ? 0.2f : 0.85f; col[2] = stripe ? 0.2f : 0.8f;
        } else {
            const float a = 6.2831853f * u01(rng), b = 6.2831853f * u01(rng), R = 0.75f, r = 0.28f;
            nrm[0] = std::cos(b) * std::cos(a); nrm[1] = std::sin(b); nrm[2] = std::cos(b) * std::sin(a);
            p[0] = (R + r * std::cos(b)) * std::cos(a) + 0.9f; p[1] = r * std::sin(b) - 0.55f; p[2] = (R + r * std::cos(b)) * std::sin(a);
            // tilt the torus a little so that it stands on its edge
            const float t = 0.9f, y = p[1] + 0.55f;
            const float ny = nrm[1] * std::cos(t) - nrm[2] * std::sin(t), nz = nrm[1] * std::sin(t) + nrm[2] * std::cos(t);
            const float py = y * std::cos(t) - p[2] * std::sin(t), pz = y * std::sin(t) + p[2] * std::cos(t);
            nrm[1] = ny; nrm[2] = nz; p[1] = py - 0.2f; p[2] = pz;
            col[0] = 0.25f + 0.6f * (0.5f + 0.5f * std::cos(a * 3)); col[1] = 0.55f; col[2] = 0.9f - 0.5f * (0.5f + 0.5f * std::cos(a * 3));
        }
        const float lambert = std::max(0.0f, (nrm[0] * light[0] + nrm[1] * light[1] + nrm[2] * light[2]) / ll);
        const float shade = 0.3f + 0.75f * lambert;
        for (int k = 0; k < 3; k++) { s.xyz[3 * i + k] = p[k]; col[k] = std::min(1.0f, col[k] * shade); }
        const float sz = 0.028f * (0.7f + 0.6f * u01(rng));
        s.scale[3 * i] = sz; s.scale[3 * i + 1] = sz; s.scale[3 * i + 2] = 0.15f * sz;
        // rotation taking the z axis to the normal
        float q[4] = {1.0f + nrm[2], -nrm[1], nrm[0], 0.0f};
        if (1.0f + nrm[2] < 1e-5f) { q[0] = 0; q[1] = 1; q[2] = 0; q[3] = 0; }
        const float ql = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        for (int k = 0; k < 4; k++) s.rot[4 * i + k] = q[k] / ql;
        s.opacity[i] = 0.9f;
        for (int c = 0; c < 3; c++) {
            s.sh[(i * K) * 3 + c] = (col[c] - 0.5f) / 0.28209479177387814f;
            // a little shine: brighter when seen along the light's mirror direction (a linear term in the view direction)
            s.sh[(i * K + 2) * 3 + c] = 0.15f * lambert * col[c];
        }
    }
    return s;
}

}  // namespace ttr::gs
