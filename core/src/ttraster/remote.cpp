// SPDX-License-Identifier: Apache-2.0

#include "ttraster/remote.h"

#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ttr::remote {

namespace {

struct Writer {
    std::vector<uint8_t> buf;
    template <class T> void put(T v) {
        const size_t n = buf.size();
        buf.resize(n + sizeof(T));
        std::memcpy(buf.data() + n, &v, sizeof(T));
    }
    template <class T> void put_array(const T* p, size_t count) {
        const size_t n = buf.size();
        buf.resize(n + sizeof(T) * count);
        if (count != 0) std::memcpy(buf.data() + n, p, sizeof(T) * count);
    }
};

struct Reader {
    const uint8_t* p;
    size_t left;
    bool ok = true;
    Reader(const uint8_t* data, size_t size) : p(data), left(size) {}
    template <class T> T get() {
        T v{};
        if (!ok || left < sizeof(T)) { ok = false; return v; }
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        left -= sizeof(T);
        return v;
    }
    // A count of elements of `elem` bytes each that must still fit in the message (so a damaged count cannot make us allocate).
    uint64_t count(size_t elem) {
        const uint64_t n = get<uint64_t>();
        if (!ok || n > left / elem) { ok = false; return 0; }
        return n;
    }
    template <class T> void get_array(T* out, size_t n) {
        if (!ok || left / sizeof(T) < n) { ok = false; return; }
        if (n != 0) std::memcpy(out, p, sizeof(T) * n);
        p += sizeof(T) * n;
        left -= sizeof(T) * n;
    }
};

constexpr uint32_t kMaxSide = 16384;
constexpr uint64_t kMaxPixels = 1ull << 26;

}  // namespace

std::vector<uint8_t> encode_frame(const Frame& f) {
    Writer w;
    w.buf.reserve(64 + f.triangles.size() * 112 + f.attrs.size() * 4);
    w.put<uint32_t>(f.width);
    w.put<uint32_t>(f.height);
    w.put_array(f.clear_color, 4);
    w.put<float>(f.clear_depth);
    w.put<uint8_t>(static_cast<uint8_t>(f.depth_func));
    w.put<uint8_t>(static_cast<uint8_t>(f.cull));
    w.put<uint64_t>(f.triangles.size());
    for (const Triangle& t : f.triangles) {
        for (const Vertex& v : t.v) {
            w.put<float>(v.x); w.put<float>(v.y); w.put<float>(v.z); w.put<float>(v.rhw);
            w.put_array(v.color, 4);
        }
        w.put<uint32_t>(t.viewport);
        w.put<uint8_t>(static_cast<uint8_t>((t.flat ? 1 : 0) | (t.perspective ? 2 : 0) | (t.own_state ? 4 : 0) | (t.depth_write ? 8 : 0)));
        w.put<uint8_t>(static_cast<uint8_t>(t.depth_func));
        w.put<uint8_t>(static_cast<uint8_t>(t.blend));
        w.put<uint16_t>(t.shader);
        w.put<uint32_t>(t.attr_offset);
    }
    w.put<uint64_t>(f.viewports.size());
    for (const Viewport& v : f.viewports) {
        w.put<int32_t>(v.x); w.put<int32_t>(v.y); w.put<int32_t>(v.width); w.put<int32_t>(v.height);
    }
    w.put<uint64_t>(f.shaders.size());
    for (const ShaderProgram& p : f.shaders) {
        w.put<uint32_t>(p.num_inputs);
        w.put<uint32_t>(p.num_regs);
        w.put<uint32_t>(p.out_reg);
        w.put<int32_t>(p.fragcoord_reg);
        w.put<uint64_t>(p.code.size());
        for (const ShaderInstr& in : p.code) w.put_array(in.w, shader::kInstrWords);
        w.put<uint64_t>(p.consts.size());
        w.put_array(p.consts.data(), p.consts.size());
    }
    w.put<uint64_t>(f.attrs.size());
    w.put_array(f.attrs.data(), f.attrs.size());
    return std::move(w.buf);
}

bool decode_frame(const uint8_t* data, size_t size, Frame* out, std::string* error) {
    auto fail = [&](const char* why) {
        if (error != nullptr) *error = why;
        return false;
    };
    Reader r(data, size);
    Frame f;
    f.width = r.get<uint32_t>();
    f.height = r.get<uint32_t>();
    r.get_array(f.clear_color, 4);
    f.clear_depth = r.get<float>();
    const uint8_t depth_func = r.get<uint8_t>(), cull = r.get<uint8_t>();
    if (!r.ok) return fail("truncated frame header");
    if (f.width == 0 || f.height == 0 || f.width > kMaxSide || f.height > kMaxSide || static_cast<uint64_t>(f.width) * f.height > kMaxPixels)
        return fail("frame size out of range");
    if (depth_func > 2 || cull > 2) return fail("bad depth function or cull mode");
    f.depth_func = static_cast<DepthFunc>(depth_func);
    f.cull = static_cast<CullMode>(cull);

    constexpr size_t kTriBytes = 3 * 8 * 4 + 4 + 1 + 1 + 1 + 2 + 4;
    const uint64_t nt = r.count(kTriBytes);
    if (!r.ok) return fail("bad triangle count");
    f.triangles.resize(nt);
    for (Triangle& t : f.triangles) {
        for (Vertex& v : t.v) {
            v.x = r.get<float>(); v.y = r.get<float>(); v.z = r.get<float>(); v.rhw = r.get<float>();
            r.get_array(v.color, 4);
        }
        t.viewport = r.get<uint32_t>();
        const uint8_t flags = r.get<uint8_t>();
        const uint8_t tdf = r.get<uint8_t>(), blend = r.get<uint8_t>();
        t.shader = r.get<uint16_t>();
        t.attr_offset = r.get<uint32_t>();
        if (!r.ok) return fail("truncated triangles");
        if ((flags & ~15u) != 0 || tdf > 2 || blend > static_cast<uint8_t>(BlendMode::AlphaAdd)) return fail("bad triangle state");
        t.flat = (flags & 1) != 0; t.perspective = (flags & 2) != 0; t.own_state = (flags & 4) != 0; t.depth_write = (flags & 8) != 0;
        t.depth_func = static_cast<DepthFunc>(tdf);
        t.blend = static_cast<BlendMode>(blend);
    }
    const uint64_t nv = r.count(16);
    if (!r.ok) return fail("bad viewport count");
    f.viewports.resize(nv);
    for (Viewport& v : f.viewports) {
        v.x = r.get<int32_t>(); v.y = r.get<int32_t>(); v.width = r.get<int32_t>(); v.height = r.get<int32_t>();
        if (!r.ok) return fail("truncated viewports");
        if (v.x < 0 || v.y < 0 || v.width <= 0 || v.height <= 0 || v.width > kMaxTargetSize || v.height > kMaxTargetSize ||
            static_cast<int64_t>(v.x) + v.width > f.padded_width() || static_cast<int64_t>(v.y) + v.height > f.padded_height())
            return fail("viewport outside the target");
    }
    const uint64_t ns = r.count(4 * 4 + 8 + 8);
    if (!r.ok || ns > 65535) return fail("bad shader count");
    f.shaders.resize(ns);
    for (ShaderProgram& p : f.shaders) {
        p.num_inputs = r.get<uint32_t>();
        p.num_regs = r.get<uint32_t>();
        p.out_reg = r.get<uint32_t>();
        p.fragcoord_reg = r.get<int32_t>();
        const uint64_t nc = r.count(shader::kInstrWords * 4);
        if (!r.ok) return fail("bad program length");
        p.code.resize(nc);
        for (ShaderInstr& in : p.code) r.get_array(in.w, shader::kInstrWords);
        const uint64_t nk = r.count(4);
        if (!r.ok) return fail("bad constant count");
        p.consts.resize(nk);
        r.get_array(p.consts.data(), nk);
        if (!r.ok) return fail("truncated program");
        const std::string why = validate_shader(p);
        if (!why.empty()) {
            if (error != nullptr) *error = "shader program refused: " + why;
            return false;
        }
    }
    const uint64_t na = r.count(4);
    if (!r.ok) return fail("bad attribute count");
    f.attrs.resize(na);
    r.get_array(f.attrs.data(), na);
    if (!r.ok || r.left != 0) return fail("trailing or missing bytes");
    for (const Triangle& t : f.triangles) {
        if (t.shader > f.shaders.size()) return fail("triangle names a shader that does not exist");
        if (t.shader != 0 && static_cast<uint64_t>(t.attr_offset) + 3ull * f.shaders[t.shader - 1].num_inputs > f.attrs.size())
            return fail("triangle varyings outside the attribute table");
        if (!f.viewports.empty() && t.viewport >= f.viewports.size()) return fail("triangle names a viewport that does not exist");
    }
    *out = std::move(f);
    return true;
}

std::vector<uint8_t> encode_reply(const Framebuffer& fb, const std::string& report) {
    Writer w;
    w.put<uint32_t>(fb.width);
    w.put<uint32_t>(fb.height);
    auto put = [&w](const auto& v) {
        w.put<uint64_t>(v.size());
        w.put_array(v.data(), v.size());
    };
    put(fb.color); put(fb.depth); put(fb.id); put(fb.cover); put(fb.ambiguous);
    for (const auto& p : fb.plane) put(p);
    w.put<uint32_t>(static_cast<uint32_t>(report.size()));
    w.put_array(report.data(), report.size());
    return std::move(w.buf);
}

bool decode_reply(const uint8_t* data, size_t size, Framebuffer* out, std::string* report, std::string* error) {
    auto fail = [&](const char* why) {
        if (error != nullptr) *error = why;
        return false;
    };
    Reader r(data, size);
    Framebuffer fb;
    fb.width = r.get<uint32_t>();
    fb.height = r.get<uint32_t>();
    if (!r.ok || fb.width > kMaxSide || fb.height > kMaxSide || static_cast<uint64_t>(fb.width) * fb.height > kMaxPixels) return fail("bad framebuffer size");
    const uint64_t pixels = static_cast<uint64_t>(fb.width) * fb.height;
    bool good = true;
    auto get = [&](auto& v) {
        using E = typename std::remove_reference_t<decltype(v)>::value_type;
        const uint64_t n = r.count(sizeof(E));
        if (!r.ok || (n != 0 && n != pixels)) { good = false; return; }
        v.resize(n);
        r.get_array(v.data(), n);
    };
    get(fb.color); get(fb.depth); get(fb.id); get(fb.cover); get(fb.ambiguous);
    for (auto& p : fb.plane) get(p);
    const uint32_t rl = r.get<uint32_t>();
    std::string rep;
    if (r.ok && rl <= r.left) {
        rep.assign(reinterpret_cast<const char*>(r.p), rl);
        r.p += rl;
        r.left -= rl;
    } else {
        r.ok = false;
    }
    if (!r.ok || !good || r.left != 0) return fail("malformed reply");
    *out = std::move(fb);
    if (report != nullptr) *report = std::move(rep);
    return true;
}

bool send_all(int fd, const void* data, size_t size) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    while (size > 0) {
        const ssize_t n = ::send(fd, p, size, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += n;
        size -= static_cast<size_t>(n);
    }
    return true;
}

bool recv_all(int fd, void* data, size_t size) {
    uint8_t* p = static_cast<uint8_t*>(data);
    while (size > 0) {
        const ssize_t n = ::recv(fd, p, size, 0);
        if (n == 0) return false;
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += n;
        size -= static_cast<size_t>(n);
    }
    return true;
}

bool send_message(int fd, uint32_t type, uint32_t planes, const std::vector<uint8_t>& payload) {
    Header h;
    h.type = type;
    h.planes = planes;
    h.length = payload.size();
    return send_all(fd, &h, sizeof h) && send_all(fd, payload.data(), payload.size());
}

bool recv_message(int fd, Header* h, std::vector<uint8_t>* payload, std::string* error) {
    if (!recv_all(fd, h, sizeof *h)) {
        if (error != nullptr) *error = "connection closed";
        return false;
    }
    if (h->magic != kMagic || h->version != kVersion) {
        if (error != nullptr) *error = "not a ttraster message (or another protocol version)";
        return false;
    }
    if (h->length > kMaxMessage) {
        if (error != nullptr) *error = "message too large";
        return false;
    }
    payload->resize(h->length);
    if (!recv_all(fd, payload->data(), payload->size())) {
        if (error != nullptr) *error = "connection closed inside a message";
        return false;
    }
    return true;
}

std::string default_socket_path() {
    if (const char* p = std::getenv("TTR_SERVER"); p != nullptr && p[0] != '\0') return p;
    if (const char* x = std::getenv("XDG_RUNTIME_DIR"); x != nullptr && x[0] != '\0') return std::string(x) + "/ttraster.sock";
    return "/tmp/ttraster-" + std::to_string(::getuid()) + ".sock";
}

}  // namespace ttr::remote

namespace ttr {

namespace {
int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

RemoteBackend::RemoteBackend(std::string path) : path_(path.empty() ? remote::default_socket_path() : std::move(path)) {}

RemoteBackend::~RemoteBackend() {
    for (Slot& s : slots_) {
        if (s.fd >= 0) ::close(s.fd);
    }
}

bool RemoteBackend::try_remote(Slot& s, const Frame& frame, uint32_t planes) {
    auto drop = [&](const std::string& why) {
        if (s.fd >= 0) ::close(s.fd);
        s.fd = -1;
        long retry_ms = 5000;   // after a failure the server is tried again this much later (TTR_SERVER_RETRY_MS)
        if (const char* r = std::getenv("TTR_SERVER_RETRY_MS")) retry_ms = std::strtol(r, nullptr, 10);
        s.retry_after_ms = now_ms() + retry_ms;
        std::lock_guard<std::mutex> lock(stats_mutex_);
        if (!warned_) {
            warned_ = true;
            std::fprintf(stderr, "ttraster: device server %s unavailable (%s); rendering on the CPU and trying the server again later\n", path_.c_str(),
                         why.c_str());
        }
        return false;
    };
    if (s.fd < 0) {
        if (now_ms() < s.retry_after_ms) return false;
        const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (fd < 0) return drop("socket");
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        if (path_.size() >= sizeof addr.sun_path) { ::close(fd); return drop("socket path too long"); }
        std::memcpy(addr.sun_path, path_.c_str(), path_.size() + 1);
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
            const std::string why = std::strerror(errno);
            ::close(fd);
            return drop(why);
        }
        // A server that hangs on a frame is killed by its own watchdog; this is only the last resort on the client side.
        long timeout_ms = 120000;
        if (const char* t = std::getenv("TTR_SERVER_TIMEOUT_MS")) timeout_ms = std::strtol(t, nullptr, 10);
        timeval tv{static_cast<time_t>(timeout_ms / 1000), static_cast<suseconds_t>((timeout_ms % 1000) * 1000)};
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        s.fd = fd;
        remote::Header h;
        std::vector<uint8_t> reply;
        std::string err;
        if (!remote::send_message(fd, remote::kHello, 0, {}) || !remote::recv_message(fd, &h, &reply, &err) || h.type != remote::kReply)
            return drop(err.empty() ? "no answer to the hello message" : err);
    }
    if (!remote::send_message(s.fd, remote::kRender, planes, remote::encode_frame(frame))) return drop("connection lost while sending a frame");
    remote::Header h;
    std::vector<uint8_t> payload;
    std::string err;
    if (!remote::recv_message(s.fd, &h, &payload, &err)) return drop(err);
    if (h.type == remote::kError) {
        // The server refused this frame (it stays connected): render this one on the CPU.
        std::lock_guard<std::mutex> lock(stats_mutex_);
        std::fprintf(stderr, "ttraster: device server refused a frame (%.*s); rendering it on the CPU\n", static_cast<int>(payload.size()),
                     reinterpret_cast<const char*>(payload.data()));
        return false;
    }
    if (h.type != remote::kReply || !remote::decode_reply(payload.data(), payload.size(), &s.fb, &s.report, &err)) return drop(err.empty() ? "unexpected reply" : err);
    return true;
}

const Framebuffer& RemoteBackend::render(const Frame& frame, uint32_t planes, int slot) {
    Slot& s = slots_[slot];
    if (try_remote(s, frame, planes)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        served_++;
        return s.fb;
    }
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        fallbacks_++;
    }
    s.report = "cpu fallback";
    s.fb = cpu_.render(frame, planes, slot);
    return s.fb;
}

std::string RemoteBackend::report(int slot) const { return slots_[slot].report; }

}  // namespace ttr
