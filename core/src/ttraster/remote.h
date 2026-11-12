// SPDX-License-Identifier: Apache-2.0
//
// A renderer in another process: frames go over a Unix socket to a device server (core/tools/ttr_serverd.cpp) that owns the chip, and the
// framebuffer comes back. Several client processes (Vulkan or OpenGL applications, Isaac Lab workers) then share one chip in turn, the
// server schedules them and a hung chip takes only the server down (scripts/serverd.sh resets it and starts it again). When the server
// cannot be reached, or a frame fails, the client renders that frame with the CPU reference, whose output is bit-identical to the chip's, and
// tries the server again a few seconds later: a client never fails because of the server.
//
// Selected by TTR_SERVER=<socket path> (the C API's "tensix" backend then talks to the server instead of opening a chip), or by the backend
// name "remote". The socket is a local stream socket with mode 0600: the server trusts the user that owns it, and still checks every frame it
// receives (decode_frame) so that a damaged message cannot crash it.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ttraster/backend.h"

namespace ttr::remote {

inline constexpr uint32_t kMagic = 0x53525454u;   // "TTRS"
inline constexpr uint32_t kVersion = 1;
inline constexpr uint64_t kMaxMessage = 1ull << 30;   // 1 GiB

enum MessageType : uint32_t { kHello = 1, kRender = 2, kReply = 3, kError = 4 };

struct Header {
    uint32_t magic = kMagic;
    uint32_t version = kVersion;
    uint32_t type = 0;
    uint32_t planes = 0;     // kRender: the planes the client needs back
    uint64_t length = 0;     // bytes of payload that follow
};

// Encoding of a frame / a reply. decode_frame refuses anything the rasterizer's own validation would refuse (sizes, shader programs, indices of
// triangles into the shader, attribute and viewport tables) and returns false with *error set.
std::vector<uint8_t> encode_frame(const Frame& frame);
bool decode_frame(const uint8_t* data, size_t size, Frame* frame, std::string* error);
std::vector<uint8_t> encode_reply(const Framebuffer& fb, const std::string& report);
bool decode_reply(const uint8_t* data, size_t size, Framebuffer* fb, std::string* report, std::string* error);

// Whole-buffer socket I/O (retries on EINTR); false on end of stream or error.
bool send_all(int fd, const void* data, size_t size);
bool recv_all(int fd, void* data, size_t size);
bool send_message(int fd, uint32_t type, uint32_t planes, const std::vector<uint8_t>& payload);
// Reads a header and its payload; false (with *error) on a bad header, an oversized message or a closed connection.
bool recv_message(int fd, Header* header, std::vector<uint8_t>* payload, std::string* error);

// $TTR_SERVER if set, otherwise $XDG_RUNTIME_DIR/ttraster.sock, otherwise /tmp/ttraster-<uid>.sock.
std::string default_socket_path();

}  // namespace ttr::remote

namespace ttr {

class RemoteBackend : public Backend {
public:
    // `path` empty: remote::default_socket_path(). Nothing is connected until the first frame.
    explicit RemoteBackend(std::string path = {});
    ~RemoteBackend() override;
    const Framebuffer& render(const Frame& frame, uint32_t planes, int slot = 0) override;
    std::string report(int slot = 0) const override;

    // Frames the server rendered and frames that fell back to the CPU, since the backend was created.
    uint64_t served() const { return served_; }
    uint64_t fallbacks() const { return fallbacks_; }

private:
    struct Slot {
        int fd = -1;
        Framebuffer fb;
        std::string report;
        int64_t retry_after_ms = 0;   // after a failure: do not try the server again before this time (steady clock)
    };
    bool try_remote(Slot& s, const Frame& frame, uint32_t planes);
    std::string path_;
    ReferenceBackend cpu_;
    Slot slots_[kRenderSlots];
    uint64_t served_ = 0, fallbacks_ = 0;
    mutable std::mutex stats_mutex_;
    bool warned_ = false;
};

}  // namespace ttr
