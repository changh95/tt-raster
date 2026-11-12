// SPDX-License-Identifier: Apache-2.0
//
// The device server: owns the chip (or chips) and renders the frames that client processes send over a Unix socket (src/ttraster/remote.h).
// Clients that set TTR_SERVER=<socket> use the C API as always and never open a device themselves, so any number of processes share one
// chip in turn; two frames overlap their host work and take the chip one after the other (the renderer's frame slots).
//
//   scripts/run.sh 5 build/core/tools/ttr_serverd --socket /tmp/ttraster.sock            # on a chip (use scripts/serverd.sh to supervise)
//   build/core/tools/ttr_serverd --backend reference --socket /tmp/ttraster.sock          # CPU reference, no device (tests)
//
// A frame that takes longer than --frame-timeout seconds means a hung chip: the server exits with status 3 (its clients finish the frame on
// the CPU and reconnect later) and scripts/serverd.sh resets the chip and starts a new server. SIGINT and SIGTERM stop it cleanly (status 0).

#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "ttraster/backend.h"
#include "ttraster/remote.h"
#ifdef TT_RASTER_WITH_TENSIX
#include "tensix/tensix_backend.h"
#endif

using namespace ttr;
using Clock = std::chrono::steady_clock;

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

int64_t now_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count(); }

struct Server {
    std::unique_ptr<Backend> backend;
    std::string backend_name;
    size_t chips = 1;
    double frame_timeout_s = 30;
    bool verbose = false;
    long hang_after = -1;                       // test hook (TTR_SERVERD_TEST_HANG_AFTER): the frame after this many rendered ones never returns

    std::mutex slot_mutex;
    std::condition_variable slot_cv;
    bool slot_free[kRenderSlots] = {true, true};
    std::atomic<int64_t> slot_started[kRenderSlots] = {{0}, {0}};   // steady-clock ms at which the frame in the slot began, 0 = idle

    std::atomic<uint64_t> frames{0}, refused{0}, clients{0};
    std::mutex fds_mutex;
    std::set<int> fds;   // open client connections; at shutdown their read side is closed so that idle clients let go at once

    void stop_reading() {
        std::lock_guard<std::mutex> lock(fds_mutex);
        for (int fd : fds) ::shutdown(fd, SHUT_RD);   // a frame in flight still gets its reply
    }

    int acquire_slot() {
        std::unique_lock<std::mutex> lock(slot_mutex);
        for (;;) {
            for (int i = 0; i < kRenderSlots; i++) {
                if (slot_free[i]) {
                    slot_free[i] = false;
                    slot_started[i] = now_ms();
                    return i;
                }
            }
            slot_cv.wait(lock);
        }
    }
    void release_slot(int i) {
        std::lock_guard<std::mutex> lock(slot_mutex);
        slot_started[i] = 0;
        slot_free[i] = true;
        slot_cv.notify_one();
    }

    void serve(int fd) {
        clients++;
        {
            std::lock_guard<std::mutex> lock(fds_mutex);
            fds.insert(fd);
        }
        remote::Header h;
        std::vector<uint8_t> payload;
        std::string err;
        while (!g_stop && remote::recv_message(fd, &h, &payload, &err)) {
            if (h.type == remote::kHello) {
                const std::string info = "ttraster-serverd backend=" + backend_name + " chips=" + std::to_string(chips);
                if (!remote::send_message(fd, remote::kReply, 0, std::vector<uint8_t>(info.begin(), info.end()))) break;
            } else if (h.type == remote::kRender) {
                Frame frame;
                if (!remote::decode_frame(payload.data(), payload.size(), &frame, &err)) {
                    refused++;
                    if (verbose) std::fprintf(stderr, "ttr_serverd: refused a frame: %s\n", err.c_str());
                    if (!remote::send_message(fd, remote::kError, 0, std::vector<uint8_t>(err.begin(), err.end()))) break;
                    continue;
                }
                std::vector<uint8_t> reply;
                bool ok = true;
                const int slot = acquire_slot();
                const auto t0 = Clock::now();
                try {
                    if (hang_after >= 0 && static_cast<long>(frames.load()) >= hang_after) {
                        std::fprintf(stderr, "ttr_serverd: test hook: this frame hangs\n");
                        for (;;) std::this_thread::sleep_for(std::chrono::hours(1));
                    }
                    const Framebuffer& fb = backend->render(frame, h.planes, slot);
                    reply = remote::encode_reply(fb, backend->report(slot));
                } catch (const std::exception& e) {
                    err = e.what();
                    ok = false;
                }
                release_slot(slot);
                if (!ok) {
                    refused++;
                    std::fprintf(stderr, "ttr_serverd: a frame failed: %s\n", err.c_str());
                    if (!remote::send_message(fd, remote::kError, 0, std::vector<uint8_t>(err.begin(), err.end()))) break;
                    continue;
                }
                frames++;
                if (verbose) {
                    std::fprintf(stderr, "ttr_serverd: frame %llu, %zu triangles, %.1f ms\n", static_cast<unsigned long long>(frames.load()),
                                 frame.triangles.size(), std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                }
                if (!remote::send_message(fd, remote::kReply, 0, reply)) break;
            } else {
                err = "unknown message type";
                if (!remote::send_message(fd, remote::kError, 0, std::vector<uint8_t>(err.begin(), err.end()))) break;
            }
        }
        {
            std::lock_guard<std::mutex> lock(fds_mutex);   // erase before close: the number may be reused at once
            fds.erase(fd);
            ::close(fd);
        }
        clients--;
    }

    void watchdog() {
        while (!g_stop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            for (int i = 0; i < kRenderSlots; i++) {
                const int64_t t = slot_started[i];
                if (t != 0 && now_ms() - t > static_cast<int64_t>(frame_timeout_s * 1000)) {
                    std::fprintf(stderr, "ttr_serverd: a frame has been running for more than %.0f s: the device is hung; exiting with status 3\n", frame_timeout_s);
                    ::unlink(socket_path.c_str());
                    ::_exit(3);
                }
            }
        }
    }
    std::string socket_path;
};

}  // namespace

int main(int argc, char** argv) {
    Server s;
    s.socket_path = remote::default_socket_path();
    s.backend_name = "tensix";
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--socket" && i + 1 < argc) s.socket_path = argv[++i];
        else if (a == "--backend" && i + 1 < argc) s.backend_name = argv[++i];
        else if (a == "--chips" && i + 1 < argc) s.chips = static_cast<size_t>(std::atoi(argv[++i]));
        else if (a == "--frame-timeout" && i + 1 < argc) s.frame_timeout_s = std::atof(argv[++i]);
        else if (a == "--verbose") s.verbose = true;
        else {
            std::fprintf(stderr, "usage: ttr_serverd [--socket PATH] [--backend tensix|reference] [--chips N] [--frame-timeout SECONDS] [--verbose]\n");
            return 2;
        }
    }
    if (const char* v = std::getenv("TTR_SERVERD_TEST_HANG_AFTER")) s.hang_after = std::atol(v);
    if (s.socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
        std::fprintf(stderr, "ttr_serverd: socket path too long\n");
        return 2;
    }

    // Another server on this socket: stop. A socket file nobody listens on (a crashed server) is replaced.
    {
        const int probe = ::socket(AF_UNIX, SOCK_STREAM, 0);
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::memcpy(addr.sun_path, s.socket_path.c_str(), s.socket_path.size() + 1);
        if (::connect(probe, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0) {
            std::fprintf(stderr, "ttr_serverd: a server is already listening on %s\n", s.socket_path.c_str());
            return 1;
        }
        ::close(probe);
        ::unlink(s.socket_path.c_str());
    }

    if (s.backend_name == "reference") {
        s.backend = std::make_unique<ReferenceBackend>();
    } else if (s.backend_name == "tensix") {
#ifdef TT_RASTER_WITH_TENSIX
        try {
            s.backend = make_tensix_backend(s.chips);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "ttr_serverd: cannot open the device: %s\n", e.what());
            return 1;
        }
#else
        std::fprintf(stderr, "ttr_serverd: built without Tensix support; use --backend reference\n");
        return 2;
#endif
    } else {
        std::fprintf(stderr, "ttr_serverd: unknown backend '%s'\n", s.backend_name.c_str());
        return 2;
    }

    ::umask(077);
    const int lfd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::memcpy(addr.sun_path, s.socket_path.c_str(), s.socket_path.size() + 1);
    if (::bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 || ::listen(lfd, 16) != 0) {
        std::perror("ttr_serverd: bind/listen");
        return 1;
    }
    ::chmod(s.socket_path.c_str(), 0600);

    struct sigaction sa {};
    sa.sa_handler = on_signal;
    ::sigaction(SIGINT, &sa, nullptr);
    ::sigaction(SIGTERM, &sa, nullptr);
    ::signal(SIGPIPE, SIG_IGN);

    std::thread(&Server::watchdog, &s).detach();
    std::fprintf(stderr, "ttr_serverd: serving %s on %s (frame timeout %.0f s)\n", s.backend_name.c_str(), s.socket_path.c_str(), s.frame_timeout_s);
    while (!g_stop) {
        pollfd p{lfd, POLLIN, 0};
        if (::poll(&p, 1, 500) <= 0) continue;
        const int fd = ::accept4(lfd, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd < 0) continue;
        if (s.clients >= 64) {
            ::close(fd);
            continue;
        }
        std::thread(&Server::serve, &s, fd).detach();
    }
    ::close(lfd);
    ::unlink(s.socket_path.c_str());
    s.stop_reading();
    // Give the frames in flight a moment to finish before the backend is torn down.
    for (int i = 0; i < 100 && s.clients > 0; i++) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::fprintf(stderr, "ttr_serverd: stopped after %llu frames (%llu refused)\n", static_cast<unsigned long long>(s.frames.load()),
                 static_cast<unsigned long long>(s.refused.load()));
    if (s.clients > 0) ::_exit(0);   // a client is still mid-frame; do not race the device teardown with it
    return 0;
}
