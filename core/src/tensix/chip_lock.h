// SPDX-License-Identifier: Apache-2.0
//
// A process (or a second renderer in the same process) that opens a device another renderer holds does not get an error from tt-metal:
// the second open in one process aborts with "context_id out of range", and a second process may wait forever. The renderer takes an
// advisory lock per device (the ids of TT_VISIBLE_DEVICES, "all" if unset) before it opens anything, and refuses with a message when
// the lock is held, so that an application with several processes (a compositor and its clients, a Vulkan loader probing a device twice)
// falls back to the CPU path instead of hanging. The locks go with the process: the kernel drops them when it exits or crashes.

#pragma once

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace ttr {

class ChipLock {
public:
    // Without TTR_DEVICE_POOL: lock the devices of TT_VISIBLE_DEVICES ("all" if unset). With TTR_DEVICE_POOL (a list such as "4,5" or "0-7",
    // used when TT_VISIBLE_DEVICES is not set): take the first `n_chips` devices of the pool that no other renderer holds and make them
    // the visible devices, so that several processes each get a chip of a tray without a daemon (the plan's device server, in its simplest
    // form: first come, first served; tt-metal reads TT_VISIBLE_DEVICES when it opens the first device, which is after this).
    explicit ChipLock(size_t n_chips = 1) {
        const char* pool = std::getenv("TTR_DEVICE_POOL");
        if (pool && *pool && !(std::getenv("TT_VISIBLE_DEVICES") && *std::getenv("TT_VISIBLE_DEVICES"))) {
            claim_pool(pool, n_chips == 0 ? 1 : n_chips);
            return;
        }
        std::vector<std::string> keys = split_ids(std::getenv("TT_VISIBLE_DEVICES"));
        if (keys.empty()) keys.push_back("all");
        for (const std::string& k : keys) {
            if (!try_lock(k)) {
                release();
                throw std::runtime_error("the Tensix device " + k + " is in use by another renderer (lock " + path(k) + " is held)");
            }
        }
    }
    ~ChipLock() { release(); }
    ChipLock(const ChipLock&) = delete;
    ChipLock& operator=(const ChipLock&) = delete;

private:
    static std::string path(const std::string& k) { return "/tmp/ttraster-device-" + k + ".lock"; }

    // "4,5", "0-7", "0-3,6" -> ids
    static std::vector<std::string> split_ids(const char* list) {
        std::vector<std::string> ids;
        std::string cur;
        for (const char* p = list ? list : ""; ; ++p) {
            if (*p == ',' || *p == '\0') {
                if (!cur.empty()) ids.push_back(cur);
                cur.clear();
                if (*p == '\0') break;
            } else if (std::isalnum(static_cast<unsigned char>(*p)) || *p == '-') {
                cur += *p;
            }
        }
        std::vector<std::string> out;
        for (const std::string& t : ids) {
            const size_t dash = t.find('-');
            if (dash == std::string::npos || dash == 0) {
                out.push_back(t);
            } else {
                const int lo = std::atoi(t.substr(0, dash).c_str()), hi = std::atoi(t.substr(dash + 1).c_str());
                for (int i = lo; i <= hi && i - lo < 64; i++) out.push_back(std::to_string(i));
            }
        }
        return out;
    }

    // true if the lock was taken (or cannot be made at all: then go on unguarded rather than refuse to run)
    bool try_lock(const std::string& k) {
        const int fd = ::open(path(k).c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
        if (fd < 0) return true;
        if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
            ::close(fd);
            return false;
        }
        fds_.push_back(fd);
        return true;
    }

    void claim_pool(const char* pool, size_t n_chips) {
        std::string chosen;
        size_t got = 0;
        for (const std::string& k : split_ids(pool)) {
            if (got == n_chips) break;
            if (try_lock(k)) {
                chosen += (chosen.empty() ? "" : ",") + k;
                got++;
            }
        }
        if (got < n_chips) {
            release();
            throw std::runtime_error("all Tensix devices of the pool " + std::string(pool) + " are in use by other renderers (need " + std::to_string(n_chips) + ")");
        }
        ::setenv("TT_VISIBLE_DEVICES", chosen.c_str(), 1);
    }

    void release() {
        for (const int fd : fds_) ::close(fd);   // closing drops the lock
        fds_.clear();
    }
    std::vector<int> fds_;
};

}  // namespace ttr
