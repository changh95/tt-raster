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
    ChipLock() {
        std::vector<std::string> keys;
        const char* env = std::getenv("TT_VISIBLE_DEVICES");
        std::string cur;
        for (const char* p = env ? env : ""; ; ++p) {
            if (*p == ',' || *p == '\0') {
                if (!cur.empty()) keys.push_back(cur);
                cur.clear();
                if (*p == '\0') break;
            } else if (std::isalnum(static_cast<unsigned char>(*p))) {
                cur += *p;
            }
        }
        if (keys.empty()) keys.push_back("all");
        for (const std::string& k : keys) {
            const std::string path = "/tmp/ttraster-device-" + k + ".lock";
            const int fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
            if (fd < 0) continue;   // no lock directory: go on unguarded rather than refuse to run
            if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
                ::close(fd);
                release();
                throw std::runtime_error("the Tensix device " + k + " is in use by another renderer (lock " + path + " is held)");
            }
            fds_.push_back(fd);
        }
    }
    ~ChipLock() { release(); }
    ChipLock(const ChipLock&) = delete;
    ChipLock& operator=(const ChipLock&) = delete;

private:
    void release() {
        for (const int fd : fds_) ::close(fd);   // closing drops the lock
        fds_.clear();
    }
    std::vector<int> fds_;
};

}  // namespace ttr
