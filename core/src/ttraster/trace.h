// SPDX-License-Identifier: Apache-2.0
//
// Optional timeline trace for pipelining analysis: with the environment variable TTR_TRACE set, the
// renderer prints "TRACE <ms since first call> slot <n> <what>" lines to stderr. Costs one branch when off.

#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace ttr {

inline void trace(const char* what, int slot) {
    static const bool on = std::getenv("TTR_TRACE") != nullptr;
    if (!on) {
        return;
    }
    static const auto t0 = std::chrono::steady_clock::now();
    std::fprintf(
        stderr, "TRACE %9.2f slot %d %s\n",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), slot, what);
}

}  // namespace ttr
