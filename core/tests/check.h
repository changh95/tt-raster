// SPDX-License-Identifier: Apache-2.0
//
// Minimal test harness: CHECK records a failure and keeps going; a test's main returns
// ttr_test::finish().

#pragma once

#include <cstdio>

namespace ttr_test {

inline int& failures() {
    static int n = 0;
    return n;
}

inline int finish() {
    if (failures() == 0) {
        std::printf("PASS\n");
        return 0;
    }
    std::printf("FAIL: %d check(s) failed\n", failures());
    return 1;
}

}  // namespace ttr_test

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ttr_test::failures()++;                                            \
        }                                                                      \
    } while (0)

#define CHECK_MSG(cond, ...)                                                   \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("%s:%d: CHECK failed: %s: ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                                          \
            std::printf("\n");                                                 \
            ttr_test::failures()++;                                            \
        }                                                                      \
    } while (0)
