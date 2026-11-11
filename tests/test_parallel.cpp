// SPDX-License-Identifier: Apache-2.0
//
// The shared worker pool behind parallel_for: every item runs exactly once, errors propagate, calls may nest and
// may come from several threads at the same time.

#include <atomic>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "check.h"
#include "ttraster/parallel.h"

using namespace ttr;

int main() {
    // Every index exactly once, for several sizes and thread limits.
    for (size_t n : {0u, 1u, 2u, 7u, 64u, 1000u, 100000u}) {
        for (size_t threads : {0u, 1u, 3u, 1000u}) {
            std::vector<std::atomic<int>> hits(n);
            parallel_for(n, threads, [&](size_t i) { hits[i]++; });
            bool ok = true;
            for (size_t i = 0; i < n; i++) ok &= hits[i].load() == 1;
            CHECK_MSG(ok, "n=%zu threads=%zu", n, threads);
        }
    }

    // The first exception reaches the caller; the pool is still usable afterwards.
    bool threw = false;
    try {
        parallel_for(1000, 0, [](size_t i) { if (i == 500) throw std::runtime_error("item failed"); });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    std::atomic<int> after{0};
    parallel_for(100, 0, [&](size_t) { after++; });
    CHECK(after == 100);

    // Nested calls make progress (the caller always helps).
    std::atomic<long> nested{0};
    parallel_for(32, 0, [&](size_t) { parallel_for(32, 0, [&](size_t) { nested++; }); });
    CHECK_MSG(nested == 32 * 32, "%ld", nested.load());

    // Several callers at once, as two frames in flight do.
    std::vector<std::thread> callers;
    std::atomic<long> total{0};
    for (int c = 0; c < 6; c++) {
        callers.emplace_back([&] {
            for (int rep = 0; rep < 50; rep++) {
                parallel_for(200, 0, [&](size_t) { total++; });
            }
        });
    }
    for (auto& t : callers) t.join();
    CHECK_MSG(total == 6L * 50 * 200, "%ld", total.load());
    return ttr_test::finish();
}
