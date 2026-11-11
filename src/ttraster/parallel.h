// SPDX-License-Identifier: Apache-2.0
//
// parallel_for on a persistent worker pool. Creating and joining a std::thread per call costs far more than
// the small host stages of a frame are worth (each stage ran on 64 fresh threads), and concurrent frames
// share the same workers instead of oversubscribing the machine.

#pragma once

#include <cstddef>
#include <functional>

namespace ttr {

// Run fn(i) for every i in [0, n) on up to `max_threads` threads (0 = all hardware threads), the calling
// thread included. Items are handed out dynamically, one at a time. Returns when all items are done; the
// first exception thrown by an item is rethrown in the caller (the other items still run to completion).
// Safe to call from several threads at once and from inside an item (the caller always helps, so it makes
// progress without a free worker).
void parallel_for(size_t n, size_t max_threads, const std::function<void(size_t)>& fn);

}  // namespace ttr
