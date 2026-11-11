// SPDX-License-Identifier: Apache-2.0

#include "ttraster/parallel.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace ttr {

namespace {

struct Job {
    const std::function<void(size_t)>* fn = nullptr;
    size_t n = 0;
    std::atomic<size_t> next{0};
    std::atomic<size_t> finished{0};
    size_t helpers_wanted = 0;   // workers that may join besides the caller
    size_t helpers_joined = 0;   // guarded by the pool mutex
    std::mutex error_mutex;
    std::exception_ptr error;
    std::mutex done_mutex;
    std::condition_variable done_cv;

    // Take items until none are left. Returns true if this call finished the last item.
    void work() {
        for (size_t i = next.fetch_add(1); i < n; i = next.fetch_add(1)) {
            try {
                (*fn)(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
            if (finished.fetch_add(1) + 1 == n) {
                std::lock_guard<std::mutex> lock(done_mutex);
                done_cv.notify_all();
            }
        }
    }
};

class Pool {
public:
    Pool() {
        const size_t hw = std::max<size_t>(1, std::thread::hardware_concurrency());
        for (size_t t = 0; t + 1 < hw; t++) {   // the caller of parallel_for is the last thread
            workers_.emplace_back([this] { loop(); });
        }
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }
    size_t size() const { return workers_.size() + 1; }

    void run(size_t n, size_t max_threads, const std::function<void(size_t)>& fn) {
        const size_t threads = std::min(max_threads != 0 ? max_threads : size(), size());
        if (n == 0) return;
        if (n == 1 || threads <= 1) {
            for (size_t i = 0; i < n; i++) fn(i);
            return;
        }
        auto job = std::make_shared<Job>();
        job->fn = &fn;
        job->n = n;
        job->helpers_wanted = std::min(threads, n) - 1;
        {
            std::lock_guard<std::mutex> lock(m_);
            jobs_.push_back(job);
        }
        if (job->helpers_wanted == 1) cv_.notify_one(); else cv_.notify_all();
        job->work();   // the caller helps
        {
            std::unique_lock<std::mutex> lock(job->done_mutex);
            job->done_cv.wait(lock, [&] { return job->finished.load() == n; });
        }
        {
            std::lock_guard<std::mutex> lock(m_);   // drop the job if no worker has seen it exhausted yet
            auto it = std::find(jobs_.begin(), jobs_.end(), job);
            if (it != jobs_.end()) jobs_.erase(it);
        }
        if (job->error) std::rethrow_exception(job->error);
    }

private:
    void loop() {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lock(m_);
                for (;;) {
                    if (stop_) return;
                    // First job that still wants helpers and still has items.
                    for (auto& j : jobs_) {
                        if (j->helpers_joined < j->helpers_wanted && j->next.load() < j->n) {
                            j->helpers_joined++;
                            job = j;
                            break;
                        }
                    }
                    if (job) break;
                    cv_.wait(lock);
                }
            }
            job->work();
        }
    }

    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> jobs_;
    std::vector<std::thread> workers_;
    bool stop_ = false;
};

Pool& pool() {
    static Pool p;
    return p;
}

}  // namespace

void parallel_for(size_t n, size_t max_threads, const std::function<void(size_t)>& fn) { pool().run(n, max_threads, fn); }

}  // namespace ttr
