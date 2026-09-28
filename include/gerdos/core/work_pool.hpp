#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace gerdos {

// A bounded worker pool: the one semantic home for backend execution
// threads. Four fixed workers (deterministic, never hardware-derived);
// a bounded pending queue (64); submission past the bound is refused
// loudly via nullopt. Destruction drains: queued work runs before
// workers join. Generic C++ machinery — no vendor vocabulary.
// Under GERDOS_NO_THREADS the pool executes inline with no threads:
// microcontroller-class targets link without threading, determinism
// preserved (submission order is execution order).
class WorkPool {
public:
    static constexpr std::size_t kWorkers = 4;
    static constexpr std::size_t kQueueBound = 64;

    WorkPool() {
#ifndef GERDOS_NO_THREADS
        for (std::size_t i = 0; i < kWorkers; ++i) {
            workers_.emplace_back([this] { serve(); });
        }
#endif
    }

    WorkPool(const WorkPool&) = delete;
    WorkPool& operator=(const WorkPool&) = delete;

    ~WorkPool() {
#ifdef GERDOS_NO_THREADS
        stopping_ = true;
#else
        {
            std::lock_guard<std::mutex> guard(mutex_);
            stopping_ = true;
        }

        available_.notify_all();

        for (auto& worker : workers_) {
            worker.join();
        }
#endif
    }

    [[nodiscard]] std::size_t pending() const {
#ifdef GERDOS_NO_THREADS
        return 0;
#else
        std::lock_guard<std::mutex> guard(mutex_);
        return tasks_.size();
#endif
    }

    // Submit work for a worker; the future completes with the task
    // result. Refuses loudly with nullopt when the bounded queue is
    // full or the pool is stopping. Launch failures also refuse.
    // Under GERDOS_NO_THREADS the task runs inline before returning:
    // the future is already ready, order is deterministic.
    template <typename Task>
    [[nodiscard]] auto try_submit(Task&& task)
        -> std::optional<std::future<decltype(task())>> {
        using Result = decltype(task());
        auto packed = std::make_shared<std::packaged_task<Result()>>(
            std::forward<Task>(task));
        std::future<Result> result = packed->get_future();

#ifdef GERDOS_NO_THREADS
        (*packed)();
        return std::optional<std::future<Result>>{std::move(result)};
#else
        {
            std::lock_guard<std::mutex> guard(mutex_);

            if (stopping_ || tasks_.size() >= kQueueBound) {
                return std::nullopt;
            }

            tasks_.emplace([packed] { (*packed)(); });
        }

        available_.notify_one();
        return std::optional<std::future<Result>>{std::move(result)};
#endif
    }

private:
    void serve() {
#ifdef GERDOS_NO_THREADS
        return;
#else
        while (true) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> guard(mutex_);
                available_.wait(guard, [this] {
                    return stopping_ || !tasks_.empty();
                });

                if (stopping_ && tasks_.empty()) {
                    return;
                }

                task = std::move(tasks_.front());
                tasks_.pop();
            }

            task();
        }
#endif
    }

#ifdef GERDOS_NO_THREADS
    // Inline mode keeps no threads, no queue, no synchronization:
    // the members below vanish so microcontroller targets link clean.
    // pending() reports zero; stopping_ still refuses post-destruction
    // submissions deterministically (always false while alive).
    bool stopping_{false};
#else
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_{false};
#endif
};

} // namespace gerdos

