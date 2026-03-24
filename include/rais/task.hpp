#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>

namespace rais {

enum class Lane : uint8_t {
    Interactive = 0, // CPU target: < 5ms submit-to-start
    Background  = 1, // CPU target: best-effort, yields to Interactive
    Bulk        = 2, // CPU target: only when Interactive + Background queues empty
    GPU         = 3, // dispatched to MetalExecutor, not CPU workers
};

inline constexpr uint64_t kBackgroundPromotionNs = 100'000'000ULL; // 100ms
inline constexpr uint64_t kBulkPromotionNs       = 500'000'000ULL; // 500ms

struct alignas(64) Task {
    std::function<void()>        fn;
    std::function<void(void*)>   gpu_fn;         // non-null only for GPU lane;
                                                  // argument is id<MTLCommandBuffer>
    Lane                         lane            = Lane::Background;
    uint64_t                     enqueue_time_ns = 0;  // set by Scheduler::submit()
    uint64_t                     deadline_ns     = 0;  // 0 = no deadline
    std::atomic<bool>            cancelled{false};
    std::atomic<bool>            completed{false};
};

/// Caller-facing handle to a submitted task.
///
/// Wraps a shared_ptr<Task> so the Task stays alive until both the
/// scheduler and the caller are done with it.
class TaskHandle {
public:
    TaskHandle() = default;
    explicit TaskHandle(std::shared_ptr<Task> t) : task_(std::move(t)) {}

    /// Spin-wait with progressive backoff until the task completes.
    void wait() const {
        if (!task_) return;
        while (!task_->completed.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    /// Cooperatively cancel the task. Returns true if cancellation was
    /// set; false if the task already completed or was already cancelled.
    bool cancel() {
        if (!task_) return false;
        if (task_->completed.load(std::memory_order_acquire)) return false;
        bool expected = false;
        return task_->cancelled.compare_exchange_strong(
            expected, true, std::memory_order_release, std::memory_order_relaxed);
    }

    /// Check whether the task has completed (either ran or was cancelled).
    bool done() const {
        return task_ && task_->completed.load(std::memory_order_acquire);
    }

    /// Access the underlying task (for scheduler internals).
    Task* get() const { return task_.get(); }

private:
    std::shared_ptr<Task> task_;
};

} // namespace rais
