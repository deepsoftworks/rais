#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace rais {

enum class Lane : uint8_t {
    Interactive = 0, // CPU target: < 5ms submit-to-start
    Background  = 1, // CPU target: best-effort, yields to Interactive
    Bulk        = 2, // CPU target: only when Interactive + Background queues empty
    GPU         = 3, // dispatched to MetalExecutor, not CPU workers
    IO          = 4, // dedicated IO threads for SSD reads, never competes with CPU/GPU
};

inline constexpr uint64_t kBackgroundPromotionNs = 100'000'000ULL; // 100ms
inline constexpr uint64_t kBulkPromotionNs       = 500'000'000ULL; // 500ms

struct Task {
    std::function<void()>        fn;
    std::function<void(void*, void*)> gpu_fn;     // non-null only for GPU lane;
                                                  // args: (cmd_buf, encoder) from MetalExecutor
    Lane                         lane            = Lane::Background;
    uint64_t                     enqueue_time_ns = 0;  // set by Scheduler::submit()
    uint64_t                     deadline_ns     = 0;  // 0 = no deadline
    std::atomic<bool>            cancelled{false};
    std::atomic<bool>            completed{false};

    // --- Continuation support ---
    // `dependents` is written by submit_after() (on any thread) and read on
    // completion by activate_dependents() (on a worker thread). Both accesses
    // are guarded by dependents_mu to avoid a data race: a predecessor may
    // still be running when submit_after() registers a new dependent.
    std::mutex                   dependents_mu;
    std::vector<Task*>           dependents;
    // Protected by dependents_mu. Cleared once completion snapshots dependents;
    // submit_after uses this to avoid late registration races.
    bool                         accepting_dependents = true;

    // Number of predecessors that haven't completed yet. The task is not
    // runnable until this reaches 0.
    std::atomic<int32_t>         pending_deps{0};

    // The scheduler stores a raw Task* in the lock-free queue (which requires
    // trivially copyable types). This self-reference keeps the Task alive
    // while it is in-flight. The worker resets it after marking the task
    // complete, breaking the ref cycle and allowing destruction if the
    // caller also dropped the TaskHandle.
    std::shared_ptr<Task>        self_ref;
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
