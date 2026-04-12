#pragma once

#include <rais/allocator.hpp>
#include <rais/clock.hpp>
#include <rais/deque.hpp>
#include <rais/queue.hpp>
#include <rais/task.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace rais {

class MetalExecutor;

enum class ShutdownPolicy { Drain, Cancel };

struct SchedulerConfig {
    size_t num_workers          = 0; // 0 = hardware_concurrency() - 1
    size_t global_queue_capacity = 65536; // must be power of two
    MetalExecutor* gpu_executor = nullptr; // optional — enables Lane::GPU dispatch

    // Number of dedicated IO threads. These threads only service Lane::IO
    // tasks and never steal from CPU worker deques. 2 is a reasonable default
    // for saturating Apple Silicon SSD bandwidth (~5-7 GB/s) without excessive
    // context switching.
    size_t io_thread_count      = 2;
    size_t io_queue_capacity    = 4096; // must be power of two
};

class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config = {});
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    /// Submit a CPU task. Sets enqueue_time_ns and returns a handle.
    TaskHandle submit(std::function<void()> fn, Lane lane = Lane::Background);

    /// Submit a CPU task with a deadline (absolute timestamp in nanoseconds,
    /// same clock as clock_ns()). Deadline tasks are served in earliest-
    /// deadline-first order, ahead of non-deadline FIFO tasks.
    TaskHandle submit(std::function<void()> fn, Lane lane, uint64_t deadline_ns);

    /// Submit a GPU task. The encode function receives (cmd_buf, encoder)
    /// from MetalExecutor. Requires gpu_executor in SchedulerConfig.
    TaskHandle submit_gpu(std::function<void(void*, void*)> gpu_fn);

    /// Submit a task that depends on other tasks completing first.
    /// The task becomes runnable when all deps have completed.
    /// enqueue_time_ns is set at creation time (not activation time) so
    /// starvation promotion reflects true wait time.
    TaskHandle submit_after(std::function<void()> fn, Lane lane,
                            std::vector<TaskHandle> deps);

    /// Convenience: single-predecessor continuation chain.
    /// Equivalent to submit_after(fn, lane, {dep}).
    TaskHandle then(TaskHandle dep, std::function<void()> fn,
                    Lane lane = Lane::Background);

    /// Shut down the scheduler. Drain finishes all pending tasks;
    /// Cancel marks pending tasks as cancelled and stops immediately.
    void shutdown(ShutdownPolicy policy = ShutdownPolicy::Drain);

    /// Number of tasks currently in a given lane (approximate).
    int32_t lane_count(Lane lane) const;

    /// Number of tasks that started execution after their deadline had passed.
    uint64_t deadline_misses() const;

private:
    struct Worker {
        std::thread thread;
        WorkStealingDeque<Task*> deque;
        size_t id = 0;

        Worker() = default;
        Worker(const Worker&) = delete;
        Worker& operator=(const Worker&) = delete;
    };

    void worker_loop(size_t worker_id);
    Task* try_steal(size_t worker_id, std::mt19937& rng);
    void check_starvation_promotions(Task* task);
    std::shared_ptr<Task> alloc_task();
    Task* pop_deadline_task();
    void activate_dependents(Task* task, WorkStealingDeque<Task*>& local_deque);
    void enqueue_task(Task* raw);
    void push_global_task(Task* raw);
    bool pop_global_task(Task*& raw);
    void io_worker_loop();
    void release_task_lifetime_ref(Task* task);

    static constexpr size_t kTaskSlabCapacity = 8192;

    MPMCQueue<Task*> global_queue_;
    MPMCQueue<Task*> io_queue_;         // separate queue for Lane::IO tasks
    std::vector<std::unique_ptr<Worker>> workers_;
    std::vector<std::thread> io_threads_; // dedicated IO threads
    MetalExecutor* gpu_executor_ = nullptr;
    SlabAllocator<Task, kTaskSlabCapacity> task_slab_;

    // Deadline min-heap: tasks with explicit deadlines, served EDF.
    struct DeadlineGreater {
        bool operator()(const Task* a, const Task* b) const {
            return a->deadline_ns > b->deadline_ns;
        }
    };
    std::mutex deadline_mutex_;
    std::vector<Task*> deadline_heap_;

    // Per-lane admission counters. Indexed by static_cast<int>(Lane).
    // 5 slots: Interactive(0), Background(1), Bulk(2), GPU(3), IO(4).
    alignas(64) std::atomic<int32_t> lane_counts_[5] = {};

    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> shutdown_called_{false};
    alignas(64) std::atomic<uint64_t> deadline_misses_{0};

    std::mutex tsan_global_mu_;
    std::deque<Task*> tsan_global_queue_;

    // TSan mode defers task destruction until worker threads have joined,
    // avoiding teardown races while sanitizer instrumentation is active.
    std::mutex tsan_retired_mu_;
    std::vector<std::shared_ptr<Task>> tsan_retired_tasks_;
};

} // namespace rais
