#pragma once

#include <rais/allocator.hpp>
#include <rais/clock.hpp>
#include <rais/deque.hpp>
#include <rais/queue.hpp>
#include <rais/task.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
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

    static constexpr size_t kTaskSlabCapacity = 8192;

    MPMCQueue<Task*> global_queue_;
    std::vector<std::unique_ptr<Worker>> workers_;
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
    alignas(64) std::atomic<int32_t> lane_counts_[4] = {};

    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> shutdown_called_{false};
    alignas(64) std::atomic<uint64_t> deadline_misses_{0};
};

} // namespace rais
