#include <rais/scheduler.hpp>
#include <rais/metal_executor.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <mutex>
#include <thread>

namespace rais {

Scheduler::Scheduler(SchedulerConfig config)
    : global_queue_(config.global_queue_capacity)
    , gpu_executor_(config.gpu_executor) {

    size_t n = config.num_workers;
    if (n == 0) {
        unsigned hw = std::thread::hardware_concurrency();
        n = (hw > 1) ? hw - 1 : 1;
    }

    // Create all workers before starting any threads, so the vector is
    // fully built and stable when worker threads begin accessing it.
    workers_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        auto w = std::make_unique<Worker>();
        w->id = i;
        workers_.push_back(std::move(w));
    }
    for (size_t i = 0; i < n; ++i) {
        workers_[i]->thread = std::thread([this, i]() { worker_loop(i); });
    }
}

Scheduler::~Scheduler() {
    if (!shutdown_called_.load(std::memory_order_relaxed)) {
        shutdown(ShutdownPolicy::Drain);
    }
}

std::shared_ptr<Task> Scheduler::alloc_task() {
    Task* raw = task_slab_.allocate();
    if (raw) {
        new (raw) Task();
        return std::shared_ptr<Task>(raw, [this](Task* t) {
            t->~Task();
            task_slab_.free(t);
        });
    }
    // Slab exhausted — fall back to heap
    return std::make_shared<Task>();
}

TaskHandle Scheduler::submit(std::function<void()> fn, Lane lane) {
    auto task = alloc_task();
    task->fn = std::move(fn);
    task->lane = lane;
    task->enqueue_time_ns = clock_ns();

    lane_counts_[static_cast<int>(lane)].fetch_add(1, std::memory_order_relaxed);

    // Self-reference keeps the Task alive while in the lock-free queue
    // (which stores raw Task*). The worker resets self_ref after completion.
    Task* raw = task.get();
    task->self_ref = task;

    while (!global_queue_.push(raw)) {
        // Global queue full — back off and retry.
        std::this_thread::yield();
    }

    return TaskHandle(std::move(task));
}

TaskHandle Scheduler::submit(std::function<void()> fn, Lane lane,
                             uint64_t deadline_ns) {
    auto task = alloc_task();
    task->fn = std::move(fn);
    task->lane = lane;
    task->deadline_ns = deadline_ns;
    task->enqueue_time_ns = clock_ns();

    lane_counts_[static_cast<int>(lane)].fetch_add(1, std::memory_order_relaxed);

    Task* raw = task.get();
    task->self_ref = task;

    {
        std::lock_guard<std::mutex> lock(deadline_mutex_);
        deadline_heap_.push_back(raw);
        std::push_heap(deadline_heap_.begin(), deadline_heap_.end(),
                       DeadlineGreater{});
    }

    return TaskHandle(std::move(task));
}

TaskHandle Scheduler::submit_gpu(std::function<void(void*, void*)> gpu_fn) {
    assert(gpu_executor_ && "submit_gpu requires a MetalExecutor in SchedulerConfig");

    auto task = alloc_task();
    task->gpu_fn = std::move(gpu_fn);
    task->lane = Lane::GPU;
    task->enqueue_time_ns = clock_ns();

    lane_counts_[static_cast<int>(Lane::GPU)].fetch_add(1, std::memory_order_relaxed);

    Task* raw = task.get();
    task->self_ref = task;

    while (!global_queue_.push(raw)) {
        std::this_thread::yield();
    }

    return TaskHandle(std::move(task));
}

void Scheduler::shutdown(ShutdownPolicy policy) {
    bool expected = false;
    if (!shutdown_called_.compare_exchange_strong(expected, true,
            std::memory_order_acq_rel)) {
        return; // already shutting down
    }

    if (policy == ShutdownPolicy::Drain) {
        // Wait for all in-flight tasks (including GPU) to complete before
        // telling workers to stop. Workers keep running their normal loop
        // during this wait, draining queues and dispatching GPU work.
        for (;;) {
            int32_t total = 0;
            for (int i = 0; i < 4; ++i) {
                total += lane_counts_[i].load(std::memory_order_acquire);
            }
            if (total == 0) break;
            std::this_thread::yield();
        }
    }

    stop_flag_.store(true, std::memory_order_release);

    for (auto& w : workers_) {
        if (w->thread.joinable()) {
            w->thread.join();
        }
    }
}

int32_t Scheduler::lane_count(Lane lane) const {
    return lane_counts_[static_cast<int>(lane)].load(std::memory_order_acquire);
}

uint64_t Scheduler::deadline_misses() const {
    return deadline_misses_.load(std::memory_order_relaxed);
}

Task* Scheduler::pop_deadline_task() {
    std::lock_guard<std::mutex> lock(deadline_mutex_);
    if (deadline_heap_.empty()) return nullptr;
    std::pop_heap(deadline_heap_.begin(), deadline_heap_.end(), DeadlineGreater{});
    Task* t = deadline_heap_.back();
    deadline_heap_.pop_back();
    return t;
}

void Scheduler::worker_loop(size_t worker_id) {
    Worker& self = *workers_[worker_id];
    std::mt19937 rng(static_cast<unsigned>(worker_id));
    uint32_t backoff_us = 0;
    constexpr uint32_t kMaxBackoffUs = 1000; // 1ms cap

    for (;;) {
        // Check stop flag. For Drain policy, we must still process remaining
        // tasks, so only break when we also fail to find any work.
        bool stopping = stop_flag_.load(std::memory_order_acquire);

        Task* task = self.deque.pop();

        // Deadline tasks get priority over the FIFO global queue
        if (!task) {
            task = pop_deadline_task();
        }

        if (!task) {
            // Try global queue
            global_queue_.pop(task);
        }

        if (!task) {
            // Try stealing from a random victim
            task = try_steal(worker_id, rng);
        }

        if (!task) {
            if (stopping) break; // Drain complete or Cancel mode

            // Exponential backoff
            if (backoff_us == 0) {
                std::this_thread::yield();
                backoff_us = 1;
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
                backoff_us = std::min(backoff_us * 2, kMaxBackoffUs);
            }
            continue;
        }

        // Found work — reset backoff
        backoff_us = 0;

        // Handle cancelled tasks
        if (task->cancelled.load(std::memory_order_acquire)) {
            lane_counts_[static_cast<int>(task->lane)].fetch_sub(1,
                std::memory_order_relaxed);
            task->completed.store(true, std::memory_order_release);
            task->self_ref.reset(); // break ref cycle
            continue;
        }

        // GPU lane: dispatch to MetalExecutor instead of running on CPU
        if (task->lane == Lane::GPU) {
            if (gpu_executor_) {
                // Capture raw pointer + prevent destruction via self_ref
                std::shared_ptr<Task> ref = task->self_ref;
                bool ok = gpu_executor_->submit(
                    task->gpu_fn,
                    [this, ref]() {
                        // Called on Metal's completion thread
                        lane_counts_[static_cast<int>(Lane::GPU)].fetch_sub(1,
                            std::memory_order_relaxed);
                        ref->completed.store(true, std::memory_order_release);
                        ref->self_ref.reset();
                    });
                if (!ok) {
                    // Backpressure — re-enqueue and let another worker retry later
                    while (!global_queue_.push(task)) {
                        std::this_thread::yield();
                    }
                }
            } else {
                // No GPU executor — mark completed immediately (nothing to run)
                lane_counts_[static_cast<int>(Lane::GPU)].fetch_sub(1,
                    std::memory_order_relaxed);
                task->completed.store(true, std::memory_order_release);
                task->self_ref.reset();
            }
            continue;
        }

        // Priority enforcement: defer Bulk tasks when higher-priority work exists
        if (task->lane == Lane::Bulk) {
            int32_t interactive = lane_counts_[static_cast<int>(Lane::Interactive)]
                .load(std::memory_order_acquire);
            int32_t background = lane_counts_[static_cast<int>(Lane::Background)]
                .load(std::memory_order_acquire);
            if (interactive > 0 || background > 0) {
                // Re-enqueue to global queue so it can be picked up later
                while (!global_queue_.push(task)) {
                    std::this_thread::yield();
                }
                continue;
            }
        }

        // Check for starvation promotions
        check_starvation_promotions(task);

        // Track deadline misses
        if (task->deadline_ns != 0 && clock_ns() > task->deadline_ns) {
            deadline_misses_.fetch_add(1, std::memory_order_relaxed);
        }

        // Execute
        if (task->fn) {
            task->fn();
        }
        lane_counts_[static_cast<int>(task->lane)].fetch_sub(1,
            std::memory_order_relaxed);
        task->completed.store(true, std::memory_order_release);
        task->self_ref.reset(); // break ref cycle
    }
}

Task* Scheduler::try_steal(size_t worker_id, std::mt19937& rng) {
    size_t n = workers_.size();
    if (n <= 1) return nullptr;

    // Try up to n-1 random victims
    std::uniform_int_distribution<size_t> dist(0, n - 2);
    for (size_t attempt = 0; attempt < n - 1; ++attempt) {
        size_t victim = dist(rng);
        if (victim >= worker_id) ++victim; // skip self

        Task* task = workers_[victim]->deque.steal();
        if (task) return task;
    }
    return nullptr;
}

void Scheduler::check_starvation_promotions(Task* task) {
    uint64_t now = clock_ns();
    uint64_t age = now - task->enqueue_time_ns;

    if (task->lane == Lane::Bulk && age >= kBulkPromotionNs) {
        lane_counts_[static_cast<int>(Lane::Bulk)].fetch_sub(1,
            std::memory_order_relaxed);
        task->lane = Lane::Background;
        lane_counts_[static_cast<int>(Lane::Background)].fetch_add(1,
            std::memory_order_relaxed);
    }

    if (task->lane == Lane::Background && age >= kBackgroundPromotionNs) {
        lane_counts_[static_cast<int>(Lane::Background)].fetch_sub(1,
            std::memory_order_relaxed);
        task->lane = Lane::Interactive;
        lane_counts_[static_cast<int>(Lane::Interactive)].fetch_add(1,
            std::memory_order_relaxed);
    }
}

} // namespace rais
