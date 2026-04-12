#include <rais/scheduler.hpp>
#include <rais/metal_executor.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <mutex>
#include <thread>

namespace rais {

namespace {

inline std::vector<Task*> snapshot_and_close_dependents(Task* task) {
    std::lock_guard<std::mutex> lock(task->dependents_mu);
    task->accepting_dependents = false;
    return task->dependents;
}

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
inline constexpr bool kTsanBuild = true;
#else
inline constexpr bool kTsanBuild = false;
#endif
#elif defined(__SANITIZE_THREAD__)
inline constexpr bool kTsanBuild = true;
#else
inline constexpr bool kTsanBuild = false;
#endif

} // namespace

Scheduler::Scheduler(SchedulerConfig config)
    : global_queue_(config.global_queue_capacity)
    , io_queue_(config.io_queue_capacity)
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

    // Dedicated IO threads — only service io_queue_, never steal from workers
    size_t io_threads = kTsanBuild ? 0 : config.io_thread_count;
    for (size_t i = 0; i < io_threads; ++i) {
        io_threads_.emplace_back([this]() { io_worker_loop(); });
    }
}

Scheduler::~Scheduler() {
    if (!shutdown_called_.load(std::memory_order_relaxed)) {
        shutdown(ShutdownPolicy::Drain);
    }
}

void Scheduler::release_task_lifetime_ref(Task* task) {
    std::shared_ptr<Task> keepalive = std::move(task->self_ref);
    if (!keepalive) return;

    if (kTsanBuild) {
        std::lock_guard<std::mutex> lock(tsan_retired_mu_);
        tsan_retired_tasks_.push_back(std::move(keepalive));
        return;
    }

    keepalive.reset();
}

std::shared_ptr<Task> Scheduler::alloc_task() {
    if (kTsanBuild) {
        return std::make_shared<Task>();
    }

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

void Scheduler::push_global_task(Task* raw) {
    if (kTsanBuild) {
        std::lock_guard<std::mutex> lock(tsan_global_mu_);
        tsan_global_queue_.push_back(raw);
        return;
    }
    while (!global_queue_.push(raw)) {
        std::this_thread::yield();
    }
}

bool Scheduler::pop_global_task(Task*& raw) {
    if (kTsanBuild) {
        std::lock_guard<std::mutex> lock(tsan_global_mu_);
        if (tsan_global_queue_.empty()) {
            raw = nullptr;
            return false;
        }
        raw = tsan_global_queue_.front();
        tsan_global_queue_.pop_front();
        return true;
    }
    return global_queue_.pop(raw);
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

    // IO-lane tasks go to the dedicated io_queue_ so they are only serviced
    // by IO threads and never compete with CPU compute work.
    bool use_io_queue = (lane == Lane::IO) && !kTsanBuild;
    if (use_io_queue) {
        while (!io_queue_.push(raw)) {
            std::this_thread::yield();
        }
    } else {
        push_global_task(raw);
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

void Scheduler::enqueue_task(Task* raw) {
    bool use_io_queue = (raw->lane == Lane::IO) && !kTsanBuild;
    if (use_io_queue) {
        while (!io_queue_.push(raw)) {
            std::this_thread::yield();
        }
    } else {
        push_global_task(raw);
    }
}

TaskHandle Scheduler::submit_after(std::function<void()> fn, Lane lane,
                                   std::vector<TaskHandle> deps) {
    auto task = alloc_task();
    task->fn = std::move(fn);
    task->lane = lane;
    task->enqueue_time_ns = clock_ns();

    lane_counts_[static_cast<int>(lane)].fetch_add(1, std::memory_order_relaxed);

    Task* raw = task.get();
    task->self_ref = task;

    if (deps.empty()) {
        // No dependencies — behaves identically to submit()
        enqueue_task(raw);
        return TaskHandle(std::move(task));
    }

    // Set pending dep count before registering with predecessors.
    // acq_rel not needed here — store is sequenced before the loop below
    // and predecessors synchronize via their own completed.store(release).
    task->pending_deps.store(static_cast<int32_t>(deps.size()),
                             std::memory_order_relaxed);

    int32_t already_done = 0;
    for (auto& dep : deps) {
        Task* pred = dep.get();
        if (!pred) {
            // Null handle — treat as already completed
            ++already_done;
            continue;
        }

        // Resolve predecessor state under dependents_mu:
        // - accepting_dependents=true: register so completion decrements us
        // - accepting_dependents=false: completion already snapshotted deps
        //   and won't observe us; count as already done.
        bool pred_completed = false;
        {
            std::lock_guard<std::mutex> lock(pred->dependents_mu);
            pred_completed = !pred->accepting_dependents;
            if (!pred_completed) {
                pred->dependents.push_back(raw);
            }
        }

        if (pred_completed) {
            ++already_done;
        }
    }

    // Subtract deps that were already complete. If all were done, enqueue now.
    if (already_done > 0) {
        int32_t prev = task->pending_deps.fetch_sub(already_done,
                                                     std::memory_order_acq_rel);
        if (prev == already_done) {
            // All deps already done — enqueue immediately
            enqueue_task(raw);
        }
    }

    return TaskHandle(std::move(task));
}

TaskHandle Scheduler::then(TaskHandle dep, std::function<void()> fn, Lane lane) {
    return submit_after(std::move(fn), lane, {std::move(dep)});
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

    push_global_task(raw);

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
            for (int i = 0; i < 5; ++i) {
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
    for (auto& t : io_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }

    if (kTsanBuild) {
        std::lock_guard<std::mutex> lock(tsan_retired_mu_);
        tsan_retired_tasks_.clear();
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

void Scheduler::activate_dependents(Task* task,
                                    WorkStealingDeque<Task*>& local_deque) {
    std::vector<Task*> deps = snapshot_and_close_dependents(task);
    for (Task* dep : deps) {
        if (task->cancelled.load(std::memory_order_acquire)) {
            // Cascading cancellation: mark dependent as cancelled+completed
            // so its own dependents and waiters unblock.
            dep->cancelled.store(true, std::memory_order_relaxed);
        }

        // acq_rel: acquire sees predecessor's writes; release publishes
        // the decrement so the dependent's fn (or next decrementer) sees
        // all predecessor side-effects.
        int32_t prev = dep->pending_deps.fetch_sub(1, std::memory_order_acq_rel);
        if (prev == 1) {
            // We were the last predecessor — this dependent is now runnable.
            if (dep->cancelled.load(std::memory_order_acquire)) {
                // Already cancelled (either directly or cascading). Complete it
                // without running fn, then propagate to its own dependents.
                lane_counts_[static_cast<int>(dep->lane)].fetch_sub(1,
                    std::memory_order_relaxed);
                dep->completed.store(true, std::memory_order_release);
                activate_dependents(dep, local_deque);
                release_task_lifetime_ref(dep);
            } else {
                // Push to the completing worker's local deque for cache locality:
                // the dependent's input data is likely still hot in this core.
                if (kTsanBuild) {
                    push_global_task(dep);
                } else {
                    local_deque.push(dep);
                }
            }
        }
    }
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
            pop_global_task(task);
        }

        if (!task && !kTsanBuild) {
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

        // Handle cancelled tasks — still activate dependents so the DAG
        // propagates cancellation and waiters unblock.
        if (task->cancelled.load(std::memory_order_acquire)) {
            lane_counts_[static_cast<int>(task->lane)].fetch_sub(1,
                std::memory_order_relaxed);
            task->completed.store(true, std::memory_order_release);
            activate_dependents(task, self.deque);
            release_task_lifetime_ref(task); // break ref cycle safely
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
                        // Called on Metal's completion thread — no local deque
                        // available, so dependents go to the global queue.
                        lane_counts_[static_cast<int>(Lane::GPU)].fetch_sub(1,
                            std::memory_order_relaxed);
                        std::vector<Task*> deps =
                            snapshot_and_close_dependents(ref.get());
                        for (Task* dep : deps) {
                            int32_t prev = dep->pending_deps.fetch_sub(1,
                                std::memory_order_acq_rel);
                            if (prev == 1) {
                                enqueue_task(dep);
                            }
                        }
                        ref->completed.store(true, std::memory_order_release);
                        release_task_lifetime_ref(ref.get());
                    });
                if (!ok) {
                    // Backpressure — re-enqueue and let another worker retry later
                    push_global_task(task);
                }
            } else {
                // No GPU executor — mark completed immediately (nothing to run)
                lane_counts_[static_cast<int>(Lane::GPU)].fetch_sub(1,
                    std::memory_order_relaxed);
                task->completed.store(true, std::memory_order_release);
                activate_dependents(task, self.deque);
                release_task_lifetime_ref(task);
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
                push_global_task(task);
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
        activate_dependents(task, self.deque);
        release_task_lifetime_ref(task); // break ref cycle safely
    }
}

void Scheduler::io_worker_loop() {
    uint32_t backoff_us = 0;
    constexpr uint32_t kMaxBackoffUs = 1000;

    // IO workers need a dummy deque for activate_dependents. We use the
    // global queue path instead (enqueue_task) by creating a local deque
    // that is only used for dependent activation within this thread.
    WorkStealingDeque<Task*> local_deque;

    for (;;) {
        bool stopping = stop_flag_.load(std::memory_order_acquire);

        Task* task = nullptr;
        io_queue_.pop(task);

        if (!task) {
            if (stopping) break;
            if (backoff_us == 0) {
                std::this_thread::yield();
                backoff_us = 1;
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
                backoff_us = std::min(backoff_us * 2, kMaxBackoffUs);
            }
            continue;
        }

        backoff_us = 0;

        if (task->cancelled.load(std::memory_order_acquire)) {
            lane_counts_[static_cast<int>(task->lane)].fetch_sub(1,
                std::memory_order_relaxed);
            task->completed.store(true, std::memory_order_release);
            activate_dependents(task, local_deque);
            release_task_lifetime_ref(task);
            // Drain any dependents that were pushed to our local deque
            // back into the appropriate global queue.
            Task* dep = nullptr;
            while ((dep = local_deque.pop()) != nullptr) {
                enqueue_task(dep);
            }
            continue;
        }

        if (task->fn) {
            task->fn();
        }
        lane_counts_[static_cast<int>(task->lane)].fetch_sub(1,
            std::memory_order_relaxed);
        task->completed.store(true, std::memory_order_release);
        activate_dependents(task, local_deque);
        release_task_lifetime_ref(task);

        // Dependents activated by IO completion likely belong to other lanes
        // (e.g. GPU compute after an SSD read). Push them to the global queue.
        Task* dep = nullptr;
        while ((dep = local_deque.pop()) != nullptr) {
            enqueue_task(dep);
        }
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
