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

inline void blocking_push(MPMCQueue<Task*>& queue, Task* raw) {
    while (!queue.push(raw)) {
        std::this_thread::yield();
    }
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
    : interactive_queue_(config.global_queue_capacity)
    , background_queue_(config.global_queue_capacity)
    , bulk_queue_(config.global_queue_capacity)
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

// Route a runnable task to its lane's queue. TSan builds funnel everything
// through a single mutex-guarded queue instead.
void Scheduler::enqueue_task(Task* raw) {
    if (kTsanBuild) {
        std::lock_guard<std::mutex> lock(tsan_global_mu_);
        tsan_global_queue_.push_back(raw);
        return;
    }
    switch (raw->lane) {
        case Lane::IO:         blocking_push(io_queue_, raw); break;
        case Lane::Background: blocking_push(background_queue_, raw); break;
        case Lane::Bulk:       blocking_push(bulk_queue_, raw); break;
        default:               blocking_push(interactive_queue_, raw); break;
    }
}

Task* Scheduler::pop_cpu_task(Worker& self) {
    if (kTsanBuild) {
        std::lock_guard<std::mutex> lock(tsan_global_mu_);
        if (tsan_global_queue_.empty()) return nullptr;
        Task* raw = tsan_global_queue_.front();
        tsan_global_queue_.pop_front();
        return raw;
    }

    Task* raw = nullptr;

    // Strict priority order would starve lower lanes under sustained
    // Interactive load, so at most once per kBackgroundPromotionNs give
    // Background/Bulk one pop ahead of Interactive. A Bulk task popped here
    // still passes through the deferral check in worker_loop — this is what
    // lets an aged Bulk task reach its promotion check at all. The clock
    // read is amortized over 256 pops to keep it off the hot path.
    if ((++self.pop_count & 0xFF) == 0) {
        uint64_t now = clock_ns();
        if (now - self.last_low_service_ns >= kBackgroundPromotionNs) {
            self.last_low_service_ns = now;
            if (background_queue_.pop(raw)) return raw;
            if (bulk_queue_.pop(raw)) return raw;
        }
    }

    if (interactive_queue_.pop(raw)) return raw;
    if (background_queue_.pop(raw)) return raw;
    if (bulk_queue_.pop(raw)) return raw;
    return nullptr;
}

TaskHandle Scheduler::submit(std::function<void()> fn, Lane lane) {
    auto task = alloc_task();
    task->fn = std::move(fn);
    task->lane = lane;
    task->enqueue_time_ns = clock_ns();

    lane_counter(lane).fetch_add(1, std::memory_order_relaxed);

    // Self-reference keeps the Task alive while in the lock-free queue
    // (which stores raw Task*). The worker resets self_ref after completion.
    Task* raw = task.get();
    task->self_ref = task;

    enqueue_task(raw);

    return TaskHandle(std::move(task));
}

TaskHandle Scheduler::submit(std::function<void()> fn, Lane lane,
                             uint64_t deadline_ns) {
    auto task = alloc_task();
    task->fn = std::move(fn);
    task->lane = lane;
    task->deadline_ns = deadline_ns;
    task->enqueue_time_ns = clock_ns();

    lane_counter(lane).fetch_add(1, std::memory_order_relaxed);

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

TaskHandle Scheduler::submit_after(std::function<void()> fn, Lane lane,
                                   std::vector<TaskHandle> deps) {
    auto task = alloc_task();
    task->fn = std::move(fn);
    task->lane = lane;
    task->enqueue_time_ns = clock_ns();

    lane_counter(lane).fetch_add(1, std::memory_order_relaxed);

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
            // The predecessor finished as cancelled — propagate, matching
            // the cascade that dependents registered before its completion
            // receive in activate_dependents.
            if (pred->cancelled.load(std::memory_order_acquire)) {
                raw->cancelled.store(true, std::memory_order_relaxed);
            }
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

    lane_counter(Lane::GPU).fetch_add(1, std::memory_order_relaxed);

    Task* raw = task.get();
    task->self_ref = task;

    enqueue_task(raw);

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
                total += lane_counts_[i].value.load(std::memory_order_acquire);
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
    return lane_counts_[static_cast<int>(lane)].value.load(std::memory_order_acquire);
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
                                    WorkStealingDeque<Task*>* local_deque) {
    // Iterative cascade: cancelled dependents that complete without running
    // are queued here instead of recursing, so an arbitrarily deep chain of
    // cancelled continuations cannot overflow the stack.
    std::vector<Task*> cascade;
    Task* current = task;

    for (;;) {
        std::vector<Task*> deps = snapshot_and_close_dependents(current);
        bool cascade_cancel = current->cancelled.load(std::memory_order_acquire);

        for (Task* dep : deps) {
            if (cascade_cancel) {
                // Cascading cancellation: mark dependent as cancelled so it
                // completes without running and its own waiters unblock.
                dep->cancelled.store(true, std::memory_order_relaxed);
            }

            // acq_rel: acquire sees predecessor's writes; release publishes
            // the decrement so the dependent's fn (or next decrementer) sees
            // all predecessor side-effects.
            int32_t prev = dep->pending_deps.fetch_sub(1, std::memory_order_acq_rel);
            if (prev != 1) continue;

            // We were the last predecessor — this dependent is now runnable.
            if (dep->cancelled.load(std::memory_order_acquire)) {
                // Complete it without running fn, then propagate to its own
                // dependents on a later iteration.
                lane_counter(dep->lane).fetch_sub(1, std::memory_order_relaxed);
                dep->completed.store(true, std::memory_order_release);
                cascade.push_back(dep);
            } else if (!kTsanBuild && local_deque) {
                // Push to the completing worker's local deque for cache
                // locality: the dependent's input is likely still hot here.
                local_deque->push(dep);
            } else {
                // No local deque (IO thread, Metal completion thread) —
                // route to the dependent's lane queue.
                enqueue_task(dep);
            }
        }

        if (current != task) {
            release_task_lifetime_ref(current);
        }
        if (cascade.empty()) break;
        current = cascade.back();
        cascade.pop_back();
    }
}

void Scheduler::finish_task(Task* task, WorkStealingDeque<Task*>* local_deque) {
    lane_counter(task->lane).fetch_sub(1, std::memory_order_relaxed);
    task->completed.store(true, std::memory_order_release);
    activate_dependents(task, local_deque);
    release_task_lifetime_ref(task); // break ref cycle safely
}

void Scheduler::worker_loop(size_t worker_id) {
    Worker& self = *workers_[worker_id];
    std::mt19937 rng(static_cast<unsigned>(worker_id));
    uint32_t backoff_us = 0;
    static constexpr uint32_t kMaxBackoffUs = 1000; // 1ms cap
    self.last_low_service_ns = clock_ns();

    auto idle_backoff = [&backoff_us]() {
        if (backoff_us == 0) {
            std::this_thread::yield();
            backoff_us = 1;
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(backoff_us));
            backoff_us = std::min(backoff_us * 2, kMaxBackoffUs);
        }
    };

    for (;;) {
        // Check stop flag. For Drain policy, we must still process remaining
        // tasks, so only break when we also fail to find any work.
        bool stopping = stop_flag_.load(std::memory_order_acquire);

        Task* task = self.deque.pop();

        // Deadline tasks get priority over the FIFO lane queues
        if (!task) {
            task = pop_deadline_task();
        }

        if (!task) {
            task = pop_cpu_task(self);
        }

        if (!task && !kTsanBuild) {
            // Try stealing from a random victim
            task = try_steal(worker_id, rng);
        }

        if (!task) {
            if (stopping) break; // Drain complete or Cancel mode
            idle_backoff();
            continue;
        }

        // Handle cancelled tasks — still activate dependents so the DAG
        // propagates cancellation and waiters unblock.
        if (task->cancelled.load(std::memory_order_acquire)) {
            backoff_us = 0;
            finish_task(task, &self.deque);
            continue;
        }

        // GPU lane: dispatch to MetalExecutor instead of running on CPU
        if (task->lane == Lane::GPU) {
            backoff_us = 0;
            if (gpu_executor_) {
                // Capture raw pointer + prevent destruction via self_ref
                std::shared_ptr<Task> ref = task->self_ref;
                bool ok = gpu_executor_->submit(
                    task->gpu_fn,
                    [this, ref]() {
                        // Called on Metal's completion thread — no local
                        // deque, so dependents go to their lane queues.
                        finish_task(ref.get(), nullptr);
                    });
                if (!ok) {
                    // Backpressure — re-enqueue and let another worker retry later
                    enqueue_task(task);
                }
            } else {
                // No GPU executor — mark completed immediately (nothing to run)
                finish_task(task, &self.deque);
            }
            continue;
        }

        // Priority enforcement: defer Bulk tasks while higher-priority work
        // is in flight — unless the task has aged past its promotion
        // threshold, in which case it runs as Background so sustained
        // Interactive load can't starve it forever. Deadline tasks are
        // exempt: deferring one into the FIFO queue would silently drop it
        // out of EDF ordering.
        if (task->lane == Lane::Bulk && task->deadline_ns == 0) {
            uint64_t age = clock_ns() - task->enqueue_time_ns;
            if (age >= kBulkPromotionNs) {
                lane_counter(Lane::Bulk).fetch_sub(1, std::memory_order_relaxed);
                task->lane = Lane::Background;
                lane_counter(Lane::Background).fetch_add(1, std::memory_order_relaxed);
            } else if (!stopping &&
                       (lane_counter(Lane::Interactive).load(std::memory_order_acquire) > 0 ||
                        lane_counter(Lane::Background).load(std::memory_order_acquire) > 0)) {
                enqueue_task(task);
                // Deferring is not progress: back off as if idle so a worker
                // alone with ineligible Bulk work doesn't spin at 100% CPU.
                idle_backoff();
                continue;
            }
        }

        backoff_us = 0;

        // Track deadline misses
        if (task->deadline_ns != 0 && clock_ns() > task->deadline_ns) {
            deadline_misses_.fetch_add(1, std::memory_order_relaxed);
        }

        // Execute
        if (task->fn) {
            task->fn();
        }
        finish_task(task, &self.deque);
    }
}

void Scheduler::io_worker_loop() {
    uint32_t backoff_us = 0;
    constexpr uint32_t kMaxBackoffUs = 1000;

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

        // IO threads run only fn (or skip it when cancelled). Dependents
        // likely belong to other lanes (e.g. GPU compute after an SSD read),
        // so finish_task routes them to their lane queues (null deque).
        if (!task->cancelled.load(std::memory_order_acquire) && task->fn) {
            task->fn();
        }
        finish_task(task, nullptr);
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

} // namespace rais
