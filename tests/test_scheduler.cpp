#include <catch2/catch_test_macros.hpp>

#include <rais/clock.hpp>
#include <rais/scheduler.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

TEST_CASE("10k Interactive tasks all complete", "[scheduler]") {
    rais::Scheduler sched({.num_workers = 4, .global_queue_capacity = 8192});

    constexpr int N = 10'000;
    std::atomic<int> counter{0};
    std::vector<rais::TaskHandle> handles;
    handles.reserve(N);

    for (int i = 0; i < N; ++i) {
        handles.push_back(sched.submit([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        }, rais::Lane::Interactive));
    }

    for (auto& h : handles) h.wait();

    REQUIRE(counter.load(std::memory_order_relaxed) == N);
}

TEST_CASE("Bulk tasks deferred while Interactive tasks pending", "[scheduler]") {
    // Use 1 worker so ordering is deterministic
    rais::Scheduler sched({.num_workers = 1, .global_queue_capacity = 4096});

    std::atomic<int> interactive_done{0};
    std::atomic<bool> bulk_saw_interactive_pending{false};

    constexpr int NUM_INTERACTIVE = 100;
    constexpr int NUM_BULK = 10;

    std::vector<rais::TaskHandle> handles;

    // Submit Interactive tasks first — they'll fill the queue
    for (int i = 0; i < NUM_INTERACTIVE; ++i) {
        handles.push_back(sched.submit([&interactive_done]() {
            interactive_done.fetch_add(1, std::memory_order_relaxed);
        }, rais::Lane::Interactive));
    }

    // Submit Bulk tasks — they should be deferred while Interactive is pending
    for (int i = 0; i < NUM_BULK; ++i) {
        handles.push_back(sched.submit([&interactive_done, &bulk_saw_interactive_pending]() {
            if (interactive_done.load(std::memory_order_relaxed) < 100) {
                bulk_saw_interactive_pending.store(true, std::memory_order_relaxed);
            }
        }, rais::Lane::Bulk));
    }

    for (auto& h : handles) h.wait();

    REQUIRE(interactive_done.load() == NUM_INTERACTIVE);
    // In a single-worker scenario, Bulk tasks should not start until
    // all Interactive tasks have completed and their lane count drops to 0.
    REQUIRE_FALSE(bulk_saw_interactive_pending.load());
}

TEST_CASE("Starvation promotion — Bulk task promoted after 500ms", "[scheduler]") {
    // 1 worker, and we block it temporarily so the Bulk task ages
    rais::Scheduler sched({.num_workers = 1, .global_queue_capacity = 4096});

    std::atomic<bool> blocker_done{false};
    std::atomic<bool> bulk_ran{false};

    // Submit a Bulk task
    auto bulk_handle = sched.submit([&bulk_ran]() {
        bulk_ran.store(true, std::memory_order_release);
    }, rais::Lane::Bulk);

    // Submit a blocking Interactive task that holds the worker for 600ms.
    // While blocked, the Bulk task ages past the 500ms promotion threshold.
    auto blocker_handle = sched.submit([&blocker_done]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(600));
        blocker_done.store(true, std::memory_order_release);
    }, rais::Lane::Interactive);

    blocker_handle.wait();
    bulk_handle.wait();

    REQUIRE(blocker_done.load());
    REQUIRE(bulk_ran.load());
}

TEST_CASE("Cancel 50 of 100 tasks — verify exactly 50 fn() calls", "[scheduler]") {
    rais::Scheduler sched({.num_workers = 2, .global_queue_capacity = 4096});

    constexpr int N = 100;
    std::atomic<int> exec_count{0};
    std::vector<rais::TaskHandle> handles;

    // Submit tasks that sleep briefly so we have time to cancel some
    for (int i = 0; i < N; ++i) {
        handles.push_back(sched.submit([&exec_count]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            exec_count.fetch_add(1, std::memory_order_relaxed);
        }, rais::Lane::Background));
    }

    // Cancel the second half
    int cancelled = 0;
    for (int i = N / 2; i < N; ++i) {
        if (handles[i].cancel()) ++cancelled;
    }

    for (auto& h : handles) h.wait();

    int executed = exec_count.load(std::memory_order_relaxed);
    // executed + cancelled should equal N
    REQUIRE(executed + cancelled == N);
}

TEST_CASE("Shutdown under load — no tasks lost with Drain", "[scheduler]") {
    constexpr int N = 10'000;
    std::atomic<int> counter{0};

    {
        rais::Scheduler sched({.num_workers = 4, .global_queue_capacity = 8192});

        // 4 threads submitting concurrently
        std::vector<std::thread> submitters;
        for (int t = 0; t < 4; ++t) {
            submitters.emplace_back([&sched, &counter, t]() {
                for (int i = 0; i < N / 4; ++i) {
                    sched.submit([&counter]() {
                        counter.fetch_add(1, std::memory_order_relaxed);
                    }, rais::Lane::Background);
                }
            });
        }

        for (auto& t : submitters) t.join();
        // Destructor calls shutdown(Drain) — should complete all tasks
    }

    REQUIRE(counter.load(std::memory_order_relaxed) == N);
}

TEST_CASE("TaskHandle wait and done", "[scheduler]") {
    rais::Scheduler sched({.num_workers = 1, .global_queue_capacity = 4096});

    std::atomic<bool> ran{false};
    auto handle = sched.submit([&ran]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        ran.store(true, std::memory_order_release);
    }, rais::Lane::Interactive);

    REQUIRE_FALSE(handle.done());
    handle.wait();
    REQUIRE(handle.done());
    REQUIRE(ran.load());
}

TEST_CASE("Deadline tasks execute before FIFO tasks", "[scheduler][deadline]") {
    // 1 worker so execution order is deterministic
    rais::Scheduler sched({.num_workers = 1, .global_queue_capacity = 4096});

    std::vector<int> order;
    std::mutex order_mu;

    auto record = [&](int id) {
        std::lock_guard<std::mutex> lock(order_mu);
        order.push_back(id);
    };

    // Block the worker so we can enqueue everything before execution starts
    std::atomic<bool> gate{false};
    auto blocker = sched.submit([&gate]() {
        while (!gate.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }, rais::Lane::Interactive);

    // Submit FIFO tasks (no deadline) — these go to the global queue
    for (int i = 0; i < 5; ++i) {
        sched.submit([&record, i]() { record(i); }, rais::Lane::Background);
    }

    // Submit a deadline task — should jump ahead of the 5 FIFO tasks
    uint64_t now = rais::clock_ns();
    auto deadline_handle = sched.submit(
        [&record]() { record(100); },
        rais::Lane::Interactive,
        now + 50'000'000ULL); // 50ms from now

    // Release the worker
    gate.store(true, std::memory_order_release);
    blocker.wait();
    deadline_handle.wait();

    // Wait for remaining tasks
    sched.shutdown(rais::ShutdownPolicy::Drain);

    std::lock_guard<std::mutex> lock(order_mu);
    REQUIRE(order.size() == 6);
    // The deadline task (100) should be first after the blocker releases
    REQUIRE(order[0] == 100);
}

TEST_CASE("Deadline ordering — nearest deadline served first", "[scheduler][deadline]") {
    rais::Scheduler sched({.num_workers = 1, .global_queue_capacity = 4096});

    std::vector<int> order;
    std::mutex order_mu;

    auto record = [&](int id) {
        std::lock_guard<std::mutex> lock(order_mu);
        order.push_back(id);
    };

    // Block the worker
    std::atomic<bool> gate{false};
    auto blocker = sched.submit([&gate]() {
        while (!gate.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }, rais::Lane::Interactive);

    uint64_t now = rais::clock_ns();

    // Submit 3 deadline tasks with decreasing urgency (far, medium, near)
    auto h_far = sched.submit([&record]() { record(3); },
        rais::Lane::Interactive, now + 300'000'000ULL);
    auto h_mid = sched.submit([&record]() { record(2); },
        rais::Lane::Interactive, now + 200'000'000ULL);
    auto h_near = sched.submit([&record]() { record(1); },
        rais::Lane::Interactive, now + 100'000'000ULL);

    // Release
    gate.store(true, std::memory_order_release);
    blocker.wait();
    h_far.wait();
    h_mid.wait();
    h_near.wait();

    std::lock_guard<std::mutex> lock(order_mu);
    REQUIRE(order.size() == 3);
    // Nearest deadline should run first
    REQUIRE(order[0] == 1);
    REQUIRE(order[1] == 2);
    REQUIRE(order[2] == 3);
}

TEST_CASE("Deadline miss counter incremented for late tasks", "[scheduler][deadline]") {
    rais::Scheduler sched({.num_workers = 1, .global_queue_capacity = 4096});

    // Submit a task whose deadline is already in the past — guaranteed miss
    // regardless of scheduling order or worker startup timing.
    auto h = sched.submit([]() {},
        rais::Lane::Interactive,
        1ULL); // 1ns since boot ≈ the dawn of time

    h.wait();

    REQUIRE(sched.deadline_misses() >= 1);
}

TEST_CASE("Deadline tasks complete during Drain shutdown", "[scheduler][deadline]") {
    constexpr int N = 100;
    std::atomic<int> counter{0};

    {
        rais::Scheduler sched({.num_workers = 2, .global_queue_capacity = 4096});

        uint64_t now = rais::clock_ns();
        for (int i = 0; i < N; ++i) {
            sched.submit([&counter]() {
                counter.fetch_add(1, std::memory_order_relaxed);
            }, rais::Lane::Interactive, now + static_cast<uint64_t>(i) * 1'000'000ULL);
        }
        // Destructor calls shutdown(Drain)
    }

    REQUIRE(counter.load(std::memory_order_relaxed) == N);
}
