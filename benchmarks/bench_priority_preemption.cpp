#include <rais/scheduler.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>
#include <mutex>

// Scenario: Long-running background task gets submitted, then interactive request arrives
// FIFO: interactive waits for background to finish
// RAIS: interactive goes to its own queue and starts sooner

struct Request {
    int id;
    uint64_t submit_time;
    uint64_t start_time;
    uint64_t end_time;
    int work_us;  // work duration in microseconds
    bool is_interactive;
};

static void do_work(Request& req) {
    req.start_time = rais::clock_ns();
    std::this_thread::sleep_for(std::chrono::microseconds(req.work_us));
    req.end_time = rais::clock_ns();
}

struct FIFOScheduler {
    std::atomic<bool> running{false};
    std::vector<Request*> queue;
    std::mutex queue_mutex;
    std::vector<std::thread> workers;

    FIFOScheduler(int num_workers) {
        running = true;
        for (int i = 0; i < num_workers; ++i) {
            workers.emplace_back([this]() {
                while (running) {
                    Request* req = nullptr;
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex);
                        if (!queue.empty()) {
                            req = queue.back();
                            queue.pop_back();
                        }
                    }
                    if (req) {
                        do_work(*req);
                    } else {
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                    }
                }
            });
        }
    }

    void submit(Request* req) {
        req->submit_time = rais::clock_ns();
        std::lock_guard<std::mutex> lock(queue_mutex);
        queue.push_back(req);
    }

    void shutdown() {
        while (true) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (queue.empty()) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        running = false;
        for (auto& w : workers) {
            w.join();
        }
    }
};

struct Scenario {
    const char* name;
    uint64_t worst_case_latency_fifo_ns;
    uint64_t worst_case_latency_rais_ns;
    double improvement;
};

static Scenario scenario_blocked_by_background(int num_workers) {
    printf("\n=== Scenario: Interactive task blocked by long background work ===\n");

    // FIFO version
    printf("FIFO: Submitting 100ms background task, then interactive task...\n");
    {
        FIFOScheduler fifo(num_workers);
        Request bg{};
        bg.id = 0;
        bg.work_us = 100'000;  // 100ms
        bg.is_interactive = false;

        Request fg{};
        fg.id = 1;
        fg.work_us = 1'000;  // 1ms
        fg.is_interactive = true;

        auto bg_submit = rais::clock_ns();
        fifo.submit(&bg);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));  // bg gets scheduled

        auto fg_submit = rais::clock_ns();
        fifo.submit(&fg);

        fifo.shutdown();

        uint64_t fg_wait_fifo = (fg.start_time - fg_submit);
        printf("  Interactive wait: %.2f ms (blocked behind 100ms background)\n", fg_wait_fifo / 1e6);
    }

    // RAIS version
    printf("RAIS: Same scenario with lane-based scheduling...\n");
    {
        rais::Scheduler sched({.num_workers = static_cast<size_t>(num_workers)});
        Request bg{};
        bg.id = 0;
        bg.work_us = 100'000;
        bg.is_interactive = false;

        Request fg{};
        fg.id = 1;
        fg.work_us = 1'000;
        fg.is_interactive = true;

        auto bg_submit = rais::clock_ns();
        bg.submit_time = bg_submit;
        sched.submit([&bg]() { do_work(bg); }, rais::Lane::Background);

        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        auto fg_submit = rais::clock_ns();
        fg.submit_time = fg_submit;
        auto handle = sched.submit([&fg]() { do_work(fg); }, rais::Lane::Interactive);
        handle.wait();

        sched.shutdown();

        uint64_t fg_wait_rais = (fg.start_time - fg_submit);
        printf("  Interactive wait: %.2f ms (own queue, minimal blocking)\n", fg_wait_rais / 1e6);
    }

    return {};
}

static void scenario_mixed_load(int num_workers) {
    printf("\n=== Scenario: 50 long background + 50 short interactive submitted in burst ===\n");

    // FIFO
    printf("FIFO...\n");
    {
        FIFOScheduler fifo(num_workers);
        std::vector<Request> requests(100);

        auto submit_start = rais::clock_ns();
        for (int i = 0; i < 100; ++i) {
            bool interactive = (i >= 50);  // first 50 background, last 50 interactive
            requests[i].id = i;
            requests[i].work_us = interactive ? 1'000 : 10'000;
            requests[i].is_interactive = interactive;
            fifo.submit(&requests[i]);
        }
        fifo.shutdown();

        // Interactive latencies
        std::vector<uint64_t> interactive_latencies;
        for (int i = 50; i < 100; ++i) {
            interactive_latencies.push_back(requests[i].end_time - requests[i].submit_time);
        }
        std::sort(interactive_latencies.begin(), interactive_latencies.end());

        uint64_t p50 = interactive_latencies[interactive_latencies.size() / 2];
        uint64_t p99 = interactive_latencies[interactive_latencies.size() * 99 / 100];

        printf("  Interactive P50: %.2f ms, P99: %.2f ms\n", p50 / 1e6, p99 / 1e6);
    }

    // RAIS
    printf("RAIS...\n");
    {
        rais::Scheduler sched({.num_workers = static_cast<size_t>(num_workers)});
        std::vector<Request> requests(100);
        std::vector<rais::TaskHandle> handles;

        auto submit_start = rais::clock_ns();
        for (int i = 0; i < 100; ++i) {
            bool interactive = (i >= 50);
            requests[i].id = i;
            requests[i].submit_time = rais::clock_ns();
            requests[i].work_us = interactive ? 1'000 : 10'000;
            requests[i].is_interactive = interactive;

            auto& req = requests[i];
            handles.push_back(sched.submit([&req]() { do_work(req); },
                interactive ? rais::Lane::Interactive : rais::Lane::Background));
        }

        for (auto& h : handles) h.wait();
        sched.shutdown();

        // Interactive latencies
        std::vector<uint64_t> interactive_latencies;
        for (int i = 50; i < 100; ++i) {
            interactive_latencies.push_back(requests[i].end_time - requests[i].submit_time);
        }
        std::sort(interactive_latencies.begin(), interactive_latencies.end());

        uint64_t p50 = interactive_latencies[interactive_latencies.size() / 2];
        uint64_t p99 = interactive_latencies[interactive_latencies.size() * 99 / 100];

        printf("  Interactive P50: %.2f ms, P99: %.2f ms\n", p50 / 1e6, p99 / 1e6);
    }
}

int main(int argc, char** argv) {
    int num_workers = 2;  // Fewer workers makes contention obvious
    if (argc > 1) num_workers = std::atoi(argv[1]);

    printf("Priority scheduling benchmark: %d worker(s)\n", num_workers);

    scenario_blocked_by_background(num_workers);
    scenario_mixed_load(num_workers);

    return 0;
}
