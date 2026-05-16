#include <rais/scheduler.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <queue>
#include <thread>
#include <vector>
#include <mutex>

// Simulate llama.cpp inference with realistic latency characteristics
struct InferenceRequest {
    int id;
    uint64_t submit_time;
    uint64_t start_time;
    uint64_t end_time;
    int tokens_to_generate;
    rais::Lane lane;
};

static constexpr int DEFAULT_TOKENS = 50;
static constexpr int INTERACTIVE_LATENCY_US = 5000;  // 5ms per batch
static constexpr int BACKGROUND_LATENCY_US = 15000;  // 15ms per batch

// Simulate llama.cpp inference
static void simulate_inference(InferenceRequest& req) {
    req.start_time = rais::clock_ns();

    // Simulate token generation latency
    int latency_us = (req.lane == rais::Lane::Interactive)
        ? INTERACTIVE_LATENCY_US
        : BACKGROUND_LATENCY_US;

    std::this_thread::sleep_for(std::chrono::microseconds(latency_us));
    req.end_time = rais::clock_ns();
}

// Baseline: naive FIFO queue with fixed worker thread
struct FIFOScheduler {
    std::queue<InferenceRequest*> queue;
    std::mutex queue_mutex;
    std::atomic<bool> running{false};
    std::vector<std::thread> workers;

    FIFOScheduler(int num_workers) {
        running = true;
        for (int i = 0; i < num_workers; ++i) {
            workers.emplace_back([this]() { worker_thread(); });
        }
    }

    void worker_thread() {
        while (running) {
            InferenceRequest* req = nullptr;
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (!queue.empty()) {
                    req = queue.front();
                    queue.pop();
                }
            }

            if (req) {
                simulate_inference(*req);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    void submit(InferenceRequest* req) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        queue.push(req);
    }

    void shutdown() {
        while (true) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (queue.empty()) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        running = false;
        for (auto& w : workers) {
            w.join();
        }
    }
};

struct BenchResult {
    const char* scheduler_name;
    int num_requests;
    uint64_t p50, p95, p99, mean, min, max;
};

static BenchResult benchmark_fifo(int num_requests, int num_workers) {
    printf("\n=== FIFO Baseline (llama.cpp native) ===\n");
    printf("Submitting %d concurrent requests with %d workers...\n", num_requests, num_workers);

    FIFOScheduler fifo(num_workers);
    std::vector<InferenceRequest> requests(num_requests);

    // Initialize and submit all requests
    auto submit_start = rais::clock_ns();
    for (int i = 0; i < num_requests; ++i) {
        requests[i].id = i;
        requests[i].submit_time = rais::clock_ns();
        requests[i].tokens_to_generate = DEFAULT_TOKENS;
        requests[i].lane = (i % 2 == 0) ? rais::Lane::Interactive : rais::Lane::Background;
        fifo.submit(&requests[i]);
    }

    fifo.shutdown();
    auto total_time = rais::clock_ns() - submit_start;

    // Compute latencies (submit to end)
    std::vector<uint64_t> latencies;
    for (const auto& req : requests) {
        if (req.end_time > req.submit_time) {
            latencies.push_back(req.end_time - req.submit_time);
        }
    }

    std::sort(latencies.begin(), latencies.end());

    BenchResult result;
    result.scheduler_name = "FIFO";
    result.num_requests = latencies.size();
    result.p50 = latencies[latencies.size() * 0.50];
    result.p95 = latencies[latencies.size() * 0.95];
    result.p99 = latencies[latencies.size() * 0.99];
    result.min = latencies.front();
    result.max = latencies.back();

    uint64_t sum = 0;
    for (auto l : latencies) sum += l;
    result.mean = sum / latencies.size();

    printf("  P50: %llu ns (%.2f ms)\n", result.p50, result.p50 / 1e6);
    printf("  P95: %llu ns (%.2f ms)\n", result.p95, result.p95 / 1e6);
    printf("  P99: %llu ns (%.2f ms)\n", result.p99, result.p99 / 1e6);
    printf("  Mean: %llu ns (%.2f ms)\n", result.mean, result.mean / 1e6);
    printf("  Total time: %.2f ms\n", total_time / 1e6);

    return result;
}

static BenchResult benchmark_rais(int num_requests, int num_workers) {
    printf("\n=== RAIS Scheduler (with priority lanes) ===\n");
    printf("Submitting %d concurrent requests with %d workers...\n", num_requests, num_workers);

    rais::Scheduler sched({.num_workers = static_cast<size_t>(num_workers)});
    std::vector<InferenceRequest> requests(num_requests);
    std::vector<rais::TaskHandle> handles;
    handles.reserve(num_requests);

    // Submit all requests with lane-based priority
    auto submit_start = rais::clock_ns();
    for (int i = 0; i < num_requests; ++i) {
        requests[i].id = i;
        requests[i].submit_time = rais::clock_ns();
        requests[i].tokens_to_generate = DEFAULT_TOKENS;
        requests[i].lane = (i % 2 == 0) ? rais::Lane::Interactive : rais::Lane::Background;

        auto lane = requests[i].lane;
        auto& req = requests[i];

        handles.push_back(sched.submit([&req]() {
            simulate_inference(req);
        }, lane));
    }

    // Wait for all to complete
    for (auto& h : handles) {
        h.wait();
    }

    sched.shutdown();
    auto total_time = rais::clock_ns() - submit_start;

    // Compute latencies (submit to end)
    std::vector<uint64_t> latencies;
    for (const auto& req : requests) {
        if (req.end_time > req.submit_time) {
            latencies.push_back(req.end_time - req.submit_time);
        }
    }

    std::sort(latencies.begin(), latencies.end());

    BenchResult result;
    result.scheduler_name = "RAIS";
    result.num_requests = latencies.size();
    result.p50 = latencies[latencies.size() * 0.50];
    result.p95 = latencies[latencies.size() * 0.95];
    result.p99 = latencies[latencies.size() * 0.99];
    result.min = latencies.front();
    result.max = latencies.back();

    uint64_t sum = 0;
    for (auto l : latencies) sum += l;
    result.mean = sum / latencies.size();

    printf("  P50: %llu ns (%.2f ms)\n", result.p50, result.p50 / 1e6);
    printf("  P95: %llu ns (%.2f ms)\n", result.p95, result.p95 / 1e6);
    printf("  P99: %llu ns (%.2f ms)\n", result.p99, result.p99 / 1e6);
    printf("  Mean: %llu ns (%.2f ms)\n", result.mean, result.mean / 1e6);
    printf("  Total time: %.2f ms\n", total_time / 1e6);

    return result;
}

int main(int argc, char** argv) {
    int num_requests = 100;
    int num_workers = 4;

    if (argc > 1) num_requests = std::atoi(argv[1]);
    if (argc > 2) num_workers = std::atoi(argv[2]);

    printf("Concurrent llama.cpp inference scheduling benchmark\n");
    printf("Requests: %d, Workers: %d\n\n", num_requests, num_workers);

    // Run both benchmarks
    auto fifo_result = benchmark_fifo(num_requests, num_workers);
    auto rais_result = benchmark_rais(num_requests, num_workers);

    // Print comparison
    printf("\n============================================================\n");
    printf("COMPARISON (lower is better)\n");
    printf("============================================================\n");
    printf("Metric       FIFO           RAIS           Ratio\n");
    printf("------------------------------------------------------------\n");

    auto print_metric = [](const char* name, uint64_t fifo_val, uint64_t rais_val) {
        double ratio = (double)fifo_val / rais_val;
        printf("%-12s %8llu ns    %8llu ns    %.2fx\n", name, fifo_val, rais_val, ratio);
    };

    print_metric("P50", fifo_result.p50, rais_result.p50);
    print_metric("P95", fifo_result.p95, rais_result.p95);
    print_metric("P99", fifo_result.p99, rais_result.p99);
    print_metric("Mean", fifo_result.mean, rais_result.mean);

    return 0;
}
