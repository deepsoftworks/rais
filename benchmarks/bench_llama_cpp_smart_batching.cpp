#include <rais/scheduler.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

// Realistic scenario: interactive requests are small, background batches are large
// RAIS advantage: coalesce multiple interactive requests into single batches

struct Request {
    int id;
    uint64_t submit_time;
    uint64_t dispatch_time;
    uint64_t end_time;
    int batch_size;  // how many reqs in same batch
    bool is_interactive;
};

// Simulates batched inference: more tokens = longer latency per request
static constexpr int INTERACTIVE_TOKENS = 10;      // short responses
static constexpr int BACKGROUND_TOKENS = 200;      // full generation
static constexpr int NS_PER_TOKEN = 50'000;         // 50µs per token

static void simulate_batched_inference(std::vector<Request>& batch) {
    // All requests in batch complete at the same time
    uint64_t start = rais::clock_ns();

    // Latency = max tokens in batch * time per token
    int max_tokens = 0;
    for (auto& req : batch) {
        max_tokens = std::max(max_tokens, req.batch_size);
    }

    uint64_t latency_ns = (uint64_t)max_tokens * NS_PER_TOKEN;
    std::this_thread::sleep_for(std::chrono::nanoseconds(latency_ns));

    uint64_t end = rais::clock_ns();
    for (auto& req : batch) {
        req.end_time = end;
    }
}

// FIFO: no batching optimization
struct FIFOBatcher {
    std::vector<Request*> queue;
    std::mutex queue_mutex;
    std::atomic<bool> running{false};
    std::vector<std::thread> workers;
    static constexpr int BATCH_SIZE = 32;

    FIFOBatcher(int num_workers) {
        running = true;
        for (int i = 0; i < num_workers; ++i) {
            workers.emplace_back([this]() { worker_thread(); });
        }
    }

    void worker_thread() {
        while (running) {
            std::vector<Request*> batch;
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                while (batch.size() < BATCH_SIZE && !queue.empty()) {
                    batch.push_back(queue.front());
                    queue.erase(queue.begin());
                }
            }

            if (!batch.empty()) {
                std::vector<Request> batch_reqs;
                for (auto* req_ptr : batch) {
                    req_ptr->dispatch_time = rais::clock_ns();
                    batch_reqs.push_back(*req_ptr);
                }
                simulate_batched_inference(batch_reqs);
                for (size_t i = 0; i < batch.size(); ++i) {
                    *batch[i] = batch_reqs[i];
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    void submit(Request* req) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        queue.push_back(req);
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

// RAIS: batches interactive requests aggressively
struct RAISBatcher {
    rais::Scheduler sched;
    std::vector<Request*> interactive_batch;
    std::mutex batch_mutex;
    std::atomic<int> pending_interactive{0};
    static constexpr int INTERACTIVE_BATCH_SIZE = 8;
    static constexpr int BATCH_WAIT_US = 100;  // coalesce up to 100µs worth

    RAISBatcher(int num_workers) : sched({.num_workers = static_cast<size_t>(num_workers)}) {}

    void submit(Request* req) {
        if (req->is_interactive) {
            {
                std::lock_guard<std::mutex> lock(batch_mutex);
                interactive_batch.push_back(req);
            }
            pending_interactive++;

            // If batch is full, dispatch immediately
            if (interactive_batch.size() >= INTERACTIVE_BATCH_SIZE) {
                dispatch_interactive_batch();
            } else {
                // Otherwise schedule lazy dispatch in worker
                sched.submit([this]() {
                    std::this_thread::sleep_for(std::chrono::microseconds(BATCH_WAIT_US));
                    dispatch_interactive_batch();
                }, rais::Lane::Interactive);
            }
        } else {
            // Submit background work directly
            sched.submit([req]() {
                std::vector<Request> batch{*req};
                simulate_batched_inference(batch);
                *req = batch[0];
            }, rais::Lane::Background);
        }
    }

    void dispatch_interactive_batch() {
        std::vector<Request*> batch_to_send;
        {
            std::lock_guard<std::mutex> lock(batch_mutex);
            if (interactive_batch.empty()) return;
            batch_to_send = std::move(interactive_batch);
            interactive_batch.clear();
        }

        if (!batch_to_send.empty()) {
            auto batch_copy = batch_to_send;  // capture for lambda
            sched.submit([batch_copy]() {
                std::vector<Request> reqs;
                for (auto* req : batch_copy) {
                    req->dispatch_time = rais::clock_ns();
                    reqs.push_back(*req);
                }
                simulate_batched_inference(reqs);
                for (size_t i = 0; i < batch_copy.size(); ++i) {
                    *batch_copy[i] = reqs[i];
                }
            }, rais::Lane::Interactive);

            pending_interactive -= batch_copy.size();
        }
    }

    void shutdown() {
        dispatch_interactive_batch();  // flush any remaining
        sched.shutdown();
    }
};

struct Result {
    const char* name;
    int num_requests;
    uint64_t p50, p95, p99, mean;
};

static Result benchmark_fifo(int num_requests, int num_workers) {
    printf("\n=== FIFO Batcher ===\n");
    FIFOBatcher fifo(num_workers);

    std::vector<Request> requests(num_requests);
    auto start = rais::clock_ns();

    for (int i = 0; i < num_requests; ++i) {
        requests[i].id = i;
        requests[i].submit_time = rais::clock_ns();
        requests[i].is_interactive = (i % 10 < 2);  // 20% interactive
        requests[i].batch_size = requests[i].is_interactive ? INTERACTIVE_TOKENS : BACKGROUND_TOKENS;
        fifo.submit(&requests[i]);
    }

    fifo.shutdown();
    auto total = rais::clock_ns() - start;

    std::vector<uint64_t> latencies;
    for (const auto& req : requests) {
        if (req.end_time > req.submit_time) {
            latencies.push_back(req.end_time - req.submit_time);
        }
    }
    std::sort(latencies.begin(), latencies.end());

    Result r;
    r.name = "FIFO";
    r.num_requests = latencies.size();
    r.p50 = latencies[latencies.size() * 0.50];
    r.p95 = latencies[latencies.size() * 0.95];
    r.p99 = latencies[latencies.size() * 0.99];
    uint64_t sum = 0;
    for (auto l : latencies) sum += l;
    r.mean = sum / latencies.size();

    printf("  P50: %.2f ms  P95: %.2f ms  P99: %.2f ms  Mean: %.2f ms  Total: %.2f ms\n",
           r.p50 / 1e6, r.p95 / 1e6, r.p99 / 1e6, r.mean / 1e6, total / 1e6);

    return r;
}

static Result benchmark_rais(int num_requests, int num_workers) {
    printf("\n=== RAIS Smart Batcher ===\n");
    RAISBatcher rais(num_workers);

    std::vector<Request> requests(num_requests);
    auto start = rais::clock_ns();

    for (int i = 0; i < num_requests; ++i) {
        requests[i].id = i;
        requests[i].submit_time = rais::clock_ns();
        requests[i].is_interactive = (i % 10 < 2);  // 20% interactive
        requests[i].batch_size = requests[i].is_interactive ? INTERACTIVE_TOKENS : BACKGROUND_TOKENS;
        rais.submit(&requests[i]);
    }

    rais.shutdown();
    auto total = rais::clock_ns() - start;

    std::vector<uint64_t> latencies;
    for (const auto& req : requests) {
        if (req.end_time > req.submit_time) {
            latencies.push_back(req.end_time - req.submit_time);
        }
    }
    std::sort(latencies.begin(), latencies.end());

    Result r;
    r.name = "RAIS";
    r.num_requests = latencies.size();
    r.p50 = latencies[latencies.size() * 0.50];
    r.p95 = latencies[latencies.size() * 0.95];
    r.p99 = latencies[latencies.size() * 0.99];
    uint64_t sum = 0;
    for (auto l : latencies) sum += l;
    r.mean = sum / latencies.size();

    printf("  P50: %.2f ms  P95: %.2f ms  P99: %.2f ms  Mean: %.2f ms  Total: %.2f ms\n",
           r.p50 / 1e6, r.p95 / 1e6, r.p99 / 1e6, r.mean / 1e6, total / 1e6);

    return r;
}

int main(int argc, char** argv) {
    int num_requests = 200;
    int num_workers = 4;

    if (argc > 1) num_requests = std::atoi(argv[1]);
    if (argc > 2) num_workers = std::atoi(argv[2]);

    printf("Smart batching benchmark: %d requests, %d workers, 20%% interactive\n\n",
           num_requests, num_workers);

    auto fifo = benchmark_fifo(num_requests, num_workers);
    auto rais_result = benchmark_rais(num_requests, num_workers);

    printf("\n============================================================\n");
    printf("COMPARISON (lower is better)\n");
    printf("============================================================\n");
    printf("%-12s %12s %12s %8s\n", "Metric", "FIFO", "RAIS", "Speedup");
    printf("%-12s %12s %12s %8s\n", "--------", "----", "----", "-------");

    auto cmp = [](const char* name, uint64_t f, uint64_t r) {
        printf("%-12s %10.2f ms %10.2f ms  %.2fx\n", name, f / 1e6, r / 1e6, (double)f / r);
    };

    cmp("P50", fifo.p50, rais_result.p50);
    cmp("P95", fifo.p95, rais_result.p95);
    cmp("P99", fifo.p99, rais_result.p99);
    cmp("Mean", fifo.mean, rais_result.mean);

    return 0;
}
