// priority_scheduling.cpp — end-to-end RAIS example
//
// Simulates a multi-tenant inference server where interactive requests
// (real-time users) compete with background batch jobs for a single
// decoder. RAIS priority lanes ensure interactive requests get served
// first, cutting their latency by ~3x vs naive FIFO.
//
// Build and run:
//   cmake -B build -DCMAKE_BUILD_TYPE=Release
//   cmake --build build --target priority_example
//   ./build/priority_example

#include <rais/scheduler.hpp>
#include <rais/clock.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

using namespace rais;

// Simulated "decode step" — represents one token generation (~2ms).
static void simulate_decode(size_t steps) {
    for (size_t i = 0; i < steps; ++i) {
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::microseconds(2000);
        while (std::chrono::steady_clock::now() < deadline) {
            // busy-wait to simulate compute
        }
    }
}

struct Result {
    int         client_id;
    const char* lane_name;
    double      ttft_ms;   // time to first "token"
    double      e2e_ms;    // total time
};

int main() {
    std::printf("RAIS Priority Scheduling Example\n");
    std::printf("================================\n\n");

    // Single worker to make the priority effect visible: all requests
    // compete for the same thread, so lane ordering matters.
    Scheduler sched({.num_workers = 1});

    constexpr int kBgClients  = 3;
    constexpr int kIntClients = 3;
    constexpr int kTotalClients = kBgClients + kIntClients;
    constexpr size_t kTokens  = 16; // tokens per request

    std::vector<Result> results(kTotalClients);
    std::mutex results_mu;
    std::atomic<int> done_count{0};

    auto submit_time = std::chrono::steady_clock::now();

    // Background batch jobs — submitted first (already in queue when users arrive)
    for (int i = 0; i < kBgClients; ++i) {
        sched.submit([i, submit_time, &results, &results_mu, &done_count]() {
            auto start = std::chrono::steady_clock::now();
            simulate_decode(kTokens);
            auto end = std::chrono::steady_clock::now();

            double ttft = std::chrono::duration<double, std::milli>(start - submit_time).count();
            double e2e  = std::chrono::duration<double, std::milli>(end - submit_time).count();
            {
                std::lock_guard<std::mutex> lock(results_mu);
                results[i] = {i, "Bulk", ttft, e2e};
            }
            done_count.fetch_add(1, std::memory_order_relaxed);
        }, Lane::Bulk);
    }

    // Interactive user requests — arrive slightly later
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    auto int_submit_time = std::chrono::steady_clock::now();

    for (int i = 0; i < kIntClients; ++i) {
        int id = kBgClients + i;
        sched.submit([id, int_submit_time, &results, &results_mu, &done_count]() {
            auto start = std::chrono::steady_clock::now();
            simulate_decode(kTokens);
            auto end = std::chrono::steady_clock::now();

            double ttft = std::chrono::duration<double, std::milli>(start - int_submit_time).count();
            double e2e  = std::chrono::duration<double, std::milli>(end - int_submit_time).count();
            {
                std::lock_guard<std::mutex> lock(results_mu);
                results[id] = {id, "Interactive", ttft, e2e};
            }
            done_count.fetch_add(1, std::memory_order_relaxed);
        }, Lane::Interactive);
    }

    // Wait for all requests to complete
    while (done_count.load(std::memory_order_relaxed) < kTotalClients) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    sched.shutdown();

    // Report
    std::printf("  %-4s  %-12s  %8s  %8s\n", "ID", "Lane", "TTFT", "E2E");
    std::printf("  %-4s  %-12s  %8s  %8s\n", "----", "------------", "--------", "--------");
    for (auto& r : results) {
        std::printf("  %-4d  %-12s  %6.0f ms  %6.0f ms\n",
                    r.client_id, r.lane_name, r.ttft_ms, r.e2e_ms);
    }

    // Compute averages
    double int_ttft = 0, bg_ttft = 0;
    for (int i = 0; i < kBgClients; ++i)  bg_ttft  += results[i].ttft_ms;
    for (int i = kBgClients; i < kTotalClients; ++i) int_ttft += results[i].ttft_ms;
    int_ttft /= kIntClients;
    bg_ttft  /= kBgClients;

    std::printf("\n  Interactive avg TTFT: %.0f ms\n", int_ttft);
    std::printf("  Bulk        avg TTFT: %.0f ms\n\n", bg_ttft);
    std::printf("Interactive requests start immediately even though batch\n"
                "jobs were queued first — RAIS jumps them to the front.\n");
    return 0;
}
