#include <rais/allocator.hpp>
#include <rais/clock.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// BenchResult + append_results (reused from bench_queue.cpp pattern)
// ---------------------------------------------------------------------------

static std::string today() {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&tt, &tm);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
    return buf;
}

static std::string build_config() {
#ifdef NDEBUG
    return "release";
#else
    return "debug";
#endif
}

struct BenchResult {
    std::string test_name;
    std::string metric;
    double value;
    std::string units;
    std::string notes;
};

static void append_results(const std::vector<BenchResult>& results) {
    const char* path = "../experiments/results.tsv";
    bool needs_header = false;
    {
        std::ifstream check(path);
        needs_header = !check.good() || check.peek() == std::ifstream::traits_type::eof();
    }
    std::ofstream out(path, std::ios::app);
    if (!out) {
        std::fprintf(stderr, "Warning: could not open %s for writing\n", path);
        return;
    }
    if (needs_header) {
        out << "date\tbuild_config\ttest_name\tmetric\tvalue\tunits\tnotes\n";
    }
    std::string date = today();
    std::string config = build_config();
    for (const auto& r : results) {
        out << date << "\t" << config << "\t" << r.test_name << "\t"
            << r.metric << "\t" << static_cast<uint64_t>(r.value) << "\t"
            << r.units << "\t" << r.notes << "\n";
    }
}

// ---------------------------------------------------------------------------
// Slab allocator benchmark
// ---------------------------------------------------------------------------

static void bench_slab_single_thread(std::vector<BenchResult>& results) {
    constexpr size_t N = 8192;
    constexpr size_t ITERS = 100'000;
    rais::SlabAllocator<uint64_t, N> slab;

    // Measure alloc/free latency
    std::vector<uint64_t> alloc_times;
    std::vector<uint64_t> free_times;
    alloc_times.reserve(ITERS);
    free_times.reserve(ITERS);

    for (size_t i = 0; i < ITERS; ++i) {
        uint64_t t0 = rais::clock_ns();
        uint64_t* p = slab.allocate();
        uint64_t t1 = rais::clock_ns();

        uint64_t t2 = rais::clock_ns();
        slab.free(p);
        uint64_t t3 = rais::clock_ns();

        alloc_times.push_back(t1 - t0);
        free_times.push_back(t3 - t2);
    }

    std::sort(alloc_times.begin(), alloc_times.end());
    std::sort(free_times.begin(), free_times.end());

    auto p50 = [](const std::vector<uint64_t>& v) { return v[v.size() / 2]; };
    auto p95 = [](const std::vector<uint64_t>& v) { return v[v.size() * 95 / 100]; };
    auto p99 = [](const std::vector<uint64_t>& v) { return v[v.size() * 99 / 100]; };

    std::printf("Slab alloc  P50=%llu P95=%llu P99=%llu ns\n",
                p50(alloc_times), p95(alloc_times), p99(alloc_times));
    std::printf("Slab free   P50=%llu P95=%llu P99=%llu ns\n",
                p50(free_times), p95(free_times), p99(free_times));

    results.push_back({"slab_alloc", "P50_latency", static_cast<double>(p50(alloc_times)), "ns", "single-thread"});
    results.push_back({"slab_alloc", "P95_latency", static_cast<double>(p95(alloc_times)), "ns", "single-thread"});
    results.push_back({"slab_alloc", "P99_latency", static_cast<double>(p99(alloc_times)), "ns", "single-thread"});
    results.push_back({"slab_free", "P50_latency", static_cast<double>(p50(free_times)), "ns", "single-thread"});
}

static void bench_slab_concurrent(std::vector<BenchResult>& results) {
    constexpr size_t N = 65536;
    constexpr size_t NUM_THREADS = 8;
    constexpr size_t OPS = 100'000;

    rais::SlabAllocator<uint64_t, N> slab;
    std::atomic<bool> start{false};

    std::vector<std::thread> threads;
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&]() {
            while (!start.load(std::memory_order_acquire)) {}
            for (size_t i = 0; i < OPS; ++i) {
                uint64_t* p = nullptr;
                while (!(p = slab.allocate())) {
                    std::this_thread::yield();
                }
                *p = i;
                slab.free(p);
            }
        });
    }

    auto wall_start = std::chrono::high_resolution_clock::now();
    start.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();
    auto wall_end = std::chrono::high_resolution_clock::now();

    double elapsed_s = std::chrono::duration<double>(wall_end - wall_start).count();
    double ops_sec = (NUM_THREADS * OPS * 2.0) / elapsed_s; // alloc + free = 2 ops
    std::printf("Slab concurrent %zu threads: %.0f ops/sec\n", NUM_THREADS, ops_sec);

    results.push_back({"slab_concurrent_8t", "throughput", ops_sec, "ops/sec", "alloc+free pairs"});
}

// ---------------------------------------------------------------------------
// Arena allocator benchmark
// ---------------------------------------------------------------------------

static void bench_arena(std::vector<BenchResult>& results) {
    constexpr size_t CAPACITY = 1024 * 1024; // 1MB
    constexpr size_t ALLOC_SIZE = 64;
    constexpr size_t ITERS = 100;

    rais::ArenaAllocator arena(CAPACITY);
    size_t allocs_per_reset = CAPACITY / ALLOC_SIZE;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t iter = 0; iter < ITERS; ++iter) {
        for (size_t i = 0; i < allocs_per_reset; ++i) {
            void* p = arena.allocate(ALLOC_SIZE);
            (void)p;
        }
        arena.reset();
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    double total_allocs = ITERS * allocs_per_reset;
    double ops_sec = total_allocs / elapsed_s;
    std::printf("Arena alloc: %.0f ops/sec (%zu-byte allocs)\n", ops_sec, ALLOC_SIZE);

    results.push_back({"arena_alloc", "throughput", ops_sec, "ops/sec", "64-byte allocs + reset"});
}

int main() {
    std::vector<BenchResult> results;

    bench_slab_single_thread(results);
    bench_slab_concurrent(results);
    bench_arena(results);

    append_results(results);
    std::printf("Results appended to experiments/results.tsv\n");

    return 0;
}
