#include <rais/queue.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

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
    // Determine path relative to executable — we expect to run from build dir,
    // results.tsv lives at repo_root/experiments/results.tsv
    const char* path = "../experiments/results.tsv";

    // Check if file exists to decide whether to write header
    bool needs_header = false;
    {
        std::ifstream check(path);
        needs_header = !check.good() || check.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream out(path, std::ios::app);
    if (!out) {
        std::cerr << "Warning: could not open " << path << " for writing\n";
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

// Returns total ops/sec.
template <typename QueueType>
static double bench_mpmc(QueueType& q, size_t num_producers, size_t num_consumers,
                         size_t items_per_producer) {
    std::atomic<bool> start{false};
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    producers.reserve(num_producers);
    consumers.reserve(num_consumers);

    size_t total_items = num_producers * items_per_producer;
    std::atomic<size_t> consumed_count{0};

    for (size_t i = 0; i < num_producers; ++i) {
        producers.emplace_back([&q, &start, i, items_per_producer]() {
            while (!start.load(std::memory_order_acquire)) {} // spin-wait for synchronized start
            uint64_t base = i * items_per_producer;
            for (uint64_t j = 0; j < items_per_producer; ++j) {
                while (!q.push(base + j)) {}
            }
        });
    }

    for (size_t i = 0; i < num_consumers; ++i) {
        consumers.emplace_back([&q, &start, &consumed_count, total_items]() {
            while (!start.load(std::memory_order_acquire)) {}
            uint64_t val = 0;
            while (consumed_count.load(std::memory_order_relaxed) < total_items) {
                if (q.pop(val)) {
                    consumed_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    start.store(true, std::memory_order_release);

    for (auto& t : producers) t.join();
    for (auto& t : consumers) t.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    return static_cast<double>(total_items) / elapsed_s;
}

int main() {
    constexpr size_t NUM_PRODUCERS = 4;
    constexpr size_t NUM_CONSUMERS = 4;
    constexpr size_t ITEMS_PER_PRODUCER = 1'000'000;
    constexpr size_t QUEUE_CAPACITY = 65536; // power of two

    std::vector<BenchResult> results;

    // Benchmark padded (production) queue
    {
        rais::MPMCQueue<uint64_t> q(QUEUE_CAPACITY);
        double ops = bench_mpmc(q, NUM_PRODUCERS, NUM_CONSUMERS, ITEMS_PER_PRODUCER);
        std::printf("Padded   MPMC %zup%zuc: %.0f ops/sec\n", NUM_PRODUCERS, NUM_CONSUMERS, ops);
        results.push_back({"queue_mpmc_4p4c", "throughput",
                           ops, "ops/sec", "with padding"});
    }

    // Benchmark unpadded queue to show false-sharing impact
    {
        rais::MPMCQueueUnpadded<uint64_t> q(QUEUE_CAPACITY);
        double ops = bench_mpmc(q, NUM_PRODUCERS, NUM_CONSUMERS, ITEMS_PER_PRODUCER);
        std::printf("Unpadded MPMC %zup%zuc: %.0f ops/sec\n", NUM_PRODUCERS, NUM_CONSUMERS, ops);
        results.push_back({"queue_mpmc_4p4c", "throughput",
                           ops, "ops/sec", "no padding"});
    }

    append_results(results);
    std::printf("Results appended to experiments/results.tsv\n");

    return 0;
}
