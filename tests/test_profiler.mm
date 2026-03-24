#include <catch2/catch_test_macros.hpp>

#import <Metal/Metal.h>

#include <rais/profiler.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

static std::string read_file(const char* path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
}

TEST_CASE("TraceBuffer — single producer/consumer correctness", "[profiler]") {
    rais::TraceBuffer buf(1024);

    REQUIRE(buf.capacity() == 1024);
    REQUIRE(buf.size() == 0);

    // Produce 100 events
    for (uint64_t i = 0; i < 100; ++i) {
        rais::TraceEvent ev{};
        ev.timestamp_ns = i;
        ev.name = "test";
        ev.category = rais::TraceCategory::Scheduler;
        ev.phase = rais::TracePhase::Begin;
        ev.arg0 = i;
        REQUIRE(buf.produce(ev));
    }

    REQUIRE(buf.size() == 100);
    REQUIRE(buf.drops() == 0);

    // Consume all 100 — verify order
    for (uint64_t i = 0; i < 100; ++i) {
        rais::TraceEvent ev{};
        REQUIRE(buf.consume(ev));
        REQUIRE(ev.timestamp_ns == i);
        REQUIRE(ev.arg0 == i);
    }

    REQUIRE(buf.size() == 0);
}

TEST_CASE("TraceBuffer — drops when full", "[profiler]") {
    rais::TraceBuffer buf(64); // small capacity

    // Fill it completely
    for (size_t i = 0; i < 64; ++i) {
        rais::TraceEvent ev{};
        ev.timestamp_ns = i;
        ev.name = "fill";
        ev.category = rais::TraceCategory::Queue;
        ev.phase = rais::TracePhase::Instant;
        REQUIRE(buf.produce(ev));
    }

    // Next produce should drop
    rais::TraceEvent overflow{};
    overflow.name = "dropped";
    overflow.category = rais::TraceCategory::Queue;
    overflow.phase = rais::TracePhase::Instant;
    REQUIRE_FALSE(buf.produce(overflow));
    REQUIRE(buf.drops() == 1);

    // Consume one to make room, then produce should succeed
    rais::TraceEvent out{};
    REQUIRE(buf.consume(out));
    REQUIRE(buf.produce(overflow));
}

TEST_CASE("TraceBuffer — multi-producer stress", "[profiler]") {
    constexpr size_t BUF_CAP = 65536;
    constexpr size_t EVENTS_PER_THREAD = 10000;
    constexpr size_t NUM_PRODUCERS = 4;

    rais::TraceBuffer buf(BUF_CAP);

    std::atomic<bool> go{false};
    std::vector<std::thread> producers;
    producers.reserve(NUM_PRODUCERS);

    for (size_t t = 0; t < NUM_PRODUCERS; ++t) {
        producers.emplace_back([&buf, &go, t]() {
            while (!go.load(std::memory_order_acquire)) {}
            for (size_t i = 0; i < EVENTS_PER_THREAD; ++i) {
                rais::TraceEvent ev{};
                ev.timestamp_ns = rais::clock_ns();
                ev.name = "stress";
                ev.category = rais::TraceCategory::User;
                ev.phase = rais::TracePhase::Instant;
                ev.arg0 = t;
                ev.arg1 = i;
                buf.produce(ev); // may drop under contention, that's OK
            }
        });
    }

    go.store(true, std::memory_order_release);
    for (auto& t : producers) t.join();

    // Drain everything
    size_t consumed = 0;
    rais::TraceEvent ev{};
    while (buf.consume(ev)) ++consumed;

    // consumed + drops should equal total produced attempts
    REQUIRE(consumed + buf.drops() == NUM_PRODUCERS * EVENTS_PER_THREAD);
}

TEST_CASE("Profiler — writes valid Chrome Tracing JSON", "[profiler]") {
    const char* path = "/tmp/rais_test_profiler.json";

    {
        rais::Profiler profiler(path);
        profiler.start();

        // Record some events across categories
        profiler.trace("submit", rais::TraceCategory::Scheduler,
                       rais::TracePhase::Instant, 42, 0);
        profiler.trace("worker_run", rais::TraceCategory::Scheduler,
                       rais::TracePhase::Begin, 1, 0);
        profiler.trace("worker_run", rais::TraceCategory::Scheduler,
                       rais::TracePhase::End, 1, 0);
        profiler.trace("slab_alloc", rais::TraceCategory::Allocator,
                       rais::TracePhase::Instant, 64, 0);

        // Give drain thread time to process
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        profiler.stop();
    }

    std::string json = read_file(path);

    // Must start with '[' and end with ']'
    REQUIRE(!json.empty());
    REQUIRE(json.front() == '[');

    // Find closing bracket (file may have trailing nulls from mmap truncation)
    auto last_bracket = json.rfind(']');
    REQUIRE(last_bracket != std::string::npos);

    // Must contain our event names
    REQUIRE(json.find("\"submit\"") != std::string::npos);
    REQUIRE(json.find("\"worker_run\"") != std::string::npos);
    REQUIRE(json.find("\"slab_alloc\"") != std::string::npos);

    // Must contain category names
    REQUIRE(json.find("\"scheduler\"") != std::string::npos);
    REQUIRE(json.find("\"allocator\"") != std::string::npos);

    // Must contain phase characters
    REQUIRE(json.find("\"ph\":\"I\"") != std::string::npos); // Instant
    REQUIRE(json.find("\"ph\":\"B\"") != std::string::npos); // Begin
    REQUIRE(json.find("\"ph\":\"E\"") != std::string::npos); // End

    std::remove(path);
}

TEST_CASE("Profiler — Span RAII generates Begin/End pairs", "[profiler]") {
    const char* path = "/tmp/rais_test_profiler_span.json";

    {
        rais::Profiler profiler(path);
        profiler.start();

        {
            rais::Profiler::Span span(profiler, "my_span",
                                       rais::TraceCategory::User, 99, 0);
            // span exists for this block
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        profiler.stop();
    }

    std::string json = read_file(path);
    REQUIRE(json.find("\"my_span\"") != std::string::npos);

    // Count occurrences of "my_span" paired with Begin vs End phase.
    // Each JSON event is a single object on one line (comma-separated).
    // Search for the combined pattern to avoid region overlap.
    size_t begin_count = 0;
    size_t end_count = 0;

    // Count "my_span" + "ph":"B" co-occurrences
    {
        std::string pattern = "\"my_span\"";
        size_t p = 0;
        while ((p = json.find(pattern, p)) != std::string::npos) {
            // Find the enclosing JSON object: scan back to '{' and forward to '}'
            size_t obj_start = json.rfind('{', p);
            size_t obj_end = json.find('}', p);
            if (obj_start != std::string::npos && obj_end != std::string::npos) {
                std::string obj = json.substr(obj_start, obj_end - obj_start + 1);
                if (obj.find("\"ph\":\"B\"") != std::string::npos) ++begin_count;
                if (obj.find("\"ph\":\"E\"") != std::string::npos) ++end_count;
            }
            p += pattern.size();
        }
    }
    REQUIRE(begin_count == 1);
    REQUIRE(end_count == 1);

    std::remove(path);
}

TEST_CASE("Profiler — GPU timestamp recording", "[profiler]") {
    const char* path = "/tmp/rais_test_profiler_gpu.json";

    {
        rais::Profiler profiler(path);
        profiler.start();

        uint64_t gpu_start = 1'000'000'000ULL; // 1s in ns
        uint64_t gpu_end   = 1'000'500'000ULL; // 1s + 500µs

        profiler.record_gpu_timestamps(gpu_start, gpu_end, 7);

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        profiler.stop();
    }

    std::string json = read_file(path);
    REQUIRE(json.find("\"gpu_command_buffer\"") != std::string::npos);
    REQUIRE(json.find("\"gpu\"") != std::string::npos); // category
    REQUIRE(json.find("\"gpu_ns\"") != std::string::npos);

    std::remove(path);
}

TEST_CASE("Profiler — drop counter tracks overflow", "[profiler]") {
    const char* path = "/tmp/rais_test_profiler_drops.json";

    {
        // Tiny buffer — will overflow easily
        rais::Profiler profiler(path, nullptr, 64);
        // Don't start drain thread — events will pile up and drop
        for (int i = 0; i < 200; ++i) {
            profiler.trace("flood", rais::TraceCategory::Queue,
                           rais::TracePhase::Instant, static_cast<uint64_t>(i));
        }

        REQUIRE(profiler.drops() > 0);
        profiler.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        profiler.stop();
    }

    std::remove(path);
}
