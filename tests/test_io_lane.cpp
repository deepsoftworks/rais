#include <rais/scheduler.hpp>
#include <rais/streaming.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <thread>
#include <unistd.h>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace {

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

// Create a temp file with known contents. Returns the path.
std::filesystem::path create_temp_file(const std::string& name,
                                       const std::vector<uint8_t>& data) {
    auto path = std::filesystem::temp_directory_path() / ("rais_test_" + name);
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
    return path;
}

} // namespace

TEST_CASE("submit_file_read reads correct data", "[io_lane]") {
    // Create a temp file with a known pattern
    std::vector<uint8_t> expected(4096);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<uint8_t>(i & 0xFF);
    }
    auto path = create_temp_file("read_basic", expected);

    rais::Scheduler sched({.num_workers = 2});

    std::vector<uint8_t> buf(4096, 0);
    auto read_h = rais::submit_file_read(sched, path, 0, 4096, buf.data());

    // Chain a continuation to verify data
    std::atomic<bool> verified{false};
    auto verify_h = sched.then(read_h, [&]() {
        verified.store(std::memcmp(buf.data(), expected.data(), 4096) == 0,
                       std::memory_order_relaxed);
    }, rais::Lane::Background);

    verify_h.wait();
    REQUIRE(verified.load(std::memory_order_relaxed));

    std::filesystem::remove(path);
}

TEST_CASE("10 concurrent IO reads all complete", "[io_lane]") {
    constexpr size_t kNumReads = 10;
    constexpr size_t kFileSize = 8192;

    // Create 10 temp files
    std::vector<std::filesystem::path> paths;
    std::vector<std::vector<uint8_t>> expected(kNumReads);
    for (size_t i = 0; i < kNumReads; ++i) {
        expected[i].resize(kFileSize);
        std::memset(expected[i].data(), static_cast<int>(i + 1), kFileSize);
        paths.push_back(create_temp_file("concurrent_" + std::to_string(i),
                                         expected[i]));
    }

    rais::Scheduler sched({.num_workers = 4});

    std::vector<std::vector<uint8_t>> buffers(kNumReads,
                                               std::vector<uint8_t>(kFileSize, 0));
    std::vector<rais::TaskHandle> handles;

    for (size_t i = 0; i < kNumReads; ++i) {
        handles.push_back(rais::submit_file_read(
            sched, paths[i], 0, kFileSize, buffers[i].data()));
    }

    // Also submit an Interactive task concurrently — it should not be
    // blocked by the IO load since IO has dedicated threads.
    auto t0 = std::chrono::steady_clock::now();
    std::atomic<bool> interactive_done{false};
    auto interactive_h = sched.submit([&]() {
        interactive_done.store(true, std::memory_order_relaxed);
    }, rais::Lane::Interactive);
    interactive_h.wait();
    auto elapsed = std::chrono::steady_clock::now() - t0;
    REQUIRE(interactive_done.load(std::memory_order_relaxed));
    // Interactive task should complete quickly (< 100ms) even under IO load
    REQUIRE(elapsed < std::chrono::milliseconds(100));

    // Wait for all reads
    for (auto& h : handles) {
        h.wait();
    }

    // Verify all buffers
    for (size_t i = 0; i < kNumReads; ++i) {
        REQUIRE(std::memcmp(buffers[i].data(), expected[i].data(), kFileSize) == 0);
    }

    for (auto& p : paths) std::filesystem::remove(p);
}

TEST_CASE("IO tasks use dedicated threads, not CPU workers", "[io_lane]") {
    if (kTsanBuild) {
        SKIP("TSan build disables dedicated IO threads in Scheduler.");
    }

    // Occupy all CPU workers with slow Background tasks, then verify
    // IO tasks still complete because they have their own threads.
    rais::Scheduler sched({.num_workers = 2, .io_thread_count = 2});

    std::atomic<int> bg_running{0};
    std::atomic<bool> bg_release{false};

    // Submit Background tasks that block all CPU workers
    std::vector<rais::TaskHandle> bg_handles;
    for (int i = 0; i < 2; ++i) {
        bg_handles.push_back(sched.submit([&]() {
            bg_running.fetch_add(1, std::memory_order_relaxed);
            while (!bg_release.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }, rais::Lane::Background));
    }

    // Wait for workers to be occupied
    while (bg_running.load(std::memory_order_relaxed) < 2) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Now submit an IO task — it should complete even with CPU workers blocked
    constexpr size_t kSize = 1024;
    std::vector<uint8_t> expected(kSize, 0xAB);
    auto path = create_temp_file("io_dedicated", expected);

    std::vector<uint8_t> buf(kSize, 0);
    auto io_h = rais::submit_file_read(sched, path, 0, kSize, buf.data());

    // IO should complete within a reasonable time
    auto t0 = std::chrono::steady_clock::now();
    io_h.wait();
    auto elapsed = std::chrono::steady_clock::now() - t0;
    REQUIRE(elapsed < std::chrono::milliseconds(500));
    REQUIRE(std::memcmp(buf.data(), expected.data(), kSize) == 0);

    // Release background workers
    bg_release.store(true, std::memory_order_release);
    for (auto& h : bg_handles) h.wait();

    std::filesystem::remove(path);
}
