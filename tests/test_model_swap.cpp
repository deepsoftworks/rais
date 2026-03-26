#include <rais/model_manager.hpp>
#include <rais/scheduler.hpp>

#include <atomic>
#include <chrono>
#include <thread>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("swap completes and on_ready fires", "[model_swap]") {
    rais::Scheduler sched({.num_workers = 2});
    rais::ModelManager manager(sched);

    std::atomic<bool> loaded{false};
    std::atomic<bool> unloaded{false};
    std::atomic<bool> ready{false};

    auto h = manager.swap(
        "/fake/model/path",
        [&](const std::filesystem::path&) {
            loaded.store(true, std::memory_order_relaxed);
            return true;
        },
        [&]() { unloaded.store(true, std::memory_order_relaxed); },
        [&]() { ready.store(true, std::memory_order_relaxed); }
    );

    h.wait();

    REQUIRE(loaded.load(std::memory_order_relaxed));
    REQUIRE(unloaded.load(std::memory_order_relaxed));
    REQUIRE(ready.load(std::memory_order_relaxed));
    REQUIRE_FALSE(manager.swap_in_progress());
}

TEST_CASE("Failed load: old model stays, on_ready never fires", "[model_swap]") {
    rais::Scheduler sched({.num_workers = 2});
    rais::ModelManager manager(sched);

    std::atomic<bool> unloaded{false};
    std::atomic<bool> ready{false};

    auto h = manager.swap(
        "/fake/model/path",
        [](const std::filesystem::path&) {
            return false; // load fails
        },
        [&]() { unloaded.store(true, std::memory_order_relaxed); },
        [&]() { ready.store(true, std::memory_order_relaxed); }
    );

    h.wait();

    // Load failed — unload and on_ready should NOT have fired
    REQUIRE_FALSE(unloaded.load(std::memory_order_relaxed));
    REQUIRE_FALSE(ready.load(std::memory_order_relaxed));
    REQUIRE_FALSE(manager.swap_in_progress());
}

TEST_CASE("Double swap: second cancels first", "[model_swap]") {
    rais::Scheduler sched({.num_workers = 2});
    rais::ModelManager manager(sched);

    std::atomic<bool> first_ready{false};
    std::atomic<bool> second_ready{false};

    // First swap: slow load
    auto h1 = manager.swap(
        "/fake/model/1",
        [](const std::filesystem::path&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            return true;
        },
        []() {},
        [&]() { first_ready.store(true, std::memory_order_relaxed); }
    );

    // Immediately issue second swap — should cancel the first
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto h2 = manager.swap(
        "/fake/model/2",
        [](const std::filesystem::path&) { return true; },
        []() {},
        [&]() { second_ready.store(true, std::memory_order_relaxed); }
    );

    h2.wait();

    REQUIRE(second_ready.load(std::memory_order_relaxed));
    REQUIRE_FALSE(manager.swap_in_progress());
}

TEST_CASE("swap_in_progress is true during swap", "[model_swap]") {
    rais::Scheduler sched({.num_workers = 2});
    rais::ModelManager manager(sched);

    std::atomic<bool> in_load{false};
    std::atomic<bool> proceed{false};

    auto h = manager.swap(
        "/fake/model",
        [&](const std::filesystem::path&) {
            in_load.store(true, std::memory_order_release);
            while (!proceed.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return true;
        },
        []() {},
        []() {}
    );

    // Wait for load to begin
    while (!in_load.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    REQUIRE(manager.swap_in_progress());

    proceed.store(true, std::memory_order_release);
    h.wait();

    REQUIRE_FALSE(manager.swap_in_progress());
}
