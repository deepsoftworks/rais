#import <Metal/Metal.h>

#include <rais/memory_pressure.hpp>
#include <rais/metal_allocator.hpp>

#include <vector>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("acquire returns nullptr when budget exceeded", "[memory_pressure]") {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);

    rais::MetalBufferPool pool((__bridge void*)device);

    // Set a small budget: 128 KB
    pool.set_memory_budget(128 * 1024);

    // Allocate 64 KB — should succeed
    void* buf1 = pool.acquire(64 * 1024);
    REQUIRE(buf1 != nullptr);

    // Allocate another 64 KB — should succeed (total = 128 KB = budget)
    void* buf2 = pool.acquire(64 * 1024);
    REQUIRE(buf2 != nullptr);

    // Allocate one more — should fail (would exceed budget)
    void* buf3 = pool.acquire(4 * 1024);
    REQUIRE(buf3 == nullptr);

    pool.release(buf1);
    pool.release(buf2);
}

TEST_CASE("total_allocated_bytes tracks correctly", "[memory_pressure]") {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);

    rais::MetalBufferPool pool((__bridge void*)device);

    REQUIRE(pool.total_allocated_bytes() == 0);

    void* buf1 = pool.acquire(4 * 1024);
    REQUIRE(buf1 != nullptr);
    REQUIRE(pool.total_allocated_bytes() == 4 * 1024);

    void* buf2 = pool.acquire(64 * 1024);
    REQUIRE(buf2 != nullptr);
    REQUIRE(pool.total_allocated_bytes() == 4 * 1024 + 64 * 1024);

    pool.release(buf1);
    REQUIRE(pool.total_allocated_bytes() == 64 * 1024);

    pool.release(buf2);
    REQUIRE(pool.total_allocated_bytes() == 0);
}

TEST_CASE("MemoryMonitor reports correct pressure levels", "[memory_pressure]") {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);

    rais::MetalBufferPool pool((__bridge void*)device);
    // Budget: 10 * 64KB = 640KB. Use 64KB buffers (exact size class) for
    // predictable accounting — no rounding surprises.
    constexpr size_t kBufSize = 64 * 1024;
    pool.set_memory_budget(10 * kBufSize);

    rais::MemoryMonitor monitor(pool, 0.75f, 0.90f);

    // No allocations — Normal
    REQUIRE(monitor.check() == rais::MemoryPressure::Normal);
    REQUIRE_FALSE(monitor.under_pressure());

    // Allocate 5/10 = 50% — Normal
    std::vector<void*> bufs;
    for (int i = 0; i < 5; ++i) {
        void* b = pool.acquire(kBufSize);
        REQUIRE(b != nullptr);
        bufs.push_back(b);
    }
    REQUIRE(monitor.check() == rais::MemoryPressure::Normal);

    // Allocate to 8/10 = 80% — Warning (>= 0.75 threshold)
    for (int i = 0; i < 3; ++i) {
        void* b = pool.acquire(kBufSize);
        REQUIRE(b != nullptr);
        bufs.push_back(b);
    }
    REQUIRE(monitor.check() == rais::MemoryPressure::Warning);
    REQUIRE(monitor.under_pressure());

    // Allocate to 10/10 = 100% — Critical (>= 0.90 threshold)
    for (int i = 0; i < 2; ++i) {
        void* b = pool.acquire(kBufSize);
        REQUIRE(b != nullptr);
        bufs.push_back(b);
    }
    REQUIRE(monitor.check() == rais::MemoryPressure::Critical);

    for (auto* b : bufs) pool.release(b);
}

TEST_CASE("No budget means always Normal", "[memory_pressure]") {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);

    rais::MetalBufferPool pool((__bridge void*)device);
    // No budget set (default 0)

    rais::MemoryMonitor monitor(pool);

    void* buf = pool.acquire(16 * 1024 * 1024); // 16 MB
    REQUIRE(buf != nullptr);
    REQUIRE(monitor.check() == rais::MemoryPressure::Normal);

    pool.release(buf);
}

TEST_CASE("Recovery: pressure drops when buffers released", "[memory_pressure]") {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);

    rais::MetalBufferPool pool((__bridge void*)device);
    pool.set_memory_budget(256 * 1024); // 256 KB

    rais::MemoryMonitor monitor(pool, 0.75f, 0.90f);

    // Fill to Critical
    void* buf1 = pool.acquire(64 * 1024);
    void* buf2 = pool.acquire(64 * 1024);
    void* buf3 = pool.acquire(64 * 1024);
    void* buf4 = pool.acquire(64 * 1024);
    REQUIRE(buf4 != nullptr);
    REQUIRE(monitor.check() == rais::MemoryPressure::Critical);

    // Release buffers to go back to Normal
    pool.release(buf3);
    pool.release(buf4);
    // Now at 128 / 256 = 50% — Normal
    REQUIRE(monitor.check() == rais::MemoryPressure::Normal);
    REQUIRE_FALSE(monitor.under_pressure());

    pool.release(buf1);
    pool.release(buf2);
}
