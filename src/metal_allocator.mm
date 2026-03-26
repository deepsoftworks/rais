#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <rais/metal_allocator.hpp>

#include <array>
#include <atomic>
#include <cassert>
#include <mutex>
#include <vector>

namespace rais {

// Size classes: 4KB, 64KB, 1MB, 16MB, 256MB
static constexpr size_t kNumSizeClasses = 5;
static constexpr size_t kSizeClasses[kNumSizeClasses] = {
    4ULL * 1024,          //   4 KB
    64ULL * 1024,         //  64 KB
    1ULL * 1024 * 1024,   //   1 MB
    16ULL * 1024 * 1024,  //  16 MB
    256ULL * 1024 * 1024, // 256 MB
};

/// Map a byte size to the smallest size class that fits it.
/// Returns kNumSizeClasses if the request exceeds the largest class.
static size_t size_class_index(size_t bytes) {
    for (size_t i = 0; i < kNumSizeClasses; ++i) {
        if (bytes <= kSizeClasses[i]) return i;
    }
    return kNumSizeClasses; // too large
}

struct MetalBufferPool::Impl {
    id<MTLDevice> device;

    // Per-size-class free list
    struct Bucket {
        std::mutex mutex;
        std::vector<id<MTLBuffer>> buffers;
    };
    std::array<Bucket, kNumSizeClasses> buckets;

    std::atomic<size_t> live_count{0};

    // Total bytes currently allocated (live buffers only). Updated
    // atomically in acquire/release using the buffer's rounded size.
    std::atomic<size_t> allocated_bytes{0};

    // Memory budget in bytes. 0 = unlimited.
    std::atomic<size_t> budget{0};
};

MetalBufferPool::MetalBufferPool(void* device)
    : impl_(std::make_unique<Impl>()) {

    impl_->device = (__bridge id<MTLDevice>)device;
    assert(impl_->device && "MetalBufferPool: nil MTLDevice");
}

MetalBufferPool::~MetalBufferPool() {
    // Release all pooled buffers. In-use buffers are the caller's
    // responsibility.
}

void* MetalBufferPool::acquire(size_t bytes) {
    size_t idx = size_class_index(bytes);
    size_t rounded_size = (idx < kNumSizeClasses) ? kSizeClasses[idx] : bytes;

    // Budget check: reject if this allocation would exceed the limit.
    // Uses a CAS loop so concurrent acquires don't overshoot.
    size_t budget_val = impl_->budget.load(std::memory_order_relaxed);
    if (budget_val > 0) {
        size_t current = impl_->allocated_bytes.load(std::memory_order_relaxed);
        while (true) {
            if (current + rounded_size > budget_val) return nullptr;
            // relaxed: budget enforcement is best-effort; no ordering
            // dependency on other data. acquire/release on allocated_bytes
            // isn't needed because we don't publish shared state through it.
            if (impl_->allocated_bytes.compare_exchange_weak(
                    current, current + rounded_size,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {
                break;
            }
        }
    } else {
        impl_->allocated_bytes.fetch_add(rounded_size, std::memory_order_relaxed);
    }

    if (idx >= kNumSizeClasses) {
        // Request exceeds largest size class — allocate directly
        id<MTLBuffer> buf = [impl_->device
            newBufferWithLength:bytes
            options:MTLResourceStorageModeShared];
        if (!buf) {
            impl_->allocated_bytes.fetch_sub(rounded_size, std::memory_order_relaxed);
            return nullptr;
        }
        impl_->live_count.fetch_add(1, std::memory_order_relaxed);
        return (__bridge void*)buf;
    }

    // Try to pop from the free list
    {
        auto& bucket = impl_->buckets[idx];
        std::lock_guard lock(bucket.mutex);
        if (!bucket.buffers.empty()) {
            id<MTLBuffer> buf = bucket.buffers.back();
            bucket.buffers.pop_back();
            impl_->live_count.fetch_add(1, std::memory_order_relaxed);
            return (__bridge void*)buf;
        }
    }

    // Allocate a new buffer
    id<MTLBuffer> buf = [impl_->device
        newBufferWithLength:rounded_size
        options:MTLResourceStorageModeShared];
    if (!buf) {
        impl_->allocated_bytes.fetch_sub(rounded_size, std::memory_order_relaxed);
        return nullptr;
    }

    // Runtime assert: on Apple Silicon, Managed mode is pointless.
    // We always use Shared. Verify the buffer got Shared mode.
    assert(buf.storageMode == MTLStorageModeShared &&
           "MetalBufferPool: expected MTLStorageModeShared");

    impl_->live_count.fetch_add(1, std::memory_order_relaxed);
    return (__bridge void*)buf;
}

void MetalBufferPool::release(void* buffer) {
    assert(buffer && "MetalBufferPool::release: null buffer");
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;

    size_t len = buf.length;
    size_t idx = size_class_index(len);

    impl_->live_count.fetch_sub(1, std::memory_order_relaxed);
    impl_->allocated_bytes.fetch_sub(len, std::memory_order_relaxed);

    if (idx >= kNumSizeClasses) {
        // Oversized buffer — not pooled, caller is done with it
        return;
    }

    // Return to the appropriate bucket (no deallocation, no zeroing)
    auto& bucket = impl_->buckets[idx];
    std::lock_guard lock(bucket.mutex);
    bucket.buffers.push_back(buf);
}

size_t MetalBufferPool::pool_size() const {
    size_t total = 0;
    for (size_t i = 0; i < kNumSizeClasses; ++i) {
        auto& bucket = impl_->buckets[i];
        std::lock_guard lock(bucket.mutex);
        total += bucket.buffers.size();
    }
    return total;
}

size_t MetalBufferPool::live_buffers() const {
    return impl_->live_count.load(std::memory_order_acquire);
}

size_t MetalBufferPool::total_allocated_bytes() const {
    return impl_->allocated_bytes.load(std::memory_order_relaxed);
}

void MetalBufferPool::set_memory_budget(size_t bytes) {
    impl_->budget.store(bytes, std::memory_order_relaxed);
}

size_t MetalBufferPool::memory_budget() const {
    return impl_->budget.load(std::memory_order_relaxed);
}

} // namespace rais

// C-linkage helper so pure C++ code (layer_streamer.cpp) can get the
// CPU-visible pointer from an MTLBuffer without including Obj-C headers.
extern "C" void* rais_mtl_buffer_contents(void* buffer) {
    if (!buffer) return nullptr;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
    return buf.contents;
}
