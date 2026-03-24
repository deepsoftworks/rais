#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <rais/profiler.hpp>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace rais {

// ─── TraceBuffer ────────────────────────────────────────────────────────────

TraceBuffer::TraceBuffer(size_t capacity)
    : capacity_(capacity), mask_(capacity - 1) {
    assert((capacity & (capacity - 1)) == 0 && "TraceBuffer capacity must be power of two");
    assert(capacity >= 16 && "TraceBuffer capacity too small");
    slots_ = std::make_unique<TraceEvent[]>(capacity);
}

TraceBuffer::~TraceBuffer() = default;

bool TraceBuffer::produce(const TraceEvent& event) {
    // Multi-producer: CAS on tail to reserve a slot
    size_t tail = tail_.load(std::memory_order_relaxed);
    for (;;) {
        size_t head = head_.load(std::memory_order_acquire);
        if (tail - head >= capacity_) {
            // Buffer full — drop the event
            drop_count_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (tail_.compare_exchange_weak(tail, tail + 1,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
            break;
        }
        // CAS failed — tail was updated, retry with new value
    }

    slots_[tail & mask_] = event;
    return true;
}

bool TraceBuffer::consume(TraceEvent& out) {
    // Single consumer: only the drain thread calls this
    size_t head = head_.load(std::memory_order_relaxed);
    size_t tail = tail_.load(std::memory_order_acquire);
    if (head >= tail) return false;

    out = slots_[head & mask_];
    head_.store(head + 1, std::memory_order_release);
    return true;
}

uint64_t TraceBuffer::drops() const {
    return drop_count_.load(std::memory_order_relaxed);
}

size_t TraceBuffer::size() const {
    size_t tail = tail_.load(std::memory_order_acquire);
    size_t head = head_.load(std::memory_order_acquire);
    return tail >= head ? tail - head : 0;
}

size_t TraceBuffer::capacity() const {
    return capacity_;
}

// ─── Profiler::Impl ─────────────────────────────────────────────────────────

/// Thread ID helper — fast, no syscall on Apple platforms
static uint32_t current_thread_id() {
    uint64_t tid;
    pthread_threadid_np(nullptr, &tid);
    return static_cast<uint32_t>(tid);
}

/// Chrome Tracing phase character
static char phase_char(TracePhase p) {
    return static_cast<char>(p);
}

/// Category name for JSON output
static const char* category_name(TraceCategory cat) {
    switch (cat) {
        case TraceCategory::Scheduler: return "scheduler";
        case TraceCategory::GPU:       return "gpu";
        case TraceCategory::Allocator: return "allocator";
        case TraceCategory::Queue:     return "queue";
        case TraceCategory::User:      return "user";
    }
    return "unknown";
}

struct Profiler::Impl {
    TraceBuffer buffer;

    // Output file — memory-mapped for efficient writes
    int fd = -1;
    char* mapped = nullptr;
    size_t mapped_size = 0;
    size_t write_offset = 0;
    static constexpr size_t kInitialMapSize = 64 * 1024 * 1024; // 64 MB
    std::mutex remap_mutex; // only taken when growing the file

    // Drain thread
    std::thread drain_thread;
    std::atomic<bool> running{false};
    std::atomic<bool> stop_requested{false};

    // GPU counter support
    id<MTLDevice> gpu_device = nil;
    id<MTLCounterSampleBuffer> counter_sample_buffer = nil;
    bool gpu_counters_available = false;

    // Stats
    uint64_t events_written = 0;

    explicit Impl(const char* output_path, void* device, size_t capacity)
        : buffer(capacity) {

        // Open output file
        fd = open(output_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            std::fprintf(stderr, "Profiler: failed to open %s: %s\n",
                         output_path, std::strerror(errno));
            return;
        }

        // Extend file to initial size for mmap
        if (ftruncate(fd, static_cast<off_t>(kInitialMapSize)) < 0) {
            std::fprintf(stderr, "Profiler: ftruncate failed: %s\n",
                         std::strerror(errno));
            close(fd);
            fd = -1;
            return;
        }

        mapped = static_cast<char*>(mmap(nullptr, kInitialMapSize,
                                          PROT_READ | PROT_WRITE,
                                          MAP_SHARED, fd, 0));
        if (mapped == MAP_FAILED) {
            std::fprintf(stderr, "Profiler: mmap failed: %s\n",
                         std::strerror(errno));
            mapped = nullptr;
            close(fd);
            fd = -1;
            return;
        }
        mapped_size = kInitialMapSize;

        // Write JSON array opening
        write_raw("[");

        // Initialize GPU counters if a Metal device is provided
        if (device) {
            init_gpu_counters(device);
        }
    }

    ~Impl() {
        if (mapped && mapped_size > 0) {
            munmap(mapped, mapped_size);
        }
        if (fd >= 0) {
            // Truncate file to actual written size
            ftruncate(fd, static_cast<off_t>(write_offset));
            close(fd);
        }
    }

    void init_gpu_counters(void* device) {
        gpu_device = (__bridge id<MTLDevice>)device;
        if (!gpu_device) return;

        // Look for the timestamp counter set
        for (id<MTLCounterSet> cs in gpu_device.counterSets) {
            if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                // Found timestamp counters — set up a sample buffer
                MTLCounterSampleBufferDescriptor* desc =
                    [[MTLCounterSampleBufferDescriptor alloc] init];
                desc.counterSet = cs;
                desc.storageMode = MTLStorageModeShared;
                desc.sampleCount = 256; // ring of 256 samples

                NSError* error = nil;
                counter_sample_buffer =
                    [gpu_device newCounterSampleBufferWithDescriptor:desc error:&error];
                if (counter_sample_buffer) {
                    gpu_counters_available = true;
                } else {
                    std::fprintf(stderr, "Profiler: GPU counter sample buffer failed: %s\n",
                                 error.localizedDescription.UTF8String);
                }
                break;
            }
        }
    }

    /// Write raw bytes into the memory-mapped file, growing if needed.
    void write_raw(const char* data) {
        size_t len = std::strlen(data);
        ensure_capacity(len);
        if (mapped) {
            std::memcpy(mapped + write_offset, data, len);
            write_offset += len;
        }
    }

    void write_raw(const char* data, size_t len) {
        ensure_capacity(len);
        if (mapped) {
            std::memcpy(mapped + write_offset, data, len);
            write_offset += len;
        }
    }

    void ensure_capacity(size_t additional) {
        if (!mapped) return;
        if (write_offset + additional <= mapped_size) return;

        std::lock_guard lock(remap_mutex);
        // Double the mapping size
        size_t new_size = mapped_size * 2;
        while (write_offset + additional > new_size) {
            new_size *= 2;
        }

        if (ftruncate(fd, static_cast<off_t>(new_size)) < 0) {
            std::fprintf(stderr, "Profiler: ftruncate grow failed\n");
            return;
        }

        munmap(mapped, mapped_size);
        mapped = static_cast<char*>(mmap(nullptr, new_size,
                                          PROT_READ | PROT_WRITE,
                                          MAP_SHARED, fd, 0));
        if (mapped == MAP_FAILED) {
            mapped = nullptr;
            return;
        }
        mapped_size = new_size;
    }

    /// Format and write a single TraceEvent as a Chrome Tracing JSON object.
    void write_event(const TraceEvent& ev) {
        char buf[512];
        int n;

        if (events_written > 0) {
            // Comma separator between JSON objects
            write_raw(",\n", 2);
        }

        // Convert ns to µs for Chrome Tracing (it expects µs)
        double ts_us = static_cast<double>(ev.timestamp_ns) / 1000.0;

        if (ev.gpu_ns != 0) {
            double gpu_us = static_cast<double>(ev.gpu_ns) / 1000.0;
            n = std::snprintf(buf, sizeof(buf),
                R"({"name":"%s","cat":"%s","ph":"%c","ts":%.3f,"pid":1,"tid":%u,)"
                R"("args":{"arg0":%llu,"arg1":%llu,"gpu_ns":%llu,"gpu_us":%.3f}})",
                ev.name, category_name(ev.category), phase_char(ev.phase),
                ts_us, ev.thread_id,
                static_cast<unsigned long long>(ev.arg0),
                static_cast<unsigned long long>(ev.arg1),
                static_cast<unsigned long long>(ev.gpu_ns),
                gpu_us);
        } else {
            n = std::snprintf(buf, sizeof(buf),
                R"({"name":"%s","cat":"%s","ph":"%c","ts":%.3f,"pid":1,"tid":%u,)"
                R"("args":{"arg0":%llu,"arg1":%llu}})",
                ev.name, category_name(ev.category), phase_char(ev.phase),
                ts_us, ev.thread_id,
                static_cast<unsigned long long>(ev.arg0),
                static_cast<unsigned long long>(ev.arg1));
        }

        if (n > 0 && n < static_cast<int>(sizeof(buf))) {
            write_raw(buf, static_cast<size_t>(n));
            ++events_written;
        }
    }

    /// Drain loop: runs on a background thread, consuming events and writing
    /// them to the memory-mapped file. Wakes periodically (1ms) to drain.
    void drain_loop() {
        TraceEvent ev;
        while (!stop_requested.load(std::memory_order_acquire)) {
            // Drain all available events
            size_t drained = 0;
            while (buffer.consume(ev)) {
                write_event(ev);
                ++drained;
            }

            if (drained == 0) {
                // Nothing to drain — sleep briefly to avoid busy-spinning
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        // Final drain after stop requested
        while (buffer.consume(ev)) {
            write_event(ev);
        }
    }

    /// Finalize the JSON file: close the array and sync to disk.
    void finalize() {
        write_raw("\n]");
        if (mapped && mapped_size > 0) {
            msync(mapped, write_offset, MS_SYNC);
        }
    }
};

// ─── Profiler public API ────────────────────────────────────────────────────

Profiler::Profiler(const char* output_path, void* gpu_device, size_t buffer_capacity)
    : impl_(std::make_unique<Impl>(output_path, gpu_device, buffer_capacity)) {}

Profiler::~Profiler() {
    if (impl_->running.load(std::memory_order_acquire)) {
        stop();
    }
}

void Profiler::start() {
    if (impl_->running.exchange(true, std::memory_order_acq_rel)) {
        return; // already running
    }
    impl_->stop_requested.store(false, std::memory_order_release);
    impl_->drain_thread = std::thread([this]() { impl_->drain_loop(); });
}

void Profiler::stop() {
    if (!impl_->running.exchange(false, std::memory_order_acq_rel)) {
        return; // not running
    }
    impl_->stop_requested.store(true, std::memory_order_release);
    if (impl_->drain_thread.joinable()) {
        impl_->drain_thread.join();
    }
    impl_->finalize();
}

void Profiler::trace(const char* name, TraceCategory category, TracePhase phase,
                     uint64_t arg0, uint64_t arg1) {
    TraceEvent ev{};
    ev.timestamp_ns = clock_ns();
    ev.name = name;
    ev.category = category;
    ev.phase = phase;
    ev.arg0 = arg0;
    ev.arg1 = arg1;
    ev.thread_id = current_thread_id();
    ev.gpu_ns = 0;

    impl_->buffer.produce(ev);
}

void Profiler::record_gpu_timestamps(uint64_t gpu_start_ns, uint64_t gpu_end_ns,
                                     uint64_t arg0) {
    // Record a Begin event at GPU start time
    TraceEvent begin{};
    begin.timestamp_ns = gpu_start_ns;
    begin.name = "gpu_command_buffer";
    begin.category = TraceCategory::GPU;
    begin.phase = TracePhase::Begin;
    begin.arg0 = arg0;
    begin.thread_id = current_thread_id();
    begin.gpu_ns = gpu_start_ns;
    impl_->buffer.produce(begin);

    // Record an End event at GPU end time
    TraceEvent end{};
    end.timestamp_ns = gpu_end_ns;
    end.name = "gpu_command_buffer";
    end.category = TraceCategory::GPU;
    end.phase = TracePhase::End;
    end.arg0 = arg0;
    end.thread_id = current_thread_id();
    end.gpu_ns = gpu_end_ns;
    impl_->buffer.produce(end);
}

uint64_t Profiler::drops() const {
    return impl_->buffer.drops();
}

bool Profiler::running() const {
    return impl_->running.load(std::memory_order_acquire);
}

} // namespace rais
