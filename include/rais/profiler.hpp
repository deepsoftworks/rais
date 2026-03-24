#pragma once

#include <rais/clock.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace rais {

/// Categories for trace events, matching the runtime's subsystems.
enum class TraceCategory : uint8_t {
    Scheduler   = 0, // worker loop, steal, submit, promote
    GPU         = 1, // command buffer encode, commit, complete
    Allocator   = 2, // slab alloc/free, arena reset, buffer pool
    Queue       = 3, // push/pop contention
    User        = 4, // application-level spans
};

/// Phase codes compatible with Chrome Tracing / Perfetto JSON format.
enum class TracePhase : uint8_t {
    Begin    = 'B',
    End      = 'E',
    Instant  = 'I',
    Counter  = 'C',
};

/// A single trace event, 64 bytes for cache-line-friendly ring buffer storage.
/// All fields are plain data — no heap allocations on the hot path.
struct TraceEvent {
    uint64_t       timestamp_ns;       //  8 — monotonic clock
    uint64_t       arg0;               //  8 — event-specific (e.g. task ptr, byte count)
    uint64_t       arg1;               //  8 — event-specific (e.g. lane, worker_id)
    const char*    name;               //  8 — static string literal, never freed
    uint32_t       thread_id;          //  4
    TraceCategory  category;           //  1
    TracePhase     phase;              //  1
    uint8_t        padding_[2] = {};   //  2 — pad to 40 bytes
    uint64_t       gpu_ns;             //  8 — GPU timestamp (0 if not applicable)
    uint8_t        reserved_[16] = {}; // 16 — future use, pad to 64 bytes total
};

static_assert(sizeof(TraceEvent) == 64, "TraceEvent must be 64 bytes");

/// Lock-free single-producer-single-consumer ring buffer for trace events.
///
/// Workers write events via produce(). The drain thread reads via consume().
/// When the buffer is full, new events are silently dropped (the profiler
/// must never block the scheduler hot path). The drop counter tracks losses.
class TraceBuffer {
public:
    /// Capacity must be a power of two.
    explicit TraceBuffer(size_t capacity = 65536);
    ~TraceBuffer();

    TraceBuffer(const TraceBuffer&) = delete;
    TraceBuffer& operator=(const TraceBuffer&) = delete;

    /// Write an event into the ring. Returns false (and increments the
    /// drop counter) if the buffer is full. Lock-free, wait-free for
    /// single-producer use or CAS-based for multi-producer.
    bool produce(const TraceEvent& event);

    /// Read the next event. Returns false if the buffer is empty.
    /// Only safe to call from a single consumer (the drain thread).
    bool consume(TraceEvent& out);

    /// Number of events dropped because the buffer was full.
    uint64_t drops() const;

    /// Number of events currently in the buffer (approximate).
    size_t size() const;

    /// Capacity of the ring buffer.
    size_t capacity() const;

private:
    std::unique_ptr<TraceEvent[]> slots_;
    size_t capacity_;
    size_t mask_;
    alignas(64) std::atomic<size_t> head_{0}; // consumer reads from here
    alignas(64) std::atomic<size_t> tail_{0}; // producer writes here
    alignas(64) std::atomic<uint64_t> drop_count_{0};
};

/// Runtime profiler that collects trace events from scheduler workers,
/// the GPU executor, and allocators, and drains them to a memory-mapped
/// file for offline analysis.
///
/// Objective-C types (MTLCounterSet, MTLCounterSampleBuffer) are hidden
/// behind PIMPL so this header stays pure C++.
///
/// Usage:
///   Profiler profiler("trace.json");
///   profiler.start();
///   // ... runtime runs, instrumenting with profiler.trace() ...
///   profiler.stop();  // flushes remaining events and closes the file
///
/// The output file is Chrome Tracing JSON (loadable in chrome://tracing
/// or Perfetto UI).
class Profiler {
public:
    /// Open a profiler that will drain events to the given file path.
    /// If gpu_device is non-null (id<MTLDevice> as void*), GPU timestamp
    /// counters are sampled via MTLCounterSampleBuffer.
    explicit Profiler(const char* output_path, void* gpu_device = nullptr,
                      size_t buffer_capacity = 65536);
    ~Profiler();

    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    /// Start the background drain thread. Events produced before start()
    /// are buffered normally.
    void start();

    /// Stop the drain thread and flush remaining events to disk.
    /// Blocks until the drain thread exits and the file is closed.
    void stop();

    /// Record a trace event. This is the hot-path API — must be lock-free
    /// and allocation-free. Intended to be called from worker threads,
    /// GPU completion handlers, and allocator paths.
    void trace(const char* name, TraceCategory category, TracePhase phase,
               uint64_t arg0 = 0, uint64_t arg1 = 0);

    /// RAII scope guard for Begin/End pairs.
    class Span {
    public:
        Span(Profiler& profiler, const char* name, TraceCategory category,
             uint64_t arg0 = 0, uint64_t arg1 = 0)
            : profiler_(profiler), name_(name), category_(category),
              arg0_(arg0), arg1_(arg1) {
            profiler_.trace(name_, category_, TracePhase::Begin, arg0_, arg1_);
        }
        ~Span() {
            profiler_.trace(name_, category_, TracePhase::End, arg0_, arg1_);
        }

        Span(const Span&) = delete;
        Span& operator=(const Span&) = delete;
    private:
        Profiler& profiler_;
        const char* name_;
        TraceCategory category_;
        uint64_t arg0_;
        uint64_t arg1_;
    };

    /// Sample GPU counters and attach timestamps to the most recent GPU
    /// trace events. Called from the MetalExecutor completion handler.
    /// gpu_start_ns / gpu_end_ns come from MTLCommandBuffer's GPU timestamps.
    void record_gpu_timestamps(uint64_t gpu_start_ns, uint64_t gpu_end_ns,
                               uint64_t arg0 = 0);

    /// Number of events dropped due to buffer overflow.
    uint64_t drops() const;

    /// Whether the profiler is currently running (drain thread active).
    bool running() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Global profiler instance. Null when profiling is disabled.
/// Set by the application before starting the scheduler.
inline Profiler* g_profiler = nullptr;

/// Convenience macros for zero-overhead when profiling is disabled.
/// The g_profiler check compiles to a single branch on a global pointer.
#define RAIS_TRACE(name, cat, phase, ...) \
    do { if (::rais::g_profiler) ::rais::g_profiler->trace(name, cat, phase, ##__VA_ARGS__); } while (0)

#define RAIS_SPAN(var, name, cat, ...) \
    ::rais::Profiler::Span var##_guard_(*::rais::g_profiler, name, cat, ##__VA_ARGS__); \
    (void)var##_guard_

#define RAIS_SPAN_IF(var, name, cat, ...) \
    std::unique_ptr<::rais::Profiler::Span> var##_guard_; \
    do { if (::rais::g_profiler) var##_guard_ = std::make_unique<::rais::Profiler::Span>( \
        *::rais::g_profiler, name, cat, ##__VA_ARGS__); } while (0)

} // namespace rais
