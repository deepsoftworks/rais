# RAIS

**RAIS** (Runtime for AI Scheduling; رئيس) is a userspace C++20 runtime that schedules
concurrent AI workloads across CPU and GPU on Apple Silicon. It provides lock-free
task scheduling with four priority lanes, deadline-aware EDF ordering, Metal GPU
dispatch, and custom memory allocators — all designed for low-latency inference
serving.

## Features

- **Lock-free work-stealing scheduler** — MPMC global queue + per-worker Chase-Lev deques with randomized stealing
- **Four priority lanes** — Interactive (<5ms submit-to-start), Background, Bulk (deferred when higher-priority work pending), and GPU
- **Deadline-aware scheduling** — tasks with deadlines are served in earliest-deadline-first order, ahead of FIFO tasks, with miss tracking
- **Starvation prevention** — Background tasks promote to Interactive after 100ms; Bulk promotes to Background after 500ms
- **Metal GPU dispatch** — `Lane::GPU` tasks route to `MetalExecutor` with in-flight ring buffer, backpressure, and async completion
- **Slab allocator** — lock-free tagged-pointer object pool for O(1) task allocation (~83ns alloc latency), with heap fallback
- **Arena allocator** — per-worker bump-pointer scratch memory with O(1) bulk reset
- **Metal buffer pool** — size-class bucketed GPU buffer reuse (4KB–256MB), eliminates per-frame `newBufferWithLength:` overhead
- **GPU shader library** — `rms_norm`, `silu`, `attention_scores`, and `elementwise_add` kernels compiled to `.metallib`
- **Chrome-compatible profiler** — nanosecond-precision CPU/GPU trace events, exportable to `chrome://tracing`

## Architecture

```
                    ┌──────────────────┐
                    │ External Submit  │
                    └────────┬─────────┘
                             │
                    ┌────────┴─────────┐
                    │                  │
                    ▼                  ▼
        ┌───────────────────┐  ┌──────────────────┐
        │ Global MPMC Queue │  │  Deadline Heap    │
        │ (lock-free ring)  │  │ (EDF min-heap)   │
        └───┬─────┬─────┬───┘  └──────┬───────────┘
            │     │     │             │
   ┌────────┘     │     └────────┐    │
   ▼              ▼              ▼    │
┌─────────────┐ ┌─────────────┐ ┌────┴────────┐
│  Worker 0   │ │  Worker 1   │ │  Worker N   │
│ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │
│ │Chase-Lev│ │ │ │Chase-Lev│ │ │ │Chase-Lev│ │
│ │  Deque  │ │ │ │  Deque  │ │ │ │  Deque  │ │
│ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │
└──────┼──────┘ └──────┼──────┘ └──────┼──────┘
       │               │               │
       └───── steal ───┴───── steal ────┘

                       │ GPU lane
                       ▼
           ┌───────────────────────┐
           │   Metal Executor      │
           │  (command queue +     │
           │   pipeline cache)     │
           └───────────┬───────────┘
                       │
           ┌───────────┴───────────┐
           │  MetalBufferPool      │
           │  (size-class buckets) │
           └───────────────────────┘
```

### Worker loop priority

Each worker thread checks sources in this order:

1. **Own deque** — local work from task spawning
2. **Deadline heap** — nearest-deadline task (EDF), served before FIFO work
3. **Global FIFO queue** — non-deadline tasks in submission order
4. **Work stealing** — random victim's deque

## Building

Requires macOS on Apple Silicon (M1+), CMake 3.20+, Xcode command line tools, and [Catch2 v3](https://github.com/catchorg/Catch2).

```bash
brew install catch2
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Running benchmarks

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bench_queue
./build/bench_deque
./build/bench_allocator
./build/bench_scheduler
./build/bench_metal
```

## Project structure

```
include/rais/
  scheduler.hpp        Scheduler interface, SchedulerConfig, ShutdownPolicy
  task.hpp             Task struct, Lane enum, TaskHandle
  queue.hpp            Lock-free MPMC ring buffer (Vyukov-style)
  deque.hpp            Chase-Lev work-stealing deque
  allocator.hpp        SlabAllocator<T,N> and ArenaAllocator
  metal_executor.hpp   MetalExecutor (PIMPL, pure C++ header)
  metal_allocator.hpp  MetalBufferPool (size-class GPU buffer pool)
  clock.hpp            clock_ns() — mach_absolute_time on Apple Silicon
  profiler.hpp         Chrome trace-event profiler
src/
  scheduler.cpp        Scheduler implementation, worker loop, deadline heap
  metal_executor.mm    MetalExecutor Objective-C++ implementation
  metal_allocator.mm   MetalBufferPool implementation
  profiler.mm          Profiler implementation
shaders/
  rais_kernels.metal   GPU kernels: rms_norm, silu, attention_scores, elementwise_add
tests/                 Catch2 test suites
benchmarks/            Microbenchmarks for each subsystem
```

## Design decisions

### Lock-free MPMC ring buffer with per-slot sequence numbers
The global submission queue is the single hottest contention point — every submitter and every worker touches it. The Vyukov-style sequence-number design gives wait-free slot state reads and lock-free push/pop via CAS, with no ABA risk.

### Deadline heap alongside FIFO queue
Deadline tasks are rare relative to total throughput, so a mutex-protected min-heap is acceptable. Workers check the heap before the FIFO queue, giving deadline tasks O(log n) insertion and guaranteed priority over non-deadline work.

### Slab allocator for task objects
Every `submit()` allocates a Task. The lock-free slab pool (~83ns alloc, ~124M ops/sec) replaces `make_shared<Task>()` heap allocation, with automatic heap fallback when the slab is exhausted.

### GPU dispatch from CPU workers
Rather than a dedicated GPU thread, CPU workers that pop a `Lane::GPU` task call `MetalExecutor::submit()` (non-blocking encode + commit). The Metal completion callback marks the task done. Backpressure from the 8-slot in-flight ring re-enqueues the task for retry.

### Cache-line padding between indices
The MPMC producer (`tail_`) and consumer (`head_`) indices are on separate cache lines to eliminate false sharing. Benchmarks show ~87% throughput improvement from `alignas(64)` padding.

### MetalBufferPool with size-class bucketing
`[MTLDevice newBufferWithLength:]` is expensive. The pool maintains per-size-class free lists (4KB–256MB) with `MTLStorageModeShared` — on unified memory, Managed mode adds overhead with no benefit.

## License

MIT
