# RAIS

RAIS (Runtime for AI Scheduling; رئيس) is a userspace C++ runtime that schedules
concurrent AI workloads across CPU and GPU on Apple Silicon.

## Architecture

```
                    ┌──────────────────┐
                    │ External Submit  │
                    └────────┬─────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  Global MPMC Queue    │
                 │  (lock-free ring buf) │
                 └─────┬─────┬─────┬────┘
                       │     │     │
              ┌────────┘     │     └────────┐
              ▼              ▼              ▼
       ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
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

## Design Decisions

### Lock-free MPMC ring buffer with per-slot sequence numbers
**Why not a mutex-based queue?** The global submission queue is the single hottest contention point in the scheduler — every external submitter and every worker thread touches it. A mutex would serialize all access. The Dmitry Vyukov–style sequence-number design gives us wait-free reads of slot state and lock-free push/pop via CAS on shared indices, with no ABA risk because each slot's sequence counter is monotonically increasing.

### Power-of-two capacity
Replacing modulo (`pos % capacity`) with bitwise AND (`pos & mask`) eliminates an expensive integer division on every push/pop. The capacity constraint is enforced at construction with an assert.

### Cache-line padding between indices
The producer index (`tail_`) and consumer index (`head_`) are accessed by disjoint sets of threads. Without padding, they share a cache line and cause constant cross-core invalidation traffic. Benchmarks confirm a ~87% throughput improvement from `alignas(64)` padding.

### Separate padded and unpadded queue variants
The unpadded variant (`MPMCQueueUnpadded`) exists solely to measure the false-sharing penalty. It is not intended for production use.

### Lock-free slab allocator with tagged pointers
`SlabAllocator<T, N>` is a fixed-capacity object pool backed by a contiguous array and a lock-free 
free list. The free list head is a 64-bit tagged pointer (16-bit generation counter + 
48-bit address) to prevent ABA. Each slot's `next` pointer is `std::atomic` so that the speculative
read in `allocate()` is data-race-free even when another thread is concurrently calling `free()`. 
The slab is O(1) alloc/free and thread-safe.

### Bump-pointer arena for per-worker scratch
`ArenaAllocator` is a simple bump-pointer allocator for per-worker temporary data. It is not thread-safe by design — each worker owns its own arena. Bulk `reset()` reclaims all memory in O(1) without per-object destructors.

### MetalBufferPool with size-class bucketing
GPU buffer allocation through `[MTLDevice newBufferWithLength:]` is expensive relative to a pool hit. `MetalBufferPool` maintains per-size-class free lists (4KB, 64KB, 1MB, 16MB, 256MB) behind per-bucket mutexes. All buffers are `MTLStorageModeShared` — on Apple Silicon's unified memory, Managed mode adds overhead with no benefit. The pool tracks `live_buffers()` for leak detection.

