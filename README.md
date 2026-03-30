# RAIS

**RAIS** (Runtime for AI Scheduling; رئيس) is a C++20 scheduling runtime for
low-latency AI inference on Apple Silicon. It sits between your inference engine
and the hardware, deciding what runs, when, and on which resource — so that
user-facing requests stay fast even when the system is under load.

## The problem

Running LLMs in production on Apple Silicon surfaces three bottlenecks that
inference engines don't solve on their own:

1. **Contention between requests.** When multiple prompts arrive at once, a
   naive FIFO queue makes real-time users wait behind long batch jobs.
   Benchmarks on Llama-3.2-1B with 6 concurrent clients show users waiting
   ~4.8 seconds for their first token when background work is ahead of them.

2. **SSD-to-GPU stalls.** Loading model weights layer-by-layer from disk is
   sequential by default — the GPU sits idle while each layer reads in. On
   real LLM weights (SmolLM2-135M, TinyLlama-1.1B), RAIS's IO pipeline
   delivers **1.15–1.20x higher throughput** by overlapping disk reads with
   compute.

3. **Model switching downtime.** Swapping between models (e.g. changing
   assistant persona or loading a fine-tune) normally requires draining all
   in-flight requests before unloading. With RAIS, the new model loads in the
   background on a dedicated task while the old model keeps serving.

## What RAIS does

RAIS provides a task scheduler with five priority lanes that map directly to
the different classes of work in an inference server:

| Lane | Target | Use |
|---|---|---|
| `Interactive` | < 5ms submit-to-start | Real-time user requests |
| `Background` | Best-effort | Model hot-swap, logging, embeddings |
| `Bulk` | Deferred | Batch jobs, eval runs |
| `GPU` | Metal async | Compute kernel dispatch |
| `IO` | Dedicated threads | SSD weight reads, never blocks CPU |

When a real-time request arrives while batch jobs are queued, RAIS jumps it
to the front. In the 6-client benchmark above, this reduces interactive TTFT
from **4,829ms to 1,438ms — a 3.4x improvement** with no change to total
throughput.

## How it works

### Priority scheduling
The scheduler maintains a lock-free MPMC ring for normal tasks and a separate
min-heap for deadline tasks (served earliest-deadline-first). Each worker
checks its local Chase-Lev deque, then the deadline heap, then the global
queue, then steals from a random peer. Interactive tasks beat Background tasks
beat Bulk tasks — with starvation promotion (100ms / 500ms) ensuring low-
priority work eventually completes.

### IO pipeline
`Lane::IO` tasks run on a dedicated thread pool (default: 2 threads) with
their own MPMC queue, completely isolated from CPU compute workers. Reads use
`pread()` for thread safety and `F_NOCACHE` to bypass the kernel page cache —
layer weights are used once per forward pass and shouldn't evict hot data.

### Layer streaming
`LayerStreamer` maintains a triple-buffered ring of GPU buffers. While the
GPU computes layer N, layer N+1 is already in a Metal buffer, and layer N+2
is being read from SSD. The IO lane drives the reads; the GPU lane drives
compute. Neither stalls waiting for the other.

### Model hot-swap
`ModelManager` submits a new model load as a `Background` task. The old model
keeps serving `Interactive` requests throughout. On success, the old model
unloads atomically and the ready callback fires. Concurrent swap requests
cancel the in-flight load and take over cleanly.

### Memory management
A lock-free slab allocator handles task objects at ~83ns per alloc
(~124M ops/sec), eliminating `make_shared` overhead on the hot path.
`MetalBufferPool` buckets GPU buffers by size class (4KB–256MB) and reuses
them across forward passes, eliminating the per-layer `newBufferWithLength:`
cost. `MemoryMonitor` tracks pool utilization against a configurable budget
with Warning/Critical thresholds for backpressure signaling.

## Measured results

All measurements on Apple Silicon (M-series), release build.

**Concurrent request scheduling** (Llama-3.2-1B-Instruct-4bit, 6 clients,
3 interactive / 3 background, background batch pre-queued):

| Metric | Naive FIFO | RAIS Priority | Speedup |
|---|---|---|---|
| Interactive TTFT | 4,829 ms | 1,438 ms | **3.4x** |
| Interactive E2E | 5,653 ms | 2,254 ms | **2.5x** |

**Layer-streaming throughput** (real weight files, IO/compute overlapped):

| Model | Naive | RAIS | Speedup |
|---|---|---|---|
| SmolLM2-135M (257 MB, 31 layers) | 157 tok/s | 188 tok/s | **1.20x** |
| TinyLlama-1.1B (2.1 GB, 23 layers) | 15.5 tok/s | 17.8 tok/s | **1.15x** |

**Subsystem microbenchmarks:**

| Component | Result |
|---|---|
| MPMC queue (4P4C, cache-line padded) | 5.87M ops/sec |
| Work-stealing deque steal (4 thieves, p50) | 125 ns |
| Slab allocator alloc (single thread, p50) | 83 ns |
| Scheduler Interactive dispatch (p50) | 228 µs |

## Quick start

Clone, build, and run the priority scheduling example in under a minute:

```bash
git clone https://github.com/deepsoftworks/rais.git && cd rais
brew install catch2
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target priority_example
./build/priority_example
```

The example simulates 6 concurrent requests (3 batch + 3 interactive) hitting
a single-threaded decoder. RAIS jumps the interactive requests to the front of
the queue so they start before the batch jobs — no code changes to your model
needed.

## Integrating with mlx-lm

RAIS sits between your inference loop and the hardware. It does not replace
`mlx-lm` or any other inference engine — it decides *which* request runs
next and when to prefetch the next layer from disk.

**Priority scheduling** — wrap your `mlx_lm.generate()` calls in RAIS tasks:

```cpp
#include <rais/scheduler.hpp>

rais::Scheduler sched({.num_workers = 4});

// User-facing request — low latency
auto h = sched.submit([&]() {
    // call your generate / decode function here
    run_inference(prompt, model, tokenizer);
}, rais::Lane::Interactive);

// Background batch job — yields to interactive
sched.submit([&]() {
    run_batch_eval(dataset, model, tokenizer);
}, rais::Lane::Bulk);

h.wait(); // interactive request finishes first
```

**Layer streaming** — overlap SSD reads with GPU compute:

```cpp
#include <rais/layer_streamer.hpp>
#include <rais/metal_allocator.hpp>

rais::MetalBufferPool pool(device);
rais::LayerStreamer streamer(sched, pool, {
    .layer_size_bytes = layer_bytes,
    .num_buffer_slots = 3,       // triple-buffer
    .model_dir = "models/llama",
    .num_layers = 32,
});

streamer.start_prefetch(0);      // fill the pipeline
for (size_t i = 0; i < 32; ++i) {
    auto h = streamer.request_layer(i, [&](void* buf) {
        // buf is a Metal shared buffer with layer weights
        run_layer_forward(i, buf);
    });
    h.wait();
    streamer.release_layer(i);   // recycle the buffer slot
}
```

**Model hot-swap** — switch models without downtime:

```cpp
#include <rais/model_manager.hpp>

rais::ModelManager mgr(sched);
mgr.swap("models/llama-v2",
    [](const std::string& path) { /* load_fn */ },
    [](const std::string& path) { /* unload_fn */ },
    [](bool ok) { printf("swap %s\n", ok ? "done" : "failed"); });

// Old model keeps serving interactive requests throughout.
// When the swap completes, old model is unloaded atomically.
```

## Building

Requires macOS on Apple Silicon (M1+), CMake 3.20+, Xcode command line tools,
and [Catch2 v3](https://github.com/catchorg/Catch2).

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

**LLM layer-streaming benchmark** (requires model files from `fetch_model.py`):
```bash
python3 experiments/fetch_model.py HuggingFaceTB/SmolLM2-135M
./build/bench_inference_llm experiments/models/smollm2-135m
```

**Concurrent request scheduling benchmark** (requires MLX):
```bash
pip install mlx mlx-lm
python3 experiments/bench_mlx_concurrent.py --model llama1b --clients 6
```

## Project structure

```
include/rais/
  scheduler.hpp        Scheduler, SchedulerConfig, five priority lanes
  task.hpp             Task, Lane enum, TaskHandle
  queue.hpp            Lock-free MPMC ring buffer (Vyukov-style)
  deque.hpp            Chase-Lev work-stealing deque
  allocator.hpp        SlabAllocator<T,N> and ArenaAllocator
  streaming.hpp        IO-lane file read helpers (pread + F_NOCACHE)
  layer_streamer.hpp   Triple-buffered SSD-to-GPU layer pipeline
  memory_pressure.hpp  MemoryMonitor (Normal / Warning / Critical)
  model_manager.hpp    Zero-downtime background model swap
  metal_executor.hpp   MetalExecutor (PIMPL, pure C++ header)
  metal_allocator.hpp  MetalBufferPool (size-class GPU buffer pool)
  clock.hpp            clock_ns() — mach_absolute_time on Apple Silicon
  profiler.hpp         Chrome trace-event profiler
src/
  scheduler.cpp        Worker loop, deadline heap, IO thread pool
  streaming.cpp        pread loop with EINTR retry and zero-fill on error
  layer_streamer.cpp   Ring slot management and prefetch logic
  memory_pressure.cpp  Threshold checks against pool budget
  model_manager.cpp    Background load with in-flight swap cancellation
  metal_executor.mm    MetalExecutor Objective-C++ implementation
  metal_allocator.mm   MetalBufferPool implementation
  profiler.mm          Chrome trace-event profiler
shaders/
  rais_kernels.metal   rms_norm, silu, attention_scores, elementwise_add
examples/
  priority_scheduling.cpp   Self-contained priority scheduling demo (no deps)
experiments/
  bench_inference_llm.cpp     Layer-streaming IO benchmark on real weights
  bench_mlx_concurrent.py     Priority scheduling benchmark with MLX
  fetch_model.py              Download + split HuggingFace models
tests/                        Catch2 test suites
benchmarks/                   Per-subsystem microbenchmarks
```

## License

MIT
