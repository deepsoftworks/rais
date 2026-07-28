<p align="center">
  <img src="rais.png" alt="Rais" />
</p>

<p align="center">
  <a href="https://github.com/deepsoftworks/rais/actions/workflows/ci.yml"><img src="https://github.com/deepsoftworks/rais/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://codeberg.org/deepsoftworks/rais/releases"><img src="https://img.shields.io/gitea/v/release/deepsoftworks/rais?gitea_url=https%3A%2F%2Fcodeberg.org&label=latest%20version" alt="Latest Version"></a>
  <img alt="macOS" src="https://img.shields.io/badge/-macOS-black?style=flat-square&logo=apple&logoColor=white" />
</p>

---

A C++ task scheduler for AI inference on Apple Silicon. Prioritizes real-time LLM requests over batch work, overlaps SSD reads with GPU compute, and hot-swaps models without downtime.

## Results

Real inference — every decode step below is an actual `mlx_lm` forward pass
scheduled by rais, not a simulation. Benchmarked against `mlx_lm.server`'s stock
continuous batching (**not** a naive-FIFO strawman) on Llama-3.2-3B-Instruct-4bit,
8 GB Mac: 12 interactive requests arriving at ~3/s while 2 background streams
saturate the model. Full harness and methodology in
[`integrations/mlx/`](integrations/mlx/).

**Interactive latency under background load** — what a scheduler exists to protect:

| System | TTFT p95 ↓ | TTFT p99 ↓ | ITL p95 ↓ | ITL p99 ↓ |
|---|---:|---:|---:|---:|
| **rais** (QoS lanes) | **626 ms** | **629 ms** | **68 ms** | **74 ms** |
| `mlx_lm.server` (batched) | 764 ms | 813 ms | 115 ms | 191 ms |
| rais, no QoS (ablation) | 1143 ms | 1431 ms | 197 ms | 253 ms |

Strict lane priority gives interactive requests the lowest tail latency of the
three — ITL p95 **1.7×** and p99 **2.6×** below `mlx_lm.server`'s fair-share
batching. The honest tradeoff: rais spends the GPU on interactive work first, so
background throughput drops to ~11% of its isolated rate (floored by
anti-starvation, never fully starved), whereas the server's batching keeps
background near full speed and higher *aggregate* throughput. Different policies —
rais protects latency; batching maximizes throughput. See
[`integrations/mlx/README.md`](integrations/mlx/README.md) for the full breakdown.

## Quick start

```bash
git clone https://codeberg.org/deepsoftworks/rais.git && cd rais
./install.sh
cmake --build build --target priority_example
./build/priority_example
```

### Minimal usage

```cpp
rais::Scheduler sched;

sched.submit([&] {
    generate(prompt);
}, rais::Lane::Interactive);
```

### Python bindings

```bash
WITH_PYTHON=1 ./install.sh
PYTHONPATH=build python3 -c "import rais; print(rais.Scheduler)"
```

## Architecture

Five priority lanes:

| Lane | Purpose |
|---|---|
| `Interactive` | Real-time user requests (< 5ms submit-to-start) |
| `Background` | Model hot-swap, logging, embeddings |
| `Bulk` | Batch jobs, eval runs |
| `GPU` | Metal compute dispatch |
| `IO` | Dedicated threads for SSD weight reads |

Key internals: lock-free MPMC ring + Chase-Lev work-stealing deques, earliest-deadline-first scheduling, starvation promotion, triple-buffered layer streaming, slab allocator (~83ns/alloc).

## Integration

**Real, runnable:** [`integrations/mlx/`](integrations/mlx/) drives live `mlx_lm`
inference through the scheduler with per-request QoS lanes — this is the
integration the Results above measure. Build the Python module with
`WITH_PYTHON=1 ./install.sh`, then:

```bash
python3 integrations/mlx/fetch_model.py
python3 integrations/mlx/rais_mlx.py --prompt "Explain unified memory" --max-tokens 48
python3 integrations/mlx/bench_concurrency.py --server   # reproduce the Results table
```

The C++ files under `examples/` are usage sketches, not wired engines:

- `examples/minimal_submit.cpp` -- basic scheduler usage
- `examples/priority_scheduling.cpp` -- QoS lanes
- `examples/llama_cpp_integration.cpp` / `examples/rais_server.cpp` -- illustrative
  handoff shapes with simulated decode (llama.cpp and a PyTorch path are roadmap,
  not yet real integrations)

## Building

Requires macOS on Apple Silicon (M1+), CMake 3.20+, Xcode CLI tools, [Catch2 v3](https://github.com/catchorg/Catch2).

```bash
brew install catch2
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## License

MIT
