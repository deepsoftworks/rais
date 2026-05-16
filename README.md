<p align="center">
  <img src="rais.png" alt="Rais" />
</p>

<p align="center">
  <a href="https://github.com/deepsoftworks/rais/actions/workflows/ci.yml"><img src="https://github.com/deepsoftworks/rais/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://github.com/deepsoftworks/rais/releases"><img src="https://img.shields.io/github/v/release/deepsoftworks/rais?label=latest%20version" alt="Latest Version"></a>
  <img alt="macOS" src="https://img.shields.io/badge/-macOS-black?style=flat-square&logo=apple&logoColor=white" />
</p>

---

A C++ task scheduler for AI inference on Apple Silicon. Prioritizes real-time LLM requests over batch work, overlaps SSD reads with GPU compute, and hot-swaps models without downtime.

## Results

**Concurrent request scheduling** (Llama-3.2-1B-Instruct-4bit, 6 clients):

| Metric | Naive FIFO | Rais | Speedup |
|---|---|---|---|
| Interactive TTFT | 4,829 ms | 1,438 ms | **3.4x** |
| Interactive E2E | 5,653 ms | 2,254 ms | **2.5x** |

**Layer-streaming throughput** (IO/compute overlapped):

| Model | Naive | Rais | Speedup |
|---|---|---|---|
| SmolLM2-135M (257 MB) | 157 tok/s | 188 tok/s | **1.20x** |
| TinyLlama-1.1B (2.1 GB) | 15.5 tok/s | 17.8 tok/s | **1.15x** |

## Quick start

```bash
git clone https://github.com/deepsoftworks/rais.git && cd rais
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

Works with MLX/mlx-lm, llama.cpp, and PyTorch. See `examples/` for integration patterns:

- `examples/minimal_submit.cpp` -- basic scheduler usage
- `examples/llama_cpp_integration.cpp` -- llama.cpp integration
- `examples/rais_server.cpp` -- server mode

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
