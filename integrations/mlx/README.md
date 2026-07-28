# rais × mlx-lm — concurrency & QoS

The first **real** (non-simulated) rais integration. Every decode step here is an
actual `mlx_lm` forward pass, scheduled by the rais lock-free QoS scheduler — no
`sleep_for` stand-ins.

## The claim

On Apple Silicon there is one GPU and (through Python) one GIL, so decode steps are
physically serial. The thing that makes concurrent LLM serving good or bad is
therefore **scheduling order**: when a latency-sensitive request arrives while a
batch job is running, does it wait behind the batch, or jump ahead?

- `mlx_lm.server` processes requests essentially FIFO → an interactive request
  submitted behind a long generation suffers head-of-line blocking.
- rais schedules each request's next-token step on a **QoS lane**. Interactive
  steps preempt background steps (strict lane priority), while rais's
  anti-starvation promotion keeps background work progressing.

So the product being measured is a scheduler: the headline metrics are **tail
latency** (interactive TTFT/ITL p95/p99 under load) and **fairness** (background
throughput retained under interactive load), not mean throughput.

## How it works

`rais_mlx.py` loads the model once and, per request, drives `mlx_lm`'s
`generate_step` generator one token at a time. Each step is submitted to
`rais.Scheduler` on the request's lane (`INTERACTIVE` vs `BULK`); when a step
completes it re-submits the next step on the same lane (continuations — the Python
bindings expose `submit`, so the DAG is expressed by re-submission). The scheduler
runs a single worker so model access stays serialized; lane priority decides the
interleaving.

## Reproduce

Prereqs: Apple Silicon, `mlx` + `mlx_lm` installed, and the built `rais` Python
module (`cmake -S . -B build -DRAIS_BUILD_PYTHON=ON && cmake --build build`; the
scripts add `build/` to `sys.path` automatically, or set `RAIS_PYTHONPATH`).

```bash
# 1. fetch the model (default: 4-bit 3B, ~1.8 GB, fits in 8 GB)
python3 integrations/mlx/fetch_model.py

# 2. smoke test — real tokens, scheduled by rais
python3 integrations/mlx/rais_mlx.py --prompt "Explain unified memory" --max-tokens 48

# 3. benchmark: rais (QoS lanes) + a no-QoS ablation, plus the mlx_lm.server baseline
python3 integrations/mlx/bench_concurrency.py --server

# 4. plots
python3 integrations/mlx/plot.py
```

Bigger model on a bigger machine:

```bash
RAIS_MLX_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit python3 integrations/mlx/fetch_model.py
python3 integrations/mlx/bench_concurrency.py --model mlx-community/Qwen2.5-7B-Instruct-4bit --server
```

## Results

`bench_concurrency.py` produces `results.tsv`; `plot.py` renders `ttft_tail.png`
and `fairness.png`. Numbers below are from an **M-series Mac, 8 GB, macOS 15,
Llama-3.2-3B-Instruct-4bit**, 12 interactive requests arriving at ~3/s while 2
background streams saturate the model.

The headline metric is **interactive latency under background load** — what a
scheduler exists to protect:

| system | TTFT p95 (ms) ↓ | TTFT p99 (ms) ↓ | ITL p95 (ms) ↓ | ITL p99 (ms) ↓ |
| ------ | ---: | ---: | ---: | ---: |
| **rais** (QoS lanes) | **626** | **629** | **68** | **74** |
| `mlx_lm.server` (batched, default) | 764 | 813 | 115 | 191 |
| rais (no QoS, ablation) | 1143 | 1431 | 197 | 253 |

rais gives the lowest interactive tail latency of the three: ITL p95 **1.7×**
lower than `mlx_lm.server` and p99 **2.6×** lower. Strict lane priority beats the
server's fair-share continuous batching for latency-sensitive requests, and the
no-QoS ablation (same driver, one lane) confirms the scheduler is the cause.

**The honest tradeoff — throughput.** `mlx_lm.server` is *not* a strawman: its
default continuous batching (`--decode-concurrency 32`) runs background streams in
parallel on the GPU, so background keeps ~full throughput (**20 tok/s**) while it
serves interactive requests. rais here runs a single serial worker and spends the
GPU on interactive work first, so background is deprioritized to **~11%** of its
isolated rate — floored by anti-starvation promotion so it never fully starves,
but far below the server's aggregate throughput.

So this is a policy contrast, not a knockout: **rais = strict QoS / lowest
interactive latency; `mlx_lm.server` = higher aggregate throughput via batching.**
For fits-in-RAM serving where you mostly want throughput, the server's batching is
excellent and rais does not beat it there. rais's edge is explicit priority when
interactive latency must be protected — and, beyond this benchmark, the
memory-oversubscription regime where the model doesn't fit at all.

## Oversubscription — bounded-memory streaming (`residency.py`)

The other half of the story: running a model at **bounded peak memory** by owning
weight *residency*. MLX materializes a model's whole weight set into resident
memory on first use and keeps it there, so a model larger than the memory budget
can't run under stock mlx-lm. `ResidencyManager` pins as many transformer blocks
as fit in a budget and **streams the rest** — evicting a streamed block's weights
after it computes and re-materializing them from the mmap'd safetensors before the
block is needed again.

Three mechanics make this work (see `../../oversubscription.md`):
re-classing block instances so model introspection keeps working; forcing a
per-block eval so MLX's lazy graph doesn't keep every block's weights alive until
end-of-token; and reloading from **fresh** lazy arrays each time (a persistent
weight dict would pin everything resident once materialized).

**Graceful degradation** (Llama-3.2-3B-4bit; `bench_oversub.py`, steady-state
decode peak, bit-exact output at every point):

| resident blocks | peak active memory | throughput |
| ---: | ---: | ---: |
| 28 / 28 (full, stock-equivalent) | 4.20 GB | 15.1 tok/s |
| 14 / 28 | 2.39 GB | 7.7 tok/s |
| 8 / 28 | 1.37 GB | 4.5 tok/s |
| 4 / 28 | 0.70 GB | 5.4 tok/s |
| 0 / 28 (stream all) | 0.25 GB | 5.0 tok/s |

Peak memory is controllable across a **16.7×** span by choosing how many blocks
stay resident, with identical output throughout. The win is **feasibility, not
speed**: streaming trades throughput (and, at heavy oversubscription, SSD
bandwidth) for the ability to run a model that otherwise wouldn't fit.

```bash
python3 integrations/mlx/residency.py --pinned 4          # bounded-memory smoke test
python3 integrations/mlx/bench_oversub.py                 # the curve above + plot
python3 integrations/mlx/bench_oversub.py --model <big> --oom-demo  # stock OOM vs runs
```

This first cut is synchronous. The next step is overlapping block *i+1*'s reload
with block *i*'s compute on rais's IO lane (the `LayerStreamer` shape) — where the
scheduler earns its keep over the OS demand-pager.

## Scope

The concurrency section proves **concurrent scheduling + QoS separation** on
fits-in-RAM models; this section proves **bounded-memory oversubscription**. A
real over-RAM headline (a model bigger than usable RAM that stock mlx-lm can't
load) needs a machine with the model on disk — run `--oom-demo` with a 7B-8bit /
14B-4bit to produce it.
