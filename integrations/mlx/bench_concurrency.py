"""Honest concurrency + QoS benchmark: rais vs mlx_lm.server.

The product being measured is a *scheduler*, so the headline numbers are tail
latency and fairness under contention, not mean throughput:

  * interactive TTFT / inter-token latency (p50/p95/p99) while a background batch
    saturates the model, versus the same interactive request in isolation
    (== interactive protection / slowdown factor), and
  * background tokens/sec under interactive load versus in isolation
    (== fairness / no-starvation).

Systems compared:
  rais        RaisEngine with QoS lanes (interactive -> INTERACTIVE, background -> BULK)
  rais-noqos  same engine, everything on one lane (ablation: isolates the QoS win;
              this is NOT the old std::queue "naive FIFO" strawman -- it is the real
              MLX driver with priority disabled)
  server      the stock `mlx_lm.server` OpenAI endpoint, driven by concurrent
              streaming HTTP clients (the real-engine baseline)

Usage:
    python3 integrations/mlx/bench_concurrency.py                 # rais + ablation
    python3 integrations/mlx/bench_concurrency.py --server        # also run mlx_lm.server
    python3 integrations/mlx/bench_concurrency.py --model <repo>  # bigger model

Writes results.tsv next to this script.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rais_mlx import DEFAULT_MODEL, RaisEngine, rais  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

INTERACTIVE_PROMPTS = [
    "What is unified memory on Apple Silicon?",
    "Give one tip for reducing latency in a chat app.",
    "Name a benefit of KV caching.",
    "Summarize what a scheduler does in one line.",
    "What does p99 latency mean?",
]
BACKGROUND_PROMPT = (
    "Write a long, detailed technical essay about memory hierarchies in modern "
    "computers, covering caches, DRAM, and storage."
)


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(name: str, xs_ms: List[float]) -> Dict[str, float]:
    if not xs_ms:
        return {"metric": name, "n": 0}
    return {
        "metric": name,
        "n": len(xs_ms),
        "mean": statistics.fmean(xs_ms),
        "p50": pct(xs_ms, 50),
        "p95": pct(xs_ms, 95),
        "p99": pct(xs_ms, 99),
        "min": min(xs_ms),
        "max": max(xs_ms),
    }


@dataclass
class RunResult:
    system: str
    scenario: str
    ttft_ms: List[float] = field(default_factory=list)
    itl_ms: List[float] = field(default_factory=list)
    bg_tokens: int = 0
    bg_seconds: float = 0.0

    @property
    def bg_tps(self) -> float:
        return self.bg_tokens / self.bg_seconds if self.bg_seconds > 0 else 0.0


# --------------------------------------------------------------------------- #
# rais / rais-noqos driver
# --------------------------------------------------------------------------- #
class BackgroundLoad:
    """Keeps `n_streams` background generations saturating a lane for a window."""

    def __init__(self, engine: RaisEngine, lane, n_streams: int, max_tokens: int):
        self.engine = engine
        self.lane = lane
        self.n_streams = n_streams
        self.max_tokens = max_tokens
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self.tokens = 0
        self._reqs = []

    def _on_token(self, _seg, _tok):
        with self._lock:
            self.tokens += 1

    def start(self):
        for _ in range(self.n_streams):
            self._reqs.append(self._launch())

    def _launch(self):
        return self.engine.submit(
            BACKGROUND_PROMPT,
            self.lane,
            max_tokens=self.max_tokens,
            on_token=self._on_token,
            stop_event=self.stop_event,
        )

    def maintain(self):
        # Relaunch any finished stream so the lane stays saturated.
        if self.stop_event.is_set():
            return
        for i, r in enumerate(self._reqs):
            if r.done:
                self._reqs[i] = self._launch()

    def stop_and_join(self, timeout: float = 30.0):
        self.stop_event.set()
        deadline = time.perf_counter() + timeout
        for r in self._reqs:
            r.wait(max(0.0, deadline - time.perf_counter()))

    def token_count(self) -> int:
        with self._lock:
            return self.tokens


def run_rais(
    engine: RaisEngine,
    system: str,
    interactive_lane,
    background_lane,
    n_interactive: int,
    interactive_interval: float,
    interactive_max_tokens: int,
    bg_streams: int,
    bg_max_tokens: int,
    with_background: bool,
) -> RunResult:
    scenario = "under_load" if with_background else "isolated"
    res = RunResult(system=system, scenario=scenario)

    bg: Optional[BackgroundLoad] = None
    if with_background:
        bg = BackgroundLoad(engine, background_lane, bg_streams, bg_max_tokens)
        bg.start()
        # Let the background load warm up and reach steady state.
        time.sleep(1.0)

    window_start = time.perf_counter()
    for i in range(n_interactive):
        prompt = INTERACTIVE_PROMPTS[i % len(INTERACTIVE_PROMPTS)]
        req = engine.submit(
            prompt, interactive_lane, max_tokens=interactive_max_tokens
        )
        req.wait()
        if req.ttft is not None:
            res.ttft_ms.append(req.ttft * 1e3)
        res.itl_ms.extend(v * 1e3 for v in req.itl)
        if bg is not None:
            bg.maintain()
        time.sleep(interactive_interval)
    window_end = time.perf_counter()

    if bg is not None:
        res.bg_seconds = window_end - window_start
        res.bg_tokens = bg.token_count()
        bg.stop_and_join()
    return res


def run_rais_bg_isolated(
    engine: RaisEngine, system: str, background_lane, bg_streams, bg_max_tokens,
    seconds: float,
) -> RunResult:
    """Background-only run to establish the isolated tok/s baseline for fairness."""
    res = RunResult(system=system, scenario="bg_isolated")
    bg = BackgroundLoad(engine, background_lane, bg_streams, bg_max_tokens)
    bg.start()
    time.sleep(1.0)  # warmup, not counted
    t0 = time.perf_counter()
    start_tokens = bg.token_count()
    while time.perf_counter() - t0 < seconds:
        bg.maintain()
        time.sleep(0.1)
    res.bg_seconds = time.perf_counter() - t0
    res.bg_tokens = bg.token_count() - start_tokens
    bg.stop_and_join()
    return res


# --------------------------------------------------------------------------- #
# mlx_lm.server baseline (concurrent streaming HTTP clients)
# --------------------------------------------------------------------------- #
class ServerClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def stream_chat(self, prompt: str, max_tokens: int):
        """Yield (event_time, token_text) for a streaming completion."""
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                text = delta.get("content", "")
                if text:
                    yield time.perf_counter(), text


def wait_for_server(base_url: str, timeout: float = 120.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.5)
    return False


def run_server(
    model: str,
    port: int,
    n_interactive: int,
    interactive_interval: float,
    interactive_max_tokens: int,
    bg_streams: int,
    bg_max_tokens: int,
    with_background: bool,
) -> RunResult:
    scenario = "under_load" if with_background else "isolated"
    res = RunResult(system="server", scenario=scenario)
    base_url = f"http://127.0.0.1:{port}"
    client = ServerClient(base_url, model)

    proc = subprocess.Popen(
        [sys.executable, "-m", "mlx_lm", "server", "--model", model,
         "--port", str(port), "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_for_server(base_url):
            raise RuntimeError("mlx_lm.server did not become ready")

        stop = threading.Event()
        bg_tokens = [0]
        bg_lock = threading.Lock()

        def bg_worker():
            while not stop.is_set():
                try:
                    for _t, _txt in client.stream_chat(BACKGROUND_PROMPT, bg_max_tokens):
                        with bg_lock:
                            bg_tokens[0] += 1
                        if stop.is_set():
                            break
                except (urllib.error.URLError, OSError):
                    time.sleep(0.2)

        bg_threads = []
        window_start = time.perf_counter()
        if with_background:
            for _ in range(bg_streams):
                th = threading.Thread(target=bg_worker, daemon=True)
                th.start()
                bg_threads.append(th)
            time.sleep(1.0)

        for i in range(n_interactive):
            prompt = INTERACTIVE_PROMPTS[i % len(INTERACTIVE_PROMPTS)]
            t_submit = time.perf_counter()
            first = None
            last = None
            for t_ev, _txt in client.stream_chat(prompt, interactive_max_tokens):
                if first is None:
                    first = t_ev
                    res.ttft_ms.append((first - t_submit) * 1e3)
                else:
                    res.itl_ms.append((t_ev - last) * 1e3)
                last = t_ev
            time.sleep(interactive_interval)
        window_end = time.perf_counter()

        if with_background:
            stop.set()
            for th in bg_threads:
                th.join(timeout=10)
            res.bg_seconds = window_end - window_start
            with bg_lock:
                res.bg_tokens = bg_tokens[0]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return res


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_results(path: str, results: List[RunResult]) -> None:
    rows = [("system", "scenario", "metric", "stat", "value")]
    for r in results:
        for name, xs in (("ttft_ms", r.ttft_ms), ("itl_ms", r.itl_ms)):
            s = summarize(name, xs)
            for stat, val in s.items():
                if stat == "metric":
                    continue
                rows.append((r.system, r.scenario, name, stat, f"{val:.3f}"))
        if r.bg_seconds > 0:
            rows.append((r.system, r.scenario, "bg_tps", "value", f"{r.bg_tps:.3f}"))
    with open(path, "w") as f:
        for row in rows:
            f.write("\t".join(str(c) for c in row) + "\n")


def print_report(results: List[RunResult]) -> None:
    by = {(r.system, r.scenario): r for r in results}

    def line(label, val):
        print(f"  {label:<34}{val}")

    print("\n" + "=" * 66)
    print("rais x mlx-lm concurrency benchmark")
    print("=" * 66)
    for r in results:
        s = summarize("ttft", r.ttft_ms)
        print(f"\n[{r.system} / {r.scenario}]")
        if r.ttft_ms:
            line("interactive TTFT p50/p95/p99 (ms)",
                 f"{s['p50']:.1f} / {s['p95']:.1f} / {s['p99']:.1f}")
            si = summarize("itl", r.itl_ms)
            line("interactive ITL p50/p95/p99 (ms)",
                 f"{si['p50']:.1f} / {si['p95']:.1f} / {si['p99']:.1f}")
        if r.bg_seconds > 0:
            line("background throughput (tok/s)", f"{r.bg_tps:.1f}")

    print("\n" + "-" * 66)
    print("Derived: interactive protection & fairness")
    print("-" * 66)
    for system in ("rais", "rais-noqos", "server"):
        load = by.get((system, "under_load"))
        iso = by.get((system, "isolated"))
        bgiso = by.get((system, "bg_isolated")) or by.get(("rais", "bg_isolated"))
        if not load:
            continue
        if iso and iso.ttft_ms and load.ttft_ms:
            prot = pct(load.ttft_ms, 95) / max(pct(iso.ttft_ms, 95), 1e-6)
            line(f"[{system}] TTFT p95 slowdown vs isolated", f"{prot:.2f}x")
        if bgiso and bgiso.bg_tps > 0 and load.bg_tps > 0:
            fair = load.bg_tps / bgiso.bg_tps
            line(f"[{system}] bg throughput retained under load", f"{fair*100:.0f}%")
    print()


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--interactive", type=int, default=20)
    ap.add_argument("--interactive-interval", type=float, default=0.3)
    ap.add_argument("--interactive-max-tokens", type=int, default=64)
    ap.add_argument("--bg-streams", type=int, default=2)
    ap.add_argument("--bg-max-tokens", type=int, default=100000)
    ap.add_argument("--server", action="store_true", help="also run mlx_lm.server")
    ap.add_argument("--server-port", type=int, default=8080)
    ap.add_argument("--out", default=os.path.join(HERE, "results.tsv"))
    args = ap.parse_args()

    common = dict(
        n_interactive=args.interactive,
        interactive_interval=args.interactive_interval,
        interactive_max_tokens=args.interactive_max_tokens,
        bg_streams=args.bg_streams,
        bg_max_tokens=args.bg_max_tokens,
    )
    results: List[RunResult] = []

    print(f"Loading {args.model} for the rais driver ...", file=sys.stderr, flush=True)
    engine = RaisEngine(args.model, num_workers=1)

    # rais with QoS lanes
    results.append(run_rais(engine, "rais", rais.Lane.INTERACTIVE, rais.Lane.BULK,
                            with_background=False, **common))
    results.append(run_rais(engine, "rais", rais.Lane.INTERACTIVE, rais.Lane.BULK,
                            with_background=True, **common))
    results.append(run_rais_bg_isolated(engine, "rais", rais.Lane.BULK,
                                        args.bg_streams, args.bg_max_tokens,
                                        seconds=args.interactive * args.interactive_interval + 3))

    # ablation: no QoS (interactive and background share one lane)
    results.append(run_rais(engine, "rais-noqos", rais.Lane.INTERACTIVE,
                            rais.Lane.INTERACTIVE, with_background=True, **common))

    engine.shutdown()

    if args.server:
        # Free the driver's model copy before the server loads its own (8 GB budget).
        import gc

        del engine
        gc.collect()
        mx.clear_cache()

        print("Running mlx_lm.server baseline ...", file=sys.stderr, flush=True)
        results.append(run_server(args.model, args.server_port,
                                  with_background=False, **common))
        results.append(run_server(args.model, args.server_port,
                                  with_background=True, **common))

    write_results(args.out, results)
    print_report(results)
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
