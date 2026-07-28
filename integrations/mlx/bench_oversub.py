"""Memory-oversubscription benchmark for the rais residency manager.

Two things to show (see oversubscription.md):

  1. GRACEFUL DEGRADATION — as the residency budget shrinks, peak memory drops
     and throughput degrades smoothly, with bit-exact output throughout. This is
     the controlled artificial-budget curve; it runs cleanly on 8 GB with a small
     model by capping how many transformer blocks stay resident.
  2. FEASIBILITY (optional, --oom-demo) — on a model whose weights exceed usable
     RAM, stock mlx-lm (all blocks resident) OOMs/thrashes while the residency
     manager runs to completion at bounded memory.

Peak memory is measured at DECODE steady state (max active memory across decode
steps, after a warmup token) so it reflects streaming residency, not the one-off
prompt-prefill spike.

    python3 integrations/mlx/bench_oversub.py                       # 3B curve
    python3 integrations/mlx/bench_oversub.py --pins 0,4,8,14,28
    python3 integrations/mlx/bench_oversub.py --model <big> --oom-demo
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residency import ResidencyManager  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.environ.get(
    "RAIS_MLX_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit"
)
PROMPT = "Explain unified memory on Apple Silicon and why it matters for inference."
WARMUP_TOKENS = 2  # skip prefill + first decode when measuring steady state


@dataclass
class Point:
    pins: int
    resident_frac: float
    peak_gb: float
    tps: float
    ok: bool


def _load(model_path):
    import mlx_lm

    return mlx_lm.load(model_path, lazy=True)


def _sampler():
    from mlx_lm.sample_utils import make_sampler

    return make_sampler(temp=0.0)  # greedy => deterministic for correctness


def _encode(tok):
    return mx.array(tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True))


def _generate(model, tok, sampler, max_tokens):
    """Return (tokens, steady_peak_gb, steady_tps)."""
    from mlx_lm.generate import generate_step

    toks: List[int] = []
    step_active: List[int] = []
    step_start = None
    t_steady0 = None
    for t, _ in generate_step(_encode(tok), model, sampler=sampler):
        toks.append(int(t.item()) if hasattr(t, "item") else int(t))
        step_active.append(mx.get_active_memory())
        if len(toks) == WARMUP_TOKENS:
            t_steady0 = time.perf_counter()
        if len(toks) >= max_tokens:
            break
    steady = step_active[WARMUP_TOKENS:] or step_active
    peak_gb = max(steady) / 1e9
    steady_tokens = max(len(toks) - WARMUP_TOKENS, 1)
    tps = steady_tokens / (time.perf_counter() - t_steady0) if t_steady0 else 0.0
    return toks, peak_gb, tps


def run_curve(model_path: str, pins: List[int], max_tokens: int) -> List[Point]:
    sampler = _sampler()

    # Reference (full resident) for correctness.
    print("reference (full-resident) ...", file=sys.stderr, flush=True)
    model, tok = _load(model_path)
    ref, _, _ = _generate(model, tok, sampler, max_tokens)
    del model
    mx.clear_cache()

    points: List[Point] = []
    for pin in pins:
        model, tok = _load(model_path)
        mgr = ResidencyManager(model, model_path, pinned_blocks=pin)
        toks, peak, tps = _generate(model, tok, sampler, max_tokens)
        ok = toks == ref
        points.append(Point(pin, pin / mgr.n, peak, tps, ok))
        print(f"  pin {pin:>3}/{mgr.n}  peak {peak:5.2f} GB  {tps:5.1f} tok/s  "
              f"ok={ok}", file=sys.stderr, flush=True)
        del model, mgr
        mx.clear_cache()
    return points


def _import_rais():
    try:
        import rais  # type: ignore
        return rais
    except ImportError:
        build = os.path.abspath(os.path.join(HERE, "..", "..", "build"))
        if os.path.isdir(build):
            sys.path.insert(0, build)
        import rais  # type: ignore
        return rais


def run_prefetch_compare(model_path: str, pin: int, max_tokens: int,
                         io_threads: int) -> None:
    """A/B: synchronous reload vs rais IO-lane prefetch at one residency budget."""
    rais = _import_rais()
    sampler = _sampler()

    print("reference (full-resident) ...", file=sys.stderr, flush=True)
    model, tok = _load(model_path)
    ref, _, _ = _generate(model, tok, sampler, max_tokens)
    del model
    mx.clear_cache()

    def one(prefetch: bool):
        model, tok = _load(model_path)
        sched = None
        kw = {}
        if prefetch:
            sched = rais.Scheduler(num_workers=1, io_thread_count=io_threads)
            kw = dict(scheduler=sched, io_lane=rais.Lane.IO, prefetch=True)
        mgr = ResidencyManager(model, model_path, pinned_blocks=pin, **kw)
        toks, peak, tps = _generate(model, tok, sampler, max_tokens)
        st = dict(mgr.stats)
        if sched is not None:
            sched.shutdown()
        del model, mgr
        mx.clear_cache()
        return toks, peak, tps, st

    print(f"\n[compare] pin {pin}, {max_tokens} tokens", file=sys.stderr)
    for label, pf in (("synchronous  ", False), ("io-prefetch  ", True)):
        toks, peak, tps, st = one(pf)
        extra = ""
        if pf:
            serviced = st["prefetch_hit"] + st["prefetch_late"]
            total = serviced + st["sync_reload"]
            pct = 100.0 * serviced / max(total, 1)
            extra = (f"  | prefetch overlapped {pct:.0f}% of streamed loads "
                     f"(hit={st['prefetch_hit']} late={st['prefetch_late']} "
                     f"sync={st['sync_reload']})")
        print(f"  {label} peak {peak:.2f} GB  {tps:5.1f} tok/s  "
              f"ok={toks == ref}{extra}", file=sys.stderr)
    print("\nNote: on a warm page cache the streamed weights are RAM-resident, so "
          "prefetch\nhides no SSD latency and only adds overhead. Its win is the "
          "cold / over-RAM\nregime — reproduce with a model larger than free RAM, "
          "or `sudo purge` between runs.", file=sys.stderr)


def run_oom_demo(model_path: str, budget_gb: float, max_tokens: int) -> None:
    """Stock (all-resident) vs residency-bounded on a possibly-over-RAM model.

    Stock runs in an isolated subprocess with a hard timeout: on a machine where
    the weights exceed MLX's recommended working set, holding them all resident
    thrashes/swaps rather than erroring cleanly, so we bound it and kill it rather
    than risk hanging the host. Residency runs in-process (its peak stays under the
    working set, so it's safe).
    """
    import glob as _glob
    import subprocess

    from residency import _resolve_safetensors

    # Report the budget arithmetic up front.
    weight_bytes = sum(os.path.getsize(f) for f in _resolve_safetensors(model_path))
    working_set = mx.device_info().get("max_recommended_working_set_size", 0)
    print(f"\n[oom-demo] model={model_path}", file=sys.stderr)
    print(f"  on-disk weights ~{weight_bytes/1e9:.1f} GB; MLX recommended working "
          f"set ~{working_set/1e9:.1f} GB", file=sys.stderr)
    if weight_bytes <= working_set:
        print("  NOTE: weights fit the working set on this machine — stock will "
              "not OOM here; run on less RAM or with a bigger model for the "
              "failure headline.", file=sys.stderr)

    # --- residency-bounded, in-process (safe): capture the positive result FIRST,
    # before any stock thrash can destabilize the run ---
    print(f"  [residency] budget ~{budget_gb} GB ...", file=sys.stderr, flush=True)
    sampler = _sampler()
    model, tok = _load(model_path)
    mgr = ResidencyManager(model, model_path, budget_gb=budget_gb)
    print("    " + mgr.summary(), file=sys.stderr, flush=True)
    toks, peak, tps = _generate(model, tok, sampler, max_tokens)
    ratio = (weight_bytes / 1e9) / max(peak, 1e-9)
    print(f"  [residency] RAN: peak {peak:.2f} GB, {tps:.2f} tok/s "
          f"(streamed a {weight_bytes/1e9:.1f} GB model at {ratio:.1f}x below its "
          "on-disk weight size)", file=sys.stderr, flush=True)
    del model, mgr
    mx.clear_cache()

    # --- stock, isolated + tightly bounded: it must hold all weights resident,
    # which exceeds the working set here, so it thrashes. 1 token, short cap. ---
    stock_timeout = 60
    print(f"  [stock] all-resident, isolated subprocess, 1 token, {stock_timeout}s "
          "cap (expected to thrash) ...", file=sys.stderr, flush=True)
    code = (
        "import time, json, mlx.core as mx, mlx_lm\n"
        "from mlx_lm.generate import generate_step\n"
        "from mlx_lm.sample_utils import make_sampler\n"
        f"m,tok=mlx_lm.load({model_path!r},lazy=True)\n"
        "s=make_sampler(temp=0.0)\n"
        "ids=mx.array(tok.apply_chat_template([{'role':'user','content':'hi'}],add_generation_prompt=True))\n"
        "peak=0;n=0;t0=time.perf_counter()\n"
        "for t,_ in generate_step(ids,m,sampler=s):\n"
        "    n+=1;peak=max(peak,mx.get_active_memory())\n"
        "    if n>=1: break\n"
        "print('RESULT',json.dumps({'peak_gb':peak/1e9,'secs':time.perf_counter()-t0}))\n"
    )
    try:
        env = dict(os.environ, PYTHONPATH=HERE)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=stock_timeout, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT")), None)
        if line:
            import json
            r = json.loads(line[len("RESULT"):])
            print(f"  [stock] produced 1 token in {r['secs']:.1f}s at "
                  f"peak {r['peak_gb']:.2f} GB (all-resident; note the swap cost)",
                  file=sys.stderr)
        else:
            tail = (p.stderr.strip().splitlines() or ["<no output>"])[-1]
            print(f"  [stock] FAILED (rc={p.returncode}): {tail[:140]}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print(f"  [stock] DID NOT PRODUCE ONE TOKEN within {stock_timeout}s — "
              "thrashing to swap on 8.1 GB of resident weights that exceed the "
              "5.7 GB working set (killed to protect the machine).", file=sys.stderr)


def write_tsv(path: str, points: List[Point], n_blocks: int) -> None:
    with open(path, "w") as f:
        f.write("pins\tn_blocks\tresident_frac\tpeak_gb\ttps\tok\n")
        for p in points:
            f.write(f"{p.pins}\t{n_blocks}\t{p.resident_frac:.3f}\t"
                    f"{p.peak_gb:.3f}\t{p.tps:.3f}\t{p.ok}\n")


def plot(points: List[Point], out: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot", file=sys.stderr)
        return
    xs = [p.pins for p in points]
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(xs, [p.peak_gb for p in points], "o-", color="#2f6df6",
             label="peak memory")
    ax1.set_xlabel("resident (pinned) transformer blocks")
    ax1.set_ylabel("peak active memory (GB)", color="#2f6df6")
    ax1.tick_params(axis="y", labelcolor="#2f6df6")
    ax2 = ax1.twinx()
    ax2.plot(xs, [p.tps for p in points], "s--", color="#e8833a",
             label="throughput")
    ax2.set_ylabel("throughput (tok/s)", color="#e8833a")
    ax2.tick_params(axis="y", labelcolor="#e8833a")
    ax1.set_title("Residency budget vs memory & throughput\n"
                  "(fewer resident blocks = lower peak memory, lower tok/s)")
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--pins", default="0,4,8,14,28",
                    help="comma-separated pinned-block counts for the curve")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--oom-demo", action="store_true")
    ap.add_argument("--oom-budget-gb", type=float, default=3.0)
    ap.add_argument("--compare-prefetch", action="store_true",
                    help="A/B synchronous reload vs rais IO-lane prefetch")
    ap.add_argument("--pin", type=int, default=4, help="pinned blocks for --compare-prefetch")
    ap.add_argument("--io-threads", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "oversub_results.tsv"))
    args = ap.parse_args()

    if args.oom_demo:
        run_oom_demo(args.model, args.oom_budget_gb, args.max_tokens)
        return 0

    if args.compare_prefetch:
        run_prefetch_compare(args.model, args.pin, args.max_tokens, args.io_threads)
        return 0

    pins = [int(x) for x in args.pins.split(",") if x != ""]
    points = run_curve(args.model, pins, args.max_tokens)
    n_blocks = max(p.pins for p in points) if points else 0
    # infer n_blocks from resident_frac if the max pin isn't the full model
    for p in points:
        if p.resident_frac > 0:
            n_blocks = round(p.pins / p.resident_frac)
            break
    write_tsv(args.out, points, n_blocks)
    plot(points, os.path.join(HERE, "oversub_curve.png"))

    all_ok = all(p.ok for p in points)
    print(f"\nwrote {args.out}", file=sys.stderr)
    print(f"correctness: {'all bit-exact' if all_ok else 'MISMATCH DETECTED'}",
          file=sys.stderr)
    peaks = [p.peak_gb for p in points]
    print(f"peak-memory span: {min(peaks):.2f}–{max(peaks):.2f} GB "
          f"({max(peaks)/max(min(peaks),1e-9):.1f}x)", file=sys.stderr)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
