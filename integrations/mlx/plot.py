"""Render the concurrency benchmark results (results.tsv) to PNGs.

Two plots, both theme-neutral:
  ttft_tail.png   interactive TTFT p50/p95/p99 per system, under background load
  fairness.png    background throughput retained under interactive load

    python3 integrations/mlx/plot.py
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Brand-neutral, colorblind-safe: rais highlighted, baselines muted.
COLORS = {"rais": "#2f6df6", "rais-noqos": "#9aa4b2", "server": "#e8833a"}


def load(path: str):
    # values[(system, scenario, metric)][stat] = float
    values = defaultdict(dict)
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row["system"], row["scenario"], row["metric"])
            values[key][row["stat"]] = float(row["value"])
    return values


def plot_ttft_tail(values, out):
    systems = [s for s in ("rais", "rais-noqos", "server")
               if (s, "under_load", "ttft_ms") in values]
    if not systems:
        print("no under_load TTFT data; skipping ttft_tail.png")
        return
    stats = ["p50", "p95", "p99"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    width = 0.25
    for i, stat in enumerate(stats):
        xs = [j + i * width for j in range(len(systems))]
        ys = [values[(s, "under_load", "ttft_ms")].get(stat, 0) for s in systems]
        ax.bar(xs, ys, width, label=stat,
               color=[COLORS[s] for s in systems],
               alpha=0.55 + 0.2 * i, edgecolor="white", linewidth=0.5)
    ax.set_xticks([j + width for j in range(len(systems))])
    ax.set_xticklabels(systems)
    ax.set_ylabel("interactive TTFT (ms)")
    ax.set_title("Interactive time-to-first-token under background load\n(lower is better)")
    ax.legend(title="percentile", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_fairness(values, out):
    rows = []
    for system in ("rais", "rais-noqos", "server"):
        load = values.get((system, "under_load", "bg_tps"), {}).get("value")
        iso = (values.get((system, "bg_isolated", "bg_tps"), {}).get("value")
               or values.get(("rais", "bg_isolated", "bg_tps"), {}).get("value"))
        if load and iso:
            rows.append((system, 100.0 * load / iso))
    if not rows:
        print("no fairness data; skipping fairness.png")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([r[0] for r in rows], [r[1] for r in rows],
           color=[COLORS[r[0]] for r in rows], edgecolor="white")
    ax.axhline(100, ls="--", lw=1, color="#888")
    ax.set_ylabel("background throughput retained (%)")
    ax.set_title("Fairness: background progress under interactive load\n(higher is better; 100% = no slowdown)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.tsv")
    if not os.path.exists(path):
        print(f"{path} not found; run bench_concurrency.py first", file=sys.stderr)
        return 1
    values = load(path)
    plot_ttft_tail(values, os.path.join(HERE, "ttft_tail.png"))
    plot_fairness(values, os.path.join(HERE, "fairness.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
