#!/usr/bin/env python3
"""Generate a markdown comparison table for scheduler baselines.

This script consumes:
1) RAIS concurrent benchmark output (mlx_concurrent_results.tsv)
2) Optional llama.cpp baseline TSV

Output: markdown file suitable for README snippets or launch posts.
"""

import argparse
import csv
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def get_metric(rows: list[dict[str, str]], approach: str, traffic_class: str, key: str) -> float:
    for row in rows:
        if row.get("approach") == approach and row.get("class") == traffic_class:
            return float(row[key])
    raise ValueError(
        f"Missing row in RAIS TSV for approach={approach}, class={traffic_class}, key={key}"
    )


def parse_llama_baseline(path: Path) -> tuple[float, float]:
    rows = read_tsv(path)
    for row in rows:
        klass = row.get("class", "").strip().lower()
        if klass != "interactive":
            continue

        ttft = row.get("ttft_ms")
        e2e = row.get("e2e_ms")
        if ttft is None or e2e is None:
            continue
        return float(ttft), float(e2e)

    raise ValueError("No interactive llama.cpp baseline row found in llama TSV")


def speedup(base: float, rais: float) -> str:
    if rais <= 0:
        return "-"
    return f"{(base / rais):.2f}x"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown benchmark comparison table.")
    parser.add_argument(
        "--rais-tsv",
        type=Path,
        default=Path("experiments/mlx_concurrent_results.tsv"),
        help="TSV from experiments/bench_mlx_concurrent.py",
    )
    parser.add_argument(
        "--llama-tsv",
        type=Path,
        default=None,
        help="Optional TSV with llama.cpp metrics (interactive row required).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/benchmark_comparison.md"),
        help="Output markdown path.",
    )
    args = parser.parse_args()

    rais_rows = read_tsv(args.rais_tsv)
    naive_ttft = get_metric(rais_rows, "naive", "interactive", "ttft_ms")
    naive_e2e = get_metric(rais_rows, "naive", "interactive", "e2e_ms")
    rais_ttft = get_metric(rais_rows, "rais", "interactive", "ttft_ms")
    rais_e2e = get_metric(rais_rows, "rais", "interactive", "e2e_ms")

    lines = [
        "# Benchmark Comparison",
        "",
        "Generated with `experiments/bench_compare_tools.py`.",
        "",
        "| System | Interactive TTFT (ms) | Interactive E2E (ms) | TTFT vs RAIS |",
        "|---|---:|---:|---:|",
        f"| Naive thread pool (FIFO) | {naive_ttft:.1f} | {naive_e2e:.1f} | {speedup(naive_ttft, rais_ttft)} |",
        f"| RAIS priority lanes | {rais_ttft:.1f} | {rais_e2e:.1f} | 1.00x |",
    ]

    if args.llama_tsv:
        llama_ttft, llama_e2e = parse_llama_baseline(args.llama_tsv)
        lines.append(
            f"| llama.cpp baseline | {llama_ttft:.1f} | {llama_e2e:.1f} | {speedup(llama_ttft, rais_ttft)} |"
        )
    else:
        lines.append("| llama.cpp baseline | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "Notes:",
            "- Keep client counts, prompt lengths, and model sizes consistent across tools.",
            "- Prefer the same model quantization for apples-to-apples TTFT comparison.",
        ]
    )

    args.out.write_text("\n".join(lines) + "\n")
    print(f"Wrote comparison markdown to {args.out}")


if __name__ == "__main__":
    main()
