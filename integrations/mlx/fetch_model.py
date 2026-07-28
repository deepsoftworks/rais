"""Download the mlx-lm model used by the rais concurrency benchmark.

Models land in the standard Hugging Face cache (~/.cache/huggingface). The
default is a 4-bit 3B that fits comfortably in 8 GB of unified memory; override
with --model or the RAIS_MLX_MODEL env var on a machine with more RAM.

    python3 integrations/mlx/fetch_model.py
    python3 integrations/mlx/fetch_model.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Kept in sync with rais_mlx.DEFAULT_MODEL, duplicated here so fetching the model
# does not require the compiled `rais` extension to be built yet.
DEFAULT_MODEL = os.environ.get(
    "RAIS_MLX_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit"
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the benchmark model")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    print(f"Fetching {args.model} into the Hugging Face cache ...", flush=True)
    t0 = time.perf_counter()
    # mlx_lm.load resolves + downloads the repo, then loads it lazily.
    import mlx_lm

    model, tokenizer = mlx_lm.load(args.model, lazy=True)
    dt = time.perf_counter() - t0
    print(f"OK: {args.model} ready ({dt:.1f}s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
