"""Per-layer weight residency manager for memory-oversubscribed mlx-lm inference.

MLX materializes a model's entire weight set into resident memory on first use and
keeps it there, so a model whose weights exceed the memory budget cannot run under
stock mlx-lm. This manager makes it run at *bounded* peak memory by owning weight
residency: it pins as many transformer blocks as fit in a budget and streams the
rest — evicting a streamed block's weights after it computes and re-materializing
them from the mmap'd safetensors before the block is needed again.

The GO/NO-GO spike established the mechanics this relies on (see oversubscription.md):

  * Evicting a block's weights (module.update with placeholders, dropping the last
    Python reference to the big arrays) lowers MLX active memory by exactly the
    block's size. NO lingering reference to those arrays may survive, or MLX cannot
    free them.
  * Reloading from the lazy, mmap-backed safetensors is bit-exact.

This first cut is synchronous (correct + memory-bounded). Overlapping the reload of
block i+1 with the compute of block i via rais's IO lane is the next step.
"""

from __future__ import annotations

import glob
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_unflatten


def _realize(out) -> None:
    """Force-evaluate a block's output (array or tuple of arrays)."""
    if isinstance(out, (tuple, list)):
        arrays = [o for o in out if isinstance(o, mx.array)]
        if arrays:
            mx.eval(arrays)
    elif isinstance(out, mx.array):
        mx.eval(out)


def _resolve_safetensors(repo_or_path: str) -> List[str]:
    """List the model's safetensors shard files (downloading the repo if needed)."""
    import os

    path = repo_or_path
    if not os.path.isdir(path):
        path = snapshot_download(repo_or_path, allow_patterns=["*.safetensors"])
    files = sorted(glob.glob(str(path) + "/*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors found for {repo_or_path}")
    return files


def _index_blocks(files: List[str], n: int) -> Dict[int, List[str]]:
    """Map block index -> shard files that hold any of its tensors.

    Reads only tensor *names* (mx.load is lazy/mmap; keys don't materialize data),
    then drops the dict so nothing stays resident.
    """
    block_files: Dict[int, set] = {i: set() for i in range(n)}
    prefixes = {i: f"model.layers.{i}." for i in range(n)}
    for f in files:
        d = mx.load(f)
        keys = list(d.keys())
        del d
        for k in keys:
            for i, pfx in prefixes.items():
                if k.startswith(pfx):
                    block_files[i].add(f)
                    break
    return {i: sorted(fs) for i, fs in block_files.items()}


def _make_streamed_class(base: type) -> type:
    """Subclass of a transformer block that drives residency around ``__call__``.

    We re-class the existing block instances into this subclass rather than wrapping
    them, so every attribute the model introspects (``use_sliding``, ``self_attn``,
    parameters, ``isinstance`` checks) keeps working — only the call is intercepted.
    """

    class _Streamed(base):  # type: ignore[valid-type, misc]
        def __call__(self, *args, **kwargs):
            mgr = self.__dict__["_rmgr"]
            idx = self.__dict__["_ridx"]
            mgr.ensure_resident(idx)
            out = base.__call__(self, *args, **kwargs)
            mgr.after_block(idx)
            # Force materialization of this block's output so the streamed
            # weights (already dereferenced by after_block) are freed before the
            # next block loads. Without this, MLX's lazy graph keeps every
            # block's weights alive until the end-of-token eval, and peak memory
            # equals the whole model — defeating the budget.
            _realize(out)
            return out

    _Streamed.__name__ = f"Streamed{base.__name__}"
    _Streamed.__qualname__ = _Streamed.__name__
    return _Streamed


class ResidencyManager:
    """Bounds resident transformer-block weights to a memory budget.

    Args:
        model: an mlx-lm model (expects ``model.model.layers`` to be the block list).
        repo_or_path: model repo id or local dir, used as the lazy reload source.
        budget_gb: soft budget for *block* weights; pins as many whole blocks as fit.
        pinned_blocks: explicit number of blocks to keep always-resident (overrides
            budget_gb). The rest are streamed.
    """

    def __init__(
        self,
        model: nn.Module,
        repo_or_path: str,
        budget_gb: Optional[float] = None,
        pinned_blocks: Optional[int] = None,
    ):
        self._model = model
        self._blocks: List[nn.Module] = list(model.model.layers)
        self.n = len(self._blocks)
        # Reload source: shard file paths, indexed per block. We re-open the
        # shard on each reload to get FRESH lazy arrays — holding a persistent
        # dict of weight arrays would pin them resident once materialized and
        # defeat eviction entirely.
        self._files = _resolve_safetensors(repo_or_path)
        self._block_files = _index_blocks(self._files, self.n)

        # Per-block byte sizes. Do NOT retain the flattened arrays (a stray ref
        # would defeat eviction) — sum and drop immediately.
        self._block_bytes: List[int] = []
        for b in self._blocks:
            leaves = tree_flatten(b.parameters())
            self._block_bytes.append(sum(a.nbytes for _, a in leaves))
            del leaves
        self.total_block_bytes = sum(self._block_bytes)
        self.avg_block_bytes = self.total_block_bytes / max(self.n, 1)

        self.pinned = self._decide_pinned(budget_gb, pinned_blocks)
        # Pin a contiguous prefix; for token-by-token decode every block is used
        # once per token, so which blocks are pinned doesn't affect reuse — only
        # how many.
        self._pinned_set = set(range(self.pinned))
        self._resident = set(range(self.n))  # all resident right after load

        # Intercept each block's forward by re-classing it (same object, same
        # attributes; only __call__ changes). All blocks share one class.
        streamed_cls = _make_streamed_class(type(self._blocks[0]))
        for i, b in enumerate(self._blocks):
            b.__dict__["_ridx"] = i
            b.__dict__["_rmgr"] = self
            b.__class__ = streamed_cls

        # Establish the bounded state immediately: evict everything not pinned.
        for i in range(self.n):
            if i not in self._pinned_set:
                self._evict(i)

    def _decide_pinned(self, budget_gb, pinned_blocks) -> int:
        if pinned_blocks is not None:
            return max(0, min(self.n, int(pinned_blocks)))
        if budget_gb is not None:
            k = int((budget_gb * 1e9) // max(self.avg_block_bytes, 1))
            return max(0, min(self.n, k))
        return self.n  # no budget => behave like stock mlx-lm

    # --- residency primitives -------------------------------------------------
    def _evict(self, idx: int) -> None:
        inner = self._blocks[idx]
        placeholder = tree_unflatten(
            [(k, mx.zeros(1)) for k, _ in tree_flatten(inner.parameters())]
        )
        inner.update(placeholder)
        self._resident.discard(idx)

    def _reload(self, idx: int) -> None:
        inner = self._blocks[idx]
        prefix = f"model.layers.{idx}."
        sub = []
        for f in self._block_files[idx]:
            d = mx.load(f)  # fresh lazy/mmap arrays; freed on next reload
            sub.extend(
                (k[len(prefix):], v) for k, v in d.items() if k.startswith(prefix)
            )
            # `sub` holds refs to this block's arrays; the rest of `d` is dropped.
            del d
        inner.update(tree_unflatten(sub))
        self._resident.add(idx)

    # --- hooks called by _StreamedBlock --------------------------------------
    def ensure_resident(self, idx: int) -> None:
        if idx not in self._resident:
            self._reload(idx)

    def after_block(self, idx: int) -> None:
        if idx not in self._pinned_set:
            self._evict(idx)

    # --- reporting ------------------------------------------------------------
    @property
    def streamed(self) -> int:
        return self.n - self.pinned

    def summary(self) -> str:
        return (
            f"{self.n} blocks, pin {self.pinned} / stream {self.streamed}; "
            f"block avg {self.avg_block_bytes/1e6:.1f} MB, "
            f"pinned-weight budget ~{self.pinned*self.avg_block_bytes/1e9:.2f} GB "
            f"of {self.total_block_bytes/1e9:.2f} GB total"
        )


def _smoke() -> int:
    """Run a short generation under a residency budget; check memory + correctness."""
    import argparse
    import time

    import mlx_lm
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    ap.add_argument("--pinned", type=int, default=4,
                    help="blocks kept always-resident (ignored if --budget-gb set)")
    ap.add_argument("--budget-gb", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args()

    prompt = "Explain unified memory on Apple Silicon briefly."
    sampler = make_sampler(temp=0.0)  # greedy => deterministic for correctness check

    def encode(tok):
        return mx.array(tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True))

    def run(model, tok, k):
        toks = []
        for t, _ in generate_step(encode(tok), model, sampler=sampler):
            toks.append(int(t.item()) if hasattr(t, 'item') else int(t))
            if len(toks) >= k:
                break
        return toks

    # Reference: stock full-resident mlx-lm.
    print("reference (full-resident) ...", flush=True)
    model, tok = mlx_lm.load(args.model, lazy=True)
    ref = run(model, tok, args.max_tokens)
    del model
    mx.clear_cache()

    # Streamed under budget.
    print("streamed (residency-bounded) ...", flush=True)
    model, tok = mlx_lm.load(args.model, lazy=True)
    mgr = ResidencyManager(
        model, args.model,
        budget_gb=args.budget_gb,
        pinned_blocks=None if args.budget_gb is not None else args.pinned,
    )
    print("  " + mgr.summary())

    peak = 0.0
    toks = []
    t0 = time.perf_counter()
    for t, _ in generate_step(encode(tok), model, sampler=sampler):
        toks.append(int(t.item()) if hasattr(t, 'item') else int(t))
        peak = max(peak, mx.get_active_memory())
        if len(toks) >= args.max_tokens:
            break
    dt = time.perf_counter() - t0

    match = toks == ref
    print(f"\n  peak active memory : {peak/1e9:.2f} GB  "
          f"(full weights are {mgr.total_block_bytes/1e9:.2f} GB of blocks)")
    print(f"  throughput         : {len(toks)/dt:.1f} tok/s")
    print(f"  output matches ref : {match}")
    print(f"  text: {tok.decode(toks)[:160]!r}")
    print(f"\n=== {'OK' if match else 'MISMATCH'} ===")
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(_smoke())
