"""Real mlx-lm inference driven by the rais scheduler.

This is the first non-simulated rais integration: instead of `sleep_for`, every
decode step is an actual `mlx_lm` forward pass, scheduled by rais.

Design (see integrations/mlx/README.md and the plan):

  One model + one GPU + the Python GIL means decode steps are physically serial.
  So rais's job here is *ordering*: each request's next-token step is submitted as
  a task on that request's QoS lane (INTERACTIVE for latency-sensitive requests,
  BULK for background batch work). When the step finishes it re-submits the next
  step on the same lane (continuations, since the Python bindings expose only
  `submit`). rais pops lanes in strict priority order, so an interactive request's
  decode preempts background decode, while rais's anti-starvation promotion keeps
  background progressing.

  We run the scheduler with a single worker (`num_workers=1`) so model access is
  serialized (MLX thread-safety + the GIL); the scheduler still enforces lane
  priority across the interleaved step stream.

Usage (smoke test):
    python3 integrations/mlx/rais_mlx.py --prompt "Explain unified memory" --max-tokens 48
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler


# --------------------------------------------------------------------------- #
# Import the compiled rais scheduler module (pyrais, output name "rais").
# --------------------------------------------------------------------------- #
def _import_rais():
    try:
        import rais  # type: ignore

        return rais
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    candidates = [
        os.environ.get("RAIS_PYTHONPATH", ""),
        os.path.join(repo_root, "build"),
        os.path.join(repo_root, "build-python"),
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            sys.path.insert(0, path)
            try:
                import rais  # type: ignore

                return rais
            except ImportError:
                sys.path.pop(0)
    raise ImportError(
        "Could not import the compiled `rais` module. Build it with "
        "`cmake -S . -B build -DRAIS_BUILD_PYTHON=ON && cmake --build build` "
        "and/or set RAIS_PYTHONPATH to the directory containing rais*.so."
    )


rais = _import_rais()

DEFAULT_MODEL = os.environ.get(
    "RAIS_MLX_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit"
)


@dataclass
class Request:
    """A single generation request tracked across its scheduled decode steps."""

    lane: object
    max_tokens: int
    on_token: Optional[Callable[[str, int], None]] = None
    # If set, the request stops issuing further steps once this event fires
    # (used to bound background streams to a benchmark window).
    stop_event: Optional[threading.Event] = None

    # Timing (perf_counter seconds).
    submit_time: float = 0.0
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    end_time: Optional[float] = None

    tokens: List[int] = field(default_factory=list)
    itl: List[float] = field(default_factory=list)  # inter-token latencies (s)
    text: str = ""

    _gen: object = None
    _detok: object = None
    _done: threading.Event = field(default_factory=threading.Event)

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.submit_time

    @property
    def e2e(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return self.end_time - self.submit_time

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout)

    @property
    def done(self) -> bool:
        return self._done.is_set()


class RaisEngine:
    """Loads an mlx-lm model once and schedules decode steps through rais."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        num_workers: int = 1,
        temp: float = 0.0,
    ):
        self.model, self.tokenizer = mlx_lm.load(model_path)
        self.sampler = make_sampler(temp=temp)
        # io_thread_count=0: no SSD reads in this path, all work is CPU/model.
        self.scheduler = rais.Scheduler(num_workers=num_workers, io_thread_count=0)
        self._eos_ids = set(getattr(self.tokenizer, "eos_token_ids", set()))
        eid = getattr(self.tokenizer, "eos_token_id", None)
        if eid is not None:
            self._eos_ids.add(eid)

    def _encode_prompt(self, prompt_text: str) -> mx.array:
        messages = [{"role": "user", "content": prompt_text}]
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        return mx.array(ids)

    def submit(
        self,
        prompt_text: str,
        lane,
        max_tokens: int = 128,
        on_token: Optional[Callable[[str, int], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Request:
        """Submit a generation request; returns immediately with a Request handle."""
        prompt = self._encode_prompt(prompt_text)
        cache = make_prompt_cache(self.model)
        gen = generate_step(
            prompt, self.model, prompt_cache=cache, sampler=self.sampler
        )
        detok = self.tokenizer.detokenizer

        req = Request(
            lane=lane,
            max_tokens=max_tokens,
            on_token=on_token,
            stop_event=stop_event,
        )
        req._gen = gen
        req._detok = detok
        req.submit_time = time.perf_counter()
        # Kick off the first decode step (which also runs prompt prefill).
        self.scheduler.submit(lambda: self._step(req), lane)
        return req

    def _step(self, req: Request) -> None:
        """Run exactly one decode step, then re-submit the next one (continuation)."""
        try:
            token, _logprobs = next(req._gen)
        except StopIteration:
            self._finalize(req)
            return

        tok = int(token.item()) if hasattr(token, "item") else int(token)
        now = time.perf_counter()

        if req.first_token_time is None:
            req.first_token_time = now
        else:
            req.itl.append(now - req.last_token_time)
        req.last_token_time = now

        is_eos = tok in self._eos_ids
        if not is_eos:
            req.tokens.append(tok)
            # Incremental detokenization for streaming output.
            try:
                req._detok.add_token(tok)
                segment = req._detok.last_segment
            except Exception:
                segment = self.tokenizer.decode([tok])
            if segment:
                req.text += segment
                if req.on_token is not None:
                    req.on_token(segment, tok)

        stopped = req.stop_event is not None and req.stop_event.is_set()
        if is_eos or stopped or len(req.tokens) >= req.max_tokens:
            self._finalize(req)
            return

        # Continuation: schedule the next step on the same lane.
        self.scheduler.submit(lambda: self._step(req), req.lane)

    def _finalize(self, req: Request) -> None:
        try:
            req._detok.finalize()
            tail = req._detok.last_segment
            if tail:
                req.text += tail
        except Exception:
            pass
        req.end_time = time.perf_counter()
        req._done.set()

    def shutdown(self) -> None:
        self.scheduler.shutdown()


def _smoke_test() -> int:
    ap = argparse.ArgumentParser(description="rais x mlx-lm smoke test")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default="Explain unified memory in one sentence.")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()

    print(f"Loading {args.model} ...", file=sys.stderr, flush=True)
    engine = RaisEngine(args.model, num_workers=1, temp=args.temp)
    print("Loaded. Generating (real tokens, scheduled by rais):\n", file=sys.stderr)

    t0 = time.perf_counter()
    req = engine.submit(
        args.prompt,
        rais.Lane.INTERACTIVE,
        max_tokens=args.max_tokens,
        on_token=lambda seg, _tok: (sys.stdout.write(seg), sys.stdout.flush()),
    )
    req.wait()
    dt = time.perf_counter() - t0

    print("\n", file=sys.stdout)
    n = len(req.tokens)
    ttft_ms = (req.ttft or 0.0) * 1e3
    tps = n / dt if dt > 0 else 0.0
    print(
        f"[rais] {n} tokens | TTFT {ttft_ms:.1f} ms | {tps:.1f} tok/s | "
        f"E2E {dt * 1e3:.1f} ms",
        file=sys.stderr,
    )
    engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
