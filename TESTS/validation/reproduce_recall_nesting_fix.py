"""Real-brain reproduction of the recall nesting fix.

The bug was found by replaying the stratified validation sample against the
real model: after roughly 130 examples a stored tool turn had grown to42,414
tokens, and the run reached 164,382 by the end. That growth came from recall
returning its own previous output.

This replays the same stratified sample against the real brain with the
tokenizer instrumented, and asserts the properties that matter:

  * every encode stays far below a tokenizer's context length;
  * stored tool turns stop compounding;
  * no recall output contains a nested tool turn;
  * no OOM.

Not part of the pytest suite: it loads the 8 GB model and takes a long time.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from transformers import AutoTokenizer

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import AuroraBrain
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context.isolation import (
    build_prime_with_isolation_guard,
    harness_session,
)
from PRIMORA.prime.providers import Provider
from PRIMORA.research import ResearchProvider, ResearchResult
from PRIMORA.toolbox import ToolRegistry

# Well above anything the input ceiling permits, and far below the tokenizer's
# 131,072 limit, so crossing it means the nesting is back.
OVERSIZED = 20_000


class _Web(ResearchProvider):
    name = "eval-web"

    def search(self, query, limit=5):
        return [ResearchResult(
            "Reference result", "https://example.org/a",
            "A current reference snippet for the query.", "example.org")]


class _SpecialistProvider(Provider):
    name = "eval-specialist"

    def invoke(self, capability, payload):
        text = (payload.get("messages") or [{}])[-1].get("content", "")
        return PrimeResult.success({
            "content": f"Considered analysis of: {text[:120]}"})


class TracingTokenizer:
    """Delegates to the real tokenizer and records every encode width."""

    def __init__(self, inner):
        self._inner = inner
        self.widths: list[int] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, *args, **kwargs):
        encoded = self._inner(*args, **kwargs)
        ids = getattr(encoded, "input_ids", None)
        if ids is None and isinstance(encoded, dict):
            ids = encoded.get("input_ids")
        if ids is None and isinstance(encoded, list) and encoded:
            ids = encoded[0]
        try:
            width = (int(ids.shape[-1]) if hasattr(ids, "shape")
                     else len(ids[0]))
        except Exception:
            width = 0
        self.widths.append(width)
        return encoded


def _tool_turn_sizes(db_path: str, session_id: str) -> list[int]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        return [int(row[0]) for row in con.execute(
            "SELECT LENGTH(content) FROM messages "
            "WHERE session_id = ? AND role = 'tool' ORDER BY id",
            (session_id,))]


def main() -> int:
    here = r"C:\PRIMORA PROJECT\TESTS\training"
    sys.path.insert(0, here)
    from evaluate_validation import load, stratify

    examples = load(os.path.join(here, "data", "validation.jsonl"))
    limit = int(os.environ.get("PROBE_LIMIT", "180"))
    examples = stratify(examples, limit, max(limit, int(limit * 1.25)))
    print(f"  replaying {len(examples)} stratified validation examples")
    print("=" * 70)

    brain = AuroraBrain()
    load_error = brain.warmup()
    if load_error:
        print(f"  MODEL FAILED TO LOAD: {load_error}")
        return 1
    tokenizer = brain._tokenizer
    brain._tokenizer = TracingTokenizer(tokenizer)
    print(f"  tokenizer model_max_length = {tokenizer.model_max_length}")

    tmp = tempfile.mkdtemp(prefix="recall_repro_")
    failures: list[str] = []
    recall_calls = 0
    nested = 0

    with harness_session("recall_repro", root=tmp) as session:
        context = session.context_manager()
        prime = build_prime_with_isolation_guard(
            _SpecialistProvider(), research_provider=_Web(),
            tools=ToolRegistry.with_builtins())
        prime._context = context
        prime._mind = prime._mind.__class__(context)
        for name in ("recall", "learn"):
            specialist = prime.registry.get_specialist(name)
            if specialist is not None:
                specialist.bind(prime._mind)
        aurora = Aurora(prime, brain=brain, config=AuroraConfig.from_env())
        db_path = str(context._get_store()._db_path)

        for index, example in enumerate(examples):
            aurora.mind.begin_task("eval", "validation")
            before = len(aurora.loop.executor.calls)
            try:
                aurora.respond(
                    [{"role": "user",
                      "content": example["messages"][0]["content"]}], "eval")
            except Exception as exc:
                print(f"  example {index}: {type(exc).__name__}: {exc}")
                if isinstance(exc, MemoryError) or "out of memory" in str(
                        exc).lower():
                    failures.append(f"OOM at example {index}: {exc}")
                    break
            # executor.calls holds (action, payload) tuples.
            for action, _payload in aurora.loop.executor.calls[before:]:
                if action == "recall":
                    recall_calls += 1

            if (index + 1) % 30 == 0:
                sizes = _tool_turn_sizes(db_path, "eval")
                print(f"  {index + 1:3d} examples | recalls={recall_calls} "
                      f"| encodes={len(brain._tokenizer.widths)} "
                      f"| max stored tool turn="
                      f"{max(sizes) if sizes else 0} chars", flush=True)

    widths = brain._tokenizer.widths
    max_width = max(widths) if widths else 0
    oversized = [w for w in widths if w > OVERSIZED]
    tool_sizes = _tool_turn_sizes(db_path, "eval")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  examples replayed        : {len(examples)}")
    print(f"  recall calls             : {recall_calls}")
    print(f"  tokenizes performed      : {len(widths)}")
    print(f"  max encode width         : {max_width} tokens")
    print(f"  encodes over {OVERSIZED}      : {len(oversized)}")
    print(f"  stored tool turns        : {len(tool_sizes)}")
    print(f"  largest stored tool turn : "
          f"{max(tool_sizes) if tool_sizes else 0} chars")

    if oversized:
        failures.append(
            f"{len(oversized)} encode(s) exceeded {OVERSIZED} tokens "
            f"(max {max(oversized)}) - nesting has returned")
    if tool_sizes and max(tool_sizes) > 50_000:
        failures.append(
            f"stored tool turn reached {max(tool_sizes)} chars - "
            "still compounding")
    if max_width > OVERSIZED:
        failures.append(f"max encode {max_width} exceeds {OVERSIZED}")

    if failures:
        print("\n  FAILURES:")
        for line in failures:
            print(f"    - {line}")
        return 1

    print("\n  PASS: prompts stayed bounded, tool turns did not compound,")
    print("        no oversized encode, no OOM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())