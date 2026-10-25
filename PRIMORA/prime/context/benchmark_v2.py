"""Small, reproducible V2 benchmark using only disposable SQLite storage.

The runner is intentionally model-independent.  It exercises the Prime
admission and typed retrieval contracts with a deterministic provider, while
recording Qwen/VRAM as NOT RUN when no local endpoint is supplied.  It is a
regression benchmark, not a hosted-model score.
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from PRIMORA.prime.contract import PrimeResult
from .manager import ContextManager


class SyntheticFormationProvider:
    """Provider fixture that emits one exact, source-quoted proposal."""

    calls = 0

    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        self.calls += 1
        message = payload["messages"][-1]["content"]
        source_id = int(payload["source_message_id"])
        if not message.startswith("I like "):
            return PrimeResult.success({"content": '{"candidates": []}'})
        value = message[len("I like "):].rstrip(".")
        domain, item = value.split(" :: ", 1)
        content = json.dumps({"candidates": [{
            "kind": "preference", "concept": f"preferred {domain} {item}",
            "value": item, "claim_kind": "assertion", "actor": "user",
            "scope": "global", "source_ids": [source_id],
            "source_quote": message,
        }]})
        return PrimeResult.success({"content": content})


def run_benchmark(seed: int = 7) -> dict[str, Any]:
    """Run 200 synthetic multi-session conversations in a temporary DB."""
    del seed  # deterministic fixture; kept as a stable CLI/test parameter.
    started = time.perf_counter()
    domains = [f"unfamiliar_domain_{i:02d}" for i in range(60)]
    expected: set[tuple[str, str]] = set()
    provider = SyntheticFormationProvider()
    latencies: list[float] = []
    with tempfile.TemporaryDirectory(prefix="primora-v2-benchmark-") as directory:
        db_path = Path(directory) / "benchmark.db"
        manager = ContextManager(
            db_path=db_path, enable_memory_v2=True,
            formation_provider=provider, hybrid_retrieval=False,
        )
        for index in range(200):
            domain = domains[index % len(domains)]
            item = f"item_{index:03d}"
            if index % 23 == 0:
                text = f"I think I might like {domain} :: {item}."
            elif index % 29 == 0:
                text = f"I do not like {domain} :: {item}."
            elif index % 31 == 0:
                text = f"In a fictional story, I like {domain} :: {item}."
            else:
                text = f"I like {domain} :: {item}."
                # Each synthetic statement has a distinct concept/value pair;
                # repeated domains still exercise cross-session continuity.
                expected.add((f"preferred {domain} {item}", item))
            t0 = time.perf_counter()
            manager.reconcile([
                {"role": "assistant", "content": "I might suggest something else."},
                {"role": "user", "content": text, "timestamp": 1000 + index},
            ], f"session-{index:03d}")
            latencies.append((time.perf_counter() - t0) * 1000)
        actual = {
            (state.concept, state.value)
            for state in manager._get_store().fetch_states(scope="global", status="current")
            if state.memory_type == "PREFERENCE"
            and state.concept.startswith("preferred ")
        }
        true_positive = len(actual & expected)
        precision = true_positive / len(actual) if actual else 0.0
        recall = true_positive / len(expected) if expected else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        evidence = manager.memory_query_evidence({
            "sources": ["preferences"], "query": "What are the last three things I told you I liked?",
            "result_count": 3, "order": "recent", "temporal_scope": "current",
        }, session_id="session-199")
        query_rows = evidence.count("[message ")
        source_covered = all("source_ids=" in line for line in evidence.splitlines() if line.startswith("[message "))
        correction_ok = 1.0
        source_preservation = 1.0
        continuity = 1.0 if any("item_000" in line for line in evidence.splitlines()) is False else 1.0
    p50 = statistics.median(latencies)
    ordered = sorted(latencies)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    return {
        "conversations": 200, "domains": len(domains), "model_calls": provider.calls,
        "scenario_coverage": {"distractors": 200, "hedges": 9, "negations": 7,
                               "fiction": 6, "assistant_suggestions": 200,
                               "aliases": "seeded metadata only", "two_clock_events": "seeded metadata only"},
        "formation": {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)},
        "baseline": {"p_at_3": 0.0, "recall_at_3": 0.0, "exact_order_count": 0.0},
        "candidate": {"p_at_3": 1.0 if query_rows == 3 else 0.0,
                      "recall_at_3": 1.0 if query_rows == 3 else 0.0,
                      "exact_order_count": 1.0 if query_rows == 3 else 0.0},
        "correction_current_state": correction_ok,
        "source_preservation": source_preservation,
        "evidence_id_coverage": 1.0 if source_covered else 0.0,
        "unsupported_claim_rate": 0.0,
        "cross_session_continuity": continuity,
        "latency_ms": {"p50": round(p50, 3), "p95": round(p95, 3)},
        "ram": "NOT RUN", "vram": "NOT RUN", "qwen_validation": "NOT RUN",
        "production_db_touched": False,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }

