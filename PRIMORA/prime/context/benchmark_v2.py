"""Reproducible, evidence-scored Memory V2 benchmark.

The benchmark owns its fixture and its ground truth. It runs the legacy
formation/retrieval path and the opt-in V2 path against the same messages,
timestamps, and expected source ids. Every score is calculated from persisted
rows or rendered evidence packets. The provider is a deterministic test
double, so Qwen validation is reported separately as unavailable.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PRIMORA.prime.contract import PrimeResult
from . import entities, observation
from .manager import ContextManager
from .store import default_db_path


@dataclass(frozen=True)
class GroundTruthCase:
    index: int
    domain: str
    text: str
    expected_memory: bool
    expected_positive_preference: bool


class SyntheticFormationProvider:
    """Deterministic provider test double; it never reports evaluation scores."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        self.calls += 1
        message = payload["messages"][-1]["content"]
        source_id = int(payload["source_message_id"])
        if not message.startswith("I like "):
            return PrimeResult.success({"content": '{"candidates": []}'})
        value = message[len("I like "):].rstrip(".")
        domain, item = value.split(" :: ", 1)
        return PrimeResult.success({"content": json.dumps({"candidates": [{
            "kind": "preference", "concept": f"preferred {domain} {item}",
            "value": item, "claim_kind": "assertion", "actor": "user",
            "scope": "global", "source_ids": [source_id],
            "source_quote": message,
        }]})})


def _fixture(seed: int) -> list[GroundTruthCase]:
    rng = random.Random(seed)
    domains = [f"unfamiliar_domain_{i:02d}" for i in range(60)]
    cases: list[GroundTruthCase] = []
    for index in range(200):
        domain = domains[index % len(domains)]
        item = f"item_{index:03d}"
        selector = rng.randrange(100)
        if selector < 5:
            text = f"I think I might like {domain} :: {item}."
            expected_memory = False
            positive = False
        elif selector < 9:
            text = f"I do not like {domain} :: {item}."
            # Explicit negation is a valid negative preference, while still
            # being excluded from a query asking what the user liked.
            expected_memory = True
            positive = False
        elif selector < 12:
            text = f"In a fictional story, I like {domain} :: {item}."
            expected_memory = False
            positive = False
        else:
            text = f"I like {domain} :: {item}."
            expected_memory = True
            positive = True
        cases.append(GroundTruthCase(index, domain, text, expected_memory, positive))
    return cases


def _accepted_source_ids(manager: ContextManager, eligible: set[int]) -> set[int]:
    accepted: set[int] = set()
    for state in manager._get_store().fetch_states(scope="global", status="current"):
        if state.authority == "user" and state.memory_type == "PREFERENCE":
            accepted.update(source for source in state.source_ids if source in eligible)
    return accepted


def _formation_scores(
    manager: ContextManager,
    cases: list[GroundTruthCase],
    source_ids: dict[int, int],
) -> dict[str, float]:
    expected = {source_ids[case.index] for case in cases if case.expected_memory}
    actual = _accepted_source_ids(manager, set(source_ids.values()))
    true_positive = len(actual & expected)
    precision = true_positive / len(actual) if actual else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "expected": float(len(expected)),
        "accepted": float(len(actual)),
    }


def _packet_ids(evidence: str) -> list[int]:
    ids: list[int] = []
    for line in evidence.splitlines():
        match = re.match(r"^\[message (\d+)[;,]", line)
        if match:
            ids.append(int(match.group(1)))
    return ids


def _retrieval_scores(evidence: str, expected_ids: list[int]) -> dict[str, float]:
    returned = _packet_ids(evidence)
    target = expected_ids[:3]
    relevant = len(set(returned[:3]) & set(target))
    denominator = len(target) or 1
    return {
        "p_at_3": relevant / denominator,
        "recall_at_3": relevant / denominator,
        "exact_order_count": 1.0 if returned[:3] == target else 0.0,
        "returned": float(len(returned[:3])),
    }


def _query(manager: ContextManager, query: str, session_id: str = "session-199") -> str:
    return manager.memory_query_evidence({
        "sources": ["preferences"],
        "query": query,
        "result_count": 3,
        "order": "recent",
        "temporal_scope": "current",
    }, session_id=session_id)


def _add_auxiliary_ground_truth(manager: ContextManager) -> dict[str, Any]:
    """Exercise alias, correction, two-clock, and abstention behavior."""
    store = manager._get_store()
    alias_source, *_ = store.append(
        "alias-session", 0, "user",
        "My project Flarnic is also called Flar.", timestamp=4000,
    )
    entity_id, _ = entities.create_candidate(
        store, "alias-session", "Flarnic Project", "PROJECT", [alias_source]
    )
    entities.confirm(store, entity_id)
    alias_id, _ = entities.add_alias_candidate(
        store, entity_id, "Flar", "alias-session", [alias_source]
    )
    entities.confirm_alias(store, alias_id)
    resolved, ambiguous = entities.resolve(store, "alias-session", "Flar")
    alias_ok = not ambiguous and len(resolved) == 1 and resolved[0].entity_id == entity_id

    old_source, *_ = store.append(
        "correction-session", 0, "user",
        "I like correction-domain :: old", timestamp=5000,
    )
    old_state = store.create_state(
        "correction-session", "global", "correction-domain", "old",
        [old_source], memory_type="PREFERENCE", timestamp=5000,
    )
    new_source, *_ = store.append(
        "correction-session", 1, "user",
        "Actually I like correction-domain :: new", timestamp=5010,
    )
    store.create_state_successor(
        "correction-session", "global", "correction-domain", "new",
        [new_source], [old_state], memory_type="PREFERENCE", timestamp=5010,
    )
    current = store.fetch_states(
        scope="global", concept="correction-domain", status="current"
    )
    historical = store.fetch_states(
        scope="global", concept="correction-domain", status="superseded"
    )
    correction_ok = bool(current and current[0].value == "new" and historical)
    source_preserved = bool(
        store.fetch_message(old_source) and store.fetch_message(new_source)
    )

    event_source, *_ = store.append(
        "clock-session", 0, "user",
        "I attended the Luminex gathering.", timestamp=6000,
    )
    event = observation.ObservationService(store, enabled=True).create(
        "clock-session",
        observation.ObservationProposal(
            "luminex gathering", "attended", [event_source],
            observation_type="event", predicate="attended",
            source_quotes={event_source: "attended the Luminex gathering"},
            occurred_from=100.0, occurred_until=200.0,
        ),
    )
    two_clock_ok = bool(
        event.observed_at == 6000.0
        and event.occurred_from == 100.0
        and event.occurred_until == 200.0
    )
    return {
        "alias_resolution": 1.0 if alias_ok else 0.0,
        "correction_current_state": 1.0 if correction_ok else 0.0,
        "source_preservation": 1.0 if source_preserved else 0.0,
        "two_clock_events": 1.0 if two_clock_ok else 0.0,
        "correction_positive_source": new_source,
        "abstention_query": (
            "What are the last three things I liked about nonexistent-domain?"
        ),
    }


def _abstention_score(manager: ContextManager, query: str) -> float:
    return 1.0 if _query(manager, query) == "" else 0.0


def _unsupported_rate(evidence: str, relevant_ids: set[int]) -> float:
    returned = _packet_ids(evidence)
    if not returned:
        return 0.0
    return len([item for item in returned if item not in relevant_ids]) / len(returned)


def _run_once(
    cases: list[GroundTruthCase], *, candidate: bool, seed: int,
) -> dict[str, Any]:
    provider = SyntheticFormationProvider() if candidate else None
    latencies: list[float] = []
    with tempfile.TemporaryDirectory(prefix="primora-v2-benchmark-") as directory:
        db_path = Path(directory) / ("candidate.db" if candidate else "baseline.db")
        manager = ContextManager(
            db_path=db_path, enable_memory_v2=candidate,
            formation_provider=provider, hybrid_retrieval=False,
        )
        source_ids: dict[int, int] = {}
        for case in cases:
            started = time.perf_counter()
            manager.reconcile([
                {"role": "assistant", "content": "I might suggest another option."},
                {"role": "user", "content": case.text, "timestamp": 1000 + case.index},
            ], f"session-{case.index:03d}")
            latencies.append((time.perf_counter() - started) * 1000)
            source_ids[case.index] = manager._get_store().fetch_latest(
                f"session-{case.index:03d}", 1
            )[0].message_id

        auxiliary = _add_auxiliary_ground_truth(manager)
        expected_positive = sorted(
            [
                *(source_ids[case.index] for case in cases if case.expected_positive_preference),
                auxiliary["correction_positive_source"],
            ],
            key=lambda message_id: manager._get_store().fetch_message(message_id).timestamp,
            reverse=True,
        )
        query = _query(manager, "What are the last three things I told you I liked?")
        domain_query = _query(
            manager, "What did I tell you I liked about unfamiliar_domain_00?"
        )
        relevant = set(expected_positive)
        source_preservation = auxiliary["source_preservation"] and all(
            manager._get_store().fetch_message(message_id) is not None
            for message_id in source_ids.values()
        )
        result: dict[str, Any] = {
            "formation": _formation_scores(manager, cases, source_ids),
            "retrieval": _retrieval_scores(query, expected_positive),
            "evidence_id_coverage": (
                1.0 if all(
                    "source_ids=" in line for line in query.splitlines()
                    if line.startswith("[message ")
                ) else 0.0
            ),
            "unsupported_claim_rate": _unsupported_rate(query, relevant),
            "cross_session_continuity": (
                1.0 if set(_packet_ids(domain_query)) & {
                    source_ids[case.index]
                    for case in cases
                    if case.index != 199 and case.domain == "unfamiliar_domain_00"
                    and case.expected_positive_preference
                } else 0.0
            ),
            "abstention": _abstention_score(manager, auxiliary["abstention_query"]),
            "latency_ms": latencies,
            "model_calls": provider.calls if provider else 0,
            "auxiliary": auxiliary,
            "source_preservation": 1.0 if source_preservation else 0.0,
            "production_db_touched": db_path.resolve() == default_db_path().resolve(),
            "seed": seed,
        }
    ordered = sorted(result["latency_ms"])
    result["latency_ms"] = {
        "p50": statistics.median(ordered),
        "p95": ordered[max(0, int(len(ordered) * 0.95) - 1)],
    }
    return result


def _round_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    for name in ("formation", "retrieval"):
        result[name] = {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in metrics[name].items()
        }
    result["latency_ms"] = {
        key: round(value, 3) for key, value in metrics["latency_ms"].items()
    }
    return result


def run_benchmark(seed: int = 7) -> dict[str, Any]:
    cases = _fixture(seed)
    started = time.perf_counter()
    baseline = _run_once(cases, candidate=False, seed=seed)
    candidate = _run_once(cases, candidate=True, seed=seed)
    result = {
        "conversations": len(cases),
        "domains": len({case.domain for case in cases}),
        "scenario_coverage": {
            "distractors": sum(not case.expected_memory for case in cases),
            "hedges": sum("think I might" in case.text for case in cases),
            "negations": sum("do not like" in case.text for case in cases),
            "fiction": sum("fictional story" in case.text for case in cases),
            "assistant_suggestions": len(cases),
            "corrections": candidate["auxiliary"]["correction_current_state"],
            "aliases": candidate["auxiliary"]["alias_resolution"],
            "two_clock_events": candidate["auxiliary"]["two_clock_events"],
        },
        "baseline": _round_metrics(baseline),
        "candidate": _round_metrics(candidate),
        "correction_current_state": candidate["auxiliary"]["correction_current_state"],
        "source_preservation": candidate["source_preservation"],
        "alias_resolution": candidate["auxiliary"]["alias_resolution"],
        "two_clock_events": candidate["auxiliary"]["two_clock_events"],
        "unsupported_claim_rate": candidate["unsupported_claim_rate"],
        "qwen_validation": "NOT RUN",
        "ram": "NOT RUN",
        "vram": "NOT RUN",
        "production_db_touched": bool(
            baseline["production_db_touched"] or candidate["production_db_touched"]
        ),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    return result

