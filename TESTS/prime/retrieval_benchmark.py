"""Retrieval benchmark for the CURRENT lexical retrieval system.

This module measures what exists today. It deliberately implements none of the
PROPOSED M6 architecture: no embeddings, no semantic channel, no rank fusion,
no shadow retrieval, no RAG. Nothing here alters retrieval ranking, scoring,
ContextView assembly, or the production database.

Two run modes are measured for every query, because the two questions M6 needs
answered are different:

``channel``
    A read-only mirror of ``ContextManager._retrieve``'s candidate gathering
    and ranking. It calls the same ``ContextStore.search`` and the same pure
    ``scoring.score``, so the returned order *is* the lexical retrieval rank.
    This is the mode used to calibrate a future semantic threshold, an RRF
    constant, and same-session dominance. Drift guards at the top of
    ``_rank_candidates`` fail loudly if production weights change.

``context``
    The real end-to-end path: the full conversation is passed to
    ``ContextManager.build()`` and the metric is set membership of the gold
    memories in the assembled view. ``ContextView`` deliberately re-emits
    retrieved material in chronological order, so retrieval *rank* is not
    observable here; only presence and budget survival are. This mode measures
    how much of what retrieval finds actually survives the budget.

Ground truth is declared in ``fixtures/retrieval_benchmark_corpus.json`` by
corpus key. It is never inferred from the system's own ranking.

Run standalone::

    python TESTS/prime/retrieval_benchmark.py
"""

import gc
import hashlib
import inspect
import json
import os
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from PRIMORA.prime.context import BudgetConfig, ContextManager, scoring
from PRIMORA.prime.context.store import ContextStore

FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures", "retrieval_benchmark_corpus.json",
)
PRODUCTION_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "PRIMORA", "data", "context", "aurora_context.db",
)

# A reduced but real budget profile. `BudgetConfig` is env-overridable in
# production, so a profile is a legitimate configuration. A smaller target
# keeps the fixture fast while still forcing retrieval to compete for room.
BENCHMARK_BUDGET = BudgetConfig(
    input_target=200, input_ceiling=260, output_reserve=20, total_budget=280,
)

RUN_MODES = ("channel", "context")
TOP_K = (1, 3, 5)
PRECISION_AT = 3
CROSS_SESSION_PREFIX = "[from another conversation]\n"
QUERY_TERM_LIMIT = 12
PER_TERM_LIMIT = 10
CROSS_TERM_LIMIT = 20
CROSS_SESSION_CAP = 3
LATENCY_REPEATS = 3


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _load_fixture() -> dict[str, Any]:
    with open(FIXTURE, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ----------------------------------------------------------------------
# Drift guards
# ----------------------------------------------------------------------
def assert_no_drift() -> None:
    """Fail loudly if production retrieval constants the mirror relies on move.

    The ranking weights are read from `scoring.score`'s signature. The query
    limits are hardcoded literals inside `manager.py`, so they are checked by
    reading the module source. This keeps the guard read-only: no production
    module is modified to make it testable.
    """
    params = inspect.signature(scoring.score).parameters
    expected = {
        "w_relevance": 0.45, "w_recency": 0.25, "w_importance": 0.30,
        "w_redundancy": 0.50,
    }
    for name, value in expected.items():
        assert name in params, (
            f"retrieval drift: scoring.score no longer takes {name!r}. "
            f"Update the benchmark mirror deliberately."
        )
        actual = params[name].default
        assert actual == value, (
            f"retrieval drift: scoring.score {name} is {actual}, mirror "
            f"assumes {value}. Update the benchmark mirror deliberately."
        )

    manager_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "PRIMORA", "prime", "context", "manager.py",
    )
    with open(manager_path, "r", encoding="utf-8") as handle:
        source = handle.read()
    required = {
        f"query_terms[:{QUERY_TERM_LIMIT}]": "per-query term cap",
        f"limit={PER_TERM_LIMIT}": "per-term same-session limit",
        f"limit={CROSS_TERM_LIMIT}": "per-term cross-session limit",
        f"cap: int = {CROSS_SESSION_CAP}": "cross-session admission cap",
    }
    for literal, label in required.items():
        assert literal in source, (
            f"retrieval drift: {label} literal {literal!r} is no longer in "
            f"manager.py. Update the benchmark mirror deliberately."
        )


# ----------------------------------------------------------------------
# Corpus construction
# ----------------------------------------------------------------------
@dataclass
class Corpus:
    db_path: str
    key_to_message_id: dict[str, int]
    key_to_session: dict[str, str]
    key_to_text: dict[str, tuple[str, str]]
    state_ids: dict[str, int] = field(default_factory=dict)
    entity_ids: dict[str, int] = field(default_factory=dict)


def build_corpus(db_path: str, fixture: dict[str, Any]) -> Corpus:
    """Materialize the fixture into a fresh database. Deterministic."""
    store = ContextStore(db_path)
    key_to_message_id: dict[str, int] = {}
    key_to_session: dict[str, str] = {}
    key_to_text: dict[str, tuple[str, str]] = {}

    for session_id, entries in fixture["sessions"].items():
        store.ensure_session(session_id)
        for seq, entry in enumerate(entries):
            message_id, is_new, conflict = store.append(
                session_id, seq, entry["role"], entry["text"], timestamp=100,
            )
            assert is_new and conflict is None, (session_id, seq, conflict)
            key_to_message_id[entry["key"]] = message_id
            key_to_session[entry["key"]] = session_id
            key_to_text[entry["key"]] = (entry["role"], entry["text"])

    def _sources(keys):
        return [key_to_message_id[key] for key in keys]

    state_ids: dict[str, int] = {}
    for spec in fixture["states"]:
        sources = _sources(spec["source_keys"])
        with store._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            cur = con.execute(
                "INSERT INTO states"
                "(session_id, scope, concept, value, source_ids, created_at,"
                " status, supersedes_id, authority, memory_type, confidence,"
                " observed_at, valid_from, valid_until, superseded_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    spec["session"], spec["scope"], spec["concept"],
                    spec["value"], ",".join(str(s) for s in sources),
                    float(spec["timestamp"]), spec["status"], None, "user",
                    spec["memory_type"], 1.0, float(spec["timestamp"]),
                    float(spec["valid_from"]) if spec["valid_from"] is not None
                    else None,
                    float(spec["valid_until"]) if spec["valid_until"] is not None
                    else None,
                    float(spec["valid_until"]) if spec["valid_until"] is not None
                    else None,
                ),
            )
            state_ids[spec["key"]] = int(cur.lastrowid)

    entity_ids: dict[str, int] = {}
    for spec in fixture["entities"]:
        sources = _sources(spec["source_keys"])
        with store._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            cur = con.execute(
                "INSERT INTO entities"
                "(session_id, scope, entity_type, display_name,"
                " normalized_name, status, merged_into_entity_id, source_ids,"
                " authority, confidence, created_at, observed_at, merged_at)"
                " VALUES (?, ?, ?, ?, ?, 'active', NULL, ?, ?, 1.0, ?, ?, NULL)",
                (
                    "personal", "personal", spec["entity_type"],
                    spec["display_name"],
                    spec["display_name"].casefold().strip(),
                    ",".join(str(s) for s in sources), "user", 100.0, 100.0,
                ),
            )
            entity_ids[spec["key"]] = int(cur.lastrowid)

    for spec in fixture.get("aliases", []):
        with store._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "INSERT INTO entity_aliases"
                "(session_id, scope, entity_id, alias_text, normalized_alias,"
                " status, source_ids, authority, confidence, created_at,"
                " observed_at, valid_from, valid_until, superseded_at)"
                " VALUES ('personal', 'personal', ?, ?, ?, 'current', ?,"
                " 'user', 1.0, 100.0, 100.0, 100.0, NULL, NULL)",
                (
                    entity_ids[spec["entity_key"]], spec["alias_text"],
                    spec["alias_text"].casefold().strip(),
                    ",".join(str(s) for s in _sources(spec["source_keys"])),
                ),
            )

    for spec in fixture.get("relationships", []):
        if "subject_state" in spec:
            subject_kind = "memory_state"
            subject_id = state_ids[spec["subject_state"]]
            object_kind, object_id = "entity", entity_ids[spec["object"]]
        else:
            subject_kind, subject_id = "entity", entity_ids[spec["subject"]]
            object_kind, object_id = "entity", entity_ids[spec["object"]]
        with store._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "INSERT INTO relationships"
                "(session_id, scope, relationship_type, subject_kind,"
                " subject_id, object_kind, object_id, status, source_ids,"
                " authority, confidence, created_at, observed_at, valid_from,"
                " valid_until, supersedes_id, superseded_at)"
                " VALUES ('personal', 'personal', ?, ?, ?, ?, ?, 'current', ?,"
                " 'user', 1.0, 100.0, 100.0, 100.0, NULL, NULL, NULL)",
                (
                    spec["type"], subject_kind, subject_id, object_kind,
                    object_id,
                    ",".join(str(s) for s in _sources(spec["source_keys"])),
                ),
            )

    # Fold the write-ahead log into the main file so the template can be
    # copied byte-for-byte per run.
    with store._connect() as con:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    corpus = Corpus(
        db_path=db_path,
        key_to_message_id=key_to_message_id,
        key_to_session=key_to_session,
        key_to_text=key_to_text,
        state_ids=state_ids,
        entity_ids=entity_ids,
    )
    _validate_fixture(fixture, corpus)
    return corpus


def _validate_fixture(fixture: dict[str, Any], corpus: Corpus) -> None:
    keys = set(corpus.key_to_message_id)
    assert len(keys) == len(corpus.key_to_message_id), "duplicate corpus key"
    texts = list(corpus.key_to_text.values())
    assert len(set(texts)) == len(texts), "duplicate (role, text) in corpus"
    for query in fixture["queries"]:
        unknown = [k for k in query["gold"] if k not in keys]
        assert not unknown, (query["id"], unknown)
        distractor = query.get("expect_distractor")
        if distractor:
            assert distractor in keys, (query["id"], distractor)
    # Ground truth must be able to express "one", "many", and "none".
    assert any(len(q["gold"]) == 1 for q in fixture["queries"])
    assert any(not q["gold"] for q in fixture["queries"])


# ----------------------------------------------------------------------
# Mode 1: lexical channel mirror (true retrieval rank)
# ----------------------------------------------------------------------
def _history_requested(text: str) -> bool:
    from PRIMORA.prime.context.manager import HISTORY_MARKERS
    lowered = (text or "").lower()
    return any(marker in lowered for marker in HISTORY_MARKERS)


def _superseded_source_ids(store, scopes) -> set:
    from PRIMORA.prime.context.manager import ContextManager
    return ContextManager._superseded_source_ids(store, scopes)


def _rank_candidates(store, session_id: str, query_text: str):
    """Mirror of ContextManager._retrieve candidate gathering and ranking.

    Deviation, documented deliberately: `selected_texts` holds only the
    current turn, whereas production also holds the recency window by this
    point. This slightly under-penalizes redundancy relative to production
    and never changes candidate membership.
    """
    query_terms = scoring.build_query(query_text, "")
    if not query_terms:
        return [], [], query_terms, []
    latest = store.fetch_latest(session_id, 200)
    now_seq = max((m.seq for m in latest), default=0)
    stale: set = set()
    if not _history_requested(query_text):
        stale = _superseded_source_ids(store, [session_id, "global"])

    selected_texts = [query_text]
    candidates = []
    seen: set = set()
    diagnostics: list[str] = []
    for term in query_terms[:QUERY_TERM_LIMIT]:
        if not term or not term.strip():
            diagnostics.append(f"empty term skipped: {term!r}")
            continue
        try:
            hits = store.search(session_id, term, limit=PER_TERM_LIMIT)
        except Exception as exc:
            diagnostics.append(f"search failed for {term!r}: {exc}")
            continue
        for hit in hits:
            if hit.message_id in seen:
                continue
            seen.add(hit.message_id)
            if hit.message_id in stale:
                continue
            candidates.append(hit)

    scored = []
    for hit in candidates:
        breakdown = scoring.score(
            text=hit.content, role=hit.role, seq=hit.seq, now_seq=now_seq,
            query_terms=query_terms, selected_texts=selected_texts,
            stored_importance=hit.importance,
        )
        scored.append((breakdown["score"], hit.seq, hit, breakdown))
    scored.sort(key=lambda row: (-row[0], row[1]))

    foreign = []
    if not _history_requested(query_text):
        foreign_scored = []
        fseen: set = set()
        for term in query_terms[:QUERY_TERM_LIMIT]:
            try:
                hits = store.search(None, term, limit=CROSS_TERM_LIMIT)
            except Exception as exc:
                diagnostics.append(f"cross search failed {term!r}: {exc}")
                continue
            for hit in hits:
                if hit.session_id == session_id or hit.message_id in fseen:
                    continue
                fseen.add(hit.message_id)
                if hit.message_id in stale:
                    continue
                foreign_scored.append((hit, term))
        session_max: dict[str, int] = {}
        rows = []
        for hit, _ in foreign_scored:
            if hit.session_id not in session_max:
                latest_f = store.fetch_latest(hit.session_id, 1)
                session_max[hit.session_id] = (
                    latest_f[0].seq if latest_f else hit.seq
                )
            breakdown = scoring.score(
                text=hit.content, role=hit.role, seq=hit.seq,
                now_seq=session_max[hit.session_id], query_terms=query_terms,
                selected_texts=selected_texts,
                stored_importance=hit.importance,
            )
            rows.append((breakdown["score"], hit.session_id, hit.seq, hit,
                         breakdown))
        rows.sort(key=lambda row: (-row[0], row[1], row[2]))
        foreign = rows[:CROSS_SESSION_CAP]
    return scored, foreign, query_terms, diagnostics


# ----------------------------------------------------------------------
# Mode 2: end-to-end membership
# ----------------------------------------------------------------------
def _session_keys(corpus: Corpus, session_id: str) -> list[str]:
    return [k for k, v in corpus.key_to_session.items() if v == session_id]


# ----------------------------------------------------------------------
# Observation
# ----------------------------------------------------------------------
@dataclass
class Observation:
    query_id: str
    category: str
    match: str
    mode: str
    session_id: str
    scope: str
    query: str
    expected_keys: list[str]
    expected_ids: list[int]
    returned_keys: list[str]
    returned_ids: list[int]
    ranks: dict[str, int]
    scores: dict[str, float]
    query_terms: list[str]
    synthesized_lines: list[str]
    state_lines: list[str]
    foreign_keys: list[str]
    latency_ms: float
    input_tokens: int
    diagnostics: list[str]
    error: Optional[str] = None

    @property
    def is_no_match(self) -> bool:
        return not self.expected_keys


def _run_channel(corpus: Corpus, db_path: str, query) -> Observation:
    store = ContextStore(db_path)
    session_id = query["session"]
    best = None
    for _ in range(LATENCY_REPEATS):
        started = time.perf_counter()
        result = _rank_candidates(store, session_id, query["text"])
        elapsed = (time.perf_counter() - started) * 1000.0
        if best is None or elapsed < best[0]:
            best = (elapsed, result)
    latency_ms, (scored, foreign, terms, diagnostics) = best

    id_to_key = {
        value: key for key, value in corpus.key_to_message_id.items()
    }
    # Production admission order: same-session hits first, then up to
    # CROSS_SESSION_CAP foreign hits appended after them.
    returned_keys: list[str] = []
    scores: dict[str, float] = {}
    for _score, _seq, hit, _breakdown in scored:
        key = id_to_key.get(hit.message_id)
        if key is not None and key not in returned_keys:
            returned_keys.append(key)
            scores[key] = round(_score, 4)
    foreign_keys: list[str] = []
    for _score, _sid, _seq, hit, _bd in foreign:
        key = id_to_key.get(hit.message_id)
        if key is not None and key not in foreign_keys:
            foreign_keys.append(key)
            scores[key] = round(_score, 4)
            if key not in returned_keys:
                returned_keys.append(key)

    return Observation(
        query_id=query["id"], category=query["category"], match=query["match"],
        mode="channel", session_id=session_id, scope=session_id,
        query=query["text"], expected_keys=list(query["gold"]),
        expected_ids=[corpus.key_to_message_id[k] for k in query["gold"]],
        returned_keys=returned_keys,
        returned_ids=[corpus.key_to_message_id[k] for k in returned_keys],
        ranks={k: i + 1 for i, k in enumerate(returned_keys)},
        scores=scores, query_terms=terms, synthesized_lines=[],
        state_lines=[], foreign_keys=foreign_keys, latency_ms=latency_ms,
        input_tokens=0, diagnostics=diagnostics,
    )


def _run_context(corpus: Corpus, db_path: str, query) -> Observation:
    session_id = query["session"]
    snapshot = [
        {"role": role, "content": text, "timestamp": 100}
        for role, text in [
            corpus.key_to_text[key] for key in _session_keys(corpus, session_id)
        ]
    ]
    snapshot.append(
        {"role": "user", "content": query["text"], "timestamp": 900}
    )
    manager = ContextManager(db_path=db_path)
    started = time.perf_counter()
    result = manager.build(
        list(snapshot), session_id, token_counter=_counter,
        max_new_tokens=20, budget=BENCHMARK_BUDGET,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    view = result.view

    text_to_key = {v: k for k, v in corpus.key_to_text.items()}
    returned_keys: list[str] = []
    synthesized: list[str] = []
    state_lines: list[str] = []
    foreign: list[str] = []
    for message in view.messages:
        content, role = message.get("content", ""), message.get("role", "")
        key = text_to_key.get((role, content))
        if key is None and content.startswith(CROSS_SESSION_PREFIX):
            key = text_to_key.get(
                (role, content[len(CROSS_SESSION_PREFIX):])
            )
            if key is not None:
                foreign.append(key)
        if key is not None:
            if key not in returned_keys:
                returned_keys.append(key)
            continue
        synthesized.append(content)
        if content.startswith("[state "):
            state_lines.append(content)

    return Observation(
        query_id=query["id"], category=query["category"], match=query["match"],
        mode="context", session_id=session_id, scope=session_id,
        query=query["text"], expected_keys=list(query["gold"]),
        expected_ids=[corpus.key_to_message_id[k] for k in query["gold"]],
        returned_keys=returned_keys,
        returned_ids=[corpus.key_to_message_id[k] for k in returned_keys],
        ranks={k: i + 1 for i, k in enumerate(returned_keys)},
        scores={}, query_terms=[], synthesized_lines=synthesized,
        state_lines=state_lines, foreign_keys=foreign, latency_ms=latency_ms,
        input_tokens=view.input_tokens, diagnostics=list(view.diagnostics),
    )


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def _recall(expected, returned, k: int) -> float:
    if not expected:
        return 0.0
    return len(set(expected) & set(returned[:k])) / len(set(expected))


def _coverage(expected, returned) -> float:
    if not expected:
        return 0.0
    return len(set(expected) & set(returned)) / len(set(expected))


def _percentile(values, fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1,
                       int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def metrics(observations: list[Observation]) -> dict[str, Any]:
    scored = [o for o in observations if not o.is_no_match]
    no_match = [o for o in observations if o.is_no_match]
    errors = [o for o in observations if o.error]
    ranked = observations and all(
        o.mode == "channel" for o in observations
    )

    out: dict[str, Any] = {
        "query_count": len(observations),
        "relevant_result_count": sum(len(o.expected_ids) for o in observations),
        "failed_count": len(errors),
        "metric_kind": "ranked" if ranked else "membership",
    }
    if ranked:
        for k in TOP_K:
            out[f"recall_at_{k}"] = round(statistics.fmean(
                [_recall(o.expected_keys, o.returned_keys, k) for o in scored]
            ), 4) if scored else 0.0
        out["recall_at_all"] = round(statistics.fmean(
            [_coverage(o.expected_keys, o.returned_keys) for o in scored]
        ), 4) if scored else 0.0
        out[f"precision_at_{PRECISION_AT}"] = round(statistics.fmean(
            [_recall(o.expected_keys, o.returned_keys, PRECISION_AT)
             for o in scored]
        ), 4) if scored else 0.0
        fp3 = sum(
            1 for o in scored for k in o.returned_keys[:PRECISION_AT]
            if k not in o.expected_keys
        )
        ret3 = sum(len(o.returned_keys[:PRECISION_AT]) for o in scored)
        out["false_positive_rate_at_3"] = round(
            fp3 / ret3, 4) if ret3 else 0.0
        out["false_positive_rate_all"] = round(sum(
            1 for o in scored for k in o.returned_keys
            if k not in o.expected_keys
        ) / sum(len(o.returned_keys) for o in scored), 4) if scored else 0.0
        out["returned_items_total"] = sum(len(o.returned_keys) for o in scored)
        # Same-session dominance, measured rather than assumed.
        out["cross_gold_in_top3"] = sum(
            1 for o in observations
            if o.category == "cross_session"
            and set(o.expected_keys) & set(o.returned_keys[:3])
        )
        out["cross_gold_returned_at_all"] = sum(
            1 for o in observations
            if o.category == "cross_session"
            and set(o.expected_keys) & set(o.returned_keys)
        )
    else:
        out["coverage"] = round(statistics.fmean(
            [_coverage(o.expected_keys, o.returned_keys) for o in scored]
        ), 4) if scored else 0.0
        out["coverage_at_1"] = round(statistics.fmean(
            [_recall(o.expected_keys, o.returned_keys, 1) for o in scored]
        ), 4) if scored else 0.0
        out["note"] = (
            "membership only. ContextView re-emits retrieved material in "
            "chronological order, so retrieval rank, precision, and "
            "no-match silence are NOT observable here; use the channel mode. "
            "The recency window always contributes messages, which is why "
            "no-match accuracy is not reported in this mode."
        )

    out["no_match_accuracy"] = round(
        sum(1 for o in no_match if not o.returned_keys) / len(no_match), 4
    ) if (no_match and ranked) else None
    out["no_match_count"] = len(no_match)

    for category in sorted({o.category for o in observations}):
        group = [o for o in observations if o.category == category]
        group_scored = [o for o in group if not o.is_no_match]
        if ranked:
            out[f"recall_at_3__{category}"] = round(statistics.fmean(
                [_recall(o.expected_keys, o.returned_keys, 3)
                 for o in group_scored]
            ), 4) if group_scored else None
        else:
            out[f"coverage__{category}"] = round(statistics.fmean(
                [_coverage(o.expected_keys, o.returned_keys)
                 for o in group_scored]
            ), 4) if group_scored else None
        out[f"count__{category}"] = len(group)

    punctuation = [o for o in observations if o.category == "punctuation"]
    out["punctuation_success"] = round(
        sum(1 for o in punctuation if o.returned_keys) / len(punctuation), 4
    ) if punctuation else 0.0
    if ranked:
        out["punctuation_success_at_3"] = round(
            sum(1 for o in punctuation
                if set(o.expected_keys) & set(o.returned_keys[:3]))
            / len(punctuation), 4
        ) if punctuation else 0.0

    controls = [o for o in observations
                if o.category == "false_positive_control"]
    out["fp_control_count"] = len(controls)
    out["fp_control_distractor_admitted"] = sum(
        1 for o in controls
        if any("lunch" in k for k in o.returned_keys)
    )
    if ranked:
        out["fp_control_distractor_at_3"] = sum(
            1 for o in controls
            if any("lunch" in k for k in o.returned_keys[:3])
        )
        out["fp_control_gold_at_3"] = sum(
            1 for o in controls
            if set(o.expected_keys) & set(o.returned_keys[:3])
        )

    latencies = [o.latency_ms for o in observations if not o.error]
    out["latency_ms"] = {
        "count": len(latencies),
        "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        "median": round(statistics.median(latencies), 3) if latencies else 0.0,
        "p95": round(_percentile(latencies, 0.95), 3) if latencies else 0.0,
        "min": round(min(latencies), 3) if latencies else 0.0,
        "max": round(max(latencies), 3) if latencies else 0.0,
    }
    if not ranked:
        out["mean_input_tokens"] = round(statistics.fmean(
            [o.input_tokens for o in observations]), 2) if observations else 0.0
    out["skipped_term_total"] = sum(
        sum(1 for d in o.diagnostics if d.startswith("retrieval_skipped_term"))
        for o in observations
    )
    out["search_failure_total"] = sum(
        sum(1 for d in o.diagnostics if d.startswith("search failed"))
        for o in observations
    )
    return out


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
@dataclass
class BenchmarkReport:
    fixture: str
    modes: dict[str, dict[str, Any]]
    observations: list[Observation]
    production_db_sha256: Optional[str]
    production_db_untouched: bool


def _sha256(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(tmpdir: Optional[str] = None) -> BenchmarkReport:
    """Execute the benchmark. Never writes to the production database."""
    assert_no_drift()
    owned = tmpdir is None
    workdir = tmpdir or tempfile.mkdtemp(prefix="primora-retrieval-bench-")
    try:
        before = _sha256(PRODUCTION_DB)
        fixture = _load_fixture()
        template = os.path.join(workdir, "template.db")
        corpus = build_corpus(template, fixture)
        runners = {"channel": _run_channel, "context": _run_context}

        observations: list[Observation] = []
        for mode in RUN_MODES:
            for query in fixture["queries"]:
                run_db = os.path.join(workdir, f"run-{mode}-{query['id']}.db")
                shutil.copyfile(template, run_db)
                try:
                    observations.append(runners[mode](corpus, run_db, query))
                except Exception as exc:  # recorded, never raised
                    observations.append(Observation(
                        query_id=query["id"], category=query["category"],
                        match=query["match"], mode=mode,
                        session_id=query["session"], scope=query["session"],
                        query=query["text"], expected_keys=list(query["gold"]),
                        expected_ids=[], returned_keys=[], returned_ids=[],
                        ranks={}, scores={}, query_terms=[],
                        synthesized_lines=[], state_lines=[],
                        foreign_keys=[], latency_ms=0.0, input_tokens=0,
                        diagnostics=[],
                        error=f"{type(exc).__name__}: {exc}",
                    ))

        after = _sha256(PRODUCTION_DB)
        return BenchmarkReport(
            fixture=fixture["name"],
            modes={
                mode: metrics([o for o in observations if o.mode == mode])
                for mode in RUN_MODES
            },
            observations=observations,
            production_db_sha256=before,
            production_db_untouched=(before == after),
        )
    finally:
        if owned:
            gc.collect()
            shutil.rmtree(workdir, ignore_errors=True)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def format_report(report: BenchmarkReport) -> str:
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add(f"PRIMORA retrieval benchmark - fixture: {report.fixture}")
    add("CURRENT lexical retrieval only. No embeddings, no RRF, no hybrid.")
    add("=" * 78)
    add(f"production DB unchanged : {report.production_db_untouched}")
    if report.production_db_sha256:
        add(f"production DB sha256   : {report.production_db_sha256}")
    titles = {
        "channel": "lexical channel mirror (true retrieval rank)",
        "context": "end-to-end build() (membership; rank not observable)",
    }
    for mode in RUN_MODES:
        data = report.modes[mode]
        add("")
        add("-" * 78)
        add(f"MODE {mode}: {titles[mode]}")
        add("-" * 78)
        for key, value in data.items():
            if key.startswith("recall_at_3__") or key.startswith("coverage__"):
                continue
            if key == "latency_ms":
                continue
            if isinstance(value, dict):
                continue
            add(f"  {key:36} {value}")
        latency = data["latency_ms"]
        add(f"  {'latency_ms.mean':36} {latency['mean']}")
        add(f"  {'latency_ms.median':36} {latency['median']}")
        add(f"  {'latency_ms.p95':36} {latency['p95']}")
        add(f"  {'latency_ms.min/max':36} {latency['min']} / {latency['max']}")
        add("  per-category:")
        for key in sorted(data):
            if key.startswith("recall_at_3__") or key.startswith("coverage__"):
                category = key.split("__", 1)[1]
                add(f"    {category:30} {data[key]}  (n={data['count__' + category]})")
    return "\n".join(lines)


def format_trace(report: BenchmarkReport, limit: int = 60) -> str:
    lines = ["", "=" * 78, "PER-QUERY TRACE", "=" * 78]
    for mode in RUN_MODES:
        lines.append(f"\nMODE {mode}")
        for obs in report.observations:
            if obs.mode != mode:
                continue
            if obs.error:
                lines.append(f"  {obs.query_id} {obs.category:24} ERROR {obs.error}")
                continue
            gold = obs.expected_keys or []
            got = obs.returned_keys[:5]
            if not got and not gold:
                continue
            rank3 = _recall(obs.expected_keys, obs.returned_keys, 3)
            lines.append(
                f"  {obs.query_id} {obs.category:24} gold={gold} "
                f"top5={got} k3={rank3:.2f} {obs.latency_ms:.0f}ms"
            )
    body = lines[4:]
    if len(body) > limit:
        body = body[:limit] + ["...truncated..."]
    return "\n".join(lines[:4] + body)


def main() -> int:
    report = run()
    print(format_report(report))
    print(format_trace(report))
    return 0 if report.production_db_untouched else 1


if __name__ == "__main__":
    raise SystemExit(main())
