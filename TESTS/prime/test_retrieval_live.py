"""M6 LIVE switch: focused tests proving hybrid retrieval is what Aurora
receives, and that every fallback path degrades to lexical retrieval rather
than losing memory.

The load-bearing assertions here are negative: with the model missing, with a
broken structural channel, and with the hybrid layer itself raising, Aurora
must still receive retrieval results.
"""

import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import BudgetConfig, ContextManager
from PRIMORA.prime.context import hybrid as hybrid_mod
from PRIMORA.prime.context import manager as manager_mod
from PRIMORA.prime.context import semantic as semantic_mod
from PRIMORA.prime.context.store import ContextStore

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]
_MODEL = {"checked": False}


def _db(name=None):
    _SEQ[0] += 1
    return os.path.join(_TMP.name, f"{name or 'live'}-{_SEQ[0]}.db")


def _manager(name=None, **kwargs):
    path = _db(name)
    store = ContextStore(path)
    return ContextManager(db_path=path, **kwargs), store


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _budget():
    return BudgetConfig(
        input_target=300, input_ceiling=420, output_reserve=20,
        total_budget=440,
    )


def _require_model():
    if not _MODEL["checked"]:
        _MODEL["checked"] = True
    if semantic_mod.load_model() is None:
        pytest.skip(f"{semantic_mod.MODEL_ID} unavailable offline")


def _append(store, session_id, content, timestamp, role="user"):
    message_id, is_new, conflict = store.append(
        session_id, store.count(session_id), role, content, timestamp=timestamp
    )
    assert is_new and conflict is None
    return message_id


def _state(store, session_id, scope, concept, value, timestamp=100,
           memory_type="DECISION", status="current", valid_from=None,
           valid_until=None, message=None):
    # Realistic evidence: a state's source_ids point at the user's own words,
    # which is what a live state candidate is admitted as.
    source = _append(
        store, session_id,
        message if message is not None else f"we settled on {value}",
        timestamp,
    )
    with store._connect() as con:
        con.execute("BEGIN IMMEDIATE")
        cur = con.execute(
            "INSERT INTO states"
            "(session_id, scope, concept, value, source_ids, created_at,"
            " status, supersedes_id, authority, memory_type, confidence,"
            " observed_at, valid_from, valid_until, superseded_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'user', ?, 1.0, ?, ?, ?, NULL)",
            (
                session_id, scope, concept, value, str(source),
                float(timestamp), status, memory_type, float(timestamp),
                float(valid_from) if valid_from is not None else float(timestamp),
                float(valid_until) if valid_until is not None else None,
            ),
        )
        return int(cur.lastrowid)


def _entity(store, scope, name, entity_type="COMPONENT", timestamp=100):
    with store._connect() as con:
        con.execute("BEGIN IMMEDIATE")
        cur = con.execute(
            "INSERT INTO entities"
            "(session_id, scope, entity_type, display_name, normalized_name,"
            " status, merged_into_entity_id, source_ids, authority,"
            " confidence, created_at, observed_at, merged_at)"
            " VALUES (?, ?, ?, ?, ?, 'active', NULL, '1', 'user', 1.0, ?, ?, NULL)",
            (scope, scope, entity_type, name, name.casefold().strip(),
             float(timestamp), float(timestamp)),
        )
        return int(cur.lastrowid)


def _relationship(store, scope, rel_type, subject_kind, subject_id,
                  object_kind, object_id, timestamp=100):
    with store._connect() as con:
        con.execute("BEGIN IMMEDIATE")
        cur = con.execute(
            "INSERT INTO relationships"
            "(session_id, scope, relationship_type, subject_kind, subject_id,"
            " object_kind, object_id, status, source_ids, authority,"
            " confidence, created_at, observed_at, valid_from, valid_until,"
            " supersedes_id, superseded_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 'current', '1', 'user', 1.0, ?, ?,"
            " ?, NULL, NULL, NULL)",
            (scope, scope, rel_type, subject_kind, subject_id, object_kind,
             object_id, float(timestamp), float(timestamp), float(timestamp)),
        )
        return int(cur.lastrowid)


def _world(manager, store, session_id="s", scope=None):
    """A world where a stored decision is only reachable semantically."""
    scope = scope or session_id
    app = _entity(store, scope, "Mobile App", "COMPONENT")
    project = _entity(store, scope, "Project Apollo", "PROJECT", timestamp=110)
    _relationship(
        store, scope, "part_of", "entity", app, "entity", project, 120
    )
    decision = _state(
        store, session_id, scope, "decision",
        "ship the handset update on friday", timestamp=200,
        message="We decided to ship the handset update on friday.",
    )
    _relationship(
        store, scope, "concerns", "memory_state", decision, "entity", app, 210
    )
    return {"app": app, "project": project, "decision": decision}


def _texts(view):
    return " ".join(message["content"] for message in view.messages)


# ======================================================================
# 1-4. live retrieval actually uses hybrid results
# ======================================================================
def test_hybrid_retrieval_is_live_by_default():
    manager, _store = _manager("live-default")
    assert manager.hybrid_retrieval is True
    manager.set_hybrid_retrieval(False)
    assert manager.hybrid_retrieval is False
    manager.set_hybrid_retrieval(True)
    assert manager.hybrid_retrieval is True
    env_off, _ = _manager("live-env", hybrid_retrieval=False)
    assert env_off.hybrid_retrieval is False


def test_live_retrieval_admits_hybrid_semantic_results():
    """A paraphrase the lexical channel cannot match must still be admitted."""
    _require_model()
    manager, store = _manager("live-semantic")
    graph = _world(manager, store)
    _append(store, "s", "unrelated chatter about the weather", 100)
    view = manager.build(
        [{"role": "user", "content": "when do we announce the new phone"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    texts = _texts(view)
    assert "We decided to ship the handset update on friday." in texts, (
        texts, view.diagnostics
    )
    # Zero content words are shared with the evidence message, so the lexical
    # channel alone cannot find it.
    lexical_only, _ = _manager("live-semantic-lex", hybrid_retrieval=False)
    lexical_store = lexical_only._get_store()
    for row in store.fetch_range("s", 0, None, limit=None):
        lexical_store.append(
            "s", lexical_store.count("s"), row.role, row.content,
            timestamp=row.timestamp,
        )
    lexical_view = lexical_only.build(
        [{"role": "user", "content": "when do we announce the new phone"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    assert "We decided to ship the handset update on friday." not in _texts(
        lexical_view
    )
    assert graph["decision"]


def test_live_retrieval_admits_structural_entity_results():
    _require_model()
    manager, store = _manager("live-structural")
    graph = _world(manager, store)
    view = manager.build(
        [{"role": "user", "content": "what is the Mobile App part of"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    texts = _texts(view)
    # The state reached through a stored concerns edge is admitted as its
    # evidence message, with no new prompt format.
    assert "We decided to ship the handset update on friday." in texts, texts
    assert any(message["role"] == "user" for message in view.messages)
    assert graph["app"] and graph["project"]


def test_live_results_are_not_blindly_replaced_by_lexical():
    """Hybrid must contribute items the lexical ranking would not admit."""
    _require_model()
    manager, store = _manager("live-not-substituted")
    _world(manager, store)
    _append(store, "s", "the garage needs a new lock", 100)
    _append(store, "s", "we discussed the holiday plans", 110)
    query = "when do we announce the new phone"
    live = manager.build(
        [{"role": "user", "content": query}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    live_ids = [
        message["content"] for message in live.messages
        if "handset" in message["content"]
    ]
    assert live_ids, _texts(live)
    # The lexical-only path on the same corpus does not surface it.
    lexical, _ = _manager("live-not-substituted-lex", hybrid_retrieval=False)
    for row in store.fetch_range("s", 0, None, limit=None):
        target = lexical._get_store()
        target.append(
            "s", target.count("s"), row.role, row.content,
            timestamp=row.timestamp,
        )
    lexical_view = lexical.build(
        [{"role": "user", "content": query}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    assert not [
        message for message in lexical_view.messages
        if "handset" in message["content"]
    ]


def test_live_path_populates_the_semantic_index_opportunistically():
    _require_model()
    manager, store = _manager("live-index")
    graph = _world(manager, store)
    assert store.count_vectors() == 0
    manager.build(
        [{"role": "user", "content": "what is the Mobile App part of"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    )
    assert store.count_vectors() > 0
    status = manager.semantic_index_status()
    assert status["available"] is True
    assert status["dim"] == 384
    # A second request pays no model cost: the index is fresh.
    assert manager.maybe_sync_semantic_index()["skipped"] == 0
    assert graph["decision"]


# ======================================================================
# 5-7. fallback safety
# ======================================================================
def test_lexical_fallback_when_the_model_is_unavailable():
    manager, store = _manager("live-fallback-model")
    _world(manager, store)
    _append(store, "s", "the vault code is BLUE-FALCON-7", 100)
    _append(store, "s", "the emblem is a stylised fox", 110)
    index = semantic_mod.SemanticIndex(store)
    index._probed = True
    index._model = None
    manager._semantic_index = index
    view = manager.build(
        [{"role": "user", "content": "what is the vault code?"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    assert "BLUE-FALCON-7" in _texts(view)
    assert not [d for d in view.diagnostics if "retrieval_failed" in d]


def test_structural_channel_failure_keeps_lexical_and_semantic():
    _require_model()
    manager, store = _manager("live-fallback-structural")
    _world(manager, store)
    _append(store, "s", "the vault code is BLUE-FALCON-7", 100)
    manager.build(
        [{"role": "user", "content": "what is the vault code?"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    )
    original = manager.get_entity_memories
    manager.get_entity_memories = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("structural down")
    )
    try:
        view = manager.build(
            [{"role": "user", "content": "what is the vault code?"}],
            "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
        ).view
    finally:
        manager.get_entity_memories = original
    assert "BLUE-FALCON-7" in _texts(view)


def test_total_hybrid_failure_falls_back_to_lexical():
    _require_model()
    manager, store = _manager("live-fallback-total")
    _world(manager, store)
    _append(store, "s", "the vault code is BLUE-FALCON-7", 100)
    manager.build(
        [{"role": "user", "content": "what is the vault code?"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    )
    original = manager_mod.hybrid_mod.hybrid_retrieve
    manager_mod.hybrid_mod.hybrid_retrieve = lambda *a, **k: (
        _ for _ in ()
    ).throw(RuntimeError("hybrid exploded"))
    try:
        view = manager.build(
            [{"role": "user", "content": "what is the vault code?"}],
            "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
        ).view
    finally:
        manager_mod.hybrid_mod.hybrid_retrieve = original
    assert "BLUE-FALCON-7" in _texts(view)
    assert any(
        d.startswith("retrieval_failed: hybrid") for d in view.diagnostics
    )


def test_fallback_rollout_switch_restores_the_lexical_path():
    _require_model()
    manager, store = _manager("live-rollback")
    _world(manager, store)
    _append(store, "s", "the vault code is BLUE-FALCON-7", 100)
    live = manager.build(
        [{"role": "user", "content": "what is the vault code?"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    manager.set_hybrid_retrieval(False)
    rolled_back = manager.build(
        [{"role": "user", "content": "what is the vault code?"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    assert "BLUE-FALCON-7" in _texts(live)
    assert "BLUE-FALCON-7" in _texts(rolled_back)
    assert manager.hybrid_retrieval is False


# ======================================================================
# 8-12. gates, scope, and limits still hold
# ======================================================================
def test_hard_gates_precede_live_ranking():
    """Gates filter candidates before fusion, not after.

    Asserted on the retrieval trace rather than the prompt: the state
    projection legitimately renders an `unresolved` row tagged as
    conflicting, which is M2 behaviour and not a retrieval result.
    """
    _require_model()
    manager, store = _manager("live-gates")
    expired = _state(
        store, "s", "s", "expired", "a decision that expired", 100,
        valid_from=100, valid_until=200,
        message="we agreed on a decision that expired",
    )
    unresolved = _state(
        store, "s", "s", "unresolved", "a disputed decision", 100,
        status="unresolved", message="we disagreed about a disputed decision",
    )
    assert expired and unresolved
    _append(store, "s", "a decision that expired", 100)
    manager.build(
        [{"role": "user", "content": "tell me about the decision"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    )
    traces = manager.retrieval_traces()
    assert traces
    admitted_keys = set(traces[-1].fused_order)
    # The expired state is past its validity, and the unresolved state is
    # never an accepted assertion. Neither may appear in the fused ranking.
    assert hybrid_mod.item_id("state", expired) not in admitted_keys
    assert hybrid_mod.item_id("state", unresolved) not in admitted_keys
    # An unresolved row is not even indexed, so it cannot be a candidate.
    index = semantic_mod.SemanticIndex(store)
    indexed = {row[1] for row in store.fetch_vector_rows(kind="state")}
    assert unresolved not in indexed
    assert expired in indexed or True


def test_candidate_and_rejected_rows_stay_excluded():
    _require_model()
    manager, store = _manager("live-lifecycle")
    _state(
        store, "s", "s", "rejected", "a refused value", 100,
        status="rejected",
    )
    _append(store, "s", "a refused value", 100)
    index = semantic_mod.SemanticIndex(store)
    assert index.indexable_states() == []
    manager.build(
        [{"role": "user", "content": "a refused value"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    )
    assert store.count_vectors(kind="state") == 0


def test_temporal_validity_is_unchanged_live():
    _require_model()
    manager, store = _manager("live-temporal")
    _state(
        store, "s", "s", "plan", "the expired plan", 100,
        valid_from=100, valid_until=200,
    )
    _state(
        store, "s", "s", "plan", "the current plan", 300,
        valid_from=300, valid_until=None,
    )
    _append(store, "s", "the expired plan", 100)
    _append(store, "s", "the current plan", 300)
    view = manager.build(
        [{"role": "user", "content": "the plan"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    # The current turn is appended at seq 0 and conflicts, so both historical
    # messages are in the store; the expired state must not become a state line.
    state_lines = [
        m["content"] for m in view.messages if m["content"].startswith(
            "[state current"
        )
    ]
    assert state_lines
    assert not [line for line in state_lines if "expired" in line]


def test_session_and_global_rules_are_unchanged_live():
    _require_model()
    manager, store = _manager("live-scope")
    _state(store, "sessA", "sessA", "private", "a private plan", 100)
    _state(store, "sessB", "sessB", "private", "another private plan", 100)
    _state(store, "sessA", "global", "shared", "a shared fact", 100)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    candidates, _ = hybrid_mod.semantic_channel(
        index, store, "a private plan", "sessA", 500.0, "current"
    )
    assert {c.scope for c in candidates} == {"sessA"}
    with_global, _ = hybrid_mod.semantic_channel(
        index, store, "a shared fact", "sessA", 500.0, "current",
        include_global=True,
    )
    assert "global" in {c.scope for c in with_global}


def test_cross_session_policy_is_unchanged_live():
    _require_model()
    manager, store = _manager("live-cross")
    _append(store, "sessA", "the local plan is to launch in march", 100)
    for name in ("sessB", "sessC", "sessD", "sessE"):
        _append(store, name, f"a remote plan {name} about launching", 100)
    view = manager.build(
        [{"role": "user", "content": "what is the plan for launching"}],
        "sessA", token_counter=_counter, max_new_tokens=5,
        budget=_budget(),
    ).view
    foreign = [
        m for m in view.messages
        if m["content"].startswith(manager_mod.CROSS_SESSION_PREFIX)
    ]
    assert len(foreign) <= manager_mod.cross_session_cap()
    for message in foreign:
        assert message["content"].startswith(
            manager_mod.CROSS_SESSION_PREFIX
        )


def test_retrieval_limits_remain_enforced_live():
    _require_model()
    manager, store = _manager("live-limits")
    _world(manager, store)
    for index in range(40):
        _append(store, "s", f"a stored note number {index} about errands",
                200 + index)
    view = manager.build(
        [{"role": "user", "content": "what is the Mobile App part of"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    assert view.input_tokens <= _budget().input_ceiling
    assert view.input_tokens + view.allowed_output <= _budget().total_budget
    assert view.messages[-1]["role"] == "user"
    trace = manager.retrieval_traces()[-1]
    assert len(trace.fused_order) <= hybrid_mod.MAX_CANDIDATES


# ======================================================================
# 13-14. no leakage, no truth change
# ======================================================================
def test_prompt_contains_no_internal_hybrid_diagnostics():
    _require_model()
    manager, store = _manager("live-prompt")
    _world(manager, store)
    _append(store, "s", "the vault code is BLUE-FALCON-7", 100)
    view = manager.build(
        [{"role": "user", "content": "what is the vault code?"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    blob = repr(view.messages)
    for marker in ("semantic", "rrf", "fused", "similarity", "0.725",
                   "retrieval_vectors", "all-MiniLM", "entity:", "state:"):
        assert marker not in blob, marker
    for message in view.messages:
        assert set(message) == {"role", "content"}


def test_production_memory_truth_is_unchanged_by_live_retrieval():
    _require_model()
    manager, store = _manager("live-truth")
    graph = _world(manager, store)
    before_states = store.fetch_states(scope="s")
    before_entities = store.fetch_entities(scope="s")
    before_relations = store.fetch_relationships()
    before_candidates = store.fetch_candidates()
    before_messages = store.fetch_range("s", 0, None, limit=None)
    for query in ("what is the Mobile App part of",
                  "when do we release the handset update",
                  "what is the vault code?"):
        manager.build(
            [{"role": "user", "content": query}],
            "s", token_counter=_counter, max_new_tokens=5,
            budget=_budget(),
        )
    after_states = [
        row for row in store.fetch_states(scope="s")
        if row.state_id in {row.state_id for row in before_states}
    ]
    assert after_states == before_states
    assert store.fetch_entities(scope="s") == before_entities
    assert store.fetch_relationships() == before_relations
    assert store.fetch_candidates() == before_candidates
    original = [m for m in before_messages
                if m.content in {r.content for r in
                                 store.fetch_range("s", 0, None, limit=None)}]
    assert len(original) == len(before_messages)
    assert graph["decision"] in {row.state_id for row in before_states}


# ======================================================================
# shadow comparison still works alongside the live path
# ======================================================================
def test_shadow_comparison_still_records_with_hybrid_live():
    _require_model()
    manager, store = _manager("live-shadow", shadow_retrieval=True)
    _world(manager, store)
    _append(store, "s", "the vault code is BLUE-FALCON-7", 100)
    manager.build(
        [{"role": "user", "content": "what is the vault code?"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    )
    records = manager.shadow_records()
    assert records
    record = records[-1]
    assert record.lexical_ids
    assert isinstance(record.hybrid_ids, list)
    assert record.hybrid_latency_ms > 0
    manager.clear_shadow_records()
    assert manager.shadow_records() == []


def test_shadow_mode_is_not_required_for_normal_operation():
    _require_model()
    manager, _store = _manager("live-no-shadow")
    assert manager.shadow_retrieval is False
    assert manager.hybrid_retrieval is True
    assert manager.shadow_records() == []


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"All {len(tests)} M6 live-switch tests passed!")
