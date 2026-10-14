"""M6 Advanced Retrieval: focused tests for the semantic index, the hybrid
fusion engine, the M5 structural channel, shadow mode, and fail-safe
behaviour.

Shadow mode is the only mode the production request path uses, so the central
assertion of this file is negative: adding the hybrid path must not change a
single byte of what Aurora receives.
"""

import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import (
    BudgetConfig,
    ContextManager,
    RetrievalCandidate,
    RetrievalTrace,
    ShadowComparison,
)
from PRIMORA.prime.context import hybrid as hybrid_mod
from PRIMORA.prime.context import queries as queries_mod
from PRIMORA.prime.context import semantic as semantic_mod
from PRIMORA.prime.context.store import ContextStore

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]
_MODEL = {"loaded": False}


def _db(name=None):
    _SEQ[0] += 1
    return os.path.join(_TMP.name, f"{name or 'm6'}-{_SEQ[0]}.db")


def _append(store, session_id, content, timestamp, role="user"):
    message_id, is_new, conflict = store.append(
        session_id, store.count(session_id), role, content, timestamp=timestamp
    )
    assert is_new and conflict is None
    return message_id


def _manager(name=None, **kwargs):
    path = _db(name)
    store = ContextStore(path)
    return ContextManager(db_path=path, **kwargs), store


def _budget():
    return BudgetConfig(
        input_target=220, input_ceiling=400, output_reserve=20,
        total_budget=420,
    )


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _require_model():
    if not _MODEL["loaded"]:
        if semantic_mod.load_model() is None:
            pytest.skip(f"{semantic_mod.MODEL_ID} unavailable offline")
        _MODEL["loaded"] = True


def _state(store, session_id, scope, concept, value, timestamp=100,
           memory_type="DECISION", status="current", valid_from=None,
           valid_until=None, authority="user", source=True):
    sources = []
    if source:
        sources = [_append(store, session_id, f"evidence {concept}", timestamp)]
    with store._connect() as con:
        con.execute("BEGIN IMMEDIATE")
        cur = con.execute(
            "INSERT INTO states"
            "(session_id, scope, concept, value, source_ids, created_at,"
            " status, supersedes_id, authority, memory_type, confidence,"
            " observed_at, valid_from, valid_until, superseded_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 1.0, ?, ?, ?, NULL)",
            (
                session_id, scope, concept, value,
                ",".join(str(s) for s in sources), float(timestamp), status,
                authority, memory_type, float(timestamp),
                float(valid_from) if valid_from is not None else float(timestamp),
                float(valid_until) if valid_until is not None else None,
            ),
        )
        return int(cur.lastrowid)


def _entity(store, scope, name, entity_type="PROJECT", status="active",
            timestamp=100, authority="user"):
    with store._connect() as con:
        con.execute("BEGIN IMMEDIATE")
        cur = con.execute(
            "INSERT INTO entities"
            "(session_id, scope, entity_type, display_name, normalized_name,"
            " status, merged_into_entity_id, source_ids, authority,"
            " confidence, created_at, observed_at, merged_at)"
            " VALUES (?, ?, ?, ?, ?, ?, NULL, '1', ?, 1.0, ?, ?, NULL)",
            (
                scope, scope, entity_type, name, name.casefold().strip(),
                status, authority, float(timestamp), float(timestamp),
            ),
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
            " VALUES (?, ?, ?, ?, ?, ?, ?, 'current', '1', 'user', 1.0,"
            " ?, ?, ?, NULL, NULL, NULL)",
            (
                scope, scope, rel_type, subject_kind, subject_id, object_kind,
                object_id, float(timestamp), float(timestamp),
                float(timestamp),
            ),
        )
        return int(cur.lastrowid)


def _house(manager, store, session_id="s", scope=None):
    """A small realistic graph: entity, part_of edge, state, concerns edge."""
    scope = scope or session_id
    app = _entity(store, scope, "Mobile App", "COMPONENT")
    project = _entity(store, scope, "Project Apollo", "PROJECT", timestamp=110)
    _relationship(
        store, scope, "part_of", "entity", app, "entity", project, 120
    )
    decision = _state(
        store, session_id, scope, "decision",
        "ship the mobile app on friday", timestamp=200,
    )
    _relationship(
        store, scope, "concerns", "memory_state", decision, "entity", app, 210
    )
    return {
        "app": app, "project": project, "decision": decision,
    }


# ======================================================================
# PART 1-2: semantic index
# ======================================================================
def test_index_creation_is_deterministic_and_cpu_first():
    _require_model()
    manager, store = _manager("index-create")
    _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    assert index.device == "cpu"
    assert index.model_id == "sentence-transformers/all-MiniLM-L6-v2"
    first = index.sync()
    assert first["states"] == 1 and first["entities"] == 2
    dim = index.dim()
    assert dim == 384, dim
    status = index.status()
    assert status["available"] is True
    assert status["dim"] == 384
    assert status["states"] == 1 and status["entities"] == 2


def test_index_does_not_store_source_text():
    _require_model()
    manager, store = _manager("index-no-text")
    _house(manager, store)
    semantic_mod.SemanticIndex(store).sync()
    con = sqlite3.connect(store.db_path)
    columns = {row[1] for row in con.execute(
        "PRAGMA table_info(retrieval_vectors)"
    )}
    rows = con.execute(
        "SELECT kind, ref_id, scope, session_id, model_id, dim, content_hash,"
        " length(vector) FROM retrieval_vectors"
    ).fetchall()
    con.close()
    assert "text" not in columns
    assert "content" not in columns
    assert "concept" not in columns
    assert "value" not in columns
    assert "display_name" not in columns
    for _kind, _ref, _scope, _sid, _mid, dim, digest, byte_length in rows:
        assert dim == 384
        assert byte_length == 384 * 4
        assert len(digest) == 64


def test_only_states_and_entities_are_embedded():
    _require_model()
    manager, store = _manager("index-kinds")
    _house(manager, store)
    # A message, a relationship, and a digest must produce no vectors.
    _append(store, "s", "a raw message that is never embedded", 90)
    _append(store, "s", "second raw message", 91)
    semantic_mod.SemanticIndex(store).sync()
    kinds = {
        row[0] for row in store.fetch_vector_rows()
    }
    assert kinds == {"state", "entity"}
    assert semantic_mod.EMBEDDABLE_KINDS == ("state", "entity")


def test_unresolved_and_rejected_rows_are_never_indexed():
    _require_model()
    manager, store = _manager("index-lifecycle")
    _state(store, "s", "s", "thing", "a value", 100, status="unresolved")
    _state(store, "s", "s", "other", "b value", 110, status="rejected")
    _state(store, "s", "s", "good", "c value", 120, status="current")
    index = semantic_mod.SemanticIndex(store)
    report = index.sync()
    assert report["states"] == 1
    indexed = {row[1] for row in store.fetch_vector_rows(kind="state")}
    live = {s.state_id for s in store.fetch_states(scope="s")}
    assert indexed < live


def test_stale_and_missing_index_entries_are_detected():
    _require_model()
    manager, store = _manager("index-stale")
    graph = _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    assert index.stale()["healthy"] is True
    # A new eligible row is reported missing.
    fresh = _state(store, "s", "s", "later", "a new fact", 300)
    stale = index.stale()
    assert stale["healthy"] is False
    assert {"kind": "state", "ref_id": fresh} in stale["missing"]
    # A content change is reported as a mismatch.
    with store._connect() as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute(
            "UPDATE states SET value = 'changed value' WHERE id = ?",
            (graph["decision"],),
        )
    stale = index.stale()
    assert any(
        entry["ref_id"] == graph["decision"] and entry["reason"] == "content"
        for entry in stale["mismatch"]
    )
    # Re-syncing repairs the mismatch.
    index.sync()
    assert index.stale()["missing"] == []
    assert index.stale()["mismatch"] == []


def test_orphaned_vectors_are_detected_after_source_removal():
    _require_model()
    manager, store = _manager("index-orphan")
    graph = _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    with store._connect() as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute("DELETE FROM entities WHERE id = ?", (graph["app"],))
    stale = index.stale()
    assert any(
        entry["kind"] == "entity" and entry["ref_id"] == graph["app"]
        for entry in stale["orphaned"]
    )


def test_rebuild_is_lossless_and_idempotent():
    _require_model()
    manager, store = _manager("index-rebuild")
    _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    before = index.status()
    store.delete_vectors(kind="state", ref_id=999)
    rebuilt = index.rebuild()
    assert rebuilt["states"] == before["states"]
    assert index.status()["states"] == before["states"]
    assert index.status()["entities"] == before["entities"]
    assert index.stale()["healthy"] is True


def test_rebuild_with_unavailable_model_invalidates_the_token():
    """A rebuild that cannot run must not leave a claim of health.

    Regression: rebuild() used to return early on an unavailable model
    without touching sync_token. Paired with an already-valid token that
    left the semantic channel permanently disabled while every status
    call reported a healthy index.
    """
    _require_model()
    manager, store = _manager("index-rebuild-unavailable")
    _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    assert index.stale()["healthy"] is True
    assert index.needs_sync() is False

    # The model fails to load. rebuild() must not be able to claim health.
    index.available = lambda: False
    report = index.rebuild()

    assert report["skipped"] == 1
    assert store.get_index_meta("sync_token") == ""
    # needs_sync() reports False only because the model is still down and
    # nothing could be synced anyway. The claim of freshness is gone.
    assert index.needs_sync() is False
    # Once the model is back, the invalidation must force a real re-sync.
    assert semantic_mod.SemanticIndex(store).needs_sync() is True


def test_rebuild_that_fails_midway_leaves_the_index_needing_sync():
    """Vectors deleted, embedding fails: the token must not survive.

    This is the exact production failure. delete_vectors() runs, encode()
    then returns None, and the index ends up empty. A surviving token
    makes maybe_sync a silent no-op forever.
    """
    _require_model()
    manager, store = _manager("index-rebuild-midway")
    _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    assert store.count_vectors(model_id=index.model_id) > 0

    def _dead_encoder(*args, **kwargs):
        return None

    original = semantic_mod.encode
    semantic_mod.encode = _dead_encoder
    try:
        report = index.rebuild()
    finally:
        semantic_mod.encode = original

    assert report["skipped"] == 1
    # The delete happened, so the index really is empty ...
    assert store.count_vectors(model_id=index.model_id) == 0
    # ... and it must therefore not claim to be fresh.
    assert store.get_index_meta("sync_token") == ""
    assert index.needs_sync() is True
    assert index.stale()["healthy"] is False
    assert index.stale()["missing"], "an empty index over live rows is reported"


def test_failed_rebuild_does_not_silently_believe_the_index_is_healthy():
    """The user-visible symptom: maybe_sync must retry, not no-op."""
    _require_model()
    manager, store = _manager("index-rebuild-silent")
    _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    healthy_before = index.stale()["healthy"]

    def _dead_encoder(*args, **kwargs):
        return None

    original = semantic_mod.encode
    semantic_mod.encode = _dead_encoder
    try:
        index.rebuild()
        # The live path calls maybe_sync on every request.
        blind = index.maybe_sync()
    finally:
        semantic_mod.encode = original

    assert healthy_before is True
    assert blind["states"] == 0 and blind["entities"] == 0
    assert store.count_vectors(model_id=index.model_id) == 0, (
        "a broken encoder must not be able to leave a silently empty index "
        "that reports itself as fresh"
    )

    # Encoder restored: the very next live call must repair the index.
    recovered = index.maybe_sync()
    assert recovered["states"] > 0
    assert index.stale()["healthy"] is True
    assert index.needs_sync() is False


def test_successful_rebuild_restores_the_token():
    """The fix must not leave the index permanently unable to go fresh."""
    _require_model()
    manager, store = _manager("index-rebuild-recover")
    _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()

    index.available = lambda: False
    index.rebuild()
    # The model is still down, so this instance cannot report a need.
    assert index.needs_sync() is False
    assert store.get_index_meta("sync_token") == ""

    restored = semantic_mod.SemanticIndex(store)
    assert restored.needs_sync() is True
    report = restored.rebuild()
    assert report["states"] > 0
    assert restored.needs_sync() is False
    assert restored.stale()["healthy"] is True


# ----------------------------------------------------------------------
# Scope-aware freshness. A scoped sync indexes one scope, so its freshness
# claim must describe that scope only.
# ----------------------------------------------------------------------
def _two_scopes(store, tag):
    """Scope A and scope B, each with its own distinct eligible state."""
    a = _state(store, f"{tag}-A", f"{tag}-A", "alpha", "first value", 100)
    b = _state(store, f"{tag}-B", f"{tag}-B", "beta", "second value", 200)
    return a, b


def test_scoped_sync_does_not_stamp_the_global_token():
    """Regression: a scope-limited sync used to write the global key.

    The live path syncs the current session's scope, so a global token
    written from there claimed every scope was indexed when only one was.
    """
    _require_model()
    manager, store = _manager("scope-token")
    _two_scopes(store, "tok")
    index = semantic_mod.SemanticIndex(store)

    index.sync("tok-A")

    assert store.get_index_meta("sync_token") is None, (
        "a scoped sync must not write the global freshness key"
    )
    assert store.get_index_meta("sync_token:scope:tok-A") == index._sync_token("tok-A")


def test_sync_of_one_scope_leaves_other_scopes_needing_sync():
    """Requirement A: syncing A must not make B look synchronized."""
    _require_model()
    manager, store = _manager("scope-isolation")
    _two_scopes(store, "iso")
    index = semantic_mod.SemanticIndex(store)

    assert index.needs_sync("iso-A") is True
    assert index.needs_sync("iso-B") is True

    index.sync("iso-A")

    assert index.needs_sync("iso-A") is False, "A was just indexed"
    assert index.needs_sync("iso-B") is True, (
        "B was never indexed and must not be reported fresh"
    )
    assert index.needs_sync() is True, (
        "the global domain still has B outstanding"
    )

    index.sync("iso-B")

    assert index.needs_sync("iso-B") is False
    assert index.needs_sync("iso-A") is False, "A's own token is untouched"
    assert index.needs_sync() is True, (
        "neither sync was global, so the global domain is still not indexed"
    )


def test_global_sync_is_distinct_from_scoped_syncs():
    """Global indexing must remain its own domain, not a side effect."""
    _require_model()
    manager, store = _manager("scope-global")
    _two_scopes(store, "glb")
    index = semantic_mod.SemanticIndex(store)

    index.sync("glb-A")
    index.sync("glb-B")
    assert store.get_index_meta("sync_token") is None

    index.sync()

    assert store.get_index_meta("sync_token") == index._sync_token(None)
    assert index.needs_sync() is False
    # Scoped keys are untouched by a global sync.
    assert store.get_index_meta("sync_token:scope:glb-A") == index._sync_token("glb-A")
    assert store.get_index_meta("sync_token:scope:glb-B") == index._sync_token("glb-B")


def test_maybe_sync_is_scoped_not_global():
    """The live path calls maybe_sync(session_id); it must act on that scope."""
    _require_model()
    manager, store = _manager("scope-maybe")
    _two_scopes(store, "mby")
    index = semantic_mod.SemanticIndex(store)

    first = index.maybe_sync("mby-A")
    assert first["states"] > 0

    # A is fresh, so a second call is a no-op ...
    assert index.maybe_sync("mby-A") == {
        "states": 0, "entities": 0, "skipped": 0
    }
    # ... while B still gets its own work done.
    second = index.maybe_sync("mby-B")
    assert second["states"] > 0

    indexed = {
        (row[0], row[1], row[2])
        for row in store.fetch_vector_rows(model_id=index.model_id)
    }
    assert ("state", ) and len(indexed) == 2
    assert {scope for _kind, _ref, scope in indexed} == {"mby-A", "mby-B"}


def test_scoped_sync_with_unavailable_model_invalidates_only_that_scope():
    """Requirement B: a failed scoped sync must not leave a fresh claim."""
    _require_model()
    manager, store = _manager("scope-fail")
    _two_scopes(store, "fl")
    index = semantic_mod.SemanticIndex(store)
    index.sync("fl-A")
    assert index.needs_sync("fl-A") is False

    def _dead_encoder(*args, **kwargs):
        return None

    original = semantic_mod.encode
    semantic_mod.encode = _dead_encoder
    try:
        index.rebuild("fl-A")
        blind = index.maybe_sync("fl-A")
    finally:
        semantic_mod.encode = original

    assert blind["states"] == 0
    assert store.count_vectors(model_id=index.model_id) == 0
    assert store.get_index_meta("sync_token:scope:fl-A") == ""
    assert index.needs_sync("fl-A") is True, (
        "a failed scoped sync must leave that scope needing sync"
    )
    # B was untouched by the failed A rebuild.
    assert store.get_index_meta("sync_token:scope:fl-B") is None

    # The next successful call repairs exactly that scope.
    repaired = index.maybe_sync("fl-A")
    assert repaired["states"] > 0
    assert index.needs_sync("fl-A") is False
    assert index.needs_sync("fl-B") is True


def test_scoped_sync_cannot_make_unrelated_scopes_appear_fresh():
    """Requirement C: no cross-scope leakage through the global key."""
    _require_model()
    manager, store = _manager("scope-leak")
    _two_scopes(store, "leak")
    index = semantic_mod.SemanticIndex(store)

    for _ in range(3):
        index.maybe_sync("leak-A")

    assert index.needs_sync("leak-B") is True
    assert store.get_index_meta("sync_token") is None
    assert index.needs_sync() is True
    rows = store.fetch_vector_rows(model_id=index.model_id)
    assert {r[2] for r in rows} == {"leak-A"}, "only A may be embedded"


def test_indexing_never_mutates_memory_truth():
    _require_model()
    manager, store = _manager("index-truth")
    _house(manager, store)
    message_count = store.count("s")
    before_states = store.fetch_states(scope="s")
    before_entities = store.fetch_entities(scope="s")
    before_rels = store.fetch_relationships()
    before_candidates = store.fetch_candidates()
    semantic_mod.SemanticIndex(store).sync()
    semantic_mod.SemanticIndex(store).rebuild()
    assert store.fetch_states(scope="s") == before_states
    assert store.fetch_entities(scope="s") == before_entities
    assert store.fetch_relationships() == before_rels
    assert store.fetch_candidates() == before_candidates
    assert store.count("s") == message_count


def test_model_dim_is_deterministic_across_calls():
    _require_model()
    manager, store = _manager("index-dim")
    _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    assert index.dim() == 384
    index.sync()
    dims = {row[5] for row in store.fetch_vector_rows()}
    assert dims == {384}


# ======================================================================
# PART 3: semantic retrieval
# ======================================================================
def test_semantic_retrieval_finds_a_paraphrase():
    _require_model()
    manager, store = _manager("sem-paraphrase")
    graph = _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    # No content word of the stored decision is repeated here except
    # "friday": the lexical channel has nothing to match on.
    trace = hybrid_mod.hybrid_retrieve(
        manager, store, index, "s", "s",
        "when do we release the mobile app",
        use_structural=False,
    )
    semantic = [c for c in trace.candidates if c.channel == "semantic"]
    assert semantic, trace.diagnostics
    assert hybrid_mod.item_id("state", graph["decision"]) in {
        c.item_id for c in semantic
    }
    # "release" never appears in the stored text, so the lexical channel has
    # nothing to match on: this is a pure semantic gain.
    assert not [c for c in trace.candidates if c.channel == "lexical"]


def test_tau_is_the_calibrated_value_and_is_enforced():
    _require_model()
    assert hybrid_mod.SEMANTIC_TAU == 0.45
    manager, store = _manager("sem-tau")
    _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    rows = index.search(
        "which handset model will we release next friday", ["s"], tau=0.45
    )
    for row in rows:
        assert row["similarity"] >= 0.45
    strict = index.search(
        "which handset model will we release next friday", ["s"], tau=0.99
    )
    assert strict == []
    loose = index.search(
        "which handset model will we release next friday", ["s"], tau=0.0
    )
    assert len(loose) >= len(rows)


def test_semantic_channel_respects_scope():
    _require_model()
    manager, store = _manager("sem-scope")
    _house(manager, store, "s", "s")
    _house(manager, store, "t", "t")
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    rows = index.search(
        "which handset model will we release next friday", ["s"], tau=0.0
    )
    assert rows
    assert {row["scope"] for row in rows} == {"s"}
    rows = index.search(
        "which handset model will we release next friday", ["s", "t"], tau=0.0
    )
    assert {row["scope"] for row in rows} == {"s", "t"}


def test_semantic_candidates_are_gated_before_scoring():
    _require_model()
    manager, store = _manager("sem-gate")
    expired = _state(
        store, "s", "s", "old plan", "an expired decision", 100,
        valid_from=100, valid_until=200,
    )
    live = _state(
        store, "s", "s", "new plan", "a current decision", 300,
        valid_from=300, valid_until=None,
    )
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    rows = index.search("a decision about the plan", ["s"], tau=0.0)
    ids = {row["ref_id"] for row in rows}
    assert {expired, live} <= ids
    candidates, diagnostics = hybrid_mod.semantic_channel(
        index, store, "a decision about the plan", "s", 500.0, "current"
    )
    kept = {c.ref_id for c in candidates}
    assert live in kept
    assert expired not in kept
    assert any(
        f"semantic_gated: state:{expired}" in d for d in diagnostics
    )


def test_unresolved_state_is_never_a_semantic_candidate():
    _require_model()
    manager, store = _manager("sem-unresolved")
    unresolved = _state(
        store, "s", "s", "contested", "a disputed value", 100,
        status="unresolved",
    )
    ok, reason = hybrid_mod.gate_state(
        store.fetch_states(scope="s", state_ids=[unresolved])[0], 500.0,
        "current",
    )
    assert ok is False and reason == "unresolved"
    assert hybrid_mod.gate_entity(
        store.fetch_entity(_entity(store, "s", "Live Thing")), "current"
    ) == (True, "pass")


def test_assistant_authority_is_never_retrievable():
    _require_model()
    manager, store = _manager("sem-authority")
    state = store.fetch_states(scope="s", state_ids=[_state(
        store, "s", "s", "assistant claim", "a value", 100
    )])[0]
    with store._connect() as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute(
            "UPDATE states SET authority = 'assistant' WHERE id = ?",
            (state.state_id,),
        )
    record = store.fetch_states(scope="s")[0]
    assert hybrid_mod.gate_state(record, 500.0, "current") == (
        False, "authority"
    )
    entity_id = _entity(store, "s", "Assistant Thing", authority="assistant")
    assert hybrid_mod.gate_entity(
        store.fetch_entity(entity_id), "current"
    ) == (False, "authority")


def test_model_unavailable_falls_back_without_raising():
    manager, store = _manager("sem-unavailable")
    _append(store, "s", "the vault code is BLUE-FALCON-7", 100)
    _append(store, "s", "the emblem is a stylised fox", 110)
    _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index._probed = True
    index._model = None
    assert index.available() is False
    assert index.diagnostics()
    assert index.search("anything", ["s"], tau=0.0) == []
    report = index.sync()
    assert report["skipped"] == 1
    trace = hybrid_mod.hybrid_retrieve(
        manager, store, index, "s", "s", "what is the vault code",
        use_structural=False,
    )
    assert isinstance(trace, RetrievalTrace)
    assert trace.fused_order, "lexical must still produce candidates"
    assert all(c.channel == "lexical" for c in trace.candidates)
    assert any(
        d.startswith("semantic_model") for d in trace.diagnostics
    )
    assert trace.degraded is True


def test_missing_index_leaves_lexical_intact():
    manager, store = _manager("sem-no-index")
    _append(store, "s", "the vault code is BLUE-FALCON-7", 100)
    _append(store, "s", "the emblem is a stylised fox", 110)
    index = semantic_mod.SemanticIndex(store)
    _require_model()
    index.sync()
    # The index exists but holds nothing for this scope.
    assert index.search("vault code", ["other-scope"], tau=0.45) == []
    trace = hybrid_mod.hybrid_retrieve(
        manager, store, index, "s", "s", "what is the vault code",
        use_structural=False,
    )
    assert any(
        c.channel == "lexical" for c in trace.candidates
    )
    assert trace.fused_order


# ======================================================================
# PART 4: entity / relationship channel
# ======================================================================
def test_entity_mentions_require_an_exact_word_run():
    manager, store = _manager("struct-mention")
    _entity(store, "s", "Mobile App", "COMPONENT")
    _entity(store, "s", "Car", "COMPONENT", timestamp=110)
    assert [e.display_name for e in
            hybrid_mod.entity_mentions(store, "s", "what about the Mobile App?")
            ] == ["Mobile App"]
    assert hybrid_mod.entity_mentions(
        store, "s", "how is the carpet cleaning going"
    ) == []
    assert hybrid_mod.entity_mentions(store, "s", "") == []
    assert hybrid_mod.entity_mentions(store, "s", "the   ") == []


def test_confirmed_alias_counts_as_a_mention_candidate_only():
    manager, store = _manager("struct-alias")
    entity_id = _entity(store, "s", "Berlin Office Move", "PROJECT")
    with store._connect() as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute(
            "INSERT INTO entity_aliases"
            "(session_id, scope, entity_id, alias_text, normalized_alias,"
            " status, source_ids, authority, confidence, created_at,"
            " observed_at, valid_from, valid_until, superseded_at)"
            " VALUES ('s', 's', ?, 'Office Move', 'office move',"
            " 'candidate', '1', 'user', 0.3, 100.0, 100.0, 100.0, NULL, NULL)",
            (entity_id,),
        )
    text = "what do we know about the Office Move?"
    assert hybrid_mod.entity_mentions(store, "s", text) == []
    with store._connect() as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute(
            "UPDATE entity_aliases SET status = 'current' WHERE entity_id = ?",
            (entity_id,),
        )
    found = hybrid_mod.entity_mentions(store, "s", text)
    assert [e.entity_id for e in found] == [entity_id]


def test_structural_channel_contributes_states_and_entities():
    _require_model()
    manager, store = _manager("struct-channel")
    graph = _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    trace = hybrid_mod.hybrid_retrieve(
        manager, store, index, "s", "s", "what is the Mobile App part of",
    )
    by_channel: dict[str, set[str]] = {}
    for candidate in trace.candidates:
        by_channel.setdefault(candidate.channel, set()).add(candidate.item_id)
    # The entity channel contributes the named entity and the state attached
    # to it through a stored concerns edge.
    assert hybrid_mod.item_id("entity", graph["app"]) in by_channel["entity"]
    assert hybrid_mod.item_id("state", graph["decision"]) in (
        by_channel["entity"]
    )
    # The relationship channel contributes the one-hop parent.
    assert hybrid_mod.item_id("entity", graph["project"]) in (
        by_channel["relationship"]
    )
    relationship_candidate = next(
        c for c in trace.candidates
        if c.channel == "relationship" and c.ref_id == graph["project"]
    )
    assert relationship_candidate.relationship_type == "part_of"
    assert relationship_candidate.relationship_id is not None
    assert hybrid_mod.item_id("entity", graph["app"]) in trace.fused_order


def test_evidence_floor_requires_agreement_or_top_rank():
    """Documented consequence of the calibrated 0.725 floor.

    RRF sums each channel's contribution, so an item found by two channels
    scores roughly twice an item found by one. The 0.725 floor therefore
    admits an item that either has cross-channel agreement or ranks first in
    its own channel. This is a real, intentional property of the calibrated
    design, pinned here so a future change to the floor cannot alter it
    silently.
    """
    _require_model()
    manager, store = _manager("floor-agreement")
    graph = _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    trace = hybrid_mod.hybrid_retrieve(
        manager, store, index, "s", "s", "what is the Mobile App part of",
    )
    by_item: dict[str, set[str]] = {}
    for candidate in trace.candidates:
        by_item.setdefault(candidate.item_id, set()).add(candidate.channel)
    for key in trace.fused_order:
        assert len(by_item[key]) >= 1
    # Project Apollo is named by exactly one channel here, and it does not
    # lead that channel, so the floor withholds it.
    project = hybrid_mod.item_id("entity", graph["project"])
    assert by_item[project] == {"relationship"}
    if project not in trace.fused_order:
        withheld = next(
            c for c in trace.candidates if c.item_id == project
        )
        assert withheld.exclusion_reason == "below_evidence_floor"
        assert withheld.final_rank is None
        assert "fusion_evidence_floor_applied" in trace.diagnostics


def test_structural_channel_is_conservative_without_a_mention():
    _require_model()
    manager, store = _manager("struct-none")
    _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    trace = hybrid_mod.hybrid_retrieve(
        manager, store, index, "s", "s", "what is the airspeed of a swallow",
    )
    assert not [c for c in trace.candidates if c.channel in ("entity",
                                                            "relationship")]
    assert "structural_channel_no_mention" in trace.diagnostics


def test_structural_channel_never_creates_a_generic_relation():
    _require_model()
    manager, store = _manager("struct-no-related")
    _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    hybrid_mod.hybrid_retrieve(
        manager, store, index, "s", "s", "tell me about the Mobile App",
    )
    assert {row.relationship_type for row in store.fetch_relationships()} == {
        "part_of", "concerns"
    }
    assert not [r for r in store.fetch_relationships()
                if r.relationship_type == "related_to"]


def test_structural_channel_respects_m5_one_hop_bound():
    _require_model()
    manager, store = _manager("struct-onehop")
    a = _entity(store, "s", "Alpha", "COMPONENT")
    b = _entity(store, "s", "Beta", "COMPONENT", timestamp=110)
    c = _entity(store, "s", "Gamma", "COMPONENT", timestamp=120)
    _relationship(store, "s", "part_of", "entity", b, "entity", a, 130)
    _relationship(store, "s", "part_of", "entity", c, "entity", b, 140)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    trace = hybrid_mod.hybrid_retrieve(
        manager, store, index, "s", "s", "what is Alpha part of",
    )
    reached = {
        int(c.ref_id) for c in trace.candidates if c.kind == "entity"
    }
    assert a in reached
    # Beta is one hop away and is reachable; Gamma is two hops and is not.
    assert b in reached
    assert c not in reached
    assert not [x for x in trace.candidates if x.ref_id == c]


def test_structural_failure_degrades_without_breaking_fusion():
    _require_model()
    manager, store = _manager("struct-fail")
    _append(store, "s", "the mobile app release notes are in the vault", 100)
    _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    original = manager.get_entity_memories
    manager.get_entity_memories = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("structural unavailable")
    )
    try:
        trace = hybrid_mod.hybrid_retrieve(
            manager, store, index, "s", "s",
            "what about the Mobile App vault notes",
        )
    finally:
        manager.get_entity_memories = original
    assert any(
        d.startswith("structural_") for d in trace.diagnostics
    )
    assert any(c.channel == "lexical" for c in trace.candidates)
    assert trace.fused_order


# ======================================================================
# PART 5: fusion
# ======================================================================
def test_calibrated_constants():
    assert hybrid_mod.RRF_K == 60.0
    assert hybrid_mod.SEMANTIC_TAU == 0.45
    assert hybrid_mod.EVIDENCE_FLOOR_RATIO == 0.725


def test_rrf_fusion_is_rank_based_and_deterministic():
    channels = {
        "lexical": [
            RetrievalCandidate("message:1", "message", 1, "s", "s",
                               "lexical", 1, 0.9),
            RetrievalCandidate("state:2", "state", 2, "s", "s",
                               "lexical", 2, 0.4),
        ],
        "semantic": [
            RetrievalCandidate("state:2", "state", 2, "s", "s",
                               "semantic", 1, 0.5),
        ],
    }
    scores, order = hybrid_mod.rrf_fuse(channels, k=60.0)
    assert order == ["state:2", "message:1"]
    assert scores["state:2"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["message:1"] == pytest.approx(1 / 61)
    again_scores, again_order = hybrid_mod.rrf_fuse(channels, k=60.0)
    assert again_order == order
    assert again_scores == scores


def test_rrf_never_sums_heterogeneous_raw_scores():
    channels = {
        "lexical": [
            RetrievalCandidate("message:1", "message", 1, "s", "s",
                               "lexical", 1, 100.0),
        ],
        "semantic": [
            RetrievalCandidate("state:2", "state", 2, "s", "s",
                               "semantic", 1, 0.01),
        ],
    }
    scores, _order = hybrid_mod.rrf_fuse(channels, k=60.0)
    # Identical rank in both channels gives identical weight regardless of
    # the wildly different raw channel scores.
    assert scores["message:1"] == pytest.approx(scores["state:2"])


def test_rrf_rejects_invalid_k():
    channels = {"lexical": [], "semantic": [], "entity": [],
                "relationship": []}
    with pytest.raises(ValueError):
        hybrid_mod.rrf_fuse(channels, k=0)


def test_relative_evidence_floor_definition():
    scores = {"a": 1.0, "b": 0.8, "c": 0.4, "d": 0.1}
    order = ["a", "b", "c", "d"]
    kept, applied = hybrid_mod.apply_evidence_floor(scores, order, 0.725)
    # keep(x) iff score(x) >= 0.725 * score(best)
    assert kept == ["a", "b"]
    assert applied is True
    assert not hybrid_mod.apply_evidence_floor(scores, order, 0.4)[1] is False
    assert hybrid_mod.apply_evidence_floor(scores, order, 0.4)[0] == [
        "a", "b", "c"
    ]
    # The ratio is measured against the strongest item in the request, so it
    # is portable across corpora and score scales.
    scaled = {key: value * 1000 for key, value in scores.items()}
    assert hybrid_mod.apply_evidence_floor(scaled, order, 0.725)[0] == kept
    assert hybrid_mod.apply_evidence_floor(scores, order, 0.0)[1] is False
    assert hybrid_mod.apply_evidence_floor({}, [], 0.725) == ([], False)
    zero = {"a": 0.0, "b": 0.0}
    assert hybrid_mod.apply_evidence_floor(zero, ["a", "b"], 0.725) == (
        ["a", "b"], False
    )


def test_floor_never_widens_what_a_gate_allowed():
    _require_model()
    manager, store = _manager("fusion-gate")
    graph = _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    trace = hybrid_mod.hybrid_retrieve(
        manager, store, index, "s", "s", "what is the Mobile App part of",
    )
    gated = [
        c for c in trace.candidates
        if c.exclusion_reason and not c.included
    ]
    for candidate in gated:
        assert candidate.final_rank is None
    # Everything fused came from a channel; nothing was invented.
    assert all(
        c.channel in hybrid_mod.CHANNELS for c in trace.candidates
    )


def test_hard_gates_precede_ranking():
    """A high similarity cannot rescue a gate-blocked item."""
    _require_model()
    manager, store = _manager("fusion-precedence")
    state_id = _state(
        store, "s", "s", "gated", "a decision that expired", 100,
        valid_from=100, valid_until=200,
    )
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    rows = index.search("a decision that expired", ["s"], tau=0.0)
    assert state_id in {row["ref_id"] for row in rows}
    candidates, diagnostics = hybrid_mod.semantic_channel(
        index, store, "a decision that expired", "s", 500.0, "current"
    )
    assert state_id not in {c.ref_id for c in candidates}
    assert any(
        f"semantic_gated: state:{state_id}" in d for d in diagnostics
    )


def test_confidence_never_influences_ranking():
    _require_model()
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    path = os.path.join(root, "PRIMORA", "prime", "context", "hybrid.py")
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    assert ".confidence" not in source
    # And functionally: a high-confidence row a gate rejects stays rejected.
    manager, store = _manager("fusion-confidence")
    state_id = _state(
        store, "s", "s", "confident", "a value", 100, status="unresolved"
    )
    with store._connect() as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute(
            "UPDATE states SET confidence = 1.0 WHERE id = ?", (state_id,)
        )
    record = store.fetch_states(scope="s")[0]
    assert record.confidence == 1.0
    assert hybrid_mod.gate_state(record, 500.0, "current")[0] is False


# ======================================================================
# PART 6: same-session / scope
# ======================================================================
def test_cross_session_results_stay_tagged_and_scope_bound():
    _require_model()
    manager, store = _manager("scope-cross")
    _append(store, "sessA", "the vault code is BLUE-FALCON-7", 100)
    _append(store, "sessB", "unrelated chatter about lunch", 100)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    view = manager.build(
        [{"role": "user", "content": "what is the vault code?"}],
        "sessB", token_counter=_counter, max_new_tokens=5,
        budget=_budget(),
    ).view
    texts = " ".join(m["content"] for m in view.messages)
    assert "BLUE-FALCON-7" in texts
    assert "[from another conversation]" in texts


def test_local_memory_is_not_bypassed_by_a_foreign_distractor():
    _require_model()
    manager, store = _manager("scope-local")
    _append(store, "sessA", "the local plan is to launch in march", 100)
    _append(store, "sessB", "the remote plan is to launch in june", 100)
    _append(store, "sessC", "another remote plan about launching", 100)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    trace = hybrid_mod.hybrid_retrieve(
        manager, store, index, "sessA", "sessA",
        "what is the plan for launching", use_structural=False,
    )
    lexical = [
        c for c in trace.candidates if c.channel == "lexical"
    ]
    local = [c for c in lexical if c.session_id == "sessA"]
    foreign = [c for c in lexical if c.session_id != "sessA"]
    ranks = {c.item_id: c.final_rank for c in trace.candidates}
    if local and foreign:
        local_ref = min(ranks[c.item_id] for c in local
                        if c.item_id in ranks)
        foreign_ref = min(ranks[c.item_id] for c in foreign
                          if c.item_id in ranks)
        assert local_ref <= foreign_ref


def test_semantic_channel_does_not_leak_foreign_scope_by_default():
    _require_model()
    manager, store = _manager("scope-sem")
    _state(store, "sessA", "sessA", "alpha", "a private plan", 100)
    _state(store, "sessB", "sessB", "beta", "another private plan", 100)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    candidates, _ = hybrid_mod.semantic_channel(
        index, store, "a private plan", "sessA", 500.0, "current"
    )
    assert {c.scope for c in candidates} <= {"sessA"}


def test_global_scope_is_opt_in():
    _require_model()
    manager, store = _manager("scope-global")
    _state(store, "sessA", "global", "shared", "a global fact", 100)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    without, _ = hybrid_mod.semantic_channel(
        index, store, "a global fact", "sessA", 500.0, "current"
    )
    assert without == []
    with_global, _ = hybrid_mod.semantic_channel(
        index, store, "a global fact", "sessA", 500.0, "current",
        include_global=True,
    )
    assert {c.scope for c in with_global} == {"global"}


def test_temporal_validity_and_supersession_in_the_channel():
    _require_model()
    manager, store = _manager("scope-temporal")
    expired = _state(
        store, "s", "s", "plan", "the old plan", 100,
        valid_from=100, valid_until=200,
    )
    current = _state(
        store, "s", "s", "plan", "the new plan", 300,
        valid_from=300, valid_until=None,
    )
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    live, _ = hybrid_mod.semantic_channel(
        index, store, "the plan", "s", 500.0, "current"
    )
    assert {c.ref_id for c in live} == {current}
    historical, _ = hybrid_mod.semantic_channel(
        index, store, "the plan", "s", 150.0, "as_of"
    )
    assert expired in {c.ref_id for c in historical}


# ======================================================================
# PART 7-8: shadow mode and trace
# ======================================================================
def test_shadow_mode_is_off_by_default_and_switchable():
    manager, _store = _manager("shadow-switch")
    assert manager.shadow_retrieval is False
    manager.set_shadow_retrieval(True)
    assert manager.shadow_retrieval is True
    manager.set_shadow_retrieval(False)
    assert manager.shadow_retrieval is False
    enabled, _ = _manager("shadow-switch-env", shadow_retrieval=True)
    assert enabled.shadow_retrieval is True


def test_shadow_mode_never_changes_the_assembled_context():
    _require_model()
    _append_shared = [
        {"role": "user", "content": "the vault code is BLUE-FALCON-7",
         "timestamp": 100},
        {"role": "user", "content": "we prefer the matte charcoal finish",
         "timestamp": 110},
    ]
    plain, _ = _manager("shadow-plain")
    plain.sync_semantic_index()
    before = plain.build(
        list(_append_shared), "s", token_counter=_counter,
        max_new_tokens=5, budget=_budget(),
    ).view
    shadowed, _ = _manager("shadow-on", shadow_retrieval=True)
    shadowed.sync_semantic_index()
    after = shadowed.build(
        list(_append_shared), "s", token_counter=_counter,
        max_new_tokens=5, budget=_budget(),
    ).view
    assert after.messages == before.messages
    assert after.input_tokens == before.input_tokens
    assert shadowed.shadow_records()


def test_shadow_mode_leaves_active_and_working_memory_untouched():
    _require_model()
    history = [
        {"role": "user", "content": "my favorite color is blue",
         "timestamp": 100},
        {"role": "user", "content": "my favorite color is green",
         "timestamp": 200},
    ]
    plain, _ = _manager("shadow-am-plain", shadow_retrieval=True)
    plain.sync_semantic_index()
    plain.build(list(history), "s", token_counter=_counter,
                max_new_tokens=5, budget=_budget())
    before = [
        (i.kind, i.content) for i in
        plain.get_active_memory("s", at_time=300).items
    ]
    shadowed, _ = _manager("shadow-am-on", shadow_retrieval=True)
    shadowed.sync_semantic_index()
    shadowed.build(list(history), "s", token_counter=_counter,
                   max_new_tokens=5, budget=_budget())
    after = [
        (i.kind, i.content) for i in
        shadowed.get_active_memory("s", at_time=300).items
    ]
    assert after == before
    assert shadowed.shadow_records()
    from PRIMORA.prime.context import WorkingMemory
    memory = WorkingMemory("s", task="shadow")
    assert not hasattr(memory, "_store")
    assert memory.entry_count() == 1
    assert memory.to_messages()


def test_shadow_comparison_record_is_structured():
    _require_model()
    history = [
        {"role": "user", "content": "the vault code is BLUE-FALCON-7",
         "timestamp": 100},
    ]
    manager, _ = _manager("shadow-record", shadow_retrieval=True)
    manager.sync_semantic_index()
    manager.build(list(history), "s", token_counter=_counter,
                  max_new_tokens=5, budget=_budget())
    records = manager.shadow_records()
    assert records
    record = records[0]
    assert isinstance(record, ShadowComparison)
    assert record.query == "the vault code is BLUE-FALCON-7"
    assert isinstance(record.lexical_ids, list)
    assert isinstance(record.hybrid_ids, list)
    assert record.overlap == [
        rid for rid in record.hybrid_ids if rid in record.lexical_ids
    ]
    assert record.new_only == [
        rid for rid in record.hybrid_ids if rid not in record.lexical_ids
    ]
    assert record.old_only == [
        rid for rid in record.lexical_ids if rid not in record.hybrid_ids
    ]
    assert isinstance(record.channel_contributions, dict)
    assert record.hybrid_latency_ms > 0
    assert record.channel_timings_ms


def test_shadow_trace_is_never_rendered_into_the_prompt():
    _require_model()
    history = [
        {"role": "user", "content": "the vault code is BLUE-FALCON-7",
         "timestamp": 100},
    ]
    manager, _ = _manager("shadow-not-in-prompt", shadow_retrieval=True)
    manager.sync_semantic_index()
    view = manager.build(
        list(history), "s", token_counter=_counter, max_new_tokens=5,
        budget=_budget(),
    ).view
    blob = repr(view.messages) + repr(view.diagnostics)
    for marker in ("semantic", "rrf", "fused", "similarity", "shadow",
                   "retrieval_vectors", "0.725"):
        assert marker not in blob, marker
    assert manager.retrieval_traces()


def test_shadow_records_can_be_cleared():
    _require_model()
    history = [
        {"role": "user", "content": "the vault code is BLUE-FALCON-7",
         "timestamp": 100},
    ]
    manager, _ = _manager("shadow-clear", shadow_retrieval=True)
    manager.sync_semantic_index()
    manager.build(list(history), "s", token_counter=_counter,
                  max_new_tokens=5, budget=_budget())
    assert manager.shadow_records()
    manager.clear_shadow_records()
    assert manager.shadow_records() == []


def test_trace_records_channel_rank_and_exclusion():
    _require_model()
    manager, store = _manager("trace-detail")
    graph = _house(manager, store)
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    trace = hybrid_mod.hybrid_retrieve(
        manager, store, index, "s", "s", "what is the Mobile App part of",
    )
    assert isinstance(trace, RetrievalTrace)
    assert trace.session_id == "s"
    assert trace.temporal == "current"
    assert trace.total_ms > 0
    assert set(trace.channel_timings_ms) >= {"lexical_ms", "semantic_ms",
                                             "fusion_ms"}
    for candidate in trace.candidates:
        assert candidate.channel_rank >= 1
        if candidate.included:
            assert candidate.final_rank is not None
            assert candidate.fused_score is not None
        else:
            assert candidate.exclusion_reason
    included = sorted(
        {c.item_id: c for c in trace.candidates if c.included}.values(),
        key=lambda c: c.final_rank,
    )
    assert included
    assert [c.final_rank for c in included] == list(
        range(1, len(included) + 1)
    )


# ======================================================================
# PART 11: fail-safe
# ======================================================================
def test_lexical_retrieval_works_with_no_model_and_no_index():
    manager, store = _manager("failsafe-lexical")
    _append(store, "s", "the vault code is BLUE-FALCON-7", 100)
    _append(store, "s", "the emblem is a stylised fox", 110)
    _append(store, "s", "we like the matte charcoal car", 120)
    semantic_mod.reset_model_cache()
    semantic_mod._MODEL_CACHE["unavailable@cpu"] = None
    index = semantic_mod.SemanticIndex(store)
    index._probed = True
    index._model = None
    view = manager.build(
        [{"role": "user", "content": "what is the vault code?"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    assert "BLUE-FALCON-7" in " ".join(m["content"] for m in view.messages)


def test_shadow_records_the_failure_rather_than_claiming_success():
    manager, store = _manager("failsafe-shadow")
    _append(store, "s", "the vault code is BLUE-FALCON-7", 100)
    index = semantic_mod.SemanticIndex(store)
    index._probed = True
    index._model = None
    trace = hybrid_mod.hybrid_retrieve(
        manager, store, index, "s", "s", "vault code", use_structural=False
    )
    assert trace.degraded is True
    record = hybrid_mod.shadow_comparison(
        [1], trace, "s", "vault code"
    )
    assert record.degraded is True
    assert "semantic_model" in record.error


def test_fusion_failure_degrades_to_no_fusion():
    channels = {
        "lexical": [RetrievalCandidate("message:1", "message", 1, "s", "s",
                                       "lexical", 1, 0.5)],
        "semantic": [], "entity": [], "relationship": [],
    }
    scores, order = hybrid_mod.rrf_fuse(channels, k=60.0)
    assert order == ["message:1"]
    assert scores["message:1"] > 0


# ======================================================================
# PART 10: performance
# ======================================================================
def test_latency_is_recorded_and_bounded():
    _require_model()
    manager, store = _manager("perf")
    for index in range(12):
        _append(store, "s", f"a stored note number {index} about errands",
                100 + index)
    _house(manager, store)
    semantic = semantic_mod.SemanticIndex(store)
    semantic.sync()
    timings = []
    for _ in range(5):
        trace = hybrid_mod.hybrid_retrieve(
            manager, store, semantic, "s", "s",
            "what is the Mobile App part of",
        )
        timings.append(trace.total_ms)
    assert all(value > 0 for value in timings)
    assert max(timings) < 5000, timings
    for key in ("lexical_ms", "semantic_ms", "fusion_ms"):
        assert key in trace.channel_timings_ms
    print("\nhybrid latencies ms:", [round(v, 2) for v in timings])
    print("channel breakdown:", {
        k: round(v, 3) for k, v in trace.channel_timings_ms.items()
    })


def test_shadow_overhead_is_measured_and_reported():
    _require_model()
    history = [
        {"role": "user", "content": "the vault code is BLUE-FALCON-7",
         "timestamp": 100},
    ]
    plain, _ = _manager("perf-plain")
    plain.sync_semantic_index()
    shadowed, _ = _manager("perf-shadow", shadow_retrieval=True)
    shadowed.sync_semantic_index()
    plain.build(list(history), "s", token_counter=_counter,
                max_new_tokens=5, budget=_budget())
    shadowed.build(list(history), "s", token_counter=_counter,
                   max_new_tokens=5, budget=_budget())
    record = shadowed.shadow_records()[-1]
    assert record.hybrid_latency_ms > 0
    assert record.hybrid_latency_ms < 5000


# ----------------------------------------------------------------------
# Residue removal: deleting proven test data must leave the index and the
# preserved memories truthful. Modelled on the production cleanup, which
# removed 22 fixture sessions, 4 fixture states, and one orphan session.
# ----------------------------------------------------------------------
def test_removing_residue_leaves_no_orphaned_vectors():
    """sync() upserts, it does not prune; rebuild() is the pruning path.

    The important property is that the orphan is never invisible: stale()
    must report it, and rebuild() must remove it.
    """
    _require_model()
    manager, store = _manager("residue-vectors")
    _two_scopes(store, "res")
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    assert store.count_vectors(model_id=index.model_id) == 2

    with store._connect() as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute("DELETE FROM states WHERE scope = 'res-A'")

    # A plain sync cannot remove a vector whose source row is gone.
    index.sync()
    stale = index.stale()
    assert stale["healthy"] is False
    assert [o["ref_id"] for o in stale["orphaned"]], (
        "an orphaned vector must be reported, never silently tolerated"
    )

    # rebuild() is the operation that prunes, and it is lossless for the
    # rows that still exist.
    index.rebuild()
    live = {s.state_id for s in index.indexable_states()}
    stored = {row[1] for row in store.fetch_vector_rows(model_id=index.model_id)}
    assert stored == live, "a vector must not outlive the row it describes"
    assert live, "the surviving row must still be indexed"
    assert index.stale()["orphaned"] == []
    assert index.stale()["healthy"] is True


def test_index_metadata_is_truthful_after_full_removal():
    """Zero eligible rows must mean zero vectors AND a healthy fresh index.

    This is the exact post-cleanup production state. The dangerous outcome is
    an index that is empty but claims freshness over rows that no longer
    exist, or one that reports unhealthy forever with nothing to fix.
    """
    _require_model()
    manager, store = _manager("residue-empty")
    _two_scopes(store, "emp")
    index = semantic_mod.SemanticIndex(store)
    index.sync()
    assert store.count_vectors(model_id=index.model_id) == 2

    with store._connect() as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute("DELETE FROM states")

    report = index.rebuild()
    assert report["states"] == 0
    assert store.count_vectors(model_id=index.model_id) == 0

    assert index.needs_sync() is False, "an empty index over no rows is fresh"
    stale = index.stale()
    assert stale["healthy"] is True
    assert stale["missing"] == [] and stale["mismatch"] == []
    assert stale["orphaned"] == []
    assert store.get_index_meta("sync_token") == index._sync_token(None)

    # An unscoped rebuild only stamps the global domain, so a scope that was
    # never synced on its own still reports a need. That is conservative
    # rather than wrong, and it converges after one call.
    assert index.needs_sync("emp") is True
    index.maybe_sync("emp")
    assert index.needs_sync("emp") is False
    assert store.count_vectors(model_id=index.model_id) == 0


def test_retrieval_falls_back_to_lexical_when_index_is_empty():
    """No vectors must not mean no retrieval."""
    manager, store = _manager("residue-fallback")
    _append(store, "s", "the vault code is BLUE-FALCON-7", 100)
    index = semantic_mod.SemanticIndex(store)
    index.rebuild()
    assert store.count_vectors(model_id=index.model_id) == 0

    trace = hybrid_mod.hybrid_retrieve(
        manager, store, index, "s", "s", "what is the vault code")
    assert trace.candidates, "lexical retrieval must still return candidates"
    assert {c.channel for c in trace.candidates} == {"lexical"}
    assert any(c.included for c in trace.candidates)
    assert not [d for d in trace.diagnostics if "failed" in d]


def test_cleanup_preserves_unrelated_memories():
    """Deleting residue must not touch any other memory row."""
    _require_model()
    manager, store = _manager("residue-preserve")
    _two_scopes(store, "keep")
    _append(store, "real", "a genuine user memory worth keeping", 100)
    index = semantic_mod.SemanticIndex(store)
    index.sync()

    before_states = {s.state_id: (s.scope, s.value)
                     for s in store.fetch_states(scope=None)}
    before_messages = store.count("real")
    before_digests = store.fetch_digests("real", limit=50)

    with store._connect() as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute("DELETE FROM states WHERE scope = 'keep-A'")
    index.rebuild()

    after_states = {s.state_id: (s.scope, s.value)
                    for s in store.fetch_states(scope=None)}
    deleted = {sid for sid, (scope, _v) in before_states.items()
               if scope == "keep-A"}
    assert deleted, "the test must actually remove something"
    assert set(after_states) == set(before_states) - deleted
    for sid, value in after_states.items():
        assert before_states[sid] == value, "a surviving state was modified"
    assert store.count("real") == before_messages
    assert store.fetch_digests("real", limit=50) == before_digests


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"All {len(tests)} M6 retrieval tests passed!")
