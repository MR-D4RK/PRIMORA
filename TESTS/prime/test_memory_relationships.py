import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import BudgetConfig, ContextManager, WorkingMemory
from PRIMORA.prime.context import relationships as relationships_mod
from PRIMORA.prime.context.store import ContextStore

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]


def _db(name=None):
    _SEQ[0] += 1
    filename = name or f"relationships-{_SEQ[0]}.db"
    return os.path.join(_TEST_TMPDIR.name, filename)


def _append(store, session_id, content, timestamp, role="user"):
    message_id, is_new, conflict = store.append(
        session_id,
        store.count(session_id),
        role,
        content,
        timestamp=timestamp,
    )
    assert is_new and conflict is None
    return message_id


def _manager(name=None):
    path = _db(name)
    store = ContextStore(path)
    return ContextManager(db_path=path), store


def _active_entity(
    manager,
    store,
    session_id,
    display_name,
    entity_type="COMPONENT",
    scope=None,
    timestamp=100,
):
    source_id = _append(
        store, session_id, f"entity evidence {display_name}", timestamp
    )
    entity_id, diagnostics = manager.create_entity_candidate(
        session_id,
        display_name,
        entity_type,
        source_ids=[source_id],
        scope=scope,
        timestamp=timestamp + 5,
    )
    assert entity_id is not None, diagnostics
    entity, diagnostics = manager.confirm_entity(entity_id)
    assert entity is not None, diagnostics
    return entity


def _state(
    store,
    session_id,
    scope,
    concept,
    value,
    timestamp=100,
    memory_type="FACT",
    source=True,
):
    source_ids = []
    if source:
        source_ids = [_append(
            store, session_id, f"state evidence {concept}", timestamp
        )]
    state_id = store.create_state(
        session_id,
        scope,
        concept,
        value,
        source_ids,
        status="current",
        authority="user",
        timestamp=timestamp,
        memory_type=memory_type,
    )
    return store.fetch_states(
        scope=scope, concept=concept, status="current"
    )[-1]


def _relationship_candidate(
    manager,
    store,
    relationship_type,
    subject_id,
    object_id,
    session_id="s",
    scope=None,
    authority="user",
    timestamp=100,
    valid_from=None,
    valid_until=None,
    source=True,
):
    source_ids = []
    if source:
        source_ids = [_append(
            store,
            session_id,
            f"relationship evidence {relationship_type}",
            timestamp,
        )]
    relationship_id, diagnostics = manager.create_relationship_candidate(
        session_id,
        relationship_type,
        subject_id,
        object_id,
        source_ids=source_ids,
        authority=authority,
        scope=scope,
        timestamp=timestamp + 5,
        valid_from=valid_from,
        valid_until=valid_until,
    )
    assert relationship_id is not None, diagnostics
    return store.fetch_relationship(relationship_id)


def _current_relationship(
    manager,
    relationship_id,
    at_time=None,
    valid_from=None,
    valid_until=None,
):
    relationship, diagnostics = manager.confirm_relationship(
        relationship_id,
        valid_from=valid_from,
        valid_until=valid_until,
        at_time=at_time,
    )
    assert relationship is not None, diagnostics
    return relationship


def _counter(messages):
    return sum(len(message.get("content", "").split()) + 2
               for message in messages)


def _budget():
    return BudgetConfig(
        input_target=220,
        input_ceiling=400,
        output_reserve=20,
        total_budget=420,
    )


def test_migration_creates_relationship_table_without_fabricating_rows():
    path = _db("legacy-relationships.db")
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            concept TEXT NOT NULL,
            value TEXT NOT NULL,
            source_ids TEXT NOT NULL,
            created_at REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'current',
            supersedes_id INTEGER,
            authority TEXT NOT NULL DEFAULT 'user'
        );
        INSERT INTO states
            (id, session_id, scope, concept, value, source_ids, created_at,
             status, supersedes_id, authority)
        VALUES (1, 'legacy', 'legacy', 'phrase', 'Blue Lantern', '1', 100,
                'current', NULL, 'user');
        """
    )
    con.commit()
    con.close()

    store = ContextStore(path)
    con = sqlite3.connect(path)
    tables = {
        row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    columns = {
        row[1] for row in con.execute("PRAGMA table_info(relationships)")
    }
    con.close()

    assert {"entities", "entity_aliases", "relationships"} <= tables
    assert {
        "id", "session_id", "scope", "relationship_type", "subject_kind",
        "subject_id", "object_kind", "object_id", "status", "source_ids",
        "authority", "confidence", "created_at", "observed_at", "valid_from",
        "valid_until", "supersedes_id", "superseded_at",
    } <= columns
    assert store.fetch_states(scope="legacy", concept="phrase")[0].value == (
        "Blue Lantern"
    )
    assert store.fetch_relationships() == []
    assert ContextStore(path).fetch_relationships() == []


def test_relationship_vocabulary_and_endpoint_matrix_are_closed():
    assert relationships_mod.RELATIONSHIP_TYPES == {
        "part_of", "concerns", "explains"
    }
    assert relationships_mod.RELATIONSHIP_ENDPOINTS == {
        "part_of": ("entity", "entity"),
        "concerns": ("memory_state", "entity"),
        "explains": ("memory_state", "memory_state"),
    }
    manager, store = _manager()
    entity = _active_entity(manager, store, "s", "Verse")
    state = _state(store, "s", "s", "phrase", "Blue Lantern")
    invalid = [
        ("part_of", "memory_state", state.state_id, "entity", entity.entity_id),
        ("part_of", "entity", entity.entity_id, "memory_state", state.state_id),
        ("concerns", "entity", entity.entity_id, "entity", entity.entity_id),
        ("concerns", "memory_state", state.state_id, "memory_state", state.state_id),
        ("explains", "memory_state", state.state_id, "entity", entity.entity_id),
        ("explains", "entity", entity.entity_id, "entity", entity.entity_id),
    ]
    for relationship_type, subject_kind, subject_id, object_kind, object_id in invalid:
        try:
            store.create_relationship(
                "s", "s", relationship_type, subject_kind, subject_id,
                object_kind, object_id, [],
            )
            assert False, "invalid endpoint matrix must be rejected"
        except ValueError as exc:
            assert "requires endpoints" in str(exc)
    relationship_id, diagnostics = manager.create_relationship_candidate(
        "s", "related_to", entity.entity_id, entity.entity_id, source_ids=[]
    )
    assert relationship_id is None
    assert any("relationship_type" in item for item in diagnostics)


def test_part_of_entity_to_entity_candidate_and_current():
    manager, store = _manager()
    verse = _active_entity(manager, store, "s", "Verse")
    primora = _active_entity(manager, store, "s", "PRIMORA", "PROJECT")
    candidate = _relationship_candidate(
        manager, store, "part_of", verse.entity_id, primora.entity_id
    )
    assert candidate.status == "candidate"
    assert candidate.subject_kind == "entity"
    assert candidate.object_kind == "entity"
    assert candidate.relationship_type == "part_of"
    assert store.fetch_relationships(status="current") == []
    current = _current_relationship(manager, candidate.relationship_id)
    assert current.status == "current"
    assert current.confidence == 1.0
    assert current.supersedes_id is None
    assert current.valid_until is None


def test_concerns_state_to_entity_candidate_and_current():
    manager, store = _manager()
    state = _state(store, "s", "s", "aurora visual", "frozen")
    entity = _active_entity(manager, store, "s", "Aurora Visual")
    candidate = _relationship_candidate(
        manager, store, "concerns", state.state_id, entity.entity_id
    )
    assert candidate.subject_kind == "memory_state"
    assert candidate.object_kind == "entity"
    current = _current_relationship(manager, candidate.relationship_id)
    assert current.status == "current"
    assert current.subject_id == state.state_id
    assert current.object_id == entity.entity_id


def test_explains_state_to_state_candidate_and_current():
    manager, store = _manager()
    rationale = _state(
        store, "s", "s", "rationale for decision",
        "it finally feels like Aurora", memory_type="RATIONALE",
    )
    decision = _state(
        store, "s", "s", "decision", "visual", memory_type="DECISION"
    )
    candidate = _relationship_candidate(
        manager, store, "explains", rationale.state_id, decision.state_id
    )
    assert candidate.subject_kind == "memory_state"
    assert candidate.object_kind == "memory_state"
    current = _current_relationship(manager, candidate.relationship_id)
    assert current.status == "current"
    assert current.subject_id == rationale.state_id
    assert current.object_id == decision.state_id


def test_candidate_requires_confirmation_and_provenance():
    manager, store = _manager()
    child = _active_entity(manager, store, "s", "Verse")
    parent = _active_entity(manager, store, "s", "PRIMORA", "PROJECT")
    without_source = _relationship_candidate(
        manager, store, "part_of", child.entity_id, parent.entity_id,
        source=False,
    )
    confirmed, diagnostics = manager.confirm_relationship(
        without_source.relationship_id
    )
    assert confirmed is None
    assert any("provenance" in item for item in diagnostics)
    assert store.fetch_relationship(
        without_source.relationship_id
    ).status == "candidate"
    candidate = _relationship_candidate(
        manager, store, "part_of", child.entity_id, parent.entity_id,
        timestamp=200,
    )
    confirmed = _current_relationship(manager, candidate.relationship_id)
    assert confirmed.source_ids
    assert confirmed.status == "current"


def test_only_user_authority_can_confirm_current_relationship():
    manager, store = _manager()
    child = _active_entity(manager, store, "s", "Verse")
    parent = _active_entity(manager, store, "s", "PRIMORA", "PROJECT")
    for authority in ("assistant", "tool", "system"):
        candidate = _relationship_candidate(
            manager, store, "part_of", child.entity_id, parent.entity_id,
            authority=authority, timestamp=200 + len(authority),
        )
        assert candidate.status == "candidate"
        confirmed, diagnostics = manager.confirm_relationship(
            candidate.relationship_id
        )
        assert confirmed is None
        assert any("not user" in item for item in diagnostics)
        assert store.fetch_relationship(
            candidate.relationship_id
        ).status == "candidate"


def test_confidence_is_metadata_not_truth():
    manager, store = _manager()
    child = _active_entity(manager, store, "s", "Verse")
    parent = _active_entity(manager, store, "s", "PRIMORA", "PROJECT")
    candidate = _relationship_candidate(
        manager, store, "part_of", child.entity_id, parent.entity_id
    )
    assert candidate.confidence == 0.3
    assert store.fetch_relationships(status="current") == []
    current = _current_relationship(manager, candidate.relationship_id)
    assert current.confidence == 1.0
    assert current.status == "current"


def test_invalid_endpoints_and_self_edges_are_rejected():
    manager, store = _manager()
    entity = _active_entity(manager, store, "s", "Verse")
    relationship_id, diagnostics = manager.create_relationship_candidate(
        "s", "part_of", 9999, entity.entity_id, source_ids=[]
    )
    assert relationship_id is None and any("not found" in item for item in diagnostics)
    relationship_id, diagnostics = manager.create_relationship_candidate(
        "s", "part_of", entity.entity_id, entity.entity_id, source_ids=[]
    )
    assert relationship_id is None
    assert any("different" in item for item in diagnostics)
    assert store.fetch_relationships() == []


def test_relationship_scope_must_match_endpoints():
    manager, store = _manager()
    verse = _active_entity(manager, store, "a", "Verse")
    primora = _active_entity(manager, store, "b", "PRIMORA", "PROJECT")
    relationship_id, diagnostics = manager.create_relationship_candidate(
        "a", "part_of", verse.entity_id, primora.entity_id, source_ids=[]
    )
    assert relationship_id is None
    assert any("scope" in item for item in diagnostics)
    assert store.fetch_relationships() == []


def test_session_relationships_are_isolated():
    manager, store = _manager()
    a_child = _active_entity(manager, store, "a", "Verse")
    a_parent = _active_entity(manager, store, "a", "PRIMORA", "PROJECT")
    b_child = _active_entity(manager, store, "b", "Verse")
    b_parent = _active_entity(manager, store, "b", "PRIMORA", "PROJECT")
    a_candidate = _relationship_candidate(
        manager, store, "part_of", a_child.entity_id, a_parent.entity_id,
        session_id="a",
    )
    b_candidate = _relationship_candidate(
        manager, store, "part_of", b_child.entity_id, b_parent.entity_id,
        session_id="b", timestamp=200,
    )
    a_current = _current_relationship(manager, a_candidate.relationship_id)
    b_current = _current_relationship(manager, b_candidate.relationship_id)
    assert store.fetch_relationships(scope="a", status="current") == [a_current]
    assert store.fetch_relationships(scope="b", status="current") == [b_current]
    assert store.fetch_relationships(scope="global") == []
    assert a_current.relationship_id != b_current.relationship_id


def test_explicit_global_relationship_preserves_session_provenance():
    manager, store = _manager()
    child = _active_entity(
        manager, store, "session-a", "Verse", scope="global"
    )
    parent = _active_entity(
        manager, store, "session-a", "PRIMORA", "PROJECT", scope="global"
    )
    candidate = _relationship_candidate(
        manager, store, "part_of", child.entity_id, parent.entity_id,
        session_id="session-a", scope="global",
    )
    current = _current_relationship(manager, candidate.relationship_id)
    assert current.scope == "global"
    assert current.session_id == "session-a"
    assert store.fetch_relationships(scope="global", status="current") == [current]
    assert store.fetch_relationships(scope="session-a") == []


def test_no_automatic_session_to_global_promotion():
    manager, store = _manager()
    child = _active_entity(manager, store, "s", "Verse")
    parent = _active_entity(manager, store, "s", "PRIMORA", "PROJECT")
    candidate = _relationship_candidate(
        manager, store, "part_of", child.entity_id, parent.entity_id
    )
    _current_relationship(manager, candidate.relationship_id)
    assert store.fetch_relationships(scope="global") == []
    assert all(
        row.scope == "s" for row in store.fetch_relationships()
    )


def test_relationship_timestamps_follow_m4_conventions():
    manager, store = _manager()
    child = _active_entity(manager, store, "s", "Verse")
    parent = _active_entity(manager, store, "s", "PRIMORA", "PROJECT")
    source_id = _append(store, "s", "relationship source", 100)
    relationship_id, diagnostics = manager.create_relationship_candidate(
        "s", "part_of", child.entity_id, parent.entity_id,
        source_ids=[source_id], timestamp=300,
    )
    assert relationship_id is not None, diagnostics
    candidate = store.fetch_relationship(relationship_id)
    assert candidate.created_at == 300
    assert candidate.observed_at == 100
    assert candidate.valid_from == 100
    assert candidate.valid_until is None
    assert candidate.superseded_at is None


def test_relationship_validity_is_half_open():
    manager, store = _manager()
    child = _active_entity(manager, store, "s", "Verse")
    parent = _active_entity(manager, store, "s", "PRIMORA", "PROJECT")
    candidate = _relationship_candidate(
        manager, store, "part_of", child.entity_id, parent.entity_id,
        valid_from=100, valid_until=200,
    )
    current = _current_relationship(manager, candidate.relationship_id)
    assert store.fetch_relationships(
        relationship_type="part_of", status="current", at_time=100
    ) == [current]
    assert store.fetch_relationships(
        relationship_type="part_of", status="current", at_time=199.999
    ) == [current]
    assert store.fetch_relationships(
        relationship_type="part_of", status="current", at_time=200
    ) == []
    relationship_id, diagnostics = manager.create_relationship_candidate(
        "s", "part_of", child.entity_id, parent.entity_id,
        source_ids=[], valid_from=200, valid_until=100,
    )
    assert relationship_id is None
    assert any("valid_until" in item for item in diagnostics)


def test_current_relationship_replacement_preserves_history_and_lineage():
    manager, store = _manager()
    verse = _active_entity(manager, store, "s", "Verse")
    primora = _active_entity(manager, store, "s", "PRIMORA", "PROJECT")
    first_candidate = _relationship_candidate(
        manager, store, "part_of", verse.entity_id, primora.entity_id,
        timestamp=100,
    )
    first = _current_relationship(
        manager, first_candidate.relationship_id, at_time=150
    )
    second_candidate = _relationship_candidate(
        manager, store, "part_of", verse.entity_id, primora.entity_id,
        timestamp=200,
    )
    second = _current_relationship(
        manager, second_candidate.relationship_id, at_time=400
    )
    old = store.fetch_relationship(first.relationship_id)
    assert old.status == "superseded"
    assert old.valid_until == 200
    assert old.superseded_at == 400
    assert old.source_ids == first.source_ids
    assert second.status == "current"
    assert second.valid_from == 200
    assert second.supersedes_id == first.relationship_id
    assert store.fetch_relationships(
        relationship_type="part_of", status="current"
    ) == [second]
    assert len(store.fetch_relationships(relationship_type="part_of")) == 2


def test_bare_supersession_creates_no_successor():
    manager, store = _manager()
    child = _active_entity(manager, store, "s", "Verse")
    parent = _active_entity(manager, store, "s", "PRIMORA", "PROJECT")
    candidate = _relationship_candidate(
        manager, store, "part_of", child.entity_id, parent.entity_id
    )
    current = _current_relationship(manager, candidate.relationship_id)
    superseded, diagnostics = manager.supersede_relationship(
        current.relationship_id, valid_until=300, at_time=400
    )
    assert superseded is not None, diagnostics
    assert superseded.status == "superseded"
    assert superseded.valid_until == 300
    assert superseded.superseded_at == 400
    assert len(store.fetch_relationships(relationship_type="part_of")) == 1


def test_conflict_lifecycle_and_rejection_on_confirmation():
    manager, store = _manager()
    child = _active_entity(manager, store, "s", "Verse")
    first_parent = _active_entity(manager, store, "s", "PRIMORA", "PROJECT")
    first_candidate = _relationship_candidate(
        manager, store, "part_of", child.entity_id, first_parent.entity_id,
        timestamp=100,
    )
    first = _current_relationship(manager, first_candidate.relationship_id)
    conflict_id, diagnostics = manager.record_relationship_conflict(
        "s", "part_of", child.entity_id, first_parent.entity_id,
        source_ids=[_append(store, "s", "assistant conflict", 150)],
    )
    assert conflict_id is not None, diagnostics
    conflict = store.fetch_relationship(conflict_id)
    assert conflict.status == "unresolved"
    assert store.fetch_relationship(first.relationship_id).status == "current"
    replacement_candidate = _relationship_candidate(
        manager, store, "part_of", child.entity_id, first_parent.entity_id,
        timestamp=200,
    )
    replacement = _current_relationship(
        manager, replacement_candidate.relationship_id, at_time=400
    )
    assert store.fetch_relationship(conflict_id).status == "rejected"
    assert store.fetch_relationship(first.relationship_id).status == "superseded"
    assert replacement.status == "current"
    assert replacement.supersedes_id == first.relationship_id


def test_current_relationship_can_be_marked_unresolved_and_candidate_rejected():
    manager, store = _manager()
    child = _active_entity(manager, store, "s", "Verse")
    parent = _active_entity(manager, store, "s", "PRIMORA", "PROJECT")
    current_candidate = _relationship_candidate(
        manager, store, "part_of", child.entity_id, parent.entity_id
    )
    current = _current_relationship(manager, current_candidate.relationship_id)
    unresolved, diagnostics = manager.mark_relationship_unresolved(
        current.relationship_id, at_time=300
    )
    assert unresolved is not None, diagnostics
    assert unresolved.status == "unresolved"
    assert unresolved.valid_until == 300
    assert unresolved.superseded_at is None
    assert store.fetch_relationships(status="current") == []
    candidate = _relationship_candidate(
        manager, store, "part_of", child.entity_id, parent.entity_id,
        timestamp=400,
    )
    rejected, diagnostics = manager.reject_relationship(
        candidate.relationship_id
    )
    assert rejected is not None, diagnostics
    assert rejected.status == "rejected"


def test_relationship_replacement_is_atomic():
    manager, store = _manager()
    child = _active_entity(manager, store, "s", "Verse")
    first_parent = _active_entity(manager, store, "s", "PRIMORA", "PROJECT")
    first_candidate = _relationship_candidate(
        manager, store, "part_of", child.entity_id, first_parent.entity_id
    )
    first = _current_relationship(manager, first_candidate.relationship_id)
    second_candidate = _relationship_candidate(
        manager, store, "part_of", child.entity_id, first_parent.entity_id,
        timestamp=200,
    )
    con = sqlite3.connect(store.db_path)
    con.executescript(
        f"""
        CREATE TRIGGER fail_relationship_supersede
        BEFORE UPDATE OF status ON relationships
        WHEN NEW.id = {first.relationship_id}
        BEGIN
            SELECT RAISE(ABORT, 'forced relationship failure');
        END;
        """
    )
    con.close()
    confirmed, diagnostics = manager.confirm_relationship(
        second_candidate.relationship_id, at_time=400
    )
    assert confirmed is None
    assert any("relationship_failed" in item for item in diagnostics)
    assert store.fetch_relationship(first.relationship_id).status == "current"
    assert store.fetch_relationship(
        second_candidate.relationship_id
    ).status == "candidate"
    assert store.fetch_relationships(status="current") == [first]


def test_no_inverse_relationship_duplication():
    manager, store = _manager()
    child = _active_entity(manager, store, "s", "Verse")
    parent = _active_entity(manager, store, "s", "PRIMORA", "PROJECT")
    candidate = _relationship_candidate(
        manager, store, "part_of", child.entity_id, parent.entity_id
    )
    current = _current_relationship(manager, candidate.relationship_id)
    rows = store.fetch_relationships()
    assert rows == [current]
    assert all(row.relationship_type == "part_of" for row in rows)
    assert not any(row.relationship_type == "contains" for row in rows)
    con = sqlite3.connect(store.db_path)
    schema_sql = " ".join(
        row[0] or "" for row in con.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table'"
        )
    )
    con.close()
    assert "contains" not in schema_sql
    assert "related_to" not in schema_sql


def test_narrow_part_of_extraction_creates_candidate_only():
    manager, store = _manager()
    verse = _active_entity(manager, store, "s", "Verse")
    primora = _active_entity(manager, store, "s", "PRIMORA", "PROJECT")
    source_id = _append(store, "s", "Verse is part of PRIMORA.", 100)
    relationship_id, diagnostics = manager.propose_relationship_from_text(
        "s", "Verse is part of PRIMORA.", source_ids=[source_id]
    )
    assert relationship_id is not None, diagnostics
    candidate = store.fetch_relationship(relationship_id)
    assert candidate.relationship_type == "part_of"
    assert candidate.subject_id == verse.entity_id
    assert candidate.object_id == primora.entity_id
    assert candidate.status == "candidate"
    assert store.fetch_relationships(status="current") == []
    relationship_id, diagnostics = manager.propose_relationship_from_text(
        "s", "Verse is friends with PRIMORA.", source_ids=[]
    )
    assert relationship_id is None
    assert any("pattern mismatch" in item for item in diagnostics)
    assert store.fetch_states() == []


def test_explains_bridge_is_canonical_and_preserves_m1_state():
    manager, store = _manager("explains-bridge.db")
    diagnostics = manager.reconcile([
        {
            "role": "user",
            "content": "We kept the visual because it finally feels like Aurora.",
            "timestamp": 100,
        }
    ], "s")
    assert not any(item.startswith("state_failed") for item in diagnostics)
    rationale = store.fetch_states(
        scope="global", concept="rationale for decision for visual",
        status="current"
    )[0]
    decision = store.fetch_states(
        scope="global", concept="decision for visual", status="current"
    )[0]
    assert rationale.memory_type == "RATIONALE"
    assert decision.memory_type == "DECISION"
    assert rationale.source_ids == decision.source_ids
    relationships = store.fetch_relationships(relationship_type="explains")
    assert len(relationships) == 1
    explains = relationships[0]
    assert explains.status == "current"
    assert explains.subject_kind == "memory_state"
    assert explains.subject_id == rationale.state_id
    assert explains.object_id == decision.state_id
    assert explains.source_ids == decision.source_ids
    assert explains.valid_from == decision.valid_from
    assert explains.confidence == 1.0


def test_explains_relationship_supersedes_with_old_rationale():
    manager, store = _manager("explains-supersede.db")
    history = [
        {
            "role": "user",
            "content": "We kept the visual because it finally feels like Aurora.",
            "timestamp": 100,
        }
    ]
    manager.reconcile(list(history), "s")
    old_decision_id = store.fetch_states(
        scope="global", concept="decision for visual", status="current"
    )[0].state_id
    old_rationale_id = store.fetch_states(
        scope="global", concept="rationale for decision for visual",
        status="current"
    )[0].state_id
    old_relationship = store.fetch_relationships(
        relationship_type="explains", status="current"
    )[0]
    history.append({
        "role": "user",
        "content": "We changed the decision to the layout.",
        "timestamp": 200,
    })
    manager.reconcile(list(history), "s")
    new_decision = store.fetch_states(
        scope="global", concept="decision for visual", status="current"
    )[0]
    old_rationale = [
        item for item in store.fetch_states(scope="global")
        if item.state_id == old_rationale_id
    ][0]
    superseded = store.fetch_relationship(old_relationship.relationship_id)
    old_decision = [
        item for item in store.fetch_states(scope="global")
        if item.state_id == old_decision_id
    ][0]
    assert old_decision.status == "superseded"
    assert old_rationale.status == "superseded"
    assert superseded.status == "superseded"
    assert superseded.valid_until == new_decision.valid_from == 200
    assert superseded.superseded_at is not None
    assert store.fetch_relationships(
        relationship_type="explains", status="current"
    ) == []


def test_relationship_rows_copy_no_state_or_message_text():
    manager, store = _manager()
    state = _state(store, "s", "s", "aurora visual", "frozen")
    entity = _active_entity(manager, store, "s", "Aurora Visual")
    candidate = _relationship_candidate(
        manager, store, "concerns", state.state_id, entity.entity_id
    )
    current = _current_relationship(manager, candidate.relationship_id)
    rendered = repr(current)
    assert "frozen" not in rendered
    assert "aurora visual" not in rendered
    con = sqlite3.connect(store.db_path)
    columns = {
        row[1] for row in con.execute("PRAGMA table_info(relationships)")
    }
    con.close()
    assert "concept" not in columns
    assert "value" not in columns
    assert "content" not in columns


def test_no_stage_c_apis_active_memory_working_or_retrieval_changes():
    manager, store = _manager()
    snapshot = [{"role": "user", "content": "My interface is Verse."}]
    manager.reconcile(list(snapshot), "s")
    evidence_id = store.fetch_latest("s", 1)[0].message_id

    child_id, diagnostics = manager.create_entity_candidate(
        "s", "Verse", "COMPONENT", source_ids=[evidence_id], authority="user"
    )
    assert child_id is not None, diagnostics
    manager.confirm_entity(child_id)
    parent_id, diagnostics = manager.create_entity_candidate(
        "s", "PRIMORA", "PROJECT",
        source_ids=[evidence_id], authority="user",
    )
    assert parent_id is not None, diagnostics
    manager.confirm_entity(parent_id)
    relationship_id, diagnostics = manager.create_relationship_candidate(
        "s", "part_of", child_id, parent_id,
        source_ids=[evidence_id], authority="user", timestamp=105,
    )
    assert relationship_id is not None, diagnostics

    before_active = [
        (item.kind, item.content)
        for item in manager.get_active_memory("s", at_time=200).items
    ]
    before_build = manager.build(
        list(snapshot), "s", token_counter=_counter,
        max_new_tokens=5, budget=_budget(),
    ).view
    before_states = store.fetch_states(scope="s")

    confirmed = manager.confirm_relationship(relationship_id, at_time=200)
    assert confirmed is not None, diagnostics

    after_active = [
        (item.kind, item.content)
        for item in manager.get_active_memory("s", at_time=200).items
    ]
    after_build = manager.build(
        list(snapshot), "s", token_counter=_counter,
        max_new_tokens=5, budget=_budget(),
    ).view
    assert after_active == before_active
    assert after_build.messages == before_build.messages
    assert after_build.input_tokens == before_build.input_tokens
    assert store.fetch_states(scope="s") == before_states
    wm = WorkingMemory("s", task="relationship check")
    assert not hasattr(wm, "_store")
    # Writes never expose a traversal or graph API: every such read lives in
    # the Stage C query layer, on a separate module.
    for name in (
        "relationships_from", "relationships_to", "entity_memories",
        "explanations", "relationship_context", "contains",
        "walk_graph", "shortest_path", "connected_components", "related_to",
    ):
        assert not hasattr(store, name)
        assert not hasattr(manager, name)
        assert not hasattr(relationships_mod, name)
    assert all(
        not str(item.get("content", "")).startswith(
            ("[entity", "[relationship", "[part_of")
        )
        for item in after_build.messages
    )


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print("All M5 Stage B relationship tests passed!")
