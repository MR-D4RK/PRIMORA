import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import ContextManager, WorkingMemory
from PRIMORA.prime.context import entities as entities_mod
from PRIMORA.prime.context import state as state_mod
from PRIMORA.prime.context.store import ContextStore

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]


def _db(name=None):
    _SEQ[0] += 1
    filename = name or f"entities-{_SEQ[0]}.db"
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


def _active(
    manager,
    store,
    session_id,
    display_name,
    entity_type="COMPONENT",
    scope=None,
    authority="user",
    timestamp=100,
    source=True,
):
    source_ids = []
    if source:
        source_ids = [_append(
            store, session_id, f"identity evidence {display_name}", timestamp
        )]
    entity_id, diagnostics = manager.create_entity_candidate(
        session_id,
        display_name,
        entity_type,
        source_ids=source_ids,
        authority=authority,
        scope=scope,
        timestamp=timestamp + 10,
    )
    assert entity_id is not None, diagnostics
    entity, diagnostics = manager.confirm_entity(entity_id)
    assert entity is not None, diagnostics
    return entity


def _current_alias(
    manager,
    store,
    entity,
    alias_text,
    session_id,
    timestamp=100,
    valid_from=None,
    valid_until=None,
):
    source_id = _append(
        store, session_id, f"alias evidence {alias_text}", timestamp
    )
    alias_id, diagnostics = manager.add_alias_candidate(
        entity.entity_id,
        alias_text,
        session_id,
        source_ids=[source_id],
        timestamp=timestamp + 10,
        valid_from=valid_from,
        valid_until=valid_until,
    )
    assert alias_id is not None, diagnostics
    alias, diagnostics = manager.confirm_alias(
        alias_id, valid_from=valid_from, valid_until=valid_until
    )
    assert alias is not None, diagnostics
    return alias


def test_migration_creates_identity_tables_without_touching_legacy_state():
    path = _db("legacy-identity.db")
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
    entity_columns = {
        row[1] for row in con.execute("PRAGMA table_info(entities)")
    }
    alias_columns = {
        row[1] for row in con.execute("PRAGMA table_info(entity_aliases)")
    }
    con.close()

    assert {"entities", "entity_aliases"} <= tables
    assert {
        "id", "session_id", "scope", "entity_type", "display_name",
        "normalized_name", "status", "merged_into_entity_id", "source_ids",
        "authority", "confidence", "created_at", "observed_at", "merged_at",
    } <= entity_columns
    assert {
        "id", "entity_id", "session_id", "scope", "alias_text",
        "normalized_alias", "status", "source_ids", "authority",
        "confidence", "created_at", "observed_at", "valid_from",
        "valid_until", "superseded_at",
    } <= alias_columns
    state = store.fetch_states(scope="legacy", concept="phrase")[0]
    assert state.value == "Blue Lantern"
    assert state.observed_at == 100 and state.valid_from == 100
    assert store.fetch_entities() == []
    assert store.fetch_entity_aliases() == []

    reopened = ContextStore(path)
    assert reopened.fetch_states(scope="legacy", concept="phrase")[0].value == (
        "Blue Lantern"
    )
    assert reopened.fetch_entities() == []


def test_entity_candidate_creation_preserves_provenance_and_timestamps():
    manager, store = _manager()
    source_id = _append(store, "s", "Verse identity evidence", 100)
    entity_id, diagnostics = manager.create_entity_candidate(
        "s",
        "Verse",
        "COMPONENT",
        source_ids=[source_id],
        timestamp=300,
    )
    assert entity_id is not None, diagnostics
    entity = store.fetch_entity(entity_id)
    assert entity.status == "candidate"
    assert entity.session_id == "s" and entity.scope == "s"
    assert entity.entity_type == "COMPONENT"
    assert entity.display_name == "Verse"
    assert entity.normalized_name == "verse"
    assert entity.source_ids == [source_id]
    assert entity.authority == "user" and entity.confidence == 0.3
    assert entity.created_at == 300 and entity.observed_at == 100
    assert entity.merged_into_entity_id is None and entity.merged_at is None


def test_normalization_is_exact_case_whitespace_and_safe_punctuation():
    assert entities_mod.normalize_entity_name("Verse") == "verse"
    assert entities_mod.normalize_entity_name("  PRIMORA   Project ") == (
        "primora project"
    )
    assert entities_mod.normalize_entity_name("The Verse") == "the verse"
    assert entities_mod.normalize_entity_name("manager.py") == "manager.py"
    assert entities_mod.normalize_entity_name(r"C:\PRIMORA\prime") == (
        r"c:\primora\prime"
    )
    assert entities_mod.normalize_entity_name("AURORA’s") == "aurora's"
    assert entities_mod.normalize_entity_name("“Verse”") == "verse"
    assert entities_mod.normalize_entity_name("Verse.") == "verse"
    assert entities_mod.normalize_entity_name("A—B") == "a-b"
    assert entities_mod.normalize_entity_name("Straße") == "strasse"
    assert entities_mod.normalize_entity_name("   ") == ""
    assert entities_mod.normalize_entity_name("...") == ""
    assert state_mod.normalize_concept("The Verse") == "verse"
    assert entities_mod.normalize_entity_name("The Verse") != (
        state_mod.normalize_concept("The Verse")
    )


def test_entity_confirmation_requires_user_authority_and_provenance():
    manager, store = _manager()
    for authority in ("assistant", "tool", "system"):
        source_id = _append(store, "s", f"{authority} identity evidence", 100)
        entity_id, diagnostics = manager.create_entity_candidate(
            "s",
            f"{authority} candidate",
            "AGENT",
            source_ids=[source_id],
            authority=authority,
        )
        assert entity_id is not None, diagnostics
        confirmed, diagnostics = manager.confirm_entity(entity_id)
        assert confirmed is None
        assert any("entity_failed" in item for item in diagnostics)
        assert store.fetch_entity(entity_id).status == "candidate"

    no_source_id, diagnostics = manager.create_entity_candidate(
        "s", "No Source", "PERSON", source_ids=[]
    )
    assert no_source_id is not None, diagnostics
    confirmed, diagnostics = manager.confirm_entity(no_source_id)
    assert confirmed is None and any("provenance" in item for item in diagnostics)

    source_id = _append(store, "s", "user identity evidence", 100)
    try:
        store.create_entity(
            "s", "s", "AGENT", "Assistant Active", "assistant active",
            [source_id], status="active", authority="assistant",
        )
        assert False, "non-user authority must not create active identity"
    except ValueError as exc:
        assert "only user-authored" in str(exc)

    entity = _active(manager, store, "s", "Confirmed User", "PERSON")
    assert entity.status == "active" and entity.confidence == 1.0


def test_confidence_does_not_change_identity_truth():
    manager, store = _manager()
    candidate_source = _append(store, "s", "candidate evidence", 100)
    candidate_id = store.create_entity(
        "s", "s", "AGENT", "Candidate", "candidate", [candidate_source],
        status="candidate", authority="user", confidence=1.0,
    )
    assert manager.resolve_entity("s", "Candidate") == ([], False)

    active_source = _append(store, "s", "active evidence", 100)
    active_id = store.create_entity(
        "s", "s", "AGENT", "Low Confidence", "low confidence",
        [active_source], status="active", authority="user", confidence=0.3,
    )
    resolved, ambiguous = manager.resolve_entity("s", "Low Confidence")
    assert not ambiguous
    assert [entity.entity_id for entity in resolved] == [active_id]
    assert resolved[0].confidence == 0.3
    assert store.fetch_entity(candidate_id).confidence == 1.0


def test_exact_canonical_resolution_is_scope_explicit():
    manager, store = _manager()
    entity = _active(manager, store, "s", "Verse")
    resolved, ambiguous = manager.resolve_entity("s", "  VERSE. ")
    assert not ambiguous
    assert [row.entity_id for row in resolved] == [entity.entity_id]
    assert manager.resolve_entity("s", "the Verse") == ([], False)
    assert manager.resolve_entity("global", "Verse") == ([], False)
    assert manager.get_entity("wrong-scope", entity.entity_id) is None


def test_exact_alias_resolution_requires_confirmation():
    manager, store = _manager()
    entity = _active(manager, store, "s", "Verse")
    source_id = _append(store, "s", "the Verse interface evidence", 100)
    alias_id, diagnostics = manager.add_alias_candidate(
        entity.entity_id,
        "the Verse interface",
        "s",
        source_ids=[source_id],
    )
    assert alias_id is not None, diagnostics
    assert manager.resolve_entity("s", "The Verse Interface") == ([], False)
    alias, diagnostics = manager.confirm_alias(alias_id)
    assert alias is not None, diagnostics
    assert alias.status == "current" and alias.confidence == 1.0
    resolved, ambiguous = manager.resolve_entity("s", "The Verse Interface")
    assert not ambiguous
    assert [row.entity_id for row in resolved] == [entity.entity_id]
    listed = manager.list_aliases("s", entity_id=entity.entity_id)
    assert [row.alias_id for row in listed] == [alias_id]


def test_alias_scope_must_match_entity_scope():
    manager, store = _manager()
    entity = _active(manager, store, "s", "Verse")
    source_id = _append(store, "s", "alias evidence", 100)
    alias_id, diagnostics = manager.add_alias_candidate(
        entity.entity_id,
        "Verse UI",
        "s",
        source_ids=[source_id],
        scope="global",
    )
    assert alias_id is None and any("scope" in item for item in diagnostics)
    assert manager.list_aliases("global") == []
    alias_id, diagnostics = manager.add_alias_candidate(
        entity.entity_id, "Verse UI", "s", source_ids=[source_id]
    )
    assert alias_id is not None, diagnostics
    assert manager.list_aliases("s", entity_id=entity.entity_id)[0].scope == "s"


def test_ambiguous_resolution_returns_candidates_without_merging():
    manager, store = _manager()
    first = _active(manager, store, "s", "Verse", "COMPONENT")
    second = _active(manager, store, "s", "verse.", "PROJECT")
    resolved, ambiguous = manager.resolve_entity("s", "VERSE")
    assert ambiguous
    assert {row.entity_id for row in resolved} == {
        first.entity_id, second.entity_id
    }
    assert store.fetch_entity(first.entity_id).status == "active"
    assert store.fetch_entity(second.entity_id).status == "active"
    assert store.fetch_entity(first.entity_id).merged_into_entity_id is None


def test_candidate_alias_candidate_entity_and_archived_do_not_resolve():
    manager, store = _manager()
    candidate_source = _append(store, "s", "candidate evidence", 100)
    candidate_id, diagnostics = manager.create_entity_candidate(
        "s", "Candidate", "AGENT", source_ids=[candidate_source]
    )
    assert candidate_id is not None, diagnostics
    assert manager.resolve_entity("s", "Candidate") == ([], False)

    archived = _active(manager, store, "s", "Archived")
    archived, diagnostics = manager.archive_entity(archived.entity_id)
    assert archived is not None and archived.status == "archived"
    assert manager.get_entity("s", archived.entity_id).entity_id == (
        archived.entity_id
    )
    assert manager.resolve_entity("s", "Archived") == ([], False)


def test_session_isolation_and_no_automatic_cross_session_merge():
    manager, store = _manager()
    first = _active(manager, store, "a", "Verse")
    second = _active(manager, store, "b", "Verse")
    first_resolved, first_ambiguous = manager.resolve_entity("a", "Verse")
    second_resolved, second_ambiguous = manager.resolve_entity("b", "Verse")
    assert not first_ambiguous and not second_ambiguous
    assert [row.entity_id for row in first_resolved] == [first.entity_id]
    assert [row.entity_id for row in second_resolved] == [second.entity_id]
    assert all(
        row.status != "merged" and row.merged_into_entity_id is None
        for row in store.fetch_entities()
    )
    assert manager.resolve_entity("a", "Verse") != (
        manager.resolve_entity("b", "Verse")
    )


def test_explicit_global_identity_preserves_originating_session():
    manager, store = _manager()
    entity = _active(manager, store, "session-a", "PRIMORA", "PROJECT", "global")
    assert entity.scope == "global" and entity.session_id == "session-a"
    resolved, ambiguous = manager.resolve_entity("global", "PRIMORA")
    assert not ambiguous and [row.entity_id for row in resolved] == [entity.entity_id]
    assert manager.resolve_entity("session-a", "PRIMORA") == ([], False)
    other_source = _append(store, "session-b", "global provenance", 200)
    other_id, diagnostics = manager.create_entity_candidate(
        "session-b", "Aurora", "AGENT", source_ids=[other_source], scope="global"
    )
    assert other_id is not None, diagnostics
    other, diagnostics = manager.confirm_entity(other_id)
    assert other is not None, diagnostics
    assert other.session_id == "session-b" and other.scope == "global"


def test_no_automatic_session_to_global_promotion():
    manager, store = _manager()
    entity = _active(manager, store, "s", "Verse")
    _current_alias(manager, store, entity, "Verse UI", "s")
    assert manager.list_aliases("global") == []
    assert all(row.scope == "s" for row in store.fetch_entities())
    assert all(row.scope == "s" for row in store.fetch_entity_aliases())
    assert store.fetch_entities(scope="global") == []


def test_archive_preserves_row_provenance_and_aliases():
    manager, store = _manager()
    entity = _active(manager, store, "s", "Verse")
    alias = _current_alias(manager, store, entity, "Verse UI", "s")
    archived, diagnostics = manager.archive_entity(entity.entity_id)
    assert archived is not None, diagnostics
    assert archived.source_ids == entity.source_ids
    assert archived.created_at == entity.created_at
    assert archived.observed_at == entity.observed_at
    assert manager.get_entity("s", entity.entity_id).status == "archived"
    assert [
        row.alias_id for row in manager.list_aliases("s", entity_id=entity.entity_id)
    ] == [alias.alias_id]
    assert manager.resolve_entity("s", "Verse") == ([], False)
    assert manager.resolve_entity("s", "Verse UI") == ([], False)


def test_merge_preserves_loser_redirects_and_aliases():
    manager, store = _manager()
    losing = _active(manager, store, "s", "Verse UI")
    canonical = _active(manager, store, "s", "Verse")
    alias = _current_alias(manager, store, losing, "Verse interface", "s")
    before_ids = {row.entity_id for row in store.fetch_entities()}
    merged, diagnostics = manager.merge_entity(
        losing.entity_id, canonical.entity_id, at_time=400
    )
    assert merged is not None, diagnostics
    assert merged.status == "merged"
    assert merged.merged_into_entity_id == canonical.entity_id
    assert merged.merged_at == 400
    assert merged.source_ids == losing.source_ids
    assert {row.entity_id for row in store.fetch_entities()} == before_ids
    assert manager.get_entity("s", losing.entity_id).status == "merged"
    assert manager.get_entity(
        "s", losing.entity_id, follow_merge=True
    ).entity_id == canonical.entity_id
    assert manager.resolve_canonical_entity(
        "s", losing.entity_id
    ).entity_id == canonical.entity_id
    resolved, ambiguous = manager.resolve_entity("s", "Verse UI")
    assert not ambiguous and [row.entity_id for row in resolved] == [
        canonical.entity_id
    ]
    resolved, ambiguous = manager.resolve_entity("s", "Verse interface")
    assert not ambiguous and [row.entity_id for row in resolved] == [
        canonical.entity_id
    ]
    assert store.fetch_entity_alias(alias.alias_id).entity_id == losing.entity_id


def test_merge_chain_resolution_respects_transition_time():
    manager, store = _manager()
    first = _active(manager, store, "s", "First")
    second = _active(manager, store, "s", "Second")
    third = _active(manager, store, "s", "Third")
    merged, diagnostics = manager.merge_entity(
        first.entity_id, second.entity_id, at_time=300
    )
    assert merged is not None, diagnostics
    merged, diagnostics = manager.merge_entity(
        second.entity_id, third.entity_id, at_time=400
    )
    assert merged is not None, diagnostics
    assert manager.resolve_canonical_entity(
        "s", first.entity_id
    ).entity_id == third.entity_id
    assert manager.resolve_canonical_entity(
        "s", first.entity_id, at_time=250
    ).entity_id == first.entity_id
    assert manager.resolve_canonical_entity(
        "s", first.entity_id, at_time=350
    ).entity_id == second.entity_id


def test_merge_rejects_non_user_and_cross_scope_targets():
    manager, store = _manager()
    first = _active(manager, store, "a", "Verse")
    second = _active(manager, store, "a", "Verse UI")
    other = _active(manager, store, "b", "Other Verse")
    merged, diagnostics = manager.merge_entity(
        first.entity_id, second.entity_id, authority="assistant"
    )
    assert merged is None and any("user" in item for item in diagnostics)
    merged, diagnostics = manager.merge_entity(
        first.entity_id, other.entity_id
    )
    assert merged is None and any("scope" in item for item in diagnostics)
    assert store.fetch_entity(first.entity_id).status == "active"
    assert store.fetch_entity(second.entity_id).status == "active"
    assert store.fetch_entity(other.entity_id).status == "active"


def test_alias_validity_is_half_open_and_invalid_interval_is_rejected():
    manager, store = _manager()
    entity = _active(manager, store, "s", "Verse")
    alias = _current_alias(
        manager, store, entity, "Verse UI", "s",
        valid_from=100, valid_until=200,
    )
    assert alias.valid_from == 100 and alias.valid_until == 200
    assert manager.resolve_entity("s", "Verse UI", at_time=100)[0]
    assert manager.resolve_entity("s", "Verse UI", at_time=199.999)[0]
    assert manager.resolve_entity("s", "Verse UI", at_time=200) == ([], False)
    source_id = _append(store, "s", "invalid alias evidence", 300)
    alias_id, diagnostics = manager.add_alias_candidate(
        entity.entity_id, "Invalid Alias", "s", source_ids=[source_id],
        valid_from=300, valid_until=200,
    )
    assert alias_id is None and any("valid_until" in item for item in diagnostics)
    assert manager.list_aliases("s", status="candidate") == []


def test_resolve_concept_regression_is_not_entity_resolution():
    manager, store = _manager("resolve-concept-regression.db")
    diagnostics = manager.reconcile([
        {"role": "user", "content": "My interface is Verse.", "timestamp": 100},
        {"role": "user", "content": "The Verse interface is frozen.", "timestamp": 200},
    ], "s")
    assert not any(item.startswith("state_failed") for item in diagnostics)
    current = store.fetch_states(
        scope="s", concept="interface", status="current"
    )[0]
    assert current.value == "frozen"
    history = store.fetch_states(scope="s", concept="interface")
    assert {row.value for row in history} == {"Verse", "frozen"}
    assert state_mod.resolve_concept(["interface"], "verse interface") == "interface"

    entity = _active(manager, store, "s", "interface")
    assert entity.normalized_name == "interface"
    assert manager.resolve_entity("s", "verse interface") == ([], False)
    assert manager.resolve_entity("s", "interface")[0][0].entity_id == (
        entity.entity_id
    )
    assert store.fetch_entities(scope="s")[0].entity_id == entity.entity_id


def test_no_generic_predicates_or_unrestricted_traversal_surface():
    manager, store = _manager()
    con = sqlite3.connect(store.db_path)
    schema_sql = " ".join(
        row[0] or "" for row in con.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table'"
        )
    )
    columns = {row[1] for row in con.execute("PRAGMA table_info(relationships)")}
    con.close()
    for banned in ("related_to", "contains", "depends_on", "works_on",
                   "owns", "located_in", "friend_of", "created_by",
                   "decided_by"):
        assert banned not in schema_sql
        assert banned not in columns
    # No graph-walking surface exists on any M5 layer.
    for name in ("walk", "walk_graph", "shortest_path", "connected_components",
                 "neighbourhood", "subgraph", "multi_hop", "traverse",
                 "relationships_from", "relationships_to", "entity_memories",
                 "explanations", "relationship_context", "contains"):
        assert not hasattr(store, name)
        assert not hasattr(manager, name)
        assert not hasattr(entities_mod, name)


def test_entity_writes_do_not_change_active_or_working_memory():
    manager, store = _manager()
    diagnostics = manager.reconcile([
        {"role": "user", "content": "My interface is Verse.", "timestamp": 100}
    ], "s")
    assert not any(item.startswith("state_failed") for item in diagnostics)
    before = manager.get_active_memory("s", at_time=200)
    before_items = [(item.kind, item.content) for item in before.items]
    state_before = manager._get_store().fetch_states(
        scope="s", concept="interface"
    )
    entity = _active(manager, store, "s", "Verse")
    _current_alias(manager, store, entity, "Verse UI", "s")
    after = manager.get_active_memory("s", at_time=200)
    assert [(item.kind, item.content) for item in after.items] == before_items
    assert all(item.kind in ("state", "session", "digest") for item in after.items)
    assert manager._get_store().fetch_states(
        scope="s", concept="interface"
    ) == state_before
    wm = WorkingMemory("s", task="identity check")
    assert not hasattr(wm, "_store")
    assert wm.to_messages() == [{"role": "system", "content": "[working task] identity check"}]


def test_entity_types_and_normalization_rejects_semantic_aliases():
    manager, store = _manager()
    assert entities_mod.ENTITY_TYPES == {
        "PERSON", "AGENT", "PROJECT", "COMPONENT", "FILE"
    }
    entity_id, diagnostics = manager.create_entity_candidate(
        "s", "Bad Type", "CONCEPT", source_ids=[]
    )
    assert entity_id is None and any("entity_type" in item for item in diagnostics)
    entity = _active(manager, store, "s", "Verse")
    source_id = _append(store, "s", "semantic alias evidence", 100)
    alias_id, diagnostics = manager.add_alias_candidate(
        entity.entity_id, "Aurora's world", "s", source_ids=[source_id]
    )
    assert alias_id is not None, diagnostics
    assert manager.resolve_entity("s", "Aurora's world") == ([], False)
    assert manager.resolve_entity("s", "Verse") [0][0].entity_id == entity.entity_id


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print("All M5 Stage A entity identity tests passed!")
