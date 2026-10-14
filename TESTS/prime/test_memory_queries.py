import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import BudgetConfig, ContextManager, WorkingMemory
from PRIMORA.prime.context import queries as queries_mod
from PRIMORA.prime.context.store import ContextStore

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]


def _db(name=None):
    _SEQ[0] += 1
    filename = name or f"queries-{_SEQ[0]}.db"
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
    source_id = _append(store, session_id, f"evidence for {display_name}",
                        timestamp)
    entity_id, diagnostics = manager.create_entity_candidate(
        session_id, display_name, entity_type, source_ids=[source_id],
        scope=scope, timestamp=timestamp + 5,
    )
    assert entity_id is not None, diagnostics
    entity, diagnostics = manager.confirm_entity(entity_id)
    assert entity is not None, diagnostics
    return entity


def _state(
    store, session_id, scope, concept, value, timestamp=100,
    memory_type="FACT",
):
    source_id = _append(store, session_id, f"evidence {concept}", timestamp)
    store.create_state(
        session_id, scope, concept, value, [source_id], status="current",
        authority="user", timestamp=timestamp, memory_type=memory_type,
    )
    return store.fetch_states(
        scope=scope, concept=concept, status="current"
    )[-1]


def _link(
    manager, store, relationship_type, subject_id, object_id,
    session_id="s", scope=None, timestamp=100, confirm=True,
    valid_from=None, valid_until=None, at_time=None,
):
    source_id = _append(
        store, session_id, f"evidence {relationship_type}", timestamp
    )
    relationship_id, diagnostics = manager.create_relationship_candidate(
        session_id, relationship_type, subject_id, object_id,
        source_ids=[source_id], scope=scope, timestamp=timestamp + 5,
        valid_from=valid_from, valid_until=valid_until,
    )
    assert relationship_id is not None, diagnostics
    if not confirm:
        return store.fetch_relationship(relationship_id)
    relationship, diagnostics = manager.confirm_relationship(
        relationship_id, at_time=at_time
    )
    assert relationship is not None, diagnostics
    return relationship


def _counter(messages):
    return sum(len(message.get("content", "").split()) + 2
               for message in messages)


def _budget():
    return BudgetConfig(
        input_target=220, input_ceiling=400, output_reserve=20,
        total_budget=420,
    )


def _apollo(manager, store, session_id="s", scope=None, timestamp=100):
    """Real-world graph: Mobile App part_of Project Apollo, settings.py
    part_of Mobile App, a decision that concerns the app, and a rationale
    that explains that decision."""
    app = _active_entity(
        manager, store, session_id, "Mobile App", "COMPONENT", scope,
        timestamp,
    )
    apollo = _active_entity(
        manager, store, session_id, "Project Apollo", "PROJECT", scope,
        timestamp + 1,
    )
    settings = _active_entity(
        manager, store, session_id, "settings.py", "FILE", scope, timestamp + 2,
    )
    app_in_apollo = _link(
        manager, store, "part_of", app.entity_id, apollo.entity_id,
        session_id, scope, timestamp + 10,
    )
    settings_in_app = _link(
        manager, store, "part_of", settings.entity_id, app.entity_id,
        session_id, scope, timestamp + 11,
    )
    decision = _state(
        store, session_id, scope or session_id, "decision",
        "ship the mobile app on friday", timestamp + 20, "DECISION",
    )
    concerns = _link(
        manager, store, "concerns", decision.state_id, app.entity_id,
        session_id, scope, timestamp + 21,
    )
    rationale = _state(
        store, session_id, scope or session_id, "decision rationale",
        "the app keeps the aurora look fast", timestamp + 30, "RATIONALE",
    )
    explains = _link(
        manager, store, "explains", rationale.state_id, decision.state_id,
        session_id, scope, timestamp + 31,
    )
    return {
        "app": app, "apollo": apollo, "settings": settings,
        "app_in_apollo": app_in_apollo, "settings_in_app": settings_in_app,
        "decision": decision, "concerns": concerns,
        "rationale": rationale, "explains": explains,
    }


# ----------------------------------------------------------------------
# Entity lookup, alias lookup, resolution
# ----------------------------------------------------------------------
def test_exact_entity_lookup_by_id_and_canonical_name():
    manager, store = _manager()
    graph = _apollo(manager, store)
    app = graph["app"]
    assert manager.get_entity("s", app.entity_id).entity_id == app.entity_id
    resolved, ambiguous = manager.resolve_entity("s", "Mobile App")
    assert not ambiguous
    assert [row.entity_id for row in resolved] == [app.entity_id]
    assert manager.get_entity("s", app.entity_id, at_time=200) is not None
    # Case/whitespace normalization is deterministic and exact.
    assert manager.resolve_entity("s", "  mobile app ")[0][0] == app
    # No semantic inference: containment never implies identity.
    assert manager.resolve_entity("s", "Mobile") == ([], False)
    assert manager.resolve_entity("s", "the Apollo project") == ([], False)
    assert manager.get_entity("other-session", app.entity_id) is None


def test_exact_alias_lookup_is_scoped_and_temporal():
    manager, store = _manager()
    app = _active_entity(manager, store, "s", "Mobile App")
    source_id = _append(store, "s", "call it the app", 150)
    alias_id, diagnostics = manager.add_alias_candidate(
        app.entity_id, "the Apollo phone app", "s", source_ids=[source_id],
        valid_from=150, valid_until=400,
    )
    assert alias_id is not None, diagnostics
    candidate = manager.list_aliases("s", status="candidate")
    assert [row.alias_text for row in candidate] == ["the Apollo phone app"]
    assert manager.resolve_entity("s", "the Apollo phone app") == ([], False)
    alias, diagnostics = manager.confirm_alias(alias_id)
    assert alias is not None, diagnostics
    assert manager.resolve_entity("s", "the Apollo phone app",
                                  at_time=200)[0][0] == app
    found = manager.list_aliases("s", alias_text="the Apollo phone app")
    assert [row.entity_id for row in found] == [app.entity_id]
    assert found[0].status == "current"
    assert found[0].source_ids == [source_id]
    # Half-open alias validity, exactly as M4 states.
    assert manager.list_aliases(
        "s", alias_text="the Apollo phone app", at_time=399.999
    )
    assert manager.list_aliases(
        "s", alias_text="the Apollo phone app", at_time=400
    ) == []
    # A session alias never leaks into another session.
    assert manager.list_aliases("s2", alias_text="the Apollo phone app") == []


def test_ambiguous_entity_resolution_stays_ambiguous():
    manager, store = _manager()
    first = _active_entity(manager, store, "s", "Collection")
    store.create_entity_alias(
        first.entity_id, "s", "s", "Collection", "collection",
        [1], status="current", authority="user", confidence=1.0,
        timestamp=100, observed_at=100, valid_from=100, valid_until=None,
    )
    second = _active_entity(manager, store, "s", "Collection", timestamp=200)
    resolved, ambiguous = manager.resolve_entity("s", "Collection")
    assert ambiguous is True
    assert {row.entity_id for row in resolved} == {
        first.entity_id, second.entity_id
    }
    # Ambiguity is never silently resolved by picking the first row.
    assert len(manager.get_related_entities("s", first.entity_id)) == 0


def test_merged_entity_resolves_to_canonical_target():
    manager, store = _manager()
    graph = _apollo(manager, store)
    app = graph["app"]
    losing = _active_entity(manager, store, "s", "Apollo Phone", timestamp=300)
    merged, diagnostics = manager.merge_entity(
        losing.entity_id, app.entity_id, at_time=400
    )
    assert merged is not None, diagnostics
    assert merged.merged_into_entity_id == app.entity_id
    resolved, ambiguous = manager.resolve_entity("s", "Apollo Phone")
    assert not ambiguous
    assert resolved[0].entity_id == app.entity_id
    assert manager.get_entity(
        "s", losing.entity_id, follow_merge=True
    ).entity_id == app.entity_id
    # Traversal from the merged id follows the canonical target.
    related = manager.get_related_entities("s", losing.entity_id)
    assert {row.entity.entity_id for row in related} == {
        graph["apollo"].entity_id, graph["settings"].entity_id
    }
    # An id that is not a memory state cannot start an explanation path,
    # even when a stored `explains` row happens to point at that number.
    assert manager.get_relationship_explanation(
        "s", graph["settings"].entity_id
    ) == []


# ----------------------------------------------------------------------
# Scope
# ----------------------------------------------------------------------
def test_session_scope_isolation_and_explicit_global():
    manager, store = _manager()
    mine = _apollo(manager, store, "s")
    theirs = _apollo(manager, store, "t")
    global_graph = _apollo(manager, store, "g", scope="global")

    mine_rows = manager.get_entity_relationships(
        "s", mine["app"].entity_id
    )
    assert [row.relationship_id
            for row in mine_rows] == [mine["app_in_apollo"].relationship_id]
    assert all(row.scope == "s" for row in mine_rows)

    # No automatic foreign-session traversal.
    assert manager.get_related_entities(
        "s", theirs["app"].entity_id
    ) == []
    # include_global adds global scope; it never adds another session.
    with_global = manager.get_related_entities(
        "s", mine["app"].entity_id, include_global=True
    )
    assert {row.relationship.relationship_id
            for row in with_global} == {
                mine["app_in_apollo"].relationship_id,
                mine["settings_in_app"].relationship_id,
            }
    assert all(row.relationship.scope == "s" for row in with_global)
    assert all(row.entity.scope == "s" for row in with_global)
    # Global rows are only visible when global scope is asked for.
    global_rows = manager.get_entity_relationships(
        "global", global_graph["app"].entity_id
    )
    assert [row.relationship_id
            for row in global_rows] == [
                global_graph["app_in_apollo"].relationship_id
            ]
    # Same display name in another scope is a separate identity.
    assert mine["app"].entity_id != theirs["app"].entity_id
    assert mine["app"].normalized_name == theirs["app"].normalized_name
    assert manager.get_entity("s", global_graph["app"].entity_id) is None


# ----------------------------------------------------------------------
# One-hop traversal, limits, derived inverse
# ----------------------------------------------------------------------
def test_one_hop_outbound_and_inbound_part_of():
    manager, store = _manager()
    graph = _apollo(manager, store)
    app = graph["app"]
    outbound = manager.get_entity_relationships("s", app.entity_id)
    assert [row.relationship_id
            for row in outbound] == [graph["app_in_apollo"].relationship_id]
    inbound = manager.get_entity_relationships(
        "s", app.entity_id, direction="inbound"
    )
    assert {row.relationship_id for row in inbound} == {
        graph["settings_in_app"].relationship_id,
        graph["concerns"].relationship_id,
    }
    both = manager.get_entity_relationships(
        "s", app.entity_id, direction="both"
    )
    assert {row.relationship_id for row in both} == {
        graph["app_in_apollo"].relationship_id,
        graph["settings_in_app"].relationship_id,
        graph["concerns"].relationship_id,
    }
    typed = manager.get_entity_relationships(
        "s", app.entity_id, relationship_type="part_of"
    )
    assert [row.relationship_type for row in typed] == ["part_of"]


def test_no_unrestricted_multi_hop_traversal():
    manager, store = _manager()
    graph = _apollo(manager, store)
    # settings.py -> Mobile App -> Project Apollo is two hops.
    from_settings = manager.get_entity_relationships(
        "s", graph["settings"].entity_id
    )
    assert [row.relationship_id
            for row in from_settings] == [
                graph["settings_in_app"].relationship_id
            ]
    assert all(row.object_id != graph["apollo"].entity_id
               for row in from_settings)
    related = manager.get_related_entities("s", graph["settings"].entity_id)
    assert [row.entity.entity_id for row in related] == [graph["app"].entity_id]
    assert queries_mod.MAX_TRAVERSAL_HOPS == 1
    for name in (
        "walk", "walk_graph", "shortest_path", "connected_components",
        "neighbourhood", "subgraph", "related_to", "multi_hop",
    ):
        assert not hasattr(queries_mod, name)
        assert not hasattr(manager, name)
        assert not hasattr(store, name)


def test_hop_limit_is_clamped_and_reported():
    manager, store = _manager()
    graph = _apollo(manager, store)
    context = manager.get_selected_relationship_context(
        "s", entity_ids=[graph["settings"].entity_id], hop_limit=4
    )
    assert context.hop_limit == 1
    assert any("hop_limit_clamped" in item for item in context.diagnostics)
    # Even a clamped request cannot reach two hops out.
    assert all(row.object_id != graph["apollo"].entity_id
               for row in context.relationships)
    default = manager.get_selected_relationship_context(
        "s", entity_ids=[graph["settings"].entity_id]
    )
    assert default.hop_limit == 1
    assert default.diagnostics == []


def test_mandatory_limit_bounds_result_size():
    manager, store = _manager()
    collection = _active_entity(manager, store, "s", "My Collection")
    for index in range(55):
        child = _active_entity(
            manager, store, "s", f"Item {index}", "COMPONENT",
            timestamp=200 + index,
        )
        _link(
            manager, store, "part_of", child.entity_id,
            collection.entity_id, timestamp=300 + index,
        )
    assert len(manager.get_contained_entities(
        "s", collection.entity_id
    )) == queries_mod.DEFAULT_LIMIT
    assert len(manager.get_contained_entities(
        "s", collection.entity_id, limit=3
    )) == 3
    # An over-large request is clamped, never honoured.
    assert len(manager.get_contained_entities(
        "s", collection.entity_id, limit=500
    )) == queries_mod.MAX_LIMIT
    assert manager.get_contained_entities(
        "s", collection.entity_id, limit=0
    ) == []
    assert manager.get_contained_entities(
        "s", collection.entity_id, limit=-1
    ) == []


def test_inverse_contains_is_derived_only_and_never_stored():
    manager, store = _manager()
    graph = _apollo(manager, store)
    apollo = graph["apollo"]
    contained = manager.get_contained_entities("s", apollo.entity_id)
    assert [row.entity.entity_id for row in contained] == [graph["app"].entity_id]
    derived = contained[0]
    assert derived.derived is True
    assert derived.direction == "contains"
    # The row itself stays the canonical stored direction.
    assert derived.relationship.relationship_type == "part_of"
    assert derived.relationship.subject_id == graph["app"].entity_id
    assert derived.relationship.object_id == apollo.entity_id
    assert derived.relationship.status == "current"
    outbound = manager.get_related_entities(
        "s", graph["app"].entity_id, direction="outbound"
    )
    assert outbound[0].derived is False
    assert outbound[0].direction == "part_of"
    rows = store.fetch_relationships()
    assert all(row.relationship_type != "contains" for row in rows)
    con = sqlite3.connect(store.db_path)
    schema_sql = " ".join(
        row[0] or "" for row in con.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table'"
        )
    )
    columns = {row[1] for row in con.execute("PRAGMA table_info(relationships)")}
    con.close()
    assert "contains" not in schema_sql
    assert "contains" not in columns


def test_no_generic_related_to_or_manufactured_associations():
    manager, store = _manager()
    graph = _apollo(manager, store)
    store.create_entity(
        "s", "s", "PERSON", "Amir", "amir", [1], status="active",
        authority="user", confidence=1.0, timestamp=100, observed_at=100,
    )
    amir = store.fetch_entities(scope="s", normalized_name="amir")[0]
    # Same session, same messages, no stored relationship: nothing links
    # Amir to the app, and no lexical co-occurrence is manufactured.
    assert manager.get_entity_relationships("s", amir.entity_id) == []
    assert manager.get_related_entities("s", amir.entity_id) == []
    assert manager.get_entity_memories("s", amir.entity_id) == []
    assert manager.get_contained_entities("s", amir.entity_id) == []
    assert {row.relationship_type
            for row in store.fetch_relationships()} == {
                "part_of", "concerns", "explains"
            }
    # resolve_concept is not entity resolution.
    assert manager.resolve_entity("s", "amir")[0][0].entity_id == (
        amir.entity_id
    )
    assert manager.resolve_entity("s", "Amir and Project Apollo") == ([], False)


# ----------------------------------------------------------------------
# Entity -> memory
# ----------------------------------------------------------------------
def test_entity_memories_follow_concerns_only():
    manager, store = _manager()
    graph = _apollo(manager, store)
    app = graph["app"]
    links = manager.get_entity_memories("s", app.entity_id)
    assert len(links) == 1
    link = links[0]
    assert link.state.state_id == graph["decision"].state_id
    assert link.relationship.relationship_type == "concerns"
    assert link.relationship.subject_id == graph["decision"].state_id
    assert link.relationship.object_id == app.entity_id
    assert link.state.concept == "decision"
    assert link.state.source_ids
    # The rationale mentions no entity, so it is not returned here.
    assert manager.get_entity_memories(
        "s", graph["apollo"].entity_id
    ) == []
    # A keyword match alone never substitutes for the relationship.
    store.create_state(
        "s", "s", "meeting", "we discussed the mobile app", [1],
        status="current", authority="user", timestamp=400,
        memory_type="FACT",
    )
    assert len(manager.get_entity_memories("s", app.entity_id)) == 1
    # ... and it is still not returned by an entity with no concerns link.
    assert manager.get_entity_memories("s", graph["settings"].entity_id) == []


def test_entity_memories_preserve_provenance_and_temporal_metadata():
    manager, store = _manager()
    graph = _apollo(manager, store)
    link = manager.get_entity_memories("s", graph["app"].entity_id)[0]
    assert link.state.source_ids
    assert link.relationship.source_ids
    assert link.state.observed_at is not None
    assert link.state.valid_from is not None
    assert link.state.valid_until is None
    assert link.relationship.observed_at is not None
    assert link.relationship.valid_from is not None
    assert link.relationship.superseded_at is None
    assert link.relationship.authority == "user"
    assert link.relationship.status == "current"
    assert link.evaluation_time is not None


# ----------------------------------------------------------------------
# Explanation path
# ----------------------------------------------------------------------
def test_explanation_path_follows_explicit_explains_rows():
    manager, store = _manager()
    graph = _apollo(manager, store)
    paths = manager.get_relationship_explanation(
        "s", graph["decision"].state_id
    )
    assert len(paths) == 1
    path = paths[0]
    assert path.rationale_state.state_id == graph["rationale"].state_id
    assert path.decision_state.state_id == graph["decision"].state_id
    assert path.relationship.relationship_type == "explains"
    assert path.relationship.subject_id == graph["rationale"].state_id
    assert path.relationship.object_id == graph["decision"].state_id
    assert path.relationship.status == "current"
    assert path.derived is False
    # The fixed second segment: the decision concerns the Mobile App.
    assert [row.entity_id for row in path.concerned_entities] == [
        graph["app"].entity_id
    ]
    assert [row.relationship_type
            for row in path.concern_relationships] == ["concerns"]
    # Provenance is preserved on both ends of the path.
    assert path.rationale_state.source_ids
    assert path.decision_state.source_ids
    assert path.relationship.source_ids
    # Outbound from the rationale is the same path.
    outbound = manager.get_relationship_explanation(
        "s", graph["rationale"].state_id, direction="outbound"
    )
    assert [row.decision_state.state_id
            for row in outbound] == [graph["decision"].state_id]


def test_explanation_path_does_not_use_legacy_rationale_convention():
    manager, store = _manager()
    path_db = _db("legacy-rationale.db")
    legacy_store = ContextStore(path_db)
    legacy_manager = ContextManager(db_path=path_db)
    message_id, is_new, conflict = legacy_store.append(
        "s", 0, "user", "We ship on Friday because Aurora stays fast.",
        timestamp=100,
    )
    assert is_new and conflict is None
    legacy_store.create_state(
        "s", "s", "decision", "ship on friday", [message_id],
        status="current", authority="user", timestamp=100,
        memory_type="DECISION",
    )
    legacy_store.create_state(
        "s", "s", "rationale for decision", "aurora stays fast",
        [message_id], status="current", authority="user", timestamp=100,
        memory_type="RATIONALE",
    )
    # A rationale state whose name follows the legacy convention but which
    # has no stored `explains` row yields no explanation path.
    rationale = legacy_store.fetch_states(
        scope="s", concept="rationale for decision"
    )[0]
    assert legacy_manager.get_relationship_explanation(
        "s", rationale.state_id, direction="outbound"
    ) == []
    assert legacy_manager.get_relationship_explanation(
        "s", rationale.state_id
    ) == []
    assert legacy_store.fetch_relationships() == []


# ----------------------------------------------------------------------
# Temporal semantics
# ----------------------------------------------------------------------
def test_current_relationship_query_excludes_non_current_statuses():
    manager, store = _manager()
    app = _active_entity(manager, store, "s", "Mobile App")
    parent = _active_entity(manager, store, "s", "Project Apollo", "PROJECT",
                            timestamp=110)
    current = _link(manager, store, "part_of", app.entity_id,
                    parent.entity_id, timestamp=230)
    pending = _link(manager, store, "part_of", app.entity_id,
                    parent.entity_id, timestamp=200, confirm=False)
    rejected_candidate = _link(
        manager, store, "part_of", app.entity_id, parent.entity_id,
        timestamp=220, confirm=False,
    )
    manager.reject_relationship(rejected_candidate.relationship_id)
    conflict, diagnostics = manager.record_relationship_conflict(
        "s", "part_of", app.entity_id, parent.entity_id,
        source_ids=[_append(store, "s", "assistant says otherwise", 240)],
    )
    assert conflict is not None, diagnostics

    rows = manager.get_entity_relationships("s", app.entity_id)
    assert [row.relationship_id for row in rows] == [
        current.relationship_id
    ]
    assert pending.status == "candidate"
    assert store.fetch_relationship(conflict).status == "unresolved"
    assert store.fetch_relationship(
        rejected_candidate.relationship_id
    ).status == "rejected"
    # None of them may reach a current relationship query, in any direction.
    current_ids = {
        row.relationship_id for row in manager.get_entity_relationships(
            "s", app.entity_id, direction="both"
        )
    }
    assert pending.relationship_id not in current_ids
    assert conflict not in current_ids
    assert rejected_candidate.relationship_id not in current_ids
    assert current.relationship_id in current_ids
    # A historical query still refuses proposals and refusals.
    historical_ids = {
        row.relationship_id for row in manager.get_entity_relationships(
            "s", app.entity_id, temporal="historical"
        )
    }
    assert historical_ids == current_ids


def test_current_query_excludes_archived_endpoint():
    manager, store = _manager()
    graph = _apollo(manager, store)
    apollo = graph["apollo"]
    before = manager.get_entity_relationships(
        "s", graph["app"].entity_id, direction="both"
    )
    assert graph["app_in_apollo"].relationship_id in {
        row.relationship_id for row in before
    }
    archived, diagnostics = manager.archive_entity(apollo.entity_id)
    assert archived is not None, diagnostics
    after = manager.get_entity_relationships(
        "s", graph["app"].entity_id, direction="both"
    )
    # Only the edges touching the archived entity drop out.
    assert graph["app_in_apollo"].relationship_id not in {
        row.relationship_id for row in after
    }
    assert {row.relationship_id for row in after} == {
        graph["settings_in_app"].relationship_id,
        graph["concerns"].relationship_id,
    }
    assert manager.get_contained_entities("s", apollo.entity_id) == []
    # History is preserved: the row itself is never deleted.
    assert store.fetch_relationship(
        graph["app_in_apollo"].relationship_id
    ).status == "current"


def test_superseded_relationship_is_historical_only():
    manager, store = _manager()
    graph = _apollo(manager, store)
    app = graph["app"]
    apollo = graph["apollo"]
    original = graph["app_in_apollo"]
    # Confirming a correction supersedes its predecessor (Stage B write
    # semantics); the predecessor is then history, not a current assertion.
    replacement = _link(
        manager, store, "part_of", app.entity_id, apollo.entity_id,
        timestamp=500,
    )
    assert store.fetch_relationship(original.relationship_id).status == (
        "superseded"
    )
    assert replacement.supersedes_id == original.relationship_id

    current_rows = manager.get_entity_relationships("s", app.entity_id)
    assert [row.relationship_id for row in current_rows] == [
        replacement.relationship_id
    ]
    as_of_rows = manager.get_entity_relationships(
        "s", app.entity_id, temporal="as_of", at_time=200
    )
    assert [row.relationship_id for row in as_of_rows] == [
        original.relationship_id
    ]
    as_of_late = manager.get_entity_relationships(
        "s", app.entity_id, temporal="as_of", at_time=550
    )
    assert [row.relationship_id for row in as_of_late] == [
        replacement.relationship_id
    ]
    historical = manager.get_entity_relationships(
        "s", app.entity_id, temporal="historical"
    )
    assert {row.relationship_id for row in historical} == {
        original.relationship_id, replacement.relationship_id
    }
    statuses = {row.relationship_id: row.status for row in historical}
    assert statuses[original.relationship_id] == "superseded"
    assert statuses[replacement.relationship_id] == "current"

    # A bare invalidation ends validity without inventing a replacement.
    superseded, diagnostics = manager.supersede_relationship(
        replacement.relationship_id, valid_until=600, at_time=700
    )
    assert superseded is not None, diagnostics
    assert manager.get_entity_relationships(
        "s", app.entity_id, at_time=700
    ) == []
    assert {row.relationship_id for row in manager.get_entity_relationships(
        "s", app.entity_id, temporal="historical"
    )} == {original.relationship_id, replacement.relationship_id}


def test_as_of_uses_half_open_validity():
    manager, store = _manager()
    app = _active_entity(manager, store, "s", "Mobile App")
    parent = _active_entity(manager, store, "s", "Project Apollo", "PROJECT",
                            timestamp=110)
    row = _link(
        manager, store, "part_of", app.entity_id, parent.entity_id,
        timestamp=200, valid_from=100, valid_until=200,
    )
    assert row.valid_from == 100 and row.valid_until == 200
    assert manager.get_entity_relationships(
        "s", app.entity_id, temporal="as_of", at_time=100
    )
    assert manager.get_entity_relationships(
        "s", app.entity_id, temporal="as_of", at_time=199.999
    )
    assert manager.get_entity_relationships(
        "s", app.entity_id, temporal="as_of", at_time=200
    ) == []
    assert manager.get_entity_relationships(
        "s", app.entity_id, at_time=100
    )
    assert manager.get_entity_relationships(
        "s", app.entity_id, at_time=200
    ) == []
    # as_of without a time, and historical with a time, are both refused.
    assert manager.get_entity_relationships(
        "s", app.entity_id, temporal="as_of"
    ) == []
    assert manager.get_entity_relationships(
        "s", app.entity_id, temporal="historical", at_time=100
    ) == []
    assert manager.get_entity_relationships(
        "s", app.entity_id, temporal="nonsense"
    ) == []


def test_as_of_returns_relationship_effective_in_the_past():
    manager, store = _manager()
    app = _active_entity(manager, store, "s", "Mobile App")
    old_parent = _active_entity(manager, store, "s", "Old Project", "PROJECT",
                                timestamp=110)
    new_parent = _active_entity(manager, store, "s", "New Project", "PROJECT",
                                timestamp=120)
    first = _link(manager, store, "part_of", app.entity_id,
                  old_parent.entity_id, timestamp=200)
    second = _link(manager, store, "part_of", app.entity_id,
                   new_parent.entity_id, timestamp=400)
    store.supersede_relationship(first.relationship_id, 400, 500)
    store.fetch_relationship(second.relationship_id)
    rows = manager.get_entity_relationships(
        "s", app.entity_id, temporal="as_of", at_time=300
    )
    assert [row.relationship_id for row in rows] == [first.relationship_id]
    rows = manager.get_entity_relationships(
        "s", app.entity_id, temporal="as_of", at_time=450
    )
    assert [row.relationship_id for row in rows] == [second.relationship_id]
    related_at_300 = manager.get_related_entities(
        "s", app.entity_id, temporal="as_of", at_time=300
    )
    assert [row.entity.entity_id
            for row in related_at_300] == [old_parent.entity_id]


# ----------------------------------------------------------------------
# Selected relationship context
# ----------------------------------------------------------------------
def test_selected_relationship_context_is_bounded_and_provenance_preserving():
    manager, store = _manager()
    graph = _apollo(manager, store)
    context = manager.get_selected_relationship_context(
        "s", entity_ids=[graph["app"].entity_id],
        state_ids=[graph["rationale"].state_id],
    )
    assert context.scope == "s"
    assert context.temporal == "current"
    assert context.hop_limit == 1
    # The selected entity plus everything exactly one hop away, in either
    # direction: Project Apollo above, settings.py below. Nothing two hops.
    assert {row.entity_id for row in context.entities} == {
        graph["app"].entity_id,
        graph["apollo"].entity_id,
        graph["settings"].entity_id,
    }
    relationship_ids = {row.relationship_id
                        for row in context.relationships}
    assert graph["app_in_apollo"].relationship_id in relationship_ids
    assert graph["concerns"].relationship_id in relationship_ids
    assert graph["explains"].relationship_id in relationship_ids
    for row in context.relationships:
        assert row.status == "current"
        assert row.authority == "user"
        assert row.source_ids
        assert row.scope == "s"
        assert row.relationship_type in ("part_of", "concerns", "explains")
    for state in context.memories:
        assert state.source_ids
        assert state.status == "current"
    assert {row.state_id for row in context.memories} >= {
        graph["decision"].state_id, graph["rationale"].state_id
    }
    # One inbound part_of is reported as the derived `contains` inverse.
    assert context.derived_relationships == 1
    assert context.truncated is False
    # A two-hop seed cannot drag Project Apollo in through settings.py.
    deep = manager.get_selected_relationship_context(
        "s", entity_ids=[graph["settings"].entity_id]
    )
    assert {row.entity_id for row in deep.entities} == {
        graph["settings"].entity_id, graph["app"].entity_id
    }
    assert graph["apollo"].entity_id not in {
        row.entity_id for row in deep.entities
    }


def test_selected_relationship_context_derives_contains_and_caps():
    manager, store = _manager()
    collection = _active_entity(manager, store, "s", "My Collection")
    for index in range(6):
        child = _active_entity(
            manager, store, "s", f"Item {index}", "COMPONENT",
            timestamp=200 + index,
        )
        _link(manager, store, "part_of", child.entity_id,
              collection.entity_id, timestamp=300 + index)
    context = manager.get_selected_relationship_context(
        "s", entity_ids=[collection.entity_id], entity_limit=3,
        relationship_limit=2,
    )
    assert context.hop_limit == 1
    assert len(context.relationships) == 2
    assert len(context.entities) <= 3
    assert context.truncated is True
    assert context.derived_relationships == 2
    assert all(row.relationship_type == "part_of" for row in context.relationships)
    # Nothing else in the graph leaked in.
    assert len(store.fetch_relationships()) == 6


def test_selected_relationship_context_requires_a_selection():
    manager, store = _manager()
    graph = _apollo(manager, store)
    empty = manager.get_selected_relationship_context("s")
    assert empty.entities == []
    assert empty.relationships == []
    assert empty.memories == []
    assert empty.truncated is False
    # An unknown id is not a wildcard.
    unknown = manager.get_selected_relationship_context(
        "s", entity_ids=[99999]
    )
    assert unknown.entities == []
    assert unknown.relationships == []


def test_selected_relationship_context_respects_scope_and_temporal():
    manager, store = _manager()
    graph = _apollo(manager, store)
    context = manager.get_selected_relationship_context(
        "other", entity_ids=[graph["app"].entity_id]
    )
    assert context.entities == []
    assert context.relationships == []
    historical = manager.get_selected_relationship_context(
        "s", entity_ids=[graph["app"].entity_id], temporal="historical"
    )
    assert historical.temporal == "historical"
    assert historical.at_time > 0
    rejected = manager.get_selected_relationship_context(
        "s", entity_ids=[graph["app"].entity_id], temporal="as_of"
    )
    assert rejected.relationships == []
    assert any("relationship_query_rejected" in item
               for item in rejected.diagnostics)


# ----------------------------------------------------------------------
# Boundaries
# ----------------------------------------------------------------------
def test_queries_do_not_change_active_memory_or_working_memory():
    manager, store = _manager()
    graph = _apollo(manager, store)
    snapshot = [{"role": "user", "content": "I prefer the Apollo phone app.",
                 "timestamp": 500}]
    manager.reconcile(list(snapshot), "s")
    before_active = [
        (item.kind, item.memory_type, item.content)
        for item in manager.get_active_memory("s", at_time=600).items
    ]
    before_build = manager.build(
        list(snapshot), "s", token_counter=_counter, max_new_tokens=5,
        budget=_budget(),
    ).view
    before_states = store.fetch_states(scope="s")
    before_messages = store.fetch_latest("s", 500)

    manager.get_related_entities("s", graph["app"].entity_id)
    manager.get_contained_entities("s", graph["apollo"].entity_id)
    manager.get_entity_memories("s", graph["app"].entity_id)
    manager.get_relationship_explanation("s", graph["decision"].state_id)
    context = manager.get_selected_relationship_context(
        "s", entity_ids=[graph["app"].entity_id]
    )
    assert context.relationships

    after_active = [
        (item.kind, item.memory_type, item.content)
        for item in manager.get_active_memory("s", at_time=600).items
    ]
    after_build = manager.build(
        list(snapshot), "s", token_counter=_counter, max_new_tokens=5,
        budget=_budget(),
    ).view
    assert after_active == before_active
    assert after_build.messages == before_build.messages
    assert after_build.input_tokens == before_build.input_tokens
    assert store.fetch_states(scope="s") == before_states
    assert store.fetch_latest("s", 500) == before_messages
    # No graph dump: no entity or relationship line is ever injected.
    assert all(item.kind in ("state", "session", "digest")
               for item in manager.get_active_memory("s", at_time=600).items)
    assert all(
        not str(item.get("content", "")).startswith(
            ("[entity", "[relationship", "[part_of", "[explains", "[concerns")
        )
        for item in after_build.messages
    )


def test_working_memory_stays_runtime_only_after_queries():
    manager, store = _manager()
    graph = _apollo(manager, store)
    manager.get_selected_relationship_context(
        "s", entity_ids=[graph["app"].entity_id]
    )
    working = WorkingMemory("s", task="apollo review")
    assert not hasattr(working, "_store")
    con = sqlite3.connect(store.db_path)
    tables = {
        row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    con.close()
    assert "working_memory" not in tables
    assert "working" not in tables
    for name in ("entities", "entity_aliases", "relationships"):
        assert name in tables
    # A query never promotes anything to a candidate or to truth.
    assert store.fetch_candidates(scope="s", status="pending") == []
    assert len(store.fetch_relationships(status="candidate")) == 0


def test_queries_do_not_change_relationship_writes():
    manager, store = _manager()
    graph = _apollo(manager, store)
    before = store.fetch_relationships()
    manager.get_entity_relationships("s", graph["app"].entity_id)
    manager.get_related_entities("s", graph["app"].entity_id)
    manager.get_contained_entities("s", graph["apollo"].entity_id)
    manager.get_entity_memories("s", graph["app"].entity_id)
    manager.get_relationship_explanation("s", graph["decision"].state_id)
    manager.get_selected_relationship_context(
        "s", entity_ids=[graph["app"].entity_id]
    )
    assert store.fetch_relationships() == before
    # The write path still behaves exactly as in Stage B.
    other = _active_entity(manager, store, "s", "Design System", "COMPONENT",
                           timestamp=600)
    created = _link(manager, store, "part_of", other.entity_id,
                    graph["app"].entity_id, timestamp=700)
    assert created.status == "current"
    assert len(store.fetch_relationships()) == len(before) + 1
    assert manager.get_entity_relationships("s", other.entity_id) == [
        created
    ]


def test_real_world_nested_part_of_stays_one_hop():
    manager, store = _manager()
    car = _active_entity(manager, store, "s", "Car", "COMPONENT")
    collection = _active_entity(
        manager, store, "s", "Car Collection", "COMPONENT", timestamp=110
    )
    garage = _active_entity(
        manager, store, "s", "Garage", "COMPONENT", timestamp=120
    )
    car_in_collection = _link(
        manager, store, "part_of", car.entity_id, collection.entity_id,
        timestamp=200,
    )
    collection_in_garage = _link(
        manager, store, "part_of", collection.entity_id, garage.entity_id,
        timestamp=210,
    )
    assert manager.get_contained_entities(
        "s", collection.entity_id
    )[0].entity.entity_id == car.entity_id
    assert manager.get_contained_entities(
        "s", garage.entity_id
    )[0].entity.entity_id == collection.entity_id
    # From the Car, only the collection is one hop away.
    rows = manager.get_entity_relationships("s", car.entity_id)
    assert [row.relationship_id for row in rows] == [
        car_in_collection.relationship_id
    ]
    assert all(row.object_id != garage.entity_id for row in rows)
    assert store.fetch_relationship(
        collection_in_garage.relationship_id
    ).status == "current"
    assert manager.get_related_entities("s", car.entity_id)[0].entity == (
        collection
    )


def test_real_world_person_entity_has_no_manufactured_links():
    manager, store = _manager()
    amir = _active_entity(manager, store, "s", "Amir", "PERSON")
    app = _active_entity(manager, store, "s", "Mobile App", timestamp=110)
    apollo = _active_entity(
        manager, store, "s", "Project Apollo", "PROJECT", timestamp=120
    )
    _link(manager, store, "part_of", app.entity_id, apollo.entity_id,
          timestamp=200)
    _append(store, "s", "Amir works on the Mobile App for Project Apollo.",
            300)
    assert manager.get_entity_relationships("s", amir.entity_id) == []
    assert manager.get_related_entities("s", amir.entity_id) == []
    assert manager.get_entity_memories("s", amir.entity_id) == []
    assert manager.get_entity_relationships(
        "s", apollo.entity_id, direction="inbound"
    )[0].subject_id == app.entity_id
    context = manager.get_selected_relationship_context(
        "s", entity_ids=[amir.entity_id]
    )
    assert context.entities == [amir]
    assert context.relationships == []
