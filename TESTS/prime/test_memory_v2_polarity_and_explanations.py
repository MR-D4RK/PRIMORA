"""Adversarial Memory V2 polarity, rationale, and lifecycle contracts."""

import json

from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context.entities import create_candidate as create_entity
from PRIMORA.prime.context.entities import confirm as confirm_entity
from PRIMORA.prime.context.manager import ContextManager
from PRIMORA.prime.context.memory_audit import MemoryAuditService
from PRIMORA.prime.context.formation import apply_proposals
from PRIMORA.prime.context.relationships import create_candidate as create_relationship
from PRIMORA.prime.context.relationships import confirm as confirm_relationship


class TypedProvider:
    def __init__(self, candidates):
        self.candidates = candidates

    def invoke(self, capability, payload):
        rows = []
        for candidate in self.candidates:
            item = dict(candidate)
            item.setdefault("source_ids", [payload["source_message_id"]])
            item.setdefault("source_quote", payload["messages"][-1]["content"])
            rows.append(item)
        return PrimeResult.success({"content": json.dumps({"candidates": rows})})


def _evidence(manager, query, session="other"):
    return manager.memory_query_evidence(
        {"sources": ["preferences"], "query": query,
         "result_count": 20, "order": "recent", "temporal_scope": None},
        session_id=session,
    )


def test_historical_positive_inventory_excludes_current_negative_and_hate_is_negative(tmp_path):
    manager = ContextManager(
        db_path=tmp_path / "polarity.db", enable_memory_v2=True,
        hybrid_retrieval=False,
    )
    manager.reconcile([
        {"role": "user", "content": "I like flarnic ceramics", "timestamp": 100},
        {"role": "user", "content": "I no longer like flarnic ceramics", "timestamp": 200},
        {"role": "user", "content": "I hate hardstyle", "timestamp": 300},
    ], "first-session")

    historical = _evidence(manager, "What did I previously like?")
    assert "I like flarnic ceramics" in historical
    assert "I no longer like flarnic ceramics" not in historical
    assert "I hate hardstyle" not in historical

    negative = _evidence(manager, "What do I dislike?")
    assert "I no longer like flarnic ceramics" in negative
    assert "I hate hardstyle" in negative
    assert "I like flarnic ceramics" not in negative


def test_explanation_query_abstains_without_verified_user_reason(tmp_path):
    manager = ContextManager(
        db_path=tmp_path / "explanation-abstain.db", enable_memory_v2=True,
        hybrid_retrieval=False,
    )
    manager.reconcile([
        {"role": "user", "content": "I like flarnic ceramics", "timestamp": 100},
        {"role": "user", "content": "I like hardstyle because its rhythm helps me focus", "timestamp": 200},
    ], "s")
    no_reason = _evidence(manager, "Why did I like flarnic ceramics?", session="other")
    assert "couldn't verify a user-stated reason" in no_reason
    assert "flarnic ceramics" not in no_reason

    with_reason = _evidence(manager, "Why did I like hardstyle?", session="other")
    assert "user-stated reason" in with_reason
    assert "its rhythm helps me focus" in with_reason
    assert "flarnic ceramics" not in with_reason


def test_preference_inventory_excludes_mistyped_reason_state_but_keeps_rationale(tmp_path):
    provider = TypedProvider([{
        # This is the failure shape observed from local Qwen: the reason is
        # proposed as a second preference with the same source message.
        "kind": "preference", "concept": "quiet libraries",
        "value": "help me focus", "claim_kind": "assertion", "actor": "user",
        "scope": "global",
    }])
    manager = ContextManager(
        db_path=tmp_path / "reason-state.db", enable_memory_v2=True,
        hybrid_retrieval=False, formation_provider=provider,
    )
    manager.reconcile([{
        "role": "user",
        "content": "I like quiet libraries because they help me focus",
        "timestamp": 100,
    }], "s")
    evidence = _evidence(manager, "What did I previously like?", session="other")
    assert evidence.count("[message ") == 1
    assert "likes for quiet libraries" in evidence
    assert "stored: quiet libraries: help me focus" not in evidence
    # The rationale and the model-produced secondary state remain available
    # to source/audit paths; retrieval alone does not delete evidence.
    states = manager._get_store().fetch_states(scope="global")
    assert any(row.memory_type == "RATIONALE" and "help me focus" in row.value
               for row in states)
    assert any(row.concept == "quiet libraries" and row.value == "help me focus"
               for row in states)
    assert manager._get_store().fetch_relationships(
        scope="global", relationship_type="explains", status="current"
    )
    assert all(row.source_ids == [1] for row in states if row.source_ids)


def test_same_message_keeps_two_preferences_but_excludes_unrelated_fact(tmp_path):
    manager = ContextManager(
        db_path=tmp_path / "multi-preference-inventory.db", enable_memory_v2=True,
        hybrid_retrieval=False,
    )
    store = manager._get_store()
    source, *_ = store.append(
        "s", 0, "user",
        "I like quiet libraries and prefer urban gardens.", timestamp=100,
    )
    store.create_state(
        "s", "global", "likes for quiet libraries", "loves quiet libraries",
        [source], memory_type="PREFERENCE", timestamp=100,
    )
    store.create_state(
        "s", "global", "preferred urban gardens", "urban gardens",
        [source], memory_type="PREFERENCE", timestamp=100,
    )
    store.create_state(
        "s", "global", "weather", "sunny", [source],
        memory_type="FACT", timestamp=100,
    )
    evidence = _evidence(manager, "What did I previously like?", session="other")
    assert evidence.count("[message ") == 2
    assert "quiet libraries" in evidence
    assert "urban gardens" in evidence
    assert "sunny" not in evidence


def test_typed_no_longer_like_deduplicates_canonical_dislike_without_losing_source(tmp_path):
    provider = TypedProvider([{
        "kind": "preference", "concept": "flarnic ceramics",
        "value": "no longer like", "claim_kind": "assertion", "actor": "user",
        "scope": "global",
    }])
    manager = ContextManager(
        db_path=tmp_path / "typed-negative.db", enable_memory_v2=True,
        hybrid_retrieval=False, formation_provider=provider,
    )
    manager.reconcile([
        {"role": "user", "content": "I no longer like flarnic ceramics"},
    ], "s")
    current = [row for row in manager._get_store().fetch_states(
        scope="global", status="current") if row.memory_type == "PREFERENCE"]
    assert len(current) == 1
    assert current[0].value == "dislikes flarnic ceramics"
    assert current[0].source_ids == [1]


def test_same_source_keeps_distinct_positive_and_negative_subjects(tmp_path):
    manager = ContextManager(db_path=tmp_path / "distinct-claims.db", hybrid_retrieval=False)
    store = manager._get_store()
    source_id, *_ = store.append(
        "s", 0, "user", "I like flarnic ceramics and hate hardstyle", timestamp=10
    )
    message = store.fetch_message(source_id)
    result = apply_proposals(store, "s", message, TypedProvider([
        {"kind": "preference", "concept": "flarnic ceramics",
         "value": "like", "claim_kind": "assertion", "actor": "user", "scope": "global"},
        {"kind": "preference", "concept": "hardstyle",
         "value": "hate", "claim_kind": "negation", "actor": "user", "scope": "global"},
    ]))
    # Predicate-only values are rejected by Prime; this test uses complete
    # semantic values so both independent claims remain source-backed.
    assert not result.admitted_ids

    result = apply_proposals(store, "s", message, TypedProvider([
        {"kind": "preference", "concept": "flarnic ceramics",
         "value": "like flarnic ceramics", "claim_kind": "assertion",
         "actor": "user", "scope": "global"},
        {"kind": "preference", "concept": "hardstyle",
         "value": "hate hardstyle", "claim_kind": "negation",
         "actor": "user", "scope": "global"},
    ]))
    assert len(result.admitted_ids) == 2
    assert len([row for row in store.fetch_states(scope="global", status="current")
                if row.memory_type == "PREFERENCE"]) == 2


def test_m7_approval_does_not_silently_resolve_unrelated_organization_candidate(tmp_path):
    manager = ContextManager(db_path=tmp_path / "m7-unresolved.db", hybrid_retrieval=False)
    store = manager._get_store()
    source, *_ = store.append(
        "s", 0, "user", "The Flarnic project belongs to Aurora Labs.", timestamp=10
    )
    entity_a, _ = create_entity(store, "s", "Flarnic project", "PROJECT", [source])
    # M7's intentionally bounded entity vocabulary has no generic
    # ORGANIZATION type; affiliation candidates remain separate review items.
    entity_b, _ = create_entity(store, "s", "Aurora Labs", "PROJECT", [source])
    confirm_entity(store, entity_a)
    confirm_entity(store, entity_b)
    relationship, _ = create_relationship(
        store, "s", "part_of", entity_a, entity_b, [source]
    )
    confirm_relationship(store, relationship)
    candidate_id = store.create_candidate(
        "s", "global", [source], "organization_affiliation",
        "organization_affiliation", "Aurora Labs",
    )

    pending = store.fetch_candidates(candidate_id=candidate_id, status="pending")
    assert len(pending) == 1
    report = MemoryAuditService(store, enabled=True).run(scope="global")
    unresolved = [item for item in report.findings
                  if item.kind == "unresolved_candidate"
                  and item.record_id == candidate_id]
    assert unresolved
    assert store.fetch_candidates(candidate_id=candidate_id, status="pending")
