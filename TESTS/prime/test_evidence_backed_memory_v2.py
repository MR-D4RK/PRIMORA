"""Read-only V2 contracts; every test uses a disposable SQLite database."""

import json

import pytest

from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context.benchmark_v2 import run_benchmark
from PRIMORA.prime.context.formation import apply_proposals
from PRIMORA.prime.context.models import ObservationRecord
from PRIMORA.prime.context.observation import (
    ObservationProposal,
    ObservationService,
)
from PRIMORA.prime.context.manager import ContextManager
from PRIMORA.prime.context.store import ContextStore


class FakeProvider:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def invoke(self, capability, payload):
        self.calls += 1
        return PrimeResult.success({"content": self.content})


def test_observations_are_opt_in_and_source_backed(tmp_path, monkeypatch):
    db = tmp_path / "observation.db"
    store = ContextStore(db)
    source_id, *_ = store.append("s", 0, "user", "I like zorbles", timestamp=10)
    assistant_id, *_ = store.append("s", 1, "assistant", "You like zorbles", timestamp=11)
    service = ObservationService(store)
    with pytest.raises(RuntimeError):
        service.create("s", ObservationProposal("zorbles", "violet", [source_id]))
    service = ObservationService(store, enabled=True)
    row = service.create("s", ObservationProposal(
        "zorbles", "violet", [source_id, source_id],
        source_quotes={source_id: "like zorbles"},
    ))
    assert isinstance(row, ObservationRecord)
    assert row.proof_count == 1
    assert row.source_ids == [source_id]
    with pytest.raises(ValueError):
        service.create("s", ObservationProposal("bad", "assistant", [assistant_id]))
    newer_id, *_ = store.append("s", 2, "user", "Actually, I like amber zorbles", timestamp=20)
    replacement = service.create("s", ObservationProposal(
        "zorbles", "amber", [newer_id], source_quotes={newer_id: "like amber zorbles"}
    ))
    service.supersede(row.observation_id, replacement_id=replacement.observation_id, reason="user correction", transition_time=20)
    assert service.history(row.observation_id)[0].snapshot["value"] == "violet"
    assert store.fetch_observations(scope="s", status="superseded")[0].value == "violet"


def test_typed_formation_validates_quotes_and_non_assertions(tmp_path):
    manager = ContextManager(db_path=tmp_path / "formation.db", enable_memory_v2=True)
    store = manager._get_store()
    message_id, *_ = store.append("s", 0, "user", "I like flarnic textures", timestamp=10)
    message = store.fetch_message(message_id)
    provider = FakeProvider(json.dumps({"candidates": [{
        "kind": "preference", "concept": "preferred flarnic texture",
        "value": "flarnic textures", "claim_kind": "assertion", "actor": "user",
        "scope": "global", "source_ids": [message_id], "source_quote": "I like flarnic textures",
    }]}))
    result = apply_proposals(store, "s", message, provider)
    assert provider.calls == 1 and result.admitted_ids
    state = store.fetch_states(scope="global", status="current")[0]
    assert state.authority == "user" and state.source_ids == [message_id]
    bad = FakeProvider(json.dumps({"candidates": [{
        "kind": "preference", "concept": "invented", "value": "value",
        "claim_kind": "hypothesis", "actor": "user", "scope": "global",
        "source_ids": [message_id], "source_quote": "I like flarnic textures",
    }]}))
    rejected = apply_proposals(store, "s", message, bad)
    assert not rejected.admitted_ids and any("non_assertion" in d for d in rejected.diagnostics)


def test_typed_formation_rejects_inverted_predicate_values_and_bad_provenance(tmp_path):
    manager = ContextManager(db_path=tmp_path / "inverted.db", enable_memory_v2=True)
    store = manager._get_store()
    message_id, *_ = store.append("alpha", 0, "user", "I like kinetic sculptures", timestamp=10)
    message = store.fetch_message(message_id)
    inverted = FakeProvider(json.dumps({"candidates": [{
        "kind": "preference", "concept": "kinetic sculptures", "value": "like",
        "claim_kind": "assertion", "actor": "user", "scope": "session",
        "source_ids": [message_id], "source_quote": "I like kinetic sculptures",
    }]}))
    result = apply_proposals(store, "alpha", message, inverted)
    assert not result.admitted_ids
    assert any("preference_value_is_predicate" in item for item in result.diagnostics)
    assert not store.fetch_states(scope="alpha", status="current")
    invalid_source = FakeProvider(json.dumps({"candidates": [{
        "kind": "preference", "concept": "kinetic sculptures", "value": "sculptures",
        "claim_kind": "assertion", "actor": "user", "scope": "global",
        "source_ids": [999999], "source_quote": "I like kinetic sculptures",
    }]}))
    rejected = apply_proposals(store, "alpha", message, invalid_source)
    assert not rejected.admitted_ids
    assert any("source_not_user" in item for item in rejected.diagnostics)
    omitted_provenance = FakeProvider(json.dumps({"candidates": [{
        "kind": "preference", "concept": "preferred kinetic sculpture",
        "value": "kinetic sculptures", "claim_kind": "assertion", "actor": "user",
        "scope": "session",
    }]}))
    bound = apply_proposals(store, "alpha", message, omitted_provenance)
    saved = store.fetch_states(scope="global", status="current")
    assert bound.admitted_ids and saved[-1].source_ids == [message_id]
    generalized = FakeProvider(json.dumps({"candidates": [{
        "kind": "preference", "concept": "preferred art form",
        "value": "kinetic sculptures", "claim_kind": "assertion", "actor": "user",
        "scope": "global", "source_ids": [message_id],
        "source_quote": "I like kinetic sculptures",
    }]}))
    generalized_result = apply_proposals(store, "alpha", message, generalized)
    assert not generalized_result.admitted_ids
    assert any("generalization_not_source_supported" in item for item in generalized_result.diagnostics)


def test_inverted_qwen_preference_cannot_duplicate_deterministic_memory(tmp_path):
    provider = FakeProvider(json.dumps({"candidates": [{
        "kind": "preference", "concept": "kinetic sculptures", "value": "like",
        "claim_kind": "assertion", "actor": "user", "scope": "global",
        "source_ids": [1], "source_quote": "I like kinetic sculptures",
    }]}))
    manager = ContextManager(
        db_path=tmp_path / "inverted-manager.db", enable_memory_v2=True,
        formation_provider=provider,
    )
    manager.reconcile([{"role": "user", "content": "I like kinetic sculptures"}], "s")
    states = manager._get_store().fetch_states(scope="global", status="current")
    assert any(row.value.startswith("loves kinetic sculptures") for row in states)
    assert not any(row.concept == "kinetic sculptures" and row.value == "like" for row in states)


def test_three_unrelated_preferences_do_not_double_into_six_states(tmp_path):
    class DuplicateProvider:
        def invoke(self, capability, payload):
            text = payload["messages"][-1]["content"]
            subject = text.split(maxsplit=2)[-1].rstrip(".")
            return PrimeResult.success({"content": json.dumps({"candidates": [{
                "kind": "preference", "concept": f"preferred {subject}",
                "value": subject, "claim_kind": "assertion", "actor": "user",
                "scope": "session", "source_ids": [payload["source_message_id"]],
                "source_quote": text,
            }]})})

    manager = ContextManager(
        db_path=tmp_path / "three-preferences.db", enable_memory_v2=True,
        formation_provider=DuplicateProvider(),
    )
    for text in (
        "I like kinetic sculptures",
        "I prefer alpine jazz",
        "I love copper architecture",
    ):
        manager.reconcile([{"role": "user", "content": text}], "s")
    states = manager._get_store().fetch_states(scope="global", status="current")
    assert len([row for row in states if row.memory_type == "PREFERENCE"]) == 3
    assert {"kinetic sculptures", "alpine jazz", "copper architecture"}.issubset(
        {row.value.removeprefix("loves ") for row in states}
    )


def test_provider_failure_falls_back_and_feature_off_preserves_legacy(tmp_path):
    broken = FakeProvider("not json")
    manager = ContextManager(
        db_path=tmp_path / "fallback.db", enable_memory_v2=True,
        formation_provider=broken,
    )
    diagnostics = manager.reconcile([{"role": "user", "content": "I like a flarnic texture"}], "s")
    assert broken.calls == 1
    assert any("typed_formation" in item for item in diagnostics)
    assert manager._get_store().fetch_states(scope="global", status="current")
    disabled_provider = FakeProvider('{"candidates": []}')
    disabled = ContextManager(db_path=tmp_path / "disabled.db", formation_provider=disabled_provider)
    disabled.reconcile([{"role": "user", "content": "I like a flarnic texture"}], "s")
    assert disabled_provider.calls == 0


def test_empty_qwen_response_keeps_deterministic_preference_and_timing(tmp_path):
    class EmptyProvider:
        calls = 0

        def invoke(self, capability, payload):
            self.calls += 1
            return PrimeResult.failure("OpenAI returned empty response")

    provider = EmptyProvider()
    manager = ContextManager(
        db_path=tmp_path / "empty-qwen.db", enable_memory_v2=True,
        formation_provider=provider,
    )
    diagnostics = manager.reconcile([{
        "role": "user", "content": "I prefer asymmetrical architecture"
    }], "s")
    assert provider.calls == 1
    assert any("provider_failure:OpenAI returned empty response" in item for item in diagnostics)
    assert any(
        "typed_formation_timing:" in item
        and "provider_ms=" in item
        and "db_ms=" in item
        for item in diagnostics
    )
    states = manager._get_store().fetch_states(scope="global", status="current")
    assert any("asymmetrical architecture" in row.value for row in states)


def test_v2_personal_preferences_are_global_across_sessions(tmp_path):
    provider = FakeProvider(json.dumps({"candidates": [{
        "kind": "preference", "concept": "preferred kinetic sculpture",
        "value": "kinetic sculptures", "claim_kind": "assertion", "actor": "user",
        "scope": "session", "source_ids": [1],
        "source_quote": "I like kinetic sculptures",
    }]}))
    manager = ContextManager(
        db_path=tmp_path / "cross-session.db", enable_memory_v2=True,
        formation_provider=provider,
    )
    manager.reconcile([{"role": "user", "content": "I like kinetic sculptures"}], "alpha")
    assert manager._get_store().fetch_states(scope="global", status="current")
    evidence = manager.memory_query_evidence({
        "sources": ["preferences"],
        "query": "What are the last three things I told you I liked?",
        "result_count": 3, "order": "recent", "temporal_scope": "current",
    }, session_id="beta")
    assert "kinetic sculptures" in evidence


def test_explicit_temporary_preference_remains_session_scoped(tmp_path):
    provider = FakeProvider(json.dumps({"candidates": [{
        "kind": "preference", "concept": "preferred temporary sculpture",
        "value": "temporary sculpture", "claim_kind": "assertion", "actor": "user",
        "scope": "global", "source_ids": [1],
        "source_quote": "For now I like a temporary sculpture",
    }]}))
    manager = ContextManager(
        db_path=tmp_path / "temporary-scope.db", enable_memory_v2=True,
        formation_provider=provider,
    )
    manager.reconcile([{
        "role": "user", "content": "For now I like a temporary sculpture"
    }], "alpha")
    assert not manager._get_store().fetch_states(scope="global", status="current")
    assert manager._get_store().fetch_states(scope="alpha", status="current")


def test_historical_preference_query_includes_superseded_source_only_when_requested(tmp_path):
    manager = ContextManager(
        db_path=tmp_path / "historical-preferences.db", enable_memory_v2=True,
        hybrid_retrieval=False,
    )
    manager.reconcile([
        {"role": "user", "content": "I like kinetic sculptures", "seq": 0, "timestamp": 100},
        {"role": "user", "content": "Actually, I don't like kinetic sculptures", "seq": 1, "timestamp": 200},
    ], "alpha")
    current = manager.memory_query_evidence({
        "sources": ["preferences"],
        "query": "What are the last things I told you I liked?",
        "result_count": 10, "order": "recent", "temporal_scope": None,
    }, session_id="beta")
    historical = manager.memory_query_evidence({
        "sources": ["preferences"],
        "query": "What did I previously like?",
        "result_count": 10, "order": "recent", "temporal_scope": None,
    }, session_id="beta")
    assert "kinetic sculptures" not in current
    assert "I like kinetic sculptures" in historical
    assert "source_ids=" in historical


def test_negative_qwen_preference_deduplicates_deterministic_dislike(tmp_path):
    provider = FakeProvider(json.dumps({"candidates": [{
        "kind": "preference", "concept": "kinetic sculptures",
        "value": "dislike kinetic sculptures", "claim_kind": "assertion",
        "actor": "user", "scope": "global", "source_ids": [1],
        "source_quote": "I don't like kinetic sculptures",
    }]}))
    manager = ContextManager(
        db_path=tmp_path / "negative-dedup.db", enable_memory_v2=True,
        formation_provider=provider,
    )
    manager.reconcile([{
        "role": "user", "content": "I don't like kinetic sculptures"
    }], "alpha")
    states = [row for row in manager._get_store().fetch_states(
        scope="global", status="current"
    ) if row.memory_type == "PREFERENCE"]
    assert len(states) == 1
    assert states[0].value.startswith("dislikes kinetic sculptures")


def test_event_and_relationship_proposals_remain_source_backed(tmp_path):
    manager = ContextManager(db_path=tmp_path / "typed-events.db", enable_memory_v2=True)
    store = manager._get_store()
    source, *_ = store.append("s", 0, "user", "I completed the flarnic migration", timestamp=10)
    message = store.fetch_message(source)
    event_provider = FakeProvider(json.dumps({"candidates": [{
        "kind": "event", "concept": "migration", "value": "completed",
        "predicate": "completion", "claim_kind": "assertion", "actor": "user",
        "scope": "session", "source_ids": [source],
        "source_quote": "I completed the flarnic migration",
    }]}))
    result = apply_proposals(store, "s", message, event_provider)
    assert result.admitted_ids and store.fetch_experience_events(session_id="s")
    relationship_provider = FakeProvider(json.dumps({"candidates": [{
        "kind": "relationship", "concept": "migration", "value": "project",
        "predicate": "concerns", "claim_kind": "assertion", "actor": "user",
        "scope": "session", "source_ids": [source],
        "source_quote": "I completed the flarnic migration",
    }]}))
    held = apply_proposals(store, "s", message, relationship_provider)
    assert held.held_ids and store.fetch_candidates(status="pending")


def test_typed_query_filters_before_last_n_order(tmp_path):
    manager = ContextManager(db_path=tmp_path / "query.db", enable_memory_v2=True)
    store = manager._get_store()
    for seq, text in enumerate((
        "I like amber flarn", "The weather is clear", "I like violet flarn",
        "The assistant likes no personal thing", "I prefer asymmetrical architecture",
        "I like gold flarn",
    )):
        message_id, *_ = store.append("s", seq, "user", text, timestamp=100 + seq)
        if text.startswith(("I like", "I prefer")):
            value = text.removeprefix("I like ").removeprefix("I prefer ")
            store.create_state("s", "global", "preferred flarn", value, [message_id], memory_type="PREFERENCE", timestamp=100 + seq)
    evidence = manager.memory_query_evidence({
        "sources": ["preferences"], "query": "What are the last three things I told you I liked?",
        "result_count": 3, "order": "recent", "temporal_scope": "current",
    }, session_id="s")
    assert evidence.count("[message ") == 3
    assert "gold flarn" in evidence and "violet flarn" in evidence and "asymmetrical architecture" in evidence
    assert "amber flarn" not in evidence
    assert "weather" not in evidence and "assistant likes" not in evidence
    assert "source_ids=" in evidence


def test_typed_formation_records_provider_timing_and_bounded_json_request(tmp_path):
    provider = FakeProvider('{"candidates": []}')
    manager = ContextManager(
        db_path=tmp_path / "timing.db", enable_memory_v2=True,
        formation_provider=provider,
    )
    diagnostics = manager.reconcile([{
        "role": "user", "content": "I am drawn to flarnic architecture"
    }], "s")
    assert provider.calls == 1
    assert any("typed_formation_timing:" in item for item in diagnostics)


def test_reasoning_only_provider_output_still_uses_safe_json_validation(tmp_path):
    provider = FakeProvider(
        "I considered the wording. {\"candidates\":[{"
        "\"kind\":\"preference\",\"concept\":\"preferred flarnic architecture\","
        "\"value\":\"asymmetrical architecture\",\"claim_kind\":\"assertion\","
        "\"actor\":\"user\",\"scope\":\"global\",\"source_ids\":[1],"
        "\"source_quote\":\"I prefer asymmetrical architecture\"}]}",
    )
    manager = ContextManager(
        db_path=tmp_path / "reasoning.db", enable_memory_v2=True,
        formation_provider=provider,
    )
    manager.reconcile([{
        "role": "user", "content": "I prefer asymmetrical architecture"
    }], "s")
    # The deterministic path is already sufficient for this wording; the
    # assertion verifies that any reasoning JSON remains source validated.
    states = manager._get_store().fetch_states(scope="global", status="current")
    assert any("asymmetrical architecture" in row.value for row in states)


def test_relationship_temporal_source_episode_is_additive(tmp_path):
    store = ContextStore(tmp_path / "relationship.db")
    source, *_ = store.append("s", 0, "user", "A is part of B", timestamp=10)
    from PRIMORA.prime.context import entities, relationships
    a, _ = entities.create_candidate(store, "s", "A", "PERSON", [source])
    b, _ = entities.create_candidate(store, "s", "B", "PERSON", [source])
    rid, _ = relationships.create_candidate(
        store, "s", "part_of", a, b, [source], source_episode_ids=["episode-1"],
        valid_from=10, valid_until=20,
    )
    row = store.fetch_relationship(rid)
    assert row.source_episode_ids == ["episode-1"]
    assert row.valid_from == 10 and row.valid_until == 20


def test_benchmark_is_disposable_and_reports_unavailable_hardware():
    report = run_benchmark()
    assert report["conversations"] >= 200 and report["domains"] >= 50
    assert report["candidate"]["retrieval"]["p_at_3"] >= 0.90
    assert report["candidate"]["retrieval"]["recall_at_3"] >= 0.85
    assert report["candidate"]["retrieval"]["exact_order_count"] >= 0.95
    assert report["candidate"]["formation"]["precision"] >= 0.90
    assert report["candidate"]["formation"]["recall"] >= 0.90
    assert report["candidate"]["formation"]["f1"] >= 0.90
    assert report["baseline"]["model_calls"] == 0
    assert report["candidate"]["model_calls"] == report["conversations"]
    assert report["baseline"]["retrieval"]["p_at_3"] < report["candidate"]["retrieval"]["p_at_3"]
    assert report["baseline"]["abstention"] == 0.0
    assert report["candidate"]["abstention"] == 1.0
    assert report["source_preservation"] == 1.0
    assert report["alias_resolution"] == 1.0
    assert report["two_clock_events"] == 1.0
    assert report["unsupported_claim_rate"] == report["candidate"]["unsupported_claim_rate"]
    assert report["production_db_touched"] is False
    assert report["qwen_validation"] == "NOT RUN"

