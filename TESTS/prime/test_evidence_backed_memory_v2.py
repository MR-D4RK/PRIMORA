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
        "The assistant likes no personal thing", "I like gold flarn",
    )):
        message_id, *_ = store.append("s", seq, "user", text, timestamp=100 + seq)
        if text.startswith("I like"):
            value = text.removeprefix("I like ")
            store.create_state("s", "global", "preferred flarn", value, [message_id], memory_type="PREFERENCE", timestamp=100 + seq)
    evidence = manager.memory_query_evidence({
        "sources": ["preferences"], "query": "What are the last three things I told you I liked?",
        "result_count": 3, "order": "recent", "temporal_scope": "current",
    }, session_id="s")
    assert evidence.count("[message ") == 3
    assert "gold flarn" in evidence and "violet flarn" in evidence and "amber flarn" in evidence
    assert "weather" not in evidence and "assistant likes" not in evidence
    assert "source_ids=" in evidence


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
    assert report["candidate"]["p_at_3"] >= 0.90
    assert report["candidate"]["recall_at_3"] >= 0.85
    assert report["candidate"]["exact_order_count"] >= 0.95
    assert report["formation"]["precision"] >= 0.90
    assert report["formation"]["recall"] >= 0.90
    assert report["formation"]["f1"] >= 0.90
    assert report["source_preservation"] == 1.0
    assert report["unsupported_claim_rate"] == 0.0
    assert report["production_db_touched"] is False
    assert report["qwen_validation"] == "NOT RUN"

