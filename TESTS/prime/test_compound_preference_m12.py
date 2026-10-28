"""Integrated semantic-first memory acceptance: disposable SQLite only.

No topic- or product-specific runtime extraction. The provider is a fixture
that stands in for Qwen's interpretation; live Qwen must be tested separately.
"""
from __future__ import annotations

import json
import pytest

from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.formation import validate_semantic_candidate, TypedMemoryCandidate
from PRIMORA.prime.context.store import ContextStore


class _ProposalProvider:
    def __init__(self, subject, predicate="love", claim_kind="assertion", fail=False):
        self.subject = subject
        self.predicate = predicate
        self.claim_kind = claim_kind
        self.fail = fail
        self.calls = []

    def invoke(self, capability, payload):
        assert capability == "memory_formation"
        self.calls.append(payload)
        if self.fail:
            return PrimeResult.failure("offline")
        if not self.subject:
            return PrimeResult.success({"content": '{"candidates":[]}'})
        source = payload["messages"][-1]["content"]
        subject = self.subject
        verb = {
            "love": "loves", "like": "likes", "adore": "adores",
            "enjoy": "enjoys", "prefer": "prefers",
            "hate": "hates", "dislike": "dislikes",
        }[self.predicate]
        candidate = {
            "kind": "preference",
            "concept": f"likes for {subject}",
            "value": f"{verb} {subject}",
            "subject": subject,
            "predicate": self.predicate,
            "claim_kind": self.claim_kind,
            "actor": "user",
            "scope": "global",
            "source_ids": [payload["source_message_id"]],
            "source_quote": source,
            "confidence": .95,
        }
        return PrimeResult.success({"content": json.dumps({"candidates":[candidate]})})


@pytest.fixture(autouse=True)
def semantic_mode(monkeypatch):
    monkeypatch.setenv("PRIMORA_MEMORY_FORMATION_MODE", "semantic")
    monkeypatch.setenv("PRIMORA_MEMORY_FORMATION_DEFERRED", "0")


@pytest.mark.parametrize("statement,subject,predicate", [
    (
        "i love cyberpunk i love every aspect of it whether its story "
        "its gameplay or visual all of those are the best",
        "cyberpunk", "love",
    ),
    (
        "I adore Silver Lanterns I love every part of it, especially the arrangement",
        "Silver Lanterns", "adore",
    ),
    (
        "I enjoy Northwind Atlas; its maps and page layout are delightful.",
        "Northwind Atlas", "enjoy",
    ),
])
def test_unfamiliar_compound_claims_are_canonical_and_source_backed(
    tmp_path, statement, subject, predicate,
):
    path = tmp_path / "semantic.db"
    provider = _ProposalProvider(subject, predicate)
    manager = ContextManager(
        db_path=path, formation_provider=provider, hybrid_retrieval=False,
    )
    assert manager.semantic_formation
    manager.reconcile([
        {"role": "user", "content": "Which items do I like most?"},
        {"role": "assistant", "content": "Tell me which one you enjoy."},
        {"role": "user", "content": statement},
    ], "conversation")
    states = manager._get_store().fetch_states(
        scope="global", concept=f"likes for {subject.casefold()}",
        status="current",
    )
    assert len(states) == 1, manager._get_store().fetch_states(scope="global")
    assert states[0].value.lower() == {
        "love": "loves", "adore": "adores", "enjoy": "enjoys",
    }[predicate] + " " + subject.casefold()
    evidence = manager._get_store().fetch_messages_by_ids(states[0].source_ids)
    assert len(evidence) == 1 and evidence[0].content == statement
    # The fallback regex must not have stored a second, giant concept.
    assert not any("every aspect" in row.concept
                   for row in manager._get_store().fetch_states())

    restarted = ContextManager(db_path=path, hybrid_retrieval=False)
    recalled = restarted._get_store().fetch_states(
        scope="global", concept=f"likes for {subject.casefold()}", status="current"
    )
    assert len(recalled) == 1
    assert recalled[0].source_ids == states[0].source_ids


def test_older_dialogue_is_context_only_never_evidence(tmp_path):
    provider = _ProposalProvider("Birch Lantern")
    manager = ContextManager(
        db_path=tmp_path / "dialogue.db", formation_provider=provider,
    )
    manager.reconcile([
        {"role": "assistant", "content": "Remember your Birch Lantern project?"},
        {"role": "user", "content": "I love Birch Lantern for its shape."},
    ], "s")
    payload = provider.calls[-1]
    assert payload["messages"][-1]["content"] == "I love Birch Lantern for its shape."
    assert any(m["role"] == "assistant" for m in payload["messages"][1:-1])
    states = manager._get_store().fetch_states(scope="global", status="current")
    assert len(states) == 1
    linked = manager._get_store().fetch_messages_by_ids(states[0].source_ids)
    assert len(linked) == 1 and linked[0].role == "user"


@pytest.mark.parametrize("statement,subject", [
    ("Suppose I loved Grey Paper, that would be fictional.", "Grey Paper"),
    ("My sister loves Birch Lantern.", "Birch Lantern"),
    ("I love this I love every aspect of it.", "this"),
    ("I love Indigo Dusk I love every aspect of it.", "Indigo Dusk I love every aspect"),
])
def test_unsafe_model_proposals_do_not_become_current_memory(tmp_path, statement, subject):
    provider = _ProposalProvider(subject)
    manager = ContextManager(db_path=tmp_path / "unsafe.db",
                             formation_provider=provider)
    manager.reconcile([{"role": "user", "content": statement}], "s")
    assert manager._get_store().fetch_states(status="current") == []
    assert manager._get_store().fetch_range("s")[0].content == statement


def test_provider_failure_leaves_raw_only_without_legacy_fallback(tmp_path):
    path = tmp_path / "down.db"
    provider = _ProposalProvider("Cedar Kite", fail=True)
    manager = ContextManager(db_path=path, formation_provider=provider)
    manager.reconcile(
        [{"role": "user", "content": "I love Cedar Kite a lot."}], "s"
    )
    assert manager._get_store().fetch_states(status="current") == []
    assert manager._get_store().fetch_range("s")
    assert manager.formation_jobs(status="failed")


def test_no_provider_is_raw_only(tmp_path):
    manager = ContextManager(db_path=tmp_path / "noprov.db")
    manager.reconcile(
        [{"role": "user", "content": "I love New Horizon."}], "s"
    )
    assert manager._get_store().fetch_states(status="current") == []
    assert manager._get_store().fetch_range("s")


def test_source_quote_cannot_be_fabricated(tmp_path):
    store = ContextStore(tmp_path / "proof.db")
    source_id, _, _ = store.append("s", 0, "user", "I enjoy Silver Prism.")
    source = store.fetch_message(source_id)
    candidate = TypedMemoryCandidate(
        kind="preference", concept="likes for Silver Prism",
        value="loves Silver Prism", subject="Silver Prism",
        predicate="love", actor="user", claim_kind="assertion",
        source_ids=[source_id], source_quote="I love Silver Prism.",
    )
    valid, reason = validate_semantic_candidate(candidate, source)
    assert not valid and reason == "semantic_missing_exact_quote"


def test_legacy_remains_explicit_rollback(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIMORA_MEMORY_FORMATION_MODE", "legacy")
    manager = ContextManager(db_path=tmp_path / "legacy.db", enable_memory_v2=False)
    manager.reconcile(
        [{"role": "user", "content": "I love Willow Trees."}], "s"
    )
    assert manager._get_store().fetch_states(scope="global", status="current")

def test_negative_preference_supersedes_positive_without_affecting_other_topics(tmp_path):
    path = tmp_path / "correction.db"
    first = _ProposalProvider("Maple Board", "love")
    manager = ContextManager(db_path=path, formation_provider=first)
    manager.reconcile([
        {"role": "user", "content": "I love Maple Board."},
    ], "session")
    assert len(manager._get_store().fetch_states(
        scope="global", concept="likes for maple board", status="current")) == 1
    second = _ProposalProvider("Maple Board", "dislike", "negation")
    manager.set_formation_provider(second)
    manager.reconcile([
        {"role": "user", "content": "I don't like Maple Board anymore."},
    ], "other-session")
    currents = manager._get_store().fetch_states(
        scope="global", concept="likes for maple board", status="current")
    assert len(currents) == 1
    assert currents[0].value == "dislikes Maple Board"
    prior = manager._get_store().fetch_states(
        scope="global", concept="likes for maple board", status="superseded")
    assert len(prior) == 1 and prior[0].value == "loves Maple Board"


def test_repeated_reconciliation_is_idempotent(tmp_path):
    provider = _ProposalProvider("Winter Orchard")
    manager = ContextManager(db_path=tmp_path / "idempotent.db",
                             formation_provider=provider)
    turn = [{"role": "user", "content": "I love Winter Orchard."}]
    manager.reconcile(turn, "session")
    count_before = len(provider.calls)
    manager.reconcile(turn, "session")
    assert len(provider.calls) == count_before
    assert len(manager._get_store().fetch_states(
        scope="global", concept="likes for winter orchard",
        status="current")) == 1
