"""Integrated semantic-first memory acceptance: disposable SQLite only.

No topic- or product-specific runtime extraction. The provider is a fixture
that stands in for Qwen's interpretation; live Qwen must be tested separately.
"""
from __future__ import annotations

import json
import os
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



def test_real_preference_about_fictional_subject_is_not_hypothetical(tmp_path):
    """The existence of a fictional object does not make the user's like fictional."""
    statement = "I really love a fictional band called Candle Harbor."
    manager = ContextManager(
        db_path=tmp_path / "fictional-entity.db",
        formation_provider=_ProposalProvider("Candle Harbor", "love"),
    )
    manager.reconcile([{"role": "user", "content": statement}], "first")
    rows = manager._get_store().fetch_states(
        scope="global", concept="likes for candle harbor", status="current",
    )
    assert len(rows) == 1, manager.formation_jobs(status="failed")
    assert manager._get_store().fetch_messages_by_ids(rows[0].source_ids)[0].content == statement


@pytest.mark.parametrize("statement,subject,predicate,claim_kind", [
    ("I recently started liking a fictional band called Willow Bay.",
     "Willow Bay", "like", "assertion"),
    ("I also like a fictional band called Copper Orchard.",
     "Copper Orchard", "like", "assertion"),
    ("I still like Copper Orchard.", "Copper Orchard", "like", "assertion"),
    ("I don't like Willow Bay anymore.", "Willow Bay", "dislike", "negation"),
    ("I no longer like Willow Bay.", "Willow Bay", "dislike", "negation"),
])
def test_first_person_preference_variants_have_source_evidence(
        tmp_path, statement, subject, predicate, claim_kind):
    store = ContextStore(tmp_path / "proof.db")
    source_id, _, _ = store.append("session", 0, "user", statement)
    source = store.fetch_message(source_id)
    verbs = {"like": "likes", "dislike": "dislikes"}
    candidate = TypedMemoryCandidate(
        kind="preference", subject=subject, predicate=predicate,
        concept="likes for " + subject,
        value=verbs[predicate] + " " + subject,
        actor="user", claim_kind=claim_kind,
        source_ids=[source_id], source_quote=statement, scope="global",
    )
    assert validate_semantic_candidate(candidate, source) == (True, "ok")



def test_semantic_formation_retains_emphasized_feature_as_reason(tmp_path):
    statement = (
        "I recently started liking a fictional band called Northglass. "
        "I especially love its guitar melodies."
    )
    manager = ContextManager(
        db_path=tmp_path / "feature-reason.db",
        formation_provider=_ProposalProvider("Northglass", "like"),
    )
    manager.reconcile([{"role": "user", "content": statement}], "original")
    store = manager._get_store()
    preference = store.fetch_states(
        scope="global", concept="likes for northglass", status="current",
    )
    reasons = store.fetch_states(
        scope="global", concept="rationale for likes for northglass",
        status="current",
    )
    assert len(preference) == 1
    assert len(reasons) == 1, manager.formation_status()
    assert reasons[0].value == "I especially love its guitar melodies"
    assert reasons[0].memory_type == "RATIONALE"
    assert reasons[0].source_ids == preference[0].source_ids
    assert store.fetch_messages_by_ids(reasons[0].source_ids)[0].content == statement

    # No in-memory chat history: the reason survives a full store reopen.
    restarted = ContextManager(db_path=tmp_path / "feature-reason.db")
    historical = restarted._get_store().fetch_states(
        scope="global", concept="rationale for likes for northglass",
    )
    assert len(historical) == 1
    assert historical[0].value == "I especially love its guitar melodies"

    # A new-session correction retires the old explanation without erasing
    # it or changing its original user-message provenance.
    manager.set_formation_provider(
        _ProposalProvider("Northglass", "dislike", "negation")
    )
    manager.reconcile(
        [{"role": "user", "content": "I don't like Northglass anymore."}],
        "corrected",
    )
    assert store.fetch_states(
        scope="global", concept="rationale for likes for northglass",
        status="current",
    ) == []
    old_reasons = store.fetch_states(
        scope="global", concept="rationale for likes for northglass",
        status="superseded",
    )
    assert len(old_reasons) == 1
    assert old_reasons[0].value == "I especially love its guitar melodies"
    current = store.fetch_states(
        scope="global", concept="likes for northglass", status="current",
    )
    assert len(current) == 1 and current[0].value == "dislikes Northglass"


def test_focused_feature_does_not_attach_to_wrong_subject(tmp_path):
    statement = (
        "I like Northglass. I also like Copper Vale. "
        "I especially love its singer's voice."
    )
    manager = ContextManager(
        db_path=tmp_path / "subject-reference.db",
        formation_provider=_ProposalProvider("Northglass", "like"),
    )
    manager.reconcile([{"role": "user", "content": statement}], "first")
    assert not manager._get_store().fetch_states(
        scope="global", concept="rationale for likes for northglass",
    )


def test_compound_correction_creates_independent_history_across_sessions(tmp_path):
    """Two independent likes plus a mixed-polarity correction retain provenance."""
    class BatchProvider:
        def __init__(self, proposals):
            self.proposals = proposals

        def invoke(self, capability, payload):
            assert capability == "memory_formation"
            source = payload["messages"][-1]["content"]
            candidates = []
            for subject, predicate in self.proposals[source]:
                verb = {
                    "like": "likes", "love": "loves", "dislike": "dislikes",
                }[predicate]
                candidates.append({
                    "kind": "preference",
                    "concept": "likes for " + subject,
                    "value": verb + " " + subject,
                    "subject": subject,
                    "predicate": predicate,
                    "claim_kind": "negation" if predicate == "dislike" else "assertion",
                    "actor": "user",
                    "scope": "global",
                    "source_ids": [payload["source_message_id"]],
                    "source_quote": source,
                    "confidence": 0.95,
                })
            return PrimeResult.success({"content": json.dumps({"candidates": candidates})})

    first = "I recently started liking a fictional band called Candle Harbor."
    second = "I also like a fictional band called Copper Orchard."
    correction = "Actually, I don't like Candle Harbor anymore, but I still like Copper Orchard."
    manager = ContextManager(
        db_path=tmp_path / "compound.db",
        formation_provider=BatchProvider({
            first: [("Candle Harbor", "like")],
            second: [("Copper Orchard", "like")],
            correction: [("Candle Harbor", "dislike"), ("Copper Orchard", "like")],
        }),
    )
    assert manager.semantic_formation
    for session, statement in [
        ("one", first), ("one", second), ("two", correction)
    ]:
        manager.reconcile([{"role": "user", "content": statement}], session)

    store = manager._get_store()
    harbor = store.fetch_states(
        scope="global", concept="likes for candle harbor", status="current",
    )
    orchard = store.fetch_states(
        scope="global", concept="likes for copper orchard", status="current",
    )
    assert len(harbor) == len(orchard) == 1, (manager.formation_status(), store.fetch_states(scope="global"))
    assert harbor[0].value == "dislikes Candle Harbor"
    assert orchard[0].value == "likes Copper Orchard"
    old_harbor = store.fetch_states(
        scope="global", concept="likes for candle harbor", status="superseded",
    )
    assert len(old_harbor) == 1
    assert old_harbor[0].value == "likes Candle Harbor"
    assert store.fetch_messages_by_ids(harbor[0].source_ids)[0].content == correction
    restarted = ContextManager(db_path=tmp_path / "compound.db")
    assert len(restarted._get_store().fetch_states(
        scope="global", concept="likes for copper orchard", status="current",
    )) == 1


def test_live_runtime_defaults_semantic_but_respects_legacy_override(monkeypatch):
    from types import SimpleNamespace
    from PRIMORA.main import configure_runtime_defaults

    server = SimpleNamespace(base_url="http://127.0.0.1:8080/v1", alias="qwen3.5-9b")
    monkeypatch.delenv("PRIMORA_MEMORY_FORMATION_MODE")
    configure_runtime_defaults(server)
    assert os.environ["PRIMORA_MEMORY_FORMATION_MODE"] == "semantic"
    monkeypatch.setenv("PRIMORA_MEMORY_FORMATION_MODE", "legacy")
    configure_runtime_defaults(server)
    assert os.environ["PRIMORA_MEMORY_FORMATION_MODE"] == "legacy"


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


@pytest.mark.parametrize("dimension,value", [
    ("favorite instrument", "cello"),
    ("favorite travel style", "slow train journeys"),
])
def test_model_formed_explicit_favorite_slots_keep_category_and_value(
    tmp_path, dimension, value,
):
    class _FactProvider:
        def invoke(self, capability, payload):
            source = payload["messages"][-1]["content"]
            return PrimeResult.success({"content": json.dumps({
                "candidates": [{
                    "kind": "state", "concept": dimension, "value": value,
                    "claim_kind": "assertion", "actor": "user",
                    "scope": "global", "source_ids": [payload["source_message_id"]],
                    "source_quote": source, "confidence": 0.95,
                }]
            })})

    manager = ContextManager(
        db_path=tmp_path / "favorite.db", formation_provider=_FactProvider(),
    )
    message = f"My {dimension} is {value}."
    manager.reconcile([{"role": "user", "content": message}], "s")
    states = manager._get_store().fetch_states(
        scope="global", concept=dimension, status="current",
    )
    assert len(states) == 1
    assert states[0].value == value
    assert manager._get_store().fetch_messages_by_ids(states[0].source_ids)[0].content == message
