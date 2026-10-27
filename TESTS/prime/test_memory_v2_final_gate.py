"""Final-gate trust-boundary regressions for Memory V2."""

import json

import pytest

from PRIMORA.learning import LearningOutcome, StructuredCandidate
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.formation import apply_proposals
from PRIMORA.prime.context.store import ContextStore
from PRIMORA.prime.contract import PrimeResult


class _Jev:
    def evaluate(self, request):
        return LearningOutcome.confirmed([
            StructuredCandidate(
                concept="favorite color", value="amber",
                memory_type="PREFERENCE", scope="global",
            )
        ])


class _Provider:
    def __init__(self, candidate):
        self.candidate = candidate

    def invoke(self, capability, payload):
        return PrimeResult.success({
            "content": json.dumps({"candidates": [self.candidate]})
        })


def test_user_learning_requires_an_existing_user_source(tmp_path):
    manager = ContextManager(db_path=tmp_path / "gate.db")
    rejected = manager.learn("The user likes amber", "user", _Jev(), session_id="s")
    assert rejected["status"] == "rejected"
    assert "matching_user_source" in rejected["reason"]
    assert manager._get_store().fetch_states(status="current") == []
    assert manager._get_store().fetch_latest("s", 10) == []


def test_user_learning_reuses_the_immutable_user_source(tmp_path):
    manager = ContextManager(db_path=tmp_path / "gate.db")
    source, *_ = manager._get_store().append(
        "s", 0, "user", "I like amber", timestamp=10,
    )
    # The request text must match the source statement exactly after
    # normalization; an assistant paraphrase cannot become user authority.
    rejected = manager.learn("The user likes amber", "user", _Jev(), session_id="s")
    assert rejected["status"] == "rejected"
    learned = manager.learn("I like amber", "user", _Jev(), session_id="s")
    assert learned["status"] == "learned"
    state = manager._get_store().fetch_states(scope="global", status="current")
    assert state and state[0].source_ids == [source]
    assert manager._get_store().fetch_message(source).role == "user"


def test_store_rejects_user_authority_state_from_assistant_source(tmp_path):
    store = ContextStore(tmp_path / "gate.db")
    assistant, *_ = store.append("s", 0, "assistant", "Your favorite color is amber")
    with pytest.raises(ValueError, match="user-authority provenance"):
        store.create_state(
            "s", "global", "favorite color", "amber", [assistant],
            authority="user", memory_type="PREFERENCE",
        )


def test_typed_reason_must_be_user_source_supported(tmp_path):
    store = ContextStore(tmp_path / "gate.db")
    source, *_ = store.append("s", 0, "user", "I like amber")
    message = store.fetch_message(source)
    provider = _Provider({
        "kind": "preference", "concept": "amber", "value": "loves amber",
        "claim_kind": "assertion", "actor": "user", "scope": "global",
        "source_ids": [source], "source_quote": "I like amber",
        "rationale": "because it improves my memory of childhood summers",
    })
    result = apply_proposals(store, "s", message, provider)
    assert not result.admitted_ids
    assert any("rationale_not_source_supported" in item
               for item in result.diagnostics)
