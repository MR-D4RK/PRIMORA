"""Verify that memory preserves user-stated preference strength."""
import json

import pytest

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context import state as state_mod
from PRIMORA.prime.context.store import ContextStore
from PRIMORA.prime.context.formation import apply_proposals
from PRIMORA.prime.contract import PrimeResult


@pytest.mark.parametrize(("user_text", "expected"), [
    ("I enjoy sketching maps.", "enjoys sketching maps"),
    ("I like paper lanterns.", "likes paper lanterns"),
    ("I love copper sculptures.", "loves copper sculptures"),
    ("I adore music.", "adores music"),
])
def test_deterministic_strength(user_text, expected):
    memories = state_mod.extract_candidates(user_text, "user")
    assert any(item.value == expected for item in memories)


class _Provider:
    def __init__(self, value):
        self.value = value

    def invoke(self, capability, payload):
        return PrimeResult.success({"content": json.dumps({"candidates": [{
            "kind": "preference",
            "concept": "copper sculptures",
            "value": self.value,
            "scope": "global",
            "source_ids": [payload["source_message_id"]],
            "source_quote": payload["messages"][-1]["content"],
        }]})})


@pytest.mark.parametrize(("original", "proposal", "admit"), [
    ("I enjoy copper sculptures.", "loves copper sculptures", False),
    ("I like copper sculptures.", "loves copper sculptures", False),
    ("I enjoy copper sculptures.", "enjoys copper sculptures", True),
    ("I love copper sculptures.", "loves copper sculptures", True),
])
def test_typed_strength_matches_user(tmp_path, original, proposal, admit):
    store = ContextStore(tmp_path / "strength.db")
    source_id, _, _ = store.append("s", 0, "user", original)
    result = apply_proposals(
        store, "s", store.fetch_message(source_id), _Provider(proposal),
    )
    assert bool(result.admitted_ids) is admit
    if not admit:
        assert "proposal_rejected:preference_strength_not_source_supported" in result.diagnostics


def test_manager_preserves_enjoy_in_sqlite(tmp_path):
    manager = ContextManager(db_path=tmp_path / "live.db")
    manager.reconcile([
        {"role": "user", "content": "I enjoy the sound of rainfall when I read."}
    ], "s")
    states = manager._get_store().fetch_states(scope="global", status="current")
    assert any("enjoys the sound of rainfall" in row.value for row in states)
    assert not any("loves" in row.value for row in states)
