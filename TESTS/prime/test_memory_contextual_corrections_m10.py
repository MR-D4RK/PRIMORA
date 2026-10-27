"""Domain-independent contextual choice formation and supersession.

All cases use disposable SQLite stores; no model server or production memory.
"""
from __future__ import annotations

import pytest

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.state import extract_candidates


@pytest.mark.parametrize(("initial", "updated", "topic", "old", "new"), [
    (
        "For travel notes I prefer the Cedar Notebook.",
        "I've changed my mind. I now prefer the Juniper Compass for travel notes. "
        "The Cedar Notebook was my old choice.",
        "travel notes", "Cedar Notebook", "Juniper Compass",
    ),
    (
        "For studio lighting I prefer the North Lamp.",
        "I now prefer the South Lamp for studio lighting.",
        "studio lighting", "North Lamp", "South Lamp",
    ),
    (
        "I prefer the Alder Layout for presentations.",
        "For presentations I prefer the Birch Layout.",
        "presentations", "Alder Layout", "Birch Layout",
    ),
])
def test_contextual_preference_correction_persists_history(
    tmp_path, initial, updated, topic, old, new,
):
    path = tmp_path / "corrected.db"
    first = ContextManager(db_path=path, hybrid_retrieval=False)
    first.reconcile([{"role": "user", "content": initial}], "session-one")
    concept = "preferred option for " + topic
    before = first._get_store().fetch_states(
        scope="global", concept=concept, status="current",
    )
    assert len(before) == 1 and before[0].value == old
    assert before[0].source_ids

    reopened = ContextManager(db_path=path, hybrid_retrieval=False)
    reopened.reconcile([{"role": "user", "content": updated}], "session-two")
    store = reopened._get_store()
    currents = store.fetch_states(scope="global", concept=concept, status="current")
    retired = store.fetch_states(scope="global", concept=concept, status="superseded")
    assert len(currents) == 1 and currents[0].value == new
    assert len(retired) == 1 and retired[0].value == old
    assert currents[0].supersedes_id == retired[0].state_id
    assert set(currents[0].source_ids).isdisjoint(retired[0].source_ids)
    evidence = store.fetch_messages_by_ids(currents[0].source_ids)
    assert len(evidence) == 1 and evidence[0].role == "user"
    assert evidence[0].content == updated
    assert not any(
        "changed my mind" in state.concept
        for state in store.fetch_states(scope="global", status="current")
    )


def test_different_contexts_never_supersede_each_other(tmp_path):
    manager = ContextManager(db_path=tmp_path / "independent.db")
    manager.reconcile([
        {"role": "user", "content": "For planning I prefer the Maple Board."},
        {"role": "user", "content": "For sketching I prefer the Willow Pad."},
        {"role": "user", "content": "I now prefer the Elm Board for planning."},
    ], "one")
    states = manager._get_store().fetch_states(scope="global", status="current")
    lookup = {state.concept: state.value for state in states}
    assert lookup["preferred option for planning"] == "Elm Board"
    assert lookup["preferred option for sketching"] == "Willow Pad"


@pytest.mark.parametrize("message", [
    "If I ever travel, I might prefer the Juniper Compass for travel notes.",
    "I could prefer the Juniper Compass for travel notes.",
    "For travel notes, which notebook should I prefer?",
    "I've changed my mind. I now prefer the Juniper Compass for travel notes. "
    "Someone else told me a story.",
])
def test_unasserted_or_unparsed_context_does_not_create_current_memory(
    tmp_path, message,
):
    assert not any(
        candidate.concept.startswith("preferred option for ") and candidate.auto_confirm
        for candidate in extract_candidates(message, "user")
    )
    manager = ContextManager(db_path=tmp_path / "unknown.db")
    manager.reconcile([{"role": "user", "content": message}], "one")
    states = manager._get_store().fetch_states(scope="global", status="current")
    assert not any(row.concept.startswith("preferred option for ") for row in states)
