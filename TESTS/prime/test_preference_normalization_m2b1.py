"""Milestone 2 Batch 1: safe canonicalization and evidence deduplication."""

import pytest

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.state import extract_candidates
from TESTS.prime.test_slots import _db, _say, _currents, _superseded


def test_shade_phrasings_use_one_canonical_preference():
    variants = [
        "I love purple whether it's bright or not",
        "I love purple regardless of its shade",
        "I love purple no matter what shade",
    ]
    candidates = [extract_candidates(text, "user") for text in variants]
    assert all(len(x) == 1 for x in candidates)
    assert all(x[0].concept == "likes for purple" for x in candidates)
    assert all(x[0].value == "loves purple in any shade" for x in candidates)
    assert all(x[0].scope == "global" for x in candidates)


def test_multiple_paraphrases_attach_evidence_without_fake_changes():
    path = _db()
    manager = ContextManager(db_path=path)
    history = []
    session = "temporal-pref-one"
    for message in ["I love purple whether its bright or not",
                    "I love purple regardless of its shade",
                    "I love purple no matter which shade"]:
        _say(manager, history, session, message)
    current = _currents(manager, "likes for purple")
    assert len(current) == 1
    assert current[0].value == "loves purple in any shade"
    assert len(current[0].source_ids) == 3
    assert _superseded(manager, "likes for purple") == []
    assert len(set(current[0].source_ids)) == 3
    # A restart still sees the same state and evidence links.
    reopened = ContextManager(db_path=path)
    again = _currents(reopened, "likes for purple")
    assert again[0].state_id == current[0].state_id
    assert again[0].source_ids == current[0].source_ids


def test_stronger_preference_preserves_earlier_record_and_origin():
    manager = ContextManager(db_path=_db())
    history = []
    _say(manager, history, "temporal-pref-two", "I love purple")
    earlier = _currents(manager, "likes for purple")[0]
    _say(manager, history, "temporal-pref-two", "I love purple regardless of its shade")
    later = _currents(manager, "likes for purple")[0]
    assert later.value == "loves purple in any shade"
    assert later.supersedes_id == earlier.state_id
    assert len(later.source_ids) == 1
    old = _superseded(manager, "likes for purple")
    assert [row.state_id for row in old] == [earlier.state_id]
    assert old[0].source_ids == earlier.source_ids


def test_generic_restating_does_not_claim_evidence_for_shade_detail():
    manager = ContextManager(db_path=_db())
    history = []
    _say(manager, history, "temporal-pref-three", "I love purple regardless of its shade")
    state = _currents(manager, "likes for purple")[0]
    _say(manager, history, "temporal-pref-three", "I love purple")
    after = _currents(manager, "likes for purple")[0]
    assert after.state_id == state.state_id
    assert after.source_ids == state.source_ids
    assert not _superseded(manager, "likes for purple")


def test_ambiguous_trailing_clause_not_silently_trimmed():
    result = extract_candidates(
        "I love purple regardless of its shade, and I used to hate violet", "user"
    )
    assert not any(c.concept == "likes for purple" and
                   c.value == "loves purple in any shade" for c in result)
    # Never corrupt titles that incidentally contain 'too' or 'shade'.
    result = extract_candidates("I love the song Me Too", "user")
    assert result[0].concept == "likes for song me too"


def test_evidence_rejects_assistant_origin_and_is_idempotent():
    manager = ContextManager(db_path=_db())
    history = []
    _say(manager, history, "temporal-pref-four", "I love purple")
    store = manager._get_store()
    current = _currents(manager, "likes for purple")[0]
    assert not store.add_state_evidence(current.state_id, current.source_ids[0])
    assistant_id, _, _ = store.append("temporal-pref-four", 999, "assistant", "I love purple")
    with pytest.raises(ValueError, match="user message"):
        store.add_state_evidence(current.state_id, assistant_id)
    unchanged = _currents(manager, "likes for purple")[0]
    assert unchanged.source_ids == current.source_ids


def test_shade_preference_with_color_comparison_keeps_both_claims():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "temporal-pref-five",
         "I love purple regardless of its shade, and it gives me a similar feeling to violet")
    likes = _currents(mgr, "likes for purple")
    feels = _currents(mgr, "color feeling for purple")
    assert [(v.concept, v.value) for v in likes] == [
        ("likes for purple", "loves purple in any shade")]
    assert len(feels) == 1
    assert "violet" in feels[0].value.lower()
    assert likes[0].source_ids == feels[0].source_ids
