"""Milestone 2 Batch 2: grounded preference storage and session-consistent recall."""

import pytest

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.state import extract_candidates
from TESTS.prime.test_slots import _db, _say, _currents, _joined, _budget


def test_mustang_natural_statement_maps_to_stable_preference():
    statements = (
        "aurora i love ford mustand regardless of their producted year",
        "I love Ford Mustang regardless of the production year",
    )
    parsed = [extract_candidates(t, "user") for t in statements]
    assert all(len(c) == 1 for c in parsed)
    assert all(c[0].concept == "likes for ford mustang" for c in parsed)
    assert all(c[0].value == "loves ford mustang across model years"
               for c in parsed)


def test_classic_boss_preserves_uncertain_model():
    cands = extract_candidates("for instance the 1969 boss is my fav classic car", "user")
    assert len(cands) == 1
    assert cands[0].concept == "favorite classic car"
    assert cands[0].value == "1969 boss"
    assert "302" not in cands[0].value and "429" not in cands[0].value


def test_car_preferences_survive_fresh_session_without_old_transcript():
    path = _db()
    manager = ContextManager(db_path=path, hybrid_retrieval=False)
    history = []
    sid = "origin-car"
    _say(manager, history, sid, "what are my fav cars?")
    _say(manager, history, sid, "I like the ZR1 too")
    _say(manager, history, sid, "my favorite car is corvette c8")
    _say(manager, history, sid,
         "aurora i love ford mustand regardless of their producted year")
    _say(manager, history, sid, "for instance the 1969 boss is my fav classic car")
    assert len(_currents(manager, "likes for ford mustang")) == 1
    assert len(_currents(manager, "favorite classic car")) == 1
    assert len(_currents(manager, "likes for zr1")) == 1

    fresh = ContextManager(db_path=path, hybrid_retrieval=False)
    assert fresh.load_transcript("fresh-car") == []
    view = _say(fresh, [], "fresh-car", "What are my favorite cars?",
                budget=_budget(input_target=500, input_ceiling=800,
                               output_reserve=20, total_budget=820)).view
    text = _joined(view).lower()
    assert "zr1" in text
    assert "corvette c8" in text
    assert "1969 boss" in text
    assert "ford mustang" in text


def test_cross_session_current_discussion_does_not_inject_past():
    path = _db()
    orig = ContextManager(db_path=path, hybrid_retrieval=False)
    _say(orig, [], "old-chat", "i love synthwave music")
    fresh = ContextManager(db_path=path, hybrid_retrieval=False)
    assert fresh.load_transcript("brand-new-chat") == []
    res = _say(fresh, [], "brand-new-chat", "what were we talking about")
    text = _joined(res.view).lower()
    assert "this session" in text or "this conversation" in text
    assert "synthwave" not in text
    assert "[from another conversation]" not in text


def test_explicit_old_session_question_is_not_misclassified_as_current():
    from PRIMORA.prime.context.manager import _LOCAL_DISCUSSION_RE
    assert _LOCAL_DISCUSSION_RE.fullmatch("what were we talking about")
    assert not _LOCAL_DISCUSSION_RE.fullmatch("what did we discuss last time?")


def test_does_not_store_guessed_mustang_from_unrelated_sentence():
    for text in (
        "I saw Ford Mustand at a car meet",
        "Maybe the 1969 Boss is the most powerful",
        "Are 1969 Boss Mustangs rare?",
    ):
        cands = extract_candidates(text, "user")
        assert not any(c.concept == "favorite classic car" or
                       c.concept == "likes for ford mustang" for c in cands)
