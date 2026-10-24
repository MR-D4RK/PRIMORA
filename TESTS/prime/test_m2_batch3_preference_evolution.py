"""M2 Batch 3: time-stamped preference evolution in existing states.

All tests use temporary databases. Never access a production profile DB.
"""

import pytest

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.state import extract_candidates
from TESTS.prime.test_slots import _db, _say, _currents


def test_natural_new_likes_and_stops_use_same_global_slot():
    for phrase in (
        "I recently started liking the Koenigsegg Regera too",
        "I've recently started liking the Koenigsegg Regera",
        "I began to love the Koenigsegg Regera",
    ):
        parsed = extract_candidates(phrase, "user")
        assert len(parsed) == 1
        assert parsed[0].concept == "likes for koenigsegg regera"
        assert parsed[0].scope == "global"
        assert parsed[0].mtype == "PREFERENCE"

    for phrase in (
        "I no longer like the Koenigsegg Regera",
        "I stopped liking the Koenigsegg Regera",
    ):
        parsed = extract_candidates(phrase, "user")
        assert len(parsed) == 1
        assert parsed[0].concept == "likes for koenigsegg regera"
        assert parsed[0].value.startswith("dislikes ")


def test_recent_like_is_one_historical_event_even_with_article_and_repeat():
    path = _db()
    manager = ContextManager(db_path=path, hybrid_retrieval=False)
    history = []
    _say(manager, history, "new-car", "I just started liking Regera", timestamp=1000)
    _say(manager, history, "new-car", "I've recently started liking the Regera", timestamp=2000)
    rows = manager.history("global", "likes for regera")
    assert len(rows) == 1
    assert rows[0].status == "current"
    assert rows[0].valid_from == 1000
    assert rows[0].observed_at == 1000
    assert len(rows[0].source_ids) == 2
    assert rows[0].value == "likes Regera"
    assert len([r for r in manager.preference_events() if r.concept == "likes for regera"]) == 1
    fresh = ContextManager(db_path=path, hybrid_retrieval=False)
    assert fresh.load_transcript("brand-new-session") == []
    assert len(fresh.preference_events()) == 1


def test_stopping_and_restarting_preference_preserves_change_history():
    mgr = ContextManager(db_path=_db(), hybrid_retrieval=False)
    history = []
    _say(mgr, history, "car-timeline", "I love Regera", timestamp=1000)
    _say(mgr, history, "car-timeline", "I no longer like Regera", timestamp=2000)
    _say(mgr, history, "car-timeline", "I've started loving Regera", timestamp=3000)
    rows = mgr.history("global", "likes for regera")
    assert len(rows) == 3
    assert [r.status for r in rows] == ["superseded", "superseded", "current"]
    assert [r.valid_from for r in rows] == [1000, 2000, 3000]
    assert [r.valid_until for r in rows] == [2000, 3000, None]
    assert [r.value.startswith(k) for r, k in zip(rows, ["loves", "dislikes", "loves"])] == [True]*3
    assert mgr.as_of("global", "likes for regera", 1500).value.startswith("loves")
    assert mgr.as_of("global", "likes for regera", 2500).value.startswith("dislikes")
    assert mgr.current("global", "likes for regera").value.startswith("loves")
    assert mgr.previous("global", "likes for regera").value.startswith("dislikes")
    assert [x.valid_from for x in mgr.preference_events(since=2000, until=4000)] == [2000, 3000]


def test_favorite_now_supersedes_old_and_time_slice_is_precise():
    mgr = ContextManager(db_path=_db(), hybrid_retrieval=False)
    history = []
    _say(mgr, history, "fav-changes", "My favorite car is Corvette C8", timestamp=1000)
    _say(mgr, history, "fav-changes", "My favorite car is Regera now", timestamp=2000)
    rows = mgr.history("global", "favorite car")
    assert [(r.value, r.status) for r in rows] == [
        ("Corvette C8", "superseded"), ("Regera", "current")
    ]
    assert rows[1].supersedes_id == rows[0].state_id
    assert mgr.as_of("global", "favorite car", 1500).value == "Corvette C8"
    assert mgr.as_of("global", "favorite car", 2500).value == "Regera"
    assert [r.value for r in mgr.preference_events(since=1500, until=2500)] == ["Regera"]


def test_timeline_excludes_assistant_claims_and_other_subjects():
    mgr = ContextManager(db_path=_db(), hybrid_retrieval=False)
    history = []
    _say(mgr, history, "safe", "I love Regera", timestamp=1000)
    _say(mgr, history, "safe", "My timezone is GMT", timestamp=2000)
    _say(mgr, history, "safe", "I think I love Apollo IE", timestamp=3000)
    assert [r.concept for r in mgr.preference_events()] == ["likes for regera"]
    with pytest.raises(ValueError):
        mgr.preference_events(since=3000, until=2000)
    assert mgr.preference_events(since=1000, until=1000) == []


def test_questions_and_uncertain_referents_never_create_timeline_events():
    for text in (
        "What cars did I start liking recently?",
        "I started liking it",
        "I no longer like that",
        "Maybe I started liking Regera",
        "Did I stop liking Regera?",
    ):
        cands = extract_candidates(text, "user")
        assert not any(c.concept == "likes for regera" and c.auto_confirm for c in cands)

def test_weak_rephrasing_does_not_downgrade_existing_love():
    mgr = ContextManager(db_path=_db(), hybrid_retrieval=False)
    hist = []
    _say(mgr, hist, "strength", "I love the Regera", timestamp=1000)
    _say(mgr, hist, "strength", "I recently started liking Regera", timestamp=2000)
    rows = mgr.history("global", "likes for regera")
    assert len(rows) == 1
    assert rows[0].value == "loves the Regera"
    assert len(rows[0].source_ids) == 2
