"""Milestone 2 Batch 4 — read-only timeline evidence in normal context builds."""
from datetime import datetime, timezone

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.timeline import parse_request
from TESTS.prime.test_slots import _budget, _db, _joined, _say

BIG = _budget(input_target=480, input_ceiling=850, total_budget=900,
              output_reserve=30)


def test_intent_is_explicit_and_category_specific():
    t = 1760000000
    assert parse_request("which cars have i started liking most recently?", now=t).mode == "started"
    assert parse_request("what new songs have I started liking recently?", now=t).category == "song"
    assert parse_request("how have my favorite cars changed?", now=t).mode == "changes"
    assert parse_request("what things have I liked lately?", now=t).category is None
    assert parse_request("what are my favorite cars", now=t) is None
    assert parse_request("i recently started liking regera", now=t) is None
    assert parse_request("tell me about the newest model of cars", now=t) is None
    last = parse_request("which songs did I like in the last 7 days?", now=t)
    assert last.since == t - 7*86400
    month = parse_request("how did my music preferences change this month?", now=t)
    dt = datetime.fromtimestamp(t, tz=timezone.utc)
    assert month.since == datetime(dt.year, dt.month, 1,
                                  tzinfo=timezone.utc).timestamp()


def test_recent_car_starts_exclude_old_favorites_and_other_interests():
    path = _db()
    mgr = ContextManager(db_path=path, hybrid_retrieval=False)
    hist = []
    sid = "source-1"
    _say(mgr, hist, sid, "what are my favorite cars?", timestamp=1000)
    _say(mgr, hist, sid, "I love Apollo IE", timestamp=1100)
    _say(mgr, hist, sid, "I really love Corvette C8", timestamp=1200)
    _say(mgr, hist, sid, "I've recently started liking Koenigsegg Regera", timestamp=1300)
    _say(mgr, hist, sid, "What are my favorite songs?", timestamp=1350)
    _say(mgr, hist, sid, "I've recently started liking synthwave", timestamp=1400)

    fresh = ContextManager(db_path=path, hybrid_retrieval=False)
    assert fresh.load_transcript("brand-new") == []
    view = _say(fresh, [], "brand-new",
                "Which cars have I started liking most recently?", budget=BIG).view
    facts = [msg["content"] for msg in view.messages if msg["role"] == "system"]
    timeline = next(x for x in facts if x.startswith("Verified user preference timeline"))
    assert "koenigsegg regera" in timeline.lower()
    assert "Apollo IE" not in timeline
    assert "Corvette C8" not in timeline
    assert "synthwave" not in timeline
    assert "not necessarily when" in timeline
    assert "source IDs" in timeline


def test_change_history_keeps_old_favorites_not_new_discoveries():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    h = []
    _say(m, h, "source-change", "My favorite car is Corvette C8", timestamp=1000)
    _say(m, h, "source-change", "My favorite car is Regera now", timestamp=2000)
    fresh = ContextManager(db_path=m._db_path, hybrid_retrieval=False)
    view = _say(fresh, [], "changes", "How have my favorite cars changed?", budget=BIG).view
    text = _joined(view)
    assert "Corvette C8" in text and "Regera" in text
    assert "superseded/historical" in text
    assert "current" in text
    assert text.index("Corvette C8") < text.index("Regera")


def test_unknown_timeline_cannot_fabricate_recent_likes():
    mgr = ContextManager(db_path=_db(), hybrid_retrieval=False)
    view = _say(mgr, [], "empty", "What songs did I start liking recently?", budget=BIG).view
    text = _joined(view)
    assert "No matching verified timeline events were found" in text


def test_current_favorites_do_not_trigger_timeline_or_change_existing_behavior():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    h = []
    _say(m, h, "original", "my favorite car is corvette c8", timestamp=1000)
    fresh = ContextManager(db_path=m._db_path, hybrid_retrieval=False)
    result = _say(fresh, [], "new", "what are my favorite cars?", budget=BIG).view
    assert "corvette c8" in _joined(result).lower()
    assert "Verified user preference timeline" not in _joined(result)


def test_repeated_preferences_remain_one_timeline_event():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    h = []
    _say(m, h, "generic", "I just started liking Regera", timestamp=1000)
    _say(m, h, "generic", "I've recently started liking the Regera", timestamp=2000)
    result = _say(m, h, "generic", "What are the latest things I liked?", budget=BIG).view
    timeline = [x["content"] for x in result.messages if x["role"] == "system" and
                x["content"].startswith("Verified user preference timeline")][0]
    assert timeline.count("likes for regera") == 1
    assert "1970-01-01" in timeline


def test_later_car_question_does_not_relabel_previous_song_like():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    h = []
    _say(m, h, "mixed", "What are my favorite songs?", timestamp=1000)
    _say(m, h, "mixed", "I recently started liking synthwave", timestamp=1100)
    _say(m, h, "mixed", "What are my favorite cars?", timestamp=1200)
    _say(m, h, "mixed", "I recently started liking Regera", timestamp=1300)
    fresh = ContextManager(db_path=m._db_path, hybrid_retrieval=False)
    view = _say(fresh, [], "fresh", "Which cars have I started liking recently?",
                budget=BIG).view
    timeline = next(x["content"] for x in view.messages
                    if x["content"].startswith("Verified user preference timeline"))
    assert "regera" in timeline.lower()
    assert "synthwave" not in timeline.lower()


def test_timeline_evidence_has_room_when_chat_history_is_long():
    m = ContextManager(db_path=_db(), hybrid_retrieval=False)
    source = []
    _say(m, source, "long-source", "what are my favorite cars?", timestamp=1000)
    _say(m, source, "long-source", "I recently started liking Regera", timestamp=1100)
    fresh = ContextManager(db_path=m._db_path, hybrid_retrieval=False)
    old = [{"role": "user", "content": "ordinary chat " + ("filler " * 45)}
           for _ in range(20)]
    old.append({"role": "user",
                "content": "What cars have I started liking recently?"})
    view = fresh.build(old, "brand-new", token_counter=lambda ms:
                       sum(2 + len(x["content"].split()) for x in ms),
                       budget=_budget(input_target=260, input_ceiling=400,
                                      total_budget=450, state_reserve=0)).view
    assert "Verified user preference timeline" in _joined(view)
    assert "likes for regera" in _joined(view)
