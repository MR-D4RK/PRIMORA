"""Slot registry and plain-sentence rendering (Stage 3, Step 1).

The contract under test: personal facts land in specific, separate
slots instead of colliding; learned favorite <x> slots are derived by
the normalization rule and exist only as concepts in states (no
registry table); rules are protected from ordinary statements and
retire only on an explicit revocation; personal facts are global; and
the model sees plain sentences - never a "[state ...]" machine format
and never a superseded value.

Conventions follow the repo: stdlib only, temp SQLite files, injected
word-based token counter, no live model.
"""
import os
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import BudgetConfig, ContextManager

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]


def _db():
    _SEQ[0] += 1
    return os.path.join(_TEST_TMPDIR.name, "slots%d.db" % _SEQ[0])


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _budget(**kwargs):
    params = dict(input_target=200, input_ceiling=300,
                  output_reserve=20, total_budget=320)
    params.update(kwargs)
    return BudgetConfig(**params)


def _say(mgr, hist, sid, text, budget=None, timestamp=None):
    msg = {"role": "user", "content": text}
    if timestamp is not None:
        msg["timestamp"] = timestamp
    hist.append(msg)
    return mgr.build(list(hist), sid, token_counter=_counter,
                     max_new_tokens=5, budget=budget or _budget())


def _currents(mgr, concept=None):
    """All current rows across session and global scope."""
    store = mgr._get_store()
    rows = store.fetch_states(status="current")
    if concept is not None:
        rows = [r for r in rows if r.concept == concept]
    return rows


def _facts(mgr):
    return sorted((s.concept, s.value) for s in _currents(mgr))


def _superseded(mgr, concept=None):
    rows = mgr._get_store().fetch_states(status="superseded")
    if concept is not None:
        rows = [r for r in rows if r.concept == concept]
    return rows


def _joined(view):
    return "\n".join(m["content"] for m in view.messages)


def _state_lines(view):
    """Model-visible state sentences: system messages, minus session info."""
    return [m["content"] for m in view.messages
            if m.get("role") == "system"
            and not m["content"].startswith("[session ")]


# 1. Names land in separate slots ---------------------------------------------
def test_first_and_last_name_are_separate_slots():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "slot-sess-a", "My first name is Amir.")
    _say(mgr, hist, "slot-sess-a", "My last name is Smith.")
    assert _facts(mgr) == [
        ("first name", "Amir"),
        ("last name", "Smith"),
    ]
    rows = _currents(mgr)
    assert all(r.scope == "global" for r in rows)


# 2. Unrelated facts all survive ------------------------------------------------
def test_pizza_darkmode_tesla_tacos_all_survive():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "slot-sess-a", "I love pizza.")
    _say(mgr, hist, "slot-sess-a", "I prefer dark mode.")
    _say(mgr, hist, "slot-sess-a", "My favorite car is a Tesla.")
    _say(mgr, hist, "slot-sess-a", "I love tacos.")
    assert _facts(mgr) == [
        ("favorite car", "a Tesla"),
        ("likes for pizza", "loves pizza"),
        ("likes for tacos", "loves tacos"),
        ("preference for dark mode", "dark mode"),
    ]
    assert _superseded(mgr) == []


# 3. Sentiment change retires only its own object -------------------------------
def test_hate_retires_love_for_same_object_only():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "slot-sess-a", "I love pizza.")
    _say(mgr, hist, "slot-sess-a", "I love tacos.")
    _say(mgr, hist, "slot-sess-a", "I hate pizza.")
    assert _facts(mgr) == [
        ("likes for pizza", "hates pizza"),
        ("likes for tacos", "loves tacos"),
    ]
    old = _superseded(mgr, "likes for pizza")
    assert [(s.concept, s.value, s.status) for s in old] == [
        ("likes for pizza", "loves pizza", "superseded")]
    assert _superseded(mgr, "likes for tacos") == []
    current = _currents(mgr, "likes for pizza")[0]
    assert current.supersedes_id == old[0].state_id


# 4. Single-value slot keeps one current and its history ------------------------
def test_favorite_color_supersedes_and_keeps_history():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "slot-sess-a", "My favorite color is yellow.")
    _say(mgr, hist, "slot-sess-a", "My favorite color is blue.")
    rows = _currents(mgr, "favorite color")
    assert [(r.scope, r.value) for r in rows] == [("global", "blue")]
    old = _superseded(mgr, "favorite color")
    assert [(s.value, s.status) for s in old] == [
        ("yellow", "superseded")]
    assert rows[0].supersedes_id == old[0].state_id


# 5. Favorite spelling aliases normalize to one slot ---------------------------
def test_fav_colour_lands_in_the_same_slot():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "slot-sess-a", "My favorite color is yellow.")
    _say(mgr, hist, "slot-sess-a", "My fav colour is red.")
    rows = _currents(mgr, "favorite color")
    assert [(r.scope, r.value) for r in rows] == [("global", "red")]
    assert [(s.value, s.status) for s in _superseded(mgr, "favorite color")] == [
        ("yellow", "superseded")]


# 6. Learned favorite slot from a fresh noun ------------------------------------
def test_learned_favorite_slot_from_fresh_noun():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "slot-sess-a", "My favorite food is sushi.")
    assert _facts(mgr) == [("favorite food", "sushi")]
    row = _currents(mgr, "favorite food")[0]
    assert row.scope == "global" and row.value == "sushi"
    # The learned slot is derived, not registered: it exists only as a
    # concept in states, and it renders as its own sentence.
    res = _say(mgr, hist, "slot-sess-a", "What is my favorite food?",
               budget=_budget(input_target=40, input_ceiling=100))
    assert "The user's favorite food is sushi." in _joined(res.view)


# 7a. Ordinary statements never retire a rule ------------------------------------
def test_rule_survives_later_preference_and_ordinary_corrections():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "slot-sess-a", "Always be concise.")
    _say(mgr, hist, "slot-sess-a", "I prefer verbose answers.")
    assert ("instruction for be concise", "be concise") in _facts(mgr)
    assert ("preference for verbose answers", "verbose answers") in _facts(mgr)
    # An ordinary invalidation may not retire the rule; it is held visible.
    res = _say(mgr, hist, "slot-sess-a",
               "My instruction is no longer correct.")
    assert ("instruction for be concise", "be concise") in _facts(mgr)
    assert any(d.startswith("rule_protected") for d in res.view.diagnostics)
    assert _superseded(mgr, "instruction for be concise") == []
    pend = mgr._get_store().fetch_candidates(status="pending")
    assert any(p.proposed_concept == "instruction for be concise"
               for p in pend)


# 7b. Only an explicit revocation retires a rule --------------------------------
def test_explicit_revocation_retires_the_rule():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "slot-sess-a", "Always be concise.")
    _say(mgr, hist, "slot-sess-a", "I prefer verbose answers.")
    _say(mgr, hist, "slot-sess-a", "Stop doing being concise.")
    assert ("instruction for be concise", "be concise") not in _facts(mgr)
    assert ("preference for verbose answers", "verbose answers") in _facts(mgr)
    old = _superseded(mgr, "instruction for be concise")
    assert [(s.concept, s.status) for s in old] == [
        ("instruction for be concise", "superseded")]
    # Bare revocation with a single rule also works.
    mgr2 = ContextManager(db_path=_db())
    hist2 = []
    _say(mgr2, hist2, "slot-sess-b", "Never store API keys.")
    _say(mgr2, hist2, "slot-sess-b", "Forget that rule.")
    assert _currents(mgr2, "instruction for never store api keys") == []
    assert _superseded(mgr2, "instruction for never store api keys")


# 8. Personal facts are visible from a new session ------------------------------
def test_personal_facts_visible_from_new_session():
    mgr = ContextManager(db_path=_db())
    _say(mgr, [], "slot-sess-a", "My first name is Amir.")
    res = _say(mgr, [], "slot-sess-b", "What is my first name?")
    joined = _joined(res.view)
    assert "The user's first name is Amir." in joined, joined


# 9. The model sees plain sentences, never the machine format --------------------
def test_rendered_block_is_plain_sentences_only():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "slot-sess-a", "My favorite color is yellow.")
    _say(mgr, hist, "slot-sess-a", "My favorite color is blue.")
    _say(mgr, hist, "slot-sess-a", "My first name is Amir.")
    res = _say(mgr, hist, "slot-sess-a",
               "What is my favorite color and my first name?")
    lines = _state_lines(res.view)
    assert lines, res.view.messages
    joined = "\n".join(lines)
    assert "[state" not in joined, joined
    assert "src:" not in joined, joined
    assert "The user's favorite color is blue." in lines, joined
    assert "The user's first name is Amir." in lines, joined
    # The heading carries the usage guidance.
    assert any(line.startswith("The user's facts") for line in lines), joined
    assert any("use only when relevant" in line for line in lines), joined
    # The superseded value never reaches the model as state (raw history
    # may still quote the user's own words; state lines may not).
    assert not any("yellow" in line for line in lines), joined


# 10. Shuffled-order survival invariant ------------------------------------------
def test_shuffled_order_keeps_the_same_surviving_facts():
    base = [
        "I love pizza.",
        "I prefer dark mode.",
        "My favorite car is a Tesla.",
        "I love tacos.",
        "My first name is Amir.",
        "Always be concise.",
    ]
    orders = [
        base,
        list(reversed(base)),
        [base[i] for i in (3, 0, 5, 1, 4, 2)],
        [base[i] for i in (2, 5, 1, 4, 0, 3)],
    ]
    expected = [
        ("favorite car", "a Tesla"),
        ("first name", "Amir"),
        ("instruction for be concise", "be concise"),
        ("likes for pizza", "loves pizza"),
        ("likes for tacos", "loves tacos"),
        ("preference for dark mode", "dark mode"),
    ]
    for order in orders:
        mgr = ContextManager(db_path=_db())
        hist = []
        for text in order:
            _say(mgr, hist, "slot-sess-a", text)
        assert _facts(mgr) == expected, order
        assert _superseded(mgr) == [], order


# Extras ------------------------------------------------------------------------
def test_structureless_personal_statement_becomes_a_note():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "slot-sess-a", "I'm allergic to penicillin.")
    assert _facts(mgr) == [
        ("notes for allergic to penicillin", "allergic to penicillin")]
    row = _currents(mgr)[0]
    assert row.scope == "global"
    # Deduplicated on repetition.
    _say(mgr, hist, "slot-sess-a", "I am allergic to penicillin.")
    assert _currents(mgr, "notes for allergic to penicillin") != []
    assert _superseded(mgr, "notes for allergic to penicillin") == []


def test_hedged_slot_statement_stays_pending_until_confirmed():
    from PRIMORA.prime.context import state as state_mod
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "slot-sess-a", "I think my favorite color is blue.")
    assert _currents(mgr, "favorite color") == []
    pend = mgr._get_store().fetch_candidates(status="pending")
    assert any(p.proposed_concept == "favorite color" for p in pend)
    diags = state_mod.confirm_pending(
        mgr._get_store(), "slot-sess-a", pend[0].candidate_id)
    assert any("candidate_confirmed" in d for d in diags), diags
    rows = _currents(mgr, "favorite color")
    assert [(r.scope, r.value) for r in rows] == [("global", "blue")]


# Name phrasings --------------------------------------------------------------
def test_name_and_last_name_phrasings_land_in_slots():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "slot-sess-a", "My name is Amir.")
    _say(mgr, hist, "slot-sess-a", "My last name is Shahbazi.")
    assert _facts(mgr) == [
        ("first name", "Amir"),
        ("last name", "Shahbazi"),
    ]
    # The legacy bare "name" concept is never created.
    assert all(r.concept != "name" for r in _currents(mgr))


def test_two_word_name_fills_both_slots_at_once():
    mgr = ContextManager(db_path=_db())
    _say(mgr, [], "slot-sess-a", "My name is Amir Shahbazi.")
    assert _facts(mgr) == [
        ("first name", "Amir"),
        ("last name", "Shahbazi"),
    ]


def test_call_me_and_called_phrasings():
    mgr = ContextManager(db_path=_db())
    _say(mgr, [], "slot-sess-a", "Call me Amir.")
    assert _facts(mgr) == [("first name", "Amir")]
    _say(mgr, [], "slot-sess-a", "I'm called Bob.", timestamp=1100.0)
    assert _facts(mgr) == [("first name", "Bob")]
    old = _superseded(mgr, "first name")
    assert [(s.value, s.status) for s in old] == [("Amir", "superseded")]


def test_i_am_adjective_creates_nothing():
    mgr = ContextManager(db_path=_db())
    res = _say(mgr, [], "slot-sess-a", "I am tired.")
    assert _currents(mgr) == []
    pend = mgr._get_store().fetch_candidates(status="pending")
    assert pend == []
    assert not any(d.startswith("state_") for d in res.view.diagnostics)


# Location phrasings ----------------------------------------------------------
def test_lived_in_then_moved_to_supersedes_with_history():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "slot-sess-a", "I live in Karaj.")
    _say(mgr, hist, "slot-sess-a", "I moved to Tehran.")
    rows = _currents(mgr, "home location")
    assert [(r.scope, r.value) for r in rows] == [("global", "Tehran")]
    old = _superseded(mgr, "home location")
    assert [(s.value, s.status) for s in old] == [("Karaj", "superseded")]
    assert rows[0].supersedes_id == old[0].state_id


def test_im_living_in_is_home_location():
    mgr = ContextManager(db_path=_db())
    _say(mgr, [], "slot-sess-a", "I'm living in Karaj.")
    assert _facts(mgr) == [("home location", "Karaj")]


def test_im_from_is_a_note_not_residence():
    mgr = ContextManager(db_path=_db())
    _say(mgr, [], "slot-sess-a", "I'm from Karaj.")
    assert _facts(mgr) == [
        ("notes for from karaj", "from Karaj")]
    assert _currents(mgr, "home location") == []


def test_dont_live_in_retires_only_matching_value():
    mgr = ContextManager(db_path=_db())
    _say(mgr, [], "slot-sess-a", "I live in Karaj.")
    _say(mgr, [], "slot-sess-a", "I don't live in Karaj.", timestamp=1100.0)
    assert _currents(mgr, "home location") == []
    assert [(s.value, s.status) for s in _superseded(mgr, "home location")] == [
        ("Karaj", "superseded")]
    # A negation naming a different city retires nothing; it is held
    # pending with a reason.
    mgr2 = ContextManager(db_path=_db())
    _say(mgr2, [], "slot-sess-b", "I live in Karaj.")
    res = _say(mgr2, [], "slot-sess-b", "I don't live in Tehran.",
               timestamp=1100.0)
    rows = _currents(mgr2, "home location")
    assert [(r.scope, r.value) for r in rows] == [("global", "Karaj")]
    assert _superseded(mgr2, "home location") == []
    pend = mgr2._get_store().fetch_candidates(status="pending")
    assert any(p.proposed_value == "Tehran" for p in pend)
    assert any(d.startswith("candidate_pending_reason") for d in
               res.view.diagnostics)


def test_what_is_my_name_renders_name_sentences():
    mgr = ContextManager(db_path=_db())
    _say(mgr, [], "slot-sess-a", "My name is Amir Shahbazi.")
    res = _say(mgr, [], "slot-sess-b", "What is my name?",
               budget=_budget(input_target=40, input_ceiling=100))
    lines = _state_lines(res.view)
    assert "The user's first name is Amir." in lines, lines
    assert "The user's last name is Shahbazi." in lines, lines


if __name__ == "__main__":
    test_first_and_last_name_are_separate_slots()
    test_pizza_darkmode_tesla_tacos_all_survive()
    test_hate_retires_love_for_same_object_only()
    test_favorite_color_supersedes_and_keeps_history()
    test_fav_colour_lands_in_the_same_slot()
    test_learned_favorite_slot_from_fresh_noun()
    test_rule_survives_later_preference_and_ordinary_corrections()
    test_explicit_revocation_retires_the_rule()
    test_personal_facts_visible_from_new_session()
    test_rendered_block_is_plain_sentences_only()
    test_shuffled_order_keeps_the_same_surviving_facts()
    test_structureless_personal_statement_becomes_a_note()
    test_hedged_slot_statement_stays_pending_until_confirmed()
    test_name_and_last_name_phrasings_land_in_slots()
    test_two_word_name_fills_both_slots_at_once()
    test_call_me_and_called_phrasings()
    test_i_am_adjective_creates_nothing()
    test_lived_in_then_moved_to_supersedes_with_history()
    test_im_living_in_is_home_location()
    test_im_from_is_a_note_not_residence()
    test_dont_live_in_retires_only_matching_value()
    test_what_is_my_name_renders_name_sentences()
    print("All slot registry tests passed!")
