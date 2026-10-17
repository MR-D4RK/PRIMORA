"""Fact-collision regressions: unrelated statements never share one
newest-wins slot.

Slot-routed personal facts land in their own registry concept ("I love
pizza" -> "likes for pizza"; "my first name is Amir" -> "first name")
and live in the global scope; un-routed type-word statements keep their
topic-bearing session concepts ("I prefer dark mode" -> "preference for
dark mode"), the way Mem0 keys facts by content and Letta keeps labelled
distinct slots. Explicit corrections still supersede the single tracked
concept of that type; ambiguous corrections are held pending; history
stays retrievable but is never presented as current.

Conventions follow the repo: stdlib only, temp SQLite files, injected
word-based token counter, no live model.
"""
import os
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import BudgetConfig, ContextManager
from PRIMORA.prime.context import state as state_mod

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]


def _db():
    _SEQ[0] += 1
    return os.path.join(_TEST_TMPDIR.name, "fc%d.db" % _SEQ[0])


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _budget(**kwargs):
    params = dict(input_target=200, input_ceiling=300,
                  output_reserve=20, total_budget=320)
    params.update(kwargs)
    return BudgetConfig(**params)


def _say(mgr, hist, sid, text, budget=None):
    hist.append({"role": "user", "content": text})
    return mgr.build(list(hist), sid, token_counter=_counter,
                     max_new_tokens=5,
                     budget=budget or _budget())


def _currents(mgr, scope=None, concept=None):
    return mgr._get_store().fetch_states(
        scope=scope, concept=concept, status="current")


def _facts(mgr, scope=None):
    return sorted((s.concept, s.value) for s in _currents(mgr, scope))


def _superseded(mgr, scope=None, concept=None):
    return mgr._get_store().fetch_states(
        scope=scope, concept=concept, status="superseded")


# 1. Unrelated preferences both survive --------------------------------------
def test_unrelated_preferences_both_survive():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I love pizza.")
    _say(mgr, hist, "s", "I prefer dark mode.")
    assert _facts(mgr) == [
        ("likes for pizza", "loves pizza"),
        ("preference for dark mode", "dark mode"),
    ]


# 2. Topic-established facts both survive ------------------------------------
def test_topic_facts_both_survive():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "My favorite car is a Tesla.")
    _say(mgr, hist, "s", "My favorite color is blue.")
    assert _facts(mgr) == [
        ("favorite car", "a Tesla"),
        ("favorite color", "blue"),
    ]


# 3. Multiple unrelated instructions both survive ---------------------------
def test_multiple_instructions_both_survive():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "Always confirm before deploying.")
    _say(mgr, hist, "s", "Never store API keys.")
    facts = _facts(mgr)
    assert ("instruction for confirm before deploying",
            "confirm before deploying") in facts
    # "API" keeps the user's own casing: the registry preserves it.
    assert ("instruction for never store api keys",
            "never store API keys".replace("api", "API")) in facts
    assert len(facts) == 2


# 4a. Explicit topic correction updates the right fact only ------------------
def test_topic_correction_updates_only_its_fact():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "My favorite car is a Tesla.")
    _say(mgr, hist, "s", "I prefer dark themes.")
    _say(mgr, hist, "s", "Actually, my favorite car is a Corvette C8.")
    assert _facts(mgr) == [
        ("favorite car", "a Corvette C8"),
        ("preference for dark themes", "dark themes"),
    ]
    old = _superseded(mgr, concept="favorite car")
    assert [(s.concept, s.value) for s in old] == [("favorite car", "a Tesla")]


# 4b. Type-word correction updates only the fact it shares a topic with ---
def test_type_word_correction_targets_shared_topic():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I love pizza.")
    _say(mgr, hist, "s", "I prefer dark mode.")
    _say(mgr, hist, "s", "Actually, I prefer light mode.")
    assert _facts(mgr) == [
        ("likes for pizza", "loves pizza"),
        ("preference for dark mode", "light mode"),
    ]
    old = _superseded(mgr, "s", "preference for dark mode")
    assert [(s.concept, s.value) for s in old] == [
        ("preference for dark mode", "dark mode")]
    current = _currents(mgr, "s", "preference for dark mode")[0]
    assert current.supersedes_id == old[0].state_id


# 5. Ambiguous corrections stay pending, overwrite nothing ------------------
def test_ambiguous_correction_held_pending():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I prefer blue car.")
    _say(mgr, hist, "s", "I prefer blue sky.")
    _say(mgr, hist, "s", "Actually, I prefer blue.")
    assert _facts(mgr, "s") == [
        ("preference for blue car", "blue car"),
        ("preference for blue sky", "blue sky"),
    ]
    pend = mgr._get_store().fetch_candidates(status="pending")
    assert any(p.proposed_value == "blue" for p in pend)
    assert _superseded(mgr, "s") == []


# 6. History retrievable, never presented as current -------------------------
def test_history_retrievable_but_not_current():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I prefer dark mode.")
    _say(mgr, hist, "s", "Actually, I prefer light mode.")
    current = _currents(mgr, "s", "preference for dark mode")
    assert [(s.concept, s.value) for s in current] == [
        ("preference for dark mode", "light mode")]
    old = _superseded(mgr, "s", "preference for dark mode")
    assert [(s.concept, s.value, s.status) for s in old] == [
        ("preference for dark mode", "dark mode", "superseded")]
    assert old[0].source_ids  # full history intact behind the supersession
    res = _say(mgr, hist, "s", "What is my preference?",
               budget=_budget(input_target=30, input_ceiling=60))
    joined = "\n".join(m["content"] for m in res.view.messages)
    assert "The user's preference for dark mode is light mode." in joined
    assert "The user's preference for dark mode is dark mode." not in joined
    assert "was: dark mode" not in joined


# Correction safety: type match alone is never a target ----------------------
def test_correction_without_shared_topic_is_additive():
    # Scenario a: a same-type correction with no shared topic adds a fact;
    # it never overwrites the unrelated one.
    for statements in (
        ["I love pizza.", "Actually, I prefer dark mode."],
        ["Actually, I prefer dark mode.", "I love pizza."],
    ):
        mgr = ContextManager(db_path=_db())
        hist = []
        for text in statements:
            _say(mgr, hist, "s", text)
        assert _facts(mgr) == [
            ("likes for pizza", "loves pizza"),
            ("preference for dark mode", "dark mode"),
        ]
        assert _superseded(mgr) == []


def test_correction_with_shared_topic_supersedes():
    # Scenario b: supersession happens because the topics share "mode".
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I prefer dark mode.")
    _say(mgr, hist, "s", "Actually, I prefer light mode.")
    assert _facts(mgr, "s") == [
        ("preference for dark mode", "light mode")]
    assert [(s.concept, s.value) for s in _superseded(mgr, "s")] == [
        ("preference for dark mode", "dark mode")]
    # reverse order: the correction lands first, so both become facts
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "Actually, I prefer light mode.")
    _say(mgr, hist, "s", "I prefer dark mode.")
    assert _facts(mgr, "s") == [
        ("preference for dark mode", "dark mode"),
        ("preference for light mode", "light mode"),
    ]


def test_no_cue_correction_keeps_both():
    # Scenario e: "No, I love tacos." shares no topic with pizza.
    for statements in (
        ["I love pizza.", "No, I love tacos."],
        ["No, I love tacos.", "I love pizza."],
    ):
        mgr = ContextManager(db_path=_db())
        hist = []
        for text in statements:
            _say(mgr, hist, "s", text)
        assert _facts(mgr) == [
            ("likes for pizza", "loves pizza"),
            ("likes for tacos", "loves tacos"),
        ]
        assert _superseded(mgr) == []


def test_bare_correction_held_when_no_referent():
    # Scenario g: a correction cue with no resolvable target changes
    # nothing existing and is held pending.
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I love pizza.")
    res = _say(mgr, hist, "s", "Actually, sage.")
    assert _facts(mgr) == [("likes for pizza", "loves pizza")]
    assert _superseded(mgr) == []
    pend = mgr._get_store().fetch_candidates(status="pending")
    assert any(p.proposed_value == "sage" for p in pend)
    assert any("candidate_pending_reason" in d for d in res.view.diagnostics)


def test_i_said_prefix_stores_own_fact():
    # Scenario g: "i said" is a correction cue; with no referent the
    # statement becomes its own fact.
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I love pizza.")
    _say(mgr, hist, "s", "I said I love durian.")
    assert _facts(mgr) == [
        ("likes for durian", "loves durian"),
        ("likes for pizza", "loves pizza"),
    ]
    assert _superseded(mgr) == []


def test_typed_bare_correction_needs_topic_share():
    # Scenario g: "Actually, prefer sage." names the type but shares no
    # topic with the stored fact; it is held, never overwriting.
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I love pizza.")
    res = _say(mgr, hist, "s", "Actually, prefer sage.")
    assert _facts(mgr) == [("likes for pizza", "loves pizza")]
    assert _superseded(mgr) == []
    pend = mgr._get_store().fetch_candidates(status="pending")
    assert any(p.proposed_value == "prefer sage" for p in pend)
    assert any("candidate_pending_reason" in d for d in res.view.diagnostics)


# Guard: prefix siblings stay distinct ---------------------------------------
def test_prefix_sibling_concepts_stay_distinct():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I prefer blue.")
    _say(mgr, hist, "s", "I prefer blue car.")
    assert _facts(mgr, "s") == [
        ("preference for blue", "blue"),
        ("preference for blue car", "blue car"),
    ]


# Guard: from-now-on directives are additive ---------------------------------
def test_from_now_on_directives_are_additive():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "From now on, keep the visual frozen.")
    _say(mgr, hist, "s", "From now on, always use UTC timestamps.")
    assert len(_facts(mgr)) == 2


# Guard: hedged type-word pendings never confirm onto an unrelated fact ------
def test_hedged_pending_confirms_as_own_fact():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I love pizza.")
    _say(mgr, hist, "s", "I think I prefer blue.")
    pend = mgr._get_store().fetch_candidates(status="pending")
    assert len(pend) == 1
    diags = state_mod.confirm_pending(
        mgr._get_store(), "s", pend[0].candidate_id)
    assert any("candidate_confirmed" in d for d in diags), diags
    assert _facts(mgr) == [
        ("likes for pizza", "loves pizza"),
        ("preference for blue", "blue"),
    ]


# Guard: hedged pendings die when a firm fact of their type arrives ----------
def test_hedged_pending_rejected_by_firm_fact():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I think I prefer blue.")
    _say(mgr, hist, "s", "I prefer green.")
    assert _facts(mgr, "s") == [("preference for green", "green")]
    rows = mgr._get_store().fetch_candidates()
    assert len(rows) == 1
    assert rows[0].status == "rejected"
    assert rows[0].proposed_value == "blue"


if __name__ == "__main__":
    test_unrelated_preferences_both_survive()
    test_topic_facts_both_survive()
    test_multiple_instructions_both_survive()
    test_topic_correction_updates_only_its_fact()
    test_type_word_correction_targets_shared_topic()
    test_ambiguous_correction_held_pending()
    test_history_retrievable_but_not_current()
    test_correction_without_shared_topic_is_additive()
    test_correction_with_shared_topic_supersedes()
    test_no_cue_correction_keeps_both()
    test_bare_correction_held_when_no_referent()
    test_i_said_prefix_stores_own_fact()
    test_typed_bare_correction_needs_topic_share()
    test_prefix_sibling_concepts_stay_distinct()
    test_from_now_on_directives_are_additive()
    test_hedged_pending_confirms_as_own_fact()
    test_hedged_pending_rejected_by_firm_fact()
    print("All memory fact-collision tests passed!")
