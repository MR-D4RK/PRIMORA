"""M1 Memory Formation: natural explicit statements become typed memory.

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
    return os.path.join(_TEST_TMPDIR.name, "f%d.db" % _SEQ[0])


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


def _currents(mgr, scope, concept=None):
    return mgr._get_store().fetch_states(
        scope=scope, concept=concept, status="current")


# 1. preference ---------------------------------------------------------------
def test_preference_confirmed():
    mgr = ContextManager(db_path=_db())
    hist = []
    res = _say(mgr, hist, "s", "I prefer dark themes.")
    rows = _currents(mgr, "s", "preference for dark themes")
    assert len(rows) == 1
    assert rows[0].value == "dark themes"
    assert rows[0].memory_type == "PREFERENCE"
    assert rows[0].authority == "user"
    assert rows[0].source_ids
    assert any("state_updated" in d for d in res.view.diagnostics)


# 2. decision -----------------------------------------------------------------
def test_decision_confirmed():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "We decided to keep the current visual.")
    rows = _currents(mgr, "s", "decision for keep the current visual")
    assert len(rows) == 1
    assert rows[0].value == "keep the current visual"
    assert rows[0].memory_type == "DECISION"


# 3. instruction --------------------------------------------------------------
def test_instruction_confirmed():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "From now on, keep the visual frozen.")
    rows = _currents(mgr, "s", "instruction for keep the visual frozen")
    assert len(rows) == 1
    assert rows[0].value == "keep the visual frozen"
    assert rows[0].memory_type == "INSTRUCTION"


def test_ordinary_requests_not_instructions():
    mgr = ContextManager(db_path=_db())
    hist = []
    for text in ("Show me this.", "Try this.", "Can we change this?"):
        _say(mgr, hist, "s", text)
    assert mgr._get_store().fetch_states(scope="s") == []


# 4. rationale ----------------------------------------------------------------
def test_rationale_linked():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s",
         "We kept the visual because it finally feels like Aurora.")
    dec = _currents(mgr, "s", "decision for visual")
    assert len(dec) == 1 and dec[0].value == "visual"
    rat = _currents(mgr, "s", "rationale for decision for visual")
    assert len(rat) == 1
    assert rat[0].value == "it finally feels like Aurora"
    assert rat[0].memory_type == "RATIONALE"
    assert rat[0].source_ids == dec[0].source_ids
    # an explicit correction supersedes the decision with its rationale
    hist2 = list(hist)
    hist2.append({"role": "user",
                  "content": "We changed the decision to the layout."})
    mgr.build(hist2, "s", token_counter=_counter,
              max_new_tokens=5, budget=_budget())
    old_rat = mgr._get_store().fetch_states(
        scope="s", concept="rationale for decision for visual",
        status="superseded")
    assert len(old_rat) == 1


# 5. correction -----------------------------------------------------------------
def test_correction_supersedes():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I prefer dark themes.")
    _say(mgr, hist, "s", "Actually, I prefer light themes.")
    rows = _currents(mgr, "s", "preference for dark themes")
    assert len(rows) == 1
    assert rows[0].value == "light themes"
    old = mgr._get_store().fetch_states(
        scope="s", concept="preference for dark themes", status="superseded")
    assert [s.value for s in old] == ["dark themes"]
    assert rows[0].supersedes_id == old[0].state_id


def test_bare_correction_unique_concept():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "My favorite color is blue.")
    _say(mgr, hist, "s", "Actually, green.")
    rows = _currents(mgr, "s", "favorite color")
    assert len(rows) == 1
    assert rows[0].value == "green"


def test_bare_correction_ambiguous_stays_pending():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "My favorite color is blue.")
    _say(mgr, hist, "s", "The theme is dark.")
    _say(mgr, hist, "s", "Actually, green.")
    rows = _currents(mgr, "s", "favorite color")
    assert len(rows) == 1 and rows[0].value == "blue"
    pend = mgr._get_store().fetch_candidates(status="pending")
    assert any(p.proposed_value == "green" for p in pend)


def test_changed_decision_inherits_concept():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "We decided to keep the current visual.")
    _say(mgr, hist, "s", "We changed the decision to the new layout.")
    rows = _currents(mgr, "s", "decision for keep the current visual")
    assert len(rows) == 1
    assert rows[0].value == "the new layout"
    old = mgr._get_store().fetch_states(
        scope="s", concept="decision for keep the current visual",
        status="superseded")
    assert [s.value for s in old] == ["keep the current visual"]


# 6. ambiguous ------------------------------------------------------------------
def test_ambiguous_not_confirmed():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "Maybe we should keep this.")
    assert mgr._get_store().fetch_states(scope="s") == []
    pend = mgr._get_store().fetch_candidates(status="pending")
    assert len(pend) == 1
    assert pend[0].authority == "user"


def test_hedged_explicit_stays_pending():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I think I prefer blue.")
    assert mgr._get_store().fetch_states(scope="s") == []
    pend = mgr._get_store().fetch_candidates(status="pending")
    assert len(pend) == 1
    assert pend[0].proposed_concept == "preference"


# 7. assistant barrier ------------------------------------------------------------
def test_assistant_never_current():
    mgr = ContextManager(db_path=_db())
    hist = []
    hist.append({"role": "assistant",
                 "content": "You probably prefer green."})
    mgr.build(list(hist), "s", token_counter=_counter,
              max_new_tokens=5, budget=_budget())
    hist.append({"role": "assistant",
                 "content": "I think you want this frozen."})
    mgr.build(list(hist), "s", token_counter=_counter,
              max_new_tokens=5, budget=_budget())
    rows = mgr._get_store().fetch_states()
    assert [s for s in rows if s.authority == "assistant"
            and s.status == "current"] == []
    assert [s for s in rows if s.authority == "user"
            and s.status == "current"] == []


# 8. repetition -------------------------------------------------------------------
def test_repetition_single_current():
    mgr = ContextManager(db_path=_db())
    hist = []
    for _ in range(3):
        _say(mgr, hist, "s", "I prefer dark themes.")
    rows = _currents(mgr, "s", "preference for dark themes")
    assert len(rows) == 1
    assert rows[0].value == "dark themes"
    assert mgr._get_store().count("s") == 3


# 9. provenance ---------------------------------------------------------------------
def test_provenance_points_to_source():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "First filler line here.")
    res = _say(mgr, hist, "s", "I prefer dark themes.")
    rows = _currents(mgr, "s", "preference for dark themes")
    assert len(rows) == 1
    (src,) = rows[0].source_ids
    stored = [m for m in mgr._get_store().fetch_range("s", 0)
              if m.message_id == src]
    assert len(stored) == 1
    assert stored[0].content == "I prefer dark themes."
    assert any("state_updated" in d for d in res.view.diagnostics)


# 10. scope --------------------------------------------------------------------------
def test_global_scope_formation():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "My favorite color is blue for all conversations.")
    rows = mgr._get_store().fetch_states(
        scope="global", concept="favorite color", status="current")
    assert len(rows) == 1
    assert rows[0].value == "blue"


# 11. idempotence -----------------------------------------------------------------------
def test_reconciliation_no_duplicates():
    mgr = ContextManager(db_path=_db())
    snap = [{"role": "user", "content": "I prefer dark themes."}]
    mgr.reconcile(list(snap), "s")
    mgr.reconcile(list(snap), "s")
    rows = _currents(mgr, "s", "preference for dark themes")
    assert len(rows) == 1
    assert mgr._get_store().fetch_candidates() == []
    assert mgr._get_store().count("s") == 1


# 12. candidate safety --------------------------------------------------------------------
def test_pending_never_injected():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "Maybe we should keep this.")
    res = _say(mgr, hist, "s", "What should we do?")
    joined = "\n".join(m["content"] for m in res.view.messages)
    assert "Maybe we should keep this." in joined  # raw history only
    assert "[state " not in joined
    assert "[memory" not in joined


if __name__ == "__main__":
    test_preference_confirmed()
    test_decision_confirmed()
    test_instruction_confirmed()
    test_ordinary_requests_not_instructions()
    test_rationale_linked()
    test_correction_supersedes()
    test_bare_correction_unique_concept()
    test_bare_correction_ambiguous_stays_pending()
    test_changed_decision_inherits_concept()
    test_ambiguous_not_confirmed()
    test_hedged_explicit_stays_pending()
    test_assistant_never_current()
    test_repetition_single_current()
    test_provenance_points_to_source()
    test_global_scope_formation()
    test_reconciliation_no_duplicates()
    test_pending_never_injected()
    print("All memory formation tests passed!")
