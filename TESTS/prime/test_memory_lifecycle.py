"""M2 Memory representation & lifecycle.

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
    return os.path.join(_TEST_TMPDIR.name, "l%d.db" % _SEQ[0])


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


# candidate lifecycle: pending -> confirmed -----------------------------------
def test_pending_confirms_to_structured_memory():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I think I prefer blue.")
    pend = mgr._get_store().fetch_candidates(status="pending")
    assert len(pend) == 1
    assert _currents(mgr, "s", "preference") == []
    diags = state_mod.confirm_pending(
        mgr._get_store(), "s", pend[0].candidate_id)
    assert any("candidate_confirmed" in d for d in diags), diags
    rows = _currents(mgr, "s", "preference for blue")
    assert len(rows) == 1
    assert rows[0].value == "blue"
    assert rows[0].authority == "user"
    assert rows[0].confidence == 1.0
    assert rows[0].source_ids == pend[0].source_ids
    assert rows[0].memory_type == "PREFERENCE"
    now = mgr._get_store().fetch_candidates(status="pending")
    assert now == []
    assert mgr._get_store().fetch_candidates(
        status="confirmed")[0].candidate_id == pend[0].candidate_id


def test_pending_rejected_on_supersession():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I think I prefer blue.")
    _say(mgr, hist, "s", "I prefer green.")
    assert _currents(mgr, "s", "preference for green")[0].value == "green"
    rows = mgr._get_store().fetch_candidates()
    assert len(rows) == 1
    assert rows[0].status == "rejected"
    assert rows[0].proposed_value == "blue"


def test_uncertain_candidate_cannot_confirm():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "Maybe we should keep this.")
    pend = mgr._get_store().fetch_candidates(status="pending")
    assert len(pend) == 1
    diags = state_mod.confirm_pending(
        mgr._get_store(), "s", pend[0].candidate_id)
    assert any("no confirmable concept" in d for d in diags), diags
    assert _currents(mgr, "s") == []
    assert mgr._get_store().fetch_candidates(
        status="pending")[0].candidate_id == pend[0].candidate_id


def test_confirm_unknown_candidate_fails_cleanly():
    mgr = ContextManager(db_path=_db())
    diags = state_mod.confirm_pending(mgr._get_store(), "s", 4242)
    assert any("not found" in d for d in diags)
    diags = state_mod.confirm_pending(mgr._get_store(), "s", 4242)
    assert any("not found" in d for d in diags)


# candidate safety --------------------------------------------------------------
def test_pending_rejected_never_injected():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I think I prefer blue.")
    res = _say(mgr, hist, "s", "What is my favorite color?")
    joined = "\n".join(m["content"] for m in res.view.messages)
    assert "[state " not in joined
    assert "[memory" not in joined
    # Raw history legitimately still shows the original turn verbatim.


# confidence ----------------------------------------------------------------------
def test_explicit_confidence_high_pending_low():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I prefer dark themes.")
    _say(mgr, hist, "s", "Maybe we should keep this.")
    rows = _currents(mgr, "s", "preference for dark themes")
    assert len(rows) == 1 and rows[0].confidence == 1.0
    pend = mgr._get_store().fetch_candidates(status="pending")
    assert len(pend) == 1 and pend[0].confidence == 0.3


def test_assistant_unresolved_confidence_low():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "The color is blue.")
    hist.append({"role": "assistant",
                 "content": "The color is green."})
    mgr.build(list(hist), "s", token_counter=_counter,
              max_new_tokens=5, budget=_budget())
    rows = mgr._get_store().fetch_states(
        scope="s", concept="color", status="unresolved")
    assert len(rows) == 1
    assert rows[0].authority == "assistant"
    assert rows[0].confidence == 0.3


def test_pending_rejected_on_supersession():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I think I prefer blue.")
    _say(mgr, hist, "s", "I prefer green.")
    assert _currents(mgr, "s", "preference for green")[0].value == "green"
    rows = mgr._get_store().fetch_candidates()
    assert len(rows) == 1
    assert rows[0].status == "rejected"
    assert rows[0].proposed_value == "blue"


def test_superseded_confidence_survives_but_excluded():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I prefer dark mode.")
    _say(mgr, hist, "s", "Actually, I prefer light mode.")
    old = mgr._get_store().fetch_states(
        scope="s", concept="preference for dark mode", status="superseded")
    assert len(old) == 1 and old[0].confidence == 1.0
    res = _say(mgr, hist, "s", "What is my preference?",
               budget=_budget(input_target=30, input_ceiling=60))
    joined = "\n".join(m["content"] for m in res.view.messages)
    assert "preference for dark mode = light mode" in joined
    assert "preference for dark mode = dark mode" not in joined


# current/superseded ----------------------------------------------------------------
def test_blue_green_single_current_with_confidence():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I prefer dark mode.")
    _say(mgr, hist, "s", "Actually, I prefer light mode.")
    rows = _currents(mgr, "s", "preference for dark mode")
    assert len(rows) == 1
    assert (rows[0].value, rows[0].confidence,
            rows[0].authority) == ("light mode", 1.0, "user")
    assert rows[0].supersedes_id is not None


def test_invalidate_retracts_without_replacement():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I prefer green.")
    _say(mgr, hist, "s", "My preference is no longer correct.")
    assert _currents(mgr, "s", "preference for green") == []
    old = mgr._get_store().fetch_states(
        scope="s", concept="preference for green", status="superseded")
    assert len(old) == 1 and old[0].value == "green"


# rationale linkage -------------------------------------------------------------------
def test_rationale_moves_with_decision():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s",
         "We kept the visual because it finally feels like Aurora.")
    dec = _currents(mgr, "s", "decision for visual")
    rat = _currents(mgr, "s", "rationale for decision for visual")
    assert len(dec) == 1 and len(rat) == 1
    assert dec[0].confidence == 1.0 and rat[0].confidence == 1.0
    assert rat[0].source_ids == dec[0].source_ids
    _say(mgr, hist, "s", "We changed the decision to the layout.")
    assert _currents(mgr, "s", "decision for visual")[0].value == "the layout"
    assert _currents(mgr, "s", "rationale for decision for visual") == []
    old_rat = mgr._get_store().fetch_states(
        scope="s", concept="rationale for decision for visual",
        status="superseded")
    assert len(old_rat) == 1
    assert old_rat[0].authority == "user"


# historical preservation -----------------------------------------------------------------
def test_superseded_rows_fully_intact():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I prefer dark mode.")
    before = mgr._get_store().fetch_states(
        scope="s", concept="preference for dark mode", status="current")[0]
    _say(mgr, hist, "s", "Actually, I prefer light mode.")
    after = mgr._get_store().fetch_states(
        scope="s", concept="preference for dark mode", status="superseded")
    assert len(after) == 1
    row = after[0]
    assert (row.value, row.authority, row.memory_type,
            row.confidence) == ("dark mode", "user", "PREFERENCE", 1.0)
    assert row.source_ids == before.source_ids


# assistant authority -----------------------------------------------------------------------
def test_assistant_cannot_become_current():
    mgr = ContextManager(db_path=_db())
    hist = []
    hist.append({"role": "assistant",
                 "content": "You probably prefer green."})
    mgr.build(list(hist), "s", token_counter=_counter,
              max_new_tokens=5, budget=_budget())
    rows = mgr._get_store().fetch_states()
    assert [r for r in rows if r.authority == "assistant"
            and r.status == "current"] == []
    assert [r for r in rows if r.authority == "user"
            and r.status == "current"] == []


def test_store_rejects_assistant_current():
    mgr = ContextManager(db_path=_db())
    store = mgr._get_store()
    try:
        store.create_state("s", "s", "color", "green", [1],
                           status="current", authority="assistant")
        assert False, "must raise"
    except ValueError as e:
        assert "only user-authored" in str(e)


# repetition ----------------------------------------------------------------------
def test_repetition_keeps_single_current():
    mgr = ContextManager(db_path=_db())
    hist = []
    for _ in range(3):
        _say(mgr, hist, "s", "I prefer dark themes.")
    rows = _currents(mgr, "s", "preference for dark themes")
    assert len(rows) == 1
    assert (rows[0].value, rows[0].confidence) == ("dark themes", 1.0)
    assert mgr._get_store().count("s") == 3


# provenance ------------------------------------------------------------------------
def test_source_ids_survive_all_transitions():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I prefer dark mode.")
    first = _currents(mgr, "s", "preference for dark mode")[0]
    _say(mgr, hist, "s", "Actually, I prefer light mode.")
    second = _currents(mgr, "s", "preference for dark mode")[0]
    assert first.source_ids != second.source_ids
    assert second.supersedes_id == first.state_id
    old = mgr._get_store().fetch_states(
        scope="s", concept="preference for dark mode",
        status="superseded")[0]
    assert old.source_ids == first.source_ids
    pend_rows = mgr._get_store().fetch_candidates()
    assert pend_rows == []


# idempotence -------------------------------------------------------------------------
def test_rebuild_no_duplicates():
    mgr = ContextManager(db_path=_db())
    snap = [{"role": "user", "content": "I prefer dark themes."}]
    mgr.reconcile(list(snap), "s")
    mgr.reconcile(list(snap), "s")
    assert len(_currents(mgr, "s", "preference for dark themes")) == 1
    assert mgr._get_store().fetch_candidates() == []
    assert mgr._get_store().count("s") == 1


# scope -----------------------------------------------------------------------------------
def test_scope_lifecycle_isolated():
    mgr = ContextManager(db_path=_db())
    ha, hb = [], []
    _say(mgr, ha, "sa", "I prefer dark mode.")
    _say(mgr, hb, "sb", "I prefer green tea.")
    ra = _currents(mgr, "sa", "preference for dark mode")
    rb = _currents(mgr, "sb", "preference for green tea")
    assert [(r.value, r.confidence) for r in ra] == [("dark mode", 1.0)]
    assert [(r.value, r.confidence) for r in rb] == [("green tea", 1.0)]
    _say(mgr, ha, "sa", "Actually, I prefer light mode.")
    assert _currents(mgr, "sa", "preference for dark mode")[0].value == \
        "light mode"
    assert _currents(mgr, "sb", "preference for green tea")[0].value == \
        "green tea"


if __name__ == "__main__":
    test_pending_confirms_to_structured_memory()
    test_pending_rejected_on_supersession()
    test_uncertain_candidate_cannot_confirm()
    test_confirm_unknown_candidate_fails_cleanly()
    test_pending_rejected_never_injected()
    test_explicit_confidence_high_pending_low()
    test_assistant_unresolved_confidence_low()
    test_superseded_confidence_survives_but_excluded()
    test_blue_green_single_current_with_confidence()
    test_invalidate_retracts_without_replacement()
    test_rationale_moves_with_decision()
    test_superseded_rows_fully_intact()
    test_assistant_cannot_become_current()
    test_store_rejects_assistant_current()
    test_repetition_keeps_single_current()
    test_source_ids_survive_all_transitions()
    test_rebuild_no_duplicates()
    test_scope_lifecycle_isolated()
    print("All memory lifecycle tests passed!")
