"""M3 Active Memory + Working Memory.

Conventions follow the repo: stdlib only, temp SQLite files, injected
word-based token counter, no live model.
"""
import os
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import (
    ActiveMemory,
    ActiveMemoryItem,
    BudgetConfig,
    ContextManager,
    WorkingMemory,
)

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]


def _db():
    _SEQ[0] += 1
    return os.path.join(_TEST_TMPDIR.name, "w%d.db" % _SEQ[0])


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


def _seed(mgr, sid="wm-a"):
    hist = []
    _say(mgr, hist, sid, "My favorite color is blue.")
    _say(mgr, hist, sid, "My favorite color is green.")
    _say(mgr, hist, sid,
         "We kept the visual because it finally feels like Aurora.")
    _say(mgr, hist, sid, "I think I prefer blue.")
    return hist


# Active Memory ---------------------------------------------------------------
def test_active_shows_current_only():
    mgr = ContextManager(db_path=_db())
    _seed(mgr)
    view = mgr.get_active_memory("wm-a")
    assert isinstance(view, ActiveMemory)
    by_concept = {(i.scope, i.concept): i for i in view.items
                  if i.kind == "state"}
    assert by_concept[("global", "favorite color")].value == "green"
    assert by_concept[("global", "decision for visual")].value == "visual"
    assert all(i.status == "current" for i in view.items
               if i.kind == "state")
    assert all(i.authority == "user" for i in view.items
               if i.kind == "state")


def test_active_excludes_stale_and_pending():
    mgr = ContextManager(db_path=_db())
    _seed(mgr)
    view = mgr.get_active_memory("wm-a")
    texts = " ".join(i.content for i in view.items)
    assert "favorite color = blue" not in texts
    assert "preference = blue" not in texts
    pend = mgr._get_store().fetch_candidates(status="pending")
    assert len(pend) == 1  # the hedged blue statement
    assert all("blue" not in i.content or "green" in i.content
               for i in view.items if i.kind == "state")


def test_active_excludes_unresolved():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "The color is blue.")
    hist.append({"role": "assistant", "content": "The color is green."})
    mgr.build(list(hist), "s", token_counter=_counter,
              max_new_tokens=5, budget=_budget())
    view = mgr.get_active_memory("s")
    assert all(i.kind != "state" or "unresolved" not in i.content.lower()
               for i in view.items)
    assert not [i for i in view.items
                if i.kind == "state" and i.authority != "user"]


def test_active_rationale_linked():
    mgr = ContextManager(db_path=_db())
    _seed(mgr)
    view = mgr.get_active_memory("wm-a")
    rat = [i for i in view.items
           if i.kind == "state" and i.memory_type == "RATIONALE"]
    dec = [i for i in view.items
           if i.kind == "state" and i.concept == "decision for visual"]
    assert len(rat) == 1 and len(dec) == 1
    assert rat[0].source_ids == dec[0].source_ids
    assert "finally feels like Aurora" in rat[0].value


def test_active_updates_after_correction():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I prefer dark mode.")
    before = {i.concept: i.value for i in
              mgr.get_active_memory("s").items if i.kind == "state"}
    assert before.get("preference for dark mode") == "dark mode"
    _say(mgr, hist, "s", "Actually, I prefer light mode.")
    after = {i.concept: i.value for i in
             mgr.get_active_memory("s").items if i.kind == "state"}
    assert after.get("preference for dark mode") == "light mode"
    assert "dark mode" not in [i.value for i in
                               mgr.get_active_memory("s").items
                               if i.kind == "state"]


def test_active_empty_session():
    mgr = ContextManager(db_path=_db())
    view = mgr.get_active_memory("nobody")
    assert view.items == []
    assert view.session_id == "nobody"


def test_active_empty_session():
    mgr = ContextManager(db_path=_db())
    view = mgr.get_active_memory("nobody")
    assert view.items == []
    assert view.session_id == "nobody"


# Working Memory ----------------------------------------------------------------
def test_working_create_populate_update_clear():
    wm = WorkingMemory(session_id="t1", task="improve memory architecture")
    assert wm.task == "improve memory architecture"
    assert wm.entry_count() == 1
    wm.set_objective("measure first")
    wm.note_decision("use explicit maps")
    wm.add_evidence("user", "raw quote here", source="msg:3")
    wm.add_file("PRIMORA/prime/context/manager.py", note="assembly")
    assert wm.entry_count() == 5
    wm.set_objective("measure second")
    assert wm.entry_count() == 5
    texts = " ".join(m["content"] for m in wm.to_messages())
    assert "improve memory architecture" in texts
    assert "measure second" in texts
    assert "measure first" not in texts
    assert "use explicit maps" in texts
    assert "raw quote here" in texts
    assert "manager.py" in texts
    wm.clear()
    assert wm.entry_count() == 0
    assert wm.to_messages() == []
    assert wm.session_id == "t1"


def test_working_bounded():
    wm = WorkingMemory(session_id="t1", task="t", max_items=4,
                       max_tokens=40)
    for i in range(10):
        wm.add_evidence("user", "evidence statement number %d here" % i)
    assert wm.entry_count() <= 4
    assert wm.size_tokens() <= 40
    assert wm.task == "t"


def test_working_task_isolation():
    a = WorkingMemory(session_id="s", task="task A")
    b = WorkingMemory(session_id="s", task="task B")
    a.note_decision("decision A")
    assert b.entry_count() == 1
    assert "decision A" not in " ".join(
        m["content"] for m in b.to_messages())


def test_working_session_isolation():
    a = WorkingMemory(session_id="s-a", task="same task")
    b = WorkingMemory(session_id="s-b", task="same task")
    a.note_decision("only in A")
    assert "only in A" not in " ".join(
        m["content"] for m in b.to_messages())


def test_working_no_persistence():
    before = set(os.listdir(_TEST_TMPDIR.name))
    wm = WorkingMemory(session_id="s", task="t")
    wm.note_decision("d")
    wm.add_evidence("user", "e")
    wm.clear()
    assert set(os.listdir(_TEST_TMPDIR.name)) == before
    assert not hasattr(wm, "_store")


# Active -> Working -----------------------------------------------------------------
def test_select_active_into_working():
    mgr = ContextManager(db_path=_db())
    _seed(mgr)
    picked = mgr.select_active_memory("wm-a", "what color do I prefer?")
    assert picked
    assert picked[0].concept == "favorite color"
    assert picked[0].value == "green"
    wm = WorkingMemory(session_id="wm-a", task="answer color question")
    for item in picked[:2]:
        wm.adopt_active(item)
    texts = " ".join(m["content"] for m in wm.to_messages())
    assert "The user's favorite color is green." in texts


def test_select_active_empty_query():
    mgr = ContextManager(db_path=_db())
    _seed(mgr)
    assert mgr.select_active_memory("wm-a", "") == []
    # A session with no facts of its own and no stored personal facts
    # anywhere selects nothing.
    fresh = ContextManager(db_path=_db())
    assert fresh.select_active_memory("nobody", "color") == []


# Raw -> Working ------------------------------------------------------------------------
def test_raw_evidence_without_source_change():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "The vault code is BLUE-FALCON-7.")
    before = mgr._get_store().count("s")
    wm = WorkingMemory(session_id="s", task="recall code")
    wm.add_evidence("user", "The vault code is BLUE-FALCON-7.",
                    source="msg:1")
    assert mgr._get_store().count("s") == before
    assert "BLUE-FALCON-7" in " ".join(
        m["content"] for m in wm.to_messages())


# Promotion ---------------------------------------------------------------------------------
def test_promote_user_item_creates_pending():
    mgr = ContextManager(db_path=_db())
    wm = WorkingMemory(session_id="s", task="t")
    wm.note_decision("the vault code is BLUE-FALCON-7")
    ids, diags = mgr.promote_to_candidate(
        "s", "the vault code is BLUE-FALCON-7", source_ids=[99])
    assert len(ids) == 1
    pend = mgr._get_store().fetch_candidates(status="pending")
    assert len(pend) == 1
    assert pend[0].source_ids == [99]
    assert pend[0].authority == "user"
    assert mgr._get_store().fetch_states() == []
    # ... which can later confirm through the normal path
    from PRIMORA.prime.context import state as state_mod
    diags2 = state_mod.confirm_pending(
        mgr._get_store(), "s", pend[0].candidate_id)
    assert any("candidate_confirmed" in d for d in diags2)
    rows = mgr._get_store().fetch_states(status="current")
    assert len(rows) == 1


def test_promote_assistant_item_refused():
    mgr = ContextManager(db_path=_db())
    ids, diags = mgr.promote_to_candidate(
        "s", "the vault code is BLUE-FALCON-7", authority="assistant")
    assert ids == []
    assert any("promotion_refused" in d for d in diags)
    assert mgr._get_store().fetch_candidates() == []
    assert mgr._get_store().fetch_states() == []


def test_promote_empty_refused():
    mgr = ContextManager(db_path=_db())
    ids, diags = mgr.promote_to_candidate("s", "   ")
    assert ids == [] and any("promotion_refused" in d for d in diags)


# One-brain guarantee --------------------------------------------------------------------------
def test_build_with_working_single_view():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I prefer blue.")
    wm = WorkingMemory(session_id="s", task="answer color question")
    for item in mgr.select_active_memory("s", "tell me my stated preference"):
        wm.adopt_active(item)
    res = mgr.build([{"role": "user", "content": "What is my color?"}],
                    "s", token_counter=_counter, max_new_tokens=5,
                    budget=_budget(), working=wm)
    texts = [m["content"] for m in res.view.messages]
    assert any("working task" in c for c in texts)
    assert any("The user's preference for blue is blue." in c
               for c in texts)
    assert texts[-1] == "What is my color?"
    # working content frames the request exactly once
    assert sum("working task" in c for c in texts) == 1


def test_build_without_working_unchanged():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "s", "I prefer blue.")
    res = mgr.build([{"role": "user", "content": "What is my color?"}],
                    "s", token_counter=_counter, max_new_tokens=5,
                    budget=_budget())
    assert not any("working task" in m["content"]
                   for m in res.view.messages)


if __name__ == "__main__":
    test_active_shows_current_only()
    test_active_excludes_stale_and_pending()
    test_active_excludes_unresolved()
    test_active_rationale_linked()
    test_active_updates_after_correction()
    test_active_empty_session()
    test_working_create_populate_update_clear()
    test_working_bounded()
    test_working_task_isolation()
    test_working_session_isolation()
    test_working_no_persistence()
    test_select_active_into_working()
    test_select_active_empty_query()
    test_raw_evidence_without_source_change()
    test_promote_user_item_creates_pending()
    test_promote_assistant_item_refused()
    test_promote_empty_refused()
    test_build_with_working_single_view()
    test_build_without_working_unchanged()
    print("All memory working tests passed!")
