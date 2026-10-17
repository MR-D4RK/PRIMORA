"""Memory correctness: current/superseded/history invariant.

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
    return os.path.join(_TEST_TMPDIR.name, "m%d.db" % _SEQ[0])


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _budget(**kwargs):
    params = dict(input_target=25, input_ceiling=60,
                  output_reserve=5, total_budget=65)
    params.update(kwargs)
    return BudgetConfig(**params)


def _say(mgr, hist, sid, text, budget=None):
    hist.append({"role": "user", "content": text})
    return mgr.build(list(hist), sid, token_counter=_counter,
                     max_new_tokens=5,
                     budget=budget or _budget())


def _texts(view):
    return [(m["role"], m["content"]) for m in view.messages]


def _setup_colors(sid="mem-a"):
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, sid, "My favorite color is blue.")
    _say(mgr, hist, sid, "My favorite color is red.")
    _say(mgr, hist, sid, "My favorite color is green.")
    return mgr, hist


# A. Correction ---------------------------------------------------------------
def test_normal_retrieval_returns_only_current():
    mgr, hist = _setup_colors()
    res = _say(mgr, hist, "mem-a", "What is my favorite color?")
    texts = _texts(res.view)
    joined = "\n".join(c for _, c in texts)
    assert "favorite color is green" in joined
    assert "blue" not in joined.replace("green", "GREEN-OK"), joined
    assert any("retrieval_suppressed_superseded" in d
               for d in res.view.diagnostics)


# B. Historical query ----------------------------------------------------------
def test_history_query_does_not_resurrect_superseded():
    # A history question may not drag superseded values back into state
    # lines; the current fact is what the model sees, and the superseded
    # rows stay intact in the store for provenance.
    mgr, hist = _setup_colors()
    res = _say(mgr, hist, "mem-a", "What was my previous favorite color?")
    state_joined = "\n".join(c for r, c in _texts(res.view) if r == "system")
    assert "The user's favorite color is green." in state_joined
    assert "blue" not in state_joined, state_joined
    assert "red" not in state_joined, state_joined
    old = mgr._get_store().fetch_states(
        concept="favorite color", status="superseded")
    assert sorted(s.value for s in old) == ["blue", "red"]


# C. Multiple corrections ------------------------------------------------------
def test_multiple_corrections_single_current():
    mgr, hist = _setup_colors()
    store = mgr._get_store()
    currents = store.fetch_states(
        scope="global", concept="favorite color", status="current")
    assert len(currents) == 1
    assert currents[0].value == "green"
    res = _say(mgr, hist, "mem-a", "What is my favorite color?")
    joined = "\n".join(c for _, c in _texts(res.view))
    state_joined = "\n".join(c for r, c in _texts(res.view) if r == "system")
    assert "The user's favorite color is green." in state_joined
    # Blue fell out of the recent window AND is suppressed from retrieval.
    assert "blue" not in joined.replace("green", "GREEN-OK"), joined
    # Red remains visible only as a plain chronological turn, never as an
    # active state line; the mechanism demonstrably engaged.
    assert "The user's favorite color is red." not in state_joined, joined
    assert any("retrieval_suppressed_superseded" in d
               for d in res.view.diagnostics)
    res2 = _say(mgr, hist, "mem-a",
                "List my previous favorite colors, old ones included.")
    state_joined2 = "\n".join(c for r, c in _texts(res2.view) if r == "system")
    # Even on an explicit history question the superseded values never
    # return as state lines; they remain intact in the store.
    assert "The user's favorite color is green." in state_joined2
    assert "blue" not in state_joined2, state_joined2
    assert "red" not in state_joined2, state_joined2
    old = store.fetch_states(concept="favorite color", status="superseded")
    assert sorted(s.value for s in old) == ["blue", "red"]


# D. Cross-session scope --------------------------------------------------------
def test_global_scope_invariant():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "mem-d",
         "My favorite color is blue for all conversations.")
    _say(mgr, hist, "mem-d",
         "My favorite color is green for all conversations.")
    store = mgr._get_store()
    currents = store.fetch_states(
        scope="global", concept="favorite color", status="current")
    assert len(currents) == 1
    assert currents[0].value == "green"
    res = _say(mgr, hist, "mem-d", "What is my favorite color?")
    joined = "\n".join(c for _, c in _texts(res.view))
    assert "favorite color is green" in joined
    assert "blue" not in joined.replace("green", "GREEN-OK"), joined


# E. Single current per scope/concept -------------------------------------------
def test_only_one_current_per_concept():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "mem-e", "My favorite color is blue.")
    _say(mgr, hist, "mem-e", "My favorite color is green.")
    _say(mgr, hist, "mem-e", "My favorite color is green.")
    store = mgr._get_store()
    currents = store.fetch_states(
        scope="global", concept="favorite color", status="current")
    assert len(currents) == 1
    assert currents[0].value == "green"
    assert currents[0].supersedes_id is not None


# F. Irrelevant old memory -------------------------------------------------------
def test_irrelevant_old_memory_not_promoted():
    mgr = ContextManager(db_path=_db())
    hist = []
    _say(mgr, hist, "mem-f", "The blue warehouse stores timber.")
    _say(mgr, hist, "mem-f", "My favorite color is green.")
    _say(mgr, hist, "mem-f", "Tell me about oak trees.")
    _say(mgr, hist, "mem-f", "Weather looks cloudy today.")
    res = _say(mgr, hist, "mem-f", "What is my favorite color?",
               budget=_budget(input_target=30, input_ceiling=45))
    joined = "\n".join(c for _, c in _texts(res.view))
    assert "favorite color is green" in joined
    assert "warehouse" not in joined, joined


if __name__ == "__main__":
    test_normal_retrieval_returns_only_current()
    test_history_query_does_not_resurrect_superseded()
    test_multiple_corrections_single_current()
    test_global_scope_invariant()
    test_only_one_current_per_concept()
    test_irrelevant_old_memory_not_promoted()
    print("All memory correctness tests passed!")
