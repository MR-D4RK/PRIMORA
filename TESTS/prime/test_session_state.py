"""Focused tests: sessions, state lifecycle, cross-session retrieval.

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
    return os.path.join(_TEST_TMPDIR.name, f"s{_SEQ[0]}.db")


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _budget(**kwargs):
    params = dict(input_target=60, input_ceiling=80,
                  output_reserve=10, total_budget=90)
    params.update(kwargs)
    return BudgetConfig(**params)


# 1. session creation -------------------------------------------------------
def test_session_creation():
    mgr = ContextManager(db_path=_db())
    sid, diags = mgr.ensure_session("sess-1")
    assert diags == []
    assert sid == "sess-1"
    rec = mgr._get_store().get_session("sess-1")
    assert rec.status == "active"
    assert rec.created_at > 0 and rec.last_activity_at >= rec.created_at
    sid2, diags2 = mgr.ensure_session("sess-1")
    assert sid2 == "sess-1" and diags2 == []
    again = mgr._get_store().get_session("sess-1")
    assert again.created_at == rec.created_at


# 2. session isolation -------------------------------------------------------
def test_session_isolation():
    mgr = ContextManager(db_path=_db())
    mgr.reconcile([{"role": "user", "content": "Blue Lantern vault fact"}], "sessA")
    view = mgr.build([{"role": "user", "content": "how is the weather today"}],
                     "sessB", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    texts = " ".join(m["content"] for m in view.messages)
    assert "Blue Lantern" not in texts


# 3. cross-session retrieval ---------------------------------------------------
def test_cross_session_retrieval():
    mgr = ContextManager(db_path=_db())
    mgr.reconcile([{"role": "user", "content": "the vault code is BLUE-FALCON-7"}], "sessA")
    view = mgr.build([{"role": "user", "content": "what is the vault code?"}],
                     "sessB", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    texts = " ".join(m["content"] for m in view.messages)
    assert "BLUE-FALCON-7" in texts
    assert "[from another conversation]" in texts


# 4. legacy default behavior -----------------------------------------------------
def test_legacy_default_behavior():
    mgr = ContextManager(db_path=_db())
    view = mgr.build([{"role": "user", "content": "hello there"}],
                     token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    assert view.session_id == "default"
    assert view.messages[-1] == {"role": "user", "content": "hello there"}
    assert mgr._get_store().get_session("default") is not None


# 5. explicit state creation --------------------------------------------------------
def test_explicit_state_creation():
    mgr = ContextManager(db_path=_db())
    mgr.reconcile([{"role": "user", "content": "The phrase is Blue Lantern."}], "s")
    rows = mgr._get_store().fetch_states(scope="s", status="current")
    assert len(rows) == 1
    assert rows[0].concept == "phrase" and rows[0].value == "Blue Lantern"
    assert rows[0].authority == "user" and rows[0].source_ids == [1]


# 6. user-authored state authority ------------------------------------------------------
def test_user_authority_over_assistant():
    mgr = ContextManager(db_path=_db())
    mgr.reconcile([{"role": "user", "content": "The phrase is Blue Lantern."}], "s")
    diags = mgr.reconcile([{"role": "assistant", "content": "The phrase is Red Herring."}], "s")
    currents = mgr._get_store().fetch_states(scope="s", status="current")
    assert len(currents) == 1 and currents[0].value == "Blue Lantern"
    assert any("state_conflict" in d for d in diags)


# 7. explicit correction -------------------------------------------------------------------
def test_explicit_correction():
    mgr = ContextManager(db_path=_db())
    mgr.reconcile([{"role": "user", "content": "The phrase is Blue Lantern."}], "s")
    mgr.reconcile([{"role": "user", "content": "The new and correct phrase is Violet Aurora."}], "s")
    currents = mgr._get_store().fetch_states(scope="s", status="current")
    assert len(currents) == 1 and currents[0].value == "Violet Aurora"


# 8. supersession --------------------------------------------------------------------------------
def test_supersession_linking():
    mgr = ContextManager(db_path=_db())
    mgr.reconcile([{"role": "user", "content": "The phrase is Blue Lantern."}], "s")
    mgr.reconcile([{"role": "user", "content": "Change the phrase to Violet Aurora."}], "s")
    old = mgr._get_store().fetch_states(scope="s", concept="phrase", status="superseded")
    new = mgr._get_store().fetch_states(scope="s", concept="phrase", status="current")
    assert len(old) == 1 and len(new) == 1
    assert new[0].supersedes_id == old[0].state_id
    assert new[0].value == "Violet Aurora"


# 9. historical value preservation ---------------------------------------------------------------------
def test_historical_value_preserved():
    mgr = ContextManager(db_path=_db())
    mgr.reconcile([{"role": "user", "content": "The phrase is Blue Lantern."}], "s")
    mgr.reconcile([{"role": "user", "content": "The new phrase is Violet Aurora."}], "s")
    rows = mgr._get_store().fetch_range("s")
    assert len(rows) == 2
    assert {r.content for r in rows} == {
        "The phrase is Blue Lantern.", "The new phrase is Violet Aurora."}
    old_states = mgr._get_store().fetch_states(scope="s", status="superseded")
    assert any(s.value == "Blue Lantern" for s in old_states)


# 10. unresolved conflicting states ----------------------------------------------------------------------------
def test_unresolved_conflict():
    mgr = ContextManager(db_path=_db())
    mgr.reconcile([{"role": "user", "content": "The color is Blue Lantern."}], "s")
    diags = mgr.reconcile([{"role": "assistant", "content": "The color is Red Herring."}], "s")
    currents = mgr._get_store().fetch_states(scope="s", status="current")
    stuck = mgr._get_store().fetch_states(scope="s", status="unresolved")
    assert len(currents) == 1 and currents[0].value == "Blue Lantern"
    assert len(stuck) == 1 and stuck[0].value == "Red Herring"
    assert any("state_conflict" in d for d in diags)


# 11. current-state preference ---------------------------------------------------------------------------------
def test_current_state_preferred():
    mgr = ContextManager(db_path=_db())
    mgr.reconcile([{"role": "user", "content": "The test phrase is Blue Lantern."}], "s")
    mgr.reconcile([{"role": "user", "content": "The new test phrase is Violet Aurora."}], "s")
    view = mgr.build([{"role": "user", "content": "What is the current test phrase?"}],
                     "s", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    texts = " ".join(m["content"] for m in view.messages)
    assert "Violet Aurora" in texts
    assert "was: Blue Lantern" not in texts


# 12. historical query never resurrects the superseded value -------------------
def test_historical_query_returns_superseded():
    # The superseded row stays intact in the store, but the rendered
    # state lines never carry it back; only the current value is shown.
    mgr = ContextManager(db_path=_db())
    mgr.reconcile([{"role": "user", "content": "The test phrase is Blue Lantern."}], "s")
    mgr.reconcile([{"role": "user", "content": "The new test phrase is Violet Aurora."}], "s")
    view = mgr.build([{"role": "user", "content": "What was the previous test phrase?"}],
                     "s", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    state_lines = [m["content"] for m in view.messages
                   if m.get("role") == "system"
                   and not m["content"].startswith("[session ")]
    assert any("The user's test phrase is Violet Aurora." in line
               for line in state_lines)
    assert not any("Blue Lantern" in line for line in state_lines)
    old = mgr._get_store().fetch_states(status="superseded")
    assert [s.value for s in old] == ["Blue Lantern"]


# 13. state provenance ---------------------------------------------------------------------
def test_state_provenance():
    # State lines are plain sentences now, but the stored row still
    # points at the exact message that produced it.
    mgr = ContextManager(db_path=_db())
    mgr.reconcile([{"role": "user", "content": "The test phrase is Blue Lantern."}], "s")
    view = mgr.build([{"role": "user", "content": "What is the current test phrase?"}],
                     "s", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    state_lines = [m["content"] for m in view.messages
                   if m.get("role") == "system"
                   and not m["content"].startswith("[session ")]
    assert any("The user's test phrase is Blue Lantern." in line
               for line in state_lines), state_lines
    row = mgr._get_store().fetch_states(
        scope="s", concept="test phrase", status="current")[0]
    assert row.source_ids
    (src,) = row.source_ids
    stored = [m for m in mgr._get_store().fetch_range("s", 0)
              if m.message_id == src]
    assert len(stored) == 1
    assert stored[0].content == "The test phrase is Blue Lantern."


# 14. state failure fallback -------------------------------------------------------------------
def test_state_failure_fallback():
    mgr = ContextManager(db_path=_db())
    diags = mgr.reconcile([{"role": "user", "content": "The phrase is Blue Lantern."}], "s")
    assert not any(d.startswith("state_failed") for d in diags)
    view = mgr.build([{"role": "user", "content": "hello again friend"}],
                     "s", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    assert view.messages[-1] == {"role": "user", "content": "hello again friend"}


# 15. session failure fallback -----------------------------------------------------------------------
def test_session_failure_fallback():
    import tempfile
    blocker = tempfile.NamedTemporaryFile(delete=False)
    blocker.close()
    mgr = ContextManager(db_path=os.path.join(blocker.name, "db.sqlite"))
    sid, diags = mgr.ensure_session("sessX")
    assert sid == "sessX"
    assert any(d.startswith("store_unavailable") for d in diags)
    view = mgr.build([{"role": "user", "content": "hello world"}],
                     "sessX", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    assert view.messages[-1] == {"role": "user", "content": "hello world"}


# 16. state + digest interaction ---------------------------------------------------------------------------
def test_state_digest_interaction():
    mgr = ContextManager(db_path=_db())
    for i in range(55):
        mgr.reconcile([{"role": "user", "content": f"filler discussion point {i} here"}], "s")
    mgr.reconcile([{"role": "user", "content": "The test phrase is Violet Aurora."}], "s")
    view = mgr.build([{"role": "user", "content": "What is the current test phrase?"}],
                     "s", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    texts = " ".join(m["content"] for m in view.messages)
    assert "Violet Aurora" in texts
    digests = mgr._get_store().fetch_digests("s")
    assert len(digests) >= 1
    # 55 fillers + 1 state message + the build's own query message.
    assert mgr._get_store().count("s") == 57


# 17. long conversation with multiple transitions ---------------------------------------------------------------
def test_long_conversation_multiple_transitions():
    mgr = ContextManager(db_path=_db())
    mgr.reconcile([{"role": "user", "content": "The color is red."}], "s")
    mgr.reconcile([{"role": "user", "content": "Change the color to green."}], "s")
    mgr.reconcile([{"role": "user", "content": "The new color is blue."}], "s")
    currents = mgr._get_store().fetch_states(scope="s", concept="color", status="current")
    assert len(currents) == 1 and currents[0].value == "blue"
    old = mgr._get_store().fetch_states(scope="s", concept="color", status="superseded")
    assert {s.value for s in old} == {"red", "green"}
    assert mgr._get_store().count("s") == 3


# Integration: Blue Lantern -> Violet Aurora across browser sessions ---------------------
def test_blue_lantern_to_violet_aurora_across_sessions():
    mgr = ContextManager(db_path=_db())
    mgr.reconcile([{"role": "user", "content": "The test phrase is Blue Lantern."}], "browser-1")
    mgr.reconcile([{"role": "user", "content": "The new and correct test phrase is Violet Aurora."}],
                  "browser-1")
    # browser-2 holds no states: nothing may be presented as ITS current
    # value, even though browser-1's rows remain retrievable history.
    current_view = mgr.build(
        [{"role": "user", "content": "What is the current test phrase?"}],
        "browser-2", token_counter=_counter,
        max_new_tokens=5, budget=_budget()).view
    current_texts = " ".join(m["content"] for m in current_view.messages)
    assert "[state current" not in current_texts

    # Explicit cross-session naming still resolves per session.
    named_view = mgr.build(
        [{"role": "user", "content": "What is the test phrase for browser-1?"}],
        "browser-2", token_counter=_counter,
        max_new_tokens=5, budget=_budget()).view
    named_texts = " ".join(m["content"] for m in named_view.messages)
    assert "Violet Aurora" in named_texts

    history_view = mgr.build(
        [{"role": "user", "content": "What was the previous test phrase?"}],
        "browser-2", token_counter=_counter,
        max_new_tokens=5, budget=_budget()).view
    history_texts = " ".join(m["content"] for m in history_view.messages)
    assert "Blue Lantern" in history_texts

    # Nothing deleted or rewritten anywhere (2 S1 rows + 3 S2 query rows).
    assert mgr._get_store().count("browser-1") == 2
    assert mgr._get_store().count("browser-2") == 3


if __name__ == "__main__":
    test_session_creation()
    test_session_isolation()
    test_cross_session_retrieval()
    test_legacy_default_behavior()
    test_explicit_state_creation()
    test_user_authority_over_assistant()
    test_explicit_correction()
    test_supersession_linking()
    test_historical_value_preserved()
    test_unresolved_conflict()
    test_current_state_preferred()
    test_historical_query_returns_superseded()
    test_state_provenance()
    test_state_failure_fallback()
    test_session_failure_fallback()
    test_state_digest_interaction()
    test_long_conversation_multiple_transitions()
    test_blue_lantern_to_violet_aurora_across_sessions()
    print("All session/state tests passed!")
