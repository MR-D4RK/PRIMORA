"""Focused tests: session-scoped state, resolution rules, metadata authority.

Conventions follow the repo: stdlib only, temp SQLite files, injected
word-based token counter, no live model.
"""
import os
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import BudgetConfig, ContextManager

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [100]


def _db():
    _SEQ[0] += 1
    return os.path.join(_TEST_TMPDIR.name, f"scope{_SEQ[0]}.db")


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _budget(**kwargs):
    params = dict(input_target=80, input_ceiling=110,
                  output_reserve=10, total_budget=120)
    params.update(kwargs)
    return BudgetConfig(**params)


def _say(mgr, session, text):
    mgr.reconcile([{"role": "user", "content": text}], session)


def _ask_texts(mgr, session, text):
    view = mgr.build([{"role": "user", "content": text}],
                     session, token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    return " ".join(m["content"] for m in view.messages)


# 1. same concept can have different session-scoped values -------------------
def test_same_concept_different_session_values():
    mgr = ContextManager(db_path=_db())
    _say(mgr, "sess-a", "The test phrase is OLDSKOOL.")
    _say(mgr, "sess-b", "The test phrase is BYLAW.")
    a = mgr._get_store().fetch_states(scope="sess-a", status="current")
    b = mgr._get_store().fetch_states(scope="sess-b", status="current")
    assert len(a) == 1 and a[0].value == "OLDSKOOL"
    assert len(b) == 1 and b[0].value == "BYLAW"


# 2. session A state does not overwrite session B state ----------------------
def test_no_cross_session_overwrite():
    mgr = ContextManager(db_path=_db())
    _say(mgr, "sess-a", "The test phrase is OLDSKOOL.")
    _say(mgr, "sess-b", "The test phrase is BYLAW.")
    _say(mgr, "sess-a", "Change the test phrase to OLDSKOOL V2.")
    a = mgr._get_store().fetch_states(scope="sess-a", status="current")
    b = mgr._get_store().fetch_states(scope="sess-b", status="current")
    assert a[0].value == "OLDSKOOL V2"
    assert b[0].value == "BYLAW"


# 3. correction supersedes only within the same session ------------------------
def test_correction_stays_in_scope():
    mgr = ContextManager(db_path=_db())
    _say(mgr, "sess-a", "The color is red.")
    _say(mgr, "sess-b", "The color is green.")
    _say(mgr, "sess-a", "The new color is blue.")
    old_a = mgr._get_store().fetch_states(scope="sess-a", status="superseded")
    assert {s.value for s in old_a} == {"red"}
    cur_b = mgr._get_store().fetch_states(scope="sess-b", status="current")
    assert len(cur_b) == 1 and cur_b[0].value == "green"


# 4. current-session state wins for current-session questions --------------------
# Raw cross-session retrieval may still mention the other session's rows
# (tagged history); what must not happen is presenting another session's
# value as a current state.
def test_current_session_wins():
    mgr = ContextManager(db_path=_db())
    _say(mgr, "sess-a", "The test phrase is OLDSKOOL.")
    _say(mgr, "sess-b", "The test phrase is BYLAW.")
    mgr2_view = mgr.build([{"role": "user", "content": "What is the current test phrase?"}],
                        "sess-b", token_counter=_counter,
                        max_new_tokens=5, budget=_budget()).view
    state_lines = [m["content"] for m in mgr2_view.messages
                   if m["content"].startswith("[state current")]
    assert any("BYLAW" in line for line in state_lines)
    assert not any("OLDSKOOL" in line for line in state_lines)


# 5. cross-session query returns distinct scoped values ------------------------------
def test_cross_session_distinct_values():
    mgr = ContextManager(db_path=_db())
    _say(mgr, "sess-a", "The test phrase is OLDSKOOL.")
    _say(mgr, "sess-b", "The test phrase is BYLAW.")
    texts = _ask_texts(mgr, "sess-c",
                       "What are the test phrases in our other sessions?")
    assert "OLDSKOOL" in texts and "BYLAW" in texts
    assert "sess-a" in texts and "sess-b" in texts
    assert "[states session sess-a" in texts
    assert "[states session sess-b" in texts


# 6. global state remains separate from session state -------------------------------------
def test_global_separate_from_session():
    mgr = ContextManager(db_path=_db())
    _say(mgr, "sess-a", "The test phrase is OLDSKOOL.")
    _say(mgr, "sess-a", "Globally, the motto is Rise Early.")
    texts = _ask_texts(mgr, "sess-b", "What is the current motto?")
    assert "Rise Early" in texts
    # Session B holds no test-phrase state: no current-state line may
    # present another session's value, even though raw history may mention it.
    texts_b = _ask_texts(mgr, "sess-b", "What is the current test phrase?")
    assert "[state current" not in texts_b


# 7. session rename updates authoritative metadata ----------------------------------------------
def test_session_rename_updates_metadata():
    mgr = ContextManager(db_path=_db())
    mgr.ensure_session("sess-a")
    rec = mgr.rename_session("sess-a", "Code Alert")
    assert rec is not None and rec.title == "Code Alert"
    rec2 = mgr.rename_session("sess-a", "Aurora Core")
    assert rec2.title == "Aurora Core"
    assert rec2.previous_titles == ["Code Alert"]
    assert rec2.session_id == "sess-a"


# 8. old session name remains historical but is not current -------------------------------------------
def test_old_name_historical_not_current():
    mgr = ContextManager(db_path=_db())
    mgr.ensure_session("sess-a")
    mgr.rename_session("sess-a", "Code Alert")
    mgr.rename_session("sess-a", "Aurora Core")
    texts = _ask_texts(mgr, "sess-a", "What is this session called?")
    assert "Aurora Core" in texts
    assert "Code Alert" not in texts.replace(
        "was previously called \"Code Alert\"", "")
    texts_old = _ask_texts(
        mgr, "sess-a", "What was this session previously called?")
    assert "Code Alert" in texts_old


# 9. current session metadata outranks historical mentions -----------------------------------------------
def test_metadata_outranks_mentions():
    mgr = ContextManager(db_path=_db())
    mgr.ensure_session("sess-a")
    mgr.rename_session("sess-a", "Aurora Core")
    # Raw history keeps mentioning the old name conversationally.
    mgr.reconcile(
        [{"role": "user", "content": "I loved the Crimson Orbit days"},
         {"role": "assistant", "content": "Crimson Orbit was fun"}], "sess-a")
    texts = _ask_texts(mgr, "sess-a", "What is this session called?")
    assert "Aurora Core" in texts
    assert "[session current]" in texts


# 10. session ID remains stable after rename ------------------------------------------------------------------
def test_session_id_stable_after_rename():
    mgr = ContextManager(db_path=_db())
    mgr.reconcile([{"role": "user", "content": "hello"}], "sess-a")
    before = mgr._get_store().count("sess-a")
    mgr.rename_session("sess-a", "Aurora Core")
    after = mgr._get_store().count("sess-a")
    assert before == after == 1
    rec = mgr._get_store().get_session("sess-a")
    assert rec.session_id == "sess-a" and rec.title == "Aurora Core"


# 11. digests do not resurrect superseded values as current ------------------------------------------------------
def test_digest_does_not_resurrect_superseded():
    mgr = ContextManager(db_path=_db())
    for i in range(55):
        mgr.reconcile(
            [{"role": "user", "content": f"filler discussion point {i} here"}], "s")
    mgr.reconcile([{"role": "user", "content": "The test phrase is Blue Lantern."}], "s")
    mgr.reconcile([{"role": "user", "content": "The new test phrase is Violet Aurora."}], "s")
    texts = _ask_texts(mgr, "s", "What is the current test phrase?")
    assert "Violet Aurora" in texts
    assert "was: Blue Lantern" not in texts
    assert "[state previous" not in texts


# 12. cross-session results retain identity and provenance --------------------------------------------------------------
def test_cross_session_provenance():
    mgr = ContextManager(db_path=_db())
    _say(mgr, "sess-a", "The test phrase is OLDSKOOL.")
    _say(mgr, "sess-b", "The test phrase is BYLAW.")
    texts = _ask_texts(mgr, "sess-c",
                       "List the test phrase for each session.")
    assert "src:" in texts
    assert "sess-a" in texts and "sess-b" in texts


# 13. long conversation still respects context budget ------------------------------------------------------------------
def test_long_conversation_budget():
    mgr = ContextManager(db_path=_db())
    for i in range(80):
        mgr.reconcile(
            [{"role": "user" if i % 2 == 0 else "assistant",
              "content": f"extended discussion turn {i} with substance here"}], "s")
    view = mgr.build([{"role": "user", "content": "continue please"}],
                     "s", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    assert view.input_tokens <= 110
    assert view.input_tokens + view.allowed_output <= 120


# 14. existing behavior: plain chat needs no state -----------------------------------------------------------
def test_plain_chat_needs_no_state():
    mgr = ContextManager(db_path=_db())
    view = mgr.build([{"role": "user", "content": "tell me a joke about cats"}],
                     "fresh", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    assert view.messages[-1] == {"role": "user", "content": "tell me a joke about cats"}
    assert "[state " not in " ".join(m["content"] for m in view.messages)


# Integration: OLDSKOOL / BYLAW across sessions --------------------------------
def test_integration_oldskool_bylaw():
    mgr = ContextManager(db_path=_db())
    mgr.ensure_session("session-a")
    mgr.rename_session("session-a", "Code Alert")
    mgr.ensure_session("session-b")
    mgr.rename_session("session-b", "Current Progress")
    mgr.reconcile([{"role": "user", "content": "The test phrase is OLDSKOOL."}],
                  "session-a")
    mgr.reconcile([{"role": "user", "content": "The test phrase is BYLAW."}],
                  "session-b")

    # Cross-session question from a third session.
    texts = _ask_texts(
        mgr, "session-c",
        "What are the test phrases from Session A and Session B?")
    assert "OLDSKOOL" in texts and "BYLAW" in texts
    assert "session-a" in texts and "session-b" in texts

    # Current question in C must not present either foreign value as a
    # current state (raw tagged history may still mention them).
    texts_cur = _ask_texts(mgr, "session-c", "What is the current test phrase?")
    assert "[state current" not in texts_cur

    # Session-specific question resolves correctly.
    texts_a = _ask_texts(mgr, "session-c",
                         "What is the test phrase for Session A?")
    assert "OLDSKOOL" in texts_a

    # Rename + authoritative metadata.
    mgr.rename_session("session-a", "Aurora Core")
    texts_name = _ask_texts(mgr, "session-a", "What is Session A called now?")
    assert "Aurora Core" in texts_name
    texts_old = _ask_texts(
        mgr, "session-a", "What was Session A previously called?")
    assert "Code Alert" in texts_old

    # Raw history intact everywhere (each build persists its own query:
    # A holds 1 statement + 2 name queries, B holds 1, C holds 3 queries).
    assert mgr._get_store().count("session-a") == 3
    assert mgr._get_store().count("session-b") == 1
    assert mgr._get_store().count("session-c") == 3


if __name__ == "__main__":
    test_same_concept_different_session_values()
    test_no_cross_session_overwrite()
    test_correction_stays_in_scope()
    test_current_session_wins()
    test_cross_session_distinct_values()
    test_global_separate_from_session()
    test_session_rename_updates_metadata()
    test_old_name_historical_not_current()
    test_metadata_outranks_mentions()
    test_session_id_stable_after_rename()
    test_digest_does_not_resurrect_superseded()
    test_cross_session_provenance()
    test_long_conversation_budget()
    test_plain_chat_needs_no_state()
    test_integration_oldskool_bylaw()
    print("All session-scope tests passed!")
