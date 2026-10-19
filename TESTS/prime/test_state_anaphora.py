"""Context-aware extraction: elliptical replies resolved against the question.

Conventions follow the repo: stdlib only, temp SQLite files, no live model.
Covers the favourite-color failure chain at the unit/extraction level and
the reconcile -> state -> build-injection path end to end on temp stores.
"""
import os
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context.manager import ContextManager
from PRIMORA.prime.context.state import (
    extract_candidates,
    _question_slot_phrase,
    _anaphora_value,
)

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def _db_path(name):
    return os.path.join(_TEST_TMPDIR.name, name)


def _mgr(name="anaphora.db"):
    return ContextManager(db_path=_db_path(name))


def _states(store, concept=None, session_id=None):
    return store.fetch_states(session_id=session_id, concept=concept)


# 1. pure function: anaphora + slot question -> direct-statement candidate --
def test_anaphora_with_slot_question_extracts():
    cands = extract_candidates(
        "It's yellow.", "user", "What is my favorite color?"
    )
    assert len(cands) == 1
    c = cands[0]
    assert (c.concept, c.value, c.kind, c.scope) == (
        "favorite color", "yellow", "establish", "global")
    assert c.auto_confirm is True
    assert c.slot == "favorite color"


def test_anaphora_forms_and_bare_value():
    q = "What's my favorite color?"
    for text in ("its yellow", "That's yellow", "yellow."):
        cands = extract_candidates(text, "user", q)
        assert len(cands) == 1, text
        assert cands[0].concept == "favorite color"
        assert cands[0].value == "yellow"


def test_colour_spelling_routes_to_color_slot():
    cands = extract_candidates(
        "It's yellow.", "user", "whats my favourite colour?"
    )
    assert len(cands) == 1
    assert cands[0].concept == "favorite color"


# 2. refusal cases: nothing is invented -----------------------------------
def test_without_context_no_extraction():
    assert extract_candidates("It's yellow.", "user") == []
    assert extract_candidates("It's yellow.", "user", None) == []


def test_unmappable_question_no_extraction():
    assert extract_candidates(
        "It's yellow.", "user", "What did we decide?"
    ) == []


def test_void_and_question_replies_no_extraction():
    q = "What is my favorite color?"
    assert extract_candidates("no", "user", q) == []
    assert extract_candidates("Is it yellow?", "user", q) == []
    assert extract_candidates("thanks", "user", q) == []


def test_instructions_and_lists_are_not_answers():
    q = "What is my favorite color?"
    # Regression: this exact message used to become the fact's value.
    assert extract_candidates(
        "List my previous favorite colors, old ones included.", "user", q
    ) == []
    assert extract_candidates("List my colors", "user", q) == []
    assert extract_candidates("Tell me now", "user", q) == []
    assert extract_candidates("red, blue and green", "user", q) == []


def test_first_person_statement_is_not_an_answer():
    assert extract_candidates(
        "I told you yesterday it's yellow", "user", "What is my favorite color?"
    ) == []


# 3. direct phrasing behaves exactly as before ----------------------------
def test_direct_statement_unchanged_without_context():
    cands = extract_candidates("my favorite color is yellow", "user")
    assert len(cands) == 1
    assert (cands[0].concept, cands[0].value, cands[0].auto_confirm) == (
        "favorite color", "yellow", True)


def test_hedge_downgrades_expanded_candidate():
    cands = extract_candidates(
        "I think it's yellow.", "user", "What is my favorite color?"
    )
    assert len(cands) == 1
    assert cands[0].value == "yellow"
    assert cands[0].auto_confirm is False


# 4. helpers ---------------------------------------------------------------
def test_question_slot_phrase_mapping():
    assert _question_slot_phrase("What is my favorite color?") == "favorite color"
    assert _question_slot_phrase("whats my name") == "name"
    assert _question_slot_phrase("What did we decide?") is None
    assert _question_slot_phrase("I told you yesterday") is None
    assert _question_slot_phrase(None) is None


def test_anaphora_value_extraction():
    assert _anaphora_value("It's yellow.") == "yellow"
    assert _anaphora_value("yellow.") == "yellow"
    assert _anaphora_value("No.") == ""
    assert _anaphora_value("Is it yellow?") == ""
    assert _anaphora_value("") == ""


# 5. reconcile -> state, both snapshot shapes ------------------------------
def test_full_snapshot_creates_state():
    mgr = _mgr("full_snap.db")
    diags = mgr.reconcile([
        {"role": "user", "content": "What is my favorite color?"},
        {"role": "assistant", "content": "I don't have that recorded."},
        {"role": "user", "content": "It's yellow."},
    ], "default")
    assert not [d for d in diags if d.startswith(("store_unavailable", "state_failed"))]
    rows = _states(mgr._get_store(), concept="favorite color")
    assert [(s.value, s.status, s.scope, s.authority) for s in rows] == [
        ("yellow", "current", "global", "user")]


def test_single_message_uses_store_question_as_context():
    # Legacy caller shape: one message per reconcile, question already stored.
    mgr = _mgr("single_msg.db")
    mgr.reconcile(
        [{"role": "user", "content": "What is my favorite color?"}], "default")
    diags = mgr.reconcile(
        [{"role": "user", "content": "its yellow"}], "default")
    assert not [d for d in diags if d.startswith(("store_unavailable", "state_failed"))]
    rows = _states(mgr._get_store(), concept="favorite color")
    assert any(s.value == "yellow" and s.status == "current" for s in rows)


def test_snapshot_context_wins_over_store():
    # Elliptical answer next to its own question inside one snapshot.
    mgr = _mgr("snapshot_ctx.db")
    mgr.reconcile([
        {"role": "user", "content": "what is my favorite color"},
        {"role": "user", "content": "It's yellow."},
    ], "s")
    rows = _states(mgr._get_store(), concept="favorite color")
    assert any(s.value == "yellow" for s in rows)


# 6. correction and persona separation ------------------------------------
def test_correction_supersedes():
    mgr = _mgr("correction.db")
    mgr.reconcile([
        {"role": "user", "content": "What is my favorite color?"},
        {"role": "user", "content": "It's yellow."},
    ], "default")
    mgr.reconcile([{"role": "user", "content": "Actually, green."}], "default")
    rows = _states(mgr._get_store(), concept="favorite color")
    assert rows[-1].value == "green"
    assert rows[-1].status == "current"


def test_assistant_claim_never_overrides_user_state():
    mgr = _mgr("persona.db")
    mgr.reconcile([
        {"role": "user", "content": "What is my favorite color?"},
        {"role": "user", "content": "It's yellow."},
    ], "default")
    diags = mgr.reconcile(
        [{"role": "assistant", "content": "My favorite color is blue."}],
        "default")
    rows = _states(mgr._get_store(), concept="favorite color")
    current = [s for s in rows if s.status == "current"]
    unresolved = [s for s in rows if s.status == "unresolved"]
    assert [s.value for s in current] == ["yellow"]
    assert [s.value for s in unresolved] == ["blue"]
    assert [s.authority for s in unresolved] == ["assistant"]
    assert diags is not None


def test_reconcile_snapshot_twice_no_duplicate_state():
    mgr = _mgr("idem_state.db")
    snap = [
        {"role": "user", "content": "What is my favorite color?"},
        {"role": "user", "content": "It's yellow."},
    ]
    first = mgr.reconcile(snap, "s")
    assert any(d.startswith("state_updated") for d in first)
    # Re-sending the identical snapshot re-persists nothing and, with it,
    # re-extracts nothing: one state, zero diagnostics.
    assert mgr.reconcile(snap, "s") == []
    rows = _states(mgr._get_store(), concept="favorite color")
    assert len(rows) == 1


# 7. the fact reaches the model view after "restart" ----------------------
def test_build_view_contains_fact_from_state():
    mgr = _mgr("build_fact.db")
    mgr.reconcile([
        {"role": "user", "content": "What is my favorite color?"},
        {"role": "user", "content": "It's yellow."},
    ], "default")
    # Same manager, new turn: the state (not the old chat) must inject.
    result = mgr.build(
        [{"role": "user", "content": "What is my favorite color?"}],
        "default",
        token_counter=lambda msgs: sum(len(m.get("content", "").split()) for m in msgs),
    )
    view = result.view if isinstance(result.view, str) else str(result.view)
    assert "favorite color" in view
    assert "yellow" in view
    assert "The user's favorite color is yellow." in view


if __name__ == "__main__":
    test_anaphora_with_slot_question_extracts()
    test_anaphora_forms_and_bare_value()
    test_colour_spelling_routes_to_color_slot()
    test_without_context_no_extraction()
    test_unmappable_question_no_extraction()
    test_void_and_question_replies_no_extraction()
    test_first_person_statement_is_not_an_answer()
    test_direct_statement_unchanged_without_context()
    test_hedge_downgrades_expanded_candidate()
    test_question_slot_phrase_mapping()
    test_anaphora_value_extraction()
    test_full_snapshot_creates_state()
    test_single_message_uses_store_question_as_context()
    test_snapshot_context_wins_over_store()
    test_correction_supersedes()
    test_assistant_claim_never_overrides_user_state()
    test_reconcile_snapshot_twice_no_duplicate_state()
    test_build_view_contains_fact_from_state()
    print("All state anaphora tests passed!")
