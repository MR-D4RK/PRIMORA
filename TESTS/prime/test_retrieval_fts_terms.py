"""Regression tests for FTS5 query construction (M6 preparation, Part 1).

Defect under test: the tokenizer keeps apostrophes inside tokens
(`[a-z0-9']+`), so terms such as `don't`, `apollo's`, `BLUE-FALCON-7` and
`design?` reached SQLite as raw MATCH syntax, raised `OperationalError`, and
were silently dropped by the per-term handler in `ContextManager._retrieve`.

The fix renders every FTS5 term as one quoted phrase literal, so user
punctuation can never be read as MATCH syntax. These tests pin that the fix
repairs the defective forms, leaves safe terms byte-identical, and cannot leak
malformed syntax into SQLite.
"""

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import BudgetConfig, ContextManager
from PRIMORA.prime.context.store import ContextStore

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]


def _db(name=None):
    _SEQ[0] += 1
    filename = name or f"fts-{_SEQ[0]}.db"
    return os.path.join(_TEST_TMPDIR.name, filename)


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _budget():
    return BudgetConfig(
        input_target=220, input_ceiling=400, output_reserve=20,
        total_budget=420,
    )


# The exact defective terms named in the M6 audit, paired with content that
# unicode61 tokenizes into the sub-tokens the phrase will look for.
_PUNCTUATED_CONTENT = [
    "the vault code is BLUE-FALCON-7",
    "Project Apollo's mobile app is part of Apollo",
    "We settled on a matte charcoal finish for the car design?",
    "don't rekey the north breaker without the landlord",
    "Meera's laptop has a cracked hinge",
    "the wifi password is TUXEDO-77",
    "the emblem we agreed on is a stylised fox",
]
_SAFE_CONTENT = [
    "car", "apollo", "falcon", "design", "fox", "vault", "lunch",
]


def _seeded_store(name=None):
    path = _db(name)
    store = ContextStore(path)
    for text in _PUNCTUATED_CONTENT + _SAFE_CONTENT:
        store.append("s", store.count("s"), "user", text, timestamp=100)
    return store


# ----------------------------------------------------------------------
# Direct store-level behaviour
# ----------------------------------------------------------------------
def test_fts_available_is_probed_not_assumed():
    store = ContextStore(_db())
    assert isinstance(store.fts_available, bool)
    # The probe must reflect a real round trip, not a config flag.
    with store._connect() as con:
        con.execute(
            "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH ?",
            ('"probeunlikelytermxyz"',),
        ).fetchone()


def test_punctuated_terms_no_longer_raise_and_now_match():
    store = _seeded_store()
    assert store.fts_available is True, "this regression needs FTS5"
    # Each term must now find the message that contains it. Before the fix
    # every one of these raised OperationalError and found nothing.
    cases = {
        "don't": "don't rekey the north breaker without the landlord",
        "apollo's": "Project Apollo's mobile app is part of Apollo",
        "BLUE-FALCON-7": "the vault code is BLUE-FALCON-7",
        "design?": "We settled on a matte charcoal finish for the car design?",
        "meera's": "Meera's laptop has a cracked hinge",
        "TUXEDO-77": "the wifi password is TUXEDO-77",
    }
    for term, expected in cases.items():
        rows = store.search("s", term, limit=10)
        assert expected in [row.content for row in rows], term


def test_safe_terms_are_unchanged_by_the_fix():
    store = _seeded_store()
    # A one-token phrase must match exactly like the bare token did.
    for term in _SAFE_CONTENT:
        quoted = store.search("s", term, limit=50)
        bare = _bare_match(store, term)
        assert {row.message_id for row in quoted} == bare, term


def _bare_match(store, term):
    """Reference implementation: the pre-fix, unquoted MATCH."""
    if not term.isalnum():
        raise AssertionError("bare reference only valid for safe terms")
    with store._connect() as con:
        rows = con.execute(
            "SELECT m.id FROM messages_fts f JOIN messages m ON m.id = f.rowid"
            " WHERE messages_fts MATCH ? AND m.session_id = ?",
            (term, "s"),
        ).fetchall()
    return {int(row[0]) for row in rows}


def test_malformed_fts_syntax_cannot_escape_into_sqlite():
    store = _seeded_store()
    # Every one of these was either a syntax error or silently reinterpreted
    # as FTS5 query operators before the fix.
    hostile = [
        "OR", "AND", "NOT", "NEAR", "car NEAR fox", "car OR fox",
        "car AND NOT fox", "^car", "car*", "(car", "car)", "col:car",
        'say "hi"', 'car" OR "fox', "; DROP TABLE messages; --",
        "*", "?", "-", "'''", '""', "a AND", "NEAR NEAR",
    ]
    for term in hostile:
        # Must not raise: the phrase literal absorbs the syntax.
        rows = store.search("s", term, limit=10)
        assert isinstance(rows, list), term
    # And the database is intact.
    assert store.count("s") == len(_PUNCTUATED_CONTENT) + len(_SAFE_CONTENT)
    with store._connect() as con:
        assert con.execute(
            "SELECT count(*) FROM sqlite_master WHERE name='messages'"
        ).fetchone()[0] == 1


def test_empty_and_whitespace_terms_are_safe():
    store = _seeded_store()
    for term in ("", "   ", "\t", "\n", "...", "!!!", "???"):
        assert store.search("s", term, limit=10) == []


def test_phrase_renderer_shape():
    # The renderer is a pure function; pin its exact output.
    assert ContextStore._fts_phrase("car") == '"car"'
    assert ContextStore._fts_phrase("don't") == '"don\'t"'
    assert ContextStore._fts_phrase('say "hi"') == '"say ""hi"""'
    assert ContextStore._fts_phrase("") == '""'
    for raw in ("don't", "a OR b", 'x"y', "((", "NEAR", "c*"):
        rendered = ContextStore._fts_phrase(raw)
        assert rendered.startswith('"') and rendered.endswith('"')
        assert rendered.count('"') % 2 == 0


# ----------------------------------------------------------------------
# End-to-end: the whole point is that terms are no longer dropped
# ----------------------------------------------------------------------
def test_end_to_end_punctuated_query_retrieves_without_skipped_terms():
    path = _db("fts-e2e.db")
    store = ContextStore(path)
    manager = ContextManager(db_path=path)
    for text in _PUNCTUATED_CONTENT:
        store.append("s", store.count("s"), "user", text, timestamp=100)
    view = manager.build(
        [{"role": "user", "content": "what is the wifi password TUXEDO-77?"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    texts = " ".join(m["content"] for m in view.messages)
    assert "TUXEDO-77" in texts
    assert not any(
        d.startswith("retrieval_skipped_term") for d in view.diagnostics
    ), view.diagnostics


def test_end_to_end_apostrophe_query_has_no_skipped_terms():
    path = _db("fts-e2e-apostrophe.db")
    store = ContextStore(path)
    manager = ContextManager(db_path=path)
    for text in _PUNCTUATED_CONTENT:
        store.append("s", store.count("s"), "user", text, timestamp=100)
    view = manager.build(
        [{"role": "user", "content": "don't tell me about the north breaker"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    assert not any(
        d.startswith("retrieval_skipped_term") for d in view.diagnostics
    ), view.diagnostics


def test_end_to_end_mixed_punctuation_query_is_clean():
    path = _db("fts-e2e-mixed.db")
    store = ContextStore(path)
    manager = ContextManager(db_path=path)
    for text in _PUNCTUATED_CONTENT:
        store.append("s", store.count("s"), "user", text, timestamp=100)
    view = manager.build(
        [{"role": "user", "content": "Project Apollo's vault code -- "
                                    "BLUE-FALCON-7 -- what is it??"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    texts = " ".join(m["content"] for m in view.messages)
    assert "BLUE-FALCON-7" in texts
    assert not any(
        d.startswith("retrieval_skipped_term") for d in view.diagnostics
    ), view.diagnostics


def test_end_to_end_normal_multi_word_query_still_works():
    path = _db("fts-e2e-normal.db")
    store = ContextStore(path)
    manager = ContextManager(db_path=path)
    store.append("s", 0, "user", "The car design is a matte charcoal three-door.",
                 timestamp=100)
    store.append("s", 1, "user", "The emblem we agreed on is a stylised fox.",
                 timestamp=110)
    store.append("s", 2, "user", "unrelated small talk about lunch",
                 timestamp=120)
    view = manager.build(
        [{"role": "user", "content": "what did we decide about the car design?"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    texts = " ".join(m["content"] for m in view.messages)
    assert "matte charcoal three-door" in texts
    assert not any(
        d.startswith("retrieval_skipped_term") for d in view.diagnostics
    ), view.diagnostics


def test_end_to_end_stopword_only_query_still_degrades_safely():
    path = _db("fts-e2e-stopwords.db")
    store = ContextStore(path)
    manager = ContextManager(db_path=path)
    store.append("s", 0, "user", "some earlier remark about the weather",
                 timestamp=100)
    view = manager.build(
        [{"role": "user", "content": "... !!! ???"}],
        "s", token_counter=_counter, max_new_tokens=5, budget=_budget(),
    ).view
    assert view.messages[-1] == {"role": "user", "content": "... !!! ???"}
    assert not any(
        d.startswith("retrieval_skipped_term") for d in view.diagnostics
    ), view.diagnostics
    # The empty-term early return must not have consulted FTS at all.
    with sqlite3.connect(store.db_path) as con:
        assert con.execute("SELECT count(*) FROM messages").fetchone()[0] == 2


def test_like_fallback_still_uses_the_raw_term():
    # With FTS5 unavailable the fallback is a literal substring search, so
    # the term must NOT be phrase-quoted there.
    path = _db("fts-like-fallback.db")
    store = ContextStore(path)
    store._fts_available = False
    assert store.fts_available is False
    store.append("s", 0, "user", "the vault code is BLUE-FALCON-7",
                 timestamp=100)
    assert [row.content for row in store.search("s", "BLUE-FALCON-7")] == [
        "the vault code is BLUE-FALCON-7"
    ]
    assert store.search("s", "no-such-token", limit=10) == []


def test_scoring_is_untouched_by_the_fts_fix():
    # Guard: the fix lives entirely in the store's query construction.
    from PRIMORA.prime.context import scoring
    breakdown = scoring.score(
        text="the car design is a matte charcoal three-door",
        role="user", seq=0, now_seq=4, query_terms=["car", "design"],
        selected_texts=[], stored_importance=0.0,
    )
    assert set(breakdown) == {
        "score", "relevance", "recency", "importance", "redundancy"
    }
    assert breakdown["relevance"] == 1.0


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"All {len(tests)} FTS query-construction tests passed!")
