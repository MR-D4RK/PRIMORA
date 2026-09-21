"""Focused tests for the Prime-owned Intelligent Context Manager.

Conventions follow the repo: stdlib only, temp SQLite files, injected
fake token counter (word-based, deterministic), no live model.
"""
import os
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import (
    BudgetConfig,
    ContextManager,
    ContextTooLargeError,
)
from PRIMORA.prime.context.digest import build_extractive_digest
from PRIMORA.prime.context.models import StoredMessage
from PRIMORA.prime.context import scoring as scoring_mod
from PRIMORA.prime.context.store import ContextStore, fingerprint

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def _db_path(name):
    return os.path.join(_TEST_TMPDIR.name, name)


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _budget(**kwargs):
    params = dict(input_target=60, input_ceiling=80,
                  output_reserve=10, total_budget=90)
    params.update(kwargs)
    return BudgetConfig(**params)


def _mgr(name="t.db", **kwargs):
    return ContextManager(db_path=_db_path(name), **kwargs)


# 1. raw persistence round-trip -------------------------------------------
def test_raw_persistence_round_trip():
    store = ContextStore(_db_path("roundtrip.db"))
    store.append("s", 0, "user", "Hello there")
    store.append("s", 1, "assistant", "Hi!")
    rows = store.fetch_range("s")
    assert [(r.seq, r.role, r.content) for r in rows] == [
        (0, "user", "Hello there"), (1, "assistant", "Hi!")]
    assert rows[0].fingerprint == fingerprint("s", 0, "user", "Hello there")
    assert rows[0].timestamp > 0


# 2. snapshot re-ingestion is idempotent -----------------------------------
def test_snapshot_reingestion_idempotent():
    mgr = _mgr("idem.db")
    snap = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    assert mgr.reconcile(snap, "s") == []
    assert mgr.reconcile(snap, "s") == []
    assert mgr._get_store().count("s") == 2


# 3. identical messages at different positions stay distinct ---------------
def test_identical_messages_distinct_positions():
    mgr = _mgr("distinct.db")
    mgr.reconcile([{"role": "user", "content": "same"},
                   {"role": "user", "content": "same"}], "s")
    rows = mgr._get_store().fetch_range("s")
    assert len(rows) == 2
    assert [r.seq for r in rows] == [0, 1]


# 4. conflicting sequence content never overwrites ------------------------
def test_seq_conflict_appends_and_flags():
    mgr = _mgr("conflict.db")
    mgr.reconcile([{"role": "user", "content": "first"}], "s")
    diags = mgr.reconcile([{"role": "user", "content": "second"}], "s")
    rows = mgr._get_store().fetch_range("s")
    assert len(rows) == 2
    assert {r.content for r in rows} == {"first", "second"}
    assert any(d.startswith("seq_conflict") for d in diags)


# 5. newest messages retained ----------------------------------------------
def test_newest_messages_retained():
    mgr = _mgr("newest.db")
    msgs = [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"message number {i} with extra padding words here"}
            for i in range(30)]
    view = mgr.build(msgs, "s", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    texts = [m["content"] for m in view.messages]
    assert any("message number 29" in t for t in texts)
    assert any("message number 28" in t for t in texts)


# 6. current user turn never silently removed ------------------------------
def test_current_user_turn_preserved():
    mgr = _mgr("current.db")
    msgs = ([{"role": "user", "content": f"old filler message {i} words words"} for i in range(40)]
            + [{"role": "user", "content": "CRITICALFINALREQUEST"}])
    view = mgr.build(msgs, "s", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    assert view.messages[-1] == {"role": "user", "content": "CRITICALFINALREQUEST"}


# 7. relevance retrieval ----------------------------------------------------
def test_relevance_retrieval():
    mgr = _mgr("retrieval.db")
    old = [
        {"role": "user", "content": "the vault code is BLUE-FALCON-7 remember this"},
        {"role": "user", "content": "i enjoy gardening and rainy weekends"},
    ]
    filler = [{"role": "user", "content": f"filler chatter line {i} about nothing much"}
              for i in range(30)]
    mgr.reconcile(old + filler, "s")
    view = mgr.build(
        [{"role": "user", "content": "what is the vault code?"}], "s",
        token_counter=_counter, max_new_tokens=5, budget=_budget()).view
    texts = " ".join(m["content"] for m in view.messages)
    assert "BLUE-FALCON-7" in texts
    assert "gardening" not in texts


# 8. recency scoring ---------------------------------------------------------
def test_recency_scoring():
    new = scoring_mod.recency(95, 100)
    old = scoring_mod.recency(10, 100)
    assert new > old > 0


# 9. importance signals -------------------------------------------------------
def test_importance_signals():
    plain = scoring_mod.importance_of("hello there friend", "user")
    flagged = scoring_mod.importance_of("remember this, it is important", "user")
    assert flagged > plain


# 10. user-authority weighting -------------------------------------------------
def test_user_authority_weighting():
    text = "remember the meeting is at noon"
    assert (scoring_mod.importance_of(text, "user")
            > scoring_mod.importance_of(text, "assistant"))


# 11. digest extraction ---------------------------------------------------------
def test_digest_extraction_verbatim():
    msgs = [StoredMessage(1, "s", 0, "user", "The vault code is BLUE-FALCON-7.", 1.0, "fp", 0.5),
            StoredMessage(2, "s", 1, "assistant", "I like rainy weekends.", 2.0, "fp2", 0.0)]
    d = build_extractive_digest(msgs, ["vault", "code"], max_sentences=4)
    assert "BLUE-FALCON-7" in d["content"]
    assert d["source_ids"] == [1, 2] or d["source_ids"] == [1]
    for line in d["content"].split("\n"):
        body = line.split("] ", 1)[-1]
        assert any(body in m.content for m in msgs)


# 12. digest provenance -----------------------------------------------------------
def test_digest_provenance():
    mgr = _mgr("prov.db")
    mgr.reconcile([{"role": "user", "content": "Alpha fact one."},
                   {"role": "user", "content": "Beta fact two."}], "s")
    recs = mgr._get_store().fetch_digests("s")
    assert recs == []
    from PRIMORA.prime.context.digest import maybe_compact
    rec = maybe_compact(mgr._get_store(), "s", 1, query_terms=["fact"])
    assert rec is not None
    assert rec["seq_start"] == 0 and rec["seq_end"] == 1
    assert len(rec["source_ids"]) == 2
    assert rec["digest_type"] == "extractive_v1"
    stored = mgr._get_store().fetch_digests("s")
    assert len(stored) == 1 and stored[0]["source_ids"] == rec["source_ids"]


# 13. input ceiling enforcement ----------------------------------------------------
def test_input_ceiling_enforcement():
    mgr = _mgr("ceiling.db")
    msgs = [{"role": "user", "content": " ".join(["word"] * 40)} for _ in range(60)]
    view = mgr.build(msgs, "s", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    assert view.input_tokens <= 80


# 14. output-budget calculation -------------------------------------------------------
def test_output_budget_calculation():
    mgr = _mgr("outbudget.db")
    view = mgr.build([{"role": "user", "content": "hi there friend"}],
                     "s", token_counter=_counter,
                     max_new_tokens=50, budget=_budget()).view
    assert view.allowed_output == min(50, 90 - view.input_tokens)
    assert view.input_tokens + view.allowed_output <= 90


# 15. duplicate removal during assembly --------------------------------------------------
def test_duplicate_removal():
    mgr = _mgr("dedupe.db")
    dup = {"role": "user", "content": "remember the vault code BLUE-FALCON-7"}
    msgs = [dup, {"role": "assistant", "content": "noted"},
            dict(dup), dict(dup)]
    view = mgr.build(msgs, "s", token_counter=_counter,
                     max_new_tokens=5, budget=_budget(input_target=200, input_ceiling=300, total_budget=400)).view
    texts = [m["content"] for m in view.messages]
    assert texts.count(dup["content"]) == 1


# 16. Prime follow-up context rebuild ------------------------------------------------------
def test_prime_followup_rebuild():
    mgr = _mgr("followup.db")
    first = [{"role": "user", "content": "what is two plus two"}]
    v1 = mgr.build(first, "s", token_counter=_counter,
                   max_new_tokens=5, budget=_budget()).view
    followup = first + [{"role": "assistant", "content": "computing"},
                        {"role": "tool", "content": '{"status":"success"}'}]
    v2 = mgr.build(followup, "s", token_counter=_counter,
                   max_new_tokens=5, budget=_budget()).view
    roles = [m["role"] for m in v2.messages]
    assert "tool" in roles
    assert v2.input_tokens <= 80
    assert v2.messages[-1]["role"] == "tool"


# 17. SQLite failure fallback -----------------------------------------------------------------
def test_db_failure_fallback():
    import tempfile
    blocker = tempfile.NamedTemporaryFile(delete=False)
    blocker.close()
    mgr = ContextManager(db_path=os.path.join(blocker.name, "db.sqlite"))
    view = mgr.build([{"role": "user", "content": "hello world"}],
                     "s", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    assert view.messages[-1] == {"role": "user", "content": "hello world"}
    assert any(d.startswith("store_unavailable") for d in view.diagnostics)


# 18. oversized current message ------------------------------------------------------------------
def test_oversized_current_message():
    mgr = _mgr("oversize.db")
    big = {"role": "user", "content": " ".join(["word"] * 500)}
    try:
        mgr.build([big], "s", token_counter=_counter,
                  max_new_tokens=5, budget=_budget())
        assert False, "expected ContextTooLargeError"
    except ContextTooLargeError:
        pass
    rows = mgr._get_store().fetch_range("s")
    assert len(rows) == 1 and rows[0].content == big["content"]


# 19. hostile punctuation must not abort retrieval -------------------------------
def test_retrieval_hostile_punctuation_end_to_end():
    mgr = _mgr("hostile.db")
    mgr.reconcile(
        [{"role": "user", "content": "the vault code is BLUE-FALCON-7"}], "s")
    view = mgr.build(
        [{"role": "user",
          "content": "don't stop *believin* (yeah) \"vault\" code?"}],
        "s", token_counter=_counter,
        max_new_tokens=5, budget=_budget()).view
    texts = " ".join(m["content"] for m in view.messages)
    assert "BLUE-FALCON-7" in texts
    assert view.input_tokens <= 80


# 20. one bad term is skipped, remaining terms still retrieve ------------------
def test_retrieval_skips_bad_term_keeps_good():
    import sqlite3
    from unittest.mock import patch
    mgr = _mgr("mixedterms.db")
    mgr.reconcile(
        [{"role": "user", "content": "the vault code is BLUE-FALCON-7"}], "s")
    store = mgr._get_store()
    real_search = store.search

    def flaky_search(session_id, query, limit=10):
        if query == "vault":
            raise sqlite3.OperationalError("fts5: syntax error near \"'\"")
        return real_search(session_id, query, limit)

    with patch.object(store, "search", side_effect=flaky_search):
        # Bypass the cached-store lookup so our patched instance is used.
        mgr._store = store
        view = mgr.build(
            [{"role": "user", "content": "vault code retrieval check"}],
            "s", token_counter=_counter,
            max_new_tokens=5, budget=_budget()).view
    texts = " ".join(m["content"] for m in view.messages)
    assert "BLUE-FALCON-7" in texts
    assert any(d.startswith("retrieval_skipped_term") for d in view.diagnostics)


# 21. stopword/punctuation-only query degrades to recent context ---------------
def test_retrieval_empty_query_terms_safe():
    mgr = _mgr("emptyterms.db")
    mgr.reconcile([{"role": "user", "content": "some earlier remark"}], "s")
    view = mgr.build([{"role": "user", "content": "... !!! ???"}],
                     "s", token_counter=_counter,
                     max_new_tokens=5, budget=_budget()).view
    assert view.messages[-1] == {"role": "user", "content": "... !!! ???"}
    assert view.input_tokens <= 80


# 22. rebuilds never re-digest already-covered material ------------------------
def test_compaction_skips_covered_ranges():
    mgr = _mgr("noredigest.db")
    for i in range(55):
        mgr.reconcile(
            [{"role": "user", "content": f"filler discussion point {i} here"}], "s")
    mgr.build([{"role": "user", "content": "first query here"}],
              "s", token_counter=_counter,
              max_new_tokens=5, budget=_budget())
    mgr.build([{"role": "user", "content": "second query here"}],
              "s", token_counter=_counter,
              max_new_tokens=5, budget=_budget())
    seen = []
    for d in mgr._get_store().fetch_digests("s", limit=20):
        ids = set(d["source_ids"])
        assert ids, "digest must cite sources"
        for prev in seen:
            assert ids.isdisjoint(prev), "digest reprocessed covered rows"
        seen.append(ids)
    assert seen, "expected at least one digest"


# 23. new digests begin from uncovered material --------------------------------
def test_new_digest_covers_only_uncovered():
    mgr = _mgr("uncovered.db")
    for i in range(55):
        mgr.reconcile(
            [{"role": "user", "content": f"filler discussion point {i} here"}], "s")
    mgr.build([{"role": "user", "content": "first query here"}],
              "s", token_counter=_counter,
              max_new_tokens=5, budget=_budget())
    covered = set()
    for d in mgr._get_store().fetch_digests("s", limit=20):
        covered.update(d["source_ids"])
    for i in range(55, 80):
        mgr.reconcile(
            [{"role": "user", "content": f"fresh discussion point {i} here"}], "s")
    mgr.build([{"role": "user", "content": "second query here"}],
              "s", token_counter=_counter,
              max_new_tokens=5, budget=_budget())
    fresh = set()
    for d in mgr._get_store().fetch_digests("s", limit=20):
        fresh.update(d["source_ids"])
    assert fresh - covered, "expected a digest covering the new material"


# 24. digest provenance describes exactly the digested messages ---------------
def test_maybe_compact_exact_provenance():
    from PRIMORA.prime.context.digest import maybe_compact
    store = ContextStore(_db_path("provenance.db"))
    for i in range(6):
        store.append("s", i, "user", f"numbered fact message {i} here.")
    messages = store.fetch_range("s", 2, 5)
    assert [m.seq for m in messages] == [2, 3, 4, 5]
    digest = maybe_compact(store, "s", 999, query_terms=["fact"],
                           max_sentences=10, messages=messages)
    assert digest is not None
    assert digest["seq_start"] == 2 and digest["seq_end"] == 5
    input_ids = {m.message_id for m in messages}
    assert set(digest["source_ids"]) <= input_ids
    assert digest["source_ids"]


if __name__ == "__main__":
    test_raw_persistence_round_trip()
    test_snapshot_reingestion_idempotent()
    test_identical_messages_distinct_positions()
    test_seq_conflict_appends_and_flags()
    test_newest_messages_retained()
    test_current_user_turn_preserved()
    test_relevance_retrieval()
    test_recency_scoring()
    test_importance_signals()
    test_user_authority_weighting()
    test_digest_extraction_verbatim()
    test_digest_provenance()
    test_input_ceiling_enforcement()
    test_output_budget_calculation()
    test_duplicate_removal()
    test_prime_followup_rebuild()
    test_db_failure_fallback()
    test_oversized_current_message()
    test_retrieval_hostile_punctuation_end_to_end()
    test_retrieval_skips_bad_term_keeps_good()
    test_retrieval_empty_query_terms_safe()
    test_compaction_skips_covered_ranges()
    test_new_digest_covers_only_uncovered()
    test_maybe_compact_exact_provenance()
    print("All context unit tests passed!")
