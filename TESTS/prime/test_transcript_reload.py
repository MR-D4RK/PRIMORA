"""Transcript reload: store fetch, token budget, session floor, idempotence.

Conventions follow the repo: stdlib only, temp SQLite files, no live model.
"""
import os
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context.manager import (
    ContextManager,
    TRANSCRIPT_RELOAD_LIMIT,
    TRANSCRIPT_TOKEN_BUDGET,
)
from PRIMORA.prime.context.store import ContextStore, CONVERSATION_CLEARED

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def _path(name):
    return os.path.join(_TEST_TMPDIR.name, name)


def _mgr(name="reload.db"):
    return ContextManager(db_path=_path(name))


# 1. fetch_conversation: order, filters, limit -----------------------------
def test_fetch_conversation_chronological_and_filtered():
    store = ContextStore(_path("order.db"))
    store.append("s", 0, "system", "SYSTEM instruction block")
    store.append("s", 1, "user", "first question")
    store.append("s", 2, "assistant", "first answer")
    store.append("s", 3, "tool", "tool payload")
    store.append("s", 4, "assistant",
                 "tool use <prime_request>hidden body</prime_request>")
    store.append("s", 5, "user", "second question")
    rows = store.fetch_conversation("s")
    assert [(r.role, r.content) for r in rows] == [
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "second question"),
    ]


def test_fetch_conversation_chronological_despite_legacy_seq_cram():
    # Old CLI rows all share seq 0; insertion id is the chronology.
    store = ContextStore(_path("cram.db"))
    store.append("s", 0, "user", "old one")
    store.append("s", 0, "user", "old two")
    store.append("s", 0, "user", "old three")
    rows = store.fetch_conversation("s")
    assert [r.content for r in rows] == ["old one", "old two", "old three"]
    assert [r.message_id for r in rows] == sorted(r.message_id for r in rows)


def test_fetch_conversation_limit_keeps_newest():
    store = ContextStore(_path("limit.db"))
    for i in range(5):
        store.append("s", i, "user", f"m{i}")
    rows = store.fetch_conversation("s", limit=2)
    assert [r.content for r in rows] == ["m3", "m4"]
    assert store.fetch_conversation("s", limit=0) == []


def test_fetch_conversation_session_isolated():
    store = ContextStore(_path("iso.db"))
    store.append("a", 0, "user", "only in a")
    store.append("b", 0, "user", "only in b")
    assert [r.content for r in store.fetch_conversation("a")] == ["only in a"]
    assert [r.content for r in store.fetch_conversation("b")] == ["only in b"]


def test_fetch_conversation_empty_session():
    assert ContextStore(_path("empty.db")).fetch_conversation("none") == []


def test_fetch_conversation_stops_at_clear_marker():
    store = ContextStore(_path("marker.db"))
    store.append("s", 0, "user", "before clear")
    store.append("s", 1, "assistant", "also before clear")
    store.append("s", 2, "system", CONVERSATION_CLEARED)
    store.append("s", 3, "user", "after clear")
    rows = store.fetch_conversation("s")
    assert [r.content for r in rows] == ["after clear"]
    # Raw truth survives: nothing was deleted.
    assert store.count("s") == 4


# 2. load_transcript: budget, silence, defaults ----------------------------
def test_load_transcript_default_shape():
    mgr = _mgr("shape.db")
    mgr.reconcile([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ], "s")
    entries = mgr.load_transcript("s")
    assert [e["role"] for e in entries] == ["user", "assistant"]
    assert all(set(e) >= {"role", "content", "seq"} for e in entries)
    assert all(isinstance(e["seq"], int) for e in entries)
    assert TRANSCRIPT_RELOAD_LIMIT == 30


def test_load_transcript_respects_limit_and_budget():
    mgr = _mgr("budget.db")
    mgr.reconcile([{"role": "user", "content": "tiny"}], "s")
    mgr.reconcile([{"role": "user",
                    "content": "x " * 500}], "s")  # 1000 chars = 250 tokens
    mgr.reconcile([{"role": "user", "content": "newest"}], "s")
    # Estimates: tiny 1 + big 250 + newest 1 = 252 tokens.
    assert len(mgr.load_transcript("s")) == 3            # default 600 fits
    assert len(mgr.load_transcript("s", token_budget=1000)) == 3
    # Tight budget drops oldest first, never the newest message.
    entries = mgr.load_transcript("s", token_budget=251)
    assert [e["content"] for e in entries] == ["x " * 500, "newest"]
    assert TRANSCRIPT_TOKEN_BUDGET == 600


def test_load_transcript_never_raises_without_store():
    mgr = _mgr("nostore.db")

    def _boom():
        raise RuntimeError("store down")

    mgr._get_store = _boom
    assert mgr.load_transcript("s") == []
    assert mgr.next_seq("s") == 0


def test_load_transcript_after_clear_marker():
    mgr = _mgr("cleared.db")
    mgr.reconcile([
        {"role": "user", "content": "kept for now"},
        {"role": "user", "content": "also kept"},
    ], "s")
    seq = mgr.next_seq("s")
    mgr.reconcile([{"role": "system", "content": CONVERSATION_CLEARED,
                    "seq": seq}], "s")
    assert mgr.load_transcript("s") == []
    mgr.reconcile([{"role": "user", "content": "fresh turn",
                    "seq": seq + 1}], "s")
    assert [e["content"] for e in mgr.load_transcript("s")] == ["fresh turn"]


# 3. next_seq --------------------------------------------------------------
def test_next_seq_tracks_true_maximum():
    mgr = _mgr("nextseq.db")
    assert mgr.next_seq("s") == 0
    store = mgr._get_store()
    store.append("s", 7, "user", "seven")
    store.append("s", 3, "user", "three")     # lower seq appended later
    store.append("s", 9, "user", "nine")
    assert mgr.next_seq("s") == 10


# 4. reload re-persist is idempotent --------------------------------------
def test_reload_repersist_is_noop_with_clean_seq():
    mgr = _mgr("clean_noop.db")
    snap = [
        {"role": "user", "content": "What is my favorite color?"},
        {"role": "assistant", "content": "I don't have that recorded."},
        {"role": "user", "content": "It's yellow."},
    ]
    first = mgr.reconcile(snap, "s")
    assert any(d.startswith("state_updated") for d in first)
    store = mgr._get_store()
    before = store.count("s")
    loaded = mgr.load_transcript("s")
    diags = mgr.reconcile(
        [{"role": e["role"], "content": e["content"], "seq": e["seq"]}
         for e in loaded],
        "s",
    )
    assert store.count("s") == before          # no rows added
    assert diags == []                          # no conflicts
    # The new turn after a reload sequences past everything stored.
    diag_new = mgr.reconcile(
        [{"role": "user", "content": "blue", "seq": mgr.next_seq("s")}],
        "s")
    assert not any(d.startswith("seq_conflict") for d in diag_new)
    assert store.count("s") == before + 1


def test_reload_repersist_adds_no_rows_even_when_seq_crammed():
    # Legacy sessions crammed every row onto one seq: reload still must not
    # duplicate anything (conflict chatter about the old mess is tolerated).
    store = ContextStore(_path("cram_reload.db"))
    store.append("s", 0, "user", "legacy one")
    store.append("s", 0, "user", "legacy two")
    store.append("s", 0, "assistant", "legacy reply")
    mgr = ContextManager(db_path=_path("cram_reload.db"))
    loaded = mgr.load_transcript("s")
    assert len(loaded) == 3
    before = store.count("s")
    mgr.reconcile(
        [{"role": e["role"], "content": e["content"], "seq": e["seq"]}
         for e in loaded],
        "s",
    )
    assert store.count("s") == before


if __name__ == "__main__":
    test_fetch_conversation_chronological_and_filtered()
    test_fetch_conversation_chronological_despite_legacy_seq_cram()
    test_fetch_conversation_limit_keeps_newest()
    test_fetch_conversation_session_isolated()
    test_fetch_conversation_empty_session()
    test_fetch_conversation_stops_at_clear_marker()
    test_load_transcript_default_shape()
    test_load_transcript_respects_limit_and_budget()
    test_load_transcript_never_raises_without_store()
    test_load_transcript_after_clear_marker()
    test_next_seq_tracks_true_maximum()
    test_reload_repersist_is_noop_with_clean_seq()
    test_reload_repersist_adds_no_rows_even_when_seq_crammed()
    print("All transcript reload tests passed!")
