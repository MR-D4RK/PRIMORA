"""CLI continuity: restored transcript, accumulated thread, stub parity.

Conventions follow the repo: stdlib only, temp SQLite files, patched
``builtins.input``, stub Aurora, no live model.
"""
import builtins
import contextlib
import io
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.main import _restore_cli_history, run_cli
from PRIMORA.prime.context.manager import ContextManager

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def _db_path(name):
    return os.path.join(_TEST_TMPDIR.name, name)


class _Result:
    def __init__(self, success=True, content="mock reply", error=None):
        self.success = success
        self.content = content
        self.error = error
        self.model = "stub"


class RecordingCliAurora:
    """CLI stub; exposes a real ContextManager when given one."""

    def __init__(self, manager=None, success=True, content="mock reply"):
        if manager is not None:
            self._prime = SimpleNamespace(context=manager)
        self._success = success
        self._content = content
        self.payloads = []
        self.sessions = []

    def respond(self, messages, session_id="default"):
        self.payloads.append([dict(m) for m in messages])
        self.sessions.append(session_id)
        return _Result(self._success, self._content,
                       None if self._success else "boom")


def _run_cli(aurora, inputs):
    """Drive run_cli with scripted inputs; returns captured stdout."""
    script = iter(list(inputs) + ["exit"])
    real_input = builtins.input
    real_env = os.environ.get("PRIMORA_SESSION_ID")
    os.environ["PRIMORA_SESSION_ID"] = "default"
    builtins.input = lambda prompt="": next(script)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            run_cli(aurora)
    finally:
        builtins.input = real_input
        if real_env is None:
            os.environ.pop("PRIMORA_SESSION_ID", None)
        else:
            os.environ["PRIMORA_SESSION_ID"] = real_env
    return buf.getvalue()


# 1. within a session the whole thread travels with every turn -------------
def test_cli_accumulates_history_within_session():
    mgr = ContextManager(db_path=_db_path("accum.db"))
    aurora = RecordingCliAurora(mgr)
    _run_cli(aurora, ["first question", "second question"])
    assert len(aurora.payloads) == 2
    # Fresh store: no restored rows, so payload stays pure role/content.
    assert aurora.payloads[0] == [{"role": "user", "content": "first question"}]
    assert aurora.payloads[1] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "mock reply"},
        {"role": "user", "content": "second question"},
    ]


# 2. restart: stored thread restored before the first turn -----------------
def test_cli_restores_transcript_after_restart():
    seed = ContextManager(db_path=_db_path("cli_restart.db"))
    seed.reconcile([
        {"role": "user", "content": "What is my favorite color?"},
        {"role": "assistant", "content": "I don't have that recorded."},
        {"role": "user", "content": "It's yellow."},
    ], "default")

    mgr = ContextManager(db_path=_db_path("cli_restart.db"))
    aurora = RecordingCliAurora(mgr)
    out = _run_cli(aurora, ["remind me what I said"])
    assert "(restored 3 earlier messages)" in out
    first = aurora.payloads[0]
    assert [(m["role"], m["content"]) for m in first] == [
        ("user", "What is my favorite color?"),
        ("assistant", "I don't have that recorded."),
        ("user", "It's yellow."),
        ("user", "remind me what I said"),
    ]
    # Restored rows keep their stored sequence; the new turn goes past it.
    assert [m.get("seq") for m in first] == [0, 1, 2, 3]
    assert aurora.sessions == ["default"]


def test_cli_failure_turn_keeps_thread_without_reply():
    mgr = ContextManager(db_path=_db_path("cli_fail.db"))
    aurora = RecordingCliAurora(mgr, success=False, content="")
    _run_cli(aurora, ["hello", "anyone there?"])
    assert len(aurora.payloads) == 2
    # The user turn stays part of the thread even when the model failed.
    assert aurora.payloads[1] == [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "anyone there?"},
    ]


# 3. legacy stub (no context loader) behaves exactly as before -------------
def test_cli_stub_without_context_unchanged():
    aurora = RecordingCliAurora(manager=None)
    out = _run_cli(aurora, ["just me"])
    assert "(restored" not in out
    assert aurora.payloads == [[{"role": "user", "content": "just me"}]]


def test_restore_cli_history_contract():
    bare = RecordingCliAurora(manager=None)
    assert _restore_cli_history(bare, "default") == ([], None)

    seed = ContextManager(db_path=_db_path("contract.db"))
    seed.reconcile([{"role": "user", "content": "stored turn"}], "default")
    history, next_seq = _restore_cli_history(
        RecordingCliAurora(seed), "default")
    assert [(h["role"], h["content"], h["seq"]) for h in history] == [
        ("user", "stored turn", 0)]
    assert next_seq == 1

    empty = ContextManager(db_path=_db_path("contract_empty.db"))
    assert _restore_cli_history(
        RecordingCliAurora(empty), "default") == ([], None)


if __name__ == "__main__":
    test_cli_accumulates_history_within_session()
    test_cli_restores_transcript_after_restart()
    test_cli_failure_turn_keeps_thread_without_reply()
    test_cli_stub_without_context_unchanged()
    test_restore_cli_history_contract()
    print("All CLI continuity tests passed!")
