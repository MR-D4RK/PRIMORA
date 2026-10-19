"""Continuity across a Verse (GUI) restart: reload, signals, clear, payload.

Conventions follow the repo: offscreen Qt, stub Aurora exposing a real
ContextManager over a temp SQLite file, no live model.
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from types import SimpleNamespace

from PySide6.QtGui import QGuiApplication

from PRIMORA.verse import PresentationMode, VerseLifecycleState
from PRIMORA.verse.app import VerseApplication
from PRIMORA.prime.context.manager import ContextManager
from PRIMORA.prime.context.store import ContextStore, CONVERSATION_CLEARED

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def _db_path(name):
    return os.path.join(_TEST_TMPDIR.name, name)


class RecordingAurora:
    """Aurora stub exposing a real ContextManager as ``_prime.context``."""

    def __init__(self, manager=None, response_content="Aurora reply"):
        self._prime = SimpleNamespace(context=manager)
        self._response_content = response_content
        self.warmup_calls = 0
        self.respond_calls = 0
        self.last_messages = None
        self.last_session_id = None

    def warmup(self):
        self.warmup_calls += 1
        return None

    def respond(self, messages, session_id="default", on_delta=None, cancelled=None):
        self.respond_calls += 1
        self.last_messages = [dict(m) for m in messages]
        self.last_session_id = session_id

        class Result:
            success = True
            content = self._response_content
            error = None
            model = "stub"

        return Result()


_app = None


def _qt_app():
    global _app
    if _app is None:
        existing = QGuiApplication.instance()
        _app = existing if existing is not None else QGuiApplication([])
    return _app


def _pump_until(condition, timeout_s=10.0):
    app = _qt_app()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    app.processEvents()
    return condition()


def _messages(app):
    return [(m.role, m.content) for m in app.conversation]


# 1. same session: history grows and every turn sees the whole thread ------
def test_same_session_sends_growing_history():
    _qt_app()
    mgr = ContextManager(db_path=_db_path("same_session.db"))
    aurora = RecordingAurora(mgr)
    app = VerseApplication(aurora)
    app.start()
    try:
        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
        app.send_message("Hello Aurora")
        assert _pump_until(lambda: aurora.respond_calls == 1, 10.0)
        assert _pump_until(lambda: len(app.conversation) == 2, 10.0)

        app.send_message("Second turn")
        assert _pump_until(lambda: aurora.respond_calls == 2, 10.0)
        assert [(m["role"], m["content"]) for m in aurora.last_messages] == [
            ("user", "Hello Aurora"),
            ("assistant", "Aurora reply"),
            ("user", "Second turn"),
        ]
    finally:
        app.shutdown()


# 2. restart: transcript restored, re-emitted to the QML model --------------
def test_restart_restores_transcript_and_reemits():
    _qt_app()
    seed = ContextManager(db_path=_db_path("restart.db"))
    seed.reconcile([
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi back"},
    ], "default")

    # "Restart": brand-new manager instance over the same database file.
    mgr = ContextManager(db_path=_db_path("restart.db"))
    aurora = RecordingAurora(mgr)
    app = VerseApplication(aurora)
    seen = []
    app.messageReceived.connect(lambda role, content, ts: seen.append((role, content)))
    app.start()
    try:
        assert _messages(app) == [
            ("user", "hello there"),
            ("assistant", "hi back"),
        ]
        # The QML model is signal-fed: each restored row was re-emitted.
        assert seen == [("user", "hello there"), ("assistant", "hi back")]
        # Restored rows carry their stored sequence.
        assert [m.seq for m in app.conversation] == [0, 1]

        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
        app.send_message("third turn")
        assert _pump_until(lambda: aurora.respond_calls == 1, 10.0)
        assert [(m["role"], m["content"]) for m in aurora.last_messages] == [
            ("user", "hello there"),
            ("assistant", "hi back"),
            ("user", "third turn"),
        ]
        # New in-process messages sequence past the restored ones.
        assert [m.get("seq") for m in aurora.last_messages] == [0, 1, 2]
        assert aurora.last_session_id == "default"
    finally:
        app.shutdown()


# 3. restart: the stored fact reaches the build view -----------------------
def test_restart_build_view_contains_stored_fact():
    _qt_app()
    seed = ContextManager(db_path=_db_path("fact_restart.db"))
    seed.reconcile([
        {"role": "user", "content": "What is my favorite color?"},
        {"role": "user", "content": "It's yellow."},
    ], "default")

    mgr = ContextManager(db_path=_db_path("fact_restart.db"))
    aurora = RecordingAurora(mgr)
    app = VerseApplication(aurora)
    app.start()
    try:
        assert _messages(app)[0][1] == "What is my favorite color?"
        result = app._context().build(
            [{"role": "user", "content": "What is my favorite color?"}],
            "default",
            token_counter=lambda msgs: sum(
                len(m.get("content", "").split()) for m in msgs),
        )
        view = result.view if isinstance(result.view, str) else str(result.view)
        assert "The user's favorite color is yellow." in view
    finally:
        app.shutdown()


# 4. clear: display + thread cleared, marker persists, restart stays empty --
def test_clear_survives_restart_without_deleting_rows():
    _qt_app()
    db = _db_path("clear.db")
    seed = ContextManager(db_path=db)
    seed.reconcile([
        {"role": "user", "content": "old turn"},
        {"role": "assistant", "content": "old reply"},
    ], "default")
    rows_before = ContextStore(db).count("default")

    mgr = ContextManager(db_path=db)
    aurora = RecordingAurora(mgr)
    app = VerseApplication(aurora)
    app.start()
    try:
        assert len(app.conversation) == 2
        app.clear_conversation()
        assert app.conversation == []
        assert mgr.load_transcript("default") == []
        # Raw truth is never deleted; the clear is a marker row.
        store = ContextStore(db)
        assert store.count("default") == rows_before + 1
        assert any(
            r.role == "system" and r.content == CONVERSATION_CLEARED
            for r in store.fetch_range("default")
        )
    finally:
        app.shutdown()

    # Restart after clear: still empty.
    mgr2 = ContextManager(db_path=db)
    aurora2 = RecordingAurora(mgr2)
    app2 = VerseApplication(aurora2)
    app2.start()
    try:
        assert app2.conversation == []
        assert _pump_until(lambda: app2.state is VerseLifecycleState.AWAKE, 10.0)
        app2.send_message("fresh")
        assert _pump_until(lambda: aurora2.respond_calls == 1, 10.0)
        # Post-clear turns keep sequencing past the marker, not over it.
        assert [(m["role"], m["content"]) for m in aurora2.last_messages] == [
            ("user", "fresh")]
        assert aurora2.last_messages[0]["seq"] == rows_before + 1
    finally:
        app2.shutdown()


# 5. fresh session payload shape stays exactly as before -------------------
def test_fresh_session_payload_has_only_role_and_content():
    _qt_app()
    mgr = ContextManager(db_path=_db_path("fresh_shape.db"))
    aurora = RecordingAurora(mgr)
    app = VerseApplication(aurora)
    app.start()
    try:
        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
        app.send_message("Hello")
        assert _pump_until(lambda: aurora.respond_calls == 1, 10.0)
        assert aurora.last_messages == [
            {"role": "user", "content": "Hello"}]
    finally:
        app.shutdown()


if __name__ == "__main__":
    test_same_session_sends_growing_history()
    test_restart_restores_transcript_and_reemits()
    test_restart_build_view_contains_stored_fact()
    test_clear_survives_restart_without_deleting_rows()
    test_fresh_session_payload_has_only_role_and_content()
    print("All verse continuity tests passed!")
