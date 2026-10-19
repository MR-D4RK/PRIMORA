"""Clear Conversation exposed in the Verse QML surface.

Conventions follow the repo: offscreen Qt, real QML engine via
``app.start()``, a stub Aurora exposing a real ContextManager over a
temp SQLite file, no live model. These tests click the actual QML
control (or drive the Python slot directly) and assert that the QML
model, the Python conversation, and the store stay in sync while
persistent memory survives the clear.
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from types import SimpleNamespace

from PySide6.QtCore import QMetaObject, QObject, Qt
from PySide6.QtGui import QGuiApplication

from PRIMORA.verse import VerseLifecycleState
from PRIMORA.verse.app import VerseApplication
from PRIMORA.prime.context.manager import ContextManager
from PRIMORA.prime.context.store import ContextStore, CONVERSATION_CLEARED

_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def _db_path(name):
    return os.path.join(_TEST_TMPDIR.name, name)


class RecordingAurora:
    """Aurora stub exposing a real ContextManager as ``_prime.context``."""

    def __init__(self, manager=None):
        self._prime = SimpleNamespace(context=manager)
        self.respond_calls = 0
        self.last_messages = None

    def warmup(self):
        return None

    def respond(self, messages, session_id="default", on_delta=None, cancelled=None):
        self.respond_calls += 1
        self.last_messages = [dict(m) for m in messages]

        class Result:
            success = True
            content = "stub reply"
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


def _click(obj):
    """Click a QML Control without assuming its concrete Python type."""
    try:
        if QMetaObject.invokeMethod(obj, "click", Qt.DirectConnection):
            return True
    except Exception:
        pass
    fn = getattr(obj, "click", None)
    if callable(fn):
        fn()
        return True
    return False


def _start_app(db):
    mgr = ContextManager(db_path=db)
    aurora = RecordingAurora(mgr)
    app = VerseApplication(aurora)
    app.start()
    assert _pump_until(
        lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
    return app, mgr, aurora


# 1. The visible QML control exists and its click clears both buffers -----
def test_clear_button_visible_and_click_clears_thread_and_marker():
    _qt_app()
    db = _db_path("clear_ui.db")
    seed = ContextManager(db_path=db)
    seed.reconcile([
        {"role": "user", "content": "old turn"},
        {"role": "assistant", "content": "old reply"},
    ], "default")
    rows_before = ContextStore(db).count("default")

    app, mgr, aurora = _start_app(db)
    try:
        # Restored rows reached the QML model through messageReceived.
        list_view = app._window.findChild(QObject, "verseMessageList")
        assert list_view is not None
        assert _pump_until(
            lambda: list_view.property("count") == 2, 10.0)
        assert len(app.conversation) == 2

        button = app._window.findChild(QObject, "clearConversationButton")
        assert button is not None, "Clear Conversation control missing"
        assert button.property("text") == "Clear Conversation"
        assert button.property("enabled") is True
        assert button.property("visible") is True

        assert _click(button), "clear button did not accept a click"
        assert _pump_until(
            lambda: len(app.conversation) == 0
            and list_view.property("count") == 0,
            10.0,
        )

        # Marker written; raw rows untouched; reload floor engaged.
        store = ContextStore(db)
        assert store.count("default") == rows_before + 1
        assert any(
            r.role == "system" and r.content == CONVERSATION_CLEARED
            for r in store.fetch_range("default")
        )
        assert mgr.load_transcript("default") == []
    finally:
        app.shutdown()


# 2. Python-initiated clear reaches the QML model via conversationCleared --
def test_python_initiated_clear_syncs_qml_model():
    _qt_app()
    db = _db_path("clear_signal.db")
    app, mgr, aurora = _start_app(db)
    try:
        list_view = app._window.findChild(QObject, "verseMessageList")
        assert list_view is not None

        assert _pump_until(
            lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
        app.send_message("hello")
        assert _pump_until(lambda: aurora.respond_calls == 1, 10.0)
        assert _pump_until(lambda: len(app.conversation) == 2, 10.0)
        assert _pump_until(lambda: list_view.property("count") == 2, 10.0)

        # Drive the Python slot directly: the signal must clear the model.
        app.clear_conversation()
        assert app.conversation == []
        assert _pump_until(
            lambda: list_view.property("count") == 0, 10.0)
    finally:
        app.shutdown()


# 3. Clearing removes the transcript but never persistent memory -----------
def test_clear_preserves_persistent_memory():
    _qt_app()
    db = _db_path("clear_memory.db")
    seed = ContextManager(db_path=db)
    seed.reconcile([
        {"role": "user", "content": "What is my favorite color?"},
        {"role": "user", "content": "It's yellow."},
    ], "default")
    states = seed._get_store().fetch_states(
        concept="favorite color", status="current")
    assert any(s.value == "yellow" for s in states), "seed state missing"
    rows_before = ContextStore(db).count("default")

    app, mgr, aurora = _start_app(db)
    try:
        assert len(app.conversation) == 2
        app.clear_conversation()
        assert app.conversation == []
        # Transcript gone, raw rows kept, marker added.
        assert mgr.load_transcript("default") == []
        assert ContextStore(db).count("default") == rows_before + 1
        # The extracted fact survives the clear untouched.
        surviving = mgr._get_store().fetch_states(
            concept="favorite color", status="current")
        assert any(s.value == "yellow" for s in surviving)
    finally:
        app.shutdown()


if __name__ == "__main__":
    test_clear_button_visible_and_click_clears_thread_and_marker()
    test_python_initiated_clear_syncs_qml_model()
    test_clear_preserves_persistent_memory()
    print("All verse clear UI tests passed!")
