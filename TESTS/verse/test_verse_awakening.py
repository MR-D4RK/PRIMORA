import os
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PySide6.QtGui import QGuiApplication

from PRIMORA.verse import PresentationMode, VerseLifecycleState
from PRIMORA.verse.app import VerseApplication


class BlockingAurora:
    """Aurora stub with controllable warmup for lifecycle tests."""

    def __init__(self, result=None):
        self._event = threading.Event()
        self._result = result
        self.warmup_calls = 0

    def release(self):
        self._event.set()

    def warmup(self):
        self.warmup_calls += 1
        self._event.wait(30)
        return self._result


class InstantAurora:
    def __init__(self, result=None):
        self._result = result
        self.warmup_calls = 0

    def warmup(self):
        self.warmup_calls += 1
        return self._result


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


def test_startup_begins_in_awakening_with_loading_shown():
    _qt_app()
    aurora = BlockingAurora()
    app = VerseApplication(aurora)
    app.start()
    try:
        assert app.state is VerseLifecycleState.AWAKENING
        assert app._window is not None
        assert app._window.property("verseState") == "awakening"
        assert _pump_until(lambda: aurora.warmup_calls == 1, 10.0)
    finally:
        aurora.release()
        app.shutdown()
        _pump_until(lambda: True, 0.5)


def test_successful_warmup_transitions_to_awake():
    _qt_app()
    app = VerseApplication(InstantAurora())
    app.start()
    try:
        assert _pump_until(
            lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
        assert app._window.property("verseState") == "awake"
    finally:
        app.shutdown()
    assert app.state is VerseLifecycleState.CLOSED


def test_failed_warmup_transitions_to_degraded():
    _qt_app()
    app = VerseApplication(InstantAurora(result="weights unavailable"))
    app.start()
    try:
        assert _pump_until(
            lambda: app.state is VerseLifecycleState.DEGRADED, 10.0)
        assert app.state is not VerseLifecycleState.AWAKE
        assert app._window.property("verseState") == "degraded"
        assert (
            app._window.property("verseError") == "weights unavailable"
        )
    finally:
        app.shutdown()
    assert app.state is VerseLifecycleState.CLOSED


def test_close_during_warmup_does_not_hang():
    _qt_app()
    aurora = BlockingAurora()
    app = VerseApplication(aurora)
    app.start()
    assert app.state is VerseLifecycleState.AWAKENING
    started = time.time()
    app.shutdown()
    elapsed = time.time() - started
    assert app.state is VerseLifecycleState.CLOSED
    assert elapsed < 15.0, f"shutdown hung for {elapsed:.1f}s"
    aurora.release()


def test_closed_app_ignores_late_worker_result():
    _qt_app()
    aurora = BlockingAurora()
    app = VerseApplication(aurora)
    app.start()
    app.shutdown()
    assert app.state is VerseLifecycleState.CLOSED
    aurora.release()
    _pump_until(lambda: True, 1.0)
    assert app.state is VerseLifecycleState.CLOSED
    if app._window is not None:
        assert app._window.property("verseState") != "awake"


def test_cli_path_preserved():
    import builtins
    from PRIMORA.main import run_cli

    class StubResult:
        success = True
        content = "mock reply"

    class StubAuroraCli:
        def __init__(self):
            self.calls = 0

        def respond(self, messages, session_id="default"):
            self.calls += 1
            return StubResult()

    aurora = StubAuroraCli()
    inputs = iter(["hi", "exit"])
    real_input = builtins.input
    builtins.input = lambda prompt="": next(inputs)
    try:
        run_cli(aurora)
    finally:
        builtins.input = real_input
    assert aurora.calls == 1


if __name__ == "__main__":
    test_startup_begins_in_awakening_with_loading_shown()
    test_successful_warmup_transitions_to_awake()
    test_failed_warmup_transitions_to_degraded()
    test_close_during_warmup_does_not_hang()
    test_closed_app_ignores_late_worker_result()
    test_cli_path_preserved()
    print("All Verse awakening tests passed!")
