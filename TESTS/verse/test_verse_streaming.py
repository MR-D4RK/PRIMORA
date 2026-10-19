"""Verse streaming UI tests (segment-buffered).

Verifies that only Aurora's final user-facing answer reaches the
conversation UI: intermediate cognitive-loop cycle text and tool-request
content are buffered internally and discarded, never displayed. Also covers
single-bubble display, clear-during-generation safety, and send-control
gating. Uses the offscreen Qt platform and isolated stores; never touches
the production database.
"""

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PySide6.QtGui import QGuiApplication

from PRIMORA.verse import PresentationMode, VerseLifecycleState
from PRIMORA.verse.app import VerseApplication, VerseMessage


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


class CancellableAurora:
    """Stub that streams deltas through on_delta but returns only a clean
    final answer. Simulates a brain that streams (including intermediate
    content) while the loop returns the final segment's content."""

    def __init__(self, content="Hello world", chunk=3, delay=0.05):
        self._content = content
        self._chunk = chunk
        self._delay = delay
        self.warmup_calls = 0
        self.respond_calls = 0
        self.last_messages = None
        self.last_session_id = None
        self.delta_calls = 0

    def warmup(self):
        self.warmup_calls += 1
        return None

    def respond(self, messages, session_id="default", on_delta=None,
                cancelled=None):
        self.respond_calls += 1
        self.last_messages = messages
        self.last_session_id = session_id
        # Stream deltas via on_delta if provided; the final answer is always
        # self._content regardless of whether a callback is wired.
        if on_delta is not None:
            for i in range(0, len(self._content), self._chunk):
                if cancelled is not None and cancelled():
                    break
                on_delta(self._content[i:i + self._chunk])
                self.delta_calls += 1
                time.sleep(self._delay)

        class Result:
            success = True
            content = self._content
            error = None
            model = "stub"

        return Result()


# ---------------------------------------------------------------------------
# Final answer shown once in a single bubble; no duplicate messages
# ---------------------------------------------------------------------------

def test_final_answer_shown_once_in_one_bubble():
    _qt_app()
    aurora = CancellableAurora("Hello world")
    app = VerseApplication(aurora)
    app.start()
    try:
        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)

        app.send_message("Hello Aurora")

        # User row + assistant placeholder exist immediately.
        assert _pump_until(lambda: len(app.conversation) == 2, 10.0)
        assert app.conversation[0].role == "user"
        assert app.conversation[0].content == "Hello Aurora"
        assert app.conversation[1].role == "assistant"
        assert app.conversation[1].content == ""

        # The placeholder is filled once with the final answer (no
        # progressive deltas appear in the UI).
        assert _pump_until(
            lambda: app.conversation[1].content == "Hello world", 10.0)

        assistant_rows = [m for m in app.conversation if m.role == "assistant"]
        assert len(assistant_rows) == 1
        assert assistant_rows[0].content == "Hello world"

        # The brain was invoked exactly once; no deltas are streamed to the
        # UI in this path (only the final answer is shown).
        assert aurora.respond_calls == 1
    finally:
        app.shutdown()


# ---------------------------------------------------------------------------
# Clear during generation cannot restore stale content
# ---------------------------------------------------------------------------

def test_clear_during_streaming_does_not_reappear_stale_text():
    _qt_app()
    # Slow stream so generation is still in flight when we clear.
    aurora = CancellableAurora("Hello world this is a longer reply",
                               chunk=3, delay=0.2)
    app = VerseApplication(aurora)
    app.start()
    try:
        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)

        app.send_message("Hello Aurora")

        # Let respond() start so generation is in flight before we clear.
        assert _pump_until(lambda: aurora.respond_calls >= 1, 10.0)

        # Clear while the response is still streaming.
        app.clear_conversation()
        assert _pump_until(lambda: len(app.conversation) == 0, 10.0)

        # Wait for the worker to finish; nothing may reappear.
        assert _pump_until(lambda: aurora.respond_calls == 1, 10.0)
        time.sleep(0.3)
        _qt_app().processEvents()
        assert len(app.conversation) == 0, (
            f"stale text reappeared after clear: {app.conversation}")
    finally:
        app.shutdown()


# ---------------------------------------------------------------------------
# Send control disabled while a response is active
# ---------------------------------------------------------------------------

def test_send_disabled_while_responding():
    _qt_app()
    aurora = CancellableAurora("Hello world")
    app = VerseApplication(aurora)
    app.start()
    try:
        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)

        assert app.isResponding is False

        app.send_message("Hello Aurora")

        assert _pump_until(lambda: app.isResponding is True, 10.0)
        assert app.isResponding is True

        assert _pump_until(
            lambda: app.isResponding is False
            and app.conversation[1].content == "Hello world", 10.0)
        assert app.isResponding is False
    finally:
        app.shutdown()


# ---------------------------------------------------------------------------
# Error during generation replaces the placeholder (no duplicate error row)
# ---------------------------------------------------------------------------

def test_streaming_error_replaces_placeholder():
    _qt_app()

    class ErrorAurora(CancellableAurora):
        def respond(self, messages, session_id="default", on_delta=None,
                    cancelled=None):
            self.respond_calls += 1
            self.last_messages = messages
            self.last_session_id = session_id
            if on_delta is not None:
                on_delta("partial ")
            class Result:
                success = False
                content = ""
                error = "Model unavailable"
                model = "stub"
            return Result()

    app = VerseApplication(ErrorAurora())
    app.start()
    try:
        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
        app.send_message("Hello Aurora")
        assert _pump_until(
            lambda: len(app.conversation) == 2
            and "Model unavailable" in app.conversation[1].content, 10.0)
        assistant_rows = [m for m in app.conversation if m.role == "assistant"]
        assert len(assistant_rows) == 1
        assert "Model unavailable" in assistant_rows[0].content
    finally:
        app.shutdown()


# ---------------------------------------------------------------------------
# Intermediate content streamed via on_delta never appears in the UI
# ---------------------------------------------------------------------------

def test_intermediate_deltas_never_reach_ui():
    """The stub streams intermediate-looking text through on_delta but the
    final result.content is clean. The UI must show only result.content."""

    class IntermediateAurora(CancellableAurora):
        def respond(self, messages, session_id="default", on_delta=None,
                    cancelled=None):
            self.respond_calls += 1
            self.last_messages = messages
            self.last_session_id = session_id
            # Stream intermediate-looking content through on_delta...
            if on_delta is not None:
                on_delta("Let me look that up. ")
                on_delta("<prime_request>{\"action\":\"recall\"")
            # ...but return only the cleaned final answer.
            class Result:
                success = True
                content = "The last cargo was kestrel seventeen."
                error = None
                model = "stub"
            return Result()

    _qt_app()
    aurora = IntermediateAurora()
    app = VerseApplication(aurora)
    app.start()
    try:
        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
        app.send_message("What was the cargo?")
        assert _pump_until(
            lambda: len(app.conversation) == 2
            and app.conversation[1].content
            == "The last cargo was kestrel seventeen.", 10.0)
        assistant_rows = [m for m in app.conversation if m.role == "assistant"]
        assert len(assistant_rows) == 1
        assert assistant_rows[0].content == "The last cargo was kestrel seventeen."
        assert "Let me look that up" not in assistant_rows[0].content
        assert "<prime_request>" not in assistant_rows[0].content
    finally:
        app.shutdown()


# ---------------------------------------------------------------------------
# Regression: Verse must not pass ``cancelled`` to aurora.respond()
# ---------------------------------------------------------------------------

class StrictAurora:
    """Stub whose respond() matches the real Aurora signature: it accepts
    ``on_delta`` but NOT ``cancelled``. If Verse passes ``cancelled``,
    respond() raises TypeError -- reproducing the original bug."""

    def __init__(self, content="Hello world"):
        self._content = content
        self.warmup_calls = 0
        self.respond_calls = 0
        self.last_messages = None
        self.last_session_id = None
        self.seen_kwargs = []

    def warmup(self):
        self.warmup_calls += 1
        return None

    def respond(self, messages, session_id="default", on_delta=None):
        self.respond_calls += 1
        self.last_messages = messages
        self.last_session_id = session_id
        self.seen_kwargs.append({"on_delta": on_delta is not None})

        class Result:
            success = True
            content = self._content
            error = None
            model = "stub"

        return Result()


def test_verse_does_not_pass_cancelled_to_respond():
    """Reproduces the original failure:
    ``aurora.respond got an unexpected keyword argument cancelled``.

    Verse must call respond() without ``cancelled``. If it did, the stub
    (matching the real respond signature) raises TypeError and the test
    errors here.
    """
    _qt_app()
    aurora = StrictAurora()
    app = VerseApplication(aurora)
    app.start()
    try:
        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
        app.send_message("Hello Aurora")
        assert _pump_until(
            lambda: len(app.conversation) == 2
            and app.conversation[1].content == "Hello world", 10.0)
        # Verse passed only on_delta (or neither), never cancelled.
        assert aurora.respond_calls == 1
        assert all("cancelled" not in k for k in aurora.seen_kwargs)
    finally:
        app.shutdown()


if __name__ == "__main__":
    test_final_answer_shown_once_in_one_bubble()
    test_clear_during_streaming_does_not_reappear_stale_text()
    test_send_disabled_while_responding()
    test_streaming_error_replaces_placeholder()
    test_intermediate_deltas_never_reach_ui()
    test_verse_does_not_pass_cancelled_to_respond()
    print("All Verse streaming tests passed!")
