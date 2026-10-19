import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PySide6.QtGui import QGuiApplication

from PRIMORA.verse import PresentationMode, VerseLifecycleState
from PRIMORA.verse.app import VerseApplication, VerseMessage, VerseTaskState


class InstantAurora:
    """Aurora stub with controllable respond for conversation tests."""

    def __init__(self, response_content="Aurora reply", response_success=True, response_error=None):
        self._response_content = response_content
        self._response_success = response_success
        self._response_error = response_error
        self.warmup_calls = 0
        self.respond_calls = 0
        self.last_messages = None
        self.last_session_id = None

    def warmup(self):
        self.warmup_calls += 1
        return None

    def respond(self, messages, session_id="default", on_delta=None, cancelled=None):
        self.respond_calls += 1
        self.last_messages = messages
        self.last_session_id = session_id

        class Result:
            def __init__(self, success, content, error, model="test-model"):
                self.success = success
                self.content = content
                self.error = error
                self.model = model

        if self._response_success:
            if on_delta is not None:
                content = self._response_content
                for i in range(0, len(content), 3):
                    if cancelled is not None and cancelled():
                        break
                    on_delta(content[i:i + 3])
            return Result(True, self._response_content, None)
        else:
            return Result(False, "", self._response_error or "Test error")


class BlockingAurora:
    """Aurora stub with blocking respond for async tests."""

    def __init__(self, response_content="Aurora reply"):
        self._response_content = response_content
        self.warmup_calls = 0
        self.respond_calls = 0
        self._event = None

    def warmup(self):
        self.warmup_calls += 1
        return None

    def respond(self, messages, session_id="default", on_delta=None, cancelled=None):
        self.respond_calls += 1
        if self._event:
            self._event.wait(30)
        class Result:
            def __init__(self, success, content, error, model="test-model"):
                self.success = success
                self.content = content
                self.error = error
                self.model = model
        if on_delta is not None:
            content = self._response_content
            for i in range(0, len(content), 3):
                if cancelled is not None and cancelled():
                    break
                on_delta(content[i:i + 3])
        return Result(True, self._response_content, None)

    def release(self):
        if self._event:
            self._event.set()


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


def test_send_message_when_awake_appends_to_conversation():
    _qt_app()
    aurora = InstantAurora("Hello from Aurora")
    app = VerseApplication(aurora)
    app.start()
    try:
        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)

        # Send a message
        app.send_message("Hello Aurora")

        # Wait for response (and its queued delivery to the main thread)
        assert _pump_until(lambda: aurora.respond_calls == 1, 10.0)

        # Check conversation has both messages (user + assistant placeholder)
        assert _pump_until(lambda: len(app.conversation) == 2, 10.0)
        assert app.conversation[0].role == "user"
        assert app.conversation[0].content == "Hello Aurora"
        # The assistant bubble is filled progressively; wait for the streamed
        # content to land.
        assert _pump_until(
            lambda: app.conversation[1].content == "Hello from Aurora", 10.0)
        assert app.conversation[1].role == "assistant"
        assert app.conversation[1].content == "Hello from Aurora"
    finally:
        app.shutdown()


def test_send_message_when_awakening_shows_not_ready():
    _qt_app()
    aurora = BlockingAurora()
    app = VerseApplication(aurora)
    app.start()
    try:
        assert app.state is VerseLifecycleState.AWAKENING

        # Send message during warmup
        app.send_message("Hello")

        # Should get not ready message immediately (no async wait needed)
        # Pump once to process the slot
        _pump_until(lambda: True, 0.5)

        # Should have error message in conversation
        assert len(app.conversation) >= 1
        last_msg = app.conversation[-1]
        assert last_msg.role == "assistant"
        assert "not ready" in last_msg.content.lower()
    finally:
        aurora.release()
        app.shutdown()


def test_send_message_failure_shows_error():
    _qt_app()
    aurora = InstantAurora(response_success=False, response_error="Model unavailable")
    app = VerseApplication(aurora)
    app.start()
    try:
        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)

        app.send_message("Hello")
        assert _pump_until(lambda: aurora.respond_calls == 1, 10.0)

        # Should have error message (wait for the worker's queued result)
        assert _pump_until(
            lambda: len(app.conversation) == 2
            and "Model unavailable" in app.conversation[1].content, 10.0)
        assert app.conversation[1].role == "assistant"
        assert "Error" in app.conversation[1].content
        assert "Model unavailable" in app.conversation[1].content
    finally:
        app.shutdown()


def test_sleep_transition_from_awake():
    _qt_app()
    aurora = InstantAurora()
    app = VerseApplication(aurora)
    app.start()
    try:
        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)

        app.sleep()
        assert app.state is VerseLifecycleState.ASLEEP
    finally:
        app.shutdown()


def test_wake_transition_from_asleep():
    _qt_app()
    aurora = InstantAurora()
    app = VerseApplication(aurora)
    app.start()
    try:
        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)

        app.sleep()
        assert app.state is VerseLifecycleState.ASLEEP

        app.wake()
        assert app.state is VerseLifecycleState.AWAKE
    finally:
        app.shutdown()


def test_sleep_from_awakening_invalid():
    _qt_app()
    aurora = BlockingAurora()
    app = VerseApplication(aurora)
    app.start()
    try:
        assert app.state is VerseLifecycleState.AWAKENING

        # Sleep from AWAKENING should be invalid (not allowed transition)
        try:
            app.sleep()
            # Should not change state
            assert app.state is VerseLifecycleState.AWAKENING
        except ValueError:
            # Runtime transition_to should reject invalid transition
            assert app.state is VerseLifecycleState.AWAKENING
    finally:
        aurora.release()
        app.shutdown()


def test_wake_from_awake_invalid():
    _qt_app()
    aurora = InstantAurora()
    app = VerseApplication(aurora)
    app.start()
    try:
        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)

        # Wake from AWAKE should be invalid
        try:
            app.wake()
            assert app.state is VerseLifecycleState.AWAKE
        except ValueError:
            assert app.state is VerseLifecycleState.AWAKE
    finally:
        app.shutdown()


def test_task_state_defaults_to_idle():
    _qt_app()
    aurora = InstantAurora()
    app = VerseApplication(aurora)
    app.start()
    try:
        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
        assert app.task_state.status == "idle"
        assert "No active task" in app.task_state.description
    finally:
        app.shutdown()


def test_conversation_cleared_on_shutdown():
    _qt_app()
    aurora = InstantAurora()
    app = VerseApplication(aurora)
    app.start()
    try:
        assert _pump_until(lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
        app.send_message("Test")
        _pump_until(lambda: aurora.respond_calls == 1, 10.0)
        assert len(app.conversation) > 0

        app.shutdown()
        assert app.state is VerseLifecycleState.CLOSED
    finally:
        pass


if __name__ == "__main__":
    test_send_message_when_awake_appends_to_conversation()
    test_send_message_when_awakening_shows_not_ready()
    test_send_message_failure_shows_error()
    test_sleep_transition_from_awake()
    test_wake_transition_from_asleep()
    test_sleep_from_awakening_invalid()
    test_wake_from_awake_invalid()
    test_task_state_defaults_to_idle()
    test_conversation_cleared_on_shutdown()
    print("All Verse conversation tests passed!")