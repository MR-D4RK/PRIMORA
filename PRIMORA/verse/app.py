from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import time

from PySide6.QtCore import Qt, QUrl, Slot, QEvent, QObject, Signal, QThread
from PySide6.QtQml import QQmlApplicationEngine

from .runtime import VerseRuntime
from .state import PresentationMode, VerseLifecycleState
from .worker import WarmupWorker


@dataclass(frozen=True)
class VerseMessage:
    """A single conversation message for QML display."""
    role: str          # "user" or "assistant"
    content: str
    timestamp: float


@dataclass(frozen=True)
class VerseTaskState:
    """Task execution state for QML display."""
    status: str        # "idle", "running", "completed", "failed"
    description: str   # Human-readable description
    result: str = ""   # Result or error message


class MessageWorker(QThread):
    """Runs Aurora.respond() off the Qt main thread."""
    finished_with = Signal(object)  # VerseMessage or error string

    def __init__(self, aurora, messages: list[dict[str, str]], session_id: str):
        super().__init__()
        self._aurora = aurora
        self._messages = messages
        self._session_id = session_id

    def run(self):
        try:
            result = self._aurora.respond(self._messages, self._session_id)
            if result.success:
                self.finished_with.emit(VerseMessage(
                    role="assistant",
                    content=result.content,
                    timestamp=0.0  # Will be set by caller
                ))
            else:
                self.finished_with.emit(f"Error: {result.error}")
        except Exception as e:
            self.finished_with.emit(f"Error: {e}")


class _CloseEventFilter(QObject):
    """Event filter to catch Escape and Close events on the window."""
    def __init__(self, verse_app):
        super().__init__()
        self._verse_app = verse_app

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape:
                self._verse_app.shutdown()
                return True
        elif event.type() == QEvent.Type.Close:
            if self._verse_app.state in (
                VerseLifecycleState.CLOSING, VerseLifecycleState.CLOSED,
            ):
                # Re-entry from the window.close() inside shutdown(): let
                # this event through so the window really closes (the
                # teardown is already running; consuming it would leave a
                # never-closed window and the application would not quit).
                return False
            # First close: run our teardown, then own the event.
            self._verse_app.shutdown()
            return True
        return super().eventFilter(watched, event)

QML_DIR = Path(__file__).resolve().parent / "qml"
SHELL_QML = QML_DIR / "Shell.qml"

# Bounded join for a still-running warmup worker during shutdown. The model
# load itself cannot be cancelled; this only bounds how long closing waits
# before releasing the thread as a last resort.
WARMUP_SHUTDOWN_TIMEOUT_MS = 5000


class VerseApplication(QObject):
    """Native shell binding one Aurora instance to one QML scene.

    Receives the already-created Aurora object (never constructs Aurora or
    Prime, never touches model logic). Owns the QML engine and window
    presentation; the QGuiApplication lifecycle stays with main.py.
    """

    # Signals for QML
    messageReceived = Signal(str, str, float)  # role, content, timestamp
    taskStateChanged = Signal(object)     # VerseTaskState
    conversationCleared = Signal()

    def __init__(
        self,
        aurora,
        runtime=None,
        mode: PresentationMode = PresentationMode.PORTAL,
        qml_dir=None,
    ):
        super().__init__()
        if runtime is None:
            runtime = VerseRuntime(aurora)
        elif runtime.aurora is not aurora:
            raise ValueError(
                "VerseApplication runtime must wrap the provided Aurora instance"
            )
        if not isinstance(mode, PresentationMode):
            raise ValueError(f"Invalid presentation mode: {mode!r}")
        self._runtime = runtime
        self._mode = mode
        self._qml_dir = Path(qml_dir) if qml_dir is not None else QML_DIR
        self._engine = None
        self._window = None
        self._worker = None
        self._dismissed = False
        self._conversation: list[VerseMessage] = []
        self._task_state = VerseTaskState(status="idle", description="No active task")
        self._session_id = "default"
        self._message_worker = None

    @property
    def runtime(self) -> VerseRuntime:
        return self._runtime

    @property
    def mode(self) -> PresentationMode:
        return self._runtime.mode

    @property
    def state(self) -> VerseLifecycleState:
        return self._runtime.state

    @property
    def conversation(self) -> list:
        return self._conversation

    @property
    def task_state(self) -> VerseTaskState:
        return self._task_state

    def set_mode(self, mode: PresentationMode) -> None:
        self._runtime.set_mode(mode)
        self._apply_mode()

    def start(self) -> "VerseApplication":
        """Show loading QML immediately, warm Aurora up on a worker thread.

        AWAKENING is entered synchronously; the AWAKE/DEGRADED transition
        happens later on the Qt main thread when the worker reports back.
        """
        self._runtime.transition_to(VerseLifecycleState.AWAKENING)
        self._runtime.set_mode(self._mode)
        self._engine = QQmlApplicationEngine()
        self._engine.rootContext().setContextProperty("verseApp", self)
        self._engine.load(QUrl.fromLocalFile(str(self._qml_dir / "Shell.qml")))
        roots = self._engine.rootObjects()
        if not roots:
            self._runtime.transition_to(VerseLifecycleState.DEGRADED)
            raise RuntimeError(
                f"Verse QML shell failed to load from {self._qml_dir}"
            )
        self._window = roots[0]
        self._window.installEventFilter(_CloseEventFilter(self))
        self._apply_mode()
        self._set_qml_state("awakening", "")
        worker = WarmupWorker(self._runtime.aurora)
        worker.finished_with.connect(self._on_warmup_finished)
        self._worker = worker
        worker.start()
        return self

    def _on_warmup_finished(self, result) -> None:
        worker, self._worker = self._worker, None
        if self._dismissed or self._runtime.state is VerseLifecycleState.CLOSED:
            self._release_worker(worker)
            return
        if result is None:
            self._set_qml_state("awake", "")
            self._runtime.transition_to(VerseLifecycleState.AWAKE)
        else:
            self._set_qml_state("degraded", str(result))
            self._runtime.transition_to(VerseLifecycleState.DEGRADED)
        self._release_worker(worker)

    @staticmethod
    def _release_worker(worker) -> None:
        if worker is None:
            return
        try:
            worker.quit()
            worker.wait()
        except RuntimeError:
            pass

    def _set_qml_state(self, state: str, error: str) -> None:
        if self._window is None:
            return
        try:
            self._window.setProperty("verseState", state)
            self._window.setProperty("verseError", error)
        except (AttributeError, RuntimeError):
            pass

    @Slot()
    def shutdown(self) -> None:
        """CLOSING -> close window -> CLOSED.

        Re-entrancy-safe: ``window.close()`` inside this method re-enters
        the close-event filter, which calls shutdown() again. A second
        entry while CLOSING (or after CLOSED) is a no-op - the outer call
        owns the teardown, and a repeated CLOSING transition would raise.
        """
        if self._runtime.state in (
            VerseLifecycleState.CLOSING, VerseLifecycleState.CLOSED,
        ):
            return
        self._dismissed = True
        self._runtime.transition_to(VerseLifecycleState.CLOSING)
        window, self._window = self._window, None
        if window is not None:
            try:
                window.close()
            except (AttributeError, RuntimeError):
                pass
        worker, self._worker = self._worker, None
        if worker is not None:
            try:
                if worker.isRunning():
                    worker.quit()
                    if not worker.wait(WARMUP_SHUTDOWN_TIMEOUT_MS):
                        worker.terminate()
                        worker.wait()
            except RuntimeError:
                pass
        # Clean up message worker
        msg_worker, self._message_worker = self._message_worker, None
        if msg_worker is not None:
            try:
                if msg_worker.isRunning():
                    msg_worker.quit()
                    msg_worker.wait(1000)
            except RuntimeError:
                pass
        engine, self._engine = self._engine, None
        if engine is not None:
            try:
                engine.deleteLater()
            except RuntimeError:
                pass
        self._runtime.transition_to(VerseLifecycleState.CLOSED)

    @Slot(str)
    def send_message(self, text: str) -> None:
        """Send a user message to Aurora and display the response."""
        text = text.strip()
        if not text:
            return
        if self._runtime.state not in (VerseLifecycleState.AWAKE, VerseLifecycleState.ASLEEP):
            # Not ready to process messages
            ts = time.time()
            self._conversation.append(VerseMessage(
                role="assistant",
                content="Aurora is not ready. Please wait.",
                timestamp=ts
            ))
            self.messageReceived.emit("assistant", "Aurora is not ready. Please wait.", ts)
            return

        # Add user message to conversation immediately
        ts = time.time()
        user_msg = VerseMessage(role="user", content=text, timestamp=ts)
        self._conversation.append(user_msg)
        self.messageReceived.emit("user", text, ts)

        # Build message history for Aurora (include system context via Aurora's internal handling)
        messages = [{"role": m.role, "content": m.content} for m in self._conversation]

        # Start worker to get Aurora's response
        worker = MessageWorker(self._runtime.aurora, messages, self._session_id)
        worker.finished_with.connect(self._on_message_finished)
        self._message_worker = worker
        worker.start()

    def _on_message_finished(self, result) -> None:
        worker, self._message_worker = self._message_worker, None
        if self._dismissed or self._runtime.state is VerseLifecycleState.CLOSED:
            return
        if isinstance(result, VerseMessage):
            # Set proper timestamp
            result = VerseMessage(
                role=result.role,
                content=result.content,
                timestamp=time.time()
            )
            self._conversation.append(result)
            # Emit as separate arguments for QML compatibility
            self.messageReceived.emit(result.role, result.content, result.timestamp)
        else:
            # Error string
            error_msg = VerseMessage(
                role="assistant",
                content=str(result),
                timestamp=time.time()
            )
            self._conversation.append(error_msg)
            self.messageReceived.emit("assistant", str(result), error_msg.timestamp)

    @Slot()
    def sleep(self) -> None:
        """Transition to sleep state (ASLEEP)."""
        if self._runtime.state is VerseLifecycleState.AWAKE:
            self._runtime.transition_to(VerseLifecycleState.ASLEEP)
            self._set_universe_active(False)

    @Slot()
    def wake(self) -> None:
        """Transition from sleep to awake state."""
        if self._runtime.state is VerseLifecycleState.ASLEEP:
            self._runtime.transition_to(VerseLifecycleState.AWAKE)
            self._set_universe_active(True)

    def _set_universe_active(self, active: bool) -> None:
        """Update the Universe component's active property."""
        if self._window is None:
            return
        try:
            # Find the Universe item in the QML hierarchy
            universe = self._window.findChild(QObject, "universeRoot")
            if universe is not None:
                universe.setProperty("active", active)
        except (AttributeError, RuntimeError):
            pass

    def _apply_mode(self) -> None:
        if self._window is None:
            return
        if self._runtime.mode is PresentationMode.IMMERSIVE:
            self._window.setFlags(Qt.FramelessWindowHint)
            self._window.showFullScreen()
        else:
            self._window.setFlags(Qt.Window)
            self._window.showNormal()
        try:
            self._window.setProperty("verseMode", self._runtime.mode.value)
        except (AttributeError, RuntimeError):
            pass
