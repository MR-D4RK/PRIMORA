from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import time

from PySide6.QtCore import Qt, QUrl, Slot, QEvent, QObject, Signal, QThread
from PySide6.QtQml import QQmlApplicationEngine

from PRIMORA.prime.context.store import CONVERSATION_CLEARED

from .runtime import VerseRuntime
from .state import PresentationMode, VerseLifecycleState
from .worker import WarmupWorker


@dataclass(frozen=True)
class VerseMessage:
    """A single conversation message for QML display.

    ``seq`` is the persisted sequence when the row was restored from the
    store and None for messages created in this process (they sequence
    past the restored ones when re-persisted).
    """
    role: str          # "user" or "assistant"
    content: str
    timestamp: float
    seq: Optional[int] = None


@dataclass(frozen=True)
class VerseTaskState:
    """Task execution state for QML display."""
    status: str        # "idle", "running", "completed", "failed"
    description: str   # Human-readable description
    result: str = ""   # Result or error message


class MessageWorker(QThread):
    """Runs Aurora.respond() off the Qt main thread.

    The cognitive loop only decides whether a model segment is a final
    user-facing answer or an internal/intermediate segment *after*
    generation completes, so Verse does not stream deltas into the UI:
    intermediate text and tool-request content are never displayed, and
    only ``result.content`` (the final answer) is shown once in the
    assistant bubble. ``on_delta``/``cancelled`` are intentionally not
    forwarded to ``Aurora.respond()``; the buffering would be unused and
    ``cancelled`` is not accepted by the current ``respond`` signature.
    """
    finished_with = Signal(object)  # VerseMessage or error string

    def __init__(
        self,
        aurora,
        messages: list[dict[str, str]],
        session_id: str,
        message_index: int,
        epoch: int,
    ):
        super().__init__()
        self._aurora = aurora
        self._messages = messages
        self._session_id = session_id
        self._message_index = message_index
        self._epoch = epoch

    def run(self):
        try:
            result = self._aurora.respond(self._messages, self._session_id)
            if result.success:
                self.finished_with.emit(VerseMessage(
                    role="assistant",
                    content=result.content,
                    timestamp=0.0,  # Will be set by caller
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
    messageUpdated = Signal(int, str, float)   # index, content, timestamp
    taskStateChanged = Signal(object)     # VerseTaskState
    respondingChanged = Signal(bool)      # True while a response streams
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
        self._next_seq: Optional[int] = None
        # Streaming / response lifecycle
        self._is_responding = False
        self._response_epoch = 0
        self._streaming_index: Optional[int] = None
        self._streaming_epoch: Optional[int] = None

    def _context(self):
        """The ContextManager behind this Aurora, or None (stubs, bare shells)."""
        prime = getattr(self._runtime.aurora, "_prime", None)
        return getattr(prime, "context", None)

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

    @property
    def isResponding(self) -> bool:
        """True while a response is being streamed to the UI."""
        return self._is_responding

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
        # Restore the persisted transcript before any interaction: the QML
        # model is signal-fed, so the rows are re-emitted now that the
        # engine is loaded.
        self._restore_conversation()
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
        # Clean up message worker. respond() cannot be cancelled mid-generation
        # in this path, so just let it finish (bounded wait) and drop any
        # stale result via the epoch guard.
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

    @Slot()
    def clear_conversation(self) -> None:
        """Clear the visible thread and this process's in-memory history.

        The clear persists as a store marker row, so a restart stays empty
        without deleting any raw message; memory states are untouched.
        """
        ctx = self._context()
        if ctx is not None:
            try:
                seq = int(ctx.next_seq(self._session_id))
                ctx.reconcile(
                    [{"role": "system", "content": CONVERSATION_CLEARED,
                      "seq": seq}],
                    self._session_id,
                )
                self._next_seq = seq + 1
            except Exception:
                pass
        # Invalidate any in-flight streaming response so stale chunks cannot
        # reappear after the conversation is cleared.
        self._response_epoch += 1
        self._streaming_index = None
        self._streaming_epoch = None
        self._conversation.clear()
        # The in-flight message worker's result will be dropped by the epoch
        # guard in _on_message_finished, so no stale text can reappear.
        # _on_message_finished resets _is_responding when it eventually runs.
        self.conversationCleared.emit()

    def _restore_conversation(self) -> None:
        """Reload this session's transcript so a restart keeps continuity."""
        ctx = self._context()
        if ctx is None:
            return
        try:
            rows = list(ctx.load_transcript(self._session_id) or [])
            next_seq = int(ctx.next_seq(self._session_id))
        except Exception:
            return
        if next_seq > 0:
            # New in-process messages sequence past everything stored, so
            # crammed legacy rows can never be collided with.
            self._next_seq = next_seq
        for row in rows:
            ts = row.get("timestamp")
            if not isinstance(ts, (int, float)) or isinstance(ts, bool):
                ts = 0.0
            raw_seq = row.get("seq")
            msg = VerseMessage(
                role=str(row.get("role", "user")),
                content=str(row.get("content", "")),
                timestamp=float(ts),
                seq=(raw_seq
                     if isinstance(raw_seq, int) and not isinstance(raw_seq, bool)
                     else None),
            )
            self._conversation.append(msg)
            self.messageReceived.emit(msg.role, msg.content, msg.timestamp)

    def _append_message(
        self, role: str, content: str, timestamp: float
    ) -> VerseMessage:
        """Append one in-process message, sequencing it past restored rows."""
        seq = None
        if self._next_seq is not None:
            seq = self._next_seq
            self._next_seq += 1
        msg = VerseMessage(role=role, content=content,
                           timestamp=timestamp, seq=seq)
        self._conversation.append(msg)
        self.messageReceived.emit(role, content, timestamp)
        return msg

    def _messages_payload(self) -> list[dict]:
        """History for Aurora; a restored row's sequence travels with it."""
        messages: list[dict] = []
        for m in self._conversation:
            entry: dict = {"role": m.role, "content": m.content}
            if isinstance(m.seq, int) and not isinstance(m.seq, bool):
                entry["seq"] = m.seq
            messages.append(entry)
        return messages

    @Slot(str)
    def send_message(self, text: str) -> None:
        """Send a user message to Aurora and display the response.

        The assistant answer is shown once the cognitive loop has produced
        its final segment: intermediate cycle text and tool-request content
        never reach the UI. Sending is disabled while a response is active.
        """
        text = text.strip()
        if not text:
            return
        if self._is_responding:
            # A response is already active: refuse to queue another turn.
            return
        if self._runtime.state not in (VerseLifecycleState.AWAKE, VerseLifecycleState.ASLEEP):
            # Not ready to process messages
            self._append_message(
                "assistant", "Aurora is not ready. Please wait.", time.time()
            )
            return

        # Bump the epoch so any in-flight response is treated as stale.
        self._response_epoch += 1
        epoch = self._response_epoch

        # Add user message to conversation immediately
        self._append_message("user", text, time.time())

        # Payload for Aurora: restored history plus this turn's user message.
        # The assistant placeholder below is UI-only and must not be sent to
        # the model as an empty assistant turn.
        messages = self._messages_payload()

        # Create a single assistant placeholder bubble at response start.
        self._append_message("assistant", "", time.time())
        self._streaming_index = len(self._conversation) - 1
        self._streaming_epoch = epoch

        self._is_responding = True
        self.respondingChanged.emit(True)

        # Start worker to get Aurora's response. Intermediate cycle text and
        # tool-request content are never displayed; only the final answer is
        # shown once in the assistant bubble.
        worker = MessageWorker(
            self._runtime.aurora,
            messages,
            self._session_id,
            self._streaming_index,
            epoch,
        )
        worker.finished_with.connect(self._on_message_finished)
        self._message_worker = worker
        worker.start()

    def _message_updated(self, index: int, content: str, timestamp: float) -> None:
        """Update the assistant bubble at ``index`` in place."""
        if 0 <= index < len(self._conversation):
            old = self._conversation[index]
            self._conversation[index] = VerseMessage(
                role=old.role, content=content, timestamp=timestamp, seq=old.seq)
        if self._window is None:
            return
        self.messageUpdated.emit(index, content, timestamp)

    def _on_message_finished(self, result) -> None:
        worker = self._message_worker
        self._message_worker = None
        self._is_responding = False
        self.respondingChanged.emit(False)
        if worker is None:
            return
        worker_epoch = getattr(worker, "_epoch", None)
        if self._dismissed or self._runtime.state is VerseLifecycleState.CLOSED:
            return
        if worker_epoch is not None and worker_epoch != self._response_epoch:
            # Stale: cleared or superseded; do not let stale text reappear.
            return
        if isinstance(result, VerseMessage):
            # Ensure the bubble holds the final cleaned answer only.
            self._message_updated(
                self._streaming_index, result.content, time.time())
        else:
            # Error: replace the placeholder with the error message.
            self._message_updated(
                self._streaming_index, "Error: " + str(result), time.time())
        self._streaming_index = None
        self._streaming_epoch = None

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
