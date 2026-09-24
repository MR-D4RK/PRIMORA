from PySide6.QtCore import QThread, Signal


class WarmupWorker(QThread):
    """Runs the existing ``aurora.warmup()`` off the Qt main thread.

    Deliberately tiny: it calls one existing synchronous method and emits
    the existing result (``None`` = ready, string = error). It never
    constructs Aurora/Prime, never touches QML, windows, or the
    QGuiApplication. Completion is delivered through a Qt-safe signal so
    all UI and state updates happen on the main thread.
    """

    finished_with = Signal(object)

    def __init__(self, aurora):
        super().__init__()
        self._aurora = aurora

    def run(self):
        try:
            result = self._aurora.warmup()
        except Exception as e:
            result = f"Warmup raised: {e}"
        self.finished_with.emit(result)
