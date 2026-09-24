import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QWindow
from PySide6.QtQml import QQmlApplicationEngine

from PRIMORA.verse import PresentationMode, VerseLifecycleState, VerseRuntime
from PRIMORA.verse.app import QML_DIR, VerseApplication


class StubAurora:
    pass


class InstantAurora:
    """Stub whose synchronous warmup stands in for the real model load."""

    def warmup(self):
        return None


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


def test_qml_shell_files_exist():
    assert (QML_DIR / "Shell.qml").is_file()
    assert (QML_DIR / "Placeholder.qml").is_file()


def test_app_constructs_with_stub_aurora():
    app = VerseApplication(StubAurora())
    assert app.mode is PresentationMode.PORTAL
    assert app.state is VerseLifecycleState.STARTING


def test_app_uses_provided_runtime_and_aurora():
    aurora = StubAurora()
    runtime = VerseRuntime(aurora)
    app = VerseApplication(aurora, runtime=runtime)
    assert app.runtime is runtime
    assert app.runtime.aurora is aurora


def test_app_rejects_mismatched_runtime():
    runtime = VerseRuntime(StubAurora())
    try:
        VerseApplication(StubAurora(), runtime=runtime)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "must wrap the provided Aurora instance" in str(e)


def test_qml_engine_loads_shell():
    _qt_app()
    engine = QQmlApplicationEngine()
    from PySide6.QtCore import QUrl
    engine.load(QUrl.fromLocalFile(str(QML_DIR / "Shell.qml")))
    assert len(engine.rootObjects()) == 1


def test_portal_mode_represented():
    _qt_app()
    app = VerseApplication(InstantAurora(), mode=PresentationMode.PORTAL)
    app.start()
    try:
        assert _pump_until(
            lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
        assert app.state is VerseLifecycleState.AWAKE
        assert app.mode is PresentationMode.PORTAL
        window = app._window
        assert not (window.flags() & Qt.FramelessWindowHint)
        assert window.visibility() == QWindow.Windowed
    finally:
        app.shutdown()


def test_immersive_mode_represented():
    _qt_app()
    app = VerseApplication(InstantAurora(), mode=PresentationMode.IMMERSIVE)
    app.start()
    try:
        assert _pump_until(
            lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
        assert app.state is VerseLifecycleState.AWAKE
        assert app.mode is PresentationMode.IMMERSIVE
        window = app._window
        assert bool(window.flags() & Qt.FramelessWindowHint)
        assert window.visibility() == QWindow.FullScreen
    finally:
        app.shutdown()
    assert app.state is VerseLifecycleState.CLOSED


def test_mode_switch_updates_runtime_and_window():
    _qt_app()
    app = VerseApplication(InstantAurora())
    app.start()
    try:
        assert _pump_until(
            lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
        app.set_mode(PresentationMode.IMMERSIVE)
        assert app.mode is PresentationMode.IMMERSIVE
        assert bool(app._window.flags() & Qt.FramelessWindowHint)
        app.set_mode(PresentationMode.PORTAL)
        assert app.mode is PresentationMode.PORTAL
        assert not (app._window.flags() & Qt.FramelessWindowHint)
    finally:
        app.shutdown()


def test_failed_qml_load_degrades_cleanly():
    import tempfile
    _qt_app()
    app = VerseApplication(StubAurora(), qml_dir=tempfile.mkdtemp())
    try:
        app.start()
        assert False, "Should raise RuntimeError"
    except RuntimeError as e:
        assert "failed to load" in str(e)
    assert app.state is VerseLifecycleState.DEGRADED


def test_closed_runtime_rejects_mode_changes():
    _qt_app()
    app = VerseApplication(StubAurora())
    app.start()
    app.shutdown()
    assert app.state is VerseLifecycleState.CLOSED
    try:
        app.set_mode(PresentationMode.IMMERSIVE)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_qml_shell_files_exist()
    test_app_constructs_with_stub_aurora()
    test_app_uses_provided_runtime_and_aurora()
    test_app_rejects_mismatched_runtime()
    test_qml_engine_loads_shell()
    test_portal_mode_represented()
    test_immersive_mode_represented()
    test_mode_switch_updates_runtime_and_window()
    test_failed_qml_load_degrades_cleanly()
    test_closed_runtime_rejects_mode_changes()
    print("All Verse application tests passed!")
