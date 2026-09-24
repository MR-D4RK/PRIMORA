import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from PRIMORA.verse import PresentationMode, VerseLifecycleState
from PRIMORA.verse.app import QML_DIR, VerseApplication


class InstantAurora:
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


def _load_universe():
    _qt_app()
    engine = QQmlApplicationEngine()
    engine.load(QUrl.fromLocalFile(str(QML_DIR / "Universe.qml")))
    roots = engine.rootObjects()
    assert len(roots) == 1
    return engine, roots[0]


def test_universe_component_files_exist():
    for name in ("Universe.qml", "Starfield.qml", "AuroraRibbons.qml",
                 "AuroraStrands.qml", "Presence.qml"):
        assert (QML_DIR / name).is_file(), name
    assert (QML_DIR / "Shell.qml").is_file()
    assert (QML_DIR / "Placeholder.qml").is_file()
    assert (QML_DIR / "Awakening.qml").is_file()


def test_universe_loads_with_layers():
    engine, universe = _load_universe()
    try:
        for child in ("starfield", "ribbons", "strands",
                      "presenceItem", "auraLight"):
            assert universe.findChild(object, child) is not None, child
        assert universe.property("active") is True
        energy = universe.property("energy")
        assert 0.4 <= energy <= 0.9
    finally:
        engine.deleteLater()
        _pump_until(lambda: True, 0.5)


def test_starfield_deterministic_layout():
    engine_a, first = _load_universe()
    engine_b, second = _load_universe()
    try:
        stars_a = first.findChild(object, "starfield").property("stars").toVariant()
        stars_b = second.findChild(object, "starfield").property("stars").toVariant()
        assert len(stars_a) == 220 + 8
        assert len(stars_a) == len(stars_b)
        assert stars_a[0] == stars_b[0]
        assert stars_a[-1] == stars_b[-1]
    finally:
        engine_a.deleteLater()
        engine_b.deleteLater()
        _pump_until(lambda: True, 0.5)


def test_active_gate_pauses_timers():
    engine, universe = _load_universe()
    try:
        fast = universe.findChild(object, "fastTimer")
        slow = universe.findChild(object, "slowTimer")
        assert fast.property("running") is True
        assert slow.property("running") is True
        universe.setProperty("active", False)
        assert _pump_until(lambda: not fast.property("running"), 5.0)
        assert slow.property("running") is False
        universe.setProperty("active", True)
        assert _pump_until(lambda: fast.property("running"), 5.0)
    finally:
        engine.deleteLater()
        _pump_until(lambda: True, 0.5)


def test_awake_view_shows_universe():
    _qt_app()
    app = VerseApplication(InstantAurora())
    app.start()
    try:
        assert _pump_until(
            lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
        universe = app._window.findChild(object, "universeRoot")
        assert universe is not None
        assert universe.property("active") is True
    finally:
        app.shutdown()
    assert app.state is VerseLifecycleState.CLOSED


def test_shutdown_with_universe_active():
    _qt_app()
    app = VerseApplication(InstantAurora())
    app.start()
    assert _pump_until(
        lambda: app.state is VerseLifecycleState.AWAKE, 10.0)
    started = time.time()
    app.shutdown()
    assert time.time() - started < 15.0
    assert app.state is VerseLifecycleState.CLOSED


if __name__ == "__main__":
    test_universe_component_files_exist()
    test_universe_loads_with_layers()
    test_starfield_deterministic_layout()
    test_active_gate_pauses_timers()
    test_awake_view_shows_universe()
    test_shutdown_with_universe_active()
    print("All Verse universe tests passed!")
