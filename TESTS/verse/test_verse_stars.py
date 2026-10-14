import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PySide6.QtCore import QUrl, QMetaObject
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from PRIMORA.verse.app import QML_DIR

PHASES = [0.0, 0.5, 1.0, 1.7, 2.4, 3.1, 4.2, 5.3, 6.0]


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


def _probe_engine():
    _qt_app()
    engine = QQmlApplicationEngine()
    here = os.path.dirname(os.path.abspath(__file__))
    engine.load(QUrl.fromLocalFile(
        os.path.join(here, "star_reveal_probe.qml")))
    roots = engine.rootObjects()
    assert len(roots) == 1
    return engine, roots[0]


def _run_gates(root, phases, reveals):
    root.setProperty("phases", phases)
    root.setProperty("reveals", reveals)
    QMetaObject.invokeMethod(root, "run")
    _pump_until(lambda: True, 0.5)
    return root.property("probe").toVariant()


def test_startup_black_no_visible_stars():
    engine, root = _probe_engine()
    try:
        rows = _run_gates(root, PHASES, [0.0])
        assert len(rows) == 1
        assert all(g == 0.0 for g in rows[0]), rows[0]
    finally:
        engine.deleteLater()
        _pump_until(lambda: True, 0.5)


def test_appearance_progression():
    engine, root = _probe_engine()
    try:
        rows = _run_gates(root, PHASES, [0.0, 0.5, 1.0])
        assert all(g == 0.0 for g in rows[0])
        mid = rows[1]
        assert all(0.0 <= g <= 1.0 for g in mid)
        assert max(mid) - min(mid) > 0.1, mid
        assert all(g == 1.0 for g in rows[2]), rows[2]
    finally:
        engine.deleteLater()
        _pump_until(lambda: True, 0.5)


def test_gates_monotonic_in_reveal():
    engine, root = _probe_engine()
    try:
        reveals = [0.0, 0.25, 0.5, 0.75, 1.0]
        rows = _run_gates(root, PHASES, reveals)
        for j in range(len(PHASES)):
            col = [rows[i][j] for i in range(len(reveals))]
            assert all(b >= a for a, b in zip(col, col[1:])), (j, col)
            assert col[0] == 0.0
            assert col[-1] == 1.0
    finally:
        engine.deleteLater()
        _pump_until(lambda: True, 0.5)


def test_gates_unsynchronized_across_stars():
    engine, root = _probe_engine()
    try:
        rows = _run_gates(root, PHASES, [0.35])
        row = rows[0]
        assert len(set(row)) > 1, row
    finally:
        engine.deleteLater()
        _pump_until(lambda: True, 0.5)


def test_reveal_defaults_full_standalone():
    _qt_app()
    engine = QQmlApplicationEngine()
    engine.load(QUrl.fromLocalFile(str(QML_DIR / "Starfield.qml")))
    try:
        assert len(engine.rootObjects()) == 1
        assert engine.rootObjects()[0].property("reveal") == 1.0
    finally:
        engine.deleteLater()
        _pump_until(lambda: True, 0.5)


def test_universe_wires_reveal_clock():
    _qt_app()
    engine = QQmlApplicationEngine()
    engine.load(QUrl.fromLocalFile(str(QML_DIR / "Universe.qml")))
    universe = engine.rootObjects()[0]
    try:
        from PySide6.QtCore import QObject
        starfield = universe.findChild(QObject, "starfield")
        assert starfield is not None
        assert universe.property("starReveal") == 0.0
        # Freeze the wall-clock driver so manual values are not overwritten.
        universe.setProperty("active", False)
        assert _pump_until(
            lambda: not universe.findChild(
                QObject, "fastTimer").property("running"), 5.0)
        for value in (0.0, 0.5, 1.0):
            universe.setProperty("starReveal", value)
            assert _pump_until(
                lambda v=value: starfield.property("reveal") == v, 5.0)
    finally:
        engine.deleteLater()
        _pump_until(lambda: True, 0.5)


def test_reveal_independent_of_model_state():
    _qt_app()
    engine = QQmlApplicationEngine()
    engine.load(QUrl.fromLocalFile(str(QML_DIR / "Universe.qml")))
    universe = engine.rootObjects()[0]
    try:
        from PySide6.QtCore import QObject
        starfield = universe.findChild(QObject, "starfield")
        universe.setProperty("active", False)
        assert _pump_until(
            lambda: not universe.findChild(
                QObject, "fastTimer").property("running"), 5.0)
        universe.setProperty("dormancy", 1.0)
        universe.setProperty("starReveal", 1.0)
        assert _pump_until(
            lambda: starfield.property("reveal") == 1.0, 5.0)
    finally:
        engine.deleteLater()
        _pump_until(lambda: True, 0.5)


def test_twinkle_parameters_vary_per_star():
    _qt_app()
    engine = QQmlApplicationEngine()
    engine.load(QUrl.fromLocalFile(str(QML_DIR / "Universe.qml")))
    universe = engine.rootObjects()[0]
    try:
        from PySide6.QtCore import QObject
        starfield = universe.findChild(QObject, "starfield")
        stars = starfield.property("stars").toVariant()
        assert len(stars) == 228
        speeds = {s["speed"] for s in stars}
        phases = {s["phase"] for s in stars}
        assert len(speeds) > 100, len(speeds)
        assert len(phases) > 100, len(phases)
        assert all(0.0 <= s["x"] <= 1.0 and 0.0 <= s["y"] <= 1.0
                   for s in stars)
        assert sum(1 for s in stars if s["bright"]) == 8
    finally:
        engine.deleteLater()
        _pump_until(lambda: True, 0.5)


if __name__ == "__main__":
    test_startup_black_no_visible_stars()
    test_appearance_progression()
    test_gates_monotonic_in_reveal()
    test_gates_unsynchronized_across_stars()
    test_reveal_defaults_full_standalone()
    test_universe_wires_reveal_clock()
    test_reveal_independent_of_model_state()
    test_twinkle_parameters_vary_per_star()
    print("All Verse star tests passed!")
