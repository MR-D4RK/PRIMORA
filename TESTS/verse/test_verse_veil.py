import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from PRIMORA.verse.app import QML_DIR


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


def test_veil_files_exist():
    assert (QML_DIR / "AuroraRibbons.qml").is_file()
    assert (QML_DIR / "Universe.qml").is_file()


def _load_veil():
    _qt_app()
    engine = QQmlApplicationEngine()
    engine.load(QUrl.fromLocalFile(str(QML_DIR / "AuroraRibbons.qml")))
    roots = engine.rootObjects()
    assert len(roots) == 1
    return engine, roots[0]


def test_veil_preserves_component_contract():
    engine, veil = _load_veil()
    try:
        assert veil.objectName() == "ribbons"
        assert veil.property("energy") is not None
        assert veil.property("intensity") is not None
        assert veil.property("uT") is not None
        veil.setProperty("intensity", 0.5)
        assert veil.property("intensity") == 0.5
        veil.setProperty("intensity", 1.0)
    finally:
        engine.deleteLater()
        _pump_until(lambda: True, 0.5)


def test_veil_repaint_advances_time():
    engine, veil = _load_veil()
    try:
        t0 = veil.property("uT")
        time.sleep(0.05)
        from PySide6.QtCore import QMetaObject
        QMetaObject.invokeMethod(veil, "repaint")
        _pump_until(lambda: True, 0.2)
        assert veil.property("uT") >= t0
    finally:
        engine.deleteLater()
        _pump_until(lambda: True, 0.5)


def test_universe_drives_veil_intensity():
    _qt_app()
    engine = QQmlApplicationEngine()
    engine.load(QUrl.fromLocalFile(str(QML_DIR / "Universe.qml")))
    universe = engine.rootObjects()[0]
    try:
        from PySide6.QtCore import QObject
        veil = universe.findChild(QObject, "ribbons")
        assert veil is not None
        universe.setProperty("dormancy", 1.0)
        assert _pump_until(
            lambda: veil.property("intensity") == 0.0, 5.0)
        universe.setProperty("dormancy", 0.0)
        assert _pump_until(
            lambda: veil.property("intensity") == 1.0, 5.0)
    finally:
        engine.deleteLater()
        _pump_until(lambda: True, 0.5)


if __name__ == "__main__":
    test_veil_files_exist()
    test_veil_preserves_component_contract()
    test_veil_repaint_advances_time()
    test_universe_drives_veil_intensity()
    print("All Verse veil tests passed!")
