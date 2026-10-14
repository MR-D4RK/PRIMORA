import os
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

import PRIMORA.main as main_mod
from PRIMORA.aurora import Aurora


class RecordingProvider:
    def __init__(self, model, base_url=None, timeout=None):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def invoke(self, capability, payload):
        from PRIMORA.prime.contract import PrimeResult
        return PrimeResult.failure("unused in startup tests")


def _temp_model_dirs():
    model_dir = tempfile.mkdtemp()
    adapter_dir = tempfile.mkdtemp()
    return model_dir, adapter_dir


def _run_main(argv, env_overrides, provider_factory=None):
    """Run main() with patched argv/env/providers, restoring everything."""
    old_argv = sys.argv
    saved_env = {}
    for key in ("PRIMORA_OLLAMA_MODEL", "PRIMORA_OLLAMA_URL",
                "PRIMORA_OLLAMA_TIMEOUT", "PRIMORA_AURORA_MODEL_PATH",
                "PRIMORA_AURORA_ADAPTER_PATH", "PRIMORA_SERVER_PORT"):
        saved_env[key] = os.environ.get(key)
        if key in env_overrides:
            os.environ[key] = env_overrides[key]
        elif key in os.environ:
            del os.environ[key]
    old_provider = main_mod.OllamaProvider
    old_run_gui = main_mod.run_gui
    old_run_cli = main_mod.run_cli
    old_serve = main_mod.serve
    calls = {"run_gui": [], "run_cli": [], "serve": []}
    main_mod.OllamaProvider = provider_factory or RecordingProvider
    main_mod.run_gui = lambda aurora: calls["run_gui"].append(aurora) or 0
    main_mod.run_cli = lambda aurora: calls["run_cli"].append(aurora)
    main_mod.serve = lambda *a, **k: calls["serve"].append((a, k))
    sys.argv = argv
    try:
        try:
            result = main_mod.main()
        except SystemExit as e:
            return ("exit", e.code, calls)
        return ("return", result, calls)
    finally:
        sys.argv = old_argv
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        main_mod.OllamaProvider = old_provider
        main_mod.run_gui = old_run_gui
        main_mod.run_cli = old_run_cli
        main_mod.serve = old_serve


def test_gui_starts_without_ollama_env():
    model_dir, adapter_dir = _temp_model_dirs()
    outcome, result, calls = _run_main(
        ["main.py"],
        {"PRIMORA_AURORA_MODEL_PATH": model_dir,
         "PRIMORA_AURORA_ADAPTER_PATH": adapter_dir},
    )
    assert outcome == "return", result
    assert result == 0
    assert len(calls["run_gui"]) == 1
    assert isinstance(calls["run_gui"][0], Aurora)
    assert len(calls["run_cli"]) == 0


def test_gui_default_ollama_model_recorded():
    model_dir, adapter_dir = _temp_model_dirs()
    seen = {}

    class ModelRecorder(RecordingProvider):
        def __init__(self, model, base_url=None, timeout=None):
            seen["model"] = model
            super().__init__(model, base_url, timeout)

    outcome, result, calls = _run_main(
        ["main.py"],
        {"PRIMORA_AURORA_MODEL_PATH": model_dir,
         "PRIMORA_AURORA_ADAPTER_PATH": adapter_dir},
        provider_factory=ModelRecorder,
    )
    assert outcome == "return"
    assert seen["model"] == "qwen3-8b-16k:latest"


def test_gui_uses_configured_ollama_model():
    model_dir, adapter_dir = _temp_model_dirs()
    seen = {}

    class ModelRecorder(RecordingProvider):
        def __init__(self, model, base_url=None, timeout=None):
            seen["model"] = model
            super().__init__(model, base_url, timeout)

    outcome, result, calls = _run_main(
        ["main.py"],
        {"PRIMORA_OLLAMA_MODEL": "custom:model",
         "PRIMORA_AURORA_MODEL_PATH": model_dir,
         "PRIMORA_AURORA_ADAPTER_PATH": adapter_dir},
        provider_factory=ModelRecorder,
    )
    assert outcome == "return"
    assert seen["model"] == "custom:model"


def test_cli_requires_ollama_env():
    outcome, result, calls = _run_main(["main.py", "--cli"], {})
    assert outcome == "exit"
    assert result == 1
    assert calls["run_gui"] == []
    assert calls["run_cli"] == []


def test_cli_preserves_sync_warmup_and_loop():
    model_dir, adapter_dir = _temp_model_dirs()
    order = []
    real_warmup = Aurora.warmup

    def recording_warmup(self):
        order.append("warmup")
        return None

    Aurora.warmup = recording_warmup
    old_run_cli = main_mod.run_cli
    old_serve = main_mod.serve
    main_mod.run_cli = lambda aurora: order.append("run_cli")
    main_mod.serve = lambda *a, **k: None
    old_argv = sys.argv
    saved = {}
    for key in ("PRIMORA_OLLAMA_MODEL", "PRIMORA_AURORA_MODEL_PATH",
                "PRIMORA_AURORA_ADAPTER_PATH"):
        saved[key] = os.environ.get(key)
    os.environ["PRIMORA_OLLAMA_MODEL"] = "custom:model"
    os.environ["PRIMORA_AURORA_MODEL_PATH"] = model_dir
    os.environ["PRIMORA_AURORA_ADAPTER_PATH"] = adapter_dir
    sys.argv = ["main.py", "--cli"]
    try:
        try:
            main_mod.main()
        except SystemExit as e:
            assert False, f"CLI should not exit, got {e.code}"
    finally:
        sys.argv = old_argv
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        Aurora.warmup = real_warmup
        main_mod.run_cli = old_run_cli
        main_mod.serve = old_serve
    assert order[0] == "warmup"
    assert order[1] == "run_cli"


def test_cli_does_not_start_web_ui():
    model_dir, adapter_dir = _temp_model_dirs()
    import threading as _threading
    thread_starts = []
    real_thread = _threading.Thread
    real_warmup = Aurora.warmup

    class RecordingThread(real_thread):
        def start(self):
            thread_starts.append(self)
            # Do not actually start: serve is patched and must not run.

    Aurora.warmup = lambda self: None
    _threading.Thread = RecordingThread
    try:
        outcome, result, calls = _run_main(
            ["main.py", "--cli"],
            {"PRIMORA_OLLAMA_MODEL": "custom:model",
             "PRIMORA_AURORA_MODEL_PATH": model_dir,
             "PRIMORA_AURORA_ADAPTER_PATH": adapter_dir},
        )
    finally:
        _threading.Thread = real_thread
        Aurora.warmup = real_warmup
    assert thread_starts == [], "CLI must not start the web UI thread"
    assert len(calls["run_cli"]) == 1


def test_gui_starts_web_ui():
    model_dir, adapter_dir = _temp_model_dirs()
    import threading as _threading
    thread_starts = []
    real_thread = _threading.Thread

    class RecordingThread(real_thread):
        def start(self):
            thread_starts.append(self)
            # Do not actually start: serve is patched and must not run.

    _threading.Thread = RecordingThread
    try:
        outcome, result, calls = _run_main(
            ["main.py"],
            {"PRIMORA_AURORA_MODEL_PATH": model_dir,
             "PRIMORA_AURORA_ADAPTER_PATH": adapter_dir},
        )
    finally:
        _threading.Thread = real_thread
    assert outcome == "return", result
    # GUI keeps the historical behaviour of exposing the web interface.
    assert len(thread_starts) == 1, "GUI should start the web UI thread"
    assert len(calls["run_gui"]) == 1


def test_async_load_flag_applied_for_transformers():
    # main.py must set the upstream switch before any model load so
    # Transformers materializes safetensors sequentially.
    assert os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == "true"
    from transformers.utils import is_env_variable_true
    assert is_env_variable_true("HF_DEACTIVATE_ASYNC_LOAD") is True


def test_gui_run_invokes_no_synchronous_warmup():
    # run_gui must not call aurora.warmup() itself; the single warmup
    # belongs to VerseApplication.start()'s worker (no double warmup).
    import threading
    from PySide6 import QtGui
    from PRIMORA.verse import app as verse_app_mod

    calls = {"warmup": 0, "starts": 0}
    lock = threading.Lock()

    class CountingAurora:
        def warmup(self):
            with lock:
                calls["warmup"] += 1
            return None

    class FakeVerse:
        def __init__(self, aurora, mode=None):
            self._aurora = aurora

        def start(self):
            calls["starts"] += 1
            return self

        def shutdown(self):
            pass

    class FakeSignal:
        def connect(self, fn):
            pass

    class FakeQtApp:
        def __init__(self, argv):
            self.aboutToQuit = FakeSignal()

        def exec(self):
            return 0

    real_qt_app = QtGui.QGuiApplication
    real_verse = verse_app_mod.VerseApplication
    QtGui.QGuiApplication = FakeQtApp
    verse_app_mod.VerseApplication = FakeVerse
    try:
        assert main_mod.run_gui(CountingAurora()) == 0
    finally:
        QtGui.QGuiApplication = real_qt_app
        verse_app_mod.VerseApplication = real_verse
    assert calls["starts"] == 1
    assert calls["warmup"] == 0


if __name__ == "__main__":
    test_gui_starts_without_ollama_env()
    test_gui_default_ollama_model_recorded()
    test_gui_uses_configured_ollama_model()
    test_cli_requires_ollama_env()
    test_cli_preserves_sync_warmup_and_loop()
    test_async_load_flag_applied_for_transformers()
    test_gui_run_invokes_no_synchronous_warmup()
    print("All main startup tests passed!")
