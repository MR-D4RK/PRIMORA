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
    """Run main() with patched argv/env/providers, restoring everything.

    ``PRIMORA_MODEL_SERVER`` defaults to ``off`` so no test can ever spawn
    a real llama-server; lifecycle tests pass an explicit value and patch
    ``main_mod.ModelServerService`` with a fake.
    """
    old_argv = sys.argv
    saved_env = {}
    managed = (
        "PRIMORA_OLLAMA_MODEL", "PRIMORA_OLLAMA_URL",
        "PRIMORA_OLLAMA_TIMEOUT", "PRIMORA_AURORA_MODEL_PATH",
        "PRIMORA_AURORA_ADAPTER_PATH", "PRIMORA_SERVER_PORT",
        "PRIMORA_AURORA_BACKEND", "PRIMORA_PROVIDER",
        "PRIMORA_OPENAI_MODEL", "PRIMORA_OPENAI_URL",
        "PRIMORA_OPENAI_TIMEOUT", "PRIMORA_MODEL_SERVER",
        "PRIMORA_MODEL_SERVER_KEEP", "PRIMORA_LLAMACPP_SERVER",
        "PRIMORA_LLAMACPP_MODEL",
    )
    overrides = {"PRIMORA_MODEL_SERVER": "off", **env_overrides}
    for key in managed:
        saved_env[key] = os.environ.get(key)
        if key in overrides:
            os.environ[key] = overrides[key]
        elif key in os.environ:
            del os.environ[key]
    old_provider = main_mod.OllamaProvider
    old_openai_provider = main_mod.OpenAICompatibleProvider
    old_service = main_mod.ModelServerService
    old_run_gui = main_mod.run_gui
    old_run_cli = main_mod.run_cli
    old_serve = main_mod.serve
    calls = {"run_gui": [], "run_cli": [], "serve": []}
    provider = provider_factory or RecordingProvider
    main_mod.OllamaProvider = provider
    main_mod.OpenAICompatibleProvider = provider
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
        main_mod.OpenAICompatibleProvider = old_openai_provider
        main_mod.ModelServerService = old_service
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


def test_gui_default_openai_model_recorded():
    # The entry point defaults to the live configuration: the OpenAI
    # provider talking to llama-server, with the served model alias.
    model_dir, adapter_dir = _temp_model_dirs()
    seen = {}

    class ModelRecorder(RecordingProvider):
        def __init__(self, model, base_url=None, timeout=None):
            seen["model"] = model
            seen["base_url"] = base_url
            super().__init__(model, base_url, timeout)

    outcome, result, calls = _run_main(
        ["main.py"],
        {"PRIMORA_AURORA_MODEL_PATH": model_dir,
         "PRIMORA_AURORA_ADAPTER_PATH": adapter_dir},
        provider_factory=ModelRecorder,
    )
    assert outcome == "return"
    assert seen["model"] == "qwen3-8b-heretic"
    assert seen["base_url"] == "http://127.0.0.1:8080/v1"


def test_gui_uses_configured_ollama_model():
    # The ollama backend remains selectable as an escape hatch.
    model_dir, adapter_dir = _temp_model_dirs()
    seen = {}

    class ModelRecorder(RecordingProvider):
        def __init__(self, model, base_url=None, timeout=None):
            seen["model"] = model
            super().__init__(model, base_url, timeout)

    outcome, result, calls = _run_main(
        ["main.py"],
        {"PRIMORA_PROVIDER": "ollama",
         "PRIMORA_OLLAMA_MODEL": "custom:model",
         "PRIMORA_AURORA_MODEL_PATH": model_dir,
         "PRIMORA_AURORA_ADAPTER_PATH": adapter_dir},
        provider_factory=ModelRecorder,
    )
    assert outcome == "return"
    assert seen["model"] == "custom:model"


def test_gui_honours_explicit_transformers_backend():
    # An explicit PRIMORA_AURORA_BACKEND=transformers still wins over the
    # entry point's openai default (the documented escape hatch).
    from PRIMORA.aurora.brain import AuroraBrain

    model_dir, adapter_dir = _temp_model_dirs()
    outcome, result, calls = _run_main(
        ["main.py"],
        {"PRIMORA_AURORA_BACKEND": "transformers",
         "PRIMORA_AURORA_MODEL_PATH": model_dir,
         "PRIMORA_AURORA_ADAPTER_PATH": adapter_dir},
    )
    assert outcome == "return", result
    assert result == 0
    assert len(calls["run_gui"]) == 1
    assert isinstance(calls["run_gui"][0]._brain, AuroraBrain)


def test_cli_requires_ollama_env_when_ollama_selected():
    # Fail closed when the operator explicitly selects ollama without a
    # model name; the default configuration needs no environment at all.
    outcome, result, calls = _run_main(
        ["main.py", "--cli"], {"PRIMORA_PROVIDER": "ollama"})
    assert outcome == "exit"
    assert result == 1
    assert calls["run_gui"] == []
    assert calls["run_cli"] == []


def test_cli_runs_with_default_configuration_no_env():
    # One official entry point: `main --cli` works with no environment.
    model_dir, adapter_dir = _temp_model_dirs()
    old_warmup = Aurora.warmup
    Aurora.warmup = lambda self: None
    try:
        outcome, result, calls = _run_main(
            ["main.py", "--cli"],
            {"PRIMORA_AURORA_MODEL_PATH": model_dir,
             "PRIMORA_AURORA_ADAPTER_PATH": adapter_dir},
        )
    finally:
        Aurora.warmup = old_warmup
    assert outcome == "return", result
    assert len(calls["run_cli"]) == 1
    assert calls["run_gui"] == []
    assert calls["serve"] == []


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
                "PRIMORA_AURORA_ADAPTER_PATH", "PRIMORA_MODEL_SERVER",
                "PRIMORA_AURORA_BACKEND", "PRIMORA_PROVIDER",
                "PRIMORA_OPENAI_MODEL", "PRIMORA_OPENAI_URL",
                "PRIMORA_OPENAI_TIMEOUT"):
        saved[key] = os.environ.get(key)
    os.environ["PRIMORA_OLLAMA_MODEL"] = "custom:model"
    os.environ["PRIMORA_AURORA_MODEL_PATH"] = model_dir
    os.environ["PRIMORA_AURORA_ADAPTER_PATH"] = adapter_dir
    # Never spawn a real llama-server from a unit test.
    os.environ["PRIMORA_MODEL_SERVER"] = "off"
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


def _fake_service_cls(ensure_error=None):
    """A ModelServerService stand-in that records calls, never any I/O."""
    from PRIMORA.model_server import ModelServerStatus

    class FakeService:
        instances = []

        def __init__(self, mode):
            self.mode = mode
            self.host = "127.0.0.1"
            self.port = 8080
            self.alias = "qwen3-8b-heretic"
            self.base_url = "http://127.0.0.1:8080/v1"
            self.ensure_calls = 0
            self.stop_calls = 0
            self.env_at_ensure = None
            self.env_at_stop = None
            FakeService.instances.append(self)

        @classmethod
        def from_env(cls, env=None):
            source = os.environ if env is None else env
            mode = str(source.get("PRIMORA_MODEL_SERVER") or "auto")
            return cls(mode)

        def ensure(self):
            self.ensure_calls += 1
            self.env_at_ensure = dict(os.environ)
            if ensure_error is not None:
                raise ensure_error
            return ModelServerStatus(
                state="adopted", base_url=self.base_url)

        def stop(self):
            self.stop_calls += 1
            self.env_at_stop = dict(os.environ)

    return FakeService


def test_auto_mode_ensures_model_server_and_stops_it():
    # Default configuration (no explicit endpoint): the local llama-server
    # is this process's responsibility - ensure it, apply the openai
    # defaults for this process only, and stop it on the way out.
    FakeService = _fake_service_cls()
    old_service = main_mod.ModelServerService
    main_mod.ModelServerService = FakeService
    try:
        outcome, result, calls = _run_main(
            ["main.py"], {"PRIMORA_MODEL_SERVER": "auto"})
    finally:
        main_mod.ModelServerService = old_service
    assert outcome == "return", result
    assert result == 0
    assert len(FakeService.instances) == 1
    inst = FakeService.instances[0]
    assert inst.ensure_calls == 1
    assert inst.stop_calls == 1
    assert inst.env_at_ensure["PRIMORA_AURORA_BACKEND"] == "openai"
    assert inst.env_at_ensure["PRIMORA_PROVIDER"] == "openai"
    assert inst.env_at_ensure["PRIMORA_OPENAI_URL"] == inst.base_url
    assert inst.env_at_ensure["PRIMORA_OPENAI_MODEL"] == "qwen3-8b-heretic"
    assert inst.env_at_ensure["PRIMORA_OPENAI_TIMEOUT"] == "300"
    assert len(calls["run_gui"]) == 1


def test_model_server_failure_exits_with_code_2(capsys):
    # A missing binary/model or a server that never becomes healthy is an
    # actionable startup error with its own exit code, before any
    # construction happens.
    from PRIMORA.model_server import ModelServerError

    FakeService = _fake_service_cls(
        ensure_error=ModelServerError("MISSING: GGUF model file"))
    old_service = main_mod.ModelServerService
    main_mod.ModelServerService = FakeService
    try:
        outcome, result, calls = _run_main(
            ["main.py"], {"PRIMORA_MODEL_SERVER": "auto"})
    finally:
        main_mod.ModelServerService = old_service
    assert outcome == "exit"
    assert result == 2
    assert calls["run_gui"] == []
    assert calls["run_cli"] == []
    assert "MISSING: GGUF model file" in capsys.readouterr().out
    # The failure happened before ensure could report, but the finally
    # still must not claim to stop a server we never started - stop() is
    # called and is a no-op in the real service when no process exists.
    assert len(FakeService.instances) == 1
    assert FakeService.instances[0].stop_calls == 1


def test_explicit_remote_endpoint_is_not_managed():
    # An operator who exported their own endpoint keeps it: no local
    # server is spawned, and the explicit URL survives the defaults.
    FakeService = _fake_service_cls()
    old_service = main_mod.ModelServerService
    main_mod.ModelServerService = FakeService
    try:
        outcome, result, calls = _run_main(
            ["main.py"],
            {"PRIMORA_MODEL_SERVER": "auto",
             "PRIMORA_OPENAI_URL": "https://remote.example:9999/v1"},
        )
    finally:
        main_mod.ModelServerService = old_service
    assert outcome == "return", result
    inst = FakeService.instances[0]
    assert inst.ensure_calls == 0, "explicit remote endpoint must be left alone"
    assert inst.stop_calls == 1
    assert inst.env_at_stop["PRIMORA_OPENAI_URL"] == \
        "https://remote.example:9999/v1"
    assert len(calls["run_gui"]) == 1


def test_cli_does_not_import_gui_stack():
    # `main --cli` is terminal-only: the Qt/Verse stack must not even be
    # imported in the process.
    import subprocess
    import textwrap

    script = textwrap.dedent("""
        import sys, tempfile
        from pathlib import Path
        sys.argv = ['main.py', '--cli']
        import PRIMORA.main as m
        from PRIMORA.aurora import Aurora
        from PRIMORA.prime.context.manager import ContextManager
        from PRIMORA.prime.context.store import ContextStore

        # Mirror TESTS/conftest.py: no test may touch production memory.
        _tmp = tempfile.mkdtemp()
        _orig = ContextManager._get_store
        def _safe(self):
            if self._db_path is None and self._store is None:
                self._store = ContextStore(str(Path(_tmp) / 'ctx.db'))
            return _orig(self)
        ContextManager._get_store = _safe

        Aurora.warmup = lambda self: None
        m.run_cli = lambda aurora: None
        m.serve = lambda *a, **k: None
        code = m.main()
        gui_loaded = ('PRIMORA.verse' in sys.modules) or \\
            ('PySide6' in sys.modules)
        print('GUI_LOADED=' + str(gui_loaded))
        rc = 0
        if gui_loaded:
            rc = 3
        elif code:
            rc = 1
        sys.exit(rc)
    """)
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    env = dict(os.environ)
    for key in ("PRIMORA_AURORA_BACKEND", "PRIMORA_PROVIDER",
                "PRIMORA_OPENAI_MODEL", "PRIMORA_OPENAI_URL",
                "PRIMORA_OPENAI_TIMEOUT", "PRIMORA_OLLAMA_MODEL"):
        env.pop(key, None)
    env["PRIMORA_MODEL_SERVER"] = "off"
    env["PYTHONPATH"] = root
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, cwd=root, timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "GUI_LOADED=False" in proc.stdout
    assert "PRIMORA ready" in proc.stdout


if __name__ == "__main__":
    test_gui_starts_without_ollama_env()
    test_gui_default_openai_model_recorded()
    test_gui_uses_configured_ollama_model()
    test_cli_requires_ollama_env_when_ollama_selected()
    test_cli_runs_with_default_configuration_no_env()
    test_cli_preserves_sync_warmup_and_loop()
    test_async_load_flag_applied_for_transformers()
    test_gui_run_invokes_no_synchronous_warmup()
    print("All main startup tests passed!")
