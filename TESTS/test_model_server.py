"""Unit tests for PRIMORA.model_server - the llama-server lifecycle.

Every test here is hermetic: no network, no real llama-server, no GGUF
load. The health probe and Popen are patched, and assets are temp files,
so these tests prove the decision logic and process bookkeeping without
ever spawning the real binary.
"""

import sys

sys.path.insert(0, "C:\\PRIMORA PROJECT")

import PRIMORA.model_server as ms
from PRIMORA.model_server import (
    ModelServerError,
    ModelServerService,
    should_manage,
)


class _FakeProc:
    def __init__(self, returncode=None, pid=4321):
        self.returncode = returncode
        self.pid = pid
        self.terminated = 0
        self.killed = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1
        self.returncode = -15

    def kill(self):
        self.killed += 1
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def _write_assets(svc):
    """Create the exe and GGUF the asset preflight requires."""
    svc.exe.parent.mkdir(parents=True, exist_ok=True)
    svc.exe.write_bytes(b"MZ")
    svc.model.write_bytes(b"GGUF")


def _service(tmp_path, **overrides):
    kwargs = dict(
        exe=tmp_path / "llama-server.exe",
        model=tmp_path / "model.gguf",
        log_path=tmp_path / "logs" / "llama-server.log",
        mode="auto",
    )
    kwargs.update(overrides)
    return ModelServerService(**kwargs)


# --- configuration ---------------------------------------------------------


def test_from_env_defaults_match_the_launcher():
    svc = ModelServerService.from_env(env={})
    assert svc.exe == ms.DEFAULT_SERVER_EXE
    assert svc.model == ms.DEFAULT_MODEL_PATH
    assert svc.host == "127.0.0.1"
    assert svc.port == 8080
    assert svc.alias == "qwen3-8b-heretic"
    assert svc.ctx == 4096
    assert svc.wait_seconds == 180
    assert svc.keep is False
    assert svc.mode == "auto"
    assert svc.base_url == "http://127.0.0.1:8080/v1"


def test_from_env_honours_overrides():
    svc = ModelServerService.from_env(env={
        "PRIMORA_LLAMACPP_SERVER": r"D:\bin\llama-server.exe",
        "PRIMORA_LLAMACPP_MODEL": r"D:\models\a.gguf",
        "PRIMORA_MODEL_SERVER_PORT": "8081",
        "PRIMORA_MODEL_SERVER_CTX": "2048",
        "PRIMORA_MODEL_SERVER_ALIAS": "served-alias",
        "PRIMORA_MODEL_SERVER_WAIT": "30",
        "PRIMORA_MODEL_SERVER_KEEP": "1",
        "PRIMORA_MODEL_SERVER": "on",
    })
    assert str(svc.exe) == r"D:\bin\llama-server.exe"
    assert str(svc.model) == r"D:\models\a.gguf"
    assert svc.port == 8081
    assert svc.ctx == 2048
    assert svc.alias == "served-alias"
    assert svc.wait_seconds == 30
    assert svc.keep is True
    assert svc.mode == "on"


def test_from_env_rejects_unknown_mode():
    try:
        ModelServerService.from_env(env={"PRIMORA_MODEL_SERVER": "maybe"})
    except ModelServerError as exc:
        assert "PRIMORA_MODEL_SERVER" in str(exc)
    else:
        raise AssertionError("unknown mode must fail closed")


# --- should_manage ---------------------------------------------------------


def test_auto_manages_local_default_endpoint():
    assert should_manage("auto", "openai", "", "127.0.0.1", 8080) is True


def test_auto_manages_url_pointing_at_this_host_and_port():
    for url in ("http://127.0.0.1:8080/v1",
                "http://localhost:8080/v1",
                "http://localhost:8080"):
        assert should_manage("auto", "openai", url, "127.0.0.1", 8080) is True, url


def test_auto_leaves_alone_a_url_somewhere_else():
    for url in ("http://127.0.0.1:8081/v1",
                "http://192.168.1.10:8080/v1",
                "https://api.example.com/v1"):
        assert should_manage("auto", "openai", url, "127.0.0.1", 8080) is False, url


def test_auto_leaves_alone_transformers_and_invalid_backends():
    assert should_manage("auto", "transformers", "", "127.0.0.1", 8080) is False
    assert should_manage("auto", "garbage", "", "127.0.0.1", 8080) is False


def test_off_never_manages_and_on_always_does():
    assert should_manage("off", "openai", "", "127.0.0.1", 8080) is False
    assert should_manage("off", "openai", "", "127.0.0.1", 8080) is False
    assert should_manage("on", "transformers", "", "127.0.0.1", 8080) is True
    assert should_manage("on", "openai",
                         "https://elsewhere:9/v1", "127.0.0.1", 8080) is True


# --- health probe ----------------------------------------------------------


class _FakeResponse:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_health_is_true_only_for_200(monkeypatch):
    import urllib.error

    statuses = [200]

    def fake_urlopen(url, timeout=None):
        return _FakeResponse(statuses[0])

    monkeypatch.setattr(ms.urllib.request, "urlopen", fake_urlopen)
    assert ms.health("127.0.0.1", 8080) is True
    statuses[0] = 503  # llama-server reports 503 while the model loads
    assert ms.health("127.0.0.1", 8080) is False

    def dead_urlopen(url, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ms.urllib.request, "urlopen", dead_urlopen)
    assert ms.health("127.0.0.1", 8080) is False


# --- ensure: adopt / start / fail -----------------------------------------


def test_healthy_server_is_adopted_without_starting_anything(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "health", lambda host, port, timeout=2.0: True)

    def forbidden(*a, **k):
        raise AssertionError("adopting must never spawn a process")

    monkeypatch.setattr(ms.subprocess, "Popen", forbidden)
    svc = _service(tmp_path)
    status = svc.ensure()
    assert status.state == "adopted"
    assert status.pid is None
    assert svc._proc is None


def test_missing_binary_is_actionable_and_downloads_nothing(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "health", lambda host, port, timeout=2.0: False)

    def forbidden(*a, **k):
        raise AssertionError("nothing may be spawned when assets are missing")

    monkeypatch.setattr(ms.subprocess, "Popen", forbidden)
    svc = _service(tmp_path)  # exe does not exist in tmp_path
    try:
        svc.ensure()
    except ModelServerError as exc:
        message = str(exc)
        assert "llama-server binary" in message
        assert str(svc.exe) in message
        assert "Nothing was downloaded or installed" in message
    else:
        raise AssertionError("missing exe must fail closed")


def test_missing_gguf_names_the_model_path(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "health", lambda host, port, timeout=2.0: False)
    svc = _service(tmp_path)
    svc.exe.write_bytes(b"MZ")  # exe present, model absent
    try:
        svc.ensure()
    except ModelServerError as exc:
        message = str(exc)
        assert "GGUF model file" in message
        assert str(svc.model) in message
        assert "Nothing was downloaded or installed" in message
    else:
        raise AssertionError("missing model must fail closed")


def test_start_uses_the_launcher_arguments_and_reports_started(
        tmp_path, monkeypatch):
    # Health: down for the initial probe, up once the server is spawned.
    probe = {"calls": 0}

    def health(host, port, timeout=2.0):
        probe["calls"] += 1
        return probe["calls"] > 1

    monkeypatch.setattr(ms, "health", health)
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        return _FakeProc()

    monkeypatch.setattr(ms.subprocess, "Popen", fake_popen)
    registered = []
    monkeypatch.setattr(ms.atexit, "register", registered.append)

    svc = _service(tmp_path)
    _write_assets(svc)
    status = svc.ensure()
    assert status.state == "started"
    assert status.pid == 4321
    assert captured["args"] == [
        str(svc.exe),
        "-m", str(svc.model),
        "--host", "127.0.0.1",
        "--port", "8080",
        "-c", "4096",
        "-ngl", "99",
        "--alias", "qwen3-8b-heretic",
    ]
    assert svc.log_path.exists()
    assert registered and registered[0] == svc._atexit_stop


def test_early_exit_reports_the_code_and_the_log_tail(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "health", lambda host, port, timeout=2.0: False)

    def fake_popen(args, **kwargs):
        return _FakeProc(returncode=7)

    monkeypatch.setattr(ms.subprocess, "Popen", fake_popen)
    svc = _service(tmp_path)
    _write_assets(svc)
    svc.log_path.parent.mkdir(parents=True, exist_ok=True)
    svc.log_path.write_text(
        "llama: error: could not load model\n", encoding="utf-8")
    try:
        svc.ensure()
    except ModelServerError as exc:
        message = str(exc)
        assert "exit code 7" in message
        assert "could not load model" in message
        assert str(svc.log_path) in message
    else:
        raise AssertionError("early exit must fail closed")


def test_timeout_terminates_only_the_server_we_started(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "health", lambda host, port, timeout=2.0: False)
    proc = _FakeProc()

    def fake_popen(args, **kwargs):
        return proc

    monkeypatch.setattr(ms.subprocess, "Popen", fake_popen)
    svc = _service(tmp_path, wait_seconds=0)
    _write_assets(svc)
    try:
        svc.ensure()
    except ModelServerError as exc:
        assert "did not become healthy" in str(exc)
        assert "0s" in str(exc)
    else:
        raise AssertionError("timeout must fail closed")
    assert proc.terminated >= 1 or proc.killed >= 1


# --- stop ------------------------------------------------------------------


def test_stop_never_touches_an_adopted_server(tmp_path, capsys):
    # Adopted = no process bookkeeping of ours, so stop() is a no-op.
    svc = _service(tmp_path)
    svc.stop()
    assert svc._proc is None
    assert capsys.readouterr().out == ""


def test_stop_terminates_a_server_it_started(tmp_path, capsys):
    svc = _service(tmp_path)
    proc = _FakeProc()
    svc._proc = proc
    svc.stop()
    assert proc.terminated == 1
    assert "Stopping llama-server" in capsys.readouterr().out


def test_keep_leaves_a_started_server_running(tmp_path, capsys):
    svc = _service(tmp_path, keep=True)
    proc = _FakeProc()
    svc._proc = proc
    svc.stop()
    assert proc.terminated == 0
    assert "Leaving llama-server running" in capsys.readouterr().out


def test_atexit_backstop_stops_a_started_server(tmp_path):
    svc = _service(tmp_path)
    proc = _FakeProc()
    svc._proc = proc
    svc._atexit_stop()
    assert proc.terminated == 1


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
