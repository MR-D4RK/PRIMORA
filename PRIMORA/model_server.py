"""Local llama-server lifecycle for Aurora's OpenAI-compatible backend.

Ported from ``launch-aurora-llamacpp.ps1`` so the official entry point
(``python -m PRIMORA.main``) owns the same connect-or-start logic the
launcher had:

*   Verify the llama-server binary and the GGUF model exist, with exact
    paths in the error. Nothing is downloaded or installed; the operator
    places the file and re-runs.
*   Reuse an already-healthy server instead of starting a second one. An
    adopted server is never stopped by this process - only a server this
    module started can be stopped, and only when keep is off.
*   Start the server with the launcher's proven arguments, wait for
    ``/health`` (llama-server reports 503 while the model loads and 200
    when ready), fail with the exit code and the server's own log tail if
    it dies early, and kill only our own process if it never becomes
    healthy.
*   Log the server's output to ``logs/llama-server.log`` so a failure is
    readable after the window is gone.

Management is opt-in per process through ``PRIMORA_MODEL_SERVER``:

``auto`` (default)
    Manage the local server when it is the endpoint Aurora will actually
    use: the selected backend is ``openai`` and ``PRIMORA_OPENAI_URL`` is
    unset or points at this module's host:port. An explicitly remote
    endpoint, or an explicit ``transformers`` backend, is left alone.
``on``
    Always ensure the server is up, whatever the backend selection is.
``off``
    Never probe, start, or stop anything; the operator runs the server
    (or the endpoint) themselves. Zero I/O.

Environment (all optional; defaults select Qwen3.5 9B vision):

``PRIMORA_LLAMACPP_SERVER``   llama-server executable
``PRIMORA_LLAMACPP_MODEL``    GGUF model file (defaults to project Qwen3.5 9B)
``PRIMORA_LLAMACPP_MMPROJ``   matching multimodal projector GGUF (default for project model)
``PRIMORA_MODEL_SERVER_PORT`` port (default 8080; host is 127.0.0.1)
``PRIMORA_MODEL_SERVER_WAIT`` readiness wait in seconds (default 180)
``PRIMORA_MODEL_SERVER_CTX``  context size passed as -c (default 4096)
``PRIMORA_MODEL_SERVER_ALIAS`` served model alias (default qwen3.5-9b)
``PRIMORA_MODEL_SERVER_KEEP`` 1/true/yes/on leaves a started server running
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# The default model assets live in the project; operator-provided paths
# still override both through environment variables.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_EXE = Path(
    r"C:\Users\M\Downloads\Llama"
    r"\llama-b10615-bin-win-cuda-12.4-x64\llama-server.exe"
)
DEFAULT_MODEL_PATH = (
    _PROJECT_ROOT / "RESOURCES" / "aurora" / "qwen3.5-9b"
    / "Qwen3.5-9B-Q4_K_M.gguf"
)
DEFAULT_MMPROJ_PATH = (
    _PROJECT_ROOT / "RESOURCES" / "aurora" / "qwen3.5-9b"
    / "mmproj-Qwen3.5-9B-f16.gguf"
)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_CTX = 4096
DEFAULT_NGL = "99"
DEFAULT_ALIAS = "qwen3.5-9b"
DEFAULT_WAIT_SECONDS = 180
DEFAULT_TIMEOUT_SECONDS = "300"

DEFAULT_LOG_PATH = _PROJECT_ROOT / "logs" / "llama-server.log"

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class ModelServerError(RuntimeError):
    """The model server could not be brought up; the message is actionable."""


@dataclass(frozen=True)
class ModelServerStatus:
    """What happened when the endpoint was ensured."""

    state: str  # "adopted" | "started" | "unmanaged"
    base_url: str
    pid: Optional[int] = None


def health(host: str, port: int, timeout: float = 2.0) -> bool:
    """True when llama-server answers /health with 200 (model ready).

    A 503 means the server is up but still loading the model, which is
    deliberately *not* healthy yet.
    """
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _served_model_ids(host: str, port: int) -> tuple[str, ...]:
    """Inspect the local llama-server model registry before adopting it.

    Do not silently connect the new Qwen3.5 runtime to an old Heretic server.
    An unreadable model registry fails closed for multimodal deployments.
    """
    url = f"http://{host}:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError, TypeError):
        return ()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return ()
    return tuple(
        item["id"] for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    )


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _url_targets(url: str, host: str, port: int) -> bool:
    """True when ``url`` points at this module's host:port."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = (parsed.hostname or "").lower()
    if hostname not in (host.lower(), "localhost", "127.0.0.1"):
        return False
    default_port = 443 if parsed.scheme == "https" else 80
    return (parsed.port or default_port) == port


def should_manage(
    mode: str,
    backend: str,
    explicit_url: str,
    host: str,
    port: int,
) -> bool:
    """Whether this process owns the model server's lifecycle.

    The decision is made from *explicit* configuration only (before any
    defaults are applied), so an operator who pointed Aurora at a remote
    endpoint never gets a local server spawned underneath them.
    """
    if mode == "off":
        return False
    if mode == "on":
        return True
    # auto
    if backend != "openai":
        return False
    if not explicit_url.strip():
        return True  # no endpoint configured: the local default is ours
    return _url_targets(explicit_url, host, port)


@dataclass
class ModelServerService:
    """Connect-or-start lifecycle for one llama-server endpoint."""

    exe: Path
    model: Path
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    alias: str = DEFAULT_ALIAS
    ctx: int = DEFAULT_CTX
    wait_seconds: int = DEFAULT_WAIT_SECONDS
    keep: bool = False
    mode: str = "auto"
    log_path: Path = DEFAULT_LOG_PATH
    mmproj: Optional[Path] = None
    _proc: Optional[subprocess.Popen] = field(
        default=None, repr=False, compare=False)
    _atexit_registered: bool = field(
        default=False, repr=False, compare=False)

    @classmethod
    def from_env(cls, env=None) -> "ModelServerService":
        source = os.environ if env is None else env

        def _int(name: str, default: int) -> int:
            try:
                return int(str(source.get(name, "")).strip() or default)
            except (TypeError, ValueError):
                return default

        mode = str(source.get("PRIMORA_MODEL_SERVER") or "auto").strip().lower()
        if mode not in ("auto", "on", "off"):
            raise ModelServerError(
                f"unsupported PRIMORA_MODEL_SERVER={mode!r}; "
                "expected 'auto', 'on' or 'off'"
            )
        configured_model = source.get("PRIMORA_LLAMACPP_MODEL")
        configured_mmproj = source.get("PRIMORA_LLAMACPP_MMPROJ")
        return cls(
            exe=Path(
                source.get("PRIMORA_LLAMACPP_SERVER") or DEFAULT_SERVER_EXE
            ),
            model=Path(configured_model or DEFAULT_MODEL_PATH),
            # Custom models opt out of the default projector unless supplied
            # explicitly; a projector from another architecture is unsafe.
            mmproj=(Path(configured_mmproj) if configured_mmproj
                    else DEFAULT_MMPROJ_PATH if not configured_model else None),
            host=DEFAULT_HOST,
            port=_int("PRIMORA_MODEL_SERVER_PORT", DEFAULT_PORT),
            alias=str(
                source.get("PRIMORA_MODEL_SERVER_ALIAS") or DEFAULT_ALIAS
            ),
            ctx=_int("PRIMORA_MODEL_SERVER_CTX", DEFAULT_CTX),
            wait_seconds=_int(
                "PRIMORA_MODEL_SERVER_WAIT", DEFAULT_WAIT_SECONDS
            ),
            keep=_truthy(str(source.get("PRIMORA_MODEL_SERVER_KEEP") or "")),
            mode=mode,
            log_path=Path(
                source.get("PRIMORA_MODEL_SERVER_LOG") or DEFAULT_LOG_PATH
            ),
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _missing(self, what: str, path: Path, hint: str) -> ModelServerError:
        return ModelServerError(
            f"MISSING: {what}\n"
            f"  Expected at: {path}\n"
            f"  {hint}\n"
            "  Nothing was downloaded or installed. Place the file and "
            "re-run."
        )

    def _log_tail(self, lines: int = 12) -> str:
        try:
            content = self.log_path.read_text(
                encoding="utf-8", errors="replace")
        except OSError:
            return ""
        tail = content.splitlines()[-lines:]
        return "\n".join(tail)

    def ensure(self) -> ModelServerStatus:
        """Adopt a healthy server or start one; never download anything."""
        if health(self.host, self.port):
            if self.mmproj is not None:
                ids = _served_model_ids(self.host, self.port)
                if self.alias not in ids:
                    raise ModelServerError(
                        f"A different or unverifiable llama-server is already "
                        f"running on port {self.port}. Expected alias "
                        f"{self.alias!r}; detected {ids or 'unknown'}. "
                        "Stop the existing llama-server and retry. "
                        "PRIMORA will not stop an adopted process."
                    )
            return ModelServerStatus(
                state="adopted", base_url=self.base_url)

        if not self.exe.is_file():
            raise self._missing(
                "llama-server binary (llama.cpp CUDA build)",
                self.exe,
                "Set PRIMORA_LLAMACPP_SERVER to the executable.",
            )
        if not self.model.is_file():
            raise self._missing(
                "GGUF model file",
                self.model,
                "Set PRIMORA_LLAMACPP_MODEL to the GGUF file.",
            )
        if self.mmproj is not None and not self.mmproj.is_file():
            raise self._missing(
                "multimodal projector GGUF",
                self.mmproj,
                "Set PRIMORA_LLAMACPP_MMPROJ to the matching projector file.",
            )

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        args = [
            str(self.exe),
            "-m", str(self.model),
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.ctx),
            "-ngl", DEFAULT_NGL,
            "--alias", self.alias,
        ]
        if self.mmproj is not None:
            args.extend(["--mmproj", str(self.mmproj), "--jinja"])
        with open(self.log_path, "ab") as log_f:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                creationflags=_NO_WINDOW,
            )
        self._proc = proc
        if not self._atexit_registered:
            atexit.register(self._atexit_stop)
            self._atexit_registered = True

        deadline = time.monotonic() + self.wait_seconds
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                tail = self._log_tail()
                raise ModelServerError(
                    f"llama-server exited early (exit code {proc.returncode}). "
                    f"Check GPU/CUDA availability and that port {self.port} "
                    f"is free. Log: {self.log_path}\n{tail}"
                )
            if health(self.host, self.port):
                return ModelServerStatus(
                    state="started", base_url=self.base_url, pid=proc.pid
                )
            time.sleep(1.0)

        # Never healthy within the window: stop only what we started.
        self._terminate()
        tail = self._log_tail()
        raise ModelServerError(
            f"llama-server did not become healthy within "
            f"{self.wait_seconds}s. Log: {self.log_path}\n{tail}"
        )

    def _terminate(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _atexit_stop(self) -> None:
        # Backstop for exits that never reach main()'s finally.
        self.stop()

    def stop(self) -> None:
        """Stop only a server this module started, and only if not kept."""
        if self.keep:
            if self._proc is not None and self._proc.poll() is None:
                print(
                    f"Leaving llama-server running (PID {self._proc.pid}; "
                    "PRIMORA_MODEL_SERVER_KEEP is set)."
                )
            return
        if self._proc is not None and self._proc.poll() is None:
            print(f"Stopping llama-server (PID {self._proc.pid})...")
            self._terminate()
