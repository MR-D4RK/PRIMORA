"""Automatic lifecycle for the local SearXNG research service.

Milestone 3, Phase 2C. The research *provider* talks to SearXNG; this module
decides whether an instance exists at all. It lives in the composition layer
(so both the CLI and the GUI paths get it) and never inside the provider
itself: a provider that starts containers would make ``search()`` a
subprocess launcher and make selection tests do I/O.

The contract, in order:

1. Probe the fixed local endpoint first. A healthy instance is *adopted*:
   someone else owns it, so PRIMORA never stops it. This is what makes a
   forced kill on a previous launch safe - the next launch simply adopts.
2. Unhealthy and auto-start enabled: start it with a fixed, validated
   ``docker``/``podman compose`` argument list - a list, never a shell
   string, so no model output, user text, or configuration value can become
   command syntax. The runtime must already be running; starting Docker
   Desktop or Podman itself is out of scope.
3. Wait up to ``PRIMORA_RESEARCH_STARTUP_TIMEOUT`` (default 30s) for
   readiness. Every failure mode - missing runtime, refused start, port
   conflict, timeout - warns and returns. Aurora always launches.
4. On shutdown, stop only an instance this process started, exactly once.
   Stopping is idempotent; adopted or never-started services are no-ops.

No secret is ever written to a tracked file: the runtime secret is generated
per start and passed through the subprocess environment, where SearXNG reads
it as ``SEARXNG_SECRET``.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import secrets
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from typing import Callable, Optional, Sequence

from PRIMORA.research.searxng import SEARXNG_ENDPOINT

# Configuration (see research_provider_from_env for provider selection).
STARTUP_TIMEOUT_ENV = "PRIMORA_RESEARCH_STARTUP_TIMEOUT"
AUTO_START_ENV = "PRIMORA_SEARXNG_AUTO_START"
DEFAULT_STARTUP_TIMEOUT_S = 30.0

# SearXNG reads its secret from this environment variable; the compose file
# forwards it into the container. Generated at start, never committed.
SEARXNG_SECRET_ENV = "SEARXNG_SECRET"

# Fixed, deterministic service identity: compose project name, compose file,
# and endpoint are constants, so every launch manages the same instance.
PROJECT_NAME = "primora-searxng"
COMPOSE_FILE = os.path.join(
    os.path.dirname(                      # .../PRIMORA/research
        os.path.dirname(                  # .../PRIMORA
            os.path.dirname(os.path.abspath(__file__)),
        ),
    ),
    "RESOURCES",
    "searxng",
    "docker-compose.yml",
)

# Readiness probing. The probe is a real JSON search (the only way to tell
# SearXNG from a foreign service on the port), so its timeout must cover a
# genuine round trip. 10s matches the research provider's own request
# budget; a tighter value misclassifies a healthy-but-slow instance (engines
# rate-limiting upstream) as down, which previously made a second process
# claim ownership of a running container and stop it on exit.
PROBE_TIMEOUT_S = 10.0
PROBE_POLL_INTERVAL_S = 0.5

# Hard maximum for a compose invocation (image pull, daemon hang). The
# effective bound for any call is min(configured startup timeout, this),
# so 600s is only reachable through an explicitly configured value and an
# ordinary startup path waits the 30s default instead.
CONTAINER_RUN_TIMEOUT_S = 600.0

# Ownership states. None means "not resolved yet".
OWNERSHIP_ADOPTED = "adopted"
OWNERSHIP_STARTED = "started"
OWNERSHIP_STOPPED = "stopped"


def _probe_endpoint(endpoint: str) -> bool:
    """Whether a SearXNG instance answers with valid search JSON here.

    Any transport, HTTP, timeout, or decode failure means unhealthy - the
    probe is a yes/no about readiness, so a broken probe never crashes the
    caller. A foreign service squatting on the port will not answer with a
    SearXNG-shaped payload, which is exactly the distinction we need before
    adopting versus starting.
    """
    params = urllib.parse.urlencode(
        {"q": "primora searxng readiness probe", "format": "json"}
    )
    request = urllib.request.Request(
        f"{endpoint}/search?{params}",
        headers={"User-Agent": "PRIMORA/1.0 (+research service probe)"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_S) as response:
            raw = response.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
    except Exception:
        return False
    return isinstance(payload, dict) and "results" in payload


def _run_container(
    args: Sequence[str],
    env: Optional[dict] = None,
    timeout: float = CONTAINER_RUN_TIMEOUT_S,
) -> None:
    """Run one fixed container command; raise with detail if it fails.

    ``shell`` is never used: arguments are a list, so nothing interpolated
    can be interpreted as syntax. Output is captured so a failure can be
    reported honestly instead of disappearing into the void. The default
    ``timeout`` is only the hard maximum; callers pass the configured
    startup budget, which is smaller unless someone raised it explicitly.
    """
    completed = subprocess.run(
        list(args),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-500:]
        raise RuntimeError(
            f"{os.path.basename(args[0])!r} exited with code "
            f"{completed.returncode}: {detail}"
        )


def _find_container_runtime() -> Optional[list[str]]:
    """Locate a supported container runtime, or None.

    The basename is re-checked against an allowlist so anything exotic that
    happens to answer to ``docker`` on PATH is rejected. Only discovery
    happens here - this never launches the runtime's GUI or daemon.
    """
    for name in ("docker", "podman"):
        path = shutil.which(name)
        if not path:
            continue
        base = os.path.basename(path).lower()
        if base in (name, name + ".exe"):
            return [path]
    return None


def _startup_timeout_from(source) -> float:
    raw = source.get(STARTUP_TIMEOUT_ENV)
    if raw is None or not str(raw).strip():
        return DEFAULT_STARTUP_TIMEOUT_S
    try:
        value = float(str(raw).strip())
    except ValueError:
        value = -1.0
    # Non-finite values (nan/inf parse cleanly) would make the readiness
    # deadline unreachable, so they fall back like any other bad input.
    if not math.isfinite(value) or value <= 0:
        print(
            f"WARNING: invalid {STARTUP_TIMEOUT_ENV}={str(raw)!r}; using "
            f"{DEFAULT_STARTUP_TIMEOUT_S:g}s."
        )
        return DEFAULT_STARTUP_TIMEOUT_S
    # Bounded: 600s is the maximum an explicitly configured value may ask
    # for; anything larger is clamped rather than accepted unbounded.
    if value > CONTAINER_RUN_TIMEOUT_S:
        print(
            f"WARNING: {STARTUP_TIMEOUT_ENV}={value:g}s exceeds the "
            f"{CONTAINER_RUN_TIMEOUT_S:g}s maximum; using "
            f"{CONTAINER_RUN_TIMEOUT_S:g}s."
        )
        return CONTAINER_RUN_TIMEOUT_S
    return value


def _auto_start_from(source) -> bool:
    raw = source.get(AUTO_START_ENV)
    if raw is None or not str(raw).strip():
        return True
    value = str(raw).strip().lower()
    if value in ("0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    # Unrecognised input fails closed: never start a service on a setting
    # nobody can have meant. Adoption still works either way.
    print(
        f"WARNING: invalid {AUTO_START_ENV}={str(raw)!r}; auto-start "
        "disabled (fail-closed)."
    )
    return False


class SearXNGService:
    """Lifecycle manager for the fixed local SearXNG instance.

    Dependency-injected by design: ``probe``, ``runner``, ``runtime_finder``,
    ``sleep``, and ``clock`` are all replaceable, which is why the test
    suite can exercise adoption, startup failure, readiness timeout, and
    idempotent stop without Docker, Podman, network access, or a live
    instance.
    """

    def __init__(
        self,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
        auto_start: bool = True,
        *,
        probe: Optional[Callable[[str], bool]] = None,
        # The runner protocol is (args, env, timeout=...): every compose
        # invocation carries its bounded budget explicitly.
        runner: Optional[Callable[..., None]] = None,
        runtime_finder: Optional[Callable[[], Optional[list[str]]]] = None,
        sleep: Optional[Callable[[float], None]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._startup_timeout_s = float(startup_timeout_s)
        self._auto_start = bool(auto_start)
        # Resolved from module globals at construction time so tests can
        # replace the defaults before a service is built (main() builds it).
        self._do_probe = probe if probe is not None else _probe_endpoint
        self._runner = runner if runner is not None else _run_container
        self._runtime_finder = (
            runtime_finder if runtime_finder is not None
            else _find_container_runtime
        )
        self._sleep = sleep if sleep is not None else time.sleep
        self._clock = clock if clock is not None else time.monotonic

        self._start_resolved = False
        self._ownership: Optional[str] = None
        self._runtime: Optional[list[str]] = None
        self._child_env: Optional[dict] = None

    @classmethod
    def from_env(cls, env=None, **injections) -> "SearXNGService":
        """Build the service from PRIMORA_* configuration.

        Performs no I/O: parsing a timeout and a flag is pure. The probe and
        any subprocess happen only when :meth:`start` is called.
        """
        source = os.environ if env is None else env
        return cls(
            startup_timeout_s=_startup_timeout_from(source),
            auto_start=_auto_start_from(source),
            **injections,
        )

    @property
    def endpoint(self) -> str:
        """The one address this service manages. Fixed, never configured."""
        return SEARXNG_ENDPOINT

    @property
    def ownership(self) -> Optional[str]:
        """adopted | started | stopped, or None before :meth:`start`."""
        return self._ownership

    @property
    def auto_start(self) -> bool:
        return self._auto_start

    @property
    def startup_timeout_s(self) -> float:
        return self._startup_timeout_s

    def start(self) -> None:
        """Probe, adopt, or start - at most once, never raising.

        Repeated calls are no-ops: the first call resolves the lifecycle and
        later calls (main's finally, atexit, an operator double-clicking
        both entry paths) cannot duplicate a startup.
        """
        if self._start_resolved:
            return
        self._start_resolved = True
        # Backstop for exits that bypass main()'s finally (sys.exit after a
        # failed warmup, for example). stop() is idempotent, so the normal
        # path running first makes this a no-op.
        atexit.register(self.stop)
        try:
            self._start()
        except Exception as exc:
            # Startup must never keep Aurora from launching, including on an
            # unforeseen failure inside the lifecycle itself.
            print(
                f"WARNING: SearXNG service startup failed unexpectedly "
                f"({exc}); continuing without it."
            )

    def _start(self) -> None:
        if self._safe_probe():
            self._ownership = OWNERSHIP_ADOPTED
            print(
                f"SearXNG already healthy at {SEARXNG_ENDPOINT}; adopting "
                "it (externally owned - PRIMORA will never stop it)."
            )
            return

        if not self._auto_start:
            print(
                f"WARNING: no SearXNG at {SEARXNG_ENDPOINT} and "
                f"{AUTO_START_ENV} disables auto-start; research requests "
                "will fail until an instance is running."
            )
            return

        runtime = self._runtime_finder()
        if not runtime:
            print(
                "WARNING: no supported container runtime (docker/podman) "
                f"found; not starting SearXNG at {SEARXNG_ENDPOINT}."
            )
            return

        secret = secrets.token_hex(32)
        child_env = dict(os.environ)
        child_env[SEARXNG_SECRET_ENV] = secret
        args = self._compose_args(runtime, "up", "-d")
        try:
            self._runner(args, child_env, timeout=self._compose_timeout())
        except Exception as exc:
            print(
                f"WARNING: failed to start SearXNG ({exc}); continuing "
                "without it."
            )
            return
        self._runtime = list(runtime)
        self._child_env = child_env

        if self._wait_ready():
            self._ownership = OWNERSHIP_STARTED
            print(f"Local SearXNG is ready at {SEARXNG_ENDPOINT}.")
        else:
            # It was started by this process, so it stays owned and will be
            # stopped on shutdown - but the failure only warns.
            self._ownership = OWNERSHIP_STARTED
            print(
                f"WARNING: SearXNG started but did not become healthy "
                f"within {self._startup_timeout_s:g}s at "
                f"{SEARXNG_ENDPOINT}; research may be unavailable."
            )

    @staticmethod
    def _compose_args(runtime: Sequence[str], *compose_args: str) -> list[str]:
        """The one argument shape this module ever executes.

        Compose verbs only (``up -d`` / ``stop``), an explicit project name,
        and the tracked compose file - no flags from the environment, no
        strings built by formatting user or model text.
        """
        return [
            *runtime,
            "compose",
            "-p", PROJECT_NAME,
            "-f", COMPOSE_FILE,
            *compose_args,
        ]

    def _safe_probe(self) -> bool:
        try:
            return bool(self._do_probe(SEARXNG_ENDPOINT))
        except Exception:
            return False

    def _compose_timeout(self) -> float:
        """Seconds any compose invocation may take.

        The configured startup budget (default 30s), hard-capped by
        ``CONTAINER_RUN_TIMEOUT_S``: the 600s figure is only a maximum for
        an explicitly configured value, so an ordinary startup path is
        bounded by 30s and can never wait 600s. The 0.1s floor only guards
        pathological programmatic input - environment parsing rejects
        non-positive values before they reach here.
        """
        return max(0.1, min(CONTAINER_RUN_TIMEOUT_S, self._startup_timeout_s))

    def _wait_ready(self) -> bool:
        deadline = self._clock() + self._startup_timeout_s
        while True:
            if self._safe_probe():
                return True
            remaining = deadline - self._clock()
            if remaining <= 0:
                return False
            self._sleep(min(PROBE_POLL_INTERVAL_S, remaining))

    def stop(self) -> None:
        """Stop a started instance, once. Idempotent by construction."""
        if self._ownership != OWNERSHIP_STARTED:
            # Adopted instances belong to someone else; never-started ones
            # have nothing to stop; a stopped service needs no second stop.
            return
        self._ownership = OWNERSHIP_STOPPED
        if not self._runtime:
            return
        try:
            # Same environment as the start, so compose re-reads the file
            # under identical variable substitution.
            self._runner(
                self._compose_args(self._runtime, "stop"),
                self._child_env,
                timeout=self._compose_timeout(),
            )
        except Exception as exc:
            print(
                f"WARNING: failed to stop the SearXNG instance PRIMORA "
                f"started ({exc})."
            )
