"""Phase 2C: automatic SearXNG service lifecycle, without a container runtime.

Everything here runs against injected probes, runners, and runtimes: no
Docker, no Podman, no network, no live instance. The properties under test
are ownership (adopted vs started), fail-closed configuration, fixed
argument lists, idempotent stop, and - through ``main()`` - that CLI exit
and GUI close stop what this process started while every startup failure
still lets Aurora launch.
"""

import os
import re
import sys
import tempfile

import pytest
import yaml

sys.path.insert(0, "C:\\PRIMORA PROJECT")

import PRIMORA.main as main_mod
import PRIMORA.research.service as service_mod
from PRIMORA.aurora import Aurora
from PRIMORA.research import SEARXNG_ENDPOINT, SearXNGService
from PRIMORA.research.service import COMPOSE_FILE, PROJECT_NAME

SETTINGS_FILE = os.path.join(os.path.dirname(COMPOSE_FILE), "settings.yml")


class RecordingRunner:
    """Stands in for the container runner and records every invocation."""

    def __init__(self, fail=None):
        self.calls = []
        self._fail = fail

    def __call__(self, args, env=None, timeout=None):
        self.calls.append({"args": list(args), "env": env, "timeout": timeout})
        if self._fail is not None:
            raise self._fail


class SequencedProbe:
    """Answers readiness probes from a script; the last answer repeats."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, endpoint):
        self.calls.append(endpoint)
        index = min(len(self.calls) - 1, len(self.outcomes) - 1)
        return self.outcomes[index]


def _make_service(probe, runner=None, runtime=("/usr/bin/docker",),
                  timeout=0.0, auto_start=True):
    runner = RecordingRunner() if runner is None else runner
    if runtime is None:
        finder = lambda: None
    else:
        finder = lambda: list(runtime)
    service = SearXNGService(
        startup_timeout_s=timeout,
        auto_start=auto_start,
        probe=probe,
        runner=runner,
        runtime_finder=finder,
        sleep=lambda seconds: None,
    )
    return service, runner


def _ups(runner):
    return [c for c in runner.calls if c["args"][-2:] == ["up", "-d"]]


def _stops(runner):
    return [c for c in runner.calls if c["args"][-1] == "stop"]


# ======================================================================
# Adoption: healthy instances belong to someone else
# ======================================================================


class TestAdoption:
    def test_a_healthy_instance_is_adopted_and_never_stopped(self):
        probe = SequencedProbe([True])
        service, runner = _make_service(probe=probe)
        service.start()
        assert service.ownership == "adopted"
        service.stop()
        service.stop()
        assert runner.calls == []

    def test_start_probes_the_fixed_local_endpoint_first(self):
        probe = SequencedProbe([True])
        service, runner = _make_service(probe=probe)
        service.start()
        assert probe.calls == [SEARXNG_ENDPOINT]
        assert service.endpoint == SEARXNG_ENDPOINT

    def test_adoption_happens_even_with_auto_start_disabled(self):
        probe = SequencedProbe([True])
        service, runner = _make_service(probe=probe, auto_start=False)
        service.start()
        service.stop()
        assert service.ownership == "adopted"
        assert runner.calls == []

    def test_an_unhealthy_instance_is_not_adopted(self):
        probe = SequencedProbe([False, True])
        service, runner = _make_service(probe=probe, timeout=5.0)
        service.start()
        assert service.ownership == "started"


# ======================================================================
# Auto-start: opt-in, never forced
# ======================================================================


class TestAutoStart:
    def test_disabled_auto_start_never_starts_a_service(self, capsys):
        probe = SequencedProbe([False])
        service, runner = _make_service(probe=probe, auto_start=False)
        service.start()
        service.stop()
        assert runner.calls == []
        assert service.ownership is None
        assert "WARNING" in capsys.readouterr().out

    def test_missing_runtime_warns_without_blocking(self, capsys):
        probe = SequencedProbe([False])
        service, runner = _make_service(probe=probe, runtime=None)
        service.start()
        service.stop()
        assert runner.calls == []
        assert "container runtime" in capsys.readouterr().out

    def test_startup_failure_warns_without_blocking(self, capsys):
        runner = RecordingRunner(fail=RuntimeError("daemon not running"))
        probe = SequencedProbe([False])
        service, _ = _make_service(probe=probe, runner=runner)
        service.start()
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "daemon not running" in out
        assert service.ownership is None

    def test_port_conflict_warns_without_blocking(self, capsys):
        runner = RecordingRunner(
            fail=RuntimeError(
                "port is already allocated: 127.0.0.1:8888"
            )
        )
        probe = SequencedProbe([False])
        service, _ = _make_service(probe=probe, runner=runner)
        service.start()
        out = capsys.readouterr().out
        assert "already allocated" in out
        assert service.ownership is None
        service.stop()
        assert len(runner.calls) == 1, "a failed start is never 'owned'"

    def test_readiness_timeout_warns_without_blocking(self, capsys):
        probe = SequencedProbe([False])
        service, runner = _make_service(probe=probe, timeout=0.0)
        service.start()
        out = capsys.readouterr().out
        assert "did not become healthy" in out
        # It was started by this process, so shutdown still stops it.
        assert service.ownership == "started"


# ======================================================================
# Ownership: only what this process started gets stopped, exactly once
# ======================================================================


class TestOwnershipAndIdempotence:
    def test_only_an_owned_instance_is_stopped(self):
        probe = SequencedProbe([False, True])
        service, runner = _make_service(probe=probe, timeout=5.0)
        service.start()
        assert service.ownership == "started"
        service.stop()
        assert service.ownership == "stopped"
        assert len(_ups(runner)) == 1
        assert len(_stops(runner)) == 1

    def test_stop_is_idempotent(self):
        probe = SequencedProbe([False, True])
        service, runner = _make_service(probe=probe, timeout=5.0)
        service.start()
        for _ in range(5):
            service.stop()
        assert len(_stops(runner)) == 1

    def test_repeated_lifecycle_calls_do_not_duplicate_startup(self):
        probe = SequencedProbe([False, True])
        service, runner = _make_service(probe=probe, timeout=5.0)
        service.start()
        service.start()
        service.start()
        assert len(_ups(runner)) == 1
        service.stop()
        service.stop()
        assert len(_stops(runner)) == 1

    def test_start_after_stop_does_nothing(self):
        probe = SequencedProbe([False, True])
        service, runner = _make_service(probe=probe, timeout=5.0)
        service.start()
        service.stop()
        service.start()
        assert len(_ups(runner)) == 1
        assert len(_stops(runner)) == 1

    def test_a_never_started_service_stops_as_a_no_op(self):
        probe = SequencedProbe([False])
        service, runner = _make_service(probe=probe, auto_start=False)
        service.stop()
        assert runner.calls == []
        assert service.ownership is None


# ======================================================================
# Fixed argument lists: no shell, no interpolation, no daemon launching
# ======================================================================


class TestCommandSafety:
    @pytest.mark.parametrize(
        "runtime",
        [("/usr/bin/docker",), ("C:/Program Files/Docker/docker.exe",),
         ("/usr/bin/podman",)],
    )
    def test_arguments_are_a_fixed_validated_list(self, runtime):
        probe = SequencedProbe([False, True])
        service, runner = _make_service(
            probe=probe, runtime=runtime, timeout=5.0
        )
        service.start()
        up = _ups(runner)[0]["args"]
        assert up == [
            runtime[0], "compose", "-p", PROJECT_NAME, "-f", COMPOSE_FILE,
            "up", "-d",
        ]
        service.stop()
        stop = _stops(runner)[0]["args"]
        assert stop == [
            runtime[0], "compose", "-p", PROJECT_NAME, "-f", COMPOSE_FILE,
            "stop",
        ]

    def test_only_compose_verbs_are_ever_executed(self):
        probe = SequencedProbe([False, True])
        service, runner = _make_service(probe=probe, timeout=5.0)
        service.start()
        service.stop()
        for call in runner.calls:
            args = call["args"]
            assert args[1] == "compose"
            assert args[6:] in (["up", "-d"], ["stop"])
            # Nothing that would launch Docker Desktop or Podman itself.
            assert "start" not in args
            assert "run" not in args

    def test_the_start_environment_carries_a_generated_secret_only(
            self):
        probe = SequencedProbe([False, True])
        service, runner = _make_service(probe=probe, timeout=5.0)
        service.start()
        env = _ups(runner)[0]["env"]
        secret = env[service_mod.SEARXNG_SECRET_ENV]
        assert re.fullmatch(r"[0-9a-f]{64}", secret)
        assert env.get("PATH") == os.environ.get("PATH")
        tracked = (
            open(COMPOSE_FILE, encoding="utf-8").read()
            + open(SETTINGS_FILE, encoding="utf-8").read()
        )
        assert secret not in tracked, "the runtime secret leaked into git"

    def test_the_provider_source_never_executes_anything(self):
        src = open(
            os.path.join("PRIMORA", "research", "service.py"),
            encoding="utf-8",
        ).read()
        for banned in (
            "shell=True", "os.system", "os.popen", "creationflags",
            "check_output", "Popen(", "sqlite", "aurora",
        ):
            assert banned not in src, f"unsafe execution: {banned}"

    def test_runtime_discovery_validates_the_binary_name(self, monkeypatch):
        monkeypatch.setattr(
            "shutil.which",
            lambda name: f"C:/tools/{name}-wrapper.exe",
        )
        assert service_mod._find_container_runtime() is None
        monkeypatch.setattr(
            "shutil.which",
            lambda name: (
                "C:/bin/docker.exe" if name == "docker" else None
            ),
        )
        assert service_mod._find_container_runtime() == [
            "C:/bin/docker.exe"
        ]
        monkeypatch.setattr(
            "shutil.which",
            lambda name: "/usr/bin/podman" if name == "podman" else None,
        )
        assert service_mod._find_container_runtime() == ["/usr/bin/podman"]


# ======================================================================
# Configuration parsing: pure, no I/O, fail-closed
# ======================================================================


class TestEnvironmentParsing:
    def test_defaults(self):
        service = SearXNGService.from_env(env={})
        assert service.startup_timeout_s == 30.0
        assert service.auto_start is True

    def test_timeout_override(self):
        service = SearXNGService.from_env(
            env={"PRIMORA_RESEARCH_STARTUP_TIMEOUT": "5"}
        )
        assert service.startup_timeout_s == 5.0

    @pytest.mark.parametrize(
        "bad", ["abc", "-3", "0", "  ", "nan", "-nan", "inf", "1e999"])
    def test_invalid_timeout_falls_back(self, bad, capsys):
        service = SearXNGService.from_env(
            env={"PRIMORA_RESEARCH_STARTUP_TIMEOUT": bad}
        )
        if bad.strip():
            assert "WARNING" in capsys.readouterr().out
        assert service.startup_timeout_s == 30.0

    def test_timeout_is_bounded_by_the_container_maximum(self, capsys):
        service = SearXNGService.from_env(
            env={"PRIMORA_RESEARCH_STARTUP_TIMEOUT": "1200"}
        )
        assert service.startup_timeout_s == 600.0
        assert "WARNING" in capsys.readouterr().out

    def test_a_configured_value_below_the_maximum_is_honoured(self):
        service = SearXNGService.from_env(
            env={"PRIMORA_RESEARCH_STARTUP_TIMEOUT": "45"}
        )
        assert service.startup_timeout_s == 45.0

    @pytest.mark.parametrize("value", ["0", "false", "no", "OFF"])
    def test_auto_start_can_be_disabled(self, value):
        service = SearXNGService.from_env(
            env={"PRIMORA_SEARXNG_AUTO_START": value}
        )
        assert service.auto_start is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
    def test_auto_start_enabled_values(self, value):
        service = SearXNGService.from_env(
            env={"PRIMORA_SEARXNG_AUTO_START": value}
        )
        assert service.auto_start is True

    def test_unrecognised_auto_start_fails_closed(self, capsys):
        service = SearXNGService.from_env(
            env={"PRIMORA_SEARXNG_AUTO_START": "banana"}
        )
        assert service.auto_start is False
        assert "WARNING" in capsys.readouterr().out

    def test_selection_and_parsing_perform_no_io(self, monkeypatch):
        def _no_io(*args, **kwargs):
            raise AssertionError("service configuration reached I/O")

        monkeypatch.setattr("urllib.request.urlopen", _no_io)
        monkeypatch.setattr("subprocess.run", _no_io)
        monkeypatch.setattr("subprocess.Popen", _no_io)
        service = SearXNGService.from_env(
            env={"PRIMORA_RESEARCH_STARTUP_TIMEOUT": "30"}
        )
        assert service.endpoint == SEARXNG_ENDPOINT

    def test_an_atexit_backstop_is_registered_on_start(self, monkeypatch):
        registered = []
        monkeypatch.setattr(service_mod.atexit, "register",
                            registered.append)
        probe = SequencedProbe([True])
        service, _ = _make_service(probe=probe)
        service.start()
        assert registered, "shutdown backstop missing"
        assert registered[0].__func__ is SearXNGService.stop
        assert registered[0].__self__ is service


# ======================================================================
# Startup budget: the compose call is bounded by the configured timeout
# ======================================================================


class TestComposeTimeoutBudget:
    @staticmethod
    def _service(env, probe, runner):
        return SearXNGService.from_env(
            env=env,
            probe=probe,
            runner=runner,
            runtime_finder=lambda: ["/usr/bin/docker"],
            sleep=lambda seconds: None,
        )

    def test_the_default_startup_never_waits_600_seconds(self):
        probe = SequencedProbe([False, True])
        runner = RecordingRunner()
        service = self._service({}, probe, runner)
        service.start()
        assert service.startup_timeout_s == 30.0
        assert _ups(runner)[0]["timeout"] == 30.0, (
            "an ordinary startup path waited on the 600s hard cap"
        )
        assert service_mod.CONTAINER_RUN_TIMEOUT_S == 600.0

    def test_an_explicitly_configured_value_binds_the_compose_call(self):
        probe = SequencedProbe([False, True])
        runner = RecordingRunner()
        service = self._service(
            {"PRIMORA_RESEARCH_STARTUP_TIMEOUT": "45"}, probe, runner)
        service.start()
        assert _ups(runner)[0]["timeout"] == 45.0

    def test_the_600_second_cap_is_the_maximum_for_configured_values(self):
        probe = SequencedProbe([False, True])
        runner = RecordingRunner()
        service = self._service(
            {"PRIMORA_RESEARCH_STARTUP_TIMEOUT": "1200"}, probe, runner)
        service.start()
        assert service.startup_timeout_s == 600.0
        assert _ups(runner)[0]["timeout"] == 600.0

    def test_stop_uses_the_same_bounded_budget(self):
        probe = SequencedProbe([False, True])
        runner = RecordingRunner()
        service = self._service({}, probe, runner)
        service.start()
        service.stop()
        assert _stops(runner)[0]["timeout"] == 30.0

    def test_the_runner_default_is_the_hard_cap_only(self):
        import inspect
        params = inspect.signature(service_mod._run_container).parameters
        assert params["timeout"].default == (
            service_mod.CONTAINER_RUN_TIMEOUT_S)


# ======================================================================
# Service configuration: committed files say exactly what they should
# ======================================================================


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class TestServiceConfiguration:
    def test_the_compose_file_exists_where_the_service_expects_it(self):
        assert os.path.isfile(COMPOSE_FILE)
        assert os.path.isfile(SETTINGS_FILE)

    def test_the_host_port_binds_loopback_only(self):
        compose = _load(COMPOSE_FILE)
        ports = compose["services"]["searxng"]["ports"]
        assert ports == ["127.0.0.1:8888:8080"]

    def test_the_image_is_pinned_never_latest(self):
        compose = _load(COMPOSE_FILE)
        image = compose["services"]["searxng"]["image"]
        repository, tag = image.split(":", 1)
        assert repository == "searxng/searxng"
        assert tag != "latest"
        assert re.match(r"^\d{4}\.\d+\.\d+", tag), f"unpinned tag: {tag}"

    def test_the_secret_is_a_reference_not_a_value(self):
        compose = _load(COMPOSE_FILE)
        environment = compose["services"]["searxng"]["environment"]
        assert environment["SEARXNG_SECRET"] == "${SEARXNG_SECRET}"
        settings = _load(SETTINGS_FILE)
        assert "secret_key" not in settings.get("server", {})

    def test_settings_enable_json_output(self):
        settings = _load(SETTINGS_FILE)
        assert "json" in settings["search"]["formats"]
        assert "html" in settings["search"]["formats"]

    def test_settings_are_local_only(self):
        settings = _load(SETTINGS_FILE)
        assert settings["server"]["public_instance"] is False
        assert settings["server"]["limiter"] is False

    def test_settings_are_mounted_read_only_from_the_repo(self):
        compose = _load(COMPOSE_FILE)
        volumes = compose["services"]["searxng"]["volumes"]
        assert volumes == ["./settings.yml:/etc/searxng/settings.yml:ro"]

    def test_the_project_name_is_fixed(self):
        compose = _load(COMPOSE_FILE)
        assert compose["name"] == PROJECT_NAME == "primora-searxng"


# ======================================================================
# main(): the lifecycle is part of startup, for CLI and GUI alike
# ======================================================================


class RecordingProvider:
    def __init__(self, model, base_url=None, timeout=None):
        # Mirrors the attribute surface of the real providers
        # (OllamaProvider / OpenAICompatibleProvider) that main() reads
        # for its startup log line.
        self._model = model
        self._base_url = base_url
        self._timeout = timeout

    def invoke(self, capability, payload):
        from PRIMORA.prime.contract import PrimeResult
        return PrimeResult.failure("unused in startup tests")


def _run_main(monkeypatch, argv, env, probe_outcomes, runner=None,
              runtime=("/usr/bin/docker",), warmup=None):
    model_dir = tempfile.mkdtemp()
    adapter_dir = tempfile.mkdtemp()
    values = {
        "PRIMORA_AURORA_MODEL_PATH": model_dir,
        "PRIMORA_AURORA_ADAPTER_PATH": adapter_dir,
        **env,
    }
    managed = (
        "PRIMORA_RESEARCH_PROVIDER",
        "PRIMORA_RESEARCH_ENDPOINT",
        "PRIMORA_SEARXNG_AUTO_START",
        "PRIMORA_RESEARCH_STARTUP_TIMEOUT",
        "PRIMORA_PROVIDER",
        "PRIMORA_OLLAMA_MODEL",
        "PRIMORA_OLLAMA_URL",
        "PRIMORA_OLLAMA_TIMEOUT",
        "PRIMORA_OPENAI_MODEL",
        "PRIMORA_OPENAI_URL",
        "PRIMORA_OPENAI_TIMEOUT",
        "PRIMORA_AURORA_MODEL_PATH",
        "PRIMORA_AURORA_ADAPTER_PATH",
    )
    for key in managed:
        if key in values:
            monkeypatch.setenv(key, values[key])
        else:
            monkeypatch.delenv(key, raising=False)

    probe = SequencedProbe(probe_outcomes)
    runner = RecordingRunner() if runner is None else runner
    monkeypatch.setattr(service_mod, "_probe_endpoint", probe)
    monkeypatch.setattr(service_mod, "_run_container", runner)
    if runtime is None:
        monkeypatch.setattr(service_mod, "_find_container_runtime",
                            lambda: None)
    else:
        monkeypatch.setattr(service_mod, "_find_container_runtime",
                            lambda: list(runtime))

    calls = {"run_gui": [], "run_cli": [], "serve": []}
    monkeypatch.setattr(main_mod, "OllamaProvider", RecordingProvider)
    monkeypatch.setattr(main_mod, "serve",
                        lambda *a, **k: calls["serve"].append((a, k)))
    monkeypatch.setattr(main_mod, "run_gui",
                        lambda aurora: calls["run_gui"].append(aurora) or 0)
    monkeypatch.setattr(main_mod, "run_cli",
                        lambda aurora: calls["run_cli"].append(aurora))
    monkeypatch.setattr(Aurora, "warmup",
                        warmup or (lambda self: None))
    monkeypatch.setattr(sys, "argv", list(argv))
    try:
        outcome = main_mod.main()
    except SystemExit as exc:
        outcome = ("exit", exc.code)
    return outcome, calls, probe, runner


_SEARXNG_ENV = {
    "PRIMORA_RESEARCH_PROVIDER": "searxng",
    "PRIMORA_RESEARCH_ENDPOINT": SEARXNG_ENDPOINT,
}


class TestStartupIntegration:
    def test_cli_exit_stops_an_owned_service(self, monkeypatch, capsys):
        outcome, calls, probe, runner = _run_main(
            monkeypatch, ["main.py", "--cli"],
            {"PRIMORA_OLLAMA_MODEL": "custom:model", **_SEARXNG_ENV},
            probe_outcomes=[False, True],
        )
        assert outcome == 0
        assert len(calls["run_cli"]) == 1
        assert len(_ups(runner)) == 1
        assert len(_stops(runner)) == 1
        assert probe.calls[0] == SEARXNG_ENDPOINT

    def test_gui_close_stops_an_owned_service(self, monkeypatch):
        outcome, calls, probe, runner = _run_main(
            monkeypatch, ["main.py"], _SEARXNG_ENV,
            probe_outcomes=[False, True],
        )
        assert outcome == 0
        assert len(calls["run_gui"]) == 1
        assert len(_ups(runner)) == 1
        assert len(_stops(runner)) == 1

    def test_an_adopted_service_is_never_stopped_at_exit(
            self, monkeypatch):
        outcome, calls, probe, runner = _run_main(
            monkeypatch, ["main.py"], _SEARXNG_ENV,
            probe_outcomes=[True],
        )
        assert outcome == 0
        assert probe.calls == [SEARXNG_ENDPOINT]
        assert runner.calls == []

    def test_missing_runtime_does_not_block_startup(self, monkeypatch,
                                                    capsys):
        outcome, calls, probe, runner = _run_main(
            monkeypatch, ["main.py"], _SEARXNG_ENV,
            probe_outcomes=[False], runtime=None,
        )
        assert outcome == 0
        assert runner.calls == []
        assert "container runtime" in capsys.readouterr().out

    def test_startup_failure_does_not_block_startup(self, monkeypatch,
                                                    capsys):
        runner = RecordingRunner(fail=RuntimeError("daemon not running"))
        outcome, calls, probe, _ = _run_main(
            monkeypatch, ["main.py"], _SEARXNG_ENV,
            probe_outcomes=[False], runner=runner,
        )
        assert outcome == 0
        assert len(runner.calls) == 1
        assert "WARNING" in capsys.readouterr().out

    def test_port_conflict_does_not_block_startup(self, monkeypatch,
                                                  capsys):
        runner = RecordingRunner(
            fail=RuntimeError("port is already allocated: 127.0.0.1:8888")
        )
        outcome, calls, probe, _ = _run_main(
            monkeypatch, ["main.py"], _SEARXNG_ENV,
            probe_outcomes=[False], runner=runner,
        )
        assert outcome == 0
        assert len(runner.calls) == 1
        assert "already allocated" in capsys.readouterr().out

    def test_readiness_timeout_does_not_block_startup(self, monkeypatch,
                                                      capsys):
        outcome, calls, probe, runner = _run_main(
            monkeypatch, ["main.py"],
            {**_SEARXNG_ENV, "PRIMORA_RESEARCH_STARTUP_TIMEOUT": "0.05"},
            probe_outcomes=[False],
        )
        assert outcome == 0
        out = capsys.readouterr().out
        assert "did not become healthy" in out
        assert len(_ups(runner)) == 1
        assert len(_stops(runner)) == 1

    def test_disabled_auto_start_never_starts_a_service(
            self, monkeypatch, capsys):
        outcome, calls, probe, runner = _run_main(
            monkeypatch, ["main.py"],
            {**_SEARXNG_ENV, "PRIMORA_SEARXNG_AUTO_START": "0"},
            probe_outcomes=[False],
        )
        assert outcome == 0
        assert runner.calls == []
        assert "WARNING" in capsys.readouterr().out

    def test_unconfigured_research_never_touches_the_service(
            self, monkeypatch):
        outcome, calls, probe, runner = _run_main(
            monkeypatch, ["main.py"], {}, probe_outcomes=[True],
        )
        assert outcome == 0
        assert probe.calls == []
        assert runner.calls == []

    def test_a_warmup_failure_exits_with_the_backstop_registered(
            self, monkeypatch):
        registered = []
        monkeypatch.setattr(service_mod.atexit, "register",
                            registered.append)
        outcome, calls, probe, runner = _run_main(
            monkeypatch, ["main.py", "--cli"],
            {"PRIMORA_OLLAMA_MODEL": "custom:model", **_SEARXNG_ENV},
            probe_outcomes=[False, True],
            warmup=lambda self: "model failed to load",
        )
        assert outcome == ("exit", 1)
        assert len(_ups(runner)) == 1
        assert _stops(runner) == [], "the finally path is never reached"
        assert registered, "no atexit backstop for early exits"
        assert registered[0].__self__.ownership == "started"

    def test_finally_and_atexit_together_stop_only_once(self, monkeypatch):
        registered = []
        monkeypatch.setattr(service_mod.atexit, "register",
                            registered.append)
        outcome, calls, probe, runner = _run_main(
            monkeypatch, ["main.py", "--cli"],
            {"PRIMORA_OLLAMA_MODEL": "custom:model", **_SEARXNG_ENV},
            probe_outcomes=[False, True],
        )
        assert outcome == 0
        assert len(_stops(runner)) == 1, "the finally path stopped it"
        registered[0]()  # the atexit backstop fires afterwards
        assert len(_stops(runner)) == 1, "double stop through atexit"

    def test_an_initialization_failure_stops_the_started_service(
            self, monkeypatch):
        registered = []
        monkeypatch.setattr(service_mod.atexit, "register",
                            registered.append)

        class Uninitialisable:
            def __init__(self, prime):
                raise RuntimeError("initialization exploded")

        monkeypatch.setattr(main_mod, "Aurora", Uninitialisable)
        outcome, calls, probe, runner = _run_main(
            monkeypatch, ["main.py", "--cli"],
            {"PRIMORA_OLLAMA_MODEL": "custom:model", **_SEARXNG_ENV},
            probe_outcomes=[False, True],
        )
        assert outcome == ("exit", 1)
        assert len(_ups(runner)) == 1
        assert len(_stops(runner)) == 1, "the except branch skipped stop"
        assert registered[0].__self__.ownership == "stopped"
