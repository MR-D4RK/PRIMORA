"""Milestone 3: the third capability, ``consult_specialist()``.

The properties that matter:

1. Aurora names a specialist and a task; Prime decides model and execution.
2. A specialist is an instrument, not a sub-agent: no identity, no memory,
   no goals, no conversation with Amir.
3. Unknown, unavailable, and failing specialists are reported, never crashed.
4. The Cognitive Loop is unmodified and still bounds the consultation.
"""

import ast
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.consult import SpecialistRegistry, SpecialistSpec
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeRequest, PrimeResult, RequestStatus
from PRIMORA.prime.providers import Provider
from PRIMORA.specialists.tools.consult import ConsultSpecialist

REASONING = SpecialistSpec(
    name="reasoning",
    description="Careful step-by-step analysis of a hard problem.",
    model="qwen3-reasoning",
    system_prompt="Think carefully and precisely.",
)
CODE = SpecialistSpec(
    name="code",
    description="Software design and implementation reasoning.",
    model="qwen3-code",
)


def _request(action, payload):
    return (
        "<prime_request>\n"
        + json.dumps({"action": action, "payload": payload,
                      "context": None, "metadata": None})
        + "\n</prime_request>"
    )


class _FakeProvider(Provider):
    """A fake model backend. Records every call; never touches a model."""

    def __init__(self, content="A considered answer.", error=None):
        self.calls = []
        self._content = content
        self._error = error

    def invoke(self, capability, payload):
        self.calls.append({"capability": capability, "payload": payload})
        if self._error:
            return self._error
        model = payload.get("model", self._content)
        if self._error is None and payload.get("_raise"):
            raise RuntimeError("model backend exploded")
        return PrimeResult.success({"content": self._content, "model": model})


class _Model(Provider):
    """Aurora's own model provider, which must never be consulted."""

    def invoke(self, capability, payload):
        raise AssertionError("Aurora's provider was used for a specialist")


class _Scripted(Brain):
    def __init__(self, *replies):
        self.replies = list(replies) or ["done"]
        self.calls = []

    def generate(self, messages, max_output_tokens=None):
        self.calls.append([dict(m) for m in messages])
        r = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(r, BrainResult):
            return r
        return BrainResult.success(r, "scripted")

    def get_model_name(self):
        return "scripted"

    def count_tokens(self, messages):
        return sum(len(m.get("content", "").split()) + 2 for m in messages)


_DB_SEQ = [0]


def _isolate(prime):
    """Point a Prime's Mind at a throwaway database.

    ``Prime()`` defaults to the *production* ContextManager and
    ``Aurora.respond`` persists, so every Prime used here is redirected
    before anything writes.
    """
    _DB_SEQ[0] += 1
    prime._context = prime._context.__class__(
        db_path=os.path.join(tempfile.mkdtemp(prefix="consult_"),
                            f"c{_DB_SEQ[0]}.db"))
    prime._mind = prime._mind.__class__(prime._context)
    recall = prime.registry.get_specialist("recall")
    if recall is not None:
        recall.bind(prime._mind)
    return prime


def _prime(specialists=None, provider=None):
    prime = Prime(provider or _Model(), specialists=specialists)
    return _isolate(prime)


def _aurora(prime, brain):
    return Aurora(prime, brain=brain,
                  config=AuroraConfig(model_path="m", adapter_path="a"))


def _catalogue():
    return SpecialistRegistry([REASONING, CODE])


def _src(module):
    """Executable source with docstrings removed."""
    import importlib
    path = importlib.import_module(module).__file__
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


# ======================================================================
# 1. The catalogue
# ======================================================================


class TestCatalogue:
    def test_empty_by_default(self):
        assert len(Prime(_Model()).specialists) == 0

    def test_register_and_get(self):
        registry = _catalogue()
        assert registry.has("reasoning")
        assert registry.get("reasoning").model == "qwen3-reasoning"

    def test_duplicate_registration_refused(self):
        registry = SpecialistRegistry([REASONING])
        with pytest.raises(ValueError):
            registry.register(REASONING)

    def test_requires_a_specialist_object(self):
        with pytest.raises(TypeError):
            SpecialistRegistry().register({"name": "x"})

    def test_name_must_be_non_empty(self):
        with pytest.raises(ValueError):
            SpecialistSpec(name="", description="d")

    def test_unavailable_is_still_listed_but_not_offered(self):
        registry = SpecialistRegistry([
            REASONING,
            SpecialistSpec(name="off", description="d", available=False),
        ])
        assert "off" in registry.names()
        assert "off" not in registry.available_names()
        assert [s["name"] for s in registry.describe()] == ["reasoning"]

    def test_known_names_survive_a_broken_registry(self):
        class Broken:
            def describe(self):
                raise RuntimeError("boom")

        class PrimeLike:
            specialists = Broken()

        from PRIMORA.aurora.configuration import build_specialist_instruction
        assert build_specialist_instruction(PrimeLike()) == ""

    def test_from_env_registers_named_specialists(self, monkeypatch):
        monkeypatch.setenv("PRIMORA_SPECIALIST_REASONING_MODEL", "some-model")
        monkeypatch.setenv("PRIMORA_SPECIALIST_REASONING_DESCRIPTION", "thinks")
        registry = SpecialistRegistry.from_env()
        assert "reasoning" in registry
        assert registry.get("reasoning").model == "some-model"

    def test_spec_is_frozen(self):
        with pytest.raises(Exception):
            REASONING.name = "changed"


# ======================================================================
# 2. The capability contract
# ======================================================================


class TestCapabilityContract:
    def test_registered(self):
        assert _prime().registry.has("consult_specialist")

    def test_manifest_shape(self):
        spec = _prime().registry.get("consult_specialist")
        assert spec.required_fields == ["specialist", "task"]
        assert spec.optional_fields == []

    def test_runs_through_prime(self):
        out = _prime(_catalogue(), _FakeProvider()).handle(PrimeRequest(
            "consult_specialist",
            {"specialist": "reasoning", "task": "compare A and B"}))
        assert out.status is RequestStatus.SUCCESS
        assert out.output["result"] == "A considered answer."

    def test_capabilities_registered_so_far(self):
        """Milestone 3 adds capabilities one at a time."""
        assert sorted(_prime().registry.get_names()) == [
            "calculate", "consult_specialist", "learn", "recall", "research",
            "research_read", "use_tool"]

    @pytest.mark.parametrize("bad", [None, "", "   ", 5, [], {}])
    def test_bad_specialist_rejected(self, bad):
        out = _prime(_catalogue(), _FakeProvider()).handle(PrimeRequest(
            "consult_specialist", {"specialist": bad, "task": "t"}))
        assert out.status is RequestStatus.FAILURE

    @pytest.mark.parametrize("bad", [None, "", "   ", 5, [], {}])
    def test_bad_task_rejected(self, bad):
        out = _prime(_catalogue(), _FakeProvider()).handle(PrimeRequest(
            "consult_specialist", {"specialist": "reasoning", "task": bad}))
        assert out.status is RequestStatus.FAILURE

    def test_oversized_task_refused(self):
        out = _prime(_catalogue(), _FakeProvider()).handle(PrimeRequest(
            "consult_specialist",
            {"specialist": "reasoning", "task": "x" * 9000}))
        assert out.status is RequestStatus.FAILURE
        assert "too long" in out.error

    def test_fails_without_a_catalogue(self):
        out = _prime(SpecialistRegistry(), _FakeProvider()).handle(
            PrimeRequest("consult_specialist",
                         {"specialist": "reasoning", "task": "t"}))
        assert out.status is RequestStatus.FAILURE
        assert "none configured" in out.error

    def test_fails_without_a_provider(self):
        capability = ConsultSpecialist(specialists=_catalogue())
        out = capability.handle({"specialist": "reasoning", "task": "t"})
        assert out.status is RequestStatus.FAILURE
        assert "no model provider" in out.error

    def test_rejects_none_provider_and_catalogue(self):
        capability = ConsultSpecialist()
        with pytest.raises(ValueError):
            capability.set_provider(None)
        with pytest.raises(ValueError):
            capability.set_specialists(None)

    def test_aurora_is_told_the_capability_exists(self):
        spec = _prime().registry.get("consult_specialist")
        for banned in ("http", "endpoint", "ollama", "api_key"):
            assert banned not in spec.description.lower()


# ======================================================================
# 3. Routing: the right specialist and model
# ======================================================================


class TestRouting:
    def test_correct_specialist_is_selected(self):
        p = _FakeProvider()
        _prime(_catalogue(), p).handle(PrimeRequest(
            "consult_specialist", {"specialist": "code", "task": "design it"}))
        assert p.calls[0]["capability"] == "code"

    def test_specialist_receives_the_intended_task(self):
        p = _FakeProvider()
        task = "Analyze these two competing database designs."
        _prime(_catalogue(), p).handle(PrimeRequest(
            "consult_specialist", {"specialist": "reasoning", "task": task}))
        user = [m for m in p.calls[0]["payload"]["messages"]
                if m["role"] == "user"]
        assert user[0]["content"] == task

    def test_each_specialist_routes_to_its_own_model(self):
        p = _FakeProvider()
        prime = _prime(_catalogue(), p)
        for name in ("reasoning", "code"):
            prime.handle(PrimeRequest(
                "consult_specialist", {"specialist": name, "task": "t"}))
        assert [c["payload"]["model"] for c in p.calls] == [
            "qwen3-reasoning", "qwen3-code"]

    def test_system_prompt_is_applied(self):
        p = _FakeProvider()
        _prime(_catalogue(), p).handle(PrimeRequest(
            "consult_specialist", {"specialist": "reasoning", "task": "t"}))
        system = [m for m in p.calls[0]["payload"]["messages"]
                  if m["role"] == "system"]
        assert system[0]["content"] == "Think carefully and precisely."

    def test_auroras_own_provider_is_not_used(self):
        """Prime's provider is handed to the capability, not Aurora's brain."""
        p = _FakeProvider()
        prime = _prime(_catalogue(), p)
        brain = _Scripted("hello")
        _aurora(prime, brain).respond(
            [{"role": "user", "content": "hi"}], "s")
        assert p.calls == []

    def test_ollama_provider_defaults_to_its_own_model(self):
        """The payload override must not change existing behaviour."""
        from PRIMORA.prime.providers import OllamaProvider
        captured = {}

        class FakeResp:
            status = 200

            def read(self):
                return json.dumps(
                    {"message": {"content": "ok"}}).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        import urllib.request
        original = urllib.request.urlopen
        urllib.request.urlopen = lambda req, timeout=None: (
            captured.update(model=json.loads(req.data)["model"]) or FakeResp())
        try:
            provider = OllamaProvider("default-model")
            provider.invoke("x", {"messages": [{"role": "user", "content": "hi"}]})
            assert captured["model"] == "default-model"
            provider.invoke("x", {"messages": [{"role": "user", "content": "hi"}],
                                  "model": "override-model"})
            assert captured["model"] == "override-model"
        finally:
            urllib.request.urlopen = original


# ======================================================================
# 4. Inside the cognitive loop
# ======================================================================


class TestInsideTheLoop:
    def _run(self, replies, specialists=None, provider=None, task="t"):
        brain = _Scripted(*replies)
        # An empty registry is falsy (``__len__`` is 0), so ``or`` would
        # silently substitute a populated default. Test emptiness explicitly.
        catalogue = _catalogue() if specialists is None else specialists
        a = _aurora(_prime(catalogue, provider or _FakeProvider()), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "hard problem")
        return brain, a

    def test_aurora_can_request_a_specialist_during_cognition(self):
        brain, a = self._run([
            _request("consult_specialist",
                     {"specialist": "reasoning", "task": "compare A and B"}),
            "A is better because of the locking model.",
        ])
        result = a.respond(
            [{"role": "user", "content": "which database design is better?"}],
            "s")
        assert "locking model" in result.content
        assert len(brain.calls) == 2

    def test_request_reaches_the_capability(self):
        brain, a = self._run([
            _request("consult_specialist",
                     {"specialist": "reasoning", "task": "t"}), "done"])
        a.respond([{"role": "user", "content": "go"}], "s")
        assert [n for n, _ in a.loop.executor.calls] == ["consult_specialist"]

    def test_result_reaches_the_next_segment(self):
        brain, a = self._run([
            _request("consult_specialist",
                     {"specialist": "reasoning", "task": "t"}), "done"])
        a.respond([{"role": "user", "content": "go"}], "s")
        second = " ".join(m["content"] for m in brain.calls[1])
        assert "A considered answer." in second

    def test_aurora_can_answer_using_the_result(self):
        brain, a = self._run([
            _request("consult_specialist",
                     {"specialist": "reasoning",
                      "task": "which is safer, X or Y?"}),
            "The specialist recommends Y, so Y is the safer choice.",
        ])
        result = a.respond(
            [{"role": "user", "content": "X or Y?"}], "s")
        assert "recommends Y" in result.content

    def test_specialist_catalogue_reaches_the_system_prompt(self):
        brain, a = self._run(["done"])
        a.respond([{"role": "user", "content": "hi"}], "s")
        system = brain.calls[0][0]["content"]
        assert "Available Specialists:" in system
        assert "reasoning" in system
        assert "Careful step-by-step analysis" in system

    def test_no_specialists_means_no_promise_in_the_prompt(self):
        brain, a = self._run(["done"], specialists=SpecialistRegistry())
        a.respond([{"role": "user", "content": "hi"}], "s")
        assert "Available Specialists:" not in brain.calls[0][0]["content"]

    def test_prompt_does_not_leak_model_names(self):
        brain, a = self._run(["done"])
        a.respond([{"role": "user", "content": "hi"}], "s")
        system = brain.calls[0][0]["content"]
        for banned in ("qwen3-reasoning", "qwen3-code", "ollama", "http"):
            assert banned not in system


# ======================================================================
# 5. Failure states
# ======================================================================


class TestFailureStates:
    def test_unknown_specialist_fails_cleanly(self):
        out = _prime(_catalogue(), _FakeProvider()).handle(PrimeRequest(
            "consult_specialist", {"specialist": "psychic", "task": "t"}))
        assert out.status is RequestStatus.FAILURE
        assert "Unknown specialist" in out.error
        assert "reasoning" in out.error, "should list what is available"

    def test_unavailable_specialist_fails_cleanly(self):
        registry = SpecialistRegistry([
            REASONING,
            SpecialistSpec(name="off", description="d", available=False),
        ])
        out = _prime(registry, _FakeProvider()).handle(PrimeRequest(
            "consult_specialist", {"specialist": "off", "task": "t"}))
        assert out.status is RequestStatus.FAILURE
        assert "unavailable" in out.error

    def test_provider_failure_fails_cleanly(self):
        p = _FakeProvider(error=PrimeResult.failure("model HTTP 500"))
        out = _prime(_catalogue(), p).handle(PrimeRequest(
            "consult_specialist", {"specialist": "reasoning", "task": "t"}))
        assert out.status is RequestStatus.FAILURE
        assert "specialist_failed" in out.error
        assert "model HTTP 500" in out.error

    def test_provider_exception_does_not_escape(self):
        class Exploding(Provider):
            def invoke(self, capability, payload):
                raise MemoryError("catastrophic")

        out = _prime(_catalogue(), Exploding()).handle(PrimeRequest(
            "consult_specialist", {"specialist": "reasoning", "task": "t"}))
        assert out.status is RequestStatus.FAILURE

    def test_a_failing_provider_is_never_reported_as_success(self):
        """Regression: ``PrimeResult.success`` is a classmethod, not a flag.

        Testing its truthiness is always true, so a failing specialist would
        once have been reported as a successful consultation that returned
        nothing.
        """
        p = _FakeProvider(error=PrimeResult.failure("model HTTP 500"))
        out = _prime(_catalogue(), p).handle(PrimeRequest(
            "consult_specialist", {"specialist": "reasoning", "task": "t"}))
        assert out.status is RequestStatus.FAILURE
        assert "specialist_failed" in out.error
        assert "returned_nothing" not in out.error

    def test_empty_specialist_output_fails_cleanly(self):
        out = _prime(_catalogue(), _FakeProvider(content="   ")).handle(
            PrimeRequest("consult_specialist",
                         {"specialist": "reasoning", "task": "t"}))
        assert out.status is RequestStatus.FAILURE
        assert "nothing" in out.error

    def test_failure_reaches_aurora_and_her_session_continues(self):
        brain = _Scripted(
            _request("consult_specialist",
                     {"specialist": "reasoning", "task": "t"}),
            "I could not consult the specialist, so I will reason it myself.",
        )
        p = _FakeProvider(error=PrimeResult.failure("model HTTP 500"))
        a = _aurora(_prime(_catalogue(), p), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "t")
        result = a.respond([{"role": "user", "content": "go"}], "s")
        assert result.success
        assert result.content.startswith("I could not consult")
        tool = [m for m in brain.calls[1] if m["role"] == "tool"][0]["content"]
        assert "failure" in tool


# ======================================================================
# 6. Still bounded by the loop
# ======================================================================


class TestBoundedByLoop:
    def test_bounded_by_max_cycles(self):
        p = _FakeProvider()
        brain = _Scripted(
            _request("consult_specialist", {"specialist": "code", "task": "1"}),
            _request("consult_specialist", {"specialist": "code", "task": "2"}),
            _request("consult_specialist", {"specialist": "code", "task": "3"}),
            "done")
        a = _aurora(_prime(_catalogue(), p), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "go"}], "s")
        assert len(brain.calls) == 2
        assert len(p.calls) <= 2

    def test_repeated_identical_consult_is_blocked(self):
        p = _FakeProvider()
        brain = _Scripted(
            _request("consult_specialist",
                     {"specialist": "reasoning", "task": "same"}))
        a = _aurora(_prime(_catalogue(), p), brain)
        a.loop.max_cycles = 20
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "go"}], "s")
        assert a.loop.last_trace.stop_reason == "repeated_request"
        assert len(p.calls) == 2, "allowed twice, then blocked"

    def test_consultation_persists_nothing(self):
        import sqlite3
        prime = _prime(_catalogue(), _FakeProvider())
        path = str(prime._context._get_store()._db_path)
        prime.handle(PrimeRequest("consult_specialist",
                                  {"specialist": "reasoning", "task": "t"}))
        with sqlite3.connect(path) as con:
            assert con.execute(
                "SELECT COUNT(*) FROM messages").fetchone()[0] == 0


# ======================================================================
# 7. It is an instrument, not an agent
# ======================================================================


class TestNotAnAgent:
    def test_specialist_spec_has_no_room_for_state(self):
        fields = set(SpecialistSpec.__dataclass_fields__)
        assert fields == {"name", "description", "model", "system_prompt",
                          "available"}

    def test_specialist_receives_only_the_task(self):
        p = _FakeProvider()
        brain = _Scripted(
            _request("consult_specialist",
                     {"specialist": "reasoning", "task": "the task"}), "done")
        a = _aurora(_prime(_catalogue(), p), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "SECRET-WORKING-MEMORY")
        a.respond([{"role": "user", "content": "MY-PRIVATE-QUESTION"}], "s")
        sent = json.dumps(p.calls[0]["payload"]["messages"])
        assert "SECRET-WORKING-MEMORY" not in sent
        assert "MY-PRIVATE-QUESTION" not in sent
        assert "the task" in sent

    def test_catalogue_has_no_memory_or_sql(self):
        src = _src("PRIMORA.consult.catalogue").lower()
        for banned in ("sqlite", "contextstore", "session", "memory",
                       "goal", "conversation"):
            assert banned not in src, banned

    def test_capability_has_no_memory_or_orchestration(self):
        src = _src("PRIMORA.specialists.tools.consult").lower()
        for banned in ("sqlite", "store", "contextmanager", "mind",
                       "subprocess", "thread", "asyncio"):
            assert banned not in src, banned

    def test_capability_does_not_model_names_or_transport(self):
        src = _src("PRIMORA.specialists.tools.consult").lower()
        for banned in ("ollama", "http", "urllib", "api_key", "qwen"):
            assert banned not in src, banned

    def test_capability_imports_no_concrete_backend(self):
        tree = ast.parse(_src("PRIMORA.specialists.tools.consult"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                imported.update((a.name or "").split(".")[0]
                                for a in node.names)
        assert "OllamaProvider" not in imported
        assert "urllib" not in imported

    def test_only_one_aurora_and_one_mind_exist(self):
        prime = _prime(_catalogue(), _FakeProvider())
        brain = _Scripted("done")
        a = _aurora(prime, brain)
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "hi"}], "s")
        assert a.loop.mind is a.mind
        assert len(prime.specialists) == 2  # instruments, not actors


# ======================================================================
# 8. Scope
# ======================================================================


class TestScope:
    def test_cognitive_loop_is_unmodified(self):
        from PRIMORA.aurora import loop as loopmod
        tree = ast.parse(open(loopmod.__file__, encoding="utf-8").read())
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        for banned in ("consult_specialist", "specialists", "SpecialistSpec"):
            assert banned not in called

    def test_memory_and_retrieval_untouched(self):
        for name in ("manager.py", "hybrid.py", "store.py", "mind.py",
                     "semantic.py"):
            src = open(os.path.join("PRIMORA", "prime", "context", name),
                       encoding="utf-8").read().lower()
            for banned in ("consult_specialist", "specialistregistry"):
                assert banned not in src, f"{name}: {banned}"

    def test_verse_untouched(self):
        assert not os.path.exists(
            os.path.join("PRIMORA", "consult", "qml"))

    def test_no_orchestration_of_many_specialists(self):
        src = _src("PRIMORA.specialists.tools.consult").lower()
        for banned in ("for name in", "chain", "pipeline", "sequence",
                       "debate", "vote"):
            assert banned not in src, banned

    def test_no_learning_or_proactive_behaviour(self):
        src = _src("PRIMORA.specialists.tools.consult").lower()
        for banned in ("train", "learn", "schedule", "trigger", "proactive"):
            assert banned not in src, banned

    def test_production_database_untouched(self):
        """Production memory must be unchanged by this test run.

        Content equality via production_digest(), not emptiness: a used
        install legitimately holds real memories (MEMORY_AUDIT C2), so
        the invariant is "unchanged", never "empty". Isolated work runs
        under harness_session, which refuses production opens and
        re-checks the digest on exit.
        """
        from PRIMORA.prime.context.isolation import (
            harness_session,
            production_db_path,
            production_digest,
        )
        if not production_db_path().exists():
            pytest.skip("no production database")
        before = production_digest()
        with harness_session("c2_untouched_") as session:
            store = session.context_manager()._get_store()
            # An empty starting state belongs to the private store, never
            # to production.
            assert store.fetch_states(limit=5) == []
        assert production_digest() == before, (
            "production memory changed during this test")


class TestConsultIsConfigured:
    """An empty catalogue must not be advertised as a working capability."""

    def test_default_specialist_is_unconfigured(self):
        prime = Prime(_Model())
        specialist = prime.registry.get_specialist("consult_specialist")
        assert specialist.is_configured is False

    def test_unconfigured_is_not_advertised(self):
        from PRIMORA.aurora.configuration import build_capability_instruction
        prime = Prime(_Model())
        block = build_capability_instruction(prime)
        assert "- consult_specialist:" not in block

    def test_configured_specialist_is_advertised(self):
        from PRIMORA.aurora.configuration import build_capability_instruction
        prime = Prime(_Model())
        specialist = prime.registry.get_specialist("consult_specialist")
        specialist.set_specialists(SpecialistRegistry([REASONING]))
        assert specialist.is_configured is True
        block = build_capability_instruction(prime)
        assert "- consult_specialist:" in block
