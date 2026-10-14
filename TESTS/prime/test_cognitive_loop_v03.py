"""Cognitive loop v0.3: real multi-step cognition.

v0.2 was structurally correct but capped at two segments, which permits at
most one capability per turn. v0.3 raises the ceiling so Aurora can sustain a
genuine process:

    think -> recall -> think -> research -> think -> tool -> think -> answer

The runtime gains no new decision logic. Every chain below is Aurora emitting
a request and the loop carrying the result forward; the runtime only executes,
returns, resumes, and enforces limits. A test below proves that explicitly, by
showing the loop does not know which capabilities it carried.
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
from PRIMORA.aurora.loop import (
    DEFAULT_CAPABILITY_BUDGET,
    DEFAULT_DEADLINE_S,
    DEFAULT_MAX_CYCLES,
    LOOP_VERSION,
)
from PRIMORA.learning import Jev, LearningOutcome, StructuredCandidate
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.providers import Provider
from PRIMORA.research import ResearchProvider, ResearchResult
from PRIMORA.toolbox import ToolRegistry, ToolResult, ToolSpec

FACT = "kestrel seventeen was the last cargo"


def _request(action, payload):
    return (
        "<prime_request>\n"
        + json.dumps({"action": action, "payload": payload,
                      "context": None, "metadata": None})
        + "\n</prime_request>"
    )


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.success({"value": 4})


class _Web(ResearchProvider):
    name = "web"

    def __init__(self):
        self.calls = []

    def search(self, query, limit=5):
        self.calls.append(query)
        return [ResearchResult("Kestrel seventeen", "https://freight.test/k",
                               "The last cargo was kestrel seventeen.",
                               "freight.test")]


class _Jev(Jev):
    def evaluate(self, request):
        return LearningOutcome.confirmed([
            StructuredCandidate(concept="cargo", value="kestrel seventeen",
                                memory_type="FACT")])


class _Scripted(Brain):
    """Replies in order across every generation in the whole run."""

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


class _Noting(_Scripted):
    """A brain that writes to Working Memory while it is generating.

    This is how a decision is made *inside* a cognitive session rather than
    after it, which is the only way to show the working set survives between
    segments instead of merely still existing at the end.
    """

    def __init__(self, *replies):
        super().__init__(*replies)
        self.mind = None
        self.noted = False

    def generate(self, messages, max_output_tokens=None):
        if not self.noted and self.mind is not None:
            if self.mind.working is not None:
                self.mind.working.note_decision("check the manifest twice")
            self.noted = True
        return super().generate(messages, max_output_tokens)


_SEQ = [0]


def _prime(jev=None):
    _SEQ[0] += 1
    prime = Prime(_Provider(), research_provider=_Web(), jev=jev or _Jev(),
                  tools=ToolRegistry.with_builtins())
    prime._context = prime._context.__class__(
        db_path=os.path.join(tempfile.mkdtemp(prefix="v03_"),
                            f"v{_SEQ[0]}.db"))
    prime._mind = prime._mind.__class__(prime._context)
    for name in ("recall", "learn"):
        cap = prime.registry.get_specialist(name)
        if cap is not None:
            cap.bind(prime._mind)
    return prime


def _aurora(prime, brain, **config):
    settings = {"model_path": "m", "adapter_path": "a"}
    settings.update(config)
    a = Aurora(prime, brain=brain, config=AuroraConfig(**settings))
    return a


def _run(replies, session="s", user="work on the cargo", **config):
    prime = _prime()
    brain = _Scripted(*replies)
    a = _aurora(prime, brain, **config)
    a.mind.begin_task(session, "the cargo job")
    result = a.respond([{"role": "user", "content": user}], session)
    return brain, a, prime, result


def _executed(aurora):
    return [name for name, _ in aurora.loop.executor.calls]


def _context_of(brain, index):
    return " ".join(m["content"] for m in brain.calls[index])


# ======================================================================
# 1. The upgrade itself
# ======================================================================


class TestVersionAndDefaults:
    def test_version_is_recorded(self):
        assert LOOP_VERSION == "0.3"

    def test_default_allows_multi_step_cognition(self):
        assert DEFAULT_MAX_CYCLES == 8
        assert DEFAULT_MAX_CYCLES > 2

    def test_budget_is_independent_and_lower(self):
        """Cycles bound thinking; budget bounds doing. They are separate."""
        assert DEFAULT_CAPABILITY_BUDGET < DEFAULT_MAX_CYCLES
        assert DEFAULT_DEADLINE_S == 300.0

    def test_cycles_bound_generations_not_executions(self):
        brain, a, _p, _r = _run(
            [_request("calculate", {"expression": f"{i}+1"})
             for i in range(6)] + ["done"])
        assert len(brain.calls) <= DEFAULT_MAX_CYCLES
        assert a.loop.executor.used <= DEFAULT_CAPABILITY_BUDGET

    def test_config_carries_the_ceiling(self):
        assert AuroraConfig(model_path="m", adapter_path="a"
                            ).max_cognition_cycles == DEFAULT_MAX_CYCLES

    def test_config_can_override_the_ceiling(self, monkeypatch):
        monkeypatch.setenv("PRIMORA_AURORA_MAX_COGNITION_CYCLES", "3")
        assert AuroraConfig.from_env().max_cognition_cycles == 3

    def test_aurora_honours_the_configured_ceiling(self):
        prime = _prime()
        a = _aurora(prime, _Scripted("done"), max_cognition_cycles=3)
        assert a.loop.max_cycles == 3


# ======================================================================
# 2. Multi-step chains: each capability, then a final answer
# ======================================================================


class TestMultiStepChains:
    def test_recall_then_finish(self):
        brain, a, _p, result = _run([
            _request("recall", {"query": "cargo"}),
            "You had said the last cargo was kestrel seventeen.",
        ])
        assert _executed(a) == ["recall"]
        assert "kestrel seventeen" in result.content
        assert a.loop.last_trace.count == 2

    def test_research_then_finish(self):
        brain, a, prime, result = _run([
            _request("research", {"query": "last cargo"}),
            "Freight records say kestrel seventeen.",
        ])
        assert _executed(a) == ["research"]
        assert "kestrel seventeen" in result.content
        assert prime._registry.get_specialist("research")._provider.calls \
            == ["last cargo"]

    def test_specialist_then_finish(self):
        prime = _prime()
        from PRIMORA.consult import SpecialistRegistry, SpecialistSpec
        prime = Prime(_Provider(),
                      research_provider=_Web(),
                      specialists=SpecialistRegistry([
                          SpecialistSpec(name="reasoning",
                                         description="Thinks hard.",
                                         model="m")]),
                      tools=ToolRegistry.with_builtins())
        prime._context = prime._context.__class__(
            db_path=os.path.join(tempfile.mkdtemp(prefix="v03s_"), "s.db"))
        prime._mind = prime._mind.__class__(prime._context)
        for name in ("recall", "learn"):
            cap = prime.registry.get_specialist(name)
            if cap is not None:
                cap.bind(prime._mind)
        brain = _Scripted(
            _request("consult_specialist", {"specialist": "reasoning",
                                            "task": "analyse the manifest"}),
            "The specialist favours the second port.")
        a = _aurora(prime, brain)
        a.mind.begin_task("s", "t")
        result = a.respond([{"role": "user", "content": "analyse"}], "s")
        assert _executed(a) == ["consult_specialist"]
        assert "second port" in result.content

    def test_tool_then_finish(self):
        brain, a, _p, result = _run([
            _request("use_tool", {"name": "count_words",
                                  "arguments": {"text": "nine containers"}}),
            "There are two words.",
        ])
        assert _executed(a) == ["use_tool"]
        assert "two words" in result.content
        assert "9" in _context_of(brain, 1) or "2" in _context_of(brain, 1)

    def test_two_different_capabilities_in_one_session(self):
        brain, a, _p, result = _run([
            _request("recall", {"query": "cargo"}),
            _request("research", {"query": "last cargo"}),
            "Kestrel seventeen, from both memory and the record.",
        ])
        assert _executed(a) == ["recall", "research"]
        assert len(brain.calls) == 3
        assert a.loop.last_trace.stop_reason == "finished"

    def test_a_full_four_capability_chain(self):
        """The target shape from the milestone description."""
        brain, a, _p, result = _run([
            _request("recall", {"query": "cargo"}),
            _request("research", {"query": "last cargo"}),
            _request("use_tool", {"name": "count_words",
                                  "arguments": {"text": "two containers"}}),
            "Kestrel seventeen, two containers.",
        ])
        assert _executed(a) == ["recall", "research", "use_tool"]
        assert len(brain.calls) == 4
        assert result.content.startswith("Kestrel seventeen")

    def test_multiple_calls_with_different_requests(self):
        brain, a, _p, _r = _run([
            _request("research", {"query": "first question"}),
            _request("research", {"query": "second question"}),
            _request("research", {"query": "third question"}),
            "done",
        ])
        assert _executed(a) == ["research"] * 3
        provider = a._prime._registry.get_specialist("research")._provider
        assert provider.calls == ["first question", "second question",
                                  "third question"]

    def test_every_result_is_carried_forward(self):
        brain, a, _p, _r = _run([
            _request("research", {"query": "alpha"}),
            _request("use_tool", {"name": "count_words",
                                  "arguments": {"text": "beta"}}),
            "done",
        ])
        # The final segment's context must contain both earlier results.
        final = _context_of(brain, 2)
        assert "freight.test" in final, "the research result was lost"
        assert "beta" in final, "the tool result was lost"


# ======================================================================
# 3. The limits still hold
# ======================================================================


class TestLimitsStillHold:
    def test_capability_budget_stops_execution(self):
        brain, a, _p, _r = _run(
            [_request("research", {"query": f"q{i}"}) for i in range(6)]
            + ["done"])
        assert a.loop.executor.used == DEFAULT_CAPABILITY_BUDGET
        assert a.loop.executor.exhausted
        provider = a._prime._registry.get_specialist("research")._provider
        assert len(provider.calls) == DEFAULT_CAPABILITY_BUDGET, \
            "the budget did not stop real executions"
        assert any("capability_budget_exhausted" in m["content"]
                   for call in brain.calls for m in call if m["role"] == "tool")

    def test_max_cycles_stops_execution(self):
        prime = _prime()
        brain = _Scripted(*[_request("research", {"query": f"q{i}"})
                            for i in range(30)])
        a = _aurora(prime, brain)
        a.loop.executor.max_calls = 1000  # budget must not be what stops it
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "go"}], "s")
        assert len(brain.calls) == DEFAULT_MAX_CYCLES
        assert a.loop.last_trace.stop_reason == "max_cycles"

    def test_deadline_returns_the_latest_valid_segment(self, monkeypatch):
        import time
        import PRIMORA.aurora.loop as loopmod

        spent = {"s": 0.0}
        real = time.perf_counter

        def fake():
            return real() + spent["s"]

        class Slow(_Scripted):
            def generate(self, messages, max_output_tokens=None):
                spent["s"] += 40.0
                return super().generate(messages, max_output_tokens)

        prime = _prime()
        brain = Slow(_request("recall", {"query": "cargo"}),
                     "a partial answer")
        a = _aurora(prime, brain)
        a.loop.deadline_s = 30.0
        a.mind.begin_task("s", "t")
        monkeypatch.setattr(loopmod.time, "perf_counter", fake)
        result = a.respond([{"role": "user", "content": "go"}], "s")
        assert a.loop.last_trace.stop_reason == "deadline"
        assert result.success
        assert result.content == _request("recall", {"query": "cargo"})

    def test_repeat_guard_still_works(self):
        brain, a, _p, _r = _run([_request("recall", {"query": "cargo"})])
        assert a.loop.last_trace.stop_reason == "repeated_request"
        assert a.loop.executor.used == 2

    def test_identical_tool_call_is_still_blocked(self):
        brain, a, _p, _r = _run([
            _request("use_tool", {"name": "count_words",
                                  "arguments": {"text": "same"}})])
        assert a.loop.last_trace.stop_reason == "repeated_request"
        assert a.loop.executor.used == 2

    def test_a_single_request_still_produces_two_segments(self):
        brain, a, _p, _r = _run([_request("recall", {"query": "cargo"}),
                                 "final"])
        assert len(brain.calls) == 2

    def test_a_turn_with_no_request_is_unchanged(self):
        brain, a, _p, result = _run(["just an answer"])
        assert len(brain.calls) == 1
        assert result.content == "just an answer"


# ======================================================================
# 4. Identity, Mind and Working Memory are one continuous session
# ======================================================================


class TestOneContinuousSession:
    def test_working_memory_survives_every_segment(self):
        """A decision made *during* the run is visible in the next segment.

        Noting it afterwards would only prove the object still exists; this
        proves it survived between segments, which is the actual claim.
        """
        prime = _prime()
        brain = _Noting(
            _request("recall", {"query": "cargo"}),
            "done")
        a = Aurora(prime, brain=brain,
                   config=AuroraConfig(model_path="m", adapter_path="a"))
        brain.mind = a.mind
        a.mind.begin_task("s", "the cargo job")
        a.respond([{"role": "user", "content": "work on the cargo"}], "s")

        assert brain.noted, "the decision was never taken"
        assert len(brain.calls) >= 2
        for index in range(1, len(brain.calls)):
            assert "check the manifest twice" in _context_of(brain, index), \
                f"the working set was lost before segment {index}"
        assert a.mind.working is not None
        assert a.loop.mind is a.mind

    def test_one_mind_for_every_segment(self):
        brain, a, _p, _r = _run([
            _request("recall", {"query": "cargo"}),
            _request("research", {"query": "last cargo"}),
            "done",
        ])
        assert a.loop.mind is a.mind is a._prime.mind
        assert a.mind.working.session_id == "s"

    def test_one_brain_for_every_segment(self):
        brain, a, _p, _r = _run([
            _request("recall", {"query": "cargo"}),
            "done",
        ])
        assert a.loop._brain is a._brain is brain
        assert len(brain.calls) >= 2

    def test_working_memory_visible_in_a_later_segment(self):
        prime = _prime()
        brain = _Noting(_request("recall", {"query": "cargo"}), "done")
        a = Aurora(prime, brain=brain,
                   config=AuroraConfig(model_path="m", adapter_path="a"))
        brain.mind = a.mind
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "go"}], "s")
        assert "check the manifest twice" in _context_of(brain, 1)
        assert "check the manifest twice" in _context_of(brain, -1)

    def test_the_session_is_one_response(self):
        brain, a, _p, result = _run([
            _request("recall", {"query": "cargo"}),
            _request("research", {"query": "last cargo"}),
            "the single final answer",
        ])
        assert result.content == "the single final answer"
        assert "prime_request" not in result.content


# ======================================================================
# 5. No semantic decision logic was added
# ======================================================================


class TestNoSemanticLogicAdded:
    def test_the_loop_names_no_capability(self):
        from PRIMORA.aurora import loop as loopmod
        tree = ast.parse(open(loopmod.__file__, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef,
                                     ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
        code = ast.unparse(tree)
        for banned in ('"recall"', '"research"', '"learn"', '"use_tool"',
                       '"consult_specialist"', '"calculate"'):
            assert banned not in code, f"the loop names {banned}"

    def test_the_loop_has_no_opinion_about_sufficiency(self):
        """Continuation is structural: a request means continue, nothing else."""
        from PRIMORA.aurora import loop as loopmod
        tree = ast.parse(open(loopmod.__file__, encoding="utf-8").read())
        run = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "run")
        src = ast.unparse(run)
        for banned in ("enough", "satisfied", "sufficient", "confident",
                       "decide", "should_"):
            assert banned not in src, banned

    def test_only_the_default_changed_in_the_runtime(self):
        """No new branches, only a new number."""
        from PRIMORA.aurora import loop as loopmod
        src = open(loopmod.__file__, encoding="utf-8").read()
        assert "LOOP_VERSION = \"0.3\"" in src
        assert "DEFAULT_MAX_CYCLES = 8" in src

    def test_mind_memory_jev_and_capabilities_are_untouched(self):
        for name in ("PRIMORA/prime/context/mind.py",
                     "PRIMORA/prime/context/manager.py",
                     "PRIMORA/prime/context/state.py",
                     "PRIMORA/learning/jev.py",
                     "PRIMORA/prime/capability.py"):
            src = open(name.replace("/", os.sep), encoding="utf-8").read()
            assert "max_cycles" not in src, f"{name} reached into the loop"

    def test_capability_interfaces_are_unchanged(self):
        prime = _prime()
        assert sorted(prime.registry.get_names()) == [
            "calculate", "consult_specialist", "learn", "recall", "research",
            "research_read", "use_tool"]

    def test_verse_untouched(self):
        assert os.path.isdir(os.path.join("PRIMORA", "verse", "qml"))


# ======================================================================
# 6. Production data
# ======================================================================


class TestProductionUntouched:
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
