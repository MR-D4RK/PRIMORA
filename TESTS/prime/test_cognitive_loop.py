"""Milestone 2: the Aurora cognitive loop.

The properties that matter, in order:

1. A one-cycle turn behaves exactly as before.
2. Continuation is possible and bounded, and can never be infinite.
3. One logical response is returned; segments are not separate turns.
4. The same Mind, and the same task-local Working Memory, span the session.
5. Nothing from M4/M5/M6 was duplicated or moved, and no model was added.
"""

import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.aurora.loop import (
    DEFAULT_MAX_CYCLES,
    CapabilityExecutor,
    CognitiveLoop,
    LoopTrace,
    NoCapabilityExecutor,
    Segment,
)
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context import BudgetConfig, ContextManager, Mind
from PRIMORA.prime.providers import Provider

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]

def _request(expression="2+2"):
    """A capability request. Distinct expressions matter: the loop refuses to
    execute a byte-identical request twice in one session, so a multi-segment
    test must vary the request to keep cognition going."""
    return (
        "<prime_request>\n"
        '{"action":"calculate","payload":{"expression":"' + expression + '"},'
        '"context":null,"metadata":null}\n'
        "</prime_request>"
    )


REQUEST = _request("2+2")


def _db(name=None):
    _SEQ[0] += 1
    return os.path.join(_TMP.name, f"{name or 'loop'}-{_SEQ[0]}.db")


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _budget():
    return BudgetConfig(
        input_target=600, input_ceiling=1200,
        output_reserve=64, total_budget=1400,
    )


def _config():
    return AuroraConfig(
        model_path=os.path.join("RESOURCES", "aurora", "v0.1", "model"),
        adapter_path=os.path.join("RESOURCES", "aurora", "v0.1", "aurora_qlora"),
    )


class _Provider(Provider):
    def __init__(self, output="4"):
        self._output = output
        self.calls = []

    def invoke(self, capability, payload):
        self.calls.append((capability, payload))
        return PrimeResult.success({"value": self._output}, "stub-calculator")


class ScriptedBrain(Brain):
    """Returns queued replies; the last one repeats if exhausted.

    A ``BrainResult`` in the queue is returned verbatim, which is how a
    generation failure is scripted.
    """

    def __init__(self, *replies):
        self._replies = list(replies) or ["plain answer"]
        self.calls = []

    def generate(self, messages, max_output_tokens=None):
        self.calls.append([dict(m) for m in messages])
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        reply = self._replies[index]
        if isinstance(reply, BrainResult):
            return reply
        if isinstance(reply, Exception):
            return BrainResult.failure(str(reply), "scripted")
        return BrainResult.success(reply, "scripted")

    def get_model_name(self):
        return "scripted"

    def count_tokens(self, messages):
        return _counter(messages)


def _aurora(brain=None, provider=None, manager=None, **loop_kw):
    prime = Prime(provider or _Provider())
    if manager is not None:
        prime._context = manager
        prime._mind = Mind(manager)
    a = Aurora(prime, brain=brain or ScriptedBrain(), config=_config(),
               context_manager=manager)
    for key, value in loop_kw.items():
        setattr(a.loop, key, value)
    return a


# ======================================================================
# 1. One-cycle behaviour is unchanged
# ======================================================================
class TestOneCycle:
    def test_plain_answer_uses_exactly_one_generation(self):
        brain = ScriptedBrain("just an answer")
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("one")))
        result = a.respond([{"role": "user", "content": "hello"}], "s")
        assert result.success
        assert result.content == "just an answer"
        assert len(brain.calls) == 1
        assert a.loop.last_trace.stop_reason == "finished"
        assert a.loop.last_trace.count == 1

    def test_capability_request_uses_two_generations_by_default(self):
        brain = ScriptedBrain(REQUEST, "the answer is 4")
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("two")))
        result = a.respond([{"role": "user", "content": "what is 2+2?"}], "s")
        assert result.content == "the answer is 4"
        assert len(brain.calls) == 2
        assert a.loop.last_trace.count == 2

    def test_default_max_cycles_allows_multi_step_cognition(self):
        """v0.2 pinned this to 2 for pre-loop parity; v0.3 raises the ceiling.

        The two-generation bound is now opt-in. What must stay true is that
        the loop *stops* when Aurora stops asking, and that the default is a
        hard ceiling rather than a suggestion.
        """
        brain = ScriptedBrain(REQUEST, "final")
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("bound")))
        a.respond([{"role": "user", "content": "go"}], "s")
        assert a.loop.max_cycles == DEFAULT_MAX_CYCLES == 8
        # She asked once, so she gets two segments regardless of the ceiling.
        assert len(brain.calls) == 2

    def test_two_cycle_behaviour_is_still_available(self):
        """The old bound is a configuration choice, not a hardwired rule."""
        brain = ScriptedBrain(REQUEST, "final")
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("legacy")))
        a.loop.max_cycles = 2
        a.respond([{"role": "user", "content": "go"}], "s")
        assert len(brain.calls) == 2
        assert a.loop.max_cycles == 2

    def test_generation_failure_returns_immediately(self):
        brain = ScriptedBrain(BrainResult.failure("model exploded", "scripted"))
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("fail")))
        result = a.respond([{"role": "user", "content": "hi"}], "s")
        assert result.success is False
        assert result.error == "model exploded"
        assert len(brain.calls) == 1
        assert a.loop.last_trace.stop_reason == "generation_failed"

    def test_return_type_is_brain_result(self):
        from PRIMORA.aurora.brain import BrainResult as BR
        a = _aurora(manager=ContextManager(db_path=_db("rtype")))
        assert isinstance(a.respond([{"role": "user", "content": "x"}], "s"), BR)


# ======================================================================
# 2. Continuation across segments
# ======================================================================
class TestContinuation:
    def test_three_segments_continue_when_cycles_allow(self):
        brain = ScriptedBrain(_request("2+2"), _request("3+3"),
                              "the real answer")
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("three")))
        a.loop.max_cycles = 3
        result = a.respond([{"role": "user", "content": "work"}], "s")
        assert result.content == "the real answer"
        assert len(brain.calls) == 3
        assert a.loop.last_trace.count == 3

    def test_each_segment_accumulates_prior_context(self):
        brain = ScriptedBrain(_request("2+2"), _request("3+3"), "done")
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("acc")))
        a.loop.max_cycles = 3
        a.respond([{"role": "user", "content": "go"}], "s")
        # Segment 3 must see the assistant and tool turns from segments 1-2.
        third = " ".join(m["content"] for m in brain.calls[2])
        assert "prime_request" in third
        assert third.count('"status"') >= 2, (
            "each continuation must carry the earlier capability result"
        )

    def test_segment_records_expose_structure(self):
        brain = ScriptedBrain(REQUEST, "answer")
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("seg")))
        a.respond([{"role": "user", "content": "go"}], "s")
        segments = a.loop.last_trace.segments
        assert isinstance(segments[0], Segment)
        assert segments[0].requested is True
        assert segments[0].capability == "calculate"
        assert segments[0].executed is True
        assert segments[1].requested is False
        assert isinstance(a.loop.last_trace, LoopTrace)
        assert a.loop.last_trace.to_dict()["segment_count"] == 2

    def test_segments_are_not_separate_assistant_turns(self):
        brain = ScriptedBrain(_request("2+2"), _request("3+3"), "final")
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("turns")))
        a.loop.max_cycles = 3
        result = a.respond([{"role": "user", "content": "go"}], "s")
        assert result.content == "final", (
            "only the last segment is the response"
        )
        assert "prime_request" not in result.content

    def test_executor_records_every_capability_call(self):
        brain = ScriptedBrain(_request("2+2"), _request("3+3"), "final")
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("exec")))
        a.loop.max_cycles = 3
        a.respond([{"role": "user", "content": "go"}], "s")
        assert [name for name, _ in a.loop.executor.calls] == [
            "calculate", "calculate"]
        assert a.loop.executor.calls[0][1] == {"expression": "2+2"}


# ======================================================================
# 3. Termination: bounded, deterministic, no infinite loop
# ======================================================================
class TestTermination:
    def test_max_cycles_is_a_hard_bound(self):
        # Distinct requests every time, so only the cycle bound can stop it.
        brain = ScriptedBrain(_request("1+1"), _request("2+2"),
                              _request("3+3"), _request("4+4"),
                              _request("5+5"), _request("6+6"))
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("hard")))
        a.loop.max_cycles = 4
        a.respond([{"role": "user", "content": "go"}], "s")
        assert len(brain.calls) == 4
        assert a.loop.last_trace.stop_reason == "max_cycles"

    def test_identical_request_is_allowed_twice_then_stops(self):
        brain = ScriptedBrain(REQUEST)  # the same request every time
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("spin")))
        a.loop.max_cycles = 50
        a.respond([{"role": "user", "content": "go"}], "s")
        # Executed on segment 0 and again on segment 1; the third identical
        # request is refused and ends the loop.
        assert len(brain.calls) == 3
        assert len(a.loop.executor.calls) == 2
        assert a.loop.last_trace.stop_reason == "repeated_request"

    def test_max_repeats_of_one_refuses_the_immediate_repeat(self):
        brain = ScriptedBrain(REQUEST)
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("rep1")))
        a.loop.max_cycles = 50
        a.loop.max_repeats = 1
        a.respond([{"role": "user", "content": "go"}], "s")
        assert len(a.loop.executor.calls) == 1
        assert a.loop.last_trace.stop_reason == "repeated_request"

    def test_max_repeats_rejects_zero(self):
        a = _aurora(brain=ScriptedBrain("x"),
                    manager=ContextManager(db_path=_db("rep0")))
        with pytest.raises(ValueError):
            CognitiveLoop(
                a.mind, ScriptedBrain("x"), max_cycles=2, max_repeats=0,
                token_counter=_counter, budget=_budget(),
            )

    def test_distinct_requests_still_run_to_the_bound(self):
        brain = ScriptedBrain(
            _request("1+1"), _request("2+2"), _request("3+3"),
            _request("4+4"), "done",
        )
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("distinct")))
        a.loop.max_cycles = 6
        result = a.respond([{"role": "user", "content": "go"}], "s")
        assert result.content == "done"
        assert len(brain.calls) == 5
        assert a.loop.last_trace.stop_reason == "finished"
        assert len(a.loop.executor.calls) == 4

    def test_max_cycles_must_be_at_least_one(self):
        mind = Mind(ContextManager(db_path=_db("bad")))
        with pytest.raises(ValueError):
            CognitiveLoop(mind, ScriptedBrain(), max_cycles=0)

    def test_max_cycles_of_one_never_executes(self):
        brain = ScriptedBrain(REQUEST, "never reached")
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("onecy")))
        a.loop.max_cycles = 1
        result = a.respond([{"role": "user", "content": "go"}], "s")
        assert len(brain.calls) == 1
        assert a.loop.last_trace.stop_reason == "max_cycles"
        assert result.content == REQUEST

    def test_deadline_stops_continuation(self):
        brain = ScriptedBrain(REQUEST)
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("dl")))
        a.loop.max_cycles = 100
        a.loop.deadline_s = -1.0  # already expired
        a.respond([{"role": "user", "content": "go"}], "s")
        assert len(brain.calls) == 0
        assert a.loop.last_trace.stop_reason == "deadline"

    def test_deadline_mid_loop_returns_the_last_segment(self, monkeypatch):
        """The case that actually happens: time runs out partway through.

        The test above fires before the first generation, which cannot happen
        in production. A long task that exhausts its budget must still deliver
        its last segment, because that segment is a real answer and returning
        a failure would throw away work the user already paid for.
        """
        import time

        import PRIMORA.aurora.loop as loop_module

        real = time.perf_counter
        spent = {"s": 0.0}

        def fake_now():
            return real() + spent["s"]

        class Slow(ScriptedBrain):
            def generate(self, messages, max_output_tokens=None):
                spent["s"] += 100.0  # one generation eats the whole budget
                return super().generate(messages, max_output_tokens)

        brain = Slow(_request("1+1"), "never reached")
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("dlmid")))
        a.loop.max_cycles = 10
        a.loop.deadline_s = 2.0

        monkeypatch.setattr(loop_module.time, "perf_counter", fake_now)
        result = a.respond([{"role": "user", "content": "go"}], "s")

        assert len(brain.calls) == 1, "should stop before a second generation"
        assert a.loop.last_trace.stop_reason == "deadline"
        assert result.success
        assert result.content == _request("1+1")

    def test_executor_exception_does_not_break_the_loop(self):
        class Boom(CapabilityExecutor):
            def execute(self, request):
                raise RuntimeError("executor exploded")

        brain = ScriptedBrain(REQUEST, "recovered anyway")
        prime = Prime(_Provider())
        manager = ContextManager(db_path=_db("boom"))
        prime._context = manager
        prime._mind = Mind(manager)
        a = Aurora(prime, brain=brain, config=_config(), context_manager=manager)
        a.loop.executor = Boom(prime)
        result = a.respond([{"role": "user", "content": "go"}], "s")
        assert result.success
        assert "executor exploded" in " ".join(
            m["content"] for m in brain.calls[1])
        assert result.content == "recovered anyway"

    def test_no_executor_ends_the_session(self):
        brain = ScriptedBrain(REQUEST, "unreachable")
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("noexec")))
        a.loop.executor = None
        a.respond([{"role": "user", "content": "go"}], "s")
        assert len(brain.calls) == 1
        assert a.loop.last_trace.stop_reason == "no_executor"

    def test_blocked_executor_is_used_safely(self):
        brain = ScriptedBrain(REQUEST, "handled")
        prime = Prime(_Provider())
        manager = ContextManager(db_path=_db("blocked"))
        prime._context = manager
        prime._mind = Mind(manager)
        a = Aurora(prime, brain=brain, config=_config(), context_manager=manager)
        a.loop.executor = NoCapabilityExecutor()
        result = a.respond([{"role": "user", "content": "go"}], "s")
        assert result.success
        assert "no capability executor" in " ".join(
            m["content"] for m in brain.calls[1])


# ======================================================================
# 4. One Mind, one Working Memory, for the whole session
# ======================================================================
class TestMindAndWorkingMemory:
    def test_the_same_mind_is_used_across_all_segments(self):
        mind = Mind(ContextManager(db_path=_db("samemind")))
        brain = ScriptedBrain(_request("2+2"), _request("3+3"), "final")
        loop = CognitiveLoop(
            mind, brain, executor=CapabilityExecutor(Prime(_Provider())),
            max_cycles=3, token_counter=_counter, budget=_budget(),
        )
        loop.run([{"role": "user", "content": "go"}], "s")
        assert loop.mind is mind
        assert len(brain.calls) == 3

    def test_working_memory_survives_between_segments(self):
        mind = Mind(ContextManager(db_path=_db("wmseg")))
        mind.begin_task("s", "long running task")
        mind.working.set_objective("stay on it")
        brain = ScriptedBrain(REQUEST, REQUEST, "final")
        loop = CognitiveLoop(
            mind, brain, executor=CapabilityExecutor(Prime(_Provider())),
            max_cycles=3, token_counter=_counter, budget=_budget(),
        )
        loop.run([{"role": "user", "content": "go"}], "s")
        for index, call in enumerate(brain.calls):
            blob = " ".join(m["content"] for m in call)
            assert "[working task]" in blob, f"segment {index} lost working task"
            assert "[working objective]" in blob, (
                f"segment {index} lost the working objective"
            )

    def test_working_decisions_added_mid_loop_are_visible(self):
        mind = Mind(ContextManager(db_path=_db("wmmid")))
        mind.begin_task("s", "a task")
        brain = ScriptedBrain(REQUEST, "final")
        mind.working.note_decision("remember this between segments")
        loop = CognitiveLoop(
            mind, brain, executor=CapabilityExecutor(Prime(_Provider())),
            max_cycles=2, token_counter=_counter, budget=_budget(),
        )
        loop.run([{"role": "user", "content": "go"}], "s")
        second = " ".join(m["content"] for m in brain.calls[1])
        assert "remember this between segments" in second

    def test_working_memory_is_never_persisted(self):
        manager = ContextManager(db_path=_db("wmpersist"))
        mind = Mind(manager)
        mind.begin_task("s", "private task")
        mind.working.note_decision("do not persist me")
        brain = ScriptedBrain(REQUEST, "final")
        loop = CognitiveLoop(
            mind, brain, executor=CapabilityExecutor(Prime(_Provider())),
            max_cycles=2, token_counter=_counter, budget=_budget(),
        )
        loop.run([{"role": "user", "content": "go"}], "s")
        con = sqlite3.connect(f"file:{manager._db_path}?mode=ro", uri=True)
        try:
            for table in ("states", "entities", "candidates", "digests"):
                for row in con.execute(f"SELECT * FROM {table}"):
                    assert "do not persist me" not in str(row), table
        finally:
            con.close()

    def test_a_new_mind_is_not_built_per_segment(self):
        mind = Mind(ContextManager(db_path=_db("nomind")))
        brain = ScriptedBrain(_request("2+2"), _request("3+3"), "final")
        built = []
        original = mind.build

        def counting_build(*args, **kwargs):
            built.append(1)
            return original(*args, **kwargs)

        mind.build = counting_build
        loop = CognitiveLoop(
            mind, brain, executor=CapabilityExecutor(Prime(_Provider())),
            max_cycles=3, token_counter=_counter, budget=_budget(),
        )
        loop.run([{"role": "user", "content": "go"}], "s")
        assert len(built) == 3, "one Mind.build per segment, no more"
        assert mind is loop.mind


# ======================================================================
# 5. Persistence semantics: same snapshot/sequence/fingerprint rules;
#    the closing reply is additionally written before run() returns.
# ======================================================================
class TestPersistenceUnchanged:
    def test_segments_are_reconciled_with_the_same_snapshot_shape(self):
        manager = ContextManager(db_path=_db("persist"))
        brain = ScriptedBrain(REQUEST, "final")
        loop = CognitiveLoop(
            Mind(manager), brain,
            executor=CapabilityExecutor(Prime(_Provider())),
            max_cycles=2, token_counter=_counter, budget=_budget(),
        )
        loop.run([{"role": "user", "content": "what is 2+2?"}], "s")
        con = sqlite3.connect(f"file:{manager._db_path}?mode=ro", uri=True)
        try:
            roles = [r[0] for r in con.execute(
                "SELECT role FROM messages ORDER BY seq").fetchall()]
            # Internal segments keep the pre-loop snapshot shape (user,
            # assistant, tool) in the same turn; the final reply joins the
            # same snapshot at the end, sequenced after them.
            assert roles == ["user", "assistant", "tool", "assistant"], (
                "internal rows stay (user, assistant, tool) and the "
                "closing answer lands after them; "
                f"got {roles}"
            )
        finally:
            con.close()

    def test_a_one_cycle_turn_persists_user_message_and_final_reply(self):
        manager = ContextManager(db_path=_db("persist1"))
        brain = ScriptedBrain("no capability needed")
        loop = CognitiveLoop(
            Mind(manager), brain, max_cycles=2,
            token_counter=_counter, budget=_budget(),
        )
        loop.run([{"role": "user", "content": "hello"}], "s")
        con = sqlite3.connect(f"file:{manager._db_path}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT role, content FROM messages ORDER BY seq"
            ).fetchall()
            # Build reconciles the user turn; the end-of-turn persist
            # writes the closing answer so an immediate exit after
            # respond() cannot lose it. Two rows, ordered, no duplicates.
            assert rows == [
                ("user", "hello"),
                ("assistant", "no capability needed"),
            ]
        finally:
            con.close()


# ======================================================================
# 6. Aurora integration and the Milestone 3 seam
# ======================================================================
class TestAuroraIntegration:
    def test_aurora_exposes_its_loop(self):
        a = _aurora(manager=ContextManager(db_path=_db("lp")))
        assert isinstance(a.loop, CognitiveLoop)
        assert a.loop.mind is a.mind

    def test_aurora_uses_the_existing_brain_not_a_second_one(self):
        brain = ScriptedBrain("ok")
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("nobrain")))
        assert a.loop._brain is brain
        assert a._brain is brain

    def test_respond_signature_is_unchanged(self):
        import inspect
        params = list(inspect.signature(Aurora.respond).parameters)
        assert params == ["self", "messages", "session_id"]

    def test_executor_is_replaceable_for_milestone_three(self):
        class FakeResearch:
            def __init__(self):
                self.calls = 0

            def execute(self, request):
                self.calls += 1
                return '{"status":"success","output":"researched","error":null}'

        brain = ScriptedBrain(REQUEST, "used research")
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("m3")))
        research = FakeResearch()
        a.loop.executor = research
        a.respond([{"role": "user", "content": "find out"}], "s")
        assert research.calls == 1
        assert "researched" in " ".join(m["content"] for m in brain.calls[1])

    def test_loop_is_reusable_across_turns(self):
        brain = ScriptedBrain("first", "second", "third")
        a = _aurora(brain=brain, manager=ContextManager(db_path=_db("reuse")))
        a.respond([{"role": "user", "content": "one"}], "s")
        a.respond([{"role": "user", "content": "two"}], "s")
        assert len(brain.calls) == 2
        assert a.loop.last_trace.count == 1

    def test_context_for_still_works(self):
        from PRIMORA.prime.context.mind import CognitiveContext
        a = _aurora(manager=ContextManager(db_path=_db("ctxfor")))
        ctx = a.context_for([{"role": "user", "content": "hi"}], "s")
        assert isinstance(ctx, CognitiveContext)
        assert ctx.messages


# ======================================================================
# 7. Scope discipline
# ======================================================================
class TestScopeDiscipline:
    def _code(self, *relative_paths):
        """Source with docstrings removed, for PRIMORA-relative paths.

        Boundary tests inspect executable code, not prose: a raw text scan of
        a module whose docstring explains why research is absent would match
        the word "research" and fail for the wrong reason.
        """
        import ast
        text = "\n".join(
            open(os.path.join("PRIMORA", p), encoding="utf-8").read()
            for p in relative_paths
        )
        tree = ast.parse(text)
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body:
                first = body[0]
                if (isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    body.pop(0)
        return ast.unparse(tree)

    def test_loop_has_no_cognitive_decisions(self):
        code = self._code("aurora/loop.py").lower()
        for banned in ("should_research", "should_act", "needs_memory",
                       "should_remember", "is_important", "decide_"):
            assert banned not in code, banned

    def test_loop_does_not_duplicate_mechanism_code(self):
        code = self._code("aurora/loop.py")
        for banned in ("SEMANTIC_TAU", "RRF_K", "EVIDENCE_FLOOR",
                       "CREATE TABLE", "INSERT INTO", "UPDATE ",
                       "DELETE FROM", "CREATE VIRTUAL", "messages_fts"):
            assert banned not in code, banned

    def test_loop_introduces_no_model(self):
        code = self._code("aurora/loop.py")
        for banned in ("import torch", "transformers", "AutoModel",
                       "from_pretrained", "sentence_transformers",
                       "SentenceTransformer", "requests", "subprocess"):
            assert banned not in code, banned

    def test_loop_makes_no_calls_into_m4_m5_m6_internals(self):
        code = self._code("aurora/loop.py")
        for banned in ("hybrid_retrieve", "semantic_channel", "rrf_fuse",
                       "apply_evidence_floor", "extract_candidates",
                       "apply_user_candidates", "create_state"):
            assert banned not in code, banned

    def test_loop_only_uses_the_mind_boundary(self):
        code = self._code("aurora/loop.py")
        assert "self._mind.build" in code
        assert "self._brain.generate" in code

    def test_no_milestone_three_capability(self):
        code = self._code("aurora/loop.py").lower()
        for banned in ("research", "memory_search", "specialist",
                       "web_search", "proactive", "trigger", "goal"):
            assert banned not in code, banned

    def test_verse_is_untouched_by_the_loop(self):
        code = self._code("aurora/loop.py")
        assert "verse" not in code.lower()
