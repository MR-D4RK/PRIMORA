"""Aurora emitting exactly one valid capability request per response.

Three real-model behaviours scored 0/5 in the repeatability run:

  * multi-step      - two ``<prime_request>`` blocks in one generation;
  * specialist      - prose narration instead of a ``consult_specialist`` request;
  * tool            - ``{"action":"count_words"}``, the tool name used as the
                      action instead of the ``use_tool`` wrapper.

The causes were not one bug. The prompt told Aurora that a request "may describe
a single goal or multiple related goals in one payload", which is an explicit
invitation to emit the two-block form; the tool and specialist catalogues each
showed their own names in a list without saying those names were never actions;
and the parser rejected a two-block reply and reported it as ``finished``, which
discarded a request Aurora really had made and made the failure invisible.

These tests pin the protocol and the loop's handling of it. The generation
itself is the real model's behaviour and is measured by
``TESTS/validation/diagnose_capability_protocol.py``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.aurora import extract_prime_request
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.configuration import (
    AuroraConfig,
    build_specialist_instruction,
    build_tool_instruction,
)
from PRIMORA.consult import SpecialistRegistry, SpecialistSpec
from PRIMORA.prime import Prime
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.mind import Mind
from PRIMORA.prime.contract import PrimeRequest, RequestStatus
from PRIMORA.prime.providers import Provider
from PRIMORA.toolbox import ToolRegistry

_SEQ = [0]
_TMP = tempfile.TemporaryDirectory(prefix="one_request_")


def _db(name: str | None = None) -> str:
    _SEQ[0] += 1
    return os.path.join(_TMP.name, f"{name or 'req'}-{_SEQ[0]}.db")


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _config() -> AuroraConfig:
    return AuroraConfig(
        model_path=os.path.join("RESOURCES", "aurora", "v0.1", "model"),
        adapter_path=os.path.join("RESOURCES", "aurora", "v0.1",
                                  "aurora_qlora"),
    )


def block(action: str, payload: dict) -> str:
    """A well-formed single request block, as Aurora is told to emit."""
    return (
        "<prime_request>\n"
        + json.dumps({"action": action, "payload": payload,
                      "context": None, "metadata": None})
        + "\n</prime_request>"
    )


class _Provider(Provider):
    def invoke(self, capability, payload):
        from PRIMORA.prime.contract import PrimeResult
        return PrimeResult.success({"result": "9", "count": 4})


class ScriptedBrain(Brain):
    """Returns queued replies; the last one repeats once exhausted."""

    def __init__(self, *replies: str):
        self._replies = list(replies) or ["plain answer"]
        self.calls: list[list[dict]] = []

    def generate(self, messages, max_output_tokens=None):
        self.calls.append([dict(m) for m in messages])
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        reply = self._replies[index]
        if isinstance(reply, BrainResult):
            return reply
        return BrainResult.success(reply, "scripted")

    def get_model_name(self):
        return "scripted"

    def count_tokens(self, messages):
        return _counter(messages)


def _prime(with_tools: bool = False,
           with_specialists: bool = False) -> Prime:
    """A Prime with the optional catalogues wired at construction.

    ``tools`` and ``specialists`` are read-only properties, so they have to be
    supplied to the constructor rather than assigned afterwards.
    """
    return Prime(
        _Provider(),
        specialists=(
            SpecialistRegistry([
                SpecialistSpec(name="reasoning",
                               description="Careful analysis of a hard problem.",
                               model="m")])
            if with_specialists else None),
        tools=ToolRegistry.with_builtins() if with_tools else None,
    )


def _aurora(*replies: str, with_tools: bool = False,
            with_specialists: bool = False, **loop_kw) -> Aurora:
    prime = _prime(with_tools=with_tools, with_specialists=with_specialists)
    manager = ContextManager(db_path=_db())
    prime._context = manager
    prime._mind = Mind(manager)
    for name in ("recall", "learn"):
        specialist = prime.registry.get_specialist(name)
        if specialist is not None:
            specialist.bind(prime._mind)
    aurora = Aurora(prime, brain=ScriptedBrain(*replies), config=_config(),
                    context_manager=manager)
    for key, value in loop_kw.items():
        setattr(aurora.loop, key, value)
    return aurora


# ======================================================================
# 1. Exactly one request is extracted
# ======================================================================


class TestExactlyOneRequest:
    def test_single_block_is_extracted(self):
        text = block("recall", {"query": "berth code"})
        request = extract_prime_request(text)
        assert request is not None
        assert request.action == "recall"
        assert request.payload["query"] == "berth code"

    def test_one_block_surrounded_by_prose_still_extracts(self):
        """Prose around a single block is tolerated, not treated as a request."""
        text = ("Sure, let me check that.\n"
                + block("recall", {"query": "berth code"})
                + "\nOne moment.")
        request = extract_prime_request(text)
        assert request is not None
        assert request.action == "recall"

    def test_no_block_yields_nothing(self):
        assert extract_prime_request("just an answer") is None

    def test_malformed_json_yields_nothing(self):
        assert extract_prime_request(
            "<prime_request>{not json}</prime_request>") is None

    def test_missing_payload_yields_nothing(self):
        assert extract_prime_request(
            '<prime_request>{"action":"recall"}</prime_request>') is None


# ======================================================================
# 2. Multiple blocks are rejected safely
# ======================================================================


class TestMultipleBlocksRejected:
    def test_two_blocks_are_not_extracted(self):
        """The parser must not silently pick one of two."""
        text = (block("recall", {"query": "one"})
                + "\n"
                + block("research", {"query": "two"}))
        assert extract_prime_request(text) is None

    def test_loop_executes_nothing_when_two_blocks_emitted(self):
        aurora = _aurora(
            block("recall", {"query": "one"})
            + "\n"
            + block("research", {"query": "two"}),
            "final answer")
        result = aurora.respond([{"role": "user", "content": "go"}], "s")
        assert result.success
        assert aurora.loop.executor.calls == [], (
            "a malformed two-block reply must execute nothing")

    def test_multiple_requests_is_reported_distinctly(self):
        """Silently calling this 'finished' hid a discarded request."""
        aurora = _aurora(
            block("recall", {"query": "one"})
            + "\n"
            + block("research", {"query": "two"}),
            "final answer")
        aurora.respond([{"role": "user", "content": "go"}], "s")
        assert aurora.loop.last_trace.stop_reason == "multiple_blocks"

    def test_one_block_still_reports_finished(self):
        aurora = _aurora("just an answer")
        aurora.respond([{"role": "user", "content": "go"}], "s")
        assert aurora.loop.last_trace.stop_reason == "finished"

    def test_three_blocks_are_rejected_too(self):
        text = "".join(block("recall", {"query": str(i)}) for i in range(3))
        assert extract_prime_request(text) is None


# ======================================================================
# 3. Multi-step runs across separate cycles
# ======================================================================


class TestMultiStepAcrossCycles:
    def test_two_requests_in_two_cycles_both_execute(self):
        aurora = _aurora(
            block("calculate", {"expression": "1+1"}),
            block("calculate", {"expression": "2+2"}),
            "the answers are 2 and 4",
        )
        aurora.loop.max_cycles = 8
        aurora.respond([{"role": "user", "content": "add then double"}], "s")
        actions = [a for a, _ in aurora.loop.executor.calls]
        assert actions == ["calculate", "calculate"]

    def test_each_request_sees_the_previous_result(self):
        aurora = _aurora(
            block("calculate", {"expression": "1+1"}),
            block("calculate", {"expression": "2+2"}),
            "done",
        )
        aurora.loop.max_cycles = 8
        aurora.respond([{"role": "user", "content": "add then double"}], "s")
        # Two requests means three generations (request, request, answer), and
        # the second generation's prompt must carry the first result.
        assert len(aurora._brain.calls) == 3
        second_prompt = aurora._brain.calls[1]
        tool_turns = [m for m in second_prompt if m.get("role") == "tool"]
        assert tool_turns, "the first result was not fed back to Aurora"
        assert '"status": "success"' in tool_turns[0]["content"]
        # calculate really ran, so the first result is 2 from 1+1.
        assert "2" in tool_turns[0]["content"]

    def test_a_second_block_in_one_generation_is_not_a_second_step(self):
        """Queueing both steps at once is the failure being fixed."""
        aurora = _aurora(
            block("calculate", {"expression": "1+1"})
            + "\n" + block("calculate", {"expression": "2+2"}),
            "done",
        )
        aurora.respond([{"role": "user", "content": "add then double"}], "s")
        assert len(aurora.loop.executor.calls) == 0
        assert aurora.loop.last_trace.stop_reason == "multiple_blocks"


# ======================================================================
# 4. consult_specialist request format
# ======================================================================


class TestSpecialistRequestFormat:
    def test_specialist_name_is_not_the_action(self):
        request = extract_prime_request(block(
            "consult_specialist",
            {"specialist": "reasoning", "task": "compare storage options"}))
        assert request is not None
        assert request.action == "consult_specialist"
        assert request.payload["specialist"] == "reasoning"

    def test_consult_specialist_executes(self):
        aurora = _aurora(
            block("consult_specialist",
                  {"specialist": "reasoning", "task": "compare"}),
            "the answer", with_specialists=True)
        aurora.respond([{"role": "user", "content": "which database?"}], "s")
        actions = [a for a, _ in aurora.loop.executor.calls]
        assert actions == ["consult_specialist"]

    def test_bare_specialist_name_as_action_is_rejected(self):
        """'action':'reasoning' must not resolve to anything."""
        aurora = _aurora(block("reasoning", {"task": "compare"}), "text",
                         with_specialists=True)
        result = aurora.respond(
            [{"role": "user", "content": "which database?"}], "s")
        # executor.calls logs every dispatch attempt, so the point is not that
        # nothing was attempted but that the attempt failed and no specialist
        # result ever reached Aurora.
        attempted = [a for a, _ in aurora.loop.executor.calls]
        assert attempted == ["reasoning"]
        assert "consult_specialist" not in attempted
        assert "Considered" not in (result.content or "")

    def test_instruction_states_the_wrapper(self):
        class _P(Provider):
            def invoke(self, capability, payload):
                from PRIMORA.prime.contract import PrimeResult
                return PrimeResult.success({})

        prime = _P()
        prime.specialists = SpecialistRegistry([
            SpecialistSpec(name="reasoning", description="d", model="m")])
        text = build_specialist_instruction(prime)
        assert "consult_specialist" in text
        assert "payload.specialist" in text
        assert "never the specialist's own name" in text


# ======================================================================
# 5. use_tool wrapper format
# ======================================================================


class TestUseToolWrapperFormat:
    def test_tool_name_travels_in_the_payload(self):
        request = extract_prime_request(block(
            "use_tool",
            {"name": "count_words",
             "arguments": {"text": "the quick brown fox"}}))
        assert request is not None
        assert request.action == "use_tool"
        assert request.payload["name"] == "count_words"

    def test_use_tool_executes_and_the_result_reaches_the_answer(self):
        aurora = _aurora(
            block("use_tool", {"name": "count_words",
                               "arguments": {"text": "the quick brown fox"}}),
            "9 words.", with_tools=True)
        result = aurora.respond(
            [{"role": "user", "content": "how many words?"}], "s")
        actions = [a for a, _ in aurora.loop.executor.calls]
        assert actions == ["use_tool"]
        assert "9" in (result.content or "")

    def test_bare_tool_name_as_action_is_rejected(self):
        """The exact real-model failure: action 'count_words'."""
        aurora = _aurora(block("count_words", {"text": "a b c"}), "text",
                         with_tools=True)
        result = aurora.respond(
            [{"role": "user", "content": "how many words?"}], "s")
        attempted = [a for a, _ in aurora.loop.executor.calls]
        assert attempted == ["count_words"]
        assert "use_tool" not in attempted
        # The tool never ran, so no tool output can have reached the answer.
        assert "3" not in (result.content or "")

    def test_rejection_tells_her_the_right_wrapper(self):
        """The rejection is her only feedback, so it has to be actionable."""
        prime = _prime(with_tools=True)
        out = prime.handle(
            PrimeRequest("count_words", {"text": "a b c"}))
        assert out.status is RequestStatus.FAILURE
        message = out.error
        assert "count_words" in message
        assert "use_tool" in message
        assert "payload.name" in message

    def test_instruction_states_the_wrapper(self):
        class _P(Provider):
            def invoke(self, capability, payload):
                from PRIMORA.prime.contract import PrimeResult
                return PrimeResult.success({})

        prime = _P()
        prime.tools = ToolRegistry.with_builtins()
        text = build_tool_instruction(prime)
        assert "use_tool" in text
        assert "never the action" in text


# ======================================================================
# 6. Narrated intent is not a capability call
# ======================================================================


class TestNarratedIntent:
    @pytest.mark.parametrize("narrated", [
        "I'll research the latest version.",
        "Let me ask the reasoning specialist about this.",
        "I will retrieve that from memory.",
        "I'll check the available tools first.",
        "Let me look that up for you.",
    ])
    def test_narration_executes_nothing(self, narrated):
        aurora = _aurora(narrated, "eventually answered")
        aurora.respond([{"role": "user", "content": "go"}], "s")
        assert aurora.loop.executor.calls == []

    def test_narration_is_treated_as_the_answer(self):
        aurora = _aurora("I'll look into that.")
        aurora.respond([{"role": "user", "content": "go"}], "s")
        assert aurora.loop.last_trace.stop_reason == "finished"
        assert aurora.loop.last_trace.segments[0].requested is False

    def test_narration_is_not_silently_retried_as_a_request(self):
        """Narration must not be re-parsed into a capability call."""
        aurora = _aurora("Let me ask the specialist.", "Let me ask again.")
        aurora.respond([{"role": "user", "content": "go"}], "s")
        assert aurora.loop.executor.calls == []
        assert len(aurora._brain.calls) == 1


# ======================================================================
# 7. A failed capability can still lead to another cycle
# ======================================================================


class TestFailureRecovery:
    def test_failed_capability_result_is_fed_back(self):
        aurora = _aurora(block("count_words", {"text": "a b c"}),
                         "let me use the tool properly", with_tools=True)
        aurora.respond([{"role": "user", "content": "how many words?"}], "s")
        # The bad action produced a tool turn, and the next generation saw it.
        assert len(aurora._brain.calls) >= 2
        assert "Unknown capability" in json.dumps(aurora._brain.calls[1])

    def test_aurora_can_recover_with_a_correct_request(self):
        aurora = _aurora(
            block("count_words", {"text": "a b c"}),
            block("use_tool", {"name": "count_words",
                               "arguments": {"text": "a b c"}}),
            "3 words.", with_tools=True)
        aurora.loop.max_cycles = 8
        result = aurora.respond(
            [{"role": "user", "content": "how many words?"}], "s")
        actions = [a for a, _ in aurora.loop.executor.calls]
        assert actions == ["count_words", "use_tool"]
        assert "3" in (result.content or "")

    def test_recovery_does_not_cost_a_second_generation_for_success(self):
        """A successful result is answered from immediately."""
        aurora = _aurora(block("use_tool", {"name": "count_words",
                                            "arguments": {"text": "a b"}}),
                         "2 words.", with_tools=True)
        aurora.respond([{"role": "user", "content": "how many words?"}], "s")
        assert len(aurora._brain.calls) == 2

    def test_a_failure_gets_one_more_cycle(self):
        """She is told what was wrong and needs a turn to use it."""
        aurora = _aurora(
            block("count_words", {"text": "a b c"}),
            "let me correct that", with_tools=True)
        aurora.loop.max_cycles = 8
        aurora.respond([{"role": "user", "content": "how many words?"}], "s")
        # 1: bad request. 2: she narrates the correction. 3: she gets another
        # turn and answers. Without the extra cycle this stops at 2.
        assert len(aurora._brain.calls) == 3

    def test_the_extra_cycle_is_granted_only_once(self):
        """Recovery must not become an unbounded retry loop."""
        aurora = _aurora(
            block("count_words", {"text": "a b c"}),
            "still thinking", "still thinking", "still thinking",
            with_tools=True)
        aurora.loop.max_cycles = 8
        aurora.respond([{"role": "user", "content": "how many words?"}], "s")
        assert len(aurora._brain.calls) == 3

    def test_recovery_never_forces_an_extra_capability_call(self):
        """Recovery grants a generation, never a call."""
        aurora = _aurora(
            block("count_words", {"text": "a b c"}),
            "let me try again", with_tools=True)
        aurora.loop.max_cycles = 8
        aurora.respond([{"role": "user", "content": "how many words?"}], "s")
        assert len(aurora._brain.calls) == 3
        # Still only the one rejected attempt: nothing extra was executed.
        assert len(aurora.loop.executor.calls) == 1

    def test_the_loop_does_not_inspect_the_executor_budget(self):
        """Budget ownership stays with the executor, not the loop."""
        import ast
        import inspect

        from PRIMORA.aurora import loop as loopmod
        tree = ast.parse(inspect.getsource(loopmod))
        run = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "run")
        source = ast.unparse(run)
        for banned in ("max_calls", "self._executor.used", "exhausted"):
            assert banned not in source, banned


# ======================================================================
# 8. Preserved guarantees
# ======================================================================


class TestPreserved:
    def test_capability_budget_still_enforced(self):
        from PRIMORA.aurora.loop import DEFAULT_CAPABILITY_BUDGET
        assert DEFAULT_CAPABILITY_BUDGET == 4

    def test_repeat_guard_still_stops_identical_requests(self):
        same = block("calculate", {"expression": "1+1"})
        aurora = _aurora(same, same, same, same, same, same, same, same)
        aurora.loop.max_cycles = 8
        aurora.respond([{"role": "user", "content": "go"}], "s")
        assert aurora.loop.last_trace.stop_reason == "repeated_request"
        assert len(aurora.loop.executor.calls) == 2

    def test_recall_fix_still_in_place(self):
        from PRIMORA.prime.context.manager import INTERNAL_ROLES
        assert INTERNAL_ROLES == frozenset({"tool", "system"})

    def test_max_cycles_default_unchanged(self):
        from PRIMORA.aurora.loop import DEFAULT_MAX_CYCLES
        assert DEFAULT_MAX_CYCLES == 8

    def test_system_prompt_fits_the_context_budget(self):
        """A protocol that overflows the ceiling breaks every request.

        The composed prompt must leave room for a conversation inside the
        measured input ceiling, or every build raises ContextTooLargeError
        before Aurora ever gets to choose.
        """
        from transformers import AutoTokenizer

        class _P(Provider):
            def invoke(self, capability, payload):
                from PRIMORA.prime.contract import PrimeResult
                return PrimeResult.success({})

        prime = _P()
        prime.tools = ToolRegistry.with_builtins()
        prime.specialists = SpecialistRegistry([
            SpecialistSpec(name="reasoning",
                           description="Careful analysis of a hard problem.",
                           model="m")])
        composed = AuroraConfig.system_instruction
        composed += build_tool_instruction(prime)
        composed += build_specialist_instruction(prime)
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(r"C:\PRIMORA PROJECT", "RESOURCES", "aurora", "v0.1",
                         "model"), trust_remote_code=True)
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": composed}],
            tokenize=False, add_generation_prompt=True)
        tokens = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        config = _config()
        assert tokens < config.context_target, (
            f"composed prompt is {tokens} tokens, at or above the "
            f"{config.context_target} token target")