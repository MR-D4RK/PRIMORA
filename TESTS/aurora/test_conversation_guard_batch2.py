"""Isolated regression tests for unwanted research, grounding, and stale replies."""
import json

from PRIMORA.aurora.aurora import extract_prime_request
from PRIMORA.aurora.brain import BrainResult
from PRIMORA.aurora.configuration import DEFAULT_SYSTEM_INSTRUCTION
from PRIMORA.aurora.conversation_guard import (
    reject_unrequested_lookup, repeats_answer_after_correction,
)
from PRIMORA.aurora.loop import CognitiveLoop
from PRIMORA.prime.context import BudgetConfig, ContextManager, Mind


def _request(action="research"):
    return ("<prime_request>" + json.dumps({
        "action": action, "payload": {"query": "why people love bright purple"},
        "context": None, "metadata": None,
    }) + "</prime_request>")


class ScriptedBrain:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, messages, max_output_tokens=None, on_delta=None):
        self.calls.append(list(messages))
        return BrainResult.success(self.replies.pop(0), "scripted")

    def get_model_name(self):
        return "scripted"

    def count_tokens(self, messages):
        return sum(len(str(m.get("content", "")).split()) + 2 for m in messages)


class Executor:
    def __init__(self):
        self.calls = []

    def begin_session(self, *_):
        pass

    def execute(self, request):
        self.calls.append(request.action)
        return json.dumps({"status": "success", "output": "Sources returned"})


def _run(tmp_path, messages, replies, *, max_cycles=3):
    manager = ContextManager(db_path=str(tmp_path / "test.db"))
    brain = ScriptedBrain(*replies)
    executor = Executor()
    loop = CognitiveLoop(
        Mind(manager), brain, executor=executor, max_cycles=max_cycles,
        max_new_tokens=96, request_extractor=extract_prime_request,
        token_counter=brain.count_tokens,
        budget=BudgetConfig(input_target=700, input_ceiling=1300,
                            output_reserve=96, total_budget=1500),
    )
    result = loop.run(messages, session_id="isolated")
    return result, executor, brain, loop


def test_preference_statement_does_not_trigger_research(tmp_path):
    result, executor, brain, _ = _run(
        tmp_path, [{"role": "user", "content": "I love bright purple too"}],
        [_request(), "I understand you love bright purple too."],
    )
    assert result.success
    assert not executor.calls
    assert len(brain.calls) == 2
    assert "not_requested" in str(brain.calls[1])


def test_clarification_does_not_trigger_research(tmp_path):
    result, executor, *_ = _run(
        tmp_path, [{"role": "user", "content":
                   "No, the feeling is quite like violet color for me"}],
        [_request(), "I see: bright purple feels like violet to you."],
    )
    assert result.success and not executor.calls


def test_explicit_research_still_runs(tmp_path):
    result, executor, *_ = _run(
        tmp_path, [{"role": "user", "content":
                   "I love bright purple; please research why people like it"}],
        [_request(), "Here's what I found."],
    )
    assert result.success
    assert executor.calls == ["research"]


def test_non_research_capability_is_unchanged():
    request = extract_prime_request(_request("learn"))
    assert reject_unrequested_lookup(request, "I love purple too") is None


def test_repeat_after_correction_gets_one_retry(tmp_path):
    previous = "The research did not find any sources explaining why people love bright purple."
    messages = [
        {"role": "user", "content": "I love bright purple"},
        {"role": "assistant", "content": previous},
        {"role": "user", "content": "No the feeling is quite like violet color for me"},
    ]
    result, executor, brain, _ = _run(
        tmp_path, messages, [previous, "I understand: it feels similar to violet."],
    )
    assert result.success
    assert result.content.startswith("I understand")
    assert len(brain.calls) == 2
    assert not executor.calls
    assert "stale_reply" in str(brain.calls[1])


def test_repeat_twice_fails_closed(tmp_path):
    previous = "Same old response"
    messages = [
        {"role": "assistant", "content": previous},
        {"role": "user", "content": "Actually, I feel different about this"},
    ]
    result, executor, _, loop = _run(
        tmp_path, messages, [previous, previous], max_cycles=3,
    )
    assert not result.success
    assert loop.last_trace.stop_reason == "stale_answer"
    assert not executor.calls


def test_final_cycle_never_leaks_a_stale_answer(tmp_path):
    old = "The research did not find sources."
    messages = [
        {"role": "assistant", "content": old},
        {"role": "user", "content": "No, I meant the color reminds me of violet"},
    ]
    result, _, _, loop = _run(tmp_path, messages, [old], max_cycles=1)
    assert not result.success
    assert loop.last_trace.stop_reason == "stale_answer"


def test_other_repeats_are_not_blocked():
    messages = [
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Hello!"},
    ]
    assert not repeats_answer_after_correction(messages, "Hello!")


def test_preference_grounding_instruction_requires_user_evidence():
    lower = DEFAULT_SYSTEM_INSTRUCTION.lower()
    assert "only give reasons they explicitly stated" in lower
    assert "do not know the reason" in lower
    assert "correction supersedes" in lower
