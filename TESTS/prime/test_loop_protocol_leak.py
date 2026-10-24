"""One request runs once, and protocol payloads never become the answer.

Production evidence (session `default`, messages 4723/4746 and 6693/6737):
the same recall request executed twice in one turn, and when the model asked
a third time the loop stopped on its cycle bound and returned that raw
``<prime_request>`` block as the reply. The CLI printed it as Aurora's answer
(session transcript shows the turn ending on the request, not an answer).

These tests pin both halves of the fix:
* an identical request after a successful execution is not run again;
* a stop that has no answer (cycle bound, repeat bound, malformed protocol)
  returns an honest failure, never the internal payload.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.providers import Provider

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]

REQUEST_MARKER = "<prime_request>"


def _db(name="loop"):
    _SEQ[0] += 1
    return os.path.join(_TMP.name, "%s-%d.db" % (name, _SEQ[0]))


def _config():
    return AuroraConfig(
        model_path=os.path.join("RESOURCES", "aurora", "v0.1", "model"),
        adapter_path=os.path.join("RESOURCES", "aurora", "v0.1", "aurora_qlora"),
    )


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.success({"value": 4})


class ScriptedBrain(Brain):
    def __init__(self, *replies):
        self._replies = list(replies) or ["plain answer"]
        self.calls = []

    def generate(self, messages, max_output_tokens=None, on_delta=None,
                 cancelled=None):
        self.calls.append([dict(m) for m in messages])
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        reply = self._replies[index]
        if isinstance(reply, BrainResult):
            return reply
        return BrainResult.success(reply, "scripted")

    def get_model_name(self):
        return "scripted"

    def count_tokens(self, messages):
        return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _request(expression="2+2"):
    return (
        "<prime_request>\n"
        '{"action":"calculate","payload":{"expression":"%s"},'
        '"context":null,"metadata":null}\n'
        "</prime_request>" % expression
    )


def _aurora(brain, manager=None, max_cycles=None):
    prime = Prime(_Provider())
    a = Aurora(
        prime, brain=brain, config=_config(),
        context_manager=manager or ContextManager(db_path=_db()),
    )
    if max_cycles is not None:
        a.loop.max_cycles = max_cycles
    return a


def _assert_not_internal(result):
    assert result.success is False
    assert REQUEST_MARKER not in result.content
    assert REQUEST_MARKER not in (result.error or "")


# ======================================================================
# 1. No redundant execution after success
# ======================================================================
def test_identical_request_after_success_is_not_executed_again():
    request = _request("2+2")
    brain = ScriptedBrain(request, request, "The answer is 4.")
    aurora = _aurora(brain, max_cycles=8)
    result = aurora.respond(
        [{"role": "user", "content": "what is 2+2?"}], "s")
    assert result.success
    assert result.content == "The answer is 4."
    assert len(aurora.loop.executor.calls) == 2
    assert len(brain.calls) == 3
    assert aurora.loop.last_trace.stop_reason == "finished"


def test_repeat_after_failure_is_still_allowed():
    # A retry after a failed call is a legitimate second attempt, not the
    # redundant success repeat this guard is about. "red+yellow" is not a
    # valid expression, so calculate fails both times.
    request = _request("red+yellow")
    brain = ScriptedBrain(request, request, "done")
    aurora = _aurora(brain, max_cycles=8)
    result = aurora.respond(
        [{"role": "user", "content": "mix red and yellow"}], "s")
    assert len(aurora.loop.executor.calls) == 2
    assert result.content == "done"


# ======================================================================
# 2. A stop with no answer must never leak the payload
# ======================================================================
def test_repeated_identical_request_never_returns_raw_payload():
    request = _request("2+2")
    brain = ScriptedBrain(*([request] * 6))
    aurora = _aurora(brain, max_cycles=8)
    result = aurora.respond(
        [{"role": "user", "content": "what is 2+2?"}], "s")
    _assert_not_internal(result)
    assert len(aurora.loop.executor.calls) == 2
    assert aurora.loop.last_trace.stop_reason == "repeated_request"


def test_cycle_bound_never_returns_raw_payload():
    requests = [_request("%d+%d" % (i, i)) for i in range(1, 10)]
    brain = ScriptedBrain(*requests)
    aurora = _aurora(brain, max_cycles=3)
    result = aurora.respond(
        [{"role": "user", "content": "calculate"}], "s")
    _assert_not_internal(result)
    assert len(aurora.loop.executor.calls) == 2
    assert aurora.loop.last_trace.stop_reason == "max_cycles"


def test_multiple_request_blocks_never_return_raw_payload():
    brain = ScriptedBrain(_request("2+2") + "\n" + _request("3+3"))
    aurora = _aurora(brain)
    result = aurora.respond(
        [{"role": "user", "content": "calculate"}], "s")
    _assert_not_internal(result)
    assert aurora.loop.last_trace.stop_reason == "multiple_blocks"


def test_malformed_request_block_never_returns_raw_payload():
    brain = ScriptedBrain(
        "<prime_request>\n{\"action\": \"calculate\"\n</prime_request>")
    aurora = _aurora(brain)
    result = aurora.respond(
        [{"role": "user", "content": "calculate"}], "s")
    _assert_not_internal(result)
    assert aurora.loop.last_trace.stop_reason == "protocol_error"


def test_plain_json_answer_is_not_mistaken_for_a_request():
    brain = ScriptedBrain('{"status": "ok", "value": 4}')
    aurora = _aurora(brain)
    result = aurora.respond(
        [{"role": "user", "content": "status?"}], "s")
    assert result.success
    assert result.content == '{"status": "ok", "value": 4}'


if __name__ == "__main__":
    test_identical_request_after_success_is_not_executed_again()
    test_repeat_after_failure_is_still_allowed()
    test_repeated_identical_request_never_returns_raw_payload()
    test_cycle_bound_never_returns_raw_payload()
    test_multiple_request_blocks_never_return_raw_payload()
    test_malformed_request_block_never_returns_raw_payload()
    test_plain_json_answer_is_not_mistaken_for_a_request()
    print("All loop protocol tests passed!")
