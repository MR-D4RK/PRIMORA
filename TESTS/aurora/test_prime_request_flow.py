import os
import sys
import json
import tempfile
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.aurora import extract_prime_request
from PRIMORA.prime import Prime
from PRIMORA.prime.providers import Provider
from PRIMORA.prime.contract import PrimeResult, RequestStatus
from PRIMORA.prime.capability import CapabilityRegistry, CapabilitySpec
from PRIMORA.aurora.brain import Brain, BrainResult
from typing import Any


VALID_BLOCK = (
    '<prime_request>\n'
    '{"action": "calculate", "payload": {"expression": "2 + 3"}, "context": null, "metadata": null}\n'
    '</prime_request>'
)

CHAT_BLOCK = (
    '<prime_request>\n'
    '{"action": "chat", "payload": {"messages": []}, "context": null, "metadata": null}\n'
    '</prime_request>'
)


class MockProvider(Provider):
    def __init__(self, results: dict):
        self._results = results
        self.calls = []

    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        self.calls.append((capability, payload))
        if capability in self._results:
            return self._results[capability]
        return PrimeResult.failure(f"Unknown capability: {capability}")


class MockScriptedBrain(Brain):
    def __init__(self, outputs: list, model: str = "test-model"):
        self._outputs = list(outputs)
        self._model = model
        self.seen_messages = []

    def generate(self, messages: list[dict[str, str]], max_output_tokens=None) -> BrainResult:
        self.seen_messages.append(list(messages))
        return self._outputs.pop(0)

    def get_model_name(self) -> str:
        return self._model


_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def _test_context():
    from PRIMORA.prime.context import ContextManager
    fd, path = tempfile.mkstemp(suffix=".db", dir=_TEST_TMPDIR.name)
    os.close(fd)
    return ContextManager(db_path=path)


def test_extract_valid_request():
    req = extract_prime_request("Here you go:\n" + VALID_BLOCK + "\nDone.")
    assert req is not None
    assert req.action == "calculate"
    assert req.payload == {"expression": "2 + 3"}
    assert req.context is None
    assert req.metadata is None


def test_extract_valid_minimal_request():
    req = extract_prime_request('<prime_request>{"action": "calculate", "payload": {"expression": "2+2"}}</prime_request>')
    assert req is not None
    assert req.action == "calculate"
    assert req.payload == {"expression": "2+2"}


def test_extract_no_block():
    assert extract_prime_request("Just a normal chat reply.") is None


def test_extract_malformed_json():
    assert extract_prime_request("<prime_request>{not valid json</prime_request>") is None


def test_extract_json_array_not_object():
    assert extract_prime_request('<prime_request>["calculate"]</prime_request>') is None


def test_extract_missing_action():
    assert extract_prime_request('<prime_request>{"payload": {}}</prime_request>') is None


def test_extract_empty_action():
    assert extract_prime_request('<prime_request>{"action": "", "payload": {}}</prime_request>') is None


def test_extract_non_string_action():
    assert extract_prime_request('<prime_request>{"action": 42, "payload": {}}</prime_request>') is None


def test_extract_missing_payload():
    assert extract_prime_request('<prime_request>{"action": "calculate"}</prime_request>') is None


def test_extract_non_object_payload():
    assert extract_prime_request('<prime_request>{"action": "calculate", "payload": "2+3"}</prime_request>') is None
    assert extract_prime_request('<prime_request>{"action": "calculate", "payload": [1, 2]}</prime_request>') is None
    assert extract_prime_request('<prime_request>{"action": "calculate", "payload": null}</prime_request>') is None


def test_extract_prose_does_not_trigger():
    assert extract_prime_request("I'll ask Prime to handle this for you.") is None
    assert extract_prime_request("I will give the numbers to Prime right away.") is None


def test_extract_bare_request_object_executes():
    """A request that arrives without its wrapper is still a request.

    Observed live: asking Aurora a memory question sometimes produced
    ``{"action":"recall","payload":{"query":"current favorite color"},
    "context":null,"metadata":null}`` printed verbatim as her answer,
    with no capability executed. A Qwen base trained for tool calling
    emits the request object itself, and OpenAI-compatible servers never
    return structured tool calls because no ``tools`` are ever sent -
    the text is the only channel, so the object alone must count.

    It is recognised because the whole reply is exactly one valid
    request object, which is the same authorization path as a wrapped
    block; not because the reply happens to be JSON.
    """
    req = extract_prime_request(
        '{"action":"recall","payload":{"query":"current favorite color"},'
        '"context":null,"metadata":null}')
    assert req is not None
    assert req.action == "recall"
    assert req.payload == {"query": "current favorite color"}
    assert req.context is None and req.metadata is None


def test_extract_bare_request_rejects_everything_that_is_not_one_object():
    """The wrapper-free path is not a blanket "execute any JSON" rule."""
    # Not the whole reply: JSON quoted inside an answer is an answer.
    assert extract_prime_request(
        'I would send {"action":"recall","payload":{"query":"x"}} for that.'
    ) is None
    # A JSON answer is not a request: no action, no payload key.
    assert extract_prime_request('{"config": {"depth": 2}}') is None
    # Two objects is ambiguous, so neither runs.
    assert extract_prime_request(
        '{"action":"recall","payload":{"query":"a"}}'
        '{"action":"recall","payload":{"query":"b"}}') is None
    # Arrays and truncated objects are not requests either.
    assert extract_prime_request('["recall"]') is None
    assert extract_prime_request('{"action":"recall","payload":{"query":"col') is None
    # Shape rules are identical to the wrapped path.
    assert extract_prime_request('{"action":""}') is None
    assert extract_prime_request('{"action": 42, "payload": {}}') is None
    assert extract_prime_request('{"action": "recall"}') is None
    assert extract_prime_request('{"action":"recall","payload":"x"}') is None
    assert extract_prime_request('{"action":"recall","payload":{},'
                                 '"context": "not a dict"}') is None


def test_extract_multiple_blocks_rejected():
    text = VALID_BLOCK + "\n" + VALID_BLOCK
    assert extract_prime_request(text) is None


def test_extract_non_string_input():
    assert extract_prime_request(None) is None
    assert extract_prime_request(123) is None


def test_respond_valid_request_reaches_prime():
    provider = MockProvider({})
    prime = Prime(provider)

    brain = MockScriptedBrain([
        BrainResult.success(VALID_BLOCK, "m"),
        BrainResult.success("Done.", "m"),
    ])

    aurora = Aurora(prime, brain=brain, context_manager=_test_context())
    result = aurora.respond([
        {"role": "user", "content": "What is 2+3?"}
    ])

    assert result.success
    assert result.content == "Done."
    assert len(provider.calls) == 0


def test_respond_prime_result_returned_to_brain():
    provider = MockProvider({"calculate": PrimeResult.success({"result": 5})})
    prime = Prime(provider)
    brain = MockScriptedBrain([
        BrainResult.success(VALID_BLOCK, "m"),
        BrainResult.success("Done: 5", "m"),
    ])
    aurora = Aurora(prime, brain=brain, context_manager=_test_context())
    aurora.respond([{"role": "user", "content": "Compute"}])
    assert len(brain.seen_messages) == 2
    followup = brain.seen_messages[1]
    tool_msgs = [m for m in followup if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    tool_data = json.loads(tool_msgs[0]["content"])
    assert tool_data["status"] == "success"
    assert tool_data["output"] == {"result": 5}
    assert tool_data["error"] is None


def test_respond_multiple_blocks_no_execution():
    provider = MockProvider({"calculate": PrimeResult.success({"result": 5})})
    prime = Prime(provider)
    brain = MockScriptedBrain([BrainResult.success(VALID_BLOCK + "\n" + VALID_BLOCK, "m")])
    aurora = Aurora(prime, brain=brain, context_manager=_test_context())
    result = aurora.respond([{"role": "user", "content": "Hi"}])
    assert len(provider.calls) == 0
    assert len(brain.seen_messages) == 1
    assert result.success


def test_respond_prose_no_execution():
    provider = MockProvider({"calculate": PrimeResult.success({"result": 5})})
    prime = Prime(provider)
    brain = MockScriptedBrain([BrainResult.success("I'll ask Prime to handle this.", "m")])
    aurora = Aurora(prime, brain=brain, context_manager=_test_context())
    result = aurora.respond([{"role": "user", "content": "Hi"}])
    assert len(provider.calls) == 0
    assert result.content == "I'll ask Prime to handle this."


def test_respond_plain_chat_unchanged():
    provider = MockProvider({})
    prime = Prime(provider)
    brain = MockScriptedBrain([BrainResult.success("Hello there!", "m")])
    aurora = Aurora(prime, brain=brain, context_manager=_test_context())
    result = aurora.respond([{"role": "user", "content": "Hi"}])
    assert len(provider.calls) == 0
    assert len(brain.seen_messages) == 1
    assert result.success
    assert result.content == "Hello there!"


def test_respond_first_think_failure_no_execution():
    provider = MockProvider({"calculate": PrimeResult.success({"result": 5})})
    prime = Prime(provider)
    brain = MockScriptedBrain([BrainResult.failure("Model unavailable", "m")])
    aurora = Aurora(prime, brain=brain, context_manager=_test_context())
    result = aurora.respond([{"role": "user", "content": "Hi"}])
    assert len(provider.calls) == 0
    assert not result.success
    assert result.error == "Model unavailable"


def test_respond_malformed_block_no_execution():
    provider = MockProvider({"calculate": PrimeResult.success({"result": 5})})
    prime = Prime(provider)
    brain = MockScriptedBrain([BrainResult.success("<prime_request>{broken</prime_request>", "m")])
    aurora = Aurora(prime, brain=brain, context_manager=_test_context())
    result = aurora.respond([{"role": "user", "content": "Hi"}])
    assert len(provider.calls) == 0
    assert len(brain.seen_messages) == 1


def test_respond_context_metadata_pass_through():
    provider = MockProvider({})
    # Register calculate capability with a mock specialist for testing
    registry = CapabilityRegistry()
    class MockCalculateSpecialist:
        name = "calculate"
        def handle(self, payload):
            return PrimeResult.success({"result": 5})
    registry._specialists = {"calculate": MockCalculateSpecialist()}
    registry._capabilities["calculate"] = CapabilitySpec(
        name="calculate",
        description="Test calculate capability",
        payload_schema={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        required_fields=["expression"],
        optional_fields=[]
    )
    prime = Prime(provider)
    prime._registry = registry
    block = ('<prime_request>{"action": "calculate", "payload": {"expression": "2+2"}, '
             '"context": {"session": "1"}, "metadata": {"trace": "a"}}</prime_request>')
    brain = MockScriptedBrain([
        BrainResult.success(block, "m"),
        BrainResult.success("ok", "m"),
    ])
    aurora = Aurora(prime, brain=brain, context_manager=_test_context())
    aurora.respond([{"role": "user", "content": "Hi"}])
    assert len(provider.calls) == 0  # Specialist handles it, not provider


# ======================================================================
# Wrapper-less request objects (Stage 3A)
#
# The live defect: a Qwen base emits the request object itself, the
# OpenAI backend has no structured tool-call channel (no ``tools`` are
# ever sent), and the loop read the object as a finished answer - so a
# recall request reached the user as printed JSON with nothing executed.
# These tests pin the fix and its boundaries.
# ======================================================================


BARE_RECALL = (
    '{"action":"recall","payload":{"query":"current favorite color"},'
    '"context":null,"metadata":null}'
)


def _isolated_aurora(*replies):
    """An Aurora whose Mind and recall capability share one temp store.

    Mirrors the wiring in test_phase_a_repair, so recall answers from the
    isolated database instead of the production one.
    """
    from PRIMORA.prime.context import ContextManager
    from PRIMORA.prime.context.mind import Mind
    from PRIMORA.prime.context import state as state_mod

    provider = MockProvider({})
    prime = Prime(provider)
    manager = ContextManager(db_path=_test_context()._db_path
                             if hasattr(_test_context(), "_db_path") else _db())
    prime._mind = Mind(manager)
    prime._context = manager
    recall = prime.registry.get_specialist("recall")
    if recall is not None:
        recall.bind(prime._mind)
    # One stored fact, so a real recall has something to return.
    store = prime._mind.mechanisms._get_store()
    state_mod.apply_user_candidates(
        store, "stage3a", 0,
        [state_mod.StateCandidate("favorite color", "yellow", "establish",
                                  "global", "FACT", slot="favorite color")])
    return Aurora(prime, brain=MockScriptedBrain(list(replies)),
                  context_manager=manager)


def _db():
    fd, path = tempfile.mkstemp(suffix=".db", dir=_TEST_TMPDIR.name)
    os.close(fd)
    return path


def test_bare_request_object_is_executed_not_printed():
    """The reported defect, end to end: nothing ran and JSON was shown."""
    aurora = _isolated_aurora(
        BrainResult.success(BARE_RECALL, "m"),
        BrainResult.success("Your favorite color is yellow.", "m"),
    )
    result = aurora.respond(
        [{"role": "user", "content": "What is my current favorite color?"}],
        "stage3a")

    assert [(a, p) for a, p in aurora.loop.executor.calls] == [
        ("recall", {"query": "current favorite color"})], (
        "a wrapper-less request object must execute through the normal path")
    assert result.success
    assert result.content == "Your favorite color is yellow."
    assert not result.content.startswith("{"), (
        "the JSON object must not reach the user as the answer")
    assert aurora.loop.last_trace.stop_reason == "finished"

    # The recall result really fed back to Aurora as the tool turn.
    followup = aurora._brain.seen_messages[-1]
    tool_msgs = [m for m in followup if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert '"status": "success"' in tool_msgs[0]["content"]


def test_bare_request_object_in_both_forms_executes_once():
    """The same request in a block and as bare text is one execution."""
    wrapped = "<prime_request>\n" + BARE_RECALL + "\n</prime_request>"
    aurora = _isolated_aurora(
        BrainResult.success(wrapped + "\n" + BARE_RECALL, "m"),
        BrainResult.success("done", "m"),
    )
    aurora.respond([{"role": "user", "content": "q"}], "stage3a")
    assert len(aurora.loop.executor.calls) == 1, (
        "a duplicated request must not execute twice")


def test_two_bare_request_objects_execute_nothing():
    aurora = _isolated_aurora(
        BrainResult.success(BARE_RECALL + "\n" + BARE_RECALL, "m"),
        BrainResult.success("I could not run that.", "m"),
    )
    result = aurora.respond([{"role": "user", "content": "q"}], "stage3a")
    assert aurora.loop.executor.calls == []
    # Corrected rather than printed: the second segment is the answer.
    assert result.content == "I could not run that."
    assert len(aurora._brain.seen_messages) == 2


def test_malformed_bare_request_never_reaches_the_user_as_an_answer():
    truncated = '{"action":"recall","payload":{"query":"col'
    aurora = _isolated_aurora(
        BrainResult.success(truncated, "m"),
        BrainResult.success("Sorry, I could not run that.", "m"),
    )
    result = aurora.respond([{"role": "user", "content": "q"}], "stage3a")
    assert aurora.loop.executor.calls == []
    assert result.content == "Sorry, I could not run that.", (
        "a truncated request is not an answer")
    # She was told it did not run, in a tool turn.
    followup = aurora._brain.seen_messages[-1]
    tool_msgs = [m for m in followup if m.get("role") == "tool"]
    assert tool_msgs and "not_a_request" in tool_msgs[0]["content"]


def test_repeated_malformed_request_stops_instead_of_printing():
    truncated = '{"action":"recall","payload":{"query":"col'
    aurora = _isolated_aurora(
        BrainResult.success(truncated, "m"),
        BrainResult.success(truncated, "m"),
    )
    result = aurora.respond([{"role": "user", "content": "q"}], "stage3a")
    assert aurora.loop.executor.calls == []
    assert not result.success, (
        "insisting on an unusable request is a failure, not an answer")
    assert aurora.loop.last_trace.stop_reason == "protocol_error"


def test_bare_unknown_action_is_refused_by_prime():
    aurora = _isolated_aurora(
        BrainResult.success('{"action":"definitely_not_a_capability",'
                            '"payload":{"q":"x"}}', "m"),
        BrainResult.success("I could not do that.", "m"),
    )
    result = aurora.respond([{"role": "user", "content": "q"}], "stage3a")
    # It reached Prime, which is the only place that can authorize it,
    # and Prime refused it by name.
    assert [a for a, _ in aurora.loop.executor.calls] == [
        "definitely_not_a_capability"]
    assert result.success
    assert result.content == "I could not do that."
    followup = aurora._brain.seen_messages[-1]
    tool_msgs = [m for m in followup if m.get("role") == "tool"]
    assert tool_msgs and "Unknown capability" in tool_msgs[0]["content"]


def test_plain_and_json_shaped_answers_are_untouched():
    for answer in ("How can I help?",
                   '{"config": {"depth": 2}}',
                   '["recall"]',
                   'I would send {"action":"recall","payload":{}} for that.'):
        aurora = _isolated_aurora(BrainResult.success(answer, "m"))
        result = aurora.respond([{"role": "user", "content": "q"}], "stage3a")
        assert aurora.loop.executor.calls == [], answer
        assert result.content == answer, answer
        assert result.success, answer
        assert aurora.loop.last_trace.stop_reason == "finished", answer


if __name__ == "__main__":
    test_extract_valid_request()
    test_extract_valid_minimal_request()
    test_extract_no_block()
    test_extract_malformed_json()
    test_extract_json_array_not_object()
    test_extract_missing_action()
    test_extract_empty_action()
    test_extract_non_string_action()
    test_extract_missing_payload()
    test_extract_non_object_payload()
    test_extract_prose_does_not_trigger()
    test_extract_bare_json_outside_block_ignored()
    test_extract_multiple_blocks_rejected()
    test_extract_non_string_input()
    test_respond_valid_request_reaches_prime()
    test_respond_prime_result_returned_to_brain()
    test_respond_multiple_blocks_no_execution()
    test_respond_prose_no_execution()
    test_respond_plain_chat_unchanged()
    test_respond_first_think_failure_no_execution()
    test_respond_malformed_block_no_execution()
    test_respond_context_metadata_pass_through()
    print("All Prime-request flow tests passed!")
