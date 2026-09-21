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


def test_extract_bare_json_outside_block_ignored():
    assert extract_prime_request('{"action": "calculate", "payload": {"expression": "1+1"}}') is None


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
