import sys
import os
import tempfile
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.configuration import AuroraConfig, DEFAULT_SYSTEM_INSTRUCTION
from PRIMORA.prime import Prime
from PRIMORA.prime.providers import Provider
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.capability import CapabilityRegistry, CapabilitySpec
from PRIMORA.prime.execution import Execution
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.prime.orchestrator import Orchestrator
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



def _make_test_prime(provider, registry=None):
    """Create a Prime instance with a custom registry for testing."""
    if registry is None:
        registry = _make_test_registry()
    # Register chat capability with mock specialist
    class MockChatSpecialist:
        name = "chat"
        def handle(self, payload):
            return PrimeResult.success({"result": "ok"})
    registry.register(
        CapabilitySpec(
            name="chat",
            description="Test chat capability",
            payload_schema={"type": "object", "properties": {"messages": {"type": "array"}}, "required": ["messages"]},
            required_fields=["messages"],
            optional_fields=[]
        ),
        MockChatSpecialist()
    )
    from PRIMORA.prime.orchestrator import Orchestrator
    from PRIMORA.prime.execution import Execution
    # Create execution with chat specialist
    execution = Execution(provider, specialists={"chat": MockChatSpecialist()})
    from PRIMORA.prime.orchestrator import Orchestrator
    orchestrator = Orchestrator(provider, registry)
    orchestrator._execution = execution
    
    prime = Prime.__new__(Prime)
    prime._orchestrator = orchestrator
    prime._registry = registry
    return prime


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


def _test_config(**kwargs):
    params = dict(
        model_path="/fake/model",
        adapter_path="/fake/adapter",
    )
    params.update(kwargs)
    return AuroraConfig(**params)


_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def _test_context():
    from PRIMORA.prime.context import ContextManager
    fd, path = tempfile.mkstemp(suffix=".db", dir=_TEST_TMPDIR.name)
    os.close(fd)
    return ContextManager(db_path=path)


def test_instruction_default_content():
    assert "Prime" in DEFAULT_SYSTEM_INSTRUCTION
    assert "exactly ONE" in DEFAULT_SYSTEM_INSTRUCTION
    assert "<prime_request>" in DEFAULT_SYSTEM_INSTRUCTION
    assert "more than one" in DEFAULT_SYSTEM_INSTRUCTION
    assert "never claim" in DEFAULT_SYSTEM_INSTRUCTION.lower()


def test_instruction_env_override():
    os.environ["PRIMORA_AURORA_SYSTEM_INSTRUCTION"] = "Custom instruction."
    try:
        config = AuroraConfig.from_env()
        assert config.system_instruction == "Custom instruction."
    finally:
        os.environ.pop("PRIMORA_AURORA_SYSTEM_INSTRUCTION", None)


def test_instruction_default_from_env():
    os.environ.pop("PRIMORA_AURORA_SYSTEM_INSTRUCTION", None)
    config = AuroraConfig.from_env()
    assert config.system_instruction == DEFAULT_SYSTEM_INSTRUCTION


def test_respond_prepends_instruction_once():
    provider = MockProvider({})
    prime = Prime(provider)
    brain = MockScriptedBrain([BrainResult.success("Hello!", "m")])
    aurora = Aurora(prime, brain=brain, config=_test_config(), context_manager=_test_context())
    aurora.respond([{"role": "user", "content": "Hi"}])

    first_seen = brain.seen_messages[0]
    assert first_seen[0]["role"] == "system"
    assert first_seen[0]["content"].startswith(DEFAULT_SYSTEM_INSTRUCTION)
    assert "Available Prime Capabilities:" in first_seen[0]["content"]
    assert "- calculate:" in first_seen[0]["content"]
    assert first_seen[1] == {"role": "user", "content": "Hi"}
    assert sum(1 for m in first_seen if m["role"] == "system") == 1


def test_respond_does_not_duplicate_caller_system():
    provider = MockProvider({})
    prime = Prime(provider)
    brain = MockScriptedBrain([BrainResult.success("Hello!", "m")])
    aurora = Aurora(prime, brain=brain, config=_test_config(), context_manager=_test_context())
    aurora.respond([
        {"role": "system", "content": "Caller system."},
        {"role": "user", "content": "Hi"},
    ])

    first_seen = brain.seen_messages[0]
    assert sum(1 for m in first_seen if m["role"] == "system") == 1
    assert first_seen[0]["content"].startswith("Caller system.")
    assert "Available Prime Capabilities:" in first_seen[0]["content"]


def test_respond_followup_keeps_single_instruction():
    provider = MockProvider({})
    prime = Prime(provider)
    brain = MockScriptedBrain([
        BrainResult.success(VALID_BLOCK, "m"),
        BrainResult.success("The answer is 5.", "m"),
    ])
    aurora = Aurora(prime, brain=brain, config=_test_config(), context_manager=_test_context())

    result = aurora.respond([{"role": "user", "content": "What is 2+3?"}])

    assert result.success
    assert result.content == "The answer is 5."

    followup = brain.seen_messages[1]
    systems = [m for m in followup if m["role"] == "system"]
    assert len(systems) == 1
    assert systems[0]["content"].startswith(DEFAULT_SYSTEM_INSTRUCTION)
    assert "Available Prime Capabilities:" in systems[0]["content"]
    assert followup[0]["role"] == "system"


def test_respond_plain_chat_no_request():
    provider = MockProvider({})
    prime = Prime(provider)
    brain = MockScriptedBrain([BrainResult.success("Hello there!", "m")])
    aurora = Aurora(prime, brain=brain, config=_test_config(), context_manager=_test_context())
    result = aurora.respond([{"role": "user", "content": "Hi"}])
    assert len(provider.calls) == 0
    assert len(brain.seen_messages) == 1
    assert result.content == "Hello there!"


def test_respond_valid_request_reaches_prime():
    provider = MockProvider({})
    prime = Prime(provider)

    brain = MockScriptedBrain([
        BrainResult.success(VALID_BLOCK, "m"),
        BrainResult.success("Done.", "m"),
    ])

    aurora = Aurora(prime, brain=brain, config=_test_config(), context_manager=_test_context())
    result = aurora.respond([
        {"role": "user", "content": "What is 2+3?"}
    ])

    assert result.success
    assert result.content == "Done."
    assert len(provider.calls) == 0



if __name__ == "__main__":
    test_instruction_default_content()
    test_instruction_env_override()
    test_instruction_default_from_env()
    test_respond_prepends_instruction_once()
    test_respond_does_not_duplicate_caller_system()
    test_respond_followup_keeps_single_instruction()
    test_respond_plain_chat_no_request()
    test_respond_valid_request_reaches_prime()
    print("All instruction tests passed!")
