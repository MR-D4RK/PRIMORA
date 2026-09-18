import sys
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from typing import Any
from PRIMORA.prime.execution import Execution
from PRIMORA.prime.providers import Provider
from PRIMORA.prime.contract import PrimeResult, RequestStatus
from PRIMORA.prime.capability import CapabilityRegistry, CapabilitySpec


class MockProvider(Provider):
    def __init__(self, results: dict):
        self._results = results

    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        if capability in self._results:
            return self._results[capability]
        return PrimeResult.failure(f"Unknown capability: {capability}")


def test_execution_requires_provider():
    try:
        Execution(None, CapabilityRegistry())
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "requires a Provider" in str(e)


def test_execution_uses_registered_specialist():
    mock_result = PrimeResult.success({"custom": "result"})
    provider = MockProvider({})

    class MockChatSpecialist:
        name = "chat"

        def handle(self, payload):
            return mock_result

    registry = CapabilityRegistry()
    registry.register(
        CapabilitySpec(
            name="chat",
            description="Test chat capability",
            payload_schema={"type": "object"},
            required_fields=[],
            optional_fields=[]
        ),
        MockChatSpecialist()
    )

    execution = Execution(provider, registry)
    result = execution.execute("chat", {})

    assert result.status == RequestStatus.SUCCESS
    assert result.output == {"custom": "result"}


def test_execution_uses_registered_provider_capability():
    mock_result = PrimeResult.success({"custom": "result"})
    provider = MockProvider({"provider_chat": mock_result})
    registry = CapabilityRegistry()

    registry.register(
        CapabilitySpec(
            name="provider_chat",
            description="Test provider-backed capability",
            payload_schema={"type": "object"},
            required_fields=[],
            optional_fields=[]
        ),
        None
    )

    execution = Execution(provider, registry)
    result = execution.execute("provider_chat", {"messages": []})

    assert result.status == RequestStatus.SUCCESS
    assert result.output == {"custom": "result"}


def test_execution_returns_provider_failure():
    provider = MockProvider({"provider_chat": PrimeResult.failure("Provider error")})
    registry = CapabilityRegistry()

    registry.register(
        CapabilitySpec(
            name="provider_chat",
            description="Test provider-backed capability",
            payload_schema={"type": "object"},
            required_fields=[],
            optional_fields=[]
        ),
        None
    )

    execution = Execution(provider, registry)
    result = execution.execute("provider_chat", {})

    assert result.status == RequestStatus.FAILURE
    assert "Provider error" in result.error


def test_execution_unknown_capability():
    provider = MockProvider({})
    registry = CapabilityRegistry()
    execution = Execution(provider, registry)

    result = execution.execute("unknown", {})

    assert result.status == RequestStatus.FAILURE
    assert "Unknown capability" in result.error


if __name__ == "__main__":
    test_execution_requires_provider()
    test_execution_uses_registered_specialist()
    test_execution_uses_registered_provider_capability()
    test_execution_returns_provider_failure()
    test_execution_unknown_capability()
    print("All Execution tests passed!")
