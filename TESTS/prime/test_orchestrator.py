import sys
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.orchestrator import Orchestrator
from PRIMORA.prime.providers import Provider
from PRIMORA.prime.contract import PrimeRequest, PrimeResult, RequestStatus
from PRIMORA.prime.capability import CapabilityRegistry, CapabilitySpec
from typing import Any


class MockProvider(Provider):
    def __init__(self, results: dict):
        self._results = results

    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        if capability in self._results:
            return self._results[capability]
        return PrimeResult.failure(f"Unknown capability: {capability}")


def test_orchestrator_requires_provider():
    try:
        Orchestrator(None)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "requires a Provider" in str(e)


def test_orchestrator_routes_and_executes():
    mock_result = PrimeResult.success({"routed": True})
    provider = MockProvider({})
    registry = CapabilityRegistry()

    class MockChatSpecialist:
        name = "chat"

        def handle(self, payload):
            return mock_result

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

    orch = Orchestrator(provider, registry)
    req = PrimeRequest(action="chat", payload={})
    result = orch.orchestrate(req)

    assert result.status == RequestStatus.SUCCESS
    assert result.output == {"routed": True}


def test_orchestrator_passes_payload_to_provider():
    captured = {}

    class CapturingProvider(Provider):
        def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
            captured["capability"] = capability
            captured["payload"] = payload
            return PrimeResult.success({"ok": True})

    provider = CapturingProvider()
    registry = CapabilityRegistry()

    registry.register(
        CapabilitySpec(
            name="provider_chat",
            description="Test provider-backed capability",
            payload_schema={
                "type": "object",
                "properties": {
                    "messages": {"type": "array"}
                },
                "required": ["messages"]
            },
            required_fields=["messages"],
            optional_fields=[]
        ),
        None
    )

    orch = Orchestrator(provider, registry)
    req = PrimeRequest(
        action="provider_chat",
        payload={"messages": []}
    )

    result = orch.orchestrate(req)

    assert captured["capability"] == "provider_chat"
    assert captured["payload"] == {"messages": []}
    assert result.status == RequestStatus.SUCCESS


def test_orchestrator_rejects_unknown_capability():
    provider = MockProvider({})
    registry = CapabilityRegistry()
    orch = Orchestrator(provider, registry)

    req = PrimeRequest(
        action="unknown_capability",
        payload={}
    )

    result = orch.orchestrate(req)

    assert result.status == RequestStatus.FAILURE
    assert "Unknown capability" in result.error


if __name__ == "__main__":
    test_orchestrator_requires_provider()
    test_orchestrator_routes_and_executes()
    test_orchestrator_passes_payload_to_provider()
    test_orchestrator_rejects_unknown_capability()
    print("All Orchestrator tests passed!")
