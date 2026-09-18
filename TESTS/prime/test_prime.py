import sys
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime import Prime
from PRIMORA.prime.providers import Provider
from PRIMORA.prime.contract import PrimeRequest, PrimeResult, RequestStatus
from PRIMORA.prime.capability import CapabilityRegistry
from typing import Any


class MockProvider(Provider):
    def __init__(self, results: dict):
        self._results = results

    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        if capability in self._results:
            return self._results[capability]
        return PrimeResult.failure(f"Unknown capability: {capability}")


def test_prime_requires_provider():
    try:
        Prime(None)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "requires a Provider" in str(e)


def test_prime_handle_routes_and_executes():
    mock_result = PrimeResult.success({"result": "ok"})
    provider = MockProvider({})
    prime = Prime(provider)
    req = PrimeRequest(action="calculate", payload={"expression": "2 + 3"})
    result = prime.handle(req)
    assert result.status == RequestStatus.SUCCESS
    assert result.output == {"result": 5}


def test_prime_handle_invalid_request():
    provider = MockProvider({})
    prime = Prime(provider)
    result = prime.handle("not a request")
    assert result.status == RequestStatus.FAILURE
    assert "Invalid request type" in result.error


def test_prime_handle_unknown_capability():
    provider = MockProvider({})
    prime = Prime(provider)
    req = PrimeRequest(action="unknown", payload={})
    result = prime.handle(req)
    assert result.status == RequestStatus.FAILURE
    assert "Unknown capability" in result.error


if __name__ == "__main__":
    test_prime_requires_provider()
    test_prime_handle_routes_and_executes()
    test_prime_handle_invalid_request()
    test_prime_handle_unknown_capability()
    print("All PRIME tests passed!")