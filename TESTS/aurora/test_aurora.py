import sys
import os
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.prime import Prime
from PRIMORA.prime.providers import Provider
from PRIMORA.prime.contract import PrimeRequest, PrimeResult, RequestStatus
from PRIMORA.aurora.brain import Brain, BrainResult
from typing import Any


class MockProvider(Provider):
    def __init__(self, results: dict):
        self._results = results

    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        if capability in self._results:
            return self._results[capability]
        return PrimeResult.failure(f"Unknown capability: {capability}")


class MockBrain(Brain):
    def __init__(self, model: str = "test-model"):
        self._model = model

    def generate(self, messages: list[dict[str, str]], max_output_tokens=None) -> BrainResult:
        return BrainResult.success("mock response", self._model)

    def get_model_name(self) -> str:
        return self._model


def test_aurora_creation():
    provider = MockProvider({})
    prime = Prime(provider)
    brain = MockBrain()
    aurora = Aurora(prime, brain=brain)
    assert aurora is not None


def test_aurora_creation_with_env():
    os.environ["PRIMORA_AURORA_MODEL_PATH"] = "RESOURCES/aurora/v0.1/model"
    os.environ["PRIMORA_AURORA_ADAPTER_PATH"] = "RESOURCES/aurora/v0.1/aurora_qlora"
    try:
        provider = MockProvider({})
        prime = Prime(provider)
        aurora = Aurora(prime)
        assert aurora is not None
    except ValueError:
        pass  # Expected if paths don't exist
    finally:
        for key in ["PRIMORA_AURORA_MODEL_PATH", "PRIMORA_AURORA_ADAPTER_PATH"]:
            os.environ.pop(key, None)


def test_aurora_request_routes_through_prime():
    mock_result = PrimeResult.success({"routed": True})
    provider = MockProvider({"calculate": mock_result})
    prime = Prime(provider)
    brain = MockBrain()
    aurora = Aurora(prime, brain=brain)
    result = aurora.request("calculate", payload={"expression": "2 + 3"})
    assert result.status == RequestStatus.SUCCESS
    assert result.output == {"result": 5}


def test_aurora_request_with_context():
    provider = MockProvider({"calculate": PrimeResult.success({"ok": True})})
    prime = Prime(provider)
    brain = MockBrain()
    aurora = Aurora(prime, brain=brain)
    result = aurora.request("calculate", payload={"expression": "2+2"}, context={"session": "123"})
    assert result.status == RequestStatus.SUCCESS


def test_aurora_request_with_metadata():
    provider = MockProvider({"calculate": PrimeResult.success({"ok": True})})
    prime = Prime(provider)
    brain = MockBrain()
    aurora = Aurora(prime, brain=brain)
    result = aurora.request("calculate", payload={"expression": "2+2"}, metadata={"trace": "abc"})
    assert result.status == RequestStatus.SUCCESS


def test_aurora_think_uses_brain():
    provider = MockProvider({})
    prime = Prime(provider)
    brain = MockBrain("thinking-model")
    aurora = Aurora(prime, brain=brain)
    result = aurora.think([{"role": "user", "content": "Hi"}])
    assert result.success
    assert result.content == "mock response"
    assert result.model == "thinking-model"


def test_aurora_get_model_name():
    provider = MockProvider({})
    prime = Prime(provider)
    brain = MockBrain("my-model")
    aurora = Aurora(prime, brain=brain)
    assert aurora.get_model_name() == "my-model"


if __name__ == "__main__":
    test_aurora_creation()
    test_aurora_creation_with_env()
    test_aurora_request_routes_through_prime()
    test_aurora_request_with_context()
    test_aurora_request_with_metadata()
    test_aurora_think_uses_brain()
    test_aurora_get_model_name()
    print("All AURORA tests passed!")