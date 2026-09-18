import sys
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import Brain, AuroraBrain, BrainResult
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.prime import Prime
from PRIMORA.prime.providers import Provider
from PRIMORA.prime.contract import PrimeResult
from typing import Any, Optional


class MockProvider(Provider):
    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        return PrimeResult.failure("unused")


class MockBrain(Brain):
    def __init__(self, model: str = "test-model"):
        self._model = model
        self.warmup_calls = 0

    def generate(self, messages: list[dict[str, str]]) -> BrainResult:
        return BrainResult.success("mock response", self._model)

    def get_model_name(self) -> str:
        return self._model

    def warmup(self) -> Optional[str]:
        self.warmup_calls += 1
        return None


class FailingBrain(MockBrain):
    def warmup(self) -> Optional[str]:
        return "weights unavailable"


class CountingBrain(AuroraBrain):
    def __init__(self):
        super().__init__(config=AuroraConfig(
            model_path="/fake/model",
            adapter_path="/fake/adapter",
        ))
        self.init_calls = 0

    def _initialize(self) -> Optional[str]:
        if self._initialized:
            return None
        self.init_calls += 1
        self._initialized = True
        return None


def test_default_brain_warmup_returns_none():
    class PlainBrain(Brain):
        def generate(self, messages):
            return BrainResult.success("ok", "m")

        def get_model_name(self):
            return "m"

    assert PlainBrain().warmup() is None


def test_aurora_warmup_delegates_to_brain():
    prime = Prime(MockProvider())
    brain = MockBrain()
    aurora = Aurora(prime, brain=brain)
    assert aurora.warmup() is None
    assert brain.warmup_calls == 1


def test_aurora_warmup_propagates_error():
    prime = Prime(MockProvider())
    brain = FailingBrain()
    aurora = Aurora(prime, brain=brain)
    assert aurora.warmup() == "weights unavailable"


def test_aurorabrain_warmup_initializes_once():
    brain = CountingBrain()
    assert brain.warmup() is None
    assert brain.warmup() is None
    assert brain.init_calls == 1


if __name__ == "__main__":
    test_default_brain_warmup_returns_none()
    test_aurora_warmup_delegates_to_brain()
    test_aurora_warmup_propagates_error()
    test_aurorabrain_warmup_initializes_once()
    print("All startup tests passed!")
