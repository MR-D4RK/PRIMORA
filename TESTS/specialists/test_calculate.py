import sys
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime import Prime
from PRIMORA.prime.providers import Provider
from PRIMORA.prime.contract import PrimeRequest, PrimeResult, RequestStatus
from PRIMORA.prime.capability import CapabilityRegistry
from PRIMORA.prime.execution import Execution
from PRIMORA.specialists import Specialist, CalculateSpecialist
from typing import Any


class MockProvider(Provider):
    def __init__(self):
        self.calls = []

    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        self.calls.append((capability, payload))
        return PrimeResult.success({"via": "provider"})


def test_specialist_interface():
    assert issubclass(CalculateSpecialist, Specialist)
    assert CalculateSpecialist.name == "calculate"
    assert CapabilityRegistry().has("calculate")


def test_calculate_basic_arithmetic():
    spec = CalculateSpecialist()
    assert spec.handle({"expression": "2 + 3"}).output == {"result": 5}
    assert spec.handle({"expression": "10 - 4"}).output == {"result": 6}
    assert spec.handle({"expression": "3 * 7"}).output == {"result": 21}
    assert spec.handle({"expression": "20 / 4"}).output == {"result": 5.0}
    assert spec.handle({"expression": "(2 + 3) * 4"}).output == {"result": 20}
    assert spec.handle({"expression": "-5 + 2"}).output == {"result": -3}
    assert spec.handle({"expression": "2 ** 10"}).output == {"result": 1024}
    assert spec.handle({"expression": "17 % 5"}).output == {"result": 2}
    assert spec.handle({"expression": "17 // 5"}).output == {"result": 3}


def test_calculate_large_expression():
    result = CalculateSpecialist().handle({"expression": "938472 * 82739"})
    assert result.status == RequestStatus.SUCCESS
    assert result.output == {"result": 77648234808}


def test_calculate_missing_expression():
    result = CalculateSpecialist().handle({})
    assert result.status == RequestStatus.FAILURE
    assert "expression" in result.error


def test_calculate_invalid_expression_type():
    for bad in (123, None, ["2+2"], ""):
        result = CalculateSpecialist().handle({"expression": bad})
        assert result.status == RequestStatus.FAILURE


def test_calculate_division_by_zero():
    result = CalculateSpecialist().handle({"expression": "1 / 0"})
    assert result.status == RequestStatus.FAILURE
    assert "Division by zero" in result.error


def test_calculate_rejects_code():
    for evil in (
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "x + 1",
        "[1, 2, 3]",
        "abs(-5)",
        "(lambda: 1)()",
        "True + 1",
    ):
        result = CalculateSpecialist().handle({"expression": evil})
        assert result.status == RequestStatus.FAILURE, evil


def test_calculate_rejects_syntax_errors():
    result = CalculateSpecialist().handle({"expression": "2 +* 3"})
    assert result.status == RequestStatus.FAILURE


def test_execution_dispatches_to_specialist():
    provider = MockProvider()
    execution = Execution(provider, CapabilityRegistry())
    result = execution.execute("calculate", {"expression": "6 * 7"})
    assert result.status == RequestStatus.SUCCESS
    assert result.output == {"result": 42}
    assert provider.calls == []


def test_execution_falls_back_to_provider():
    provider = MockProvider()
    execution = Execution(provider, CapabilityRegistry())
    result = execution.execute("chat", {"messages": []})
    assert result.status == RequestStatus.SUCCESS
    assert result.output == {"via": "provider"}
    assert provider.calls == [("chat", {"messages": []})]


def test_end_to_end_calculate():
    provider = MockProvider()
    prime = Prime(provider)
    result = prime.handle(PrimeRequest(action="calculate", payload={"expression": "938472 * 82739"}))
    assert result.status == RequestStatus.SUCCESS
    assert result.output == {"result": 77648234808}
    assert provider.calls == []


def test_end_to_end_calculate_error():
    provider = MockProvider()
    prime = Prime(provider)
    result = prime.handle(PrimeRequest(action="calculate", payload={"expression": "1 / 0"}))
    assert result.status == RequestStatus.FAILURE
    assert "Division by zero" in result.error


if __name__ == "__main__":
    test_specialist_interface()
    test_calculate_basic_arithmetic()
    test_calculate_large_expression()
    test_calculate_missing_expression()
    test_calculate_invalid_expression_type()
    test_calculate_division_by_zero()
    test_calculate_rejects_code()
    test_calculate_rejects_syntax_errors()
    test_execution_dispatches_to_specialist()
    test_execution_falls_back_to_provider()
    test_end_to_end_calculate()
    test_end_to_end_calculate_error()
    print("All calculate tests passed!")
