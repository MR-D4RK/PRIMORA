import sys
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.contract import PrimeRequest, PrimeResult, RequestStatus


def test_prime_request_creation():
    req = PrimeRequest(action="test_action", payload={"key": "value"})
    assert req.action == "test_action"
    assert req.payload == {"key": "value"}
    assert req.context is None
    assert req.metadata is None


def test_prime_request_with_optional_fields():
    req = PrimeRequest(
        action="test",
        payload={"data": 123},
        context={"session": "abc"},
        metadata={"trace_id": "xyz"}
    )
    assert req.context == {"session": "abc"}
    assert req.metadata == {"trace_id": "xyz"}


def test_prime_result_success():
    result = PrimeResult.success(output={"result": "ok"}, metadata={"time": 0.1})
    assert result.status == RequestStatus.SUCCESS
    assert result.output == {"result": "ok"}
    assert result.error is None
    assert result.metadata == {"time": 0.1}


def test_prime_result_failure():
    result = PrimeResult.failure(error="Something went wrong", metadata={"code": 500})
    assert result.status == RequestStatus.FAILURE
    assert result.output is None
    assert result.error == "Something went wrong"
    assert result.metadata == {"code": 500}


def test_prime_result_immutable():
    req = PrimeRequest(action="test")
    try:
        req.action = "modified"
        assert False, "Should be frozen"
    except Exception:
        pass


if __name__ == "__main__":
    test_prime_request_creation()
    test_prime_request_with_optional_fields()
    test_prime_result_success()
    test_prime_result_failure()
    test_prime_result_immutable()
    print("All PRIME contract tests passed!")