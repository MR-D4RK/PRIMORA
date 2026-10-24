"""Model-agnostic OpenAI transport tests for strict chat templates."""
from PRIMORA.prime.providers import _normalize_openai_system_messages, OpenAICompatibleProvider
from PRIMORA.prime.contract import RequestStatus
import json


def test_multiple_system_messages_are_merged_before_history():
    original = [
        {"role": "system", "content": "Aurora identity"},
        {"role": "system", "content": "Memory source: user said X"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "system", "content": "Verified update: user said Y"},
        {"role": "user", "content": "What did I say?"},
    ]
    normalized = _normalize_openai_system_messages(original)
    assert [m["role"] for m in normalized] == ["system", "user", "assistant", "user"]
    assert normalized[0]["content"] == (
        "Aurora identity\n\nMemory source: user said X\n\nVerified update: user said Y")
    assert normalized[1:] == [original[2], original[3], original[5]]
    assert original[4]["role"] == "system"  # Original list remains untouched.


def test_no_system_or_single_leading_system_unchanged():
    for original in (
        [{"role": "user", "content": "hi"}],
        [{"role": "system", "content": "be helpful"}, {"role": "user", "content": "hi"}],
    ):
        assert _normalize_openai_system_messages(original) == original


def test_late_single_system_moves_to_front_without_dropping_history():
    original = [{"role": "user", "content": "Question"},
                {"role": "system", "content": "Source-backed context"},
                {"role": "assistant", "content": "Answer"}]
    assert _normalize_openai_system_messages(original) == [
        {"role": "system", "content": "Source-backed context"},
        original[0], original[2]]


def test_provider_sends_single_leading_system(monkeypatch):
    requests = []

    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "Ready"}}]}).encode()

    def fake_urlopen(req, timeout=None):
        requests.append(json.loads(req.data.decode()))
        return Response()

    monkeypatch.setattr("PRIMORA.prime.providers.urllib.request.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider("qwen3.5-9b", "http://127.0.0.1:8080/v1")
    original = [{"role": "system", "content": "base"},
                {"role": "system", "content": "memory context"},
                {"role": "user", "content": "question"}]
    result = provider.invoke("chat", {"messages": original})
    assert result.status == RequestStatus.SUCCESS
    assert requests[0]["messages"] == [
        {"role": "system", "content": "base\n\nmemory context"}, original[-1]]
    assert len(original) == 3
