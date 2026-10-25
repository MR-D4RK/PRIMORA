"""llama.cpp/Qwen response-shape regressions for Memory V2 formation."""

import json

from PRIMORA.prime.context.formation import parse_candidates
from PRIMORA.prime.providers import OpenAICompatibleProvider


class _Response:
    status = 200

    def __init__(self, body):
        self._body = body

    def read(self):
        return json.dumps(self._body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_reasoning_content_fallback_and_llamacpp_json_controls(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response({
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": "",
                    "reasoning_content": (
                        "<think>short internal step</think> "
                        '{"candidates":[]}'
                    ),
                },
            }],
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        "Qwen3.5-9B-Q4_K_M", base_url="http://127.0.0.1:8080/v1", timeout=17
    )
    result = provider.invoke("memory_formation", {
        "messages": [{"role": "user", "content": "I prefer asymmetry"}],
        "options": {
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "response_format": {"type": "json_object"},
    })
    assert result.status.value == "success"
    assert result.output["reasoning_fallback"] is True
    assert parse_candidates(result.output["content"]) == []
    assert captured["body"]["max_tokens"] == 256
    assert captured["body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["timeout"] == 17


def test_empty_nonformation_response_remains_failure(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _Response({
            "choices": [{"message": {"content": "", "reasoning_content": ""}}]
        }),
    )
    provider = OpenAICompatibleProvider("Qwen3.5-9B-Q4_K_M")
    result = provider.invoke("provider_chat", {
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert result.status.value == "failure"
    assert "empty response" in (result.error or "")

