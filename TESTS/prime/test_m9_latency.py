"""M9 internal-memory latency controls and fail-closed regressions."""

import json
from types import SimpleNamespace

from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.loop import CognitiveLoop
from PRIMORA.learning import LearningRequest, ModelJev
from PRIMORA.prime.contract import PrimeRequest, PrimeResult, RequestStatus
from PRIMORA.prime.context.formation import provider_proposals
from PRIMORA.prime.providers import OpenAICompatibleProvider, Provider


class _JevProvider(Provider):
    def __init__(self, output=None, failure=None):
        self.output = output
        self.failure = failure
        self.calls = []

    def invoke(self, capability, payload):
        self.calls.append((capability, payload))
        if self.failure:
            return PrimeResult.failure(self.failure)
        return PrimeResult.success({"content": self.output})


def _jev_json():
    return json.dumps({"candidates": [{
        "concept": "favorite color", "value": "amber",
        "memory_type": "PREFERENCE", "scope": "global",
    }]})


def test_jev_uses_bounded_strict_json_controls(monkeypatch):
    monkeypatch.setenv("PRIMORA_JEV_MAX_TOKENS", "77")
    monkeypatch.setenv("PRIMORA_JEV_TEMPERATURE", "0.1")
    monkeypatch.setenv("PRIMORA_JEV_ENABLE_THINKING", "0")
    provider = _JevProvider(_jev_json())
    outcome = ModelJev(provider).evaluate(
        LearningRequest("I like amber", "user", "s"))

    assert outcome.disposition == "confirm"
    payload = provider.calls[0][1]
    assert payload["options"] == {
        "temperature": 0.1,
        "max_tokens": 77,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert payload["response_format"] == {"type": "json_object"}
    assert outcome.timings_ms.keys() >= {"provider_ms", "parse_ms", "total_ms"}
    assert any(item.startswith("jev_timing:") for item in outcome.diagnostics)


def test_jev_rejects_truncated_json_and_provider_failures_with_timing():
    truncated = _JevProvider('{"candidates":[{"concept":"x"')
    outcome = ModelJev(truncated).evaluate(LearningRequest("x", "user", "s"))
    assert outcome.disposition == "reject"
    assert outcome.candidates == []
    assert "total_ms" in outcome.timings_ms

    failed = _JevProvider(failure="timeout")
    outcome = ModelJev(failed).evaluate(LearningRequest("x", "user", "s"))
    assert outcome.disposition == "reject"
    assert outcome.reason == "jev_failed: timeout"
    assert "provider_ms" in outcome.timings_ms


def test_jev_accepts_provider_reasoning_field_only_after_json_validation():
    class ReasoningProvider(Provider):
        def invoke(self, capability, payload):
            return PrimeResult.success({"content": "", "reasoning_content": _jev_json()})

    outcome = ModelJev(ReasoningProvider()).evaluate(
        LearningRequest("I like amber", "user", "s"))
    assert outcome.disposition == "confirm"
    assert outcome.candidates[0].value == "amber"


def test_learn_result_reports_jev_and_admission_latency(tmp_path):
    from PRIMORA.prime.context.manager import ContextManager

    provider = _JevProvider(_jev_json())
    summary = ContextManager(db_path=tmp_path / "learn.db").learn(
        "I like amber", "user", ModelJev(provider), session_id="s")
    assert summary["status"] == "learned"
    assert summary["timings_ms"].keys() >= {
        "jev_ms", "jev_provider_ms", "jev_parse_ms", "admission_ms",
        "total_ms",
    }


def test_typed_formation_generation_controls_are_configurable(monkeypatch):
    monkeypatch.setenv("PRIMORA_MEMORY_FORMATION_MAX_TOKENS", "91")
    monkeypatch.setenv("PRIMORA_MEMORY_FORMATION_TEMPERATURE", "0.2")
    monkeypatch.setenv("PRIMORA_MEMORY_FORMATION_ENABLE_THINKING", "0")
    provider = _JevProvider('{"candidates":[]}')

    class Message:
        message_id = 7
        content = "I like amber"

    candidates, diagnostic = provider_proposals(provider, Message(), "s")
    assert candidates == []
    assert diagnostic == "ok"
    payload = provider.calls[0][1]
    assert payload["options"]["max_tokens"] == 91
    assert payload["options"]["temperature"] == 0.2
    assert payload["options"]["chat_template_kwargs"]["enable_thinking"] is False
    assert payload["response_format"] == {"type": "json_object"}


class _Response:
    status = 200

    def __init__(self, body):
        self.body = body

    def read(self):
        return json.dumps(self.body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_openai_reasoning_channel_is_scoped_to_internal_json_capabilities(monkeypatch):
    def fake_urlopen(request, timeout):
        return _Response({"choices": [{"message": {
            "content": "",
            "reasoning_content": _jev_json(),
        }}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider("qwen")
    jev = provider.invoke("jev", {"messages": []})
    assert jev.status is RequestStatus.SUCCESS
    assert jev.output["reasoning_fallback"] is True

    chat = provider.invoke("chat", {"messages": []})
    assert chat.status is RequestStatus.FAILURE
    assert "empty response" in chat.error


class _Mechanisms:
    def __init__(self):
        self.samples = [{"total_ms": 4.0, "provider_ms": 3.0,
                         "admission_ms": 1.0}]

    def consume_formation_timings(self):
        return self.samples.pop(0) if self.samples else {}


class _Mind:
    def __init__(self):
        self.mechanisms = _Mechanisms()

    def build(self, messages, session_id, **kwargs):
        return SimpleNamespace(messages=messages, allowed_output=32)

    def reconcile(self, messages, session_id):
        return []


class _Brain(Brain):
    def generate(self, messages, max_output_tokens=None):
        return BrainResult.success("done", "test")

    def get_model_name(self):
        return "test"

    def count_tokens(self, messages):
        return 1


def test_cognitive_trace_reports_planning_formation_and_response_latency():
    loop = CognitiveLoop(_Mind(), _Brain(), max_cycles=1)
    result = loop.run([{"role": "user", "content": "hello"}], "s")
    assert result.success
    timings = loop.last_trace.to_dict()["timings_ms"]
    assert timings.keys() >= {
        "planning_ms", "formation_ms", "formation_provider_ms",
        "response_generation_ms",
    }
    assert timings["formation_ms"] == 4.0


def test_capability_executor_measurement_is_separate_from_model_generation():
    from PRIMORA.aurora.loop import CapabilityExecutor

    class Prime:
        def handle(self, request):
            return PrimeResult.success({"ok": True})

    executor = CapabilityExecutor(Prime())
    executor.begin_session("s")
    executor.execute(PrimeRequest("learn", {}))
    assert executor.timings_ms["capability_learn_ms"] >= 0
