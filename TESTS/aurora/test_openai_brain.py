import sys
import json
import tempfile
import threading
import time
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
from unittest.mock import patch

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import (
    AuroraBrain,
    OpenAIBrain,
    aurora_backend_from_env,
)
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.prime import Prime, ContextManager
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.providers import Provider


class MockProvider(Provider):
    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        return PrimeResult.failure(f"Unknown capability: {capability}")


def _isolated_prime() -> Prime:
    """Prime wired to a temp ContextManager, never the production DB.

    Bare ``Prime(provider)`` hard-codes ``ContextManager()``, whose
    default db_path is the production database; every construction here
    is redirected to a throwaway file.
    """
    db = os.path.join(tempfile.mkdtemp(), "prime_test.db")
    with patch(
        "PRIMORA.prime.prime.ContextManager",
        lambda: ContextManager(db_path=db),
    ):
        return Prime(MockProvider())


class _RecordedRequest:
    path = None
    body = None


class MockOpenAIHandler(BaseHTTPRequestHandler):
    response_data = {
        "choices": [{"message": {"role": "assistant", "content": "Hello there"}}],
        "model": "test-gguf",
    }
    status_code = 200

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        _RecordedRequest.path = self.path
        try:
            _RecordedRequest.body = json.loads(raw.decode("utf-8")) if raw else None
        except json.JSONDecodeError:
            _RecordedRequest.body = raw
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.response_data).encode("utf-8"))

    def log_message(self, format, *args):
        pass


def _start_server(handler_cls=MockOpenAIHandler):
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    return server, port


def _config(**overrides) -> AuroraConfig:
    defaults = dict(
        model_path="/fake/model",
        adapter_path="/fake/adapter",
        max_new_tokens=128,
        temperature=0.5,
        top_p=0.8,
        repetition_penalty=1.1,
    )
    defaults.update(overrides)
    return AuroraConfig(**defaults)


def _brain(port=None, config=None, **kwargs) -> OpenAIBrain:
    base_url = f"http://127.0.0.1:{port}/v1" if port else "http://localhost:8080/v1"
    return OpenAIBrain(
        config=config or _config(), base_url=base_url, **kwargs)


# ======================================================================
# Backend selection
# ======================================================================

def test_backend_selection_default_is_transformers():
    os.environ.pop("PRIMORA_AURORA_BACKEND", None)
    assert aurora_backend_from_env() == "transformers"
    assert aurora_backend_from_env(env={}) == "transformers"
    assert aurora_backend_from_env(env={"PRIMORA_AURORA_BACKEND": ""}) == "transformers"


def test_backend_selection_openai_and_alias():
    assert aurora_backend_from_env(env={"PRIMORA_AURORA_BACKEND": "openai"}) == "openai"
    assert aurora_backend_from_env(env={"PRIMORA_AURORA_BACKEND": "llamacpp"}) == "openai"
    assert aurora_backend_from_env(env={"PRIMORA_AURORA_BACKEND": "OPENAI"}) == "openai"


def test_backend_selection_unknown_fails_closed():
    try:
        aurora_backend_from_env(env={"PRIMORA_AURORA_BACKEND": "mystery"})
    except ValueError as e:
        assert "PRIMORA_AURORA_BACKEND" in str(e)
    else:
        raise AssertionError("expected ValueError for unsupported backend")


# ======================================================================
# Wire format against a mocked server
# ======================================================================

def test_generate_success_maps_generation_settings():
    server, port = _start_server()
    try:
        brain = _brain(port, model="test-gguf")
        messages = [
            {"role": "system", "content": "You are Aurora."},
            {"role": "user", "content": "Hi"},
        ]
        result = brain.generate(messages, max_output_tokens=64)
        assert result.success
        assert result.content == "Hello there"
        assert result.model == "test-gguf"
        assert brain.get_model_name() == "test-gguf"

        assert _RecordedRequest.path == "/v1/chat/completions"
        body = _RecordedRequest.body
        assert body["model"] == "test-gguf"
        # Same messages the transformers path receives, verbatim.
        assert body["messages"] == messages
        assert body["stream"] is False
        assert body["max_tokens"] == 64
        assert body["temperature"] == 0.5
        assert body["top_p"] == 0.8
        assert body["repeat_penalty"] == 1.1
        # Thinking is off by default: a Qwen3-class template that thinks
        # can spend the whole max_tokens budget on reasoning and leave
        # content empty, which the live session hit as "Model returned
        # empty response".
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
    finally:
        server.shutdown()


def test_thinking_can_be_enabled_by_env():
    os.environ["PRIMORA_OPENAI_ENABLE_THINKING"] = "1"
    server, port = _start_server()
    try:
        brain = _brain(port)
        brain.generate([{"role": "user", "content": "Hi"}])
        assert _RecordedRequest.body["chat_template_kwargs"] == {
            "enable_thinking": True}
    finally:
        server.shutdown()
        os.environ.pop("PRIMORA_OPENAI_ENABLE_THINKING", None)


def test_generate_max_tokens_capped_by_config():
    server, port = _start_server()
    try:
        brain = _brain(port)  # config.max_new_tokens = 128
        brain.generate([{"role": "user", "content": "Hi"}], max_output_tokens=999)
        assert _RecordedRequest.body["max_tokens"] == 128
    finally:
        server.shutdown()


def test_generate_max_output_tokens_zero_fails():
    brain = _brain()
    result = brain.generate(
        [{"role": "user", "content": "Hi"}], max_output_tokens=0)
    assert not result.success
    assert "No generation capacity" in result.error


def test_generate_missing_messages():
    brain = _brain()
    result = brain.generate("not a list")
    assert not result.success
    assert "Missing or invalid 'messages'" in result.error


def test_generate_http_error():
    class ErrorHandler(MockOpenAIHandler):
        status_code = 500
        response_data = {"error": "boom"}

    server, port = _start_server(ErrorHandler)
    try:
        result = _brain(port).generate([{"role": "user", "content": "Hi"}])
        assert not result.success
        assert "HTTP error 500" in result.error
    finally:
        server.shutdown()


def test_generate_connection_error():
    result = _brain(port=9999).generate([{"role": "user", "content": "Hi"}])
    assert not result.success
    assert "connection" in result.error.lower()


def test_generate_empty_content():
    class EmptyHandler(MockOpenAIHandler):
        response_data = {
            "choices": [{"message": {"role": "assistant", "content": ""}}],
        }

    server, port = _start_server(EmptyHandler)
    try:
        result = _brain(port).generate([{"role": "user", "content": "Hi"}])
        assert not result.success
        assert "empty" in result.error.lower()
    finally:
        server.shutdown()


def test_empty_content_reports_reasoning_and_finish_reason():
    """The live-session failure shape: content empty, reasoning present.

    A thinking budget that consumed max_tokens leaves content empty while
    reasoning_content carries the reasoning. The failure must name both so
    the defect is diagnosable from the error alone.
    """
    class ReasoningOnlyHandler(MockOpenAIHandler):
        response_data = {
            "choices": [{
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "x" * 296,
                },
            }],
        }

    server, port = _start_server(ReasoningOnlyHandler)
    try:
        result = _brain(port).generate([{"role": "user", "content": "Hi"}])
        assert not result.success
        assert "empty" in result.error.lower()
        assert "finish_reason='length'" in result.error
        assert "reasoning_chars=296" in result.error
        assert "enable_thinking=False" in result.error
    finally:
        server.shutdown()


def test_generate_missing_choices():
    class NoChoicesHandler(MockOpenAIHandler):
        response_data = {"model": "test-gguf"}

    server, port = _start_server(NoChoicesHandler)
    try:
        result = _brain(port).generate([{"role": "user", "content": "Hi"}])
        assert not result.success
        assert "empty" in result.error.lower()
    finally:
        server.shutdown()


# ======================================================================
# Thinking-tag stripping
# ======================================================================

def test_thinking_fence_stripped():
    class ThinkHandler(MockOpenAIHandler):
        response_data = {
            "choices": [{"message": {
                "role": "assistant",
                "content": "```think\nhidden reasoning\n```\nThe answer is 4.",
            }}],
        }

    server, port = _start_server(ThinkHandler)
    try:
        result = _brain(port).generate([{"role": "user", "content": "2+2?"}])
        assert result.success
        assert result.content == "The answer is 4."
        assert "```" not in result.content
        assert "hidden reasoning" not in result.content
    finally:
        server.shutdown()


def test_stray_thinking_tags_stripped():
    class StrayHandler(MockOpenAIHandler):
        response_data = {
            "choices": [{"message": {
                "role": "assistant",
                "content": "Visible answer```",
            }}],
        }

    server, port = _start_server(StrayHandler)
    try:
        result = _brain(port).generate([{"role": "user", "content": "Hi"}])
        assert result.success
        assert result.content == "Visible answer"
    finally:
        server.shutdown()


def test_only_thinking_output_is_empty_failure():
    class OnlyThinkHandler(MockOpenAIHandler):
        response_data = {
            "choices": [{"message": {
                "role": "assistant",
                "content": "```think\nall reasoning, no answer\n```",
            }}],
        }

    server, port = _start_server(OnlyThinkHandler)
    try:
        result = _brain(port).generate([{"role": "user", "content": "Hi"}])
        assert not result.success
        assert "empty" in result.error.lower()
    finally:
        server.shutdown()


# ======================================================================
# Config surface (same env vars as the Prime OpenAI provider)
# ======================================================================

def test_openai_endpoint_env_defaults():
    for key in ("PRIMORA_OPENAI_URL", "PRIMORA_OPENAI_MODEL",
                "PRIMORA_OPENAI_TIMEOUT"):
        os.environ.pop(key, None)
    brain = OpenAIBrain(config=_config())
    assert brain._base_url == "http://localhost:8080/v1"
    assert brain._model == "llama-server"
    assert brain._timeout == 120


def test_openai_endpoint_env_and_constructor_overrides():
    os.environ["PRIMORA_OPENAI_URL"] = "http://127.0.0.1:9999/v1"
    os.environ["PRIMORA_OPENAI_MODEL"] = "env-model"
    os.environ["PRIMORA_OPENAI_TIMEOUT"] = "30"
    try:
        brain = OpenAIBrain(config=_config())
        assert brain._base_url == "http://127.0.0.1:9999/v1"
        assert brain._model == "env-model"
        assert brain._timeout == 30

        brain = OpenAIBrain(
            config=_config(),
            base_url="http://override/v1",
            model="ctor-model",
            timeout=7,
        )
        assert brain._base_url == "http://override/v1"
        assert brain._model == "ctor-model"
        assert brain._timeout == 7
    finally:
        for key in ("PRIMORA_OPENAI_URL", "PRIMORA_OPENAI_MODEL",
                    "PRIMORA_OPENAI_TIMEOUT"):
            os.environ.pop(key, None)


def test_generation_settings_from_config_env():
    os.environ["PRIMORA_AURORA_MAX_TOKENS"] = "64"
    os.environ["PRIMORA_AURORA_TEMPERATURE"] = "0.25"
    os.environ["PRIMORA_AURORA_TOP_P"] = "0.7"
    os.environ["PRIMORA_AURORA_REPETITION_PENALTY"] = "1.2"
    server, port = _start_server()
    try:
        brain = _brain(port, config=AuroraConfig.from_env())
        brain.generate([{"role": "user", "content": "Hi"}])
        body = _RecordedRequest.body
        assert body["max_tokens"] == 64
        assert body["temperature"] == 0.25
        assert body["top_p"] == 0.7
        assert body["repeat_penalty"] == 1.2
    finally:
        server.shutdown()
        for key in ("PRIMORA_AURORA_MAX_TOKENS", "PRIMORA_AURORA_TEMPERATURE",
                    "PRIMORA_AURORA_TOP_P", "PRIMORA_AURORA_REPETITION_PENALTY"):
            os.environ.pop(key, None)


# ======================================================================
# No-HF guarantees and token counting
# ======================================================================

def test_count_tokens_needs_no_warmup_or_tokenizer():
    brain = _brain()
    messages = [
        {"role": "system", "content": "0123456789"},
        {"role": "user", "content": "abcdefgh"},
    ]
    # Same formula as ContextManager._estimate_messages: len//4 (+floor 1)
    # per message, plus 4 per message.
    assert brain.count_tokens(messages) == (10 // 4 + 4) + (8 // 4 + 4)


def test_warmup_is_a_noop_and_loads_nothing():
    brain = _brain()
    assert brain.warmup() is None


def test_context_budget_env_raise_is_honoured():
    os.environ["PRIMORA_AURORA_CONTEXT_TARGET"] = "3000"
    os.environ["PRIMORA_AURORA_CONTEXT_CEILING"] = "3200"
    os.environ["PRIMORA_AURORA_CONTEXT_OUTPUT_RESERVE"] = "400"
    os.environ["PRIMORA_AURORA_CONTEXT_TOTAL"] = "3600"
    try:
        config = AuroraConfig.from_env()
        assert config.context_total == 3600
        assert config.validate(require_paths=False) is None
    finally:
        for key in ("PRIMORA_AURORA_CONTEXT_TARGET", "PRIMORA_AURORA_CONTEXT_CEILING",
                    "PRIMORA_AURORA_CONTEXT_OUTPUT_RESERVE",
                    "PRIMORA_AURORA_CONTEXT_TOTAL"):
            os.environ.pop(key, None)


def test_validate_budget_invariants_still_enforced_without_paths():
    config = _config(context_total=9999)  # inconsistent with ceiling+reserve
    err = config.validate(require_paths=False)
    assert err is not None
    assert "context_total" in err


# ======================================================================
# Aurora wiring
# ======================================================================

def test_aurora_selects_openai_brain_without_hf_paths():
    os.environ["PRIMORA_AURORA_BACKEND"] = "openai"
    os.environ["PRIMORA_OPENAI_MODEL"] = "wired-model"
    try:
        prime = _isolated_prime()
        aurora = Aurora(
            prime,
            config=_config(
                model_path="C:/nonexistent/hf/model",
                adapter_path="C:/nonexistent/hf/adapter",
            ),
        )
        assert isinstance(aurora._brain, OpenAIBrain)
        assert aurora.get_model_name() == "wired-model"
        # No HF load: brain warmup is a no-op with no model directories
        # present (the transformers brain would fail initialization here).
        assert aurora._brain.warmup() is None
    finally:
        os.environ.pop("PRIMORA_AURORA_BACKEND", None)
        os.environ.pop("PRIMORA_OPENAI_MODEL", None)


def test_aurora_defaults_to_transformers_brain():
    os.environ.pop("PRIMORA_AURORA_BACKEND", None)
    model_dir = tempfile.mkdtemp()
    adapter_dir = tempfile.mkdtemp()
    prime = _isolated_prime()
    aurora = Aurora(
        prime,
        config=_config(model_path=model_dir, adapter_path=adapter_dir),
    )
    assert isinstance(aurora._brain, AuroraBrain)


def test_aurora_unknown_backend_fails_closed():
    os.environ["PRIMORA_AURORA_BACKEND"] = "mismatch"
    try:
        model_dir = tempfile.mkdtemp()
        adapter_dir = tempfile.mkdtemp()
        prime = _isolated_prime()
        try:
            Aurora(
                prime,
                config=_config(model_path=model_dir, adapter_path=adapter_dir),
            )
        except ValueError as e:
            assert "PRIMORA_AURORA_BACKEND" in str(e)
        else:
            raise AssertionError("expected ValueError for unsupported backend")
    finally:
        os.environ.pop("PRIMORA_AURORA_BACKEND", None)


def test_explicit_brain_still_bypasses_backend_selection():
    os.environ["PRIMORA_AURORA_BACKEND"] = "openai"
    try:
        class _Fixed(OpenAIBrain):
            pass

        brain = _Fixed(config=_config(), base_url="http://127.0.0.1:1/v1")
        prime = _isolated_prime()
        aurora = Aurora(prime, brain=brain)
        assert aurora._brain is brain
    finally:
        os.environ.pop("PRIMORA_AURORA_BACKEND", None)


if __name__ == "__main__":
    test_backend_selection_default_is_transformers()
    test_backend_selection_openai_and_alias()
    test_backend_selection_unknown_fails_closed()
    test_generate_success_maps_generation_settings()
    test_generate_max_tokens_capped_by_config()
    test_generate_max_output_tokens_zero_fails()
    test_generate_missing_messages()
    test_generate_http_error()
    test_generate_connection_error()
    test_generate_empty_content()
    test_empty_content_reports_reasoning_and_finish_reason()
    test_generate_missing_choices()
    test_thinking_can_be_enabled_by_env()
    test_thinking_fence_stripped()
    test_stray_thinking_tags_stripped()
    test_only_thinking_output_is_empty_failure()
    test_openai_endpoint_env_defaults()
    test_openai_endpoint_env_and_constructor_overrides()
    test_generation_settings_from_config_env()
    test_count_tokens_needs_no_warmup_or_tokenizer()
    test_warmup_is_a_noop_and_loads_nothing()
    test_context_budget_env_raise_is_honoured()
    test_validate_budget_invariants_still_enforced_without_paths()
    test_aurora_selects_openai_brain_without_hf_paths()
    test_aurora_defaults_to_transformers_brain()
    test_aurora_unknown_backend_fails_closed()
    test_explicit_brain_still_bypasses_backend_selection()
    print("All OpenAIBrain tests passed!")
