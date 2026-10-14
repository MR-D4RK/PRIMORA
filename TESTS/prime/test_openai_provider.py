import sys
import json
import threading
import time
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.providers import OpenAICompatibleProvider
from PRIMORA.prime.contract import PrimeResult, RequestStatus


class _RecordedRequest:
    path = None
    body = None


class MockOpenAIHandler(BaseHTTPRequestHandler):
    response_data = {
        "choices": [{"message": {"role": "assistant", "content": "Test response"}}],
        "model": "test-model",
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


def test_openai_provider_success():
    server, port = _start_server()
    try:
        provider = OpenAICompatibleProvider(
            model="test-model", base_url=f"http://127.0.0.1:{port}/v1")
        result = provider.invoke("chat", {
            "messages": [{"role": "user", "content": "Hello"}]
        })
        assert result.status == RequestStatus.SUCCESS
        assert result.output["content"] == "Test response"
        assert result.output["model"] == "test-model"
        assert _RecordedRequest.path == "/v1/chat/completions"
        body = _RecordedRequest.body
        assert body["model"] == "test-model"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]
        assert body["stream"] is False
    finally:
        server.shutdown()


def test_openai_provider_payload_model_override():
    server, port = _start_server()
    try:
        provider = OpenAICompatibleProvider(
            model="default-model", base_url=f"http://127.0.0.1:{port}/v1")
        result = provider.invoke("chat", {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "specialist-model",
        })
        assert result.status == RequestStatus.SUCCESS
        assert result.output["model"] == "specialist-model"
        assert _RecordedRequest.body["model"] == "specialist-model"
    finally:
        server.shutdown()


def test_openai_provider_options_merged_top_level():
    server, port = _start_server()
    try:
        provider = OpenAICompatibleProvider(
            model="test-model", base_url=f"http://127.0.0.1:{port}/v1")
        result = provider.invoke("chat", {
            "messages": [{"role": "user", "content": "Hello"}],
            "options": {"temperature": 0.2, "max_tokens": 64},
        })
        assert result.status == RequestStatus.SUCCESS
        body = _RecordedRequest.body
        assert body["temperature"] == 0.2
        assert body["max_tokens"] == 64
        assert "options" not in body
        assert body["stream"] is False
    finally:
        server.shutdown()


def test_openai_provider_http_error():
    class ErrorHandler(MockOpenAIHandler):
        status_code = 500
        response_data = {"error": "Internal server error"}

    server, port = _start_server(ErrorHandler)
    try:
        provider = OpenAICompatibleProvider(
            model="test-model", base_url=f"http://127.0.0.1:{port}/v1")
        result = provider.invoke("chat", {"messages": [{"role": "user", "content": "Hi"}]})
        assert result.status == RequestStatus.FAILURE
        assert "HTTP error 500" in result.error
    finally:
        server.shutdown()


def test_openai_provider_malformed_response():
    class BadResponseHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            # Consume the request body first: closing a socket with unread
            # received data makes Windows send RST, which discards this
            # response and surfaces as an intermittent WinError 10053 abort
            # instead of the malformed-JSON behavior under test.
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            body = b"not valid json"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server, port = _start_server(BadResponseHandler)
    try:
        provider = OpenAICompatibleProvider(
            model="test-model", base_url=f"http://127.0.0.1:{port}/v1")
        result = provider.invoke("chat", {"messages": [{"role": "user", "content": "Hi"}]})
        assert result.status == RequestStatus.FAILURE
        assert "malformed" in result.error.lower()
    finally:
        server.shutdown()


def test_openai_provider_missing_messages():
    provider = OpenAICompatibleProvider(
        model="test-model", base_url="http://localhost:8080/v1")
    result = provider.invoke("chat", {"not_messages": "bad"})
    assert result.status == RequestStatus.FAILURE
    assert "Missing or invalid 'messages'" in result.error


def test_openai_provider_empty_content():
    class EmptyContentHandler(MockOpenAIHandler):
        response_data = {
            "choices": [{"message": {"role": "assistant", "content": ""}}],
            "model": "test-model",
        }

    server, port = _start_server(EmptyContentHandler)
    try:
        provider = OpenAICompatibleProvider(
            model="test-model", base_url=f"http://127.0.0.1:{port}/v1")
        result = provider.invoke("chat", {"messages": [{"role": "user", "content": "Hi"}]})
        assert result.status == RequestStatus.FAILURE
        assert "empty" in result.error.lower()
    finally:
        server.shutdown()


def test_openai_provider_missing_choices():
    class NoChoicesHandler(MockOpenAIHandler):
        response_data = {"model": "test-model"}

    server, port = _start_server(NoChoicesHandler)
    try:
        provider = OpenAICompatibleProvider(
            model="test-model", base_url=f"http://127.0.0.1:{port}/v1")
        result = provider.invoke("chat", {"messages": [{"role": "user", "content": "Hi"}]})
        assert result.status == RequestStatus.FAILURE
        assert "empty" in result.error.lower()
    finally:
        server.shutdown()


def test_openai_provider_connection_error():
    provider = OpenAICompatibleProvider(
        model="test-model", base_url="http://127.0.0.1:9999/v1", timeout=1)
    result = provider.invoke("chat", {"messages": [{"role": "user", "content": "Hi"}]})
    assert result.status == RequestStatus.FAILURE
    assert "connection" in result.error.lower()


def test_openai_provider_timeout_config_env():
    os.environ["PRIMORA_OPENAI_TIMEOUT"] = "99"
    try:
        provider = OpenAICompatibleProvider(
            model="test-model", base_url="http://localhost:8080/v1")
        assert provider._timeout == 99
    finally:
        os.environ.pop("PRIMORA_OPENAI_TIMEOUT", None)


def test_openai_provider_timeout_config_constructor():
    os.environ["PRIMORA_OPENAI_TIMEOUT"] = "99"
    try:
        provider = OpenAICompatibleProvider(
            model="test-model", base_url="http://localhost:8080/v1", timeout=42)
        assert provider._timeout == 42
    finally:
        os.environ.pop("PRIMORA_OPENAI_TIMEOUT", None)


def test_openai_provider_timeout_default():
    os.environ.pop("PRIMORA_OPENAI_TIMEOUT", None)
    provider = OpenAICompatibleProvider(
        model="test-model", base_url="http://localhost:8080/v1")
    assert provider._timeout == 120


def test_openai_provider_base_url_default_and_env():
    os.environ.pop("PRIMORA_OPENAI_URL", None)
    provider = OpenAICompatibleProvider(model="test-model")
    assert provider._base_url == "http://localhost:8080/v1"

    os.environ["PRIMORA_OPENAI_URL"] = "http://127.0.0.1:9999/v1"
    try:
        provider = OpenAICompatibleProvider(model="test-model")
        assert provider._base_url == "http://127.0.0.1:9999/v1"
        provider = OpenAICompatibleProvider(
            model="test-model", base_url="http://override/v1")
        assert provider._base_url == "http://override/v1"
    finally:
        os.environ.pop("PRIMORA_OPENAI_URL", None)


def test_prime_provider_from_env_default_is_ollama():
    import PRIMORA.main as main_mod
    from PRIMORA.prime.providers import OllamaProvider

    provider = main_mod.prime_provider_from_env(env={})
    assert isinstance(provider, OllamaProvider)
    assert provider._base_url == "http://127.0.0.1:11434"
    assert provider._model == main_mod.DEFAULT_OLLAMA_MODEL
    assert provider._timeout == 120

    provider = main_mod.prime_provider_from_env(
        env={"PRIMORA_PROVIDER": "ollama", "PRIMORA_OLLAMA_MODEL": "m",
             "PRIMORA_OLLAMA_URL": "http://x:1", "PRIMORA_OLLAMA_TIMEOUT": "7"})
    assert isinstance(provider, OllamaProvider)
    assert provider._model == "m"
    assert provider._base_url == "http://x:1"
    assert provider._timeout == 7


def test_prime_provider_from_env_openai():
    import PRIMORA.main as main_mod

    provider = main_mod.prime_provider_from_env(env={"PRIMORA_PROVIDER": "openai"})
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._base_url == "http://localhost:8080/v1"
    assert provider._model == main_mod.DEFAULT_OPENAI_MODEL
    assert provider._timeout == 120

    provider = main_mod.prime_provider_from_env(env={
        "PRIMORA_PROVIDER": "openai",
        "PRIMORA_OPENAI_MODEL": "my-gguf",
        "PRIMORA_OPENAI_URL": "http://127.0.0.1:9999/v1",
        "PRIMORA_OPENAI_TIMEOUT": "30",
    })
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._model == "my-gguf"
    assert provider._base_url == "http://127.0.0.1:9999/v1"
    assert provider._timeout == 30

    provider = main_mod.prime_provider_from_env(
        env={"PRIMORA_PROVIDER": "llamacpp"})
    assert isinstance(provider, OpenAICompatibleProvider)

    provider = main_mod.prime_provider_from_env(
        env={"PRIMORA_PROVIDER": "OPENAI"})
    assert isinstance(provider, OpenAICompatibleProvider)


def test_prime_provider_from_env_unknown_fails_closed():
    import PRIMORA.main as main_mod

    try:
        main_mod.prime_provider_from_env(env={"PRIMORA_PROVIDER": "mystery"})
    except ValueError as e:
        assert "PRIMORA_PROVIDER" in str(e)
    else:
        raise AssertionError("expected ValueError for unsupported provider")


if __name__ == "__main__":
    test_openai_provider_success()
    test_openai_provider_payload_model_override()
    test_openai_provider_options_merged_top_level()
    test_openai_provider_http_error()
    test_openai_provider_malformed_response()
    test_openai_provider_missing_messages()
    test_openai_provider_empty_content()
    test_openai_provider_missing_choices()
    test_openai_provider_connection_error()
    test_openai_provider_timeout_config_env()
    test_openai_provider_timeout_config_constructor()
    test_openai_provider_timeout_default()
    test_openai_provider_base_url_default_and_env()
    test_prime_provider_from_env_default_is_ollama()
    test_prime_provider_from_env_openai()
    test_prime_provider_from_env_unknown_fails_closed()
    print("All OpenAICompatibleProvider unit tests passed!")
