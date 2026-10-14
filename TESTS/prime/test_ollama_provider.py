import sys
import json
import threading
import time
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.providers import OllamaProvider
from PRIMORA.prime.contract import PrimeResult, RequestStatus


class MockOllamaHandler(BaseHTTPRequestHandler):
    response_data = {"message": {"role": "assistant", "content": "Test response"}}
    status_code = 200

    def _read_request_body(self) -> bytes:
        # The POST body must be consumed before this handler returns.
        # Closing a socket with unread received data makes Windows send
        # RST instead of FIN, which discards the already-written response
        # and surfaces on the client as an intermittent WinError 10053
        # abort instead of the behavior under test.
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _send_json(self):
        body = json.dumps(self.response_data).encode("utf-8")
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self._read_request_body()
        self._send_json()

    def log_message(self, format, *args):
        pass


def test_ollama_provider_success():
    server = HTTPServer(("127.0.0.1", 0), MockOllamaHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    try:
        provider = OllamaProvider(model="test-model", base_url=f"http://127.0.0.1:{port}")
        result = provider.invoke("chat", {
            "messages": [{"role": "user", "content": "Hello"}]
        })
        assert result.status == RequestStatus.SUCCESS
        assert result.output["content"] == "Test response"
        assert result.output["model"] == "test-model"
    finally:
        server.shutdown()


def test_ollama_provider_http_error():
    class ErrorHandler(MockOllamaHandler):
        status_code = 500
        response_data = {"error": "Internal server error"}

    server = HTTPServer(("127.0.0.1", 0), ErrorHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    try:
        provider = OllamaProvider(model="test-model", base_url=f"http://127.0.0.1:{port}")
        result = provider.invoke("chat", {"messages": [{"role": "user", "content": "Hi"}]})
        assert result.status == RequestStatus.FAILURE
        assert "HTTP error 500" in result.error
    finally:
        server.shutdown()


def test_ollama_provider_malformed_response():
    class BadResponseHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
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

    server = HTTPServer(("127.0.0.1", 0), BadResponseHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    try:
        provider = OllamaProvider(model="test-model", base_url=f"http://127.0.0.1:{port}")
        result = provider.invoke("chat", {"messages": [{"role": "user", "content": "Hi"}]})
        assert result.status == RequestStatus.FAILURE
        assert "malformed" in result.error.lower()
    finally:
        server.shutdown()


def test_ollama_provider_missing_messages():
    provider = OllamaProvider(model="test-model", base_url="http://127.0.0.1:11434")
    result = provider.invoke("chat", {"not_messages": "bad"})
    assert result.status == RequestStatus.FAILURE
    assert "Missing or invalid 'messages'" in result.error


def test_ollama_provider_empty_content():
    class EmptyContentHandler(BaseHTTPRequestHandler):
        response_data = {"message": {"role": "assistant", "content": ""}}
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            if length:
                self.rfile.read(length)
            body = json.dumps(self.response_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), EmptyContentHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    try:
        provider = OllamaProvider(model="test-model", base_url=f"http://127.0.0.1:{port}")
        result = provider.invoke("chat", {"messages": [{"role": "user", "content": "Hi"}]})
        assert result.status == RequestStatus.FAILURE
        assert "empty" in result.error.lower()
    finally:
        server.shutdown()


def test_ollama_provider_connection_error():
    provider = OllamaProvider(model="test-model", base_url="http://127.0.0.1:9999", timeout=1)
    result = provider.invoke("chat", {"messages": [{"role": "user", "content": "Hi"}]})
    assert result.status == RequestStatus.FAILURE
    assert "connection" in result.error.lower()


def test_ollama_provider_timeout_config_env():
    os.environ["PRIMORA_OLLAMA_TIMEOUT"] = "99"
    try:
        provider = OllamaProvider(model="test-model", base_url="http://127.0.0.1:11434")
        assert provider._timeout == 99
    finally:
        os.environ.pop("PRIMORA_OLLAMA_TIMEOUT", None)


def test_ollama_provider_timeout_config_constructor():
    provider = OllamaProvider(model="test-model", base_url="http://127.0.0.1:11434", timeout=42)
    assert provider._timeout == 42


def test_ollama_provider_timeout_default():
    os.environ.pop("PRIMORA_OLLAMA_TIMEOUT", None)
    provider = OllamaProvider(model="test-model", base_url="http://127.0.0.1:11434")
    assert provider._timeout == 120


if __name__ == "__main__":
    test_ollama_provider_success()
    test_ollama_provider_http_error()
    test_ollama_provider_malformed_response()
    test_ollama_provider_missing_messages()
    test_ollama_provider_empty_content()
    test_ollama_provider_connection_error()
    test_ollama_provider_timeout_config_env()
    test_ollama_provider_timeout_config_constructor()
    test_ollama_provider_timeout_default()
    print("All OllamaProvider unit tests passed!")