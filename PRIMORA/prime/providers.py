from abc import ABC, abstractmethod
from typing import Any
import os
import json
import urllib.request
import urllib.error
from .contract import PrimeResult


class Provider(ABC):
    @abstractmethod
    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        pass


class OllamaProvider(Provider):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        timeout: int | None = None
    ):
        self._model = model
        self._base_url = base_url or os.environ.get("PRIMORA_OLLAMA_URL", "http://127.0.0.1:11434")
        self._timeout = timeout or int(os.environ.get("PRIMORA_OLLAMA_TIMEOUT", "120"))

    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return PrimeResult.failure("Missing or invalid 'messages' in payload")

        request_body = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }

        if "options" in payload:
            request_body["options"] = payload["options"]

        url = f"{self._base_url.rstrip('/')}/api/chat"

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(request_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status != 200:
                    return PrimeResult.failure(f"Ollama HTTP {resp.status}: {resp.read().decode('utf-8')}")
                response_data = json.loads(resp.read().decode("utf-8"))

        except urllib.error.HTTPError as e:
            return PrimeResult.failure(f"Ollama HTTP error {e.code}: {e.read().decode('utf-8')}")
        except urllib.error.URLError as e:
            return PrimeResult.failure(f"Ollama connection error: {e.reason}")
        except json.JSONDecodeError as e:
            return PrimeResult.failure(f"Ollama malformed response: {e}")
        except TimeoutError:
            return PrimeResult.failure(f"Ollama timeout after {self._timeout}s")
        except Exception as e:
            return PrimeResult.failure(f"Ollama unexpected error: {e}")

        try:
            message = response_data.get("message", {})
            content = message.get("content", "")
            if not content:
                return PrimeResult.failure("Ollama returned empty response")
            return PrimeResult.success({"content": content, "model": self._model})
        except Exception as e:
            return PrimeResult.failure(f"Ollama response parse error: {e}")