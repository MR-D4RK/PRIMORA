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

        # A payload may select a different model than this provider's default.
        # Added so one provider can serve several specialist models, which is
        # what consult_specialist needs. When absent, behaviour is unchanged.
        model = payload.get("model") or self._model

        request_body = {
            "model": model,
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
            return PrimeResult.success({"content": content, "model": model})
        except Exception as e:
            return PrimeResult.failure(f"Ollama response parse error: {e}")


def _normalize_openai_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Send one leading system message to strict OpenAI-compatible templates.

    PRIMORA assembles its context from several source-backed system blocks.
    Some model chat templates, including Qwen3.5, reject every system message
    except the first one. Preserve the original non-system conversation order
    and consolidate system instructions in their original relative order.
    No memory content is modified or persisted by this transport adjustment.
    """
    positions = [index for index, message in enumerate(messages)
                 if isinstance(message, dict) and message.get("role") == "system"]
    if not positions or (positions == [0]):
        return messages
    contents = [messages[index].get("content") for index in positions]
    if any(not isinstance(content, str) for content in contents):
        # Do not discard or stringify structured system content silently.
        return messages
    combined = "\n\n".join(content for content in contents if content.strip())
    conversation = [message for message in messages
                    if not (isinstance(message, dict) and message.get("role") == "system")]
    return ([{"role": "system", "content": combined}] if combined else []) + conversation


class OpenAICompatibleProvider(Provider):
    """Chat provider for OpenAI-compatible HTTP servers (llama.cpp
    llama-server, vLLM, LM Studio, ...). Same ``invoke`` contract and
    result shape as ``OllamaProvider``; only the wire format differs:
    POST ``{base_url}/chat/completions`` with OpenAI-style body and
    ``choices[0].message.content`` extraction. The ``options`` payload
    key is merged into the request body top level (OpenAI places
    ``temperature``/``max_tokens`` there, not under a nested key).
    Construction performs no network I/O.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        timeout: int | None = None
    ):
        self._model = model
        self._base_url = base_url or os.environ.get("PRIMORA_OPENAI_URL", "http://localhost:8080/v1")
        self._timeout = timeout or int(os.environ.get("PRIMORA_OPENAI_TIMEOUT", "120"))

    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return PrimeResult.failure("Missing or invalid 'messages' in payload")

        # Same payload-level model override as OllamaProvider, so one
        # provider instance can serve several specialist models.
        model = payload.get("model") or self._model

        request_body: dict[str, Any] = {
            "model": model,
            "messages": _normalize_openai_system_messages(messages),
            "stream": False,
        }

        if isinstance(payload.get("options"), dict):
            request_body.update(payload["options"])

        url = f"{self._base_url.rstrip('/')}/chat/completions"

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(request_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status != 200:
                    return PrimeResult.failure(f"OpenAI HTTP {resp.status}: {resp.read().decode('utf-8')}")
                response_data = json.loads(resp.read().decode("utf-8"))

        except urllib.error.HTTPError as e:
            return PrimeResult.failure(f"OpenAI HTTP error {e.code}: {e.read().decode('utf-8')}")
        except urllib.error.URLError as e:
            return PrimeResult.failure(f"OpenAI connection error: {e.reason}")
        except json.JSONDecodeError as e:
            return PrimeResult.failure(f"OpenAI malformed response: {e}")
        except TimeoutError:
            return PrimeResult.failure(f"OpenAI timeout after {self._timeout}s")
        except Exception as e:
            return PrimeResult.failure(f"OpenAI unexpected error: {e}")

        try:
            choices = response_data.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            content = message.get("content") or ""
            if not content:
                return PrimeResult.failure("OpenAI returned empty response")
            return PrimeResult.success({"content": content, "model": model})
        except Exception as e:
            return PrimeResult.failure(f"OpenAI response parse error: {e}")