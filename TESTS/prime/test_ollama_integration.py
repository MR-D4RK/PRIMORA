import sys
import os
import json
import urllib.request
import urllib.error

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.providers import OllamaProvider
from PRIMORA.prime.contract import PrimeResult, RequestStatus


def is_ollama_available(base_url: str = "http://127.0.0.1:11434") -> bool:
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_available_models(base_url: str = "http://127.0.0.1:11434") -> list[str]:
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def test_ollama_integration():
    base_url = os.environ.get("PRIMORA_OLLAMA_URL", "http://127.0.0.1:11434")

    if not is_ollama_available(base_url):
        print(f"SKIP: Ollama not available at {base_url}")
        return

    models = get_available_models(base_url)
    if not models:
        print(f"SKIP: No models available in Ollama at {base_url}")
        return

    model = os.environ.get("PRIMORA_OLLAMA_MODEL") or models[0]
    print(f"Using model: {model} from {base_url}")

    provider = OllamaProvider(model=model, base_url=base_url)
    result = provider.invoke("chat", {
        "messages": [{"role": "user", "content": "Say 'hello' and nothing else"}]
    })

    if result.status == RequestStatus.FAILURE:
        if "not found" in result.error.lower() or "does not exist" in result.error.lower():
            print(f"SKIP: Model '{model}' not available")
            return
        print(f"FAIL: {result.error}")
        assert False, f"Ollama request failed: {result.error}"

    assert result.status == RequestStatus.SUCCESS
    assert "content" in result.output
    assert result.output["model"] == model
    assert len(result.output["content"]) > 0
    print(f"SUCCESS: Got response: {result.output['content'][:50]}...")


if __name__ == "__main__":
    test_ollama_integration()
    print("Ollama integration test completed!")