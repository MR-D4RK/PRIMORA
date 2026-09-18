import sys
import json
import threading
import time
import urllib.request
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.prime import Prime
from PRIMORA.prime.providers import Provider
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.capability import CapabilityRegistry, CapabilitySpec
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.configuration import AuroraConfig
from typing import Any, Optional

import PRIMORA.aurora.interface.server as server_module

TEST_PORT = 18420
BLOCK = ('<prime_request>\n{"action": "calculate", "payload": {"expression": "2 + 3"}, '
         '"context": null, "metadata": null}\n</prime_request>')

CHAT_BLOCK = (
    '<prime_request>\n'
    '{"action": "chat", "payload": {"messages": []}, "context": null, "metadata": null}\n'
    '</prime_request>'
)


class MockProvider(Provider):
    def __init__(self):
        self.calls = []

    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        self.calls.append((capability, payload))
        return PrimeResult.success({"result": 5})


class MockScriptedBrain(Brain):
    def __init__(self, outputs: list, model: str = "merge-test-model"):
        self._outputs = list(outputs)
        self._model = model

    def generate(self, messages: list[dict[str, str]]) -> BrainResult:
        return self._outputs.pop(0)

    def get_model_name(self) -> str:
        return self._model


def _test_config(**kwargs):
    params = dict(model_path="/fake/model", adapter_path="/fake/adapter")
    params.update(kwargs)
    return AuroraConfig(**params)


def _start_server(aurora, port):
    thread = threading.Thread(
        target=server_module.serve, args=(aurora,),
        kwargs={"host": "127.0.0.1", "port": port}, daemon=True)
    thread.start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1).read()
            return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError("test server did not start")


def _post(port, messages):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/chat",
        data=json.dumps({"messages": messages}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_server_has_no_duplicate_init():
    assert not hasattr(server_module, "init_aurora")
    assert hasattr(server_module, "serve")


def test_ui_uses_same_aurora_instance():
    provider = MockProvider()
    brain = MockScriptedBrain([BrainResult.success("Hello from the one Aurora!", "merge-test-model")])
    aurora = Aurora(Prime(provider), brain=brain, config=_test_config())
    _start_server(aurora, TEST_PORT)
    data = _post(TEST_PORT, [{"role": "user", "content": "Hi"}])
    assert data["success"] is True
    assert data["content"] == "Hello from the one Aurora!"
    assert data["model"] == "merge-test-model"
    assert len(provider.calls) == 0


def test_ui_prime_flow_over_http():
    provider = MockProvider()

    brain = MockScriptedBrain([
        BrainResult.success(BLOCK, "merge-test-model"),
        BrainResult.success("The answer is 5.", "merge-test-model"),
    ])

    aurora = Aurora(
        Prime(provider),
        brain=brain,
        config=_test_config()
    )

    _start_server(aurora, TEST_PORT + 1)

    data = _post(TEST_PORT + 1, [
        {"role": "user", "content": "What is 2+3?"}
    ])

    assert data["success"] is True
    assert data["content"] == "The answer is 5."
    assert len(provider.calls) == 0


if __name__ == "__main__":
    test_server_has_no_duplicate_init()
    test_ui_uses_same_aurora_instance()
    test_ui_prime_flow_over_http()
    print("All server merge tests passed!")
