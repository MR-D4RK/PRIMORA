"""Regression: direct Aurora brain HTTP path must support strict templates."""
import json
from unittest.mock import patch

from PRIMORA.aurora.brain import OpenAIBrain
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.openai_messages import leading_system_message


def _brain():
    config = AuroraConfig(model_path="/fake", adapter_path="/fake")
    return OpenAIBrain(config=config, model="test", base_url="http://localhost:8080/v1")


class FakeResponse:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self):
        return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()


def test_aurora_brain_sends_exactly_one_leading_system_message():
    requests = []
    def capture(req, timeout=None):
        requests.append(json.loads(req.data))
        return FakeResponse()

    input_messages = [
        {"role": "system", "content": "Identity"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "system", "content": "Memory evidence"},
        {"role": "user", "content": "What do you remember?"},
    ]
    with patch("PRIMORA.aurora.brain.urllib.request.urlopen", capture):
        result = _brain().generate(input_messages)
    assert result.success
    assert requests[0]["messages"] == [
        {"role": "system", "content": "Identity\n\nMemory evidence"},
        input_messages[1], input_messages[2], input_messages[4],
    ]
    assert len(input_messages) == 5  # wire conversion is nonmutating


def test_system_message_added_after_initial_user():
    messages = [{"role": "user", "content": "Hello"},
                {"role": "system", "content": "Extra context"},
                {"role": "user", "content": "And now?"}]
    assert leading_system_message(messages) == [
        {"role": "system", "content": "Extra context"}, messages[0], messages[2]]


def test_ordinary_messages_and_multimodal_user_content_unchanged():
    messages = [{"role": "system", "content": "Identity"},
                {"role": "user", "content": [{"type": "text", "text": "Look"}]}]
    assert leading_system_message(messages) is messages
    no_system = [{"role": "user", "content": "Hi"}]
    assert leading_system_message(no_system) is no_system


def test_invalid_structured_system_message_rejected_before_http():
    result = _brain().generate([
        {"role": "system", "content": "Identity"},
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": [{"type": "text", "text": "Not supported"}]},
    ])
    assert not result.success
    assert "System message content must be text" in result.error
