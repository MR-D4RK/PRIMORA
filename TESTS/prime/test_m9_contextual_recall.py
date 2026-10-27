"""M9: model-led memory requests stay inside the Prime evidence boundary."""

import json

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.prime import Prime
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.providers import Provider


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.failure("unused")


def _request(query, *, subject=None, sources=("preferences",), reasons=False):
    payload = {"query": query, "sources": list(sources), "reasons": reasons}
    if subject is not None:
        payload["subject"] = subject
    return "<prime_request>\n" + json.dumps({
        "action": "recall", "payload": payload,
        "context": None, "metadata": None,
    }) + "\n</prime_request>"


class _Model(Brain):
    """A scripted Qwen stand-in: outputs are supplied, never classified here."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, messages, max_output_tokens=None, on_delta=None,
                 cancelled=None):
        self.calls.append(messages)
        if not self.outputs:
            return BrainResult.failure("script exhausted", self.get_model_name())
        return BrainResult.success(self.outputs.pop(0), self.get_model_name())

    def get_model_name(self):
        return "test-qwen"

    def count_tokens(self, messages):
        return sum(max(1, len(str(item.get("content", ""))) // 4)
                   for item in messages)


def _seed(manager):
    store = manager._get_store()
    music, *_ = store.append(
        "seed-music", 0, "user",
        "I enjoy techno music when working. It helps me stay focused and calm.",
    )
    art, *_ = store.append(
        "seed-art", 0, "user",
        "I enjoy watercolor painting on weekends.",
    )
    store.create_state(
        "seed-music", "global", "likes for techno music when working",
        "loves techno music when working", [music], memory_type="PREFERENCE",
    )
    store.create_state_with_explains(
        "seed-music", "global",
        "rationale for likes for techno music when working",
        "It helps me stay focused and calm", [music], 1,
        memory_type="RATIONALE",
    )
    store.create_state(
        "seed-art", "global", "likes for watercolor painting",
        "loves watercolor painting", [art], memory_type="PREFERENCE",
    )


def _aurora(tmp_path, outputs):
    manager = ContextManager(
        db_path=tmp_path / "contextual.db", enable_memory_v2=True,
        hybrid_retrieval=False,
    )
    _seed(manager)
    model = _Model(outputs)
    return Aurora(
        Prime(_Provider(), context_manager=manager), brain=model,
        config=AuroraConfig("model", "adapter"),
    ), model


def test_model_requests_recall_for_unseen_followup_and_gets_verified_reason(tmp_path):
    model_outputs = [
        _request("Why did I listen to techno music when working?",
                 subject="techno music when working", reasons=True),
        "It helps you stay focused and calm.",
    ]
    aurora, model = _aurora(tmp_path, model_outputs)
    result = aurora.respond([
        {"role": "user", "content": "What music helps me focus at work?"},
        {"role": "assistant", "content": "Techno."},
        {"role": "user", "content": "And the reason?"},
    ], session_id="follow-up")

    assert result.success
    assert "focused and calm" in result.content
    assert len(model.calls) == 2  # one model-led request, one answer
    tool = next(message for message in model.calls[1] if message["role"] == "tool")
    assert "focused and calm" in tool["content"]
    assert "Techno." not in tool["content"]


def test_model_abstains_when_prime_returns_no_verified_reason(tmp_path):
    aurora, model = _aurora(tmp_path, [
        _request("Explain why I enjoy watercolor", subject="watercolor",
                 reasons=True),
        "I couldn't verify a user-stated reason.",
    ])
    result = aurora.respond([
        {"role": "user", "content": "I like watercolor on weekends."},
        {"role": "assistant", "content": "Watercolor."},
        {"role": "user", "content": "What is the reason?"},
    ], session_id="no-reason")
    assert result.success
    assert "couldn't verify" in result.content
    assert len(model.calls) == 2


def test_assistant_reply_is_context_only_and_never_source_evidence(tmp_path):
    aurora, model = _aurora(tmp_path, [
        _request("When did I tell you about the glass harp?",
                 subject="glass harp", sources=("facts",)),
        "You told me in the earlier conversation.",
    ])
    store = aurora.mind.mechanisms._get_store()
    source, *_ = store.append("old", 0, "user", "My favorite instrument is a glass harp")
    store.create_state("old", "global", "favorite instrument", "glass harp",
                       [source], memory_type="FACT")
    result = aurora.respond([
        {"role": "user", "content": "Which instrument did I mention?"},
        {"role": "assistant", "content": "The glass harp."},
        {"role": "user", "content": "When did I tell you about it?"},
    ], session_id="new")
    assert result.success
    evidence = next(message["content"] for message in model.calls[1]
                     if message["role"] == "tool")
    assert "glass harp" in evidence
    assert "The glass harp." not in evidence


def test_ordinary_turn_has_one_model_generation_and_no_planner(tmp_path):
    aurora, model = _aurora(tmp_path, ["That sounds interesting."])
    result = aurora.respond([
        {"role": "user", "content": "Tell me about an unfamiliar orbital garden."}
    ], session_id="ordinary")
    assert result.success and result.content == "That sounds interesting."
    assert len(model.calls) == 1
    assert aurora.loop.last_trace.timings_ms["planning_ms"] < 1.0


def test_cross_session_model_led_recall_uses_same_prime_store(tmp_path):
    db = tmp_path / "cross-session.db"
    manager = ContextManager(db_path=db, enable_memory_v2=True,
                             hybrid_retrieval=False)
    store = manager._get_store()
    source, *_ = store.append("first", 0, "user",
                              "I prefer the moonstone atlas for planning.")
    store.create_state("first", "global", "likes for moonstone atlas",
                       "prefers moonstone atlas", [source],
                       memory_type="PREFERENCE")
    first = _Model([_request("What planning guide do I prefer?",
                             subject="moonstone atlas"),
                    "You prefer the moonstone atlas."])
    Aurora(Prime(_Provider(), context_manager=manager), brain=first,
           config=AuroraConfig("model", "adapter")).respond(
               [{"role": "user", "content": "What planning guide do I prefer?"}],
               session_id="second")
    assert len(first.calls) == 2
