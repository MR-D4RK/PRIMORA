"""M9 Batch 2: contextual follow-ups stay grounded in Prime evidence."""

import json
import time

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import BrainResult, OpenAIBrain
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.aurora.memory_coordinator import interpret
from PRIMORA.prime import Prime
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.providers import Provider
from PRIMORA.prime.contract import PrimeResult


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.failure("unused")


class _RuntimeBrain(OpenAIBrain):
    """OpenAI-brain-shaped fake so Aurora enables the production interpreter."""

    def __init__(self):
        self._config = AuroraConfig("model", "adapter")
        self._model = "test-qwen"
        self.calls = []

    def count_tokens(self, messages):
        return sum(max(1, len(str(item.get("content", ""))) // 4)
                   for item in messages)

    def generate(self, messages, max_output_tokens=None, on_delta=None,
                 cancelled=None):
        self.calls.append(messages)
        system = str(messages[0].get("content", ""))
        if "interpreting the LATEST USER MESSAGE" in system:
            # Force the deterministic contextual resolver to do the follow-up
            # work; this mirrors a model that conservatively says "other" for
            # a bare pronoun/question.
            return BrainResult.success(json.dumps({"intent": "other"}),
                                       self.get_model_name())
        if "using ONLY the evidence below" in system:
            evidence = str(messages[-1].get("content", ""))
            return BrainResult.success(
                "The verified reason is in your original statement.",
                self.get_model_name(),
            ) if "focused and calm" in evidence else BrainResult.success(
                "I couldn't verify a reason for that preference.",
                self.get_model_name(),
            )
        return BrainResult.success("Techno. It helps you stay focused and calm.",
                                   self.get_model_name())


def _seed_preferences(manager):
    store = manager._get_store()
    now = time.time()
    music, *_ = store.append(
        "seed-music", 0, "user",
        "I enjoy techno music when working. It helps me stay focused and calm.",
        timestamp=now - 30,
    )
    watercolor, *_ = store.append(
        "seed-art", 0, "user",
        "I enjoy watercolor painting on weekends because it slows me down.",
        timestamp=now - 20,
    )
    store.create_state(
        "seed-music", "global", "likes for techno music when working",
        "loves techno music when working", [music], memory_type="PREFERENCE",
    )
    store.create_state(
        "seed-music", "global",
        "rationale for likes for techno music when working",
        "it helps me stay focused and calm", [music], memory_type="RATIONALE",
    )
    store.create_state(
        "seed-art", "global", "likes for watercolor painting",
        "loves watercolor painting", [watercolor], memory_type="PREFERENCE",
    )
    store.create_state(
        "seed-art", "global", "rationale for likes for watercolor painting",
        "it slows me down", [watercolor], memory_type="RATIONALE",
    )


def _aurora(tmp_path):
    manager = ContextManager(
        db_path=tmp_path / "contextual.db", enable_memory_v2=True,
        hybrid_retrieval=False,
    )
    _seed_preferences(manager)
    brain = _RuntimeBrain()
    prime = Prime(_Provider(), context_manager=manager)
    return Aurora(
        prime, brain=brain, config=AuroraConfig("model", "adapter"),
    ), brain


def test_aurora_respond_resolves_why_and_keeps_unrelated_preference_out(tmp_path):
    aurora, brain = _aurora(tmp_path)
    result = aurora.respond([
        {"role": "user", "content": "What music do I listen to when working?"},
        {"role": "assistant", "content":
         "Techno. It helps you stay focused and calm."},
        {"role": "user", "content": "Why?"},
    ], session_id="follow-up")

    assert result.success
    assert "verified reason" in result.content
    evidence = str(brain.calls[-1][-1]["content"])
    assert "focused and calm" in evidence
    assert "watercolor" not in evidence
    # The assistant's topic pointer is present in planning context only; it
    # never becomes a source quote or an evidence line.
    assert "Techno. It helps you" not in evidence


def test_confirmation_followup_reuses_verified_reason_across_sessions(tmp_path):
    aurora, brain = _aurora(tmp_path)
    result = aurora.respond([
        {"role": "user", "content": "What music do I listen to when working?"},
        {"role": "assistant", "content":
         "Techno. It helps you stay focused and calm."},
        {"role": "user", "content": "Why?"},
        {"role": "assistant", "content": "You haven't explicitly stated why."},
        {"role": "user", "content": "I think I have told you."},
    ], session_id="new-session")

    assert result.success
    assert "verified reason" in result.content
    evidence = str(brain.calls[-1][-1]["content"])
    assert "focused and calm" in evidence
    assert "watercolor" not in evidence

    already = aurora.respond([
        {"role": "user", "content": "What music do I listen to when working?"},
        {"role": "assistant", "content": "Techno."},
        {"role": "user", "content": "Why?"},
        {"role": "assistant", "content": "You haven't explicitly stated why."},
        {"role": "user", "content": "I already told you."},
    ], session_id="new-session-2")
    assert already.success
    assert "verified reason" in already.content


def test_aurora_abstains_when_contextual_reason_has_no_user_evidence(tmp_path):
    aurora, brain = _aurora(tmp_path)
    store = aurora.mind.mechanisms._get_store()
    # Remove only the verified rationale; the preference itself remains.
    with store._connection() as connection:
        connection.execute(
            "DELETE FROM states WHERE concept LIKE 'rationale for %'"
        )
    result = aurora.respond([
        {"role": "user", "content": "What music do I listen to when working?"},
        {"role": "assistant", "content": "Techno."},
        {"role": "user", "content": "Why?"},
    ], session_id="no-reason")

    assert result.success
    assert "couldn't verify a user-stated reason" in result.content
    # The explicit abstention is returned before a planner or answer
    # generation; this clear follow-up needs no model call at all.
    assert len(brain.calls) == 0


def test_contextual_when_and_topic_followups_get_bounded_plans():
    class _Planner:
        def generate(self, messages, max_output_tokens=None):
            return BrainResult.success(json.dumps({"intent": "other"}), "test")

    previous = (
        {"role": "user", "content": "What music do I listen to when working?"},
        {"role": "assistant", "content":
         "Techno. It helps you stay focused and calm."},
    )
    when = interpret(_Planner(), "When?", previous)
    topic = interpret(_Planner(), "What about it?", previous)
    assert when.subject == "Techno"
    assert when.first_mention is True and when.temporal_scope == "historical"
    assert topic.subject == "Techno"
    assert topic.reasons is False

    assistant_pointer = interpret(
        _Planner(), "Why?",
        ({"role": "user", "content": "Can you remind me?"},
         {"role": "assistant", "content": "Techno."}),
    )
    assert assistant_pointer.subject == "Techno"


def test_ordinary_conversation_summary_does_not_invoke_memory_planner(tmp_path):
    planner = type("Planner", (), {
        "generate": lambda self, *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("summary should not invoke memory planning"))
    })()
    assert interpret(planner, "What have you said so far?", ()) == "other"

    # The same guard holds through Aurora.respond(), where a Prime capability
    # request would otherwise add an unnecessary memory operation.
    aurora, brain = _aurora(tmp_path)
    result = aurora.respond(
        [{"role": "user", "content": "What have you said so far?"}],
        session_id="summary",
    )
    assert result.success
    assert len(brain.calls) == 1


def test_ambiguous_why_still_clarifies():
    class _Planner:
        def generate(self, messages, max_output_tokens=None):
            return BrainResult.success(json.dumps({"intent": "other"}), "test")

    plan = interpret(
        _Planner(), "Why?",
        ({"role": "user", "content": "That was an unusual day."},
         {"role": "assistant", "content": "I agree."}),
    )
    assert plan.clarify is True

