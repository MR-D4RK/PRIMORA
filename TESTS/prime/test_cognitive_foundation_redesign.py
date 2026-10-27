"""Capability-led cognition and cross-session source retrieval without phrase routing."""
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from PRIMORA.aurora.loop import CognitiveLoop
from PRIMORA.prime.context.manager import ContextManager
from PRIMORA.prime.context.mind import Mind
from PRIMORA.prime.prime import Prime
from PRIMORA.prime.providers import Provider
from PRIMORA.specialists.tools.memory_evidence import MemoryEvidenceSpecialist


def memory(tmp_path: Path):
    cm = ContextManager(db_path=tmp_path / "isolated.db", hybrid_retrieval=False)
    store = cm._get_store()
    moment = (datetime.now().astimezone() - timedelta(hours=1)).replace(second=0, microsecond=0)
    first = "I'm currently watching a robotics tutorial on a video website"
    second = "It's an amazing tutorial"
    a, *_ = store.append("prior-session", 0, "user", first, timestamp=moment.timestamp())
    b, *_ = store.append("prior-session", 1, "user", second, timestamp=(moment + timedelta(minutes=1)).timestamp())
    store.append("prior-session", 2, "assistant", "I saved your favorite tutorial", timestamp=(moment + timedelta(minutes=2)).timestamp())
    return cm, a, b, moment


def test_unindexed_ongoing_experience_is_found_across_sessions(tmp_path):
    cm, a, b, moment = memory(tmp_path)
    # The old completion-only extractor did not have an index for this report.
    assert not cm._get_store().fetch_experience_events()
    result = MemoryEvidenceSpecialist(Mind(cm)).handle({
        "sources": ["experiences"], "subject": "robotics tutorial",
        "date_from": moment.date().isoformat(), "date_to": moment.date().isoformat(),
    })
    assert result.status.value == "success"
    evidence = result.output["evidence"]
    assert first_line(a, evidence) and first_line(b, evidence)
    assert "amazing tutorial" in evidence
    assert "I saved" not in evidence
    assert "possible_context_for_" in evidence


def first_line(message_id, evidence):
    return f"message {message_id}," in evidence


def test_exact_local_calendar_window_and_no_found_when_outside(tmp_path):
    cm, _, _, moment = memory(tmp_path)
    specialist = MemoryEvidenceSpecialist(Mind(cm))
    yesterday = (moment.date() - timedelta(days=1)).isoformat()
    result = specialist.handle({"sources": ["experiences"], "subject": "robotics",
                                "date_from": yesterday, "date_to": yesterday})
    assert result.status.value == "success"
    assert result.output["evidence"] == "No matching user-source evidence found."
    for payload in (
        {"sources": ["experiences"], "date_from": "2026-05-04"},
        {"sources": ["experiences"], "date_from": "2030-01-01", "date_to": "2029-12-01"},
        {"sources": ["experiences"], "subject": ""},
        {"sources": ["experiences"], "arbitrary_query": "DROP TABLE messages"},
    ):
        assert specialist.handle(payload).status.value == "failure"


def test_prime_registers_and_binds_memory_evidence():
    class StubProvider:
        def execute(self, *args, **kwargs):
            raise AssertionError("memory evidence must not call a model provider")
    prime = Prime(StubProvider())
    specialist = prime.registry.get_specialist("recall")
    assert specialist is not None and specialist._mind is prime.mind
    assert "recall" in prime.registry.get_names()
    assert "memory_evidence" not in prime.registry.get_names()


def test_production_backends_do_not_use_pre_turn_interpreter():
    # Construction is intentionally independent of model runtime initialization.
    from PRIMORA.aurora.brain import OpenAIBrain
    brain = object.__new__(OpenAIBrain)
    cm = SimpleNamespace()
    loop = CognitiveLoop(cm, brain)
    assert loop._enable_memory_interpreter is False


def test_one_cognitive_loop_uses_prime_memory_tool_without_intent_prepass(tmp_path):
    from PRIMORA.aurora.brain import BrainResult
    from PRIMORA.aurora.loop import CapabilityExecutor
    import json

    cm, first_id, _, moment = memory(tmp_path)
    prime = Prime(SimpleNamespace())
    prime._context, prime._mind = cm, Mind(cm)
    prime.registry.get_specialist("recall").bind(prime.mind)
    cmd = "<prime_request>\n" + json.dumps({
        "action": "recall",
        "payload": {"query": "When did I watch a robotics tutorial?",
                    "sources": ["experiences"], "subject": "robotics tutorial",
                    "date_from": moment.date().isoformat(),
                    "date_to": moment.date().isoformat()},
        "context": None, "metadata": None,
    }) + "\n</prime_request>"

    class Scripted:
        def __init__(self):
            self.calls = []

        def generate(self, messages, max_output_tokens=None, on_delta=None):
            self.calls.append(messages)
            content = cmd if len(self.calls) == 1 else "You told me you were watching a robotics tutorial."
            return BrainResult.success(content, "scripted")

        def get_model_name(self):
            return "scripted"

        def count_tokens(self, messages):
            return 50

    brain = Scripted()
    loop = CognitiveLoop(prime.mind, brain, executor=CapabilityExecutor(prime))
    # Production never calls the preflight interpreter or phrase handlers.
    result = loop.run([{"role": "user", "content": "What did I watch?"}], session_id="new-session")
    assert result.success
    assert len(brain.calls) == 2
    assert f"message {first_id}," in json.dumps(brain.calls[1])
    assert result.content.startswith("You told me")
