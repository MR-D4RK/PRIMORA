"""Regression contracts for current-turn versus historical memory cognition."""

import json
import time

from PRIMORA.aurora.brain import BrainResult
from PRIMORA.aurora.loop import CognitiveLoop
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.mind import Mind
from PRIMORA.aurora.memory_coordinator import validate_plan


class _PlannerBrain:
    def __init__(self, plan, answer="Understood."):
        self.plan = plan
        self.answer = answer
        self.calls = []

    def get_model_name(self):
        return "planner-test"

    def count_tokens(self, messages):
        return sum(len(str(m.get("content", ""))) // 4 + 1 for m in messages)

    def generate(self, messages, max_output_tokens=None, on_delta=None,
                 cancelled=None):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return BrainResult.success(json.dumps(self.plan), self.get_model_name())
        return BrainResult.success(self.answer, self.get_model_name())


def _current_plan(operation):
    return {
        "intent": "memory", "operation": operation, "sources": ["preferences"],
        "subject": None, "recent_days": None, "first_mention": False,
        "reasons": False, "clarify": False, "result_count": None,
        "order": "relevance", "temporal_scope": "current",
    }


def test_current_disclosure_returns_to_normal_conversation(tmp_path):
    brain = _PlannerBrain(_current_plan("current_turn"), "That is useful context.")
    loop = CognitiveLoop(
        Mind(ContextManager(db_path=tmp_path / "disclosure.db")), brain,
        enable_memory_interpreter=True,
    )
    result = loop.run(
        [{"role": "user", "content": "My new reason is that I enjoy it."}],
        "s",
    )
    assert result.success and result.content == "That is useful context."
    assert len(brain.calls) == 2


def test_current_correction_is_not_replaced_by_historical_lookup(tmp_path):
    brain = _PlannerBrain(_current_plan("correction"), "I understand the correction.")
    loop = CognitiveLoop(
        Mind(ContextManager(db_path=tmp_path / "correction.db")), brain,
        enable_memory_interpreter=True,
    )
    result = loop.run(
        [{"role": "user", "content": "Actually, the earlier example was fictional."}],
        "s",
    )
    assert result.success and "correction" in result.content
    assert len(brain.calls) == 2


def test_confirmation_does_not_trigger_an_unrelated_memory_answer(tmp_path):
    brain = _PlannerBrain(_current_plan("confirmation"), "Yes, understood.")
    loop = CognitiveLoop(
        Mind(ContextManager(db_path=tmp_path / "confirmation.db")), brain,
        enable_memory_interpreter=True,
    )
    result = loop.run(
        [{"role": "assistant", "content": "Should I keep that distinction?"},
         {"role": "user", "content": "Yes."}],
        "s",
    )
    assert result.success and result.content == "Yes, understood."
    assert len(brain.calls) == 2


def test_memory_plan_supports_bounded_recent_results():
    plan = validate_plan({
        **_current_plan("recall"),
        "result_count": 3,
        "order": "recent",
        "temporal_scope": "recent",
    })
    assert plan.result_count == 3
    assert plan.order == "recent"
    assert plan.temporal_scope == "recent"


def test_recent_evidence_uses_requested_order_and_count(tmp_path):
    manager = ContextManager(db_path=tmp_path / "recent.db")
    store = manager._get_store()
    now = time.time()
    for seq, value in enumerate(("alpha", "beta", "gamma", "delta")):
        message_id, *_ = store.append(
            "s", seq, "user", f"I like {value}", timestamp=now + seq)
        store.create_state(
            "s", "global", f"likes for {value}", f"loves {value}",
            [message_id], timestamp=now + seq,
        )
    evidence = manager.memory_query_evidence({
        "sources": ["preferences"], "subject": None,
        "result_count": 3, "order": "recent", "temporal_scope": "recent",
        "first_mention": False, "reasons": False,
    }, session_id="s")
    assert "Requested results: 3; order: recent" in evidence
    assert evidence.count("[message ") == 3
    assert "delta" in evidence and "alpha" not in evidence

