"""Regression for short speech/text failing before the brain receives it.

Runs with a fake char-counting llama.cpp brain and a temporary SQLite DB.
The real Qwen server is not needed.
"""
from __future__ import annotations

import pytest

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import BrainResult
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.aurora.prompt_fit import fit_system_prompt
from PRIMORA.prime import Prime
from PRIMORA.prime.context import ContextManager, ContextTooLargeError
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.toolbox import ToolRegistry


class _Provider:
    def invoke(self, capability, payload):
        return PrimeResult.success({"ok": True})


class _CharCountingBrain:
    def __init__(self):
        self.seen = []
    def count_tokens(self, messages):
        return sum(max(1, len(str(m.get("content", ""))) // 4) + 4
                   for m in messages)
    def get_model_name(self):
        return "stub-llama"
    def generate(self, messages, max_output_tokens=None):
        self.seen.append(list(messages))
        return BrainResult.success("Yes, I hear you.", self.get_model_name())


def _aurora(tmp_path):
    manager = ContextManager(
        db_path=tmp_path / "voice-regression.db",
        enable_memory_v2=False, hybrid_retrieval=False,
    )
    prime = Prime(
        _Provider(), context_manager=manager,
        tools=ToolRegistry.with_builtins(),
    )
    brain = _CharCountingBrain()
    aurora = Aurora(
        prime, brain=brain,
        config=AuroraConfig(model_path="not-installed", adapter_path="not-installed"),
    )
    return aurora, brain, manager


def test_short_text_turn_is_not_blocked_by_catalogue(tmp_path):
    aurora, brain, manager = _aurora(tmp_path)
    turn = "can you hear me"
    result = aurora.respond(
        [{"role": "user", "content": turn}], session_id="test"
    )
    assert result.success, result.error
    assert result.content == "Yes, I hear you."
    assert len(brain.seen) == 1
    assert any(m["role"] == "user" and m["content"] == turn
               for m in brain.seen[0])
    assert any(m.role == "user" and m.content == turn
               for m in manager._get_store().fetch_range("test"))


def test_fitted_prompt_keeps_identity_and_memory_safety(tmp_path):
    aurora, brain, _ = _aurora(tmp_path)
    assembled = aurora._with_instruction([
        {"role": "user", "content": "can you hear me"}
    ])
    prompt = assembled[0]["content"]
    assert "IDENTITY AND VERSION" in prompt
    assert "You are Aurora" in prompt
    assert "GROUNDING." in prompt
    assert "Never do Prime's work yourself" in prompt
    assert "research_read" in prompt
    assert brain.count_tokens([assembled[0]]) <= (
        aurora._config.context_target - 250
    )


def test_optional_sections_shrink_but_core_and_required_fields_remain():
    core = "Never invent user facts. " * 48
    identity = "\nYou are Aurora and must preserve original sources.\n"
    capability = "\nAvailable Prime Capabilities:\n" + "".join(
        f"- operation_{i}: " + ("optional explanation " * 25)
        + "\n  Required: query\n" for i in range(6)
    )
    specialist = "\nAvailable Specialists:\n- reviewer: " + "analysis " * 180
    tools = "\nAvailable Tools:\n- helper: " + "details " * 200
    guidance = "\nDo not answer with fabricated tool results."
    count = lambda text: max(1, len(text) // 4) + 4
    output = fit_system_prompt(
        core, identity, capability, specialist, tools, guidance,
        count=count, ceiling=1600, target=1500,
    )
    assert count(output) <= 1250
    assert core in output
    assert identity in output
    assert guidance in output
    for i in range(6):
        assert f"operation_{i}" in output
    assert "Required: query" in output


def test_oversized_user_turn_gets_user_limit_error(tmp_path):
    manager = ContextManager(
        db_path=tmp_path / "too-long.db", hybrid_retrieval=False
    )
    from PRIMORA.prime.context.models import BudgetConfig

    budget = BudgetConfig(
        input_target=80, input_ceiling=100,
        output_reserve=20, total_budget=120, state_reserve=5,
    )
    with pytest.raises(ContextTooLargeError, match="Current user message alone"):
        manager.build(
            [{"role": "system", "content": "Short system."},
             {"role": "user", "content": "A" * 900}],
            "overlong", budget=budget,
        )


def test_oversized_system_is_not_mislabeled_user_error(tmp_path):
    manager = ContextManager(
        db_path=tmp_path / "system-too-long.db", hybrid_retrieval=False,
    )
    from PRIMORA.prime.context.models import BudgetConfig

    budget = BudgetConfig(
        input_target=80, input_ceiling=100,
        output_reserve=20, total_budget=120, state_reserve=5,
    )
    with pytest.raises(ContextTooLargeError, match="System instructions exceed"):
        manager.build(
            [{"role": "system", "content": "S" * 900},
             {"role": "user", "content": "can you hear me"}],
            "system", budget=budget,
        )


def test_repeated_system_instruction_keeps_identity_once(tmp_path):
    aurora, _, _ = _aurora(tmp_path)
    first = aurora._with_instruction([
        {"role": "user", "content": "hello"}
    ])[0]
    second = aurora._with_instruction([
        first, {"role": "user", "content": "hello"}
    ])[0]
    assert second["content"].count("IDENTITY AND VERSION") == 1
    assert second["content"].count("REQUEST DISCIPLINE") <= 1
