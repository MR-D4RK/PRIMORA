"""Production-path contracts for the unified evidence memory milestone."""

import json

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import Brain, BrainResult, OpenAIBrain
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.prime import Prime
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.models import BudgetConfig
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.providers import Provider
from PRIMORA.main import build_prime


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.failure("unused")


class _Brain(Brain):
    def generate(self, messages, max_output_tokens=None, on_delta=None,
                 cancelled=None):
        return BrainResult.success("done", "test")

    def get_model_name(self):
        return "test"

    def count_tokens(self, messages):
        return sum(len(str(m.get("content", ""))) // 4 + 1 for m in messages)


def test_explicit_context_injection_rebinds_prime_capabilities(tmp_path):
    manager = ContextManager(db_path=tmp_path / "shared.db")
    prime = Prime(_Provider())
    aurora = Aurora(
        prime,
        brain=_Brain(),
        config=AuroraConfig("model", "adapter"),
        context_manager=manager,
    )

    assert prime.context is manager
    assert prime.mind is aurora.mind
    assert prime.registry.get_specialist("recall")._mind is aurora.mind
    assert prime.registry.get_specialist("learn")._mind is aurora.mind


def test_latest_capability_evidence_is_retained_under_budget_pressure(tmp_path):
    manager = ContextManager(db_path=tmp_path / "evidence.db", hybrid_retrieval=False)
    request = {
        "role": "assistant",
        "content": '<prime_request>{"action":"recall"}</prime_request>',
    }
    result = {
        "role": "tool",
        "content": json.dumps({
            "status": "success",
            "output": {"results": [{"content": "verified " + "x" * 5000}]},
        }),
    }
    built = manager.build(
        [
            {"role": "system", "content": "system " * 20},
            {"role": "user", "content": "What do I remember?"},
            request,
            result,
        ],
        "s",
        token_counter=lambda messages: sum(
            len(str(message.get("content", ""))) // 4 + 1
            for message in messages
        ),
        max_new_tokens=16,
        budget=BudgetConfig(
            input_target=60, input_ceiling=90,
            output_reserve=16, total_budget=106, state_reserve=0,
        ),
        protected_messages=[request, result],
    )

    roles = [message["role"] for message in built.view.messages]
    assert "user" in roles and "tool" in roles
    assert roles.index("user") < roles.index("tool")
    assert any("evidence_" in diagnostic for diagnostic in built.view.diagnostics)
    assert any("truncated" in message["content"] for message in built.view.messages
               if message["role"] == "tool")


def test_production_openai_brain_enables_semantic_memory_interpreter(monkeypatch):
    monkeypatch.setenv("PRIMORA_AURORA_MEMORY_INTERPRETER", "1")
    prime = Prime(_Provider())
    brain = OpenAIBrain(AuroraConfig("model", "adapter"))
    aurora = Aurora(prime, brain=brain, config=AuroraConfig("model", "adapter"))
    assert aurora.loop._enable_memory_interpreter is True


def test_production_composition_wires_learning_and_safe_tools():
    prime = build_prime(_Provider())
    assert prime.registry.get_specialist("learn").is_configured is True
    assert prime.registry.get_specialist("use_tool").is_configured is True
    assert prime.registry.get_specialist("consult_specialist").is_configured is False


def test_formation_status_is_idempotent_and_replayable(tmp_path):
    manager = ContextManager(db_path=tmp_path / "formation.db")
    manager.reconcile(
        [{"role": "user", "content": "I love an unfamiliar instrument."}],
        "s",
    )
    statuses = manager.formation_status("s")
    assert statuses and statuses[0]["status"] == "complete"
    message_id = manager._get_store().fetch_latest("s", 1)[0].message_id

    manager.replay_formation("s", message_id)
    assert manager.formation_status("s")[0]["status"] == "complete"
    assert manager._get_store().count("s") == 1

