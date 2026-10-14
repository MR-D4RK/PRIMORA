"""Phase A: repairing the JSON a local model damages, and not lying about it.

The v0.4 adapter answers in prose where v0.1 and v0.3 emit a request block.
Two separate defects sit behind that, and this file pins the fixes for both.

The first is repairable damage. A model that has learned the shape of
``<prime_request>`` still arrives at max_tokens mid-object, leaves a trailing
comma, or truncates a string. Strict JSON rejects all three, the request is
discarded, and the session reports a normal answer even though Aurora asked
for something and nothing ran. ``_repair_json_object`` restores structure the
model already emitted - never a key, a value, or an action - so those requests
execute.

The second is the silence around what is left. A block that is still unusable
after repair is a protocol failure, and reporting it as ``finished`` hides it
exactly the way reporting a discarded multi-block reply as finished once did.
The loop now stops with ``protocol_error`` so the failure is visible.

Phase A also stops promising a reach the system does not have: with no search
provider wired, ``research`` fails closed every time, so it is left out of the
capability instruction instead of advertising a request guaranteed to fail.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.aurora import (
    _repair_json_object,
    extract_prime_request,
)
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.configuration import (
    DEFAULT_SYSTEM_INSTRUCTION,
    AuroraConfig,
    build_capability_instruction,
)
from PRIMORA.prime import Prime
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.mind import Mind
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.providers import Provider
from PRIMORA.research.provider import ResearchProvider

_SEQ = [0]
_TMP = tempfile.TemporaryDirectory(prefix="phase_a_")


def _db(name: str | None = None) -> str:
    _SEQ[0] += 1
    return os.path.join(_TMP.name, f"{name or 'pa'}-{_SEQ[0]}.db")


def _counter(messages):
    return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _config() -> AuroraConfig:
    return AuroraConfig(
        model_path=os.path.join("RESOURCES", "aurora", "v0.1", "model"),
        adapter_path=os.path.join("RESOURCES", "aurora", "v0.1",
                                  "aurora_qlora"),
    )


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.success({"result": "ok"})


class _ResearchProvider(ResearchProvider):
    """A real provider object, so the specialist counts as configured."""

    def search(self, query, limit=5):
        return []


class ScriptedBrain(Brain):
    def __init__(self, *replies: str):
        self._replies = list(replies) or ["plain answer"]
        self.calls: list[list[dict]] = []

    def generate(self, messages, max_output_tokens=None):
        self.calls.append([dict(m) for m in messages])
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        reply = self._replies[index]
        if isinstance(reply, BrainResult):
            return reply
        return BrainResult.success(reply, "scripted")

    def get_model_name(self):
        return "scripted"

    def count_tokens(self, messages):
        return _counter(messages)


def _aurora(*replies: str) -> Aurora:
    prime = Prime(_Provider())
    manager = ContextManager(db_path=_db())
    prime._context = manager
    prime._mind = Mind(manager)
    for name in ("recall", "learn"):
        specialist = prime.registry.get_specialist(name)
        if specialist is not None:
            specialist.bind(prime._mind)
    return Aurora(prime, brain=ScriptedBrain(*replies), config=_config(),
                  context_manager=manager)


def _wrap(body: str) -> str:
    return f"<prime_request>{body}</prime_request>"


# ======================================================================
# A1. Repair: structure the model already emitted is restored
# ======================================================================


class TestRepairRestoresStructure:
    def test_trailing_comma_is_repaired(self):
        body = '{"action":"recall","payload":{"query":"berth"},}'
        request = extract_prime_request(_wrap(body))
        assert request is not None
        assert request.action == "recall"
        assert request.payload == {"query": "berth"}

    def test_missing_outer_closer_is_repaired(self):
        body = '{"action":"recall","payload":{"query":"berth"}}'
        request = extract_prime_request(_wrap(body))
        assert request is not None
        assert request.action == "recall"

    def test_missing_inner_and_outer_closers_are_repaired(self):
        body = '{"action":"recall","payload":{"query":"berth"'
        request = extract_prime_request(_wrap(body))
        assert request is not None
        assert request.payload == {"query": "berth"}

    def test_unterminated_string_is_repaired(self):
        body = '{"action":"recall","payload":{"query":"berth'
        request = extract_prime_request(_wrap(body))
        assert request is not None
        assert request.payload == {"query": "berth"}

    def test_raw_newline_inside_a_string_is_tolerated(self):
        body = '{"action":"recall","payload":{"query":"line\nbreak"}}'
        request = extract_prime_request(_wrap(body))
        assert request is not None

    def test_trailing_comma_before_an_array_closer(self):
        body = '{"action":"recall","payload":{"query":"a","tags":["x",]}}'
        request = extract_prime_request(_wrap(body))
        assert request is not None
        assert request.payload["tags"] == ["x"]


class TestRepairStaysConservative:
    """A repair that invents content would be worse than no repair."""

    def test_structurally_broken_input_still_yields_none(self):
        assert extract_prime_request(_wrap("{broken")) is None

    def test_prose_inside_the_block_still_yields_none(self):
        assert extract_prime_request(_wrap("{not json}")) is None

    def test_a_comma_closer_inside_a_string_is_left_alone(self):
        body = '{"action":"recall","payload":{"query":"a,b}"}}'
        request = extract_prime_request(_wrap(body))
        assert request is not None
        assert request.payload["query"] == "a,b}"

    def test_a_repair_never_invents_an_action(self):
        assert extract_prime_request(_wrap('{"payload":{"query":"x"}}')) is None

    def test_a_repair_never_invents_a_payload(self):
        assert extract_prime_request(_wrap('{"action":"recall"}')) is None

    def test_a_declined_request_is_not_repaired_into_one(self):
        assert extract_prime_request(_wrap("null")) is None
        assert extract_prime_request(_wrap("None")) is None

    def test_the_repairer_returns_none_when_nothing_is_recoverable(self):
        assert _repair_json_object("{broken") is not None  # closers only
        assert _repair_json_object("") is None
        assert _repair_json_object("   ") is None
        assert _repair_json_object("None") is None


# ======================================================================
# A1. The loop stops visibly instead of finishing silently
# ======================================================================


class TestUnrepairableBlockIsVisible:
    def test_an_unrepairable_block_stops_with_protocol_error(self):
        aurora = _aurora(_wrap("{broken"))
        result = aurora.respond([{"role": "user", "content": "go"}], "s")
        assert result.success
        assert aurora.loop.last_trace.stop_reason == "protocol_error"

    def test_nothing_ran(self):
        aurora = _aurora(_wrap("{broken"))
        aurora.respond([{"role": "user", "content": "go"}], "s")
        assert aurora.loop.executor.calls == []

    def test_a_plain_answer_still_finishes(self):
        aurora = _aurora("just an answer")
        aurora.respond([{"role": "user", "content": "go"}], "s")
        assert aurora.loop.last_trace.stop_reason == "finished"

    def test_a_repaired_block_finishes_by_executing(self):
        aurora = _aurora(_wrap('{"action":"recall","payload":{"query":"x"},}'))
        aurora.respond([{"role": "user", "content": "go"}], "s")
        assert aurora.loop.last_trace.stop_reason != "protocol_error"
        assert len(aurora.loop.executor.calls) >= 1, (
            "a repairable block must reach the executor")


# ======================================================================
# A2. The few-shot example is a capability the system really has
# ======================================================================


class TestTheExampleIsARealCapability:
    def test_the_example_is_research(self):
        assert '"action":"research"' in DEFAULT_SYSTEM_INSTRUCTION

    def test_the_example_is_not_a_tool_wrapper(self):
        assert "count_words" not in DEFAULT_SYSTEM_INSTRUCTION

    def test_the_instruction_says_exactly_one(self):
        assert "exactly ONE" in DEFAULT_SYSTEM_INSTRUCTION


# ======================================================================
# A3. Nothing is advertised that cannot run
# ======================================================================


class TestCapabilitiesAdvertisedMatchReality:
    def test_research_is_omitted_when_no_provider_is_wired(self):
        block = build_capability_instruction(Prime(_Provider()))
        assert "- research:" not in block

    def test_research_is_advertised_when_a_provider_is_wired(self):
        prime = Prime(_Provider(), research_provider=_ResearchProvider())
        block = build_capability_instruction(prime)
        assert "- research:" in block

    def test_capabilities_that_always_work_are_unchanged(self):
        block = build_capability_instruction(Prime(_Provider()))
        assert "- recall:" in block

    def test_use_tool_is_omitted_when_registry_is_empty(self):
        block = build_capability_instruction(Prime(_Provider()))
        assert "- use_tool:" not in block

    def test_use_tool_is_advertised_when_a_tool_is_registered(self):
        from PRIMORA.toolbox import ToolRegistry, ToolSpec, ToolResult

        def _handler(arguments):
            return ToolResult.success(arguments)

        prime = Prime(_Provider())
        prime.registry.get_specialist("use_tool").set_tools(ToolRegistry([
            ToolSpec(name="echo", description="Echo.",
                     input_schema={"type": "object"}, handler=_handler),
        ]))
        block = build_capability_instruction(prime)
        assert "- use_tool:" in block

    def test_consult_specialist_is_omitted_when_catalogue_is_empty(self):
        block = build_capability_instruction(Prime(_Provider()))
        assert "- consult_specialist:" not in block

    def test_consult_specialist_is_advertised_when_a_specialist_exists(self):
        from PRIMORA.consult import SpecialistRegistry, SpecialistSpec
        prime = Prime(_Provider())
        prime.registry.get_specialist("consult_specialist").set_specialists(
            SpecialistRegistry([SpecialistSpec(
                name="reasoning", description="Step by step.")]))
        block = build_capability_instruction(prime)
        assert "- consult_specialist:" in block

    def test_learn_is_omitted_when_no_jev_is_wired(self):
        block = build_capability_instruction(Prime(_Provider()))
        assert "- learn:" not in block

    def test_learn_is_advertised_when_a_jev_is_wired(self):
        from PRIMORA.learning import Jev, LearningOutcome

        class _Jev(Jev):
            name = "adv-jev"

            def evaluate(self, request):
                return LearningOutcome.rejected("noop")

        prime = Prime(_Provider(), jev=_Jev())
        block = build_capability_instruction(prime)
        assert "- learn:" in block

    def test_the_opt_in_cannot_take_away_a_working_capability(self):
        """A specialist that reports no state is always advertised."""
        prime = Prime(_Provider(), research_provider=_ResearchProvider())
        assert prime.registry.get_specialist("research").is_configured is True
        assert (prime.registry.get_specialist("research")
                .is_configured is not False)


# ======================================================================
# The budget still has room for a conversation
# ======================================================================


class TestTheBudgetStillClearsTheFloor:
    def test_headroom_is_at_least_four_hundred_tokens(self):
        from transformers import AutoTokenizer
        from PRIMORA.aurora.configuration import (
            build_specialist_instruction,
            build_tool_instruction,
        )
        from PRIMORA.aurora.identity import build_identity_instruction

        prime = Prime(_Provider(), research_provider=_ResearchProvider())
        prompt = (
            DEFAULT_SYSTEM_INSTRUCTION
            + build_identity_instruction(_config())
            + build_capability_instruction(prime)
            + build_specialist_instruction(prime)
            + build_tool_instruction(prime)
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        tokens = len(tokenizer(prompt)["input_ids"])
        headroom = _config().context_ceiling - tokens
        assert headroom >= 400, (
            f"only {headroom} tokens left for the conversation and any "
            f"tool result, on a prompt of {tokens}"
        )
