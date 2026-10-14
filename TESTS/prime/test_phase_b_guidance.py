"""Phase B: scaffolding the base family that under-issues request blocks.

The v0.4 defect is not only that it answers in prose. It is that v0.1 and
v0.3, on the same base family and the same instruction, do emit a request
block where v0.4 does not. That makes the difference partly a property of
the base, not only of the tuning: a Qwen-family base under-issues tool calls
relative to what the instruction asks for, so the instruction alone cannot
be the whole fix.

Phase B adds a guidance block keyed to the model's declared architecture
family. It is read from the model's own ``config.json``, not guessed from a
directory name, so it is keyed to what the model actually is. Only the
measured family gets it; any other family, or a model whose configuration
cannot be read, gets nothing rather than scaffolding that may not apply.

The block is placed last, after the catalogues, because it is a behavioural
nudge and the nearest instruction to generation is the one that has the most
effect on whether a block is emitted at all.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.configuration import (
    GUIDANCE_HEADER,
    QWEN_GUIDANCE,
    AuroraConfig,
    build_guidance_instruction,
    detect_base_family,
)
from PRIMORA.prime import Prime
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.providers import Provider

_SEQ = [0]
_TMP = tempfile.TemporaryDirectory(prefix="phase_b_")

REAL_MODEL = os.path.join("RESOURCES", "aurora", "v0.1", "model")


def _db(name: str | None = None) -> str:
    _SEQ[0] += 1
    return os.path.join(_TMP.name, f"{name or 'pb'}-{_SEQ[0]}.db")


def _model_dir(model_type: str | None) -> str:
    """A directory carrying a config.json declaring ``model_type``."""
    _SEQ[0] += 1
    path = os.path.join(_TMP.name, f"model-{_SEQ[0]}")
    os.makedirs(path)
    if model_type is not None:
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fh:
            json.dump({"model_type": model_type}, fh)
    return path


def _config(model_path: str) -> AuroraConfig:
    return AuroraConfig(model_path=model_path,
                        adapter_path=os.path.join("RESOURCES", "aurora",
                                                  "v0.1", "aurora_qlora"))


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.success({"result": "ok"})


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
        return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _aurora(model_path: str, brain: ScriptedBrain | None = None) -> Aurora:
    prime = Prime(_Provider())
    manager = ContextManager(db_path=_db())
    prime._context = manager
    return Aurora(prime, brain=brain or ScriptedBrain(),
                  config=_config(model_path), context_manager=manager)


def _system_message(aurora: Aurora) -> str:
    seen = aurora._with_instruction([{"role": "user", "content": "hi"}])
    return seen[0]["content"]


# ======================================================================
# Detection reads the model, not the path
# ======================================================================


class TestTheFamilyIsReadFromTheModel:
    def test_the_configured_model_is_declared_qwen(self):
        assert detect_base_family(REAL_MODEL) == "qwen3"

    def test_a_qwen25_model_is_recognised(self):
        assert detect_base_family(_model_dir("qwen2.5")) == "qwen2.5"

    def test_a_llama_model_is_not_qwen(self):
        assert detect_base_family(_model_dir("llama")) == "llama"

    def test_a_mistral_model_is_not_qwen(self):
        assert detect_base_family(_model_dir("mistral")) == "mistral"

    def test_a_missing_config_file_reports_nothing(self):
        assert detect_base_family(_model_dir(None)) is None

    def test_a_missing_directory_reports_nothing(self):
        assert detect_base_family(os.path.join(_TMP.name, "absent")) is None

    def test_an_empty_path_reports_nothing(self):
        assert detect_base_family("") is None
        assert detect_base_family(None) is None

    def test_an_unreadable_config_reports_nothing(self):
        path = _model_dir(None)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fh:
            fh.write("{ not json")
        assert detect_base_family(path) is None

    def test_a_config_without_model_type_reports_nothing(self):
        _SEQ[0] += 1
        path = os.path.join(_TMP.name, f"model-{_SEQ[0]}")
        os.makedirs(path)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fh:
            json.dump({"architectures": ["X"]}, fh)
        assert detect_base_family(path) is None


# ======================================================================
# The guidance is emitted only for the family it was measured on
# ======================================================================


class TestTheGuidanceIsKeyedToTheFamily:
    def test_a_qwen_base_gets_the_guidance(self):
        assert build_guidance_instruction(_config(REAL_MODEL)) != ""

    def test_a_non_qwen_base_gets_nothing(self):
        assert build_guidance_instruction(_config(_model_dir("llama"))) == ""

    def test_an_unreadable_base_gets_nothing(self):
        assert build_guidance_instruction(_config(_model_dir(None))) == ""

    def test_the_guidance_carries_its_header(self):
        assert GUIDANCE_HEADER in build_guidance_instruction(_config(REAL_MODEL))

    def test_the_guidance_says_the_thing_the_instruction_does_not(self):
        text = build_guidance_instruction(_config(REAL_MODEL))
        assert "BEFORE YOU ANSWER" in text
        assert "request block first" in text

    def test_the_guidance_is_not_a_copy_of_the_instruction(self):
        from PRIMORA.aurora.configuration import DEFAULT_SYSTEM_INSTRUCTION

        text = build_guidance_instruction(_config(REAL_MODEL))
        assert "ACT, DO NOT ANNOUNCE" not in text
        assert "<prime_request>" not in text
        assert text.strip() != DEFAULT_SYSTEM_INSTRUCTION.strip()


# ======================================================================
# It reaches the prompt the runtime assembles, exactly once
# ======================================================================


class TestTheGuidanceReachesThePrompt:
    def test_a_qwen_run_is_guided(self):
        assert GUIDANCE_HEADER in _system_message(_aurora(REAL_MODEL))

    def test_a_non_qwen_run_is_not_guided(self):
        message = _system_message(_aurora(_model_dir("llama")))
        assert GUIDANCE_HEADER not in message
        assert "BEFORE YOU ANSWER" not in message

    def test_the_guidance_is_the_last_block(self):
        """Nearest to generation, which is where a nudge has most effect."""
        message = _system_message(_aurora(REAL_MODEL))
        assert GUIDANCE_HEADER in message
        assert message.rindex(GUIDANCE_HEADER) > message.index("Available")

    def test_an_echoed_prompt_does_not_get_a_second_copy(self):
        aurora = _aurora(REAL_MODEL)
        already = "system text\n" + GUIDANCE_HEADER + "\n" + QWEN_GUIDANCE
        seen = aurora._with_instruction(
            [{"role": "system", "content": already},
             {"role": "user", "content": "hi"}])
        assert seen[0]["content"].count(GUIDANCE_HEADER) == 1

    def test_a_qwen_run_still_carries_the_identity_and_catalogues(self):
        from PRIMORA.aurora.identity import IDENTITY_HEADER

        message = _system_message(_aurora(REAL_MODEL))
        assert IDENTITY_HEADER in message
        assert "Available Prime Capabilities:" in message
        assert GUIDANCE_HEADER in message


# ======================================================================
# The budget still has room for a conversation
# ======================================================================


class TestTheBudgetStillClearsTheFloor:
    def test_the_measured_prompt_still_clears_the_floor(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "tc", os.path.join("TESTS", "prime", "test_context_budget_fit.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        tokenizer = module._tokenizer()
        config = AuroraConfig.from_env()
        prompt = module._full_system_prompt()
        # The helper must actually be measuring the guidance, or this
        # floor is not guarding it.
        assert GUIDANCE_HEADER in prompt
        tokens = len(tokenizer(prompt)["input_ids"])
        headroom = config.context_ceiling - tokens
        assert headroom >= 400, (
            f"only {headroom} tokens left for the conversation and any "
            f"retrieved memory, on a prompt of {tokens}"
        )

    def test_the_real_runtime_prompt_fits_the_ceiling(self):
        """Instruction, identity, catalogues and guidance, as assembled."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "tc", os.path.join("TESTS", "prime", "test_context_budget_fit.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        prime = module._fully_loaded_prime()
        config = AuroraConfig.from_env()
        from PRIMORA.aurora.configuration import (
            build_capability_instruction,
            build_specialist_instruction,
            build_tool_instruction,
        )
        from PRIMORA.aurora.identity import build_identity_instruction

        prompt = (
            config.system_instruction
            + build_identity_instruction(config)
            + build_capability_instruction(prime)
            + build_specialist_instruction(prime)
            + build_tool_instruction(prime)
            + build_guidance_instruction(config)
        )
        assert GUIDANCE_HEADER in prompt
        tokens = len(module._tokenizer()(prompt)["input_ids"])
        assert tokens < config.context_ceiling, (
            f"the runtime prompt is {tokens} tokens, over the "
            f"{config.context_ceiling} ceiling"
        )
