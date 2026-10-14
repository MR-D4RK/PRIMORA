"""The context budget must fit the instruction Aurora actually receives.

This is a regression test for a real failure found only by running the real
model: the rendered system prompt grew past the configured input ceiling as
capabilities were added, and every single turn failed with
``context_too_large`` before Aurora generated a character. 1143 tests did not
catch it, because their token counter is a word-count stub that never
overflows.

So this test counts the real thing, with the real tokenizer, when the model is
available. It is the check that would have caught it.
"""

import os
import sys

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora.configuration import (
    AuroraConfig,
    build_capability_instruction,
    build_specialist_instruction,
    build_tool_instruction,
)
from PRIMORA.consult import SpecialistRegistry, SpecialistSpec
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.providers import Provider
from PRIMORA.toolbox import ToolRegistry


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.success({"value": 4})


def _fully_loaded_prime():
    """Prime with every catalogue populated: the worst realistic prompt."""
    catalogue = SpecialistRegistry([
        SpecialistSpec(
            name="reasoning",
            description=(
                "Careful step-by-step analysis of a hard problem. Use for "
                "difficult reasoning, trade-offs, and checking your own work."
            ),
            model="m",
            system_prompt="Think carefully and precisely, then answer.",
        )
    ])
    return Prime(_Provider(), specialists=catalogue,
                 tools=ToolRegistry.with_builtins())


def _full_system_prompt(include_identity: bool = False) -> str:
    prime = _fully_loaded_prime()
    config = AuroraConfig.from_env()
    parts = [config.system_instruction]
    if include_identity:
        # The runtime appends this to every prompt it assembles; it is part
        # of what Aurora receives, so it must be measured like everything
        # else. Off by default so the original instruction-and-catalogue
        # baseline keeps its own thresholds.
        from PRIMORA.aurora.identity import build_identity_instruction

        parts.append(build_identity_instruction(config))
    parts += [
        build_capability_instruction(prime),
        build_specialist_instruction(prime),
        build_tool_instruction(prime),
    ]
    # The base-family guidance is part of what Aurora receives whenever the
    # base is Qwen-family, which every configured model here is, so it has
    # to be counted here too or the floor would not catch it growing.
    from PRIMORA.aurora.configuration import build_guidance_instruction

    parts.append(build_guidance_instruction(config))
    return "".join(parts)


MODEL = os.path.join("RESOURCES", "aurora", "v0.1", "model")


def _tokenizer():
    if not os.path.isdir(MODEL):
        pytest.skip(f"no local model at {MODEL}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


class TestSystemPromptFitsTheBudget:
    def test_the_real_system_prompt_fits_the_default_ceiling(self):
        """The failure this exists to catch, measured with the real tokenizer."""
        tokenizer = _tokenizer()
        config = AuroraConfig.from_env()
        prompt_tokens = len(tokenizer(_full_system_prompt())["input_ids"])
        assert prompt_tokens < config.context_ceiling, (
            f"the system prompt is {prompt_tokens} tokens but the input "
            f"ceiling is {config.context_ceiling}; every turn would fail with "
            f"context_too_large"
        )

    def test_there_is_room_left_for_a_conversation(self):
        """A prompt that exactly fills the ceiling is still unusable."""
        tokenizer = _tokenizer()
        config = AuroraConfig.from_env()
        prompt_tokens = len(tokenizer(_full_system_prompt())["input_ids"])
        headroom = config.context_ceiling - prompt_tokens
        assert headroom >= 400, (
            f"only {headroom} tokens left for the conversation and any "
            f"retrieved memory"
        )

    def test_a_typical_turn_fits(self):
        tokenizer = _tokenizer()
        config = AuroraConfig.from_env()
        budget = len(tokenizer(_full_system_prompt())["input_ids"]) + 15
        assert budget < config.context_total

    def test_the_ceiling_stays_far_below_the_model_window(self):
        config = AuroraConfig.from_env()
        assert config.context_total <= 40960, "leave the model window intact"
        assert config.context_total < 40960 * 0.1, (
            "the budget should stay a small fraction of the model's window"
        )


    def test_the_prompt_with_the_identity_block_fits_the_ceiling(self):
        """The identity block rides on every assembled prompt; count it, for real.

        The ceiling test above measures the instruction-and-catalogue
        baseline. This one measures what the runtime actually assembles once
        the config-derived identity is included, so an oversized identity can
        never push a turn past the limit that bricked one before.
        """
        tokenizer = _tokenizer()
        config = AuroraConfig.from_env()
        prompt_tokens = len(tokenizer(
            _full_system_prompt(include_identity=True))["input_ids"])
        assert prompt_tokens < config.context_ceiling, (
            f"the prompt with identity is {prompt_tokens} tokens but the input "
            f"ceiling is {config.context_ceiling}; every turn would fail with "
            f"context_too_large"
        )

    def test_the_identity_block_stays_bounded(self):
        """Identity costs conversation room, so its price is capped.

        The history fill stops at context_target and the baseline prompt
        already sits close under it, so every token the identity block spends
        is a token of history it displaces. The cap keeps a later rewording
        from quietly eating the conversation budget.
        """
        from PRIMORA.aurora.identity import build_identity_instruction

        tokenizer = _tokenizer()
        config = AuroraConfig.from_env()
        identity_tokens = len(tokenizer(
            build_identity_instruction(config))["input_ids"])
        assert identity_tokens <= 200, (
            f"the identity block alone costs {identity_tokens} tokens"
        )


class TestBudgetIsInternallyConsistent:
    def test_total_is_ceiling_plus_reserve(self):
        report = AuroraConfig.from_env().context_budget_report()
        assert report["consistent_total"], report

    def test_reserve_can_hold_a_default_answer(self):
        config = AuroraConfig.from_env()
        assert config.context_output_reserve >= config.max_new_tokens, (
            "the output reserve must fit a full-length answer"
        )

    def test_the_brain_ceiling_equals_the_budget(self):
        """They must not disagree.

        A brain ceiling below context_total silently trims a context the
        mechanism believed would fit. A brain ceiling above it lets the two
        ask for more than the card can hold, which is how a long turn turned
        into "CUDA out of memory, 0 bytes free" instead of a slow answer.
        """
        config = AuroraConfig.from_env()
        from PRIMORA.aurora.brain import AuroraBrain

        brain = AuroraBrain(config)
        assert brain._max_context_tokens == config.context_total, (
            f"brain ceiling {brain._max_context_tokens} != context_total "
            f"{config.context_total}"
        )

    def test_no_default_can_request_more_than_the_measured_safe_input(self):
        """1800 is the measured boundary; nothing may exceed it by default."""
        config = AuroraConfig.from_env()
        from PRIMORA.aurora.brain import AuroraBrain

        brain = AuroraBrain(config)
        assert config.context_ceiling <= 1800, config.context_ceiling
        assert config.context_target <= config.context_ceiling
        assert brain._max_context_tokens <= 1800 + config.context_output_reserve

    def test_the_ceiling_is_the_documented_budget(self):
        """1600 is the documented ceiling after the prompt compression.

        The measured OOM boundary on this card is ~1824 input tokens; the
        old 1800 ceiling sat a whisper under it and left no margin for a
        long turn. The system prompt lost ~230 tokens, so the target keeps
        the same conversation room as the old 1600 while the ceiling drops
        200 tokens of worst-case prefill and KV cache.
        """
        config = AuroraConfig.from_env()
        assert config.context_ceiling == 1600
        assert config.context_target == 1500
        assert config.context_output_reserve == 256
        assert config.context_total == 1856
        assert config.context_state_reserve == 100

    def test_target_is_below_ceiling(self):
        config = AuroraConfig.from_env()
        assert config.context_target < config.context_ceiling

    def test_target_still_fits_the_system_prompt(self):
        """The target must leave room for the instruction, not just be small."""
        config = AuroraConfig.from_env()
        tokenizer = _tokenizer()
        prompt = len(tokenizer(_full_system_prompt())["input_ids"])
        assert prompt < config.context_target, (
            f"the system prompt is {prompt} tokens and the target is "
            f"{config.context_target}; no room would be left for the turn"
        )

    def test_every_part_stays_configurable(self):
        config = AuroraConfig(
            model_path="m", adapter_path="a", context_target=10,
            context_ceiling=20, context_output_reserve=5, context_total=25,
        )
        assert config.context_budget_report()["consistent_total"]

    def test_the_allocator_setting_is_applied_before_loading(self):
        """A weak, source-level check, and honestly so.

        The setting is only readable by PyTorch once the CUDA allocator
        initialises, so proving it behaviourally would mean loading the model
        - a GPU test the main suite must not depend on. What this can prove
        cheaply is that the setting is applied where it is needed (inside
        _initialize, before the load) and that an operator can still override
        it. The behavioural half is covered by the real-model probe.
        """
        import PRIMORA.aurora.brain as brainmod

        source = open(brainmod.__file__, encoding="utf-8").read()
        assert "expandable_segments:True" in source
        assert "setdefault" in source, "an operator override must win"
        initialize = source[source.index("def _initialize"):]
        load = initialize.index("from_pretrained")
        assert initialize.index("expandable_segments") < load, (
            "the allocator setting must be applied before the model loads"
        )


class TestTheInstructionIsPresent:
    def test_the_prompt_carries_the_act_dont_announce_rule(self):
        text = _full_system_prompt()
        assert "ACT, DO NOT ANNOUNCE" in text
        for marker in ("I'll research", "I'll ask the reasoning specialist",
                       "I'll retrieve that from memory"):
            assert marker in text, marker

    def test_the_prompt_still_allows_answering_directly(self):
        """It must not push her toward a capability every time."""
        text = _full_system_prompt()
        assert "If no technical execution is required, answer normally" in text
        assert "most questions need nothing at all" in text

    def test_the_prompt_still_forbids_fake_claiming(self):
        text = _full_system_prompt()
        assert "never claim that you did" in text
        assert "Never emit more than one <prime_request>" in text


class TestTheCompressionKeptEveryLoadBearingLine:
    """The prompt was cut from ~1141 tokens to ~905.

    Almost every cut removed something an earlier test or an earlier defect
    had pinned as load-bearing. This class is the receipt: it asserts the
    lines that survived the cut are still there, so a later trimming pass
    cannot quietly remove the ones that matter while still passing the
    behavioural tests that only fire on a live model.
    """

    def test_the_size_actually_dropped(self):
        """Measured, not estimated: ~1524 tokens before, must stay under 1350.

        The pre-fix figure is the real tokenizer count of the whole assembled
        system prompt - instruction, identity, capability and tool catalogues
        and guidance - taken during the diagnosis. This is the guard against
        the prompt drifting back to that size one edit at a time.
        """
        tokenizer = _tokenizer()
        tokens = len(tokenizer(_full_system_prompt(
            include_identity=True))["input_ids"])
        assert tokens < 1350, (
            f"the assembled system prompt is back to {tokens} tokens; the "
            f"pre-fix measurement was ~1524"
        )

    def test_the_research_read_capability_is_named(self):
        """It was merged into the action line when the list was cut."""
        text = _full_system_prompt()
        assert "research_read" in text

    def test_a_failed_research_may_not_be_reported_as_a_success(self):
        """The fabrication defect needs the failure wording, not just the ban."""
        text = _full_system_prompt()
        assert "A result carrying a failure or an error did not happen" in text
        assert "never claim that you did" in text

    def test_a_stored_fact_is_offered_as_memory_not_as_ground_truth(self):
        """The [state ...] line must be framed: it is memory, not evidence."""
        text = _full_system_prompt()
        assert "[state ...] line is your own stored memory" in text

    def test_the_prompt_never_asks_for_a_word_count(self):
        """The instruction may not demand a tool the loop does not offer."""
        config = AuroraConfig.from_env()
        assert "count_words" not in config.system_instruction

    def test_the_state_reserve_is_smaller_than_the_target(self):
        """A reserve larger than the target would evict the whole turn."""
        config = AuroraConfig.from_env()
        assert 0 < config.context_state_reserve < config.context_target
