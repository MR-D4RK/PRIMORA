"""Identity tests: the runtime prompt must carry a config-derived identity.

The failure these guard against is behavioural, not cosmetic: Aurora used to
deny her own project and version because neither the prompt nor the training
data ever stated them, and because the version that matters is whichever one
this run's configuration resolved - not a string anyone could safely hardcode.

None of this loads model weights: the identity lives in the system prompt, so
a scripted brain and a temporary store are the whole runtime needed to observe
it.
"""

import os
import re
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.configuration import (
    AuroraConfig,
    DEFAULT_SYSTEM_INSTRUCTION,
    build_capability_instruction,
    build_specialist_instruction,
    build_tool_instruction,
)
from PRIMORA.aurora.identity import (
    IDENTITY_HEADER,
    build_identity_instruction,
    resolve_identity,
)
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.providers import Provider
from typing import Any


class MockProvider(Provider):
    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        return PrimeResult.success({"result": "ok"})


class MockScriptedBrain(Brain):
    def __init__(self, outputs: list, model: str = "identity-test-model"):
        self._outputs = list(outputs)
        self._model = model
        self.seen_messages = []

    def generate(self, messages: list[dict[str, str]], max_output_tokens=None) -> BrainResult:
        self.seen_messages.append(list(messages))
        return self._outputs.pop(0)

    def get_model_name(self) -> str:
        return self._model


def _test_config(**kwargs):
    params = dict(
        model_path=os.path.join("C:\\fake", "aurora", "v0.1", "model"),
        adapter_path=os.path.join("C:\\fake", "aurora", "v0.4", "aurora_qlora"),
    )
    params.update(kwargs)
    return AuroraConfig(**params)


_TEST_TMPDIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def _test_context():
    from PRIMORA.prime.context import ContextManager
    fd, path = tempfile.mkstemp(suffix=".db", dir=_TEST_TMPDIR.name)
    os.close(fd)
    return ContextManager(db_path=path)


def _runtime_system(messages, config=None) -> str:
    """The system prompt this runtime actually assembles for a turn."""
    provider = MockProvider()
    brain = MockScriptedBrain([BrainResult.success("Hello!", "m")])
    aurora = Aurora(
        Prime(provider),
        brain=brain,
        config=config or _test_config(),
        context_manager=_test_context(),
    )
    aurora.respond(messages)
    seen = brain.seen_messages[0]
    systems = [m for m in seen if m["role"] == "system"]
    assert len(systems) == 1, "exactly one system message must reach the brain"
    return systems[0]["content"]


def test_identity_block_is_in_the_runtime_system_prompt():
    prompt = _runtime_system([{"role": "user", "content": "Hi"}])
    assert prompt.startswith(DEFAULT_SYSTEM_INSTRUCTION)
    assert prompt.count(IDENTITY_HEADER) == 1
    assert "You are Aurora, the conversational identity in PRIMORA" in prompt
    assert "PRIMORA is the project this runtime belongs to" in prompt
    assert "Prime is the orchestration and tool-execution layer" in prompt


def test_identity_is_present_when_the_caller_supplies_a_system_message():
    prompt = _runtime_system([
        {"role": "system", "content": "Caller system."},
        {"role": "user", "content": "Hi"},
    ])
    assert prompt.startswith("Caller system.")
    assert prompt.count(IDENTITY_HEADER) == 1
    assert "the conversational identity in PRIMORA" in prompt


def test_identity_is_not_duplicated_when_the_prompt_already_carries_it():
    # A caller that echoes a system message already carrying the block gets
    # no second copy. (Only the identity block is under test here: the
    # capability catalogue is appended to any caller system message, before
    # and after this change, and duplicating that would be a different bug.)
    identity = build_identity_instruction(_test_config())
    provider = MockProvider()
    brain = MockScriptedBrain([BrainResult.success("Hello!", "m")])
    aurora = Aurora(
        Prime(provider),
        brain=brain,
        config=_test_config(),
        context_manager=_test_context(),
    )
    aurora.respond([
        {"role": "system", "content": "Caller system." + identity},
        {"role": "user", "content": "Hi"},
    ])
    system = brain.seen_messages[0][0]["content"]
    assert system.count(IDENTITY_HEADER) == 1


def test_adapter_identity_follows_the_configured_path():
    newer = build_identity_instruction(_test_config(
        adapter_path=os.path.join("C:\\fake", "aurora", "v0.4", "aurora_qlora")))
    older = build_identity_instruction(_test_config(
        adapter_path=os.path.join("C:\\fake", "aurora", "v0.3", "aurora_qlora")))

    assert "version v0.4" in newer
    assert "v0.3" not in newer
    assert "version v0.3" in older
    assert "v0.4" not in older

    facts = resolve_identity(_test_config(
        adapter_path=os.path.join("C:\\fake", "aurora", "v0.4", "aurora_qlora")))
    assert facts["adapter"]["configured"] is True
    assert facts["adapter"]["version"] == "v0.4"
    assert facts["model"]["version"] == "v0.1"


def test_no_version_is_hardcoded_in_the_identity_module():
    import PRIMORA.aurora.identity as identity_module

    source = open(identity_module.__file__, encoding="utf-8").read()
    assert re.search(r"v\d+\.\d+", source) is None, (
        "the identity must be derived from configuration, never baked in"
    )


def test_base_model_only_run_is_described_honestly():
    config = _test_config(adapter_path="")
    text = build_identity_instruction(config)

    assert resolve_identity(config)["adapter"]["configured"] is False
    assert resolve_identity(config)["adapter"]["version"] is None
    assert "no adapter is configured for this run" in text
    assert "you run on the base model alone" in text
    assert "aurora_qlora" not in text, "no adapter may be described when none is set"


def test_paths_missing_from_disk_are_reported_not_hidden():
    absent = os.path.join(
        tempfile.gettempdir(), "primora_no_such_runtime", "v0.9", "aurora_qlora")
    config = _test_config(
        model_path=os.path.join(tempfile.gettempdir(), "primora_no_such_model"),
        adapter_path=absent,
    )
    text = build_identity_instruction(config)

    assert "version v0.9" in text, "the configured version still comes from config"
    assert "whose path is not present on disk" in text
    assert "its path is not present on disk" in text

    root = _TEST_TMPDIR.name
    present = _test_config(
        model_path=os.path.join(root, "v0.1", "model"),
        adapter_path=os.path.join(root, "v0.4", "aurora_qlora"),
    )
    os.makedirs(present.model_path, exist_ok=True)
    os.makedirs(present.adapter_path, exist_ok=True)
    assert "not present on disk" not in build_identity_instruction(present)


def test_identity_coexists_with_the_grounding_rules():
    prompt = _runtime_system([{"role": "user", "content": "Who are you?"}])

    # The original honesty rules survive untouched, and the identity speaks
    # the same language: facts to state, claims to attribute, nothing to
    # invent.
    assert "Never invent what someone said or agreed to" in prompt
    assert "never deny your identity, PRIMORA" in prompt
    assert "no statement is automatically true" in prompt
    assert "unsupported claims" in prompt
    assert "memories you do not hold" in prompt
    assert "Answer identity questions directly, in the first person" in prompt

    grandiose = ("most advanced", "perfect", "sentient", "superintelligence",
                 "always right", "flawless", "omniscient")
    identity_text = build_identity_instruction(_test_config())
    lowered = identity_text.lower()
    for phrase in grandiose:
        assert phrase not in lowered, phrase


def test_identity_responds_to_the_config_that_was_built():
    """A config that differs from the default must produce a different block."""
    custom = build_identity_instruction(_test_config(
        model_path=os.path.join("C:\\deploy", "base", "v0.3", "model"),
        adapter_path="",
    ))
    default = build_identity_instruction(_test_config())
    assert custom != default
    assert "v0.3" in custom
    assert "no adapter is configured for this run" in custom


def test_identity_overrides_earlier_denial_replies():
    """Retrieval injects this session's own stored replies beside the block.

    Pre-identity turns answered version questions with 'no version recorded';
    the block must say, in the prompt that actually runs, that such replies
    predate the configuration instead of only asking for the facts.
    """
    prompt = _runtime_system(
        [{"role": "user", "content": "what is your brain version ?"}])
    assert ("retrieved history or your own earlier replies say you have none"
            in prompt)
    assert "those predate this configuration" in prompt
    assert prompt.count(IDENTITY_HEADER) == 1


def test_the_runtime_prompt_states_both_configured_versions_as_facts():
    prompt = _runtime_system([{"role": "user", "content": "Hi"}])
    assert "version v0.1" in prompt, "the base model version must reach the brain"
    assert "version v0.4" in prompt, "the adapter version must reach the brain"
    assert "These are configuration facts, not memories" in prompt
    # The versions come from the config-derived block, nowhere else.
    identity = build_identity_instruction(_test_config())
    body = prompt.replace(identity, "")
    assert "version v0.1" not in body and "version v0.4" not in body


def test_actual_production_paths_resolve_both_versions():
    """The real on-disk layout this runtime ships with resolves both markers."""
    root = os.path.join("C:\\PRIMORA PROJECT", "RESOURCES", "aurora")
    config = AuroraConfig(
        model_path=os.path.join(root, "v0.1", "model"),
        adapter_path=os.path.join(root, "v0.4", "aurora_qlora"),
    )
    text = build_identity_instruction(config)
    facts = resolve_identity(config)

    assert facts["model"]["version"] == "v0.1"
    assert facts["adapter"]["version"] == "v0.4"
    assert "version v0.1" in text
    assert "version v0.4" in text
    assert facts["model"]["exists"] and facts["adapter"]["exists"]
    assert "not present on disk" not in text


def test_no_instruction_elsewhere_denies_her_version():
    """Every other instruction the brain receives must stay silent on it.

    The identity block is the only place allowed to speak about versions;
    if the base instruction or a capability description said she has none,
    the prompt would contradict itself the way the live CLI used to.
    """
    prime = Prime(MockProvider())
    pieces = {
        "default system": DEFAULT_SYSTEM_INSTRUCTION,
        "capabilities": build_capability_instruction(prime),
        "specialists": build_specialist_instruction(prime),
        "tools": build_tool_instruction(prime),
    }
    denials = ("no version", "not running a version", "no versioning",
               "have no version", "version recorded")
    for name, text in pieces.items():
        lowered = text.lower()
        for phrase in denials:
            assert phrase not in lowered, (
                f"{name} tells her {phrase!r}; only the identity block may "
                f"describe versions"
            )


if __name__ == "__main__":
    test_identity_block_is_in_the_runtime_system_prompt()
    test_identity_is_present_when_the_caller_supplies_a_system_message()
    test_identity_is_not_duplicated_when_the_prompt_already_carries_it()
    test_adapter_identity_follows_the_configured_path()
    test_no_version_is_hardcoded_in_the_identity_module()
    test_base_model_only_run_is_described_honestly()
    test_paths_missing_from_disk_are_reported_not_hidden()
    test_identity_coexists_with_the_grounding_rules()
    test_identity_responds_to_the_config_that_was_built()
    test_identity_overrides_earlier_denial_replies()
    test_the_runtime_prompt_states_both_configured_versions_as_facts()
    test_actual_production_paths_resolve_both_versions()
    test_no_instruction_elsewhere_denies_her_version()
    print("All identity tests passed!")
