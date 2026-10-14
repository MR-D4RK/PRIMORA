"""Capture what the real Aurora brain actually emits for the failing scenarios.

The three capability behaviours that scored 0/5 share one symptom - a request
that the loop could not execute - but the causes are not known to be the same.
Multi-step emitted two blocks in 4/5 runs and one recall in the fifth.
Specialist and tool emitted the wrong action. This prints the raw generations
so the cause is read off the output instead of assumed.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import AuroraBrain
from PRIMORA.aurora.configuration import (
    AuroraConfig,
    build_capability_instruction,
    build_specialist_instruction,
    build_tool_instruction,
)
from PRIMORA.aurora.aurora import extract_prime_request
from PRIMORA.consult import SpecialistRegistry, SpecialistSpec
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context.isolation import (
    build_prime_with_isolation_guard,
    harness_session,
)
from PRIMORA.prime.providers import Provider
from PRIMORA.research import ResearchProvider, ResearchResult
from PRIMORA.toolbox import ToolRegistry

SCENARIOS = [
    ("multi-step",
     "We berthed at berth nine with the kestrel seventeen cargo. Confirm the "
     "berth code we used, then check the current Python release, and tell me "
     "both."),
    ("specialist",
     "Should a single server use a relational database or a key-value store "
     "for session storage, and why? Consult the reasoning specialist if it "
     "would help."),
    ("tool",
     "How many words are in this note: the quick brown fox jumps over the "
     "lazy dog?"),
]


class _Web(ResearchProvider):
    name = "diag-web"

    def search(self, query, limit=5):
        return [ResearchResult(
            "Python release", "https://example.org/py",
            "Python 3.14 is the latest stable release.", "example.org")]


class _SpecialistProvider(Provider):
    name = "diag-specialist"

    def invoke(self, capability, payload):
        text = (payload.get("messages") or [{}])[-1].get("content", "")
        return PrimeResult.success({"content": "Considered analysis."})


def main() -> int:
    brain = AuroraBrain()
    load_error = brain.warmup()
    if load_error:
        print(f"  MODEL FAILED TO LOAD: {load_error}")
        return 1

    tmp = tempfile.mkdtemp(prefix="capdiag_")
    with harness_session("capdiag", root=tmp) as session:
        context = session.context_manager()
        prime = build_prime_with_isolation_guard(
            _SpecialistProvider(), research_provider=_Web(),
            specialists=SpecialistRegistry([
                SpecialistSpec(name="reasoning",
                               description="Careful analysis of a hard problem.",
                               model="m")]),
            tools=ToolRegistry.with_builtins())
        prime._context = context
        prime._mind = prime._mind.__class__(context)
        for name in ("recall", "learn"):
            specialist = prime.registry.get_specialist(name)
            if specialist is not None:
                specialist.bind(prime._mind)
        aurora = Aurora(prime, brain=brain, config=AuroraConfig.from_env())

        from transformers import AutoTokenizer
        probe = AutoTokenizer.from_pretrained(
            r"C:\PRIMORA PROJECT\RESOURCES\aurora\v0.1\model",
            trust_remote_code=True)
        composed = aurora._config.system_instruction
        for builder in (build_capability_instruction, build_specialist_instruction,
                        build_tool_instruction):
            try:
                composed += builder(prime)
            except Exception:
                pass
        rendered = probe.apply_chat_template(
            [{"role": "system", "content": composed}],
            tokenize=False, add_generation_prompt=True)
        prompt_tokens = len(
            probe(rendered, add_special_tokens=False)["input_ids"])
        config = AuroraConfig.from_env()
        print(f"  composed system prompt : {prompt_tokens} tokens")
        print(f"  target / ceiling       : {config.context_target} / "
              f"{config.context_ceiling}")
        print(f"  room for conversation  : "
              f"{config.context_target - prompt_tokens} tokens")

        repeat = int(os.environ.get("DIAG_REPEAT", "1"))
        for name, prompt in SCENARIOS:
            for attempt in range(1, repeat + 1):
                aurora.mind.begin_task("diag", "diagnosis")
                if name == "multi-step":
                    context.reconcile([
                        {"role": "user",
                         "content": "the berth code is foghorn-42, remember that"},
                        {"role": "assistant",
                         "content": "noted, the berth code is foghorn-42"},
                    ], "diag")
                before = len(aurora.loop.executor.calls)
                result = aurora.respond(
                    [{"role": "user", "content": prompt}], "diag")
                actions = [a for a, _ in aurora.loop.executor.calls[before:]]
                trace = aurora.loop.last_trace

                print("\n" + "=" * 72)
                label = name if repeat == 1 else f"{name} run {attempt}"
                print(f"SCENARIO: {label}")
                print("=" * 72)
                print(f"  stop_reason : {trace.stop_reason}")
                print(f"  cycles      : {len(trace.segments)}")
                print(f"  executed    : {actions}")
                for segment in trace.segments:
                    parsed = extract_prime_request(segment.content or "")
                    print(f"  --- segment {segment.index} "
                          f"(requested={segment.requested}) ---")
                    raw = (segment.content or "").strip()
                    print(f"  RAW:\n{_indent(raw[:600])}")
                    print(f"  PARSED: "
                          f"{'None' if parsed is None else parsed.action}")
                print(f"  FINAL: {_one_line(result.content or '')}")

    return 0


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())


def _one_line(text: str) -> str:
    flat = " ".join(text.split())
    return flat[:300] + ("..." if len(flat) > 300 else "")


if __name__ == "__main__":
    raise SystemExit(main())