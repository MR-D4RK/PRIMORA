"""Verifies the capability-awareness and grounding prompt change on the real brain.

Before the change, first-attempt capability selection scored 1 of 5, and an
empty recall result produced a fabricated memory: given "remind me what we
agreed about the shed roof" and an empty result, Aurora answered "we agreed to
remove the old tiles before starting the replacement".

This measures the same six behaviours the earlier experiment measured, plus a
fabrication check that greps the answers for invented specifics rather than
trusting a keyword. Temperature and configuration are untouched, and the
composed prompt is built exactly as the runtime builds it.

Isolated runtime, production memory never opened.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from transformers import AutoTokenizer

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.aurora import extract_prime_request
from PRIMORA.aurora.brain import AuroraBrain
from PRIMORA.aurora.configuration import (
    AuroraConfig,
    build_capability_instruction,
    build_specialist_instruction,
    build_tool_instruction,
)
from PRIMORA.consult import SpecialistRegistry, SpecialistSpec
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context.isolation import (
    build_prime_with_isolation_guard,
    harness_session,
)
from PRIMORA.prime.providers import Provider
from PRIMORA.research import ResearchProvider, ResearchResult
from PRIMORA.toolbox import ToolRegistry

TRIALS = int(os.environ.get("TRIALS", "6"))
TEXT = "the quick brown fox jumps over the lazy dog"

# Invented specifics. An empty result must not produce any of these.
FABRICATIONS = (
    "tiles", "contractor", "weather-resistant", "pitch", "under the",
    "we agreed", "we decided", "you said", "i recorded", "i've learned",
    "as i mentioned", "last time you",
)


class _Web(ResearchProvider):
    name = "verify-web"

    def search(self, query, limit=5):
        return [ResearchResult(
            "Python release", "https://example.org/py",
            "Python 3.14 is the latest stable release.", "example.org")]


class _P(Provider):
    name = "verify-provider"

    def invoke(self, capability, payload):
        return PrimeResult.success({"content": "ok"})


class Prime_holder:
    prime = None


def composed_system() -> str:
    config = AuroraConfig.from_env()
    prime = Prime_holder.prime
    return "".join([
        config.system_instruction,
        build_capability_instruction(prime),
        build_specialist_instruction(prime),
        build_tool_instruction(prime),
    ])


def fabricated(text: str) -> list[str]:
    low = (text or "").lower()
    return [f for f in FABRICATIONS if f in low]


def main() -> int:
    Prime_holder.prime = Prime(
        _P(),
        research_provider=_Web(),
        specialists=SpecialistRegistry([
            SpecialistSpec(name="reasoning",
                           description="Careful analysis of a hard problem.",
                           model="m")]),
        tools=ToolRegistry.with_builtins())

    brain = AuroraBrain()
    load_error = brain.warmup()
    if load_error:
        print(f"  MODEL FAILED TO LOAD: {load_error}")
        return 1
    config = AuroraConfig.from_env()
    system = composed_system()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True)
    tokens = len(tokenizer(system)["input_ids"])

    print("=" * 78)
    print("CAPABILITY AWARENESS + GROUNDING VERIFICATION - real brain")
    print("=" * 78)
    print(f"  temperature={config.temperature} top_p={config.top_p} "
          f"max_new_tokens={config.max_new_tokens} trials={TRIALS}")
    print(f"  composed system prompt: {tokens} tokens "
          f"(headroom {config.context_ceiling - tokens})")

    results: dict[str, dict] = {}

    # -- 1. first-attempt capability selection -----------------------------
    print("\n" + "=" * 78)
    print("FIRST-ATTEMPT CAPABILITY SELECTION")
    print("=" * 78)
    selection = [
        ("recall", "Remind me what we agreed about the shed roof."),
        ("learn", "Remember for next time: I take my coffee black."),
        ("research", "What's the latest Python release right now?"),
        ("use_tool", f"How many words are in this note: {TEXT}?"),
        ("consult_specialist",
         "Work through this trade-off properly before I commit."),
    ]
    for expected, user in selection:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        correct = no_request = wrong = 0
        for _ in range(TRIALS):
            result = brain.generate(messages, max_output_tokens=256)
            request = extract_prime_request(result.content or "")
            if request is None:
                no_request += 1
            elif request.action == expected:
                correct += 1
            else:
                wrong += 1
        results[f"first_attempt:{expected}"] = {
            "correct": correct, "no_request": no_request, "wrong": wrong}
        print(f"    {expected:<20} correct {correct}/{TRIALS}  "
              f"no request {no_request}/{TRIALS}  wrong {wrong}/{TRIALS}")

    # -- 2. empty recall result must not be invented ------------------------
    print("\n" + "=" * 78)
    print("GROUNDING - recall returned nothing")
    print("=" * 78)
    empty = json.dumps({"status": "success", "output": {"results": []},
                        "error": None})
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Remind me what we agreed about the shed "
                                    "roof."},
        {"role": "assistant", "content":
            '<prime_request>\n{"action": "recall", "payload": {"query": '
            '"shed roof"}, "context": null, "metadata": null}\n'
            "</prime_request>"},
        {"role": "tool", "content": empty},
    ]
    fabricated_count = 0
    honest = 0
    re_requested = 0
    samples = []
    for _ in range(TRIALS):
        result = brain.generate(messages, max_output_tokens=256)
        text = (result.content or "").strip()
        samples.append(text)
        hits = fabricated(text)
        if hits:
            fabricated_count += 1
        elif extract_prime_request(text) is not None:
            re_requested += 1
        else:
            honest += 1
    results["empty_result"] = {
        "fabricated": fabricated_count, "re_requested": re_requested,
        "honest": honest}
    print(f"    fabricated an answer  : {fabricated_count}/{TRIALS}")
    print(f"    made another request  : {re_requested}/{TRIALS}")
    print(f"    honest plain answer   : {honest}/{TRIALS}")
    for index, text in enumerate(samples[:4]):
        print(f"      [{index}] {' '.join(text.split())[:150]}")

    # -- 3. end to end through the real loop -------------------------------
    print("\n" + "=" * 78)
    print("END TO END - real loop, isolated session")
    print("=" * 78)
    tmp = tempfile.mkdtemp(prefix="grounding_verify_")

    with harness_session("recall_hit", root=tmp) as session:
        context = session.context_manager()
        prime = build_prime_with_isolation_guard(
            _P(), research_provider=_Web(),
            specialists=SpecialistRegistry([
                SpecialistSpec(name="reasoning",
                               description="Careful analysis of a hard problem.",
                               model="m")]),
            tools=ToolRegistry.with_builtins())
        prime._context = context
        prime._mind = prime._mind.__class__(context)
        for name in ("recall", "learn"):
            cap = prime.registry.get_specialist(name)
            if cap is not None:
                cap.bind(prime._mind)
        aurora = Aurora(prime, brain=brain, config=config)

        context.reconcile([
            {"role": "user", "content": "the berth code is foghorn-42"},
            {"role": "assistant", "content": "noted, foghorn-42"},
        ], "s")
        aurora.mind.begin_task("s", "verify")
        result = aurora.respond(
            [{"role": "user", "content": "Remind me the berth code."}], "s")
        actions = [a for a, _ in aurora.loop.executor.calls]
        print(f"    recall existing memory : executed={actions}")
        print(f"      final: {' '.join((result.content or '').split())[:150]}")
        print(f"      grounded in result   : "
              f"{'foghorn' in (result.content or '').lower()}")
        results["recall_existing"] = {"executed": actions}

    out = os.path.join(tempfile.gettempdir(), "grounding_verify.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"results": results, "empty_samples": samples}, fh,
                  indent=2, ensure_ascii=False)
    print(f"\n  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())