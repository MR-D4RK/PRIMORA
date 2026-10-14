"""Is recovery driven by context structure, or by sampling variance?

The context dump showed something thatundermines the whole live-versus-primed
framing: in the live run, prompt 1 and prompt 2 were byte-identical message
lists, and produced a narration and then a correction. If identical inputs
produce different behaviours, sampling is doing the work, not the context.

This tests that directly. It takes one fixed message list - user, bad request,
rejection - and generates from it N times, counting how often a valid
corrected request appears versus narration. Then it varies one thing at a time:

  variant1the identical prompt, repeated
  variant2  same, but the rejection is the immediately preceding turn
            (no user turn before the assistant request)
  variant3  same, but with a fresh system-message suffix asking for one
            request only

The point is to find the smallest change that flips narration to a correction,
or to establish that no prompt-level change reliably does.

Investigation only. No production code, no temperature change, no training data.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from transformers import AutoTokenizer

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
from PRIMORA.prime.providers import Provider
from PRIMORA.toolbox import ToolRegistry

PROMPT = ("How many words are in this note: the quick brown fox jumps over "
          "the lazy dog?")

BAD_REQUEST = (
    "<prime_request>\n"
    '{"action": "count_words", "payload": {"text": "the quick brown fox '
    'jumps over the lazy dog"}, "context": null, "metadata": null}\n'
    "</prime_request>"
)

REJECTION = json.dumps({
    "status": "failure", "output": None,
    "error": (
        "Unknown capability: 'count_words'. Registered capabilities are: "
        "calculate, consult_specialist, learn, recall, research, use_tool. "
        "A tool name is never an action: use action 'use_tool' with the tool "
        "name in payload.name. A specialist name is never an action: use "
        "action 'consult_specialist' with the specialist name in "
        "payload.specialist."
    ),
})

NARRATION_MARKERS = (
    "i missed", "let me fix", "i'll fix", "i will fix", "let me correct",
    "you're right", "you are right", "i made the wrong", "i used the wrong",
    "i'll correct", "let me try again",
)


class _P(Provider):
    name = "variance-provider"

    def invoke(self, capability, payload):
        return PrimeResult.success({"content": "ok"})


def _system_prompt() -> str:
    prime = Prime_holder.prime
    config = AuroraConfig.from_env()
    composed = config.system_instruction
    composed += build_capability_instruction(prime)
    composed += build_specialist_instruction(prime)
    composed += build_tool_instruction(prime)
    return composed


class Prime_holder:
    prime = None


def classify(text: str) -> str:
    request = extract_prime_request(text or "")
    if request is not None:
        if (request.action == "use_tool"
                and isinstance(request.payload.get("name"), str)
                and "arguments" in request.payload):
            return "CORRECT"
        return f"OTHER_REQUEST:{request.action}"
    low = (text or "").lower()
    if any(marker in low for marker in NARRATION_MARKERS):
        return "NARRATED"
    return "OTHER"


def build_messages(system: str, variant: str) -> list[dict]:
    bad = {"role": "assistant", "content": BAD_REQUEST}
    tool = {"role": "tool", "content": REJECTION}
    if variant == "immediate":
        # No user turn before the failed request: the rejection is the last
        # thing that happened, with no intervening user turn to close the topic.
        return [{"role": "system", "content": system},
                {"role": "user", "content": PROMPT},
                bad, tool]
    if variant == "reminder":
        return [{"role": "system", "content": system + (
            "\n\nThe last request was refused. Reply with the corrected "
            "request block and nothing else.")},
            {"role": "user", "content": PROMPT},
            bad, tool]
    # baseline
    return [{"role": "system", "content": system},
            {"role": "user", "content": PROMPT},
            bad, tool]


def main() -> int:
    prime = Prime_holder.prime
    Prime_holder.prime = Prime(
        _P(),
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
    system = _system_prompt()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True)

    print("=" * 78)
    print("RECOVERY VARIANCE TEST - is it structure or sampling?")
    print("=" * 78)
    print(f"  temperature={config.temperature} top_p={config.top_p} "
          f"repetition_penalty={config.repetition_penalty} "
          f"max_new_tokens={config.max_new_tokens}")
    print(f"  N samples per variant")

    trials = int(os.environ.get("N_TRIALS", "8"))
    results: dict[str, dict[str, int]] = {}
    samples: dict[str, list[str]] = {}

    for variant in ("baseline", "immediate", "reminder"):
        messages = build_messages(system, variant)
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        tokens = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        counts: dict[str, int] = {}
        texts: list[str] = []
        for _ in range(trials):
            result = brain.generate(messages, max_output_tokens=256)
            verdict = classify(result.content)
            counts[verdict] = counts.get(verdict, 0) + 1
            texts.append((result.content or "").strip())
        results[variant] = counts
        samples[variant] = texts
        correct = counts.get("CORRECT", 0)
        narrated = counts.get("NARRATED", 0)
        print(f"\n  variant '{variant}'  ({tokens} prompt tokens, "
              f"{trials} samples)")
        for key, value in sorted(counts.items(),
                                 key=lambda kv: -kv[1]):
            print(f"    {key:<28} {value}")
        print(f"    -> correct {correct}/{trials}  "
              f"({correct / trials * 100:.0f}%)")

    print("\n" + "=" * 78)
    print("SAMPLES - variant 'baseline'")
    print("=" * 78)
    for index, text in enumerate(samples["baseline"][:8]):
        verdict = classify(text)
        print(f"  [{index}] {verdict}")
        print(f"      {' '.join(text.split())[:200]}")

    out = os.path.join(tempfile.gettempdir(),
                       "recovery_variance_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"results": results, "samples": samples,
                   "trials": trials}, fh, indent=2, ensure_ascii=False)
    print(f"\n  full samples written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())