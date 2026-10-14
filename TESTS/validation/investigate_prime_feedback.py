"""Does clear Prime feedback reliably improve recovery across failure types?

The previous investigation showed that one added sentence in the system prompt
took recovery from roughly a third to 14/14 on a single failure type. That is
one type and one phrasing, so it does not yet justify anything about the seed
design. This tests whether the effect generalises.

Three failure types, three materially different but semantically equivalent
feedback wordings, repeated enough times to distinguish a stable rate from a
lucky sample. Failure type C is different in kind: the request was valid and
executed, so there is nothing for Prime to reject - the feedback describes an
empty result instead.

It also measures first-attempt capability use separately, because a model that
recovers reliably but rarely gets it right the first time has a different
problem from one that never emits a request at all.

Runtime experiment only: no training data, no seed changes, no temperature
change, no production code change.
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
from PRIMORA.research import ResearchProvider, ResearchResult
from PRIMORA.toolbox import ToolRegistry

TRIALS = int(os.environ.get("TRIALS", "6"))

NARRATION_MARKERS = (
    "i missed", "let me fix", "i'll fix", "i will fix", "let me correct",
    "you're right", "you are right", "i made the wrong", "i used the wrong",
    "i'll correct", "let me try again", "i should have", "i'll resubmit",
    "let me resend", "next time",
)

# Three wordings, same meaning, different surface form. Chosen to differ in
# imperatives, tone and length so a result is not an artefact of one phrasing.
FEEDBACK = {
    "A": "That request was rejected. Send the corrected request block and "
         "nothing else.",
    "B": "The request failed. Correct it and emit the new request now. "
         "Do not explain.",
    "C": "Try the corrected capability request next. Output only the request "
         "block.",
    "none": "",
}

TEXT = "the quick brown fox jumps over the lazy dog"

FAILURES = {
    "wrong_action": {
        "user": f"How many words are in this note: {TEXT}?",
        "bad": {
            "role": "assistant",
            "content": (
                "<prime_request>\n"
                + json.dumps({"action": "count_words",
                              "payload": {"text": TEXT},
                              "context": None, "metadata": None})
                + "\n</prime_request>"),
        },
        "tool": {
            "status": "failure", "output": None,
            "error": (
                "Unknown capability: 'count_words'. Registered capabilities "
                "are: calculate, consult_specialist, learn, recall, research, "
                "use_tool. A tool name is never an action: use action 'use_tool' "
                "with the tool name in payload.name. A specialist name is never "
                "an action: use action 'consult_specialist' with the specialist "
                "name in payload.specialist."),
        },
        "correct_action": "use_tool",
        "correct_key": "name",
    },
    "missing_field": {
        "user": f"How many words are in this note: {TEXT}?",
        "bad": {
            "role": "assistant",
            "content": (
                "<prime_request>\n"
                + json.dumps({"action": "use_tool",
                              "payload": {"name": "count_words"},
                              "context": None, "metadata": None})
                + "\n</prime_request>"),
        },
        "tool": {
            "status": "failure", "output": None,
            "error": ("Invalid arguments for tool 'count_words': missing "
                      "required argument 'text'"),
        },
        "correct_action": "use_tool",
        "correct_key": "arguments",
    },
    "insufficient_result": {
        "user": "Remind me what we agreed about the shed roof.",
        "bad": {
            "role": "assistant",
            "content": (
                "<prime_request>\n"
                + json.dumps({"action": "recall",
                              "payload": {"query": "shed roof"},
                              "context": None, "metadata": None})
                + "\n</prime_request>"),
        },
        "tool": {
            "status": "success", "output": {"results": []}, "error": None,
        },
        "correct_action": None,
        "correct_key": None,
    },
}


class _Web(ResearchProvider):
    name = "gen-web"

    def search(self, query, limit=5):
        return [ResearchResult(
            "Reference result", "https://example.org/a",
            "A current reference snippet.", "example.org")]


class _P(Provider):
    name = "gen-provider"

    def invoke(self, capability, payload):
        return PrimeResult.success({"content": "ok"})


class Prime_holder:
    prime = None


def compose_system() -> str:
    config = AuroraConfig.from_env()
    text = config.system_instruction
    text += build_capability_instruction(Prime_holder.prime)
    text += build_specialist_instruction(Prime_holder.prime)
    text += build_tool_instruction(Prime_holder.prime)
    return text


def classify(text: str) -> str:
    request = extract_prime_request(text or "")
    if request is not None:
        return "REQUEST"
    low = (text or "").lower()
    if any(m in low for m in NARRATION_MARKERS):
        return "NARRATED"
    if not (text or "").strip():
        return "EMPTY"
    return "OTHER"


def is_correct(text: str, failure: dict) -> bool:
    request = extract_prime_request(text or "")
    if request is None:
        return False
    if failure["correct_action"] is None:
        # Type C: a different, valid request is the right outcome. The only
        # wrong answer here would be repeating the failed recall verbatim.
        return request.action != "recall"
    if request.action != failure["correct_action"]:
        return False
    return failure["correct_key"] in request.payload


def build(system: str, failure: dict, feedback: str) -> list[dict]:
    text = system
    if feedback:
        text = system + f"\n\n{feedback}"
    return [
        {"role": "system", "content": text},
        {"role": "user", "content": failure["user"]},
        dict(failure["bad"]),
        {"role": "tool", "content": json.dumps(failure["tool"])},
    ]


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
    system = compose_system()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True)

    print("=" * 78)
    print("PRIME FEEDBACK GENERALISATION - runtime experiment")
    print("=" * 78)
    print(f"  temperature={config.temperature} top_p={config.top_p} "
          f"max_new_tokens={config.max_new_tokens}")
    print(f"  trials per cell: {TRIALS}")

    # -- does the emitted request actually execute? ------------------------
    print("\n  --- does Prime pick the capability, or does Aurora? ---")
    checks = [
        ("use_tool", {"name": "count_words", "arguments": {"text": TEXT}}),
        ("count_words", {"text": TEXT}),
        ("use_tool", {"name": "count_words"}),
    ]
    from PRIMORA.prime.contract import PrimeRequest
    for action, payload in checks:
        result = Prime_holder.prime.handle(PrimeRequest(action, payload))
        print(f"    {action:<12} {str(payload)[:46]:<48} "
              f"-> {result.status.value}")
    print("    Prime validates and executes; it never substitutes a different")
    print("    capability for the one Aurora sent.")

    rows: list[dict] = []

    # -- recovery grid -----------------------------------------------------
    print("\n" + "=" * 78)
    print("RECOVERY GRID")
    print("=" * 78)
    for failure_name, failure in FAILURES.items():
        print(f"\n  failure type: {failure_name}")
        for key in ("none", "A", "B", "C"):
            feedback = FEEDBACK[key]
            messages = build(system, failure, feedback)
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            tokens = len(tokenizer(rendered,
                                   add_special_tokens=False)["input_ids"])
            counts = {"CORRECT": 0, "NARRATED": 0, "OTHER": 0, "EMPTY": 0,
                      "REQUEST_WRONG": 0}
            for _ in range(TRIALS):
                result = brain.generate(messages, max_output_tokens=256)
                text = result.content or ""
                if is_correct(text, failure):
                    counts["CORRECT"] += 1
                else:
                    counts[classify(text)] = counts.get(
                        classify(text), 0) + 1
            label = "no feedback" if key == "none" else f"wording {key}"
            correct = counts["CORRECT"]
            narrated = counts["NARRATED"]
            print(f"    {label:<14} {tokens:>5} tok  "
                  f"correct {correct}/{TRIALS}  "
                  f"narrated {narrated}/{TRIALS}  "
                  f"other {counts['OTHER'] + counts['EMPTY']}/"
                  f"{TRIALS - correct - narrated}")
            rows.append({
                "failure": failure_name, "wording": key,
                "tokens": tokens, "trials": TRIALS, **counts})

    # -- first-attempt capability use, separately --------------------------
    print("\n" + "=" * 78)
    print("FIRST-ATTEMPT CAPABILITY USE (no rejection exists yet)")
    print("=" * 78)
    first_attempts = [
        ("word count -> use_tool",
         f"How many words are in this note: {TEXT}?", "use_tool"),
        ("weather -> research",
         "What's the weather in Lisbon right now?", "research"),
        ("remember -> learn",
         "Remember for next time: I take my coffee black.", "learn"),
        ("stored detail -> recall",
         "Remind me what we agreed about the shed roof.", "recall"),
        ("trade-off -> consult_specialist",
         "Work through this trade-off properly before I commit.",
         "consult_specialist"),
    ]
    for label, user, expected in first_attempts:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        counts = {"CORRECT": 0, "NO_REQUEST": 0, "WRONG_ACTION": 0}
        for _ in range(TRIALS):
            result = brain.generate(messages, max_output_tokens=256)
            text = result.content or ""
            request = extract_prime_request(text)
            if request is None:
                counts["NO_REQUEST"] += 1
            elif request.action == expected:
                counts["CORRECT"] += 1
            else:
                counts["WRONG_ACTION"] += 1
        print(f"    {label:<30} correct {counts['CORRECT']}/{TRIALS}  "
              f"no request {counts['NO_REQUEST']}/{TRIALS}  "
              f"wrong action {counts['WRONG_ACTION']}/{TRIALS}")

    out = os.path.join(tempfile.gettempdir(), "prime_feedback_grid.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print(f"\n  results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())