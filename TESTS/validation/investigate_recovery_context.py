"""Why does Aurora correct in the live loop but narrate from history?

The hypothesis test showed live recovery working once and primed recovery
failing 3/3. Both were given a bad request and a rejection. This isolates the
difference by dumping the exact message list in each case and diffing them,
then varying one thing at a time.

Investigation only: no production change, no temperature change, no training
data. The only new code is this probe plus a temporary instrumentation shim
that records prompts at the brain boundary, which the production brain does
not do and which this investigation needs.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.aurora import extract_prime_request
from PRIMORA.aurora.brain import AuroraBrain
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.consult import SpecialistRegistry, SpecialistSpec
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context.isolation import (
    build_prime_with_isolation_guard,
    harness_session,
)
from PRIMORA.prime.providers import Provider
from PRIMORA.research import ResearchProvider, ResearchResult
from PRIMORA.toolbox import ToolRegistry

PROMPT = ("How many words are in this note: the quick brown fox jumps over "
          "the lazy dog?")

BAD_REQUEST = (
    "<prime_request>\n"
    '{"action": "count_words", "payload": {"text": "the quick brown fox '
    'jumps over the lazy dog"}, "context": null, "metadata": null}\n'
    "</prime_request>"
)

REJECTION = {
    "status": "failure",
    "output": None,
    "error": (
        "Unknown capability: 'count_words'. Registered capabilities are: "
        "calculate, consult_specialist, learn, recall, research, use_tool. "
        "A tool name is never an action: use action 'use_tool' with the tool "
        "name in payload.name. A specialist name is never an action: use "
        "action 'consult_specialist' with the specialist name in "
        "payload.specialist."
    ),
}


class _Web(ResearchProvider):
    name = "probe-web"

    def search(self, query, limit=5):
        return [ResearchResult(
            "Reference result", "https://example.org/a",
            "A current reference snippet for the query.", "example.org")]


class _P(Provider):
    name = "probe-provider"

    def invoke(self, capability, payload):
        return PrimeResult.success({"content": "ok"})


class _CapturingBrain(AuroraBrain):
    """The real brain plus a record of every prompt it generated from."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompts: list[list[dict]] = []

    def generate(self, messages, max_output_tokens=None):
        self.prompts.append([dict(m) for m in messages])
        return super().generate(messages, max_output_tokens=max_output_tokens)


def _runtime(tmp: str):
    prime = build_prime_with_isolation_guard(
        _P(), research_provider=_Web(),
        specialists=SpecialistRegistry([
            SpecialistSpec(name="reasoning",
                           description="Careful analysis of a hard problem.",
                           model="m")]),
        tools=ToolRegistry.with_builtins())
    manager = prime.context
    prime._context = manager
    prime._mind = prime._mind.__class__(manager)
    for name in ("recall", "learn"):
        cap = prime.registry.get_specialist(name)
        if cap is not None:
            cap.bind(prime._mind)
    return prime


def dump(label: str, messages: list[dict]) -> None:
    print(f"\n  --- {label}: {len(messages)} messages ---")
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        flat = " ".join(content.split())
        has_block = "<prime_request>" in content
        is_failure = '"status": "failure"' in content or '"failure"' in content
        tags = []
        if has_block:
            tags.append("BLOCK")
        if is_failure:
            tags.append("FAILURE")
        tag = f"  [{','.join(tags)}]" if tags else ""
        print(f"    [{index}] {role:<9} {len(content):>6} chars{tag}")
        print(f"         {flat[:230]}")


def summarise(messages: list[dict]) -> dict:
    return {
        "count": len(messages),
        "roles": [m.get("role") for m in messages],
        "system_chars": len(messages[0].get("content", ""))
        if messages and messages[0].get("role") == "system" else 0,
        "user_chars": sum(len(m.get("content", "")) for m in messages
                          if m.get("role") == "user"),
        "assistant_chars": sum(len(m.get("content", "")) for m in messages
                               if m.get("role") == "assistant"),
        "tool_chars": sum(len(m.get("content", "")) for m in messages
                          if m.get("role") == "tool"),
    }


def correct(text: str) -> bool:
    request = extract_prime_request(text or "")
    if request is None:
        return False
    return (request.action == "use_tool"
            and isinstance(request.payload.get("name"), str)
            and "arguments" in request.payload)


def narrated(text: str) -> bool:
    low = (text or "").lower()
    return any(p in low for p in ("i missed", "let me fix", "i'll fix",
                                  "i will fix", "let me correct", "you're right"))


def main() -> int:
    print("=" * 78)
    print("RECOVERY CONTEXT INVESTIGATION - messages, structure, state")
    print("=" * 78)

    brain = _CapturingBrain()
    load_error = brain.warmup()
    if load_error:
        print(f"  MODEL FAILED TO LOAD: {load_error}")
        return 1
    config = AuroraConfig.from_env()
    print(f"  model ready | temperature={config.temperature} "
          f"| top_p={config.top_p} | max_new_tokens={config.max_new_tokens} "
          f"| ceiling={config.context_ceiling}")

    tmp = tempfile.mkdtemp(prefix="recovery_investigation_")
    captured: dict[str, list] = {}

    # -- A: live loop, nothing supplied -----------------------------------
    print("\n" + "=" * 78)
    print("A. LIVE LOOP (nothing supplied)")
    print("=" * 78)
    with harness_session("a_live", root=tmp):
        prime = _runtime(tmp)
        aurora = Aurora(prime, brain=brain, config=config)
        brain.prompts.clear()
        aurora.respond([{"role": "user", "content": PROMPT}], "s")
        trace = aurora.loop.last_trace
        captured["A_live"] = list(brain.prompts)
        captured["A_trace"] = [s.content or "" for s in trace.segments]
        print(f"  cycles={len(trace.segments)} stop={trace.stop_reason} "
              f"executed={[a for a, _ in aurora.loop.executor.calls]}")
        for index, text in enumerate(captured["A_trace"]):
            flag = "CORRECT" if correct(text) else (
                "NARRATED" if narrated(text) else "other")
            print(f"    segment {index}: {flag}")
            print(f"      {' '.join(text.split())[:190]}")

    # -- B: primed, bad request and rejection as plain history -------------
    print("\n" + "=" * 78)
    print("B. PRIMED (history supplied verbatim)")
    print("=" * 78)
    with harness_session("b_primed", root=tmp):
        prime = _runtime(tmp)
        aurora = Aurora(prime, brain=brain, config=config)
        brain.prompts.clear()
        messages = [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": BAD_REQUEST},
            {"role": "tool", "content": json.dumps(REJECTION)},
        ]
        aurora.respond(messages, "s")
        trace = aurora.loop.last_trace
        captured["B_primed"] = list(brain.prompts)
        captured["B_trace"] = [s.content or "" for s in trace.segments]
        print(f"  cycles={len(trace.segments)} stop={trace.stop_reason}")
        for index, text in enumerate(captured["B_trace"]):
            flag = "CORRECT" if correct(text) else (
                "NARRATED" if narrated(text) else "other")
            print(f"    segment {index}: {flag}")
            print(f"      {' '.join(text.split())[:190]}")

    # -- message-level diff -------------------------------------------------
    print("\n" + "=" * 78)
    print("1. MESSAGES - the prompt that produced each recovery attempt")
    print("=" * 78)
    a_prompt = captured["A_live"][1] if len(captured["A_live"]) > 1 else None
    b_prompt = captured["B_primed"][0] if captured["B_primed"] else None
    if a_prompt is not None:
        dump("A live - prompt that led to the correction", a_prompt)
    if b_prompt is not None:
        dump("B primed - prompt that led to the narration", b_prompt)

    print("\n  --- summary ---")
    for label, prompt in (("A live ", a_prompt), ("B primed", b_prompt)):
        if prompt:
            print(f"    {label}: {summarise(prompt)}")

    if a_prompt is not None and b_prompt is not None:
        print("\n  --- non-system message comparison ---")
        a_tail = [m for m in a_prompt if m.get("role") != "system"]
        b_tail = [m for m in b_prompt if m.get("role") != "system"]
        a_sys = next((m["content"] for m in a_prompt
                      if m.get("role") == "system"), "")
        b_sys = next((m["content"] for m in b_prompt
                      if m.get("role") == "system"), "")
        print(f"    system message identical : {a_sys == b_sys}")
        print(f"    A non-system roles       : {[m.get('role') for m in a_tail]}")
        print(f"    B non-system roles       : {[m.get('role') for m in b_tail]}")
        print(f"    non-system text identical: "
              f"{[m.get('content') for m in a_tail] == [m.get('content') for m in b_tail]}")
        for index, (am, bm) in enumerate(zip(a_tail, b_tail)):
            same = am.get("content") == bm.get("content")
            print(f"      [{index}] A role={am.get('role'):<9} "
                  f"B role={bm.get('role'):<9} identical={same}")

    out = os.path.join(tempfile.gettempdir(),
                       "recovery_investigation_dump.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "A_prompts": captured.get("A_live", []),
            "B_prompts": captured.get("B_primed", []),
            "A_segments": captured.get("A_trace", []),
            "B_segments": captured.get("B_trace", []),
        }, fh, indent=2, ensure_ascii=False)
    print(f"\n  full prompts dumped to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())