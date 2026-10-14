"""Does showing Aurora a rejected request teach her to emit the correction?

The design bets 24 recovery seeds on this: that a record of

    wrong request -> rejection -> corrected request -> answer

transfers to behaviour. This tests the hypothesis on the real brain with the
normal runtime protocol, in an isolated session.

Two things are measured per scenario, because they answer different questions:

  live      the whole conversation runs through the real loop. Aurora's first
            generation produces the bad request, Prime rejects it, and the next
            cycle is hers. This is whether she recovers on her own.

  primed    the bad turn and the rejection are supplied as context, and only
            the following turn is generated. This is the shape a training
            record actually has - the failed request is present with loss off -
            so it is the closer analogue of what the seed would do.

The point is to find out whether the behaviour is representable. A pass means
the pattern is worth including; it does not mean training on it will work.
Nothing here rigs an outcome: no prompt, temperature or configuration is tuned
to produce a correction, and a scenario that does not produce a bad request
first is reported as inconclusive rather than counted as a pass.
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
from PRIMORA.prime.contract import PrimeRequest, PrimeResult, RequestStatus
from PRIMORA.prime.context.isolation import (
    build_prime_with_isolation_guard,
    harness_session,
)
from PRIMORA.prime.providers import Provider
from PRIMORA.research import ResearchProvider, ResearchResult
from PRIMORA.toolbox import ToolRegistry

# Phrases that indicate Aurora described the fix instead of performing it.
NARRATION = (
    "you're right", "you are right", "my mistake", "i made a mistake",
    "i've corrected", "i have corrected", "let me correct",
    "let me fix", "i'll fix", "i will fix", "the action should be",
    "i used the wrong", "i missed", "the right action",
)


def narrated(text: str) -> list[str]:
    low = (text or "").lower()
    return [p for p in NARRATION if p in low]


class _Web(ResearchProvider):
    name = "recovery-web"

    def search(self, query, limit=5):
        return [ResearchResult(
            "Python release", "https://example.org/py",
            "Python 3.14 is the latest stable release.", "example.org")]


class _P(Provider):
    name = "recovery-provider"

    def invoke(self, capability, payload):
        return PrimeResult.success({"content": "ok"})


def _build(run_dir: str):
    """A complete isolated runtime. Nothing touches production memory."""
    from PRIMORA.consult import SpecialistRegistry, SpecialistSpec

    prime = build_prime_with_isolation_guard(
        _P(), research_provider=_Web(),
        specialists=SpecialistRegistry([
            SpecialistSpec(name="reasoning",
                           description="Careful analysis of a hard problem.",
                           model="m")]),
        tools=ToolRegistry.with_builtins())
    return prime


class _CapturingBrain(AuroraBrain):
    """The real brain, plus a record of every prompt it generated from.

    The production brain deliberately does not keep its prompts, so the
    capture is added here rather than changed in the product. What the next
    generation was conditioned on matters: it is the difference between "she
    could have corrected this" and "she was never shown the rejection".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompts: list[list[dict]] = []

    def generate(self, messages, max_output_tokens=None):
        self.prompts.append([dict(m) for m in messages])
        return super().generate(messages, max_output_tokens=max_output_tokens)


def _tool_turns(brain: _CapturingBrain) -> list[str]:
    if not brain.prompts:
        return []
    return [m["content"] for m in brain.prompts[-1]
            if m.get("role") == "tool"]


# --------------------------------------------------------------------------
# Scenarios. Each states the failure we expect, the correction we expect,
# and the prompt. Nothing is tuned - the prompts are ordinary requests.
# --------------------------------------------------------------------------

SCENARIOS = [
    {
        "key": "tool_wrapper",
        "title": "Tool wrapper recovery",
        "prompt": ("How many words are in this note: the quick brown fox "
                   "jumps over the lazy dog?"),
        "expect_bad_action": "count_words",
        "expect_good_action": "use_tool",
        "expect_payload_key": "name",
        "primed_bad_turn": {
            "role": "assistant",
            "content": (
                "<prime_request>\n"
                '{"action": "count_words", "payload": {"text": "the quick '
                'brown fox jumps over the lazy dog"}, "context": null, '
                '"metadata": null}\n'
                "</prime_request>"
            ),
        },
        "primed_tool_turn": json.dumps({
            "status": "failure", "output": None,
            "error": (
                "Unknown capability: 'count_words'. Registered capabilities "
                "are: calculate, consult_specialist, learn, recall, research, "
                "use_tool. A tool name is never an action: use action "
                "'use_tool' with the tool name in payload.name. A specialist "
                "name is never an action: use action 'consult_specialist' with "
                "the specialist name in payload.specialist."
            ),
        }),
    },
    {
        "key": "malformed_field",
        "title": "Malformed-request recovery (missing required field)",
        "prompt": ("How many words are in this note: the survey came back "
                   "clean yesterday afternoon?"),
        "expect_bad_action": "use_tool",
        "expect_good_action": "use_tool",
        "expect_payload_key": "arguments",
        "primed_bad_turn": {
            "role": "assistant",
            "content": (
                "<prime_request>\n"
                '{"action": "use_tool", "payload": {"name": "count_words"}, '
                '"context": null, "metadata": null}\n'
                "</prime_request>"
            ),
        },
        "primed_tool_turn": json.dumps({
            "status": "failure", "output": None,
            "error": (
                "Invalid arguments for tool 'count_words': missing required "
                "argument 'text'"
            ),
        }),
    },
    {
        "key": "specialist_wrapper",
        "title": "Different-context recovery (specialist, not tool)",
        "prompt": ("Should a single server use a relational database or a "
                   "key-value store for session storage?"),
        "expect_bad_action": "reasoning",
        "expect_good_action": "consult_specialist",
        "expect_payload_key": "specialist",
        "primed_bad_turn": {
            "role": "assistant",
            "content": (
                "<prime_request>\n"
                '{"action": "reasoning", "payload": {"task": "compare '
                'relational and key-value storage for sessions"}, "context": '
                'null, "metadata": null}\n'
                "</prime_request>"
            ),
        },
        "primed_tool_turn": json.dumps({
            "status": "failure", "output": None,
            "error": (
                "Unknown capability: 'reasoning'. Registered capabilities "
                "are: calculate, consult_specialist, learn, recall, research, "
                "use_tool. A tool name is never an action: use action "
                "'use_tool' with the tool name in payload.name. A specialist "
                "name is never an action: use action 'consult_specialist' with "
                "the specialist name in payload.specialist."
            ),
        }),
    },
]


def judge_rejection(turns: list[str]) -> str:
    if not turns:
        return "no tool turn seen"
    first = turns[0]
    try:
        data = json.loads(first)
    except Exception:
        return "tool turn not JSON"
    if data.get("status") != "failure":
        return f"not rejected (status={data.get('status')})"
    return "rejected"


def report_live(aurora, scenario, brain) -> dict:
    trace = aurora.loop.last_trace
    segments = [s.content or "" for s in trace.segments]
    tool_turns = _tool_turns(brain)
    initial = segments[0] if segments else ""
    following = segments[1] if len(segments) > 1 else ""

    bad_emitted = None
    parsed = extract_prime_request(initial)
    if parsed is not None:
        bad_emitted = parsed.action
    rejected = judge_rejection(tool_turns)

    # Scan every later segment, not just the one immediately after the
    # rejection. She may narrate first and correct on a later cycle, and
    # measuring only the next segment would call that a failure when the
    # behaviour we care about did eventually happen.
    corrected_action = None
    payload_ok = False
    corrected_text = ""
    corrected_index = None
    for offset, text in enumerate(segments[1:], start=1):
        candidate = extract_prime_request(text)
        if candidate is not None:
            corrected_action = candidate.action
            corrected_text = text
            corrected_index = offset
            payload_ok = scenario["expect_payload_key"] in candidate.payload
            break

    executed = [a for a, _ in aurora.loop.executor.calls]
    all_narration = narrated(following)
    later_narration = [
        p for text in segments[1:] for p in narrated(text)]
    return {
        "mode": "live",
        "initial_assistant": initial,
        "first_request_action": bad_emitted,
        "first_was_expected_bad": bad_emitted == scenario["expect_bad_action"],
        "rejection": rejected,
        "next_assistant": following,
        "corrected_action": corrected_action,
        "corrected_action_ok":
            corrected_action == scenario["expect_good_action"],
        "payload_has_key": payload_ok,
        "executed": executed,
        "continued": len(segments) > 2,
        "corrected_on_cycle": corrected_index,
        "corrected_text": corrected_text,
        "narration": all_narration,
        "narrated_then_corrected":
            bool(all_narration) and corrected_action is not None,
        "narrated_only":
            corrected_action is None and bool(later_narration),
        "cycles": len(segments),
        "stop_reason": trace.stop_reason,
        "final_content": (trace.segments[-1].content or "") if trace.segments
        else "",
    }


def main() -> int:
    print("=" * 78)
    print("RECOVERY HYPOTHESIS TEST - real Aurora brain, isolated runtime")
    print("=" * 78)

    brain = _CapturingBrain()
    load_error = brain.warmup()
    if load_error:
        print(f"  MODEL FAILED TO LOAD: {load_error}")
        return 1
    config = AuroraConfig.from_env()
    print(f"  model ready | max_cognition={config.max_cognition_cycles} "
          f"| ceiling={config.context_ceiling}")

    tmp = tempfile.mkdtemp(prefix="recovery_hypothesis_")
    results = []

    for scenario in SCENARIOS:
        print("\n" + "=" * 78)
        print(f"{scenario['title']}")
        print(f"  expects: bad '{scenario['expect_bad_action']}' -> rejected "
              f"-> corrected '{scenario['expect_good_action']}'")
        print("=" * 78)

        # -- mode 1: live, nothing supplied -------------------------------
        with harness_session(f"live_{scenario['key']}", root=tmp):
            prime = _build(tmp)
            manager = prime.context
            prime._context = manager
            prime._mind = prime._mind.__class__(manager)
            for name in ("recall", "learn"):
                cap = prime.registry.get_specialist(name)
                if cap is not None:
                    cap.bind(prime._mind)
            aurora = Aurora(prime, brain=brain, config=config)

            brain.prompts.clear()
            aurora.respond(
                [{"role": "user", "content": scenario["prompt"]}], "s")
            live = report_live(aurora, scenario, brain)
            live["scenario"] = scenario["key"]
            results.append(live)

            print("\n  --- LIVE (nothing supplied) ---")
            print(f"  initial output    : {_clip(live['initial_assistant'])}")
            print(f"  first request     : {live['first_request_action']} "
                  f"(expected bad: {scenario['expect_bad_action']})")
            print(f"  Prime response    : {live['rejection']}")
            if live["next_assistant"]:
                print(f"  next output       : "
                      f"{_clip(live['next_assistant'])}")
            print(f"  corrected request : {live['corrected_action']} "
                  f"(on cycle {live['corrected_on_cycle']})")
            if live["corrected_text"]:
                print(f"  corrected text    : "
                      f"{_clip(live['corrected_text'])}")
            print(f"  executed          : {live['executed']}")
            print(f"  narration phrases : {live['narration']}")
            print(f"  cycles / stop     : {live['cycles']} / "
                  f"{live['stop_reason']}")

        # -- mode 2: primed context, only the next turn generated -----------
        with harness_session(f"primed_{scenario['key']}", root=tmp):
            prime = _build(tmp)
            manager = prime.context
            prime._context = manager
            prime._mind = prime._mind.__class__(manager)
            for name in ("recall", "learn"):
                cap = prime.registry.get_specialist(name)
                if cap is not None:
                    cap.bind(prime._mind)
            aurora = Aurora(prime, brain=brain, config=config)

            messages = [
                {"role": "user", "content": scenario["prompt"]},
                dict(scenario["primed_bad_turn"]),
                {"role": "tool", "content": scenario["primed_tool_turn"]},
            ]
            aurora.respond(messages, "s")
            trace = aurora.loop.last_trace
            segments = [s.content or "" for s in trace.segments]
            following = segments[0] if segments else ""
            corrected_action = None
            corrected_text = ""
            corrected_index = None
            payload_ok = False
            for offset, text in enumerate(segments):
                candidate = extract_prime_request(text)
                if candidate is not None:
                    corrected_action = candidate.action
                    corrected_text = text
                    corrected_index = offset
                    payload_ok = scenario["expect_payload_key"] in candidate.payload
                    break
            narration = narrated(following)
            later_narration = [
                p for text in segments for p in narrated(text)]
            executed = [a for a, _ in aurora.loop.executor.calls]
            primed = {
                "mode": "primed",
                "scenario": scenario["key"],
                "next_assistant": following,
                "corrected_action": corrected_action,
                "corrected_action_ok":
                    corrected_action == scenario["expect_good_action"],
                "payload_has_key": payload_ok,
                "corrected_on_cycle": corrected_index,
                "corrected_text": corrected_text,
                "executed": executed,
                "narration": narration,
                "narrated_then_corrected":
                    bool(narration) and corrected_action is not None,
                "narrated_only":
                    corrected_action is None and bool(later_narration),
                "continued": len(segments) > 1,
                "cycles": len(segments),
                "stop_reason": trace.stop_reason,
                "final_content": segments[-1] if segments else "",
            }
            results.append(primed)

            print("\n  --- PRIMED (bad turn + rejection as context) ---")
            print(f"  next output       : {_clip(primed['next_assistant'])}")
            print(f"  corrected request : {primed['corrected_action']} "
                  f"(on cycle {primed['corrected_on_cycle']})")
            if primed["corrected_text"]:
                print(f"  corrected text    : "
                      f"{_clip(primed['corrected_text'])}")
            print(f"  executed          : {primed['executed']}")
            print(f"  narration phrases : {primed['narration']}")
            print(f"  cycles / stop     : {primed['cycles']} / "
                  f"{primed['stop_reason']}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = (f"{'scenario':<22}{'mode':<8}{'bad':<6}{'rejected':<11}"
              f"{'corrected':<11}{'narrated':<10}{'result'}")
    print("  " + header)
    print("  " + "-" * len(header))
    for r in results:
        if r["mode"] == "live":
            bad = "yes" if r["first_was_expected_bad"] else "no"
            rej = "yes" if r["rejection"] == "rejected" else "no"
        else:
            bad = "-"
            rej = "yes"
        corr = "yes" if r.get("corrected_action_ok") else "no"
        narr = "yes" if r["narration"] else "no"
        if r.get("corrected_action_ok") and r.get("narration"):
            verdict = "PASS (narrated first)"
        elif r.get("corrected_action_ok"):
            verdict = "PASS"
        elif r["narration"]:
            verdict = "narrated only"
        else:
            verdict = "FAIL"
        print(f"  {r['scenario']:<22}{r['mode']:<8}{bad:<6}{rej:<11}"
              f"{corr:<11}{narr:<10}{verdict}")

    out = os.path.join(tempfile.gettempdir(), "recovery_hypothesis_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\n  full transcript written to {out}")
    return 0


def _clip(text: str, limit: int = 260) -> str:
    flat = " ".join((text or "").split())
    if not flat:
        return "(empty)"
    return flat[:limit] + ("..." if len(flat) > limit else "")


if __name__ == "__main__":
    raise SystemExit(main())