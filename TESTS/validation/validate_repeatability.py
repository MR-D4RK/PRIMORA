"""Repeatability of the real Aurora brain: five runs per scenario.

Measurement only. Nothing in PRIMORA is changed: same default configuration,
same prompt, same capabilities, same protections. The question is not whether
Aurora can do these things - the previous run showed she sometimes can - but
whether the *failures* are stable enough to justify training.

A stable failure is a training target. An unstable one is a measurement
artefact, and training at it would be fitting noise.

Success is counted strictly. Narrating an intention is never a success, and a
multi-step scenario only passes on an actual executed sequence of two or more
distinct capabilities plus a correct answer.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time

sys.path.insert(0, r"C:\PRIMORA PROJECT")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PRIMORA.aurora.brain import AuroraBrain  # noqa: E402
from PRIMORA.aurora.configuration import AuroraConfig  # noqa: E402
from PRIMORA.prime.contract import PrimeRequest  # noqa: E402

from validate_loop_v03_real_model import (  # noqa: E402
    RECALL_NOISE,
    FixedResearchProvider,
    build,
)

BLOCK = re.compile(r"<prime_request>(.*?)</prime_request>", re.DOTALL)
NARRATION = re.compile(
    r"\b(i'?ll|i will|let me|i'?m going to|i can (try )?(research|look|check|ask|"
    r"retrieve|compare|get|find))\b", re.IGNORECASE)

CAPABILITIES = {"recall", "research", "consult_specialist", "learn", "use_tool"}


def classify(obs):
    """Turn one run into a strict verdict. Narration is never a success."""
    actions = [a for a, _ in obs.requests]
    final = obs.final or ""
    blocks = BLOCK.findall(final)
    parsed = []
    for raw in blocks:
        try:
            data = json.loads(raw.strip())
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("action"), str):
            parsed.append(data["action"])

    malformed_reasons = []
    if len(blocks) > 1:
        malformed_reasons.append(f"{len(blocks)}_blocks")
    if blocks and not actions:
        malformed_reasons.append("no_valid_action("
                                 + ",".join(parsed or ["unparsable"]) + ")")
    if blocks and actions and parsed and parsed[0] not in actions:
        malformed_reasons.append("action_mismatch")

    narrated = (not blocks) and bool(NARRATION.search(final))
    executed = [a for a in actions if a in CAPABILITIES]
    distinct = [a for a in dict.fromkeys(executed)]

    return {
        "executed": executed,
        "distinct": distinct,
        "blocks": len(blocks),
        "parsed_actions": parsed,
        "malformed": malformed_reasons,
        "narrated": narrated,
        "generations": obs.generations,
        "calls": len(obs.requests),
        "stop": obs.stop_reason,
        "final": final,
        "correct": obs.correct(),
    }


def judge(name, r):
    """Strict pass/fail. Narrated intent never counts."""
    if r["malformed"]:
        return "malformed"
    if name == "multi-step":
        if len(r["distinct"]) >= 2 and r["correct"] and r["generations"] >= 3:
            return "pass"
        return "fail"
    if name == "no-capability":
        return "pass" if (not r["executed"] and r["correct"]) else "fail"
    if not r["executed"]:
        return "fail"
    if not r["correct"]:
        return "fail"
    return "pass"


def build_scenarios():
    return [
        ("recall",
         "What are my notes on the yard equipment? Also remind me which crane "
         "inspection is booked and for when.",
         ["eleventh"],
         [("user", note, "old-session") for note in RECALL_NOISE],
         True),
        ("research",
         "What is the latest stable release of Python, and when was it "
         "released?",
         ["3.14"],
         [],
         True),
        ("multi-step",
         "We berthed at berth nine with the kestrel seventeen cargo. Confirm "
         "the berth code we used, then check the current Python release, and "
         "tell me both.",
         ["foghorn-42", "3.14"],
         [("user", "the berth code is foghorn-42, remember that"),
          ("assistant", "noted, the berth code is foghorn-42")],
         True),
        ("specialist",
         "Should a single server use a relational database or a key-value "
         "store for session storage, and why? Consult the reasoning "
         "specialist if it would help.",
         [],
         [],
         False),
("tool",
             "How many words are in this note: the quick brown fox jumps over the "
             "lazy dog?",
             [("9", "nine")],
             [],
             True),
        ("no-capability",
         "In one short sentence, what is the capital of France?",
         ["paris"],
         [],
         True),
    ]


def main():
    runs = 5
    config = AuroraConfig.from_env()
    print("=" * 72)
    print(f"REPEATABILITY - {runs} runs per scenario, real brain, defaults")
    print("=" * 72)
    print(f"  max_cognition={config.max_cognition_cycles}  "
          f"ceiling={config.context_ceiling}  total={config.context_total}")
    print(f"  model: {os.path.basename(config.model_path)}")

    brain = AuroraBrain()
    if brain.warmup():
        print("MODEL FAILED TO LOAD")
        return 1

    tmpdir = tempfile.mkdtemp(prefix="aurora_repeat_root_")
    research = FixedResearchProvider()
    scenarios = build_scenarios()
    all_results = {}

    for name, prompt, expected, seed, expects_capability in scenarios:
        print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
        records = []
        for attempt in range(1, runs + 1):
            # A fresh store per run. Reusing one leaks between runs: the loop
            # persists each conversation, so run 1's capability output is
            # retrievable as memory by run 2. That contaminates the result.
            run_dir = tempfile.mkdtemp(dir=tmpdir, prefix=f"{name}_{attempt}_")
            prime, aurora = build(brain, run_dir, research)
            for entry in seed:
                role, content = entry[0], entry[1]
                where = entry[2] if len(entry) > 2 else "s"
                prime._context.reconcile(
                    [{"role": role, "content": content}], where)
            aurora.mind.begin_task("s", "repeatability")
            started = time.time()
            try:
                result = aurora.respond([{"role": "user", "content": prompt}], "s")
                final, success, stop = result.content, result.success, None
                trace = aurora.loop.last_trace
                generations = trace.count
                stop = trace.stop_reason
                calls = list(aurora.loop.executor.calls)
            except Exception as exc:
                final, success, stop = f"ERROR {exc}", False, "error"
                generations, calls = 0, []
            elapsed = time.time() - started

            class _O:
                pass
            obs = _O()
            obs.requests = calls
            obs.final = final
            obs.generations = generations
            obs.stop_reason = stop
            # An expectation may be a bare string, or a tuple of acceptable
            # alternatives. Aurora writing the number as a word is the same
            # answer, and scoring it wrong would have reported a protocol
            # failure that never happened.
            obs.correct = lambda e=expected: (
                all(
                    (any(alt.lower() in (final or "").lower()
                         for alt in (item if isinstance(item, (tuple, list))
                                     else (item,)))
                     if expected else True)
                    for item in (e or ())))

            record = classify(obs)
            record["verdict"] = judge(name, record)
            record["elapsed_s"] = round(elapsed, 1)
            record["expects_capability"] = expects_capability
            records.append(record)

            flags = []
            if record["malformed"]:
                flags.append("MALFORMED:" + "/".join(record["malformed"]))
            if record["narrated"]:
                flags.append("narrated")
            if record["executed"]:
                flags.append("called=" + ",".join(record["executed"]))
            print(f"  run {attempt}: {record['verdict']:10s} "
                  f"gens={record['generations']} calls={record['calls']} "
                  f"stop={record['stop']:16s} "
                  f"{' | '.join(flags) if flags else '-'}")
            # The verdict alone hid why a run failed: a correct answer spelled
            # out ("nine words") was scored wrong against an expected "9", and
            # there was no way to tell that from a real protocol failure. The
            # answer is printed so a wrong verdict is visible rather than
            # something to be taken on trust.
            flat = " ".join((final or "").split())
            print(f"      final: {flat[:150] or '(empty)'}")
            if record["narrated"]:
                print("      NOTE: narration phrasing present in the answer")
        all_results[name] = records

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    summary = {}
    for name, records in all_results.items():
        verdicts = [r["verdict"] for r in records]
        passes = verdicts.count("pass")
        fails = verdicts.count("fail")
        malformed = verdicts.count("malformed")
        if passes == len(records):
            stability = "consistent success"
        elif passes == 0 and (fails + malformed) == len(records):
            stability = "consistent failure"
        else:
            stability = "mixed/unstable"
        summary[name] = {
            "verdicts": verdicts,
            "pass": passes,
            "fail": fails,
            "malformed": malformed,
            "stability": stability,
            "always_called": all(r["executed"] for r in records),
            "never_called": not any(r["executed"] for r in records),
            "actions_seen": sorted({a for r in records
                                    for a in r["executed"]}),
        }
        print(f"  {name:15s} pass={passes} fail={fails} "
              f"malformed={malformed}  -> {stability}")
        print(f"                  actions seen: "
              f"{summary[name]['actions_seen'] or '(none)'}")

    out = os.path.join(tmpdir, "repeatability.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "runs": all_results}, handle, indent=2)
    print(f"\nFull record: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
