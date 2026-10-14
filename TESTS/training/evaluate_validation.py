"""Run the real Aurora brain on the validation split and measure it.

This is the pre-training baseline. The numbers say what v0.1 has to beat, and
they also say whether the dataset's categories match the failures the brain
actually makes - a category that is already at 100% is wasted capacity.

No training, no holdout, no production memory.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import AuroraBrain
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.aurora.loop import DEFAULT_CAPABILITY_BUDGET
from PRIMORA.consult import SpecialistRegistry, SpecialistSpec
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context.isolation import harness_session
from PRIMORA.prime.providers import Provider
from PRIMORA.research import ResearchProvider, ResearchResult
from PRIMORA.toolbox import ToolRegistry

NARRATION = re.compile(
    r"\b(i'?ll|i will|let me|i'?m going to|"
    r"i can (try )?(research|look|check|ask|retrieve|compare|get|find))",
    re.IGNORECASE)
BLOCK = re.compile(r"<prime_request>(.*?)</prime_request>", re.DOTALL)


class _Web(ResearchProvider):
    name = "eval-web"

    def search(self, query, limit=5):
        return [ResearchResult(
            "Reference result", "https://example.org/a",
            "A current reference snippet for the query.", "example.org")]


class _SpecialistProvider(Provider):
    """A small deterministic stand-in so consult_specialist can complete."""

    name = "eval-specialist"

    def invoke(self, capability, payload):
        text = (payload.get("messages") or [{}])[-1].get("content", "")
        return PrimeResult.success({
            "content": f"Considered analysis of: {text[:120]}"})


def _parses(raw: str) -> bool:
    try:
        return isinstance(json.loads(raw.strip()), dict)
    except Exception:
        return False


def load(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stratify(examples, per_category: int, c5_per_category: int):
    """A proportional-by-category sample, with C5 kept at its 20% share.

    The full validation split is ~55 minutes of real generation on this card.
    A stratified sample keeps every category represented and keeps C5 - the
    over-trigger guard - at its designed proportion, so the sample still
    measures the thing that matters rather than a convenient slice.
    """
    by_category = defaultdict(list)
    for example in examples:
        by_category[example["category"]].append(example)
    chosen = []
    for category, rows in by_category.items():
        cap = (c5_per_category if category == "C5_answer_directly"
               else per_category)
        chosen.extend(rows[:cap])
    chosen.sort(key=lambda e: e["id"])
    return chosen


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    examples = load(os.path.join(here, "data", "validation.jsonl"))
    cap = int(os.environ.get("EVAL_LIMIT", "0"))
    if cap:
        examples = stratify(examples, cap, max(cap, int(cap * 1.25)))
    print("=" * 72)
    print("AURORA v0.1 PRE-TRAINING VALIDATION RUN (real brain)")
    print("=" * 72)
    print(f"  validation examples : {len(examples)}")
    print(f"  max_cognition_cycles : {AuroraConfig.from_env().max_cognition_cycles}")
    print(f"  capability budget    : {DEFAULT_CAPABILITY_BUDGET}")

    brain = AuroraBrain()
    if brain.warmup():
        print("  MODEL FAILED TO LOAD")
        return 1

    tmp = tempfile.mkdtemp(prefix="aurora_eval_")
    per_category = defaultdict(lambda: {
        "n": 0, "block": 0, "valid": 0, "multi": 0, "narrated": 0,
        "direct_ok": 0, "action_ok": 0, "steps": 0,
    })

    started = time.time()
    # Written incrementally: an aborted run still leaves usable measurements
    # rather than nothing, which is what happened to the first attempt.
    per_example_path = os.path.join(here, "data", "validation_results.jsonl")
    os.makedirs(os.path.dirname(per_example_path), exist_ok=True)
    results_handle = open(per_example_path, "w", encoding="utf-8")
    with harness_session("eval_", root=tmp) as session:
        context = session.context_manager()
        from PRIMORA.prime.context.isolation import build_prime_with_isolation_guard
        prime = build_prime_with_isolation_guard(
            _SpecialistProvider(),
            research_provider=_Web(),
            specialists=SpecialistRegistry([
                SpecialistSpec(name="reasoning",
                               description="Careful analysis of a hard problem.",
                               model="m")]),
            tools=ToolRegistry.with_builtins())
        prime._context = context
        prime._mind = prime._mind.__class__(context)
        for name in ("recall", "learn"):
            cap_ = prime.registry.get_specialist(name)
            if cap_ is not None:
                cap_.bind(prime._mind)
        aurora = Aurora(prime, brain=brain, config=AuroraConfig.from_env())

        for index, example in enumerate(examples):
            aurora.mind.begin_task("eval", "validation")
            # The executor's call log is cumulative and is not reset per turn,
            # so snapshot its length and slice this example's calls out of it.
            # Without this, every example reads every earlier example's calls.
            calls_before = len(aurora.loop.executor.calls)
            try:
                result = aurora.respond(
                    [{"role": "user", "content": example["messages"][0]["content"]}],
                    "eval")
                final = result.content or ""
                trace = aurora.loop.last_trace
                actions = [a for a, _ in
                           aurora.loop.executor.calls[calls_before:]]
            except Exception as exc:
                final, trace, actions = f"ERROR {exc}", None, []

            blocks = BLOCK.findall(final)
            record = per_category[example["category"]]
            record["n"] += 1
            if blocks:
                record["block"] += 1
                if len(blocks) > 1:
                    record["multi"] += 1
            else:
                if NARRATION.search(final):
                    record["narrated"] += 1
            for raw in blocks[:1]:
                try:
                    parsed = json.loads(raw.strip())
                    if isinstance(parsed, dict) and parsed.get("action"):
                        record["valid"] += 1
                        if example["expect_action"] and \
                                parsed["action"] == example["expect_action"]:
                            record["action_ok"] += 1
                except Exception:
                    pass
            if example["expect_action"] is None and not blocks:
                record["direct_ok"] += 1
            if len(set(actions)) >= 2:
                record["steps"] += 1
            results_handle.write(json.dumps({
                "id": example["id"],
                "category": example["category"],
                "expect_block": example["expect_block"],
                "expect_action": example["expect_action"],
                "blocks": len(blocks),
                "parsed_action": (json.loads(blocks[0].strip()).get("action")
                                  if blocks and _parses(blocks[0]) else None),
                "actions": actions,
                "narrated": bool(not blocks and NARRATION.search(final)),
                "final": final[:400],
            }, ensure_ascii=False) + "\n")
            results_handle.flush()
            if (index + 1) % 20 == 0:
                print(f"    {index + 1}/{len(examples)} "
                      f"({time.time() - started:.0f}s)", flush=True)
    results_handle.close()

    emit = [c for c in per_category if c not in
            ("C5_answer_directly", "C8_stay_direct", "C9_uncertainty_natural")]
    totals = {"n": 0, "block": 0, "valid": 0, "multi": 0, "narrated": 0,
              "direct_ok": 0, "action_ok": 0, "steps": 0}
    for record in per_category.values():
        for key in totals:
            totals[key] += record[key]

    def pct(a, b):
        return f"{100 * a / b:.1f}%" if b else "n/a"

    print("\n" + "=" * 72)
    print("METRICS")
    print("=" * 72)
    n_emit = sum(per_category[c]["n"] for c in emit)
    emitted = sum(per_category[c]["block"] for c in emit)
    print(f"  capability-use success   : {pct(emitted, n_emit)} "
          f"({emitted}/{n_emit} expected-a-capability examples emitted one)")
    print(f"  valid request rate       : "
          f"{pct(totals['valid'], totals['block'])} "
          f"({totals['valid']}/{totals['block']} emitted blocks parsed)")
    print(f"  multi-block rate         : "
          f"{pct(sum(r['multi'] for r in per_category.values()), n_emit)}")
    print(f"  narrated-intent rate     : "
          f"{pct(totals['narrated'], totals['n'] - totals['block'])} "
          f"of replies with no block")
    direct = sum(per_category[c]["n"] for c in
                 ("C5_answer_directly", "C8_stay_direct",
                  "C9_uncertainty_natural"))
    print(f"  correct direct-answer    : {pct(totals['direct_ok'], direct)} "
          f"(no capability needed and none emitted)")
    print(f"  over-trigger on C5       : "
          f"{pct(per_category['C5_answer_directly']['block'], per_category['C5_answer_directly']['n'])}")
    print(f"  specialist-selection     : "
          f"{pct(per_category['C6_delegate_when_useful']['action_ok'], per_category['C6_delegate_when_useful']['n'])}")
    print(f"  tool-wrapper correctness : "
          f"{pct(per_category['C3_action_wrapper']['action_ok'], per_category['C3_action_wrapper']['n'])}")
    print(f"  multi-step continuation  : "
          f"{pct(per_category['C4_continue_cognition']['steps'], per_category['C4_continue_cognition']['n'])}")

    print("\n  per category:")
    for category in sorted(per_category):
        r = per_category[category]
        print(f"    {category:26s} n={r['n']:3d} block={r['block']:3d} "
              f"valid={r['valid']:3d} multi={r['multi']:3d} "
              f"narr={r['narrated']:3d} action_ok={r['action_ok']:3d} "
              f"steps={r['steps']:3d}")

    out = os.path.join(here, "data", "validation_baseline.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({
            "examples": len(examples),
            "totals": totals,
            "per_category": {k: dict(v) for k, v in per_category.items()},
        }, handle, indent=2)
    print(f"\n  baseline written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
