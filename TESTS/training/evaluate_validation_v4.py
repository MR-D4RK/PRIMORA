"""Run the real Aurora brain on the v0.4 validation split and score it.

This is the **pre-v0.4 behavioral baseline**: untrained Aurora, measured
against the v0.4 behavioral contract (``responses_v4`` specs, via the shared
``validate_dataset_v4.score_reply`` checker) on the 180 validation records.

Rules kept deliberately boring:

*   validation split only - never train, never holdout;
*   no training, no configuration change, no model change: the brain that
    loads here is the same one production uses today;
*   every record's message prefix is fed to ``aurora.respond`` and the
    reply is scored with the exact same gates the dataset itself had to
    pass, so the numbers describe behaviour, not vibes.

Outputs (under ``TESTS/training/data/v4/``):

*   ``real_model_results.jsonl`` - one line per record, written
    incrementally so an aborted run still leaves usable measurements;
*   ``real_model_report.json``   - overall / per-category / per-difficulty
    pass rates plus the failed-check tally, labelled as the baseline.

Run: python TESTS/training/evaluate_validation_v4.py
Env:  EVAL_LIMIT=n  score only the first n records (0 = all 180)
"""

from __future__ import annotations

import collections
import json
import os
import sys
import tempfile
import time

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import seed_bank_v4 as SB4  # noqa: E402
import validate_dataset_v4 as VD4  # noqa: E402

from PRIMORA.aurora import Aurora  # noqa: E402
from PRIMORA.aurora.brain import AuroraBrain  # noqa: E402
from PRIMORA.aurora.configuration import AuroraConfig  # noqa: E402
from PRIMORA.aurora.loop import DEFAULT_CAPABILITY_BUDGET  # noqa: E402
from PRIMORA.consult import SpecialistRegistry, SpecialistSpec  # noqa: E402
from PRIMORA.prime import Prime  # noqa: E402
from PRIMORA.prime.contract import PrimeResult  # noqa: E402
from PRIMORA.prime.context.isolation import harness_session  # noqa: E402
from PRIMORA.prime.providers import Provider  # noqa: E402
from PRIMORA.research import ResearchProvider, ResearchResult  # noqa: E402
from PRIMORA.toolbox import ToolRegistry  # noqa: E402

LABEL = "pre-v0.4 behavioral baseline"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "v4")
SEED_INFO = {seed["id"]: seed for seed in SB4.SEEDS}


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


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prefix_of(row: dict) -> list[dict[str, str]]:
    """Everything before the supervised target: the user's turns plus any
    context acknowledgement or prior exchange the record grounds on."""
    return row["messages"][:-1]


def percent(passed: int, total: int) -> float:
    return round(100.0 * passed / total, 1) if total else 0.0


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    records = load(os.path.join(DATA_DIR, "validation.jsonl"))
    cap = int(os.environ.get("EVAL_LIMIT", "0"))
    if cap:
        records = records[:cap]

    print("=" * 72)
    print(f"AURORA BEHAVIORAL BASELINE ({LABEL})")
    print("=" * 72)
    print(f"  validation records    : {len(records)}")
    print(f"  max_cognition_cycles  : "
          f"{AuroraConfig.from_env().max_cognition_cycles}")
    print(f"  capability budget     : {DEFAULT_CAPABILITY_BUDGET}")

    brain = AuroraBrain()
    if brain.warmup():
        print("  MODEL FAILED TO LOAD")
        return 1

    tmp = tempfile.mkdtemp(prefix="aurora_eval_v4_")
    results_path = os.path.join(DATA_DIR, "real_model_results.jsonl")
    os.makedirs(DATA_DIR, exist_ok=True)

    per_category: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "pass": 0})
    per_difficulty: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "pass": 0})
    failed_checks: collections.Counter = collections.Counter()
    history_fallbacks = 0
    errors = 0
    started = time.time()

    with open(results_path, "w", encoding="utf-8") as results_handle:
        with harness_session("eval_v4_", root=tmp) as session:
            context = session.context_manager()
            from PRIMORA.prime.context.isolation import (
                build_prime_with_isolation_guard)
            prime = build_prime_with_isolation_guard(
                _SpecialistProvider(),
                research_provider=_Web(),
                specialists=SpecialistRegistry([
                    SpecialistSpec(name="reasoning",
                                   description="Careful analysis of a hard "
                                               "problem.",
                                   model="m")]),
                tools=ToolRegistry.with_builtins())
            prime._context = context
            prime._mind = prime._mind.__class__(context)
            for name in ("recall", "learn"):
                cap_ = prime.registry.get_specialist(name)
                if cap_ is not None:
                    cap_.bind(prime._mind)
            aurora = Aurora(prime, brain=brain, config=AuroraConfig.from_env())

            for index, row in enumerate(records):
                sid = row["seed_id"]
                seed = SEED_INFO[sid]
                difficulty = str(seed["difficulty"])
                category = row["category"]

                aurora.mind.begin_task("eval", "validation")
                # The executor's call log is cumulative and is not reset per
                # turn, so snapshot its length and slice this example's calls
                # out of it.
                calls_before = len(aurora.loop.executor.calls)
                fallback = False
                error = None
                try:
                    result = aurora.respond(prefix_of(row), "eval")
                    reply = result.content or ""
                except Exception as exc:
                    error = f"prefix error: {exc}"
                    reply = ""
                if not reply.strip():
                    # History-shaped prefixes are the one structural thing
                    # this baseline did not train for; retry as a single
                    # turn and record that the history was dropped.
                    fallback = True
                    try:
                        single = [{"role": "user",
                                   "content": prefix_of(row)[-1]["content"]}]
                        result = aurora.respond(single, "eval")
                        reply = result.content or ""
                    except Exception as exc:
                        error = f"single-turn error: {exc}"
                        reply = ""
                if fallback:
                    history_fallbacks += 1
                if error:
                    errors += 1

                fails = VD4.score_reply(sid, reply)
                if error:
                    fails.append("error")
                passed = not fails

                per_category[category]["n"] += 1
                per_category[category]["pass"] += int(passed)
                per_difficulty[difficulty]["n"] += 1
                per_difficulty[difficulty]["pass"] += int(passed)
                for fail in fails:
                    failed_checks[fail.split(":", 1)[0]] += 1

                results_handle.write(json.dumps({
                    "id": row["id"],
                    "seed_id": sid,
                    "category": category,
                    "difficulty": seed["difficulty"],
                    "variant": row["variant"],
                    "passed": passed,
                    "failed_checks": fails,
                    "history_fallback": fallback,
                    "error": error,
                    "reply": reply[:600],
                }, ensure_ascii=False) + "\n")
                results_handle.flush()

                if (index + 1) % 20 == 0:
                    print(f"    {index + 1}/{len(records)} "
                          f"({time.time() - started:.0f}s)", flush=True)

    total = len(records)
    total_pass = sum(v["pass"] for v in per_category.values())
    report = {
        "label": LABEL,
        "rule": ("Pre-v0.4 behavioral baseline: untrained Aurora scored on "
                 "the 180-record validation split with the dataset's own "
                 "static gates (validate_dataset_v4.score_reply). No "
                 "training, no config/model changes, holdout untouched."),
        "records": total,
        "overall": {"pass": total_pass,
                    "fail": total - total_pass,
                    "pass_rate_percent": percent(total_pass, total)},
        "per_category": {
            cat: {**counts,
                  "pass_rate_percent": percent(counts["pass"], counts["n"])}
            for cat, counts in sorted(per_category.items())},
        "per_difficulty": {
            diff: {**counts,
                   "pass_rate_percent": percent(counts["pass"], counts["n"])}
            for diff, counts in sorted(per_difficulty.items(), key=lambda kv: kv[0])},
        "failed_checks": dict(failed_checks.most_common()),
        "history_fallbacks": history_fallbacks,
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 1),
        "results_jsonl": results_path,
    }
    report_path = os.path.join(DATA_DIR, "real_model_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n" + "=" * 72)
    print(f"RESULT - {LABEL}")
    print("=" * 72)
    print(f"  overall pass          : "
          f"{report['overall']['pass_rate_percent']}% "
          f"({total_pass}/{total})")
    for cat, counts in report["per_category"].items():
        print(f"    {cat}: {counts['pass_rate_percent']:>5}% "
              f"({counts['pass']}/{counts['n']})")
    for diff, counts in report["per_difficulty"].items():
        print(f"    difficulty {diff}: {counts['pass_rate_percent']:>5}% "
              f"({counts['pass']}/{counts['n']})")
    if failed_checks:
        print("  failed checks:")
        for name, count in failed_checks.most_common(12):
            print(f"    {name:<24} {count}")
    print(f"  history fallbacks     : {history_fallbacks}")
    print(f"  errors                : {errors}")
    print(f"  report                : {report_path}")
    print(f"  results               : {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
