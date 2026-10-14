"""Run the real Aurora brain on the PROTECTED holdout split and score it.

This is the v0.4-vs-v0.5 comparison harness. It is a faithful adaptation of
``evaluate_validation_v4.py``: the scorer, the prompt construction, the
fallback and the harness are byte-for-byte the same logic, pointed at
``holdout.jsonl`` instead of ``validation.jsonl``.

Rules kept deliberately boring:

*   holdout split only - never train, never validation;
*   no training, no configuration change, no model change: the only thing
    that differs between the two runs is ``PRIMORA_AURORA_ADAPTER_PATH``;
*   every record's message prefix is fed to ``aurora.respond`` and the reply
    is scored with the exact same gates the dataset itself had to pass
    (``validate_dataset_v4.score_reply``), so the two adapters are measured
    by one unchanging yardstick.

Outputs (under ``TESTS/training/holdout_eval/``, a directory that holds
nothing the dataset builders read):

*   ``<label>_results.jsonl`` - one line per record, written incrementally
    so an aborted run still leaves usable measurements;
*   ``<label>_report.json``   - overall / per-category / per-difficulty pass
    rates plus the failed-check tally.

Additive evidence, NOT scoring: each record also records which capability
actions the executor actually ran (``actions``). The pass/fail verdict is
produced only by ``score_reply``; ``actions`` is reported so tool-call
behaviour can be inspected without touching the gate.

Run: python TESTS/training/evaluate_holdout_v5.py
Env:  EVAL_LABEL=v0.4         label used in output filenames (required)
      PRIMORA_AURORA_ADAPTER_PATH  which adapter to load (required)
      EVAL_LIMIT=n            score only the first n records (0 = all 180)
      EVAL_SEED=n             if set, seed torch/RNG identically before each
                              record so the two adapters draw the same noise
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

SPLIT = "holdout"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "v5")
OUT_DIR = os.path.join(HERE, "holdout_eval")
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


def _seed_everything(seed: int) -> None:
    """Identical RNG start state per record for both adapters.

    Only reached when EVAL_SEED is set. It changes which random draws are
    taken, never the sampling parameters themselves.
    """
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> int:
    label = os.environ.get("EVAL_LABEL", "").strip()
    adapter = os.environ.get("PRIMORA_AURORA_ADAPTER_PATH", "").strip()
    if not label or not adapter:
        print("EVAL_LABEL and PRIMORA_AURORA_ADAPTER_PATH are required")
        return 2

    records = load(os.path.join(DATA_DIR, f"{SPLIT}.jsonl"))
    cap = int(os.environ.get("EVAL_LIMIT", "0"))
    if cap:
        records = records[:cap]

    seed_env = os.environ.get("EVAL_SEED", "").strip()
    per_record_seed = int(seed_env) if seed_env else None

    config = AuroraConfig.from_env()
    print("=" * 72)
    print(f"AURORA HOLDOUT EVALUATION - {label}")
    print("=" * 72)
    print(f"  split                 : {SPLIT} (protected, eval-only)")
    print(f"  records               : {len(records)}")
    print(f"  model_path            : {config.model_path}")
    print(f"  adapter_path          : {config.adapter_path}")
    print(f"  temperature/top_p/rep : "
          f"{config.temperature}/{config.top_p}/{config.repetition_penalty}")
    print(f"  max_new_tokens        : {config.max_new_tokens}")
    print(f"  per-record seed       : {per_record_seed}")
    print(f"  max_cognition_cycles  : {config.max_cognition_cycles}")
    print(f"  capability budget     : {DEFAULT_CAPABILITY_BUDGET}")
    print(f"  scorer                : validate_dataset_v4.score_reply")
    print(f"  system instruction    : "
          f"{'env override' if os.environ.get('PRIMORA_AURORA_SYSTEM_INSTRUCTION') else 'DEFAULT_SYSTEM_INSTRUCTION'}")

    brain = AuroraBrain(config)
    if brain.warmup():
        print("  MODEL FAILED TO LOAD")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    results_path = os.path.join(OUT_DIR, f"{label}_results.jsonl")
    tmp = tempfile.mkdtemp(prefix=f"aurora_holdout_{label}_")

    per_category: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "pass": 0})
    per_difficulty: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "pass": 0})
    failed_checks: collections.Counter = collections.Counter()
    action_counter: collections.Counter = collections.Counter()
    history_fallbacks = 0
    errors = 0
    started = time.time()

    with open(results_path, "w", encoding="utf-8") as results_handle:
        with harness_session(f"holdout_{label}_", root=tmp) as session:
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
            aurora = Aurora(prime, brain=brain, config=config)

            for index, row in enumerate(records):
                sid = row["seed_id"]
                seed = SEED_INFO[sid]
                difficulty = str(seed["difficulty"])
                category = row["category"]

                if per_record_seed is not None:
                    _seed_everything(per_record_seed + index)

                aurora.mind.begin_task("eval", SPLIT)
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

                actions = [a for a, _ in aurora.loop.executor.calls[
                    calls_before:]]
                for action in actions:
                    action_counter[action] += 1

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
                    "actions": actions,
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
        "label": label,
        "split": SPLIT,
        "rule": ("Protected 180-record holdout, scored with the dataset's "
                 "own static gates (validate_dataset_v4.score_reply). "
                 "No training, no config/model changes; the only variable "
                 "between the two adapters is PRIMORA_AURORA_ADAPTER_PATH."),
        "adapter_path": config.adapter_path,
        "model_path": config.model_path,
        "generation": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            "max_new_tokens": config.max_new_tokens,
            "system_instruction": "env override" if os.environ.get(
                "PRIMORA_AURORA_SYSTEM_INSTRUCTION") else "DEFAULT",
            "per_record_seed": per_record_seed,
        },
        "scorer": "validate_dataset_v4.score_reply",
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
            for diff, counts in sorted(per_difficulty.items(),
                                       key=lambda kv: kv[0])},
        "failed_checks": dict(failed_checks.most_common()),
        "actions": dict(action_counter.most_common()),
        "history_fallbacks": history_fallbacks,
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 1),
        "results_jsonl": results_path,
    }
    report_path = os.path.join(OUT_DIR, f"{label}_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n" + "=" * 72)
    print(f"RESULT - {label}")
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
    if action_counter:
        print("  capability actions executed:")
        for name, count in action_counter.most_common(12):
            print(f"    {name:<24} {count}")
    print(f"  history fallbacks     : {history_fallbacks}")
    print(f"  errors                : {errors}")
    print(f"  report                : {report_path}")
    print(f"  results               : {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
