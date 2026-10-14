"""T1 interrogation CLI.

    python TESTS/t1/interrogator.py --auto
    python TESTS/t1/interrogator.py --interactive
    python TESTS/t1/interrogator.py --auto --category contradiction
    python TESTS/t1/interrogator.py --auto --scenario D1,E1 --json out.json
    python TESTS/t1/interrogator.py --list

CURRENT: both modes, using the real Aurora generation path against a
throwaway database.

Safety invariants, asserted at run time:
  * no ContextManager or ContextStore is ever built with the default path,
  * the Prime provider refuses all tool calls,
  * the production database hash is recorded before and after and compared,
  * interactive mode can only ever write to the harness's own workdir.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from TESTS.t1.evaluate import AMBIGUOUS, FAIL, PASS, evaluate
from TESTS.t1.report import (
    InterrogationReport,
    build_result,
    expected_behavior_text,
)
from TESTS.t1.runtime import (
    PRODUCTION_DB,
    T1IsolationError,
    T1Runtime,
    production_data_digest,
    production_data_fingerprint,
    production_db_sha256,
)
from TESTS.t1.scenarios import (
    ALL_FAILURE_TYPES,
    SCENARIOS,
    Scenario,
    all_categories,
    load_scenarios,
    validate_corpus,
)

RUNTIME_INFO = {
    "temperature": None,
    "top_p": None,
    "max_new_tokens": None,
    "context_ceiling": None,
    "hybrid_retrieval": None,
    "isolation": "per-scenario temporary database",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="interrogator",
        description="Interrogate the real Aurora brain for T1 grounding "
                    "behaviour and report structured results.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--auto", action="store_true",
                      help="run every selected scenario and report")
    mode.add_argument("--interactive", action="store_true",
                      help="walk the scenarios and judge each answer yourself")
    mode.add_argument("--list", action="store_true",
                      help="list scenarios and exit")

    p.add_argument("--category", help="restrict to one category")
    p.add_argument("--scenario", help="comma-separated scenario ids")
    p.add_argument("--json", dest="json_path", help="write the JSON report here")
    p.add_argument("--verbose", action="store_true",
                   help="include per-check evidence and language signals")
    p.add_argument("--samples", type=int, default=1,
                   help="generations per scenario (sampling is stochastic)")
    p.add_argument("--no-hybrid", action="store_true",
                   help="disable hybrid retrieval (lexical only)")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="override the generation budget")
    p.add_argument("--keep-db", action="store_true",
                   help="keep each scenario database for inspection")
    p.add_argument("--failures-out", help="write reusable failure records here")
    return p


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _runtime_info(runtime: T1Runtime, args: argparse.Namespace) -> dict:
    info = dict(RUNTIME_INFO)
    info.update({
        "temperature": runtime.config.temperature,
        "top_p": runtime.config.top_p,
        "max_new_tokens": runtime.config.max_new_tokens,
        "context_ceiling": runtime.config.context_ceiling,
        "hybrid_retrieval": not args.no_hybrid,
        "samples_per_scenario": args.samples,
        "warmup_s": round(runtime.warmup_s or 0.0, 1),
    })
    return info


def _guard_production(label: str, before: str) -> str:
    after = production_data_digest()
    if after != before:
        detail = production_data_fingerprint()
        raise T1IsolationError(
            f"production memory changed during {label}: {before} -> {after}\n"
            f"  fingerprint: {json.dumps(detail, indent=2)}"
        )
    return after


def cmd_list(args: argparse.Namespace) -> int:
    validate_corpus()
    print(f"{len(SCENARIOS)} T1 scenarios across {len(all_categories())} categories\n")
    current = None
    for s in SCENARIOS:
        if s.category != current:
            current = s.category
            print(f"  [{current}]")
        required = sum(1 for e in s.expectations if e.severity == "required")
        preferred = len(s.expectations) - required
        print(f"    {s.id:<4} {s.title}")
        print(f"         memory={s.memory.label}  checks={required} required, "
              f"{preferred} preferred")
    print(f"\n  failure taxonomy ({len(ALL_FAILURE_TYPES)}): "
          f"{', '.join(ALL_FAILURE_TYPES)}")
    return 0


# --------------------------------------------------------------------------
# Automated mode
# --------------------------------------------------------------------------


def run_auto(args: argparse.Namespace) -> int:
    validate_corpus()
    scenarios = load_scenarios(args.category, args.scenario)
    if not scenarios:
        print("no scenarios matched the selection", file=sys.stderr)
        return 2

    prod_before = production_data_digest()
    print(f"[t1] production memory digest before : {prod_before}")

    runtime = T1Runtime(
        hybrid_retrieval=not args.no_hybrid,
        max_new_tokens=args.max_new_tokens,
    )
    results = []
    notes = [
        "Automated verdicts come from an inspectable pattern rubric, not from "
        "a judge model and not from exact-wording matching.",
        "AMBIGUOUS means a required check passed but a preferred signal did "
        "not. Adjudicate those with --interactive.",
        "Sampling is stochastic (do_sample=True). Use --samples > 1 to measure "
        "stability; a single sample is not a stability estimate.",
    ]
    try:
        runtime.warmup(verbose=True)
        print(f"[t1] running {len(scenarios)} scenario(s), "
              f"{args.samples} sample(s) each")

        for index, scenario in enumerate(scenarios, start=1):
            print(f"[t1] ({index}/{len(scenarios)}) {scenario.id} "
                  f"{scenario.category}: {scenario.title}")
            for sample in range(args.samples):
                world = runtime.build_scenario(scenario, keep=args.keep_db)
                try:
                    turn = world.ask(scenario.prompt, scenario.history)
                    verdict = evaluate(scenario, turn.content)
                    result = build_result(
                        scenario=scenario,
                        prompt=scenario.prompt,
                        response=turn.content,
                        evaluation=verdict,
                        latency_s=turn.latency_s,
                        sample_index=sample,
                        error=turn.error,
                        memory_label=scenario.memory.label,
                    )
                    results.append(result)
                    counts = world.row_counts()
                    print(f"      {verdict.verdict:<9} "
                          f"({turn.latency_s:.1f}s) "
                          f"{_short(turn.content, 70)}")
                    if args.verbose:
                        print(f"      isolated db rows: {counts}")
                    if turn.error:
                        print(f"      error: {turn.error}")
                finally:
                    world.dispose()
            _guard_production(f"scenario {scenario.id}", prod_before)
    finally:
        runtime.close()

    report = InterrogationReport(
        model=runtime.model_name() if runtime._brain else "unloaded",
        adapter=Path(runtime.config.adapter_path).name,
        generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
        runtime=_runtime_info(runtime, args),
        results=results,
        notes=notes,
    )
    _emit(report, args)
    _guard_production("automated run", prod_before)
    print(f"\n[t1] production memory digest after  : {production_data_digest()} (unchanged)")
    return 0 if report.counts()[FAIL] == 0 else 1


def _short(text: str, limit: int) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# --------------------------------------------------------------------------
# Interactive mode
# --------------------------------------------------------------------------

_VERDICT_ALIASES = {
    "p": PASS, "pass": PASS, "y": PASS, "yes": PASS, "1": PASS,
    "f": FAIL, "fail": FAIL, "n": FAIL, "no": FAIL, "0": FAIL,
    "a": AMBIGUOUS, "amb": AMBIGUOUS, "ambiguous": AMBIGUOUS, "?": AMBIGUOUS,
}


def _parse_verdict(raw: str) -> Optional[str]:
    key = raw.strip().lower()
    if not key:
        return None
    return _VERDICT_ALIASES.get(key) or _VERDICT_ALIASES.get(key[0])


def run_interactive(args: argparse.Namespace) -> int:
    validate_corpus()
    scenarios = load_scenarios(args.category, args.scenario)
    if not scenarios:
        print("no scenarios matched the selection", file=sys.stderr)
        return 2

    prod_before = production_data_digest()
    print(f"[t1] production memory digest before : {prod_before}")
    print("[t1] interactive mode. Verdicts are yours; the rubric only advises.")
    print("[t1] commands: p/f/a + optional note | "
          "s = show rubric | r = re-run | n = next | q = quit")

    runtime = T1Runtime(
        hybrid_retrieval=not args.no_hybrid,
        max_new_tokens=args.max_new_tokens,
    )
    results = []
    try:
        runtime.warmup(verbose=True)
        for index, scenario in enumerate(scenarios, start=1):
            print()
            print("=" * 74)
            print(f"T1 Scenario {index}/{len(scenarios)}  "
                  f"[{scenario.id}]  ({scenario.category})")
            print("=" * 74)
            print(f"  title    : {scenario.title}")
            print(f"  memory   : {scenario.memory.label}")
            if scenario.history:
                for turn in scenario.history:
                    print(f"  prior    : {turn['role']}: "
                          f"{_short(turn['content'], 70)}")
            print(f"  prompt   : {scenario.prompt}")
            print()
            print("  expected behaviour:")
            for line in expected_behavior_text(scenario).splitlines():
                print(f"  {line}")
            print()

            world = runtime.build_scenario(scenario, keep=args.keep_db)
            turn = None
            try:
                while True:
                    print("  press Enter to send the prompt, or:")
                    print("    s = show rubric   r = re-run   q = quit")
                    cmd = input("  > ").strip().lower()
                    if cmd == "q":
                        print("\n[t1] quit.")
                        return _finish_interactive(
                            runtime, results, args, prod_before)
                    if cmd == "s":
                        print(expected_behavior_text(scenario))
                        continue
                    if cmd == "n":
                        turn = None
                        break
                    if cmd == "r":
                        turn = None
                    elif cmd:
                        print("  (unrecognised; sending the prompt)")
                    turn = world.ask(scenario.prompt, scenario.history)
                    break

                if turn is None:
                    continue

                print()
                print("  " + "-" * 70)
                print(f"  Aurora ({turn.latency_s:.1f}s):")
                for line in (turn.content or turn.error or "").splitlines() or [""]:
                    print(f"    {line}")
                print("  " + "-" * 70)

                verdict = evaluate(scenario, turn.content)
                print(f"  rubric suggests : {verdict.verdict}")
                print(f"  observed        : {verdict.observed}")
                if verdict.failure_type:
                    print(f"  failure type    : {verdict.failure_type}")
                if args.verbose:
                    for c in verdict.checks:
                        flag = "ok  " if c.passed else "MISS"
                        print(f"    {flag} {c.key} ({c.severity}) "
                              f"evidence={c.evidence}")

                result = build_result(
                    scenario=scenario, prompt=scenario.prompt,
                    response=turn.content, evaluation=verdict,
                    latency_s=turn.latency_s,
                    error=turn.error, memory_label=scenario.memory.label,
                )
                while True:
                    raw = input("  your verdict [p/f/a, or Enter to accept "
                                "rubric]: ").strip()
                    human = _parse_verdict(raw)
                    if human is None and raw == "":
                        human = verdict.verdict
                    if human is not None:
                        break
                    print("  please answer p, f, a, or press Enter")
                note = input("  note (optional): ").strip()
                result.human_verdict = human
                result.human_note = note
                results.append(result)
                print(f"  recorded: {human}"
                      + (f" — {note}" if note else ""))
            finally:
                world.dispose()
    finally:
        runtime.close()

    return _finish_interactive(runtime, results, args, prod_before)


def _finish_interactive(
    runtime: T1Runtime, results: list, args: argparse.Namespace, prod_before: str,
) -> int:
    report = InterrogationReport(
        model=runtime.model_name() if runtime._brain else "unloaded",
        adapter=Path(runtime.config.adapter_path).name,
        generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
        runtime=_runtime_info(runtime, args),
        results=results,
        notes=["Interactive run. human_verdict overrides the rubric where set."],
    )
    _emit(report, args)
    _guard_production("interactive run", prod_before)
    print(f"\n[t1] production memory digest after  : {production_data_digest()} (unchanged)")
    return 0


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def _emit(report: InterrogationReport, args: argparse.Namespace) -> None:
    print()
    print(report.to_text(verbose=args.verbose))
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.to_json(), encoding="utf-8")
        print(f"\n[t1] JSON report written to {path}")
    if args.failures_out:
        _emit_failures(report, Path(args.failures_out))


def _emit_failures(report: InterrogationReport, path: Path) -> None:
    """Reusable failure records for future training-data authoring.

    NOT IMPLEMENTED: automatic conversion into training examples. This only
    captures the material a human needs to write one deliberately.
    """
    import json

    records = []
    for r in report.results:
        if r.effective_verdict != FAIL:
            continue
        records.append({
            "scenario_id": r.scenario_id,
            "category": r.category,
            "title": r.title,
            "intent": r.intent,
            "prompt": r.prompt,
            "memory": r.memory_label,
            "aurora_response": r.response,
            "expected_behavior": r.expected_behavior,
            "observed_behavior": r.observed_behavior,
            "failure_type": r.failure_type,
            "dimensions_failed": r.dimensions_failed,
            "model": report.model,
            "adapter": report.adapter,
            "generated_at": report.generated_at,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"[t1] {len(records)} failure record(s) written to {path}")


# --------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        return cmd_list(args)
    if args.interactive:
        return run_interactive(args)
    if args.auto:
        return run_auto(args)
    build_parser().print_help()
    print("\nChoose one of --auto, --interactive, or --list.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
