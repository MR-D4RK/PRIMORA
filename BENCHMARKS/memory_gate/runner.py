"""Disposable end-to-end Memory V2 smoke gate for the real Aurora CLI.

No dialogue classifier or PRIMORA fact assumptions live here. Expectations
belong to an external, replaceable evaluation fixture.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from contextlib import closing

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = Path(__file__).with_name("cases.json")
ANSWERS = re.compile(
    r"Aurora:\s*(.*?)(?=\n\s*(?:Timings:|You:|Goodbye!|ERROR:)|\Z)",
    re.DOTALL,
)
TIMINGS = re.compile(r"(?m)^Timings:\s*(\{[^\n]+\})")
ANCHORS_A = ("Larkspur", "Juniper", "Marigold", "Saffron", "Cedar", "Willow")
ANCHORS_B = ("Atlas", "Lantern", "Compass", "Notebook", "Cabinet", "Sketchbook")


def render(value, variables):
    if isinstance(value, str):
        for name, replacement in variables.items():
            value = value.replace("{{" + name + "}}", replacement)
    elif isinstance(value, list):
        value = [render(item, variables) for item in value]
    elif isinstance(value, dict):
        value = {key: render(item, variables) for key, item in value.items()}
    return value


def anchors(seed, case_index):
    rnd = random.Random(str(seed) + "/" + str(case_index))
    first = rnd.sample(ANCHORS_A, 2)
    second = rnd.sample(ANCHORS_B, 2)
    return {
        "object_a": first[0] + " " + second[0],
        "object_b": first[1] + " " + second[1],
    }


def bench_env(db_path, session_id):
    env = os.environ.copy()
    # Force disposable SQLite even if operator has a production DB selected.
    env.update({
        "PRIMORA_MEMORY_V2_DB": str(db_path.resolve()),
        "PRIMORA_MEMORY_V2": "1",
        "PRIMORA_MEMORY_V2_FORMATION": "1",
        "PRIMORA_MEMORY_V2_RETRIEVAL": "1",
        "PRIMORA_AURORA_MEMORY_INTERPRETER": "0",
        "PRIMORA_PROVIDER": "openai",
        "PRIMORA_AURORA_BACKEND": "openai",
        "PRIMORA_MODEL_SERVER": "off",
        "PRIMORA_RESEARCH_PROVIDER": "disabled",
        "PRIMORA_OPENAI_URL": env.get(
            "PRIMORA_OPENAI_URL", "http://127.0.0.1:8080/v1"),
        "PRIMORA_OPENAI_MODEL": env.get(
            "PRIMORA_OPENAI_MODEL", "Qwen3.5-9B-Q4_K_M"),
        "PRIMORA_OPENAI_ENABLE_THINKING": "0",
        "PRIMORA_JEV_ENABLE_THINKING": "0",
        "PRIMORA_MEMORY_FORMATION_ENABLE_THINKING": "0",
        "PRIMORA_CLI_TIMINGS": "1",
        "PRIMORA_SESSION_ID": session_id,
        "PYTHONUNBUFFERED": "1",
    })
    return env


def run_session(db_path, session_id, turns, timeout):
    prompts = [turn["prompt"] for turn in turns]
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "PRIMORA.main", "--cli"],
            input="\n".join(prompts + ["exit"]) + "\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=ROOT,
            env=bench_env(db_path, session_id),
            timeout=timeout,
            check=False,
        )
        output = proc.stdout
        answers = [x.strip() for x in ANSWERS.findall(output)]
        timing_rows = []
        for match in TIMINGS.findall(output):
            try:
                timing_rows.append(json.loads(match))
            except json.JSONDecodeError:
                timing_rows.append({})
        return {
            "returncode": proc.returncode, "answers": answers,
            "timings": timing_rows, "log": output,
            "stderr": proc.stderr[-4000:],
            "error": None if proc.returncode == 0 and len(answers) == len(turns)
                     else "CLI did not return one answer per input",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None, "answers": [], "timings": [],
            "log": "", "stderr": "", "error": "session_timeout: " + str(exc),
        }


def judge(turn, answer):
    """Lexical probes only; semantic accuracy still requires human review."""
    required = turn.get("expect_contains", [])
    forbidden = turn.get("expect_not_contains", [])
    if not required and not forbidden:
        return None
    lowered = answer.casefold()
    return (all(item.casefold() in lowered for item in required)
            and all(item.casefold() not in lowered for item in forbidden))


def source_audit(db_path):
    """Read-only integrity check; do not confuse integrity with recall quality."""
    if not db_path.is_file():
        return {"error": "database_not_created"}
    uri = db_path.resolve().as_uri() + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as con:
            integrity = con.execute("PRAGMA quick_check").fetchone()[0]
            source_roles = dict(con.execute("SELECT id, role FROM messages"))
            rows = con.execute(
                "SELECT id, authority, source_ids, concept, value, "
                "status, supersedes_id FROM states ORDER BY id").fetchall()
            violations = []
            state_summary = []
            for (state_id, authority, source_json, concept, value,
                 status, supersedes_id) in rows:
                state_summary.append({
                    "id": state_id, "authority": authority,
                    "source_ids": source_json, "concept": concept,
                    "value": value, "status": status,
                    "supersedes_id": supersedes_id,
                })
                if authority != "user":
                    continue
                try:
                    # The store serializes state source IDs as CSV ("1,2"),
                    # not JSON. Match ContextStore._state_rows_to_records.
                    pieces = str(source_json or "").split(",")
                    if not pieces or not all(
                        piece.strip().isdecimal() and int(piece.strip()) > 0
                        for piece in pieces
                    ):
                        raise ValueError("missing or invalid source IDs")
                    ids = [int(piece.strip()) for piece in pieces]
                    for item in ids:
                        if source_roles.get(item) != "user":
                            violations.append({
                                "state_id": state_id, "source_id": item,
                                "error": "user-authority state lacks user source",
                            })
                except (ValueError, TypeError):
                    violations.append({
                        "state_id": state_id, "error": "unparseable source IDs",
                    })
            episodes = con.execute(
                "SELECT e.id FROM experience_events e LEFT JOIN messages m "
                "ON e.source_message_id=m.id WHERE e.authority='user' "
                "AND (m.id IS NULL OR m.role!='user')").fetchall()
            for (event_id,) in episodes:
                violations.append({
                    "event_id": event_id,
                    "error": "user-authority episode lacks user source",
                })
            return {
                "sqlite_quick_check": integrity, "message_count": len(source_roles),
                "state_count": len(rows), "state_summary": state_summary,
                "episode_integrity_violations": len(episodes),
                "violations": violations,
            }
    except (OSError, sqlite3.DatabaseError) as exc:
        return {"error": type(exc).__name__ + ": " + str(exc)}


def percentile(numbers, fraction):
    if not numbers:
        return None
    ordered = sorted(numbers)
    index = round((len(ordered) - 1) * fraction)
    return round(ordered[index], 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=Path("BENCHMARKS/memory_gate/runs"))
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--timeout", type=int, default=900,
                        help="Maximum seconds per CLI session")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and expand cases without starting Qwen")
    args = parser.parse_args()

    dataset = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or not dataset:
        parser.error("cases must be a nonempty JSON array")
    ids = [case["id"] for case in dataset]
    if len(set(ids)) != len(ids):
        parser.error("case IDs must be unique")
    if args.dry_run:
        for index, case in enumerate(dataset):
            expanded = render(case, anchors(args.seed, index))
            assert expanded["sessions"]
            for session in expanded["sessions"]:
                assert session["turns"]
                for turn in session["turns"]:
                    assert turn["prompt"] not in ("", "exit", "quit")
        print(json.dumps({
            "valid": True, "cases": len(dataset),
            "sessions": sum(len(c["sessions"]) for c in dataset),
            "seed": args.seed,
        }, indent=2))
        return 0

    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = args.out.resolve() / (run_name + "-" + str(args.seed))
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    timings = []
    probe_count = 0
    probe_passes = 0
    session_errors = 0

    for index, raw in enumerate(dataset):
        case = render(raw, anchors(args.seed, index))
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", case["id"])[:50]
        case_dir = directory / (str(index) + "-" + safe_id)
        case_dir.mkdir()
        db = case_dir / "synthetic-only.db"
        evaluated = {"case_id": case["id"], "sessions": []}
        for session_index, session in enumerate(case["sessions"]):
            identifier = ("gate-" + str(index) + "-" +
                          str(session_index) + "-" + str(args.seed))
            run = run_session(db, identifier, session["turns"], args.timeout)
            (case_dir / ("session-" + str(session_index) + ".log")).write_text(
                run["log"] + "\nSTDERR:\n" + run["stderr"], encoding="utf-8")
            if run["error"]:
                session_errors += 1
            answered = []
            for i, turn in enumerate(session["turns"]):
                answer = run["answers"][i] if i < len(run["answers"]) else ""
                score = judge(turn, answer) if answer else False
                if score is not None:
                    probe_count += 1
                    probe_passes += bool(score)
                answered.append({
                    "prompt": turn["prompt"], "answer": answer,
                    "lexical_probe_pass": score,
                    "human_review": "PENDING",
                    "expected_contains": turn.get("expect_contains", []),
                    "expected_not_contains": turn.get("expect_not_contains", []),
                })
            timings.extend(
                t.get("total_ms") for t in run["timings"]
                if isinstance(t.get("total_ms"), (int, float)))
            evaluated["sessions"].append({
                "id": identifier, "turns": answered, "error": run["error"],
                "timings_ms": run["timings"],
                "capability_log_lines": [
                    line for line in run["log"].splitlines()
                    if "[Prime] Executing Capability:" in line],
            })
        evaluated["source_audit"] = source_audit(db)
        results.append(evaluated)
        print(case["id"] + ": " +
              ("complete" if all(not s["error"] for s in evaluated["sessions"])
               else "needs investigation"), flush=True)

    summary = {
        "status": "HUMAN_REVIEW_REQUIRED",
        "repo_baseline": "2caf190", "seed": args.seed,
        "cases": len(dataset), "session_errors": session_errors,
        "lexical_probes": probe_count, "lexical_passes": probe_passes,
        "lexical_probe_rate": round(probe_passes / probe_count, 3)
        if probe_count else None,
        "latency_total_ms_p50": percentile(timings, .5),
        "latency_total_ms_p95": percentile(timings, .95),
        "claim": "Probes are smoke checks, NOT semantic accuracy or competitor scores.",
        "cases_detail": results,
    }
    outfile = directory / "results.json"
    outfile.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print("Results: " + str(outfile))
    print("Probe pass: " + str(probe_passes) + "/" + str(probe_count))
    print("Session infrastructure errors: " + str(session_errors))
    return 1 if session_errors or probe_count != probe_passes else 0


if __name__ == "__main__":
    raise SystemExit(main())
