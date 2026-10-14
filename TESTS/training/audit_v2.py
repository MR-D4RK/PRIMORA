"""Audit the generated Aurora Brain v0.1 dataset.

Runs after generation and reports what it finds rather than assuming it is
clean. The duplicate, near-duplicate, contradiction and template-concentration
checks are the ones that matter most here: the old corpus shipped 3,400
examples built from 1,506 distinct prompts, so "it generated" was never evidence
of a usable dataset.

Holdout handling: the holdout split is counted and checked for identity, but its
prompts are never opened for wording comparison. Auditing it against train is
exactly the contamination the holdout exists to prevent, so near-duplicate
scans deliberately exclude it.

Run: python TESTS/training/audit_v2.py
"""

from __future__ import annotations

import collections
import difflib
import json
import re
import os
import sys

sys.path.insert(0, r"C:\PRIMORA PROJECT")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from TESTS.training.design_v2 import (  # noqa: E402
    CATEGORY_BY_KEY,
    SEEDS,
    SEED_BUDGET,
    TRAIN_MAX,
    VALIDATION_SIZE,
    HOLDOUT_SIZE,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "v2")
REPORT_PATH = os.path.join(DATA_DIR, "audit_report.json")

NEAR_DUP_THRESHOLD = 0.90
MAX_TARGET_TEMPLATE_SHARE = 0.25

# Matches the runtime parser in PRIMORA/aurora/aurora.py.
BLOCK = re.compile(r"<prime_request>(.*?)</prime_request>", re.DOTALL)


def load(split: str) -> list[dict]:
    path = os.path.join(DATA_DIR, f"{split}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalise(text: str) -> str:
    return " ".join("".join(c.lower() if c.isalnum() or c == " " else " "
                             for c in text).split())


def first_user(rec: dict) -> str:
    for m in rec["messages"]:
        if m["role"] == "user":
            return m["content"]
    return ""


def final_assistant(rec: dict) -> str:
    for m in reversed(rec["messages"]):
        if m["role"] == "assistant":
            return m["content"]
    return ""


def check_tool_representation(splits: dict) -> dict:
    """Every tool turn must be the runtime's own shape.

    The runtime builds a tool turn as json.dumps({"status","output","error"})
    and the Qwen3 template supplies <tool_response>. An earlier dataset wrapped
    the body in <tool_result>, which nested a second tag pair inside
    <tool_response> - a structure the runtime has never produced.
    """
    findings: list[str] = []
    counts = {"success": 0, "failure": 0}
    seen_ids: set[str] = set()

    for split, rows in splits.items():
        for rec in rows:
            for i, m in enumerate(rec["messages"]):
                if m.get("role") != "tool":
                    continue
                content = m.get("content", "")
                if "<tool_result>" in content or "<tool_response>" in content:
                    findings.append(
                        f"{rec['id']} message {i}: ChatML markup inside tool content")
                    continue
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    findings.append(f"{rec['id']} message {i}: tool content is not JSON")
                    continue
                if not isinstance(parsed, dict):
                    findings.append(f"{rec['id']} message {i}: tool JSON is not an object")
                    continue
                if list(parsed.keys()) != ["status", "output", "error"]:
                    findings.append(
                        f"{rec['id']} message {i}: keys {list(parsed.keys())} are not "
                        f"the runtime order ['status','output','error']")
                    continue
                if parsed["status"] not in ("success", "failure"):
                    findings.append(
                        f"{rec['id']} message {i}: status {parsed['status']!r} is not a "
                        f"runtime RequestStatus value")
                    continue
                if parsed["status"] == "failure" and not parsed["error"]:
                    findings.append(f"{rec['id']} message {i}: failure with no error text")
                    continue
                counts[parsed["status"]] += 1
                seen_ids.add(rec["id"])

    return {
        "tool_turns": counts["success"] + counts["failure"],
        "success": counts["success"],
        "failure": counts["failure"],
        "records_with_tool_turns": len(seen_ids),
        "findings": findings[:20],
        "finding_count": len(findings),
    }


# A capability payload is a topic, not a sentence. These are the shapes that
# mean the query extractor has drifted back into returning user prose.
_SENTINEL = ("?", " would you", " could you", " do you ", " did you ",
             " shall we", " is it ", " are we ")


def check_payload_sanity(splits: dict) -> dict:
    """Capability arguments must be well-formed and topic-shaped."""
    findings: list[str] = []
    checked = 0

    for split, rows in splits.items():
        for rec in rows:
            for m in rec["messages"]:
                if m.get("role") != "assistant":
                    continue
                content = m.get("content", "")
                if "<prime_request>" not in content:
                    continue
                checked += 1
                match = BLOCK.search(content)
                if not match:
                    findings.append(f"{rec['id']}: request block does not parse")
                    continue
                try:
                    request = json.loads(match.group(1).strip())
                except json.JSONDecodeError as exc:
                    findings.append(f"{rec['id']}: request JSON invalid ({exc})")
                    continue
                payload = request.get("payload")
                if not isinstance(payload, dict):
                    findings.append(f"{rec['id']}: payload is not an object")
                    continue
                for key, value in payload.items():
                    if not isinstance(value, str):
                        continue
                    lowered = f" {value.lower()} "
                    if any(s in lowered or s in value for s in _SENTINEL):
                        findings.append(
                            f"{rec['id']}: payload[{key!r}] reads as a sentence: "
                            f"{value[:60]!r}")
                    # A noun jammed onto the end of a clause: "would you the van".
                    if re.search(r"\b(would you|could you|do you|did you)\s+the\b",
                                 value.lower()):
                        findings.append(
                            f"{rec['id']}: payload[{key!r}] has a noun appended to "
                            f"a clause: {value[:60]!r}")
                    if re.search(r"\b(\w+)\s+\1\b", value.lower()):
                        findings.append(
                            f"{rec['id']}: payload[{key!r}] repeats a word: "
                            f"{value[:60]!r}")

    return {
        "requests_checked": checked,
        "finding_count": len(findings),
        "findings": findings[:20],
    }


def audit() -> dict:
    splits = {s: load(s) for s in ("train", "validation", "holdout")}
    train, validation, holdout = splits["train"], splits["validation"], splits["holdout"]
    report: dict = {
        "counts": {s: len(v) for s, v in splits.items()},
        "total": sum(len(v) for v in splits.values()),
    }

    # -- identity --------------------------------------------------------
    ids = [r["id"] for r in train + validation + holdout]
    report["identity"] = {
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "invalid_categories": sorted({r["category"] for r in train + validation + holdout
                                      if r["category"] not in CATEGORY_BY_KEY}),
        "split_mismatches": sum(
            1 for r in train if r["split"] != "train")
            + sum(1 for r in validation if r["split"] != "validation")
            + sum(1 for r in holdout if r["split"] != "holdout"),
        "valid_splits": all(
            r["split"] in ("train", "validation", "holdout")
            for r in train + validation + holdout),
    }

    seed_ids = {s.seed_id for s in SEEDS}
    covered = {r["seed_id"] for r in train + validation + holdout}
    anchors = {s.seed_id for s in SEEDS if not s.training_allowed}
    report["seed_coverage"] = {
        "seeds_approved": len(seed_ids),
        "seeds_covered": len(covered & seed_ids),
        "missing_seeds": sorted(seed_ids - covered),
        "anchors_correctly_absent": sorted(anchors - covered),
        "unknown_seed_ids": sorted(covered - seed_ids),
    }

    # -- duplicates ------------------------------------------------------
    # Holdout excluded from near-duplicate comparison on purpose.
    open_pool = train + validation

    def key_fn(rec: dict) -> str:
        return normalise(first_user(rec))

    counts = collections.Counter(key_fn(r) for r in open_pool)
    report["exact_duplicate_prompts"] = {
        "groups": sum(1 for v in counts.values() if v > 1),
        "examples": sum(v for v in counts.values() if v > 1),
    }

    tgt = collections.Counter(normalise(final_assistant(r)) for r in open_pool)
    report["duplicate_final_targets"] = {
        "groups": sum(1 for v in tgt.values() if v > 1),
        "max_repeat": max(tgt.values()) if tgt else 0,
    }

    pair = collections.Counter(
        (key_fn(r), normalise(final_assistant(r))) for r in open_pool)
    report["duplicate_prompt_target_pairs"] = {
        "groups": sum(1 for v in pair.values() if v > 1),
    }

    # -- near duplicates (prompt text) -----------------------------------
    prompts = sorted({key_fn(r) for r in open_pool})
    near = []
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            ratio = difflib.SequenceMatcher(None, prompts[i], prompts[j]).ratio()
            if ratio >= NEAR_DUP_THRESHOLD:
                near.append({"a": prompts[i][:70], "b": prompts[j][:70],
                             "ratio": round(ratio, 3)})
    report["near_duplicate_prompts"] = {
        "threshold": NEAR_DUP_THRESHOLD,
        "distinct_prompts": len(prompts),
        "pairs": len(near),
        "examples": near[:12],
    }

    # -- template concentration -----------------------------------------
    shapes = collections.Counter()
    for rec in open_pool:
        shapes[rec["note"]] += 1
    worst_shapes = shapes.most_common(8)
    report["template_concentration"] = {
        "distinct_target_shapes": len(shapes),
        "largest_shape_share": round(
            max(shapes.values()) / len(open_pool), 3) if open_pool else 0,
        "limit": MAX_TARGET_TEMPLATE_SHARE,
        "over_limit": [
            {"note": n, "count": c,
             "share": round(c / len(open_pool), 3)}
            for n, c in worst_shapes
            if c / len(open_pool) > MAX_TARGET_TEMPLATE_SHARE],
    }

    openings = collections.Counter(
        normalise(first_user(r))[:18] for r in open_pool)
    report["opening_concentration"] = {
        "distinct_openings": len(openings),
        "most_common": openings.most_common(5),
    }

    # -- contradictions within a seed -----------------------------------
    # Two examples from one seed that both end in a request block must request
    # the same capability; disagreement means the seed teaches two behaviours.
    contradictions = []
    by_seed: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in open_pool:
        by_seed[rec["seed_id"]].append(rec)
    for sid, recs in by_seed.items():
        actions = {r.get("expect_action") for r in recs if r.get("expect_action")}
        if len(actions) > 1:
            contradictions.append({"seed_id": sid, "actions": sorted(actions)})
    report["contradictions"] = {
        "seeds_teaching_multiple_capabilities": len(contradictions),
        "examples": contradictions[:10],
    }

    report["tool_representation"] = check_tool_representation(splits)
    report["payload_sanity"] = check_payload_sanity(splits)

    # -- masking ---------------------------------------------------------
    mask_problems = []
    masked_seeds = {s.seed_id for s in SEEDS if s.train_on is not None}
    for rec in train + validation:
        assistants = [m for m in rec["messages"] if m["role"] == "assistant"]
        train_on = rec["train_on"] or []
        if not train_on:
            mask_problems.append(f"{rec['id']}: no train_on")
            continue
        if max(train_on) != len(assistants) - 1:
            mask_problems.append(
                f"{rec['id']}: loss ends at {max(train_on)} of {len(assistants)}")
        if rec["seed_id"] in masked_seeds and 0 in train_on:
            mask_problems.append(
                f"{rec['id']}: loss on the masked failure turn")
    report["masking"] = {
        "masked_seeds": sorted(masked_seeds),
        "examples_checked": len(train) + len(validation),
        "problems": mask_problems[:15],
        "problem_count": len(mask_problems),
    }

    # -- system context --------------------------------------------------
    fps = {r["system_fingerprint"] for r in train + validation + holdout}
    refs = {r["system_ref"] for r in train + validation + holdout}
    report["system_context"] = {
        "distinct_fingerprints": len(fps),
        "fingerprint": sorted(fps)[0][:32] if fps else None,
        "distinct_refs": sorted(refs),
        "all_present": all(r.get("system_fingerprint") for r in
                           train + validation + holdout),
    }

    # -- schema ----------------------------------------------------------
    required = {"id", "seed_id", "category", "split", "messages", "train_on",
                "system_ref", "system_fingerprint", "schema_version",
                "generator_version"}
    missing_fields = sum(
        1 for r in train + validation + holdout if not required.issubset(r))
    schema_versions = {r["schema_version"] for r in train + validation + holdout}
    report["schema"] = {
        "missing_required_fields": missing_fields,
        "schema_versions": sorted(schema_versions),
        "generator_versions": sorted(
            {r["generator_version"] for r in train + validation + holdout}),
    }

    # -- split discipline ------------------------------------------------
    seed_splits: dict[str, set[str]] = collections.defaultdict(set)
    for rec in train + validation + holdout:
        seed_splits[rec["seed_id"]].add(rec["split"])
    report["split_discipline"] = {
        "seeds_straddling_splits": sorted(
            k for k, v in seed_splits.items() if len(v) > 1),
    }

    # -- balance ---------------------------------------------------------
    report["by_category"] = {
        cat: dict(collections.Counter(
            r["split"] for r in train + validation + holdout if r["category"] == cat))
        for cat in CATEGORY_BY_KEY
    }
    report["by_seed"] = {
        sid: sum(1 for r in train + validation + holdout if r["seed_id"] == sid)
        for sid in sorted(covered)
    }

    report["targets"] = {
        "train": 1400, "validation": VALIDATION_SIZE,
        "holdout": HOLDOUT_SIZE, "train_max": TRAIN_MAX,
        "seed_budget": SEED_BUDGET,
    }
    return report


def main() -> int:
    report = audit()
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)

    c = report["counts"]
    print("=" * 72)
    print("AURORA BRAIN v0.1 - DATASET AUDIT")
    print("=" * 72)
    print(f"  total        : {report['total']}")
    print(f"  train        : {c['train']}")
    print(f"  validation   : {c['validation']}")
    print(f"  holdout      : {c['holdout']}  (protected)")
    print()
    ident = report["identity"]
    print(f"  unique ids       : {ident['unique_ids']} (dupes {ident['duplicate_ids']})")
    print(f"  invalid categories: {ident['invalid_categories'] or 'none'}")
    print(f"  split mismatches : {ident['split_mismatches']}")
    print()
    cov = report["seed_coverage"]
    print(f"  seeds covered    : {cov['seeds_covered']}/{cov['seeds_approved']}")
    print(f"  anchors absent   : {cov['anchors_correctly_absent']}")
    print(f"  missing seeds    : {cov['missing_seeds'] or 'none'}")
    print()
    print(f"  exact duplicate prompts  : "
          f"{report['exact_duplicate_prompts']['groups']} groups")
    print(f"  duplicate prompt+target  : "
          f"{report['duplicate_prompt_target_pairs']['groups']} groups")
    print(f"  near-duplicate prompts   : "
          f"{report['near_duplicate_prompts']['pairs']} pairs "
          f"(threshold {NEAR_DUP_THRESHOLD})")
    print(f"  max repeat of one target : "
          f"{report['duplicate_final_targets']['max_repeat']}")
    print()
    tc = report["template_concentration"]
    print(f"  distinct target shapes   : {tc['distinct_target_shapes']}")
    print(f"  largest shape share      : {tc['largest_shape_share']} "
          f"(limit {tc['limit']})")
    print(f"  over-limit shapes        : {len(tc['over_limit'])}")
    tr = report["tool_representation"]
    print(f"  tool turns (runtime JSON) : {tr['tool_turns']} in "
          f"{tr['records_with_tool_turns']} records "
          f"(success {tr['success']} / failure {tr['failure']})")
    print(f"  tool representation issues: {tr['finding_count']}")
    for f in tr["findings"][:5]:
        print(f"      TOOL {f}")
    ps = report["payload_sanity"]
    print(f"  requests payload-checked : {ps['requests_checked']}")
    print(f"  payload sanity issues    : {ps['finding_count']}")
    for f in ps["findings"][:5]:
        print(f"      PAYLOAD {f}")
    print(f"  contradictions           : "
          f"{report['contradictions']['seeds_teaching_multiple_capabilities']}")
    print()
    m = report["masking"]
    print(f"  masking problems         : {m['problem_count']} "
          f"(over {m['examples_checked']} examples)")
    print(f"  system fingerprints      : "
          f"{report['system_context']['distinct_fingerprints']} "
          f"({report['system_context']['fingerprint']})")
    print(f"  schema missing fields    : "
          f"{report['schema']['missing_required_fields']}")
    print(f"  seeds straddling splits  : "
          f"{len(report['split_discipline']['seeds_straddling_splits'])}")
    print()
    print(f"  report written to        : {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
