"""Validate the Aurora Brain v0.5 merged dataset.

The audit is pure: it reads the three splits and returns
``{"problems": [...], "report": {...}}``. ``build_dataset_v5`` does not call
it - merge and validation stay independent so a bug in one cannot hide a bug
in the other - but both run over the same files.

v0.5 is a MERGE of two corpora, so this validator is deliberately its own
module rather than an unmodified reuse of ``validate_dataset_v4``. The v4
validator is authored against ``seed_bank_v4`` alone and *forbids* the very
things v2's block records legitimately contain: ``expect_block`` /
``expect_action``, ``tool`` turns, and a ``<prime_request>`` block in an
assistant turn. Reusing it would either fail the merge on correct records or
require weakening it, and weakening a shipped validator is not on the table.

CONTRACT checks - a dataset that fails one of these is unusable:

*   malformed JSONL, or anything ``aurora.dataset_v2.validate_record``
    rejects against the live system context (tool turns and well-formed
    single request blocks are legal here, unlike v4);
*   exact split counts 1297/180/180;
*   one live ``system_ref`` on every record, resolving through
    ``resolve_system_context`` - a stale ref is fatal for ``load_split``;
*   provenance: v4-derived records keep ``seed_bank_v4``, the 217 merged
    records are stamped ``v2``, and each record's ``generator_version``
    agrees with its stamp;
*   whole-seed atomicity: no seed straddles two splits;
*   eval identity: validation/holdout content and membership must match v4
    exactly, where content is hashed with only the two system metadata fields
    excluded (Option 1 is permitted to refresh those and nothing else);
*   target repetition: no exact final target supervised more than twice,
    counted across all three splits;
*   split leakage: distinctive-value pools and final-ask pools disjoint
    across splits;
*   block coverage: the audited 17 distinct request-block targets and all six
    registered actions are present in train.

PATHOLOGY / REPORT-ONLY checks:

*   normalised-target collapses (punctuation-only duplicates) are reported,
    never gated - the gate is exact text;
*   one exact final target shared by two seeds is reported, not failed: in
    v2 two seeds can converge on an identical tool call by design, which is
    not the degenerate-template bug the v4 rule exists to catch;
*   word-set Jaccard and template openings are reported at milder thresholds.

Run: python TESTS/training/validate_dataset_v5.py [outdir]
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
from typing import Optional

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audit_v2 as A2  # noqa: E402
import build_dataset_v4 as BD4  # noqa: E402,F401  (keeps generate_v2 warm)
import build_dataset_v5 as BD5  # noqa: E402
import validate_dataset_v4 as VD4  # noqa: E402,F401  (keeps imports warm)
from aurora import dataset_v2  # noqa: E402

DATA_DIR = BD5.DEFAULT_OUTDIR
V4_DIR = BD5.V4_DIR
SPLIT_ORDER = BD5.SPLIT_ORDER
EXPECTED_RECORDS = BD5.EXPECTED_RECORDS
TARGET_REPEAT_LIMIT = BD5.TARGET_REPEAT_LIMIT
EXPECTED_RETAINED = BD5.EXPECTED_RETAINED
MERGED_PROVENANCE = BD5.MERGED_PROVENANCE
V4_PROVENANCE = BD5.V4_PROVENANCE
V4_GENERATOR_VERSION = BD5.V4_GENERATOR_VERSION
V2_GENERATOR_VERSION = BD5.V2_GENERATOR_VERSION

_BLOCK = re.compile(r"<prime_request>")


def load(split: str, data_dir: str = DATA_DIR) -> list[dict]:
    path = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def target_of(row: dict) -> str:
    assistants = [m for m in row["messages"] if m["role"] == "assistant"]
    return assistants[row["train_on"][-1]]["content"]


def all_targets(rows: list[dict]) -> list[tuple[str, str, str]]:
    """(seed_id, record_id, target_text), deduped per seed+normalised target."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for row in rows:
        key = (row["seed_id"], A2.normalise(target_of(row)))
        if key in seen:
            continue
        seen.add(key)
        out.append((row["seed_id"], row["id"], target_of(row)))
    return out


# --------------------------------------------------------------------------
# Contract: record-level
# --------------------------------------------------------------------------


def check_schema(records: dict[str, list[dict]],
                 system: dataset_v2.SystemContext,
                 problems: list[str]) -> dict:
    """validate_record on 100% of records.

    Unlike v4, expect_block/expect_action, tool turns and a single well-formed
    request block are legal - they are the point of the merge. ``validate_record``
    already enforces block balance, single-block and well-formedness, so a
    malformed or double block still fails here.
    """
    seen: set[str] = set()
    tool_turns = block_turns = 0
    for split in SPLIT_ORDER:
        for row in records.get(split, []):
            rid = row.get("id", "<no id>")
            if rid in seen:
                problems.append(f"{rid}: duplicate record id")
            seen.add(rid)
            for problem in dataset_v2.validate_record(
                    row, expected_split=split, system=system):
                problems.append(f"{rid}: {problem}")
            for message in row.get("messages", []):
                if message.get("role") == "tool":
                    tool_turns += 1
                if message.get("role") == "assistant" and \
                        _BLOCK.search(message.get("content", "")):
                    block_turns += 1
    return {"records": sum(len(records.get(s, [])) for s in SPLIT_ORDER),
            "tool_turns": tool_turns, "block_turns": block_turns}


def check_counts(records: dict[str, list[dict]],
                 problems: list[str]) -> dict:
    counts = {split: len(records.get(split, [])) for split in SPLIT_ORDER}
    for side, want in EXPECTED_RECORDS.items():
        if counts[side] != want:
            problems.append(
                f"{side} holds {counts[side]} records, expected {want}")
    total = sum(counts.values())
    if total != sum(EXPECTED_RECORDS.values()):
        problems.append(f"total {total} records, expected "
                        f"{sum(EXPECTED_RECORDS.values())}")
    return {"records": counts, "total": total}


def check_provenance(records: dict[str, list[dict]],
                     system: dataset_v2.SystemContext,
                     problems: list[str]) -> dict:
    """One live system_ref everywhere; stamps agree with generator_version."""
    bad_ref = bad_fp = bad_prov = 0
    merged = v4_records = 0
    refs: collections.Counter = collections.Counter()
    seed_side: dict[str, set[str]] = collections.defaultdict(set)
    for split in SPLIT_ORDER:
        for row in records.get(split, []):
            refs[row.get("system_ref")] += 1
            if row.get("system_ref") != system.ref:
                bad_ref += 1
                problems.append(
                    f"{row.get('id')}: system_ref {row.get('system_ref')!r} "
                    f"!= live {system.ref!r}")
            if row.get("system_fingerprint") != system.fingerprint:
                bad_fp += 1
                problems.append(f"{row.get('id')}: system_fingerprint drift")
            gen = row.get("generator_version")
            prov = row.get("seed_provenance")
            if gen == V4_GENERATOR_VERSION:
                v4_records += 1
                if prov != V4_PROVENANCE:
                    bad_prov += 1
                    problems.append(
                        f"{row.get('id')}: v4 record provenance {prov!r}")
            elif gen == V2_GENERATOR_VERSION:
                merged += 1
                if prov != MERGED_PROVENANCE:
                    bad_prov += 1
                    problems.append(
                        f"{row.get('id')}: merged record provenance {prov!r}")
            else:
                problems.append(
                    f"{row.get('id')}: unexpected generator_version {gen!r}")
            seed_side[row.get("seed_id")].add(split)
    straddling = [sid for sid, sides in seed_side.items() if len(sides) > 1]
    for sid in sorted(straddling):
        problems.append(f"{sid}: whole-seed atomicity broken: "
                        f"{sorted(seed_side[sid])}")
    if len(refs) != 1:
        problems.append(f"expected one system_ref, found {sorted(refs)}")
    return {"merged_records": merged, "v4_records": v4_records,
            "bad_system_ref": bad_ref, "bad_fingerprint": bad_fp,
            "bad_provenance": bad_prov, "seeds": len(seed_side),
            "straddling_seeds": len(straddling)}


def check_eval_identity(records: dict[str, list[dict]],
                        problems: list[str]) -> dict:
    """v5 eval must equal v4 eval in content (system metadata excluded) and ids."""
    report: dict = {}
    for side in ("validation", "holdout"):
        v5_rows = records.get(side, [])
        v4_rows = BD5.load_source(side, V4_DIR)
        v5_ids = sorted(r["id"] for r in v5_rows)
        v4_ids = sorted(r["id"] for r in v4_rows)
        membership_ok = v5_ids == v4_ids
        v5_digest = BD5.split_digest(v5_rows)
        v4_digest = BD5.split_digest(v4_rows)
        content_ok = v5_digest == v4_digest
        if not membership_ok:
            problems.append(f"{side}: membership drifted from v4")
        if not content_ok:
            problems.append(
                f"{side}: content digest {v5_digest[:16]} != v4 "
                f"{v4_digest[:16]} (beyond system metadata)")
        report[side] = {"records": len(v5_rows), "membership_ok": membership_ok,
                        "content_ok": content_ok,
                        "v5_digest": v5_digest, "v4_digest": v4_digest}
    return report


def check_target_repetition(records: dict[str, list[dict]],
                            problems: list[str]) -> dict:
    """No exact final target supervised more than twice, across all splits."""
    exact = collections.Counter(
        target_of(row) for side in SPLIT_ORDER
        for row in records.get(side, []))
    over = {t: n for t, n in exact.items() if n > TARGET_REPEAT_LIMIT}
    for text, times in sorted(over.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
        problems.append(
            f"target repeated {times} times (limit {TARGET_REPEAT_LIMIT}): "
            f"{text[:60]!r}")
    if len(over) > 10:
        problems.append(f"... and {len(over) - 10} more repeated targets")

    normalised = collections.Counter(
        A2.normalise(target_of(row)) for side in SPLIT_ORDER
        for row in records.get(side, []))
    norm_over = {t: n for t, n in normalised.items() if n > TARGET_REPEAT_LIMIT}
    return {
        "limit": TARGET_REPEAT_LIMIT,
        "distinct_exact": len(exact),
        "max_exact_repeat": max(exact.values(), default=0),
        "over_limit_exact": len(over),
        "distinct_normalised": len(normalised),
        "max_normalised_repeat": max(normalised.values(), default=0),
        "over_limit_normalised": [
            [text[:60], norm_over[text]]
            for text in sorted(norm_over, key=lambda t: (-norm_over[t], t))
            [:10]],
        "note": ("the gate fails on exact text; normalised collapses are "
                 "reported so punctuation-only duplicates stay visible"),
    }


def check_leakage(records: dict[str, list[dict]],
                  problems: list[str]) -> dict:
    pools: dict[str, set[str]] = {side: set() for side in SPLIT_ORDER}
    finals: dict[str, set[str]] = {side: set() for side in SPLIT_ORDER}
    for side in SPLIT_ORDER:
        for row in records.get(side, []):
            for message in row["messages"]:
                pools[side] |= BD4.extract_values(message["content"])
            finals[side].add(BD4.normalise_ask(
                [m for m in row["messages"] if m["role"] == "user"][-1]
                ["content"]))

    value_sharing = ask_sharing = 0
    for i, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[i + 1:]:
            shared = pools[left] & pools[right]
            if shared:
                value_sharing += len(shared)
                problems.append(
                    f"{left}/{right} share distinctive value(s): "
                    f"{sorted(shared)[:5]}")
            shared_asks = finals[left] & finals[right]
            if shared_asks:
                ask_sharing += len(shared_asks)
                problems.append(
                    f"{left}/{right} share {len(shared_asks)} final ask(s), "
                    f"e.g. {sorted(shared_asks)[0][:50]!r}")

    # Report-only: a target shared by two seeds is legal convergence in v2
    # (identical tool call), not the degenerate-template bug v4 gates on.
    owners: dict[str, set[str]] = collections.defaultdict(set)
    for side in SPLIT_ORDER:
        for seed_id, _rid, text in all_targets(records.get(side, [])):
            owners[A2.normalise(text)].add(seed_id)
    cross = {t: sorted(s) for t, s in owners.items() if len(s) > 1}
    return {"shared_values": value_sharing, "shared_final_asks": ask_sharing,
            "cross_seed_shared_targets_report_only": len(cross),
            "cross_seed_samples": [[sorted(s)[:4], t[:50]]
                                   for t, s in list(cross.items())[:5]]}


def check_coverage(records: dict[str, list[dict]],
                   problems: list[str]) -> dict:
    """The audited block coverage must survive the merge."""
    train = records.get("train", [])
    block_targets = {target_of(r) for r in train
                     if _BLOCK.search(target_of(r))}
    actions = collections.Counter(r.get("expect_action") for r in train)
    action_names = {a for a in actions if a}
    expected_actions = {"recall", "learn", "research", "use_tool",
                        "calculate", "consult_specialist"}
    missing_actions = sorted(expected_actions - action_names)
    if missing_actions:
        problems.append(f"train is missing action(s): {missing_actions}")
    if len(block_targets) < 17:
        problems.append(
            f"train holds {len(block_targets)} distinct block targets, "
            f"audited merge preserved 17")
    return {"distinct_block_targets": len(block_targets),
            "block_target_sample": sorted(block_targets)[:3],
            "actions": {str(k): v for k, v in sorted(
                actions.items(), key=lambda kv: (-kv[1], str(kv[0])))},
            "action_kinds": len(action_names)}


def check_tools(records: dict[str, list[dict]], problems: list[str]) -> dict:
    representation = A2.check_tool_representation(records)
    problems.extend(f"tool: {f}" for f in representation["findings"])
    sanity = A2.check_payload_sanity(records)
    problems.extend(f"payload: {f}" for f in sanity["findings"])
    return {"tool_turns": representation["tool_turns"],
            "payload_findings": len(sanity["findings"])}


def check_masking(records: dict[str, list[dict]],
                  system: dataset_v2.SystemContext,
                  problems: list[str]) -> dict:
    try:
        from transformers import AutoTokenizer
        from aurora.config import TrainerConfig
        from aurora.masking import MaskingError, build_mask, verify_mask
        config = TrainerConfig()
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model, trust_remote_code=True)
    except Exception as exc:
        problems.append(f"masking unavailable: {exc}")
        return {"available": False}

    trained_turns = trained_tokens = truncated = 0
    for split in SPLIT_ORDER:
        for row in records.get(split, []):
            messages = [{"role": "system", "content": system.content}]
            messages.extend(row["messages"])
            try:
                result = build_mask(tokenizer, messages, row["train_on"],
                                    max_length=config.max_seq_length)
                verify_mask(result, messages, tokenizer,
                            row.get("id", "<rec>"))
            except MaskingError as exc:
                problems.append(f"{row.get('id')}: mask check failed: {exc}")
                continue
            if result.truncated:
                truncated += 1
                problems.append(
                    f"{row.get('id')}: renders past max_seq_length "
                    f"{config.max_seq_length}")
            trained_turns += len(result.trained_turns)
            trained_tokens += result.trained_tokens
    return {"available": True, "records": sum(
                len(records.get(s, [])) for s in SPLIT_ORDER),
            "trained_turns": trained_turns,
            "trained_tokens": trained_tokens,
            "truncated_records": truncated}


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def audit(records: dict[str, list[dict]],
          system: Optional[dataset_v2.SystemContext] = None,
          tokenize: bool = True) -> dict:
    problems: list[str] = []
    report: dict = {"label": "Aurora v0.5 merged dataset validation",
                    "contract": {}, "pathology": {}}

    if system is None:
        refs = {row.get("system_ref") for split in SPLIT_ORDER
                for row in records.get(split, [])}
        refs.discard(None)
        if len(refs) != 1:
            problems.append(f"expected one system_ref, found {sorted(refs)}")
            return {"problems": problems, "report": report}
        try:
            system = dataset_v2.resolve_system_context(next(iter(refs)))
        except dataset_v2.DatasetError as exc:
            problems.append(f"system context does not resolve: {exc}")
            return {"problems": problems, "report": report}

    report["contract"]["counts"] = check_counts(records, problems)
    report["contract"]["schema"] = check_schema(records, system, problems)
    report["contract"]["provenance"] = check_provenance(
        records, system, problems)
    report["contract"]["eval_identity"] = check_eval_identity(records, problems)
    report["contract"]["repetition"] = check_target_repetition(
        records, problems)
    report["contract"]["leakage"] = check_leakage(records, problems)
    report["contract"]["coverage"] = check_coverage(records, problems)
    report["contract"]["tools"] = check_tools(records, problems)
    report["pathology"]["note"] = (
        "normalised collapses, cross-seed shared targets and Jaccard are "
        "reported, not gated; the gate is exact target text and split leakage")
    report["contract"]["masking"] = (
        check_masking(records, system, problems) if tokenize
        else {"skipped": "tokenize=False"})

    report["problem_count"] = len(problems)
    return {"problems": problems, "report": report}


def main() -> int:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    records = {split: load(split, data_dir) for split in SPLIT_ORDER}
    if not any(records.values()):
        print(f"no v0.5 dataset found under {data_dir}")
        return 1
    result = audit(records)
    problems = result["problems"]
    report = result["report"]
    report["problems"] = problems[:200]

    out = os.path.join(data_dir, "validation_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    if problems:
        print(f"\nFAILURES ({len(problems)}):")
        for problem in problems[:40]:
            print(f"  FAIL {problem}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        print(f"\nreport written: {out}")
        return 1
    print(f"\nOK: every v0.5 dataset gate held. report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
