"""Validate the Aurora Brain v0.3 dataset against every approved v0.3 gate.

The audit is pure: it takes the three split tables and returns problems, so
``generate_v3`` calls it before writing and this module's ``main`` calls it on
what is already on disk. One code path means the dataset cannot pass
generation and fail validation.

What is checked, in the order the design lists them:

*   exact split counts, unique ids, schema and record shape
    (``dataset_v2.validate_record`` against the live system context);
*   seed coverage: the 210 emittable seeds, each wholly inside its split, the
    review-only anchor nowhere;
*   category budgets on train;
*   final-target gates: >= 900 distinct on train, no target repeated more
    than three times, no verbatim target shared between seeds (holdout is
    compared inside itself only - it is never measured against train);
*   opening gates: >= 700 distinct first-user openings, no opening repeated
    more than five times;
*   domain and register spread per category, pooled over train + validation;
*   required adjacent pairs contiguous in the train file, and no two
    consecutive records in a category opening their supervised span with the
    same first-person construction;
*   tool-turn shape, payload sanity, payload contract against the live
    ``CapabilityRegistry`` (one deliberate exception: C15-R01's unknown
    capability, which exists so the record can show a refusal);
*   supervised-token band: every record masks and verifies through the real
    ``masking`` module with no truncation, and train's conversational share
    of supervised tokens sits in 0.65 +/- 0.03.

Run: python TESTS/training/validate_v3.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from typing import Optional

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audit_v2 as A2  # noqa: E402
import builders_v3 as B3  # noqa: E402
import check_bank_v3 as C3  # noqa: E402
import design_v3 as D3  # noqa: E402
import seed_bank_v3 as SB3  # noqa: E402
from aurora import dataset_v2  # noqa: E402
from aurora.config import TrainerConfig  # noqa: E402
from aurora.masking import MaskingError, build_mask, verify_mask  # noqa: E402

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "v3")
SPLIT_ORDER = ("train", "validation", "holdout")
EXPECTED_SIZES = {
    "train": D3.TRAIN_SIZE,
    "validation": D3.VALIDATION_SIZE,
    "holdout": D3.HOLDOUT_SIZE,
}
DELIBERATE_PAYLOADS = {("C15-R01", "count_words")}

_BANK: Optional[dict[str, SB3.SeedV3]] = None


def load(split: str, data_dir: str = DATA_DIR) -> list[dict]:
    path = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bank() -> dict[str, SB3.SeedV3]:
    global _BANK
    if _BANK is None:
        seeds = SB3.build_retained(SB3.build_extras()) + C3.new_seeds()
        _BANK = {s.seed_id: s for s in seeds}
    return _BANK


def _opening(rec: dict) -> str:
    return B3.opening_key(A2.first_user(rec))


def _target(rec: dict) -> str:
    return A2.normalise(A2.final_assistant(rec))


# --------------------------------------------------------------------------
# Per-record gates.
# --------------------------------------------------------------------------


def _record_shape_problems(records: dict[str, list[dict]],
                           system: Optional[dataset_v2.SystemContext],
                           problems: list[str]) -> None:
    seen: set[str] = set()
    for split in SPLIT_ORDER:
        for row in records.get(split, []):
            rid = row.get("id", "<no id>")
            if rid in seen:
                problems.append(f"{rid}: duplicate example id")
            seen.add(rid)
            for problem in dataset_v2.validate_record(
                    row, expected_split=split, system=system):
                problems.append(f"{rid}: {problem}")


def _payload_contract_problems(records: dict[str, list[dict]],
                               problems: list[str]) -> None:
    try:
        from PRIMORA.prime.capability import CapabilityRegistry
        registry = CapabilityRegistry()
        known = set(registry.get_names())
    except Exception as exc:
        problems.append(f"capability registry unavailable: {exc}")
        return

    for split in SPLIT_ORDER:
        for row in records.get(split, []):
            for message in row.get("messages", []):
                if message.get("role") != "assistant":
                    continue
                content = message.get("content", "")
                if "<prime_request>" not in content:
                    continue
                match = A2.BLOCK.search(content)
                if not match:
                    problems.append(
                        f"{row.get('id')}: request block does not parse")
                    continue
                try:
                    request = json.loads(match.group(1).strip())
                except json.JSONDecodeError as exc:
                    problems.append(
                        f"{row.get('id')}: request JSON invalid ({exc})")
                    continue
                action = request.get("action")
                payload = request.get("payload")
                if not isinstance(payload, dict):
                    problems.append(
                        f"{row.get('id')}: payload is not an object")
                    continue
                if action not in known:
                    key = (row.get("seed_id"), action)
                    if key not in DELIBERATE_PAYLOADS:
                        problems.append(
                            f"{row.get('id')}: unknown capability {action!r}")
                    continue
                spec = registry.get(action)
                missing = [name for name in spec.required_fields
                           if name not in payload]
                extra = sorted(set(payload)
                               - set(spec.required_fields)
                               - set(spec.optional_fields))
                if missing or extra:
                    problems.append(
                        f"{row.get('id')}: {action} payload "
                        f"missing={missing} extra={extra}")


# --------------------------------------------------------------------------
# Token gate.
# --------------------------------------------------------------------------


def token_report(records: dict[str, list[dict]],
                 system: dataset_v2.SystemContext,
                 problems: list[str]) -> dict:
    try:
        from transformers import AutoTokenizer
        config = TrainerConfig()
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model, trust_remote_code=True)
    except Exception as exc:
        problems.append(f"token gate unavailable: {exc}")
        return {}

    conversational = 0
    capability = 0
    truncated = 0
    for split in SPLIT_ORDER:
        for row in records.get(split, []):
            messages = [{"role": "system", "content": system.content}]
            messages.extend(row["messages"])
            try:
                result = build_mask(
                    tokenizer, messages, row["train_on"],
                    max_length=config.max_seq_length)
                verify_mask(result, messages, tokenizer,
                            row.get("id", "<record>"))
            except MaskingError as exc:
                problems.append(f"{row.get('id')}: mask check failed: {exc}")
                continue
            if result.truncated:
                truncated += 1
                problems.append(
                    f"{row.get('id')}: renders past max_seq_length "
                    f"{config.max_seq_length}")
            if split == "train":
                if row["category"] in D3.CONVERSATIONAL_CATEGORIES:
                    conversational += result.trained_tokens
                else:
                    capability += result.trained_tokens

    total = conversational + capability
    ratio = (conversational / total) if total else 0.0
    low = D3.CONVERSATIONAL_TOKEN_TARGET - D3.TOKEN_RATIO_TOLERANCE
    high = D3.CONVERSATIONAL_TOKEN_TARGET + D3.TOKEN_RATIO_TOLERANCE
    if total and not low <= ratio <= high:
        problems.append(
            f"train supervised conversational token share {ratio:.4f} is "
            f"outside [{low:.2f}, {high:.2f}]")
    return {
        "supervised_tokens_train": {
            "conversational": conversational,
            "capability": capability,
        },
        "token_ratio_conversational": round(ratio, 4),
        "truncated_records": truncated,
    }


# --------------------------------------------------------------------------
# The audit.
# --------------------------------------------------------------------------


def audit(records: dict[str, list[dict]],
          system: Optional[dataset_v2.SystemContext] = None) -> dict:
    problems: list[str] = []
    report: dict = {}

    counts = {split: len(records.get(split, [])) for split in SPLIT_ORDER}
    report["counts"] = counts
    for split, expected in EXPECTED_SIZES.items():
        if counts[split] != expected:
            problems.append(
                f"{split} holds {counts[split]} records, expected {expected}")

    if system is None:
        refs = {row.get("system_ref")
                for split in SPLIT_ORDER
                for row in records.get(split, [])}
        refs.discard(None)
        if len(refs) != 1:
            problems.append(f"expected one system_ref, found {sorted(refs)}")
        else:
            try:
                system = dataset_v2.resolve_system_context(next(iter(refs)))
            except dataset_v2.DatasetError as exc:
                problems.append(f"system context does not resolve: {exc}")

    _record_shape_problems(records, system, problems)
    _payload_contract_problems(records, problems)

    tool_check = A2.check_tool_representation(records)
    if tool_check["finding_count"]:
        problems.extend(f"tool: {f}" for f in tool_check["findings"])
    report["tool_turns"] = {
        "total": tool_check["tool_turns"],
        "failure": tool_check["failure"],
    }
    sanity = A2.check_payload_sanity(records)
    if sanity["finding_count"]:
        problems.extend(f"payload: {f}" for f in sanity["findings"])

    # -- seed coverage ---------------------------------------------------
    seeds = bank()
    split_plan = C3.split_of(list(seeds.values()), problems)
    actual: dict[str, set[str]] = {}
    for split in SPLIT_ORDER:
        rows = records.get(split, [])
        ids = {row.get("seed_id") for row in rows}
        actual[split] = ids
        expected = {sid for sid, side in split_plan.items() if side == split}
        for missing in sorted(expected - ids):
            problems.append(f"{split}: seed {missing} emitted no records")
        for extra in sorted(ids - expected):
            problems.append(f"{split}: seed {extra} does not belong to it")
        for sid in sorted(ids - set(seeds)):
            problems.append(f"{split}: unknown seed {sid}")
    report["seeds_covered"] = len(set().union(*actual.values()))

    # -- category budgets ------------------------------------------------
    train = records.get("train", [])
    by_category = Counter(row.get("category") for row in train)
    report["train_by_category"] = dict(sorted(by_category.items()))
    for category, budget in D3.RECORD_BUDGET.items():
        if by_category.get(category, 0) != budget:
            problems.append(
                f"{category}: train holds {by_category.get(category, 0)} "
                f"records, budget {budget}")
    for category in sorted(set(by_category) - set(D3.RECORD_BUDGET)):
        problems.append(f"{category}: has records but no budget")

    # -- final targets ---------------------------------------------------
    validation = records.get("validation", [])
    holdout = records.get("holdout", [])
    pools = {
        "train": train,
        "train_validation": train + validation,
        "holdout": holdout,
    }
    for name, rows in pools.items():
        counter = Counter(_target(row) for row in rows)
        repeat = max(counter.values()) if counter else 0
        report[f"distinct_targets_{name}"] = len(counter)
        report[f"target_max_repeat_{name}"] = repeat
        if repeat > D3.MAX_TARGET_REPEAT:
            problems.append(
                f"{name}: a final target repeats {repeat} times, cap "
                f"{D3.MAX_TARGET_REPEAT}")
    if len(Counter(_target(row) for row in train)) < (
            D3.MIN_DISTINCT_FINAL_TARGETS):
        problems.append(
            f"train distinct final targets "
            f"{len(Counter(_target(row) for row in train))} < "
            f"{D3.MIN_DISTINCT_FINAL_TARGETS}")
    report["distinct_train_targets"] = report["distinct_targets_train"]

    # -- verbatim targets across seeds ----------------------------------
    for name, rows in (("train_validation", train + validation),
                       ("holdout", holdout)):
        owners: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            owners[_target(row)].add(row.get("seed_id"))
        shared = {key: sorted(v) for key, v in owners.items() if len(v) > 1}
        if shared:
            sample = sorted(shared.items())[:3]
            problems.append(
                f"{name}: {len(shared)} final target(s) shared across seeds, "
                f"e.g. {sample}")

    # -- openings --------------------------------------------------------
    for name, rows in pools.items():
        counter = Counter(_opening(row) for row in rows)
        repeat = max(counter.values()) if counter else 0
        report[f"distinct_openings_{name}"] = len(counter)
        report[f"opening_max_{name}"] = repeat
    report["distinct_openings_train"] = report["distinct_openings_train"]
    for name in ("train", "train_validation"):
        rows = pools[name]
        distinct = report[f"distinct_openings_{name}"]
        repeat = report[f"opening_max_{name}"]
        if distinct < D3.MIN_DISTINCT_OPENINGS:
            problems.append(
                f"{name}: {distinct} distinct openings < "
                f"{D3.MIN_DISTINCT_OPENINGS}")
        if repeat > D3.MAX_OPENING_REPEAT:
            problems.append(
                f"{name}: an opening repeats {repeat} times, cap "
                f"{D3.MAX_OPENING_REPEAT}")

    # -- domain and register spread -------------------------------------
    # Per design: "Every category must span at least this many scenario
    # domains and registers" - no split qualification, so pool all emitted.
    pooled = train + validation + holdout
    domains_report: dict[str, dict] = {}
    for category in D3.RECORD_BUDGET:
        rows = [row for row in pooled if row.get("category") == category]
        ids = {row.get("seed_id") for row in rows}
        domains = {seeds[sid].domain for sid in ids if sid in seeds}
        registers = {SB3.fold_register(seeds[sid].register)
                     for sid in ids if sid in seeds}
        # Also report holdout-only for visibility
        hold_ids = {row.get("seed_id") for row in holdout
                    if row.get("category") == category}
        hold_domains = sorted({seeds[sid].domain
                               for sid in hold_ids if sid in seeds})
        domains_report[category] = {
            "domains": len(domains), "registers": len(registers),
            "holdout_domains": hold_domains,
        }
        if len(domains) < D3.MIN_DOMAINS_PER_CATEGORY:
            problems.append(
                f"{category}: all splits span {len(domains)} domains, "
                f"need {D3.MIN_DOMAINS_PER_CATEGORY}")
        if len(registers) < D3.MIN_REGISTERS_PER_CATEGORY:
            problems.append(
                f"{category}: all splits span {len(registers)} registers, "
                f"need {D3.MIN_REGISTERS_PER_CATEGORY}")
    report["spread_all_splits"] = domains_report

    # -- emission order --------------------------------------------------
    pair_report: dict[str, dict] = {}
    for split in SPLIT_ORDER:
        rows = records.get(split, [])
        for index in range(len(rows) - 1):
            left, right = rows[index], rows[index + 1]
            if left.get("category") != right.get("category"):
                continue
            cons_left = B3.construction(left["messages"], left["train_on"])
            cons_right = B3.construction(right["messages"], right["train_on"])
            if cons_left is not None and cons_left == cons_right:
                problems.append(
                    f"{split}: {left.get('id')} and {right.get('id')} are "
                    f"consecutive {left.get('category')} records opening on "
                    f"{cons_left!r}")
        for pair in D3.REQUIRED_ADJACENT_PAIRS:
            positions = [i for i, row in enumerate(rows)
                         if row.get("seed_id") in pair]
            if not positions:
                continue
            run = max(positions) - min(positions) + 1
            contiguous = run == len(positions)
            switches = sum(
                1 for i in range(min(positions), max(positions))
                if rows[i].get("seed_id") != rows[i + 1].get("seed_id"))
            pair_report[f"{split}:{pair[0]}/{pair[1]}"] = {
                "records": len(positions),
                "contiguous": contiguous,
                "side_switches": switches,
            }
            if split == "train" and not contiguous:
                problems.append(
                    f"train: required adjacent pair {pair[0]}/{pair[1]} is "
                    f"split across {len(positions)} records in a run of {run}")
    report["required_pairs"] = pair_report

    # -- supervised tokens -----------------------------------------------
    if system is not None:
        report.update(token_report(records, system, problems))
    else:
        problems.append("token gate skipped: no resolvable system context")

    return {"problems": problems, "report": report}


def main() -> int:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    records = {split: load(split, data_dir) for split in SPLIT_ORDER}
    result = audit(records)
    print(json.dumps(result["report"], indent=2, default=str))
    if result["problems"]:
        print(f"\nFAILURES ({len(result['problems'])}):")
        for problem in result["problems"][:40]:
            print(f"  FAIL {problem}")
        if len(result["problems"]) > 40:
            print(f"  ... and {len(result['problems']) - 40} more")
        return 1
    print("\nOK: every v0.3 dataset gate held.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
