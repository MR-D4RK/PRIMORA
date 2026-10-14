"""Validate the Aurora Brain v0.3 dataset.

The audit is pure: it reads the three splits and returns ``(problems, report)``.
``build_dataset_v3`` does not call it - generation and validation stay
independent code paths so a bug in one cannot hide a bug in the other - but
both are run in the same order over the same files.

What fails the build:

*   malformed JSONL, or a record the accepted v2 schema rejects
    (``aurora.dataset_v2.validate_record`` against the live system context);
*   a split count other than the one the builder's plan produces;
*   a seed that appears in the wrong split, appears in two, or is missing;
*   a final target shared between splits, or shared between two seeds;
*   a mask that fails through the real ``aurora.masking`` module, or a record
    that renders past ``max_seq_length``;
*   a tool turn or capability payload that does not match the runtime's own
    contract.

What is reported but never fails: supervised-token statistics. v0.3 no longer
targets a conversational token share, so the number is printed and nothing more.
Tokenising still has to happen - that is the only way to know a mask is real and
a record fits - but its result is information, not a gate.

Run: python TESTS/training/validate_dataset_v3.py [outdir]
"""

from __future__ import annotations

import collections
import json
import os
import sys
from typing import Optional

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audit_v2 as A2  # noqa: E402
import build_dataset_v3 as BD  # noqa: E402
import design_v3 as D3  # noqa: E402
import seed_bank_v3 as SB3  # noqa: E402
from aurora import dataset_v2  # noqa: E402
from aurora.config import TrainerConfig  # noqa: E402
from aurora.masking import MaskingError, build_mask, verify_mask  # noqa: E402

DATA_DIR = BD.DEFAULT_OUTDIR
SPLIT_ORDER = BD.SPLIT_ORDER

# The one request the dataset makes on purpose that the registry refuses, so
# the record can show Aurora being rejected and recovering. Same exception the
# seed bank and the v2 audit use.
DELIBERATE_PAYLOADS = {("C15-R01", "count_words")}




def load(split: str, data_dir: str = DATA_DIR) -> list[dict]:
    path = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _target(row: dict) -> str:
    return A2.normalise(A2.final_assistant(row))


def _opening(row: dict) -> str:
    return A2.normalise(A2.first_user(row))[:18]


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_schema(records: dict[str, list[dict]],
                 system: dataset_v2.SystemContext,
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


def check_counts(records: dict[str, list[dict]],
                 problems: list[str]) -> dict[str, int]:
    counts = {split: len(records.get(split, [])) for split in SPLIT_ORDER}
    all_seeds = BD.seeds()
    split = BD.split_of(all_seeds)
    expected = BD.record_counts(split, all_seeds)
    wanted: collections.Counter = collections.Counter()
    for seed_id, want in expected.items():
        wanted[split[seed_id]] += want
    for side, want in wanted.items():
        if counts[side] != want:
            problems.append(
                f"{side} holds {counts[side]} records, the build plan "
                f"produces {want}")
    return counts


def check_provenance(records: dict[str, list[dict]],
                     problems: list[str]) -> dict:
    all_seeds = BD.seeds()
    by_id = {s.seed_id: s for s in all_seeds}
    plan = BD.split_of(all_seeds)

    actual: dict[str, set[str]] = {}
    for split in SPLIT_ORDER:
        ids = {row.get("seed_id") for row in records.get(split, [])}
        actual[split] = ids
        for unknown in sorted(ids - set(by_id)):
            problems.append(f"{split}: unknown seed {unknown}")
        for misplaced in sorted(ids - {
                sid for sid, side in plan.items() if side == split}):
            problems.append(
                f"{split}: seed {misplaced} belongs to "
                f"{plan.get(misplaced, 'no split')}")
        for absent in sorted({
                sid for sid, side in plan.items()
                if side == split and by_id[sid].training_allowed} - ids):
            problems.append(f"{split}: seed {absent} emitted no records")

    for seed_id, seed in by_id.items():
        if not seed.training_allowed:
            for split in SPLIT_ORDER:
                for row in records.get(split, []):
                    if row.get("seed_id") == seed_id:
                        problems.append(
                            f"{row.get('id')}: {seed_id} is the review-only "
                            f"anchor and must not be emitted")

    return {
        "seeds_in_bank": len(all_seeds),
        "seeds_emitted": len(set().union(*actual.values())) if actual else 0,
        "seeds_by_split": {split: len(ids) for split, ids in actual.items()},
        "by_provenance": dict(collections.Counter(
            row.get("seed_provenance")
            for split in SPLIT_ORDER for row in records.get(split, []))),
    }


def check_leakage(records: dict[str, list[dict]],
                  problems: list[str]) -> dict:
    """No final target may appear in more than one split."""
    pools = {split: {_target(row) for row in records.get(split, [])}
             for split in SPLIT_ORDER}
    for i, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[i + 1:]:
            shared = pools[left] & pools[right]
            if shared:
                problems.append(
                    f"{left} and {right} share {len(shared)} final target(s), "
                    f"e.g. {sorted(shared)[0][:60]!r}")

    owners: dict[str, set[str]] = collections.defaultdict(set)
    for split in SPLIT_ORDER:
        for row in records.get(split, []):
            owners[_target(row)].add(row.get("seed_id"))
    shared = {t: sorted(s) for t, s in owners.items() if len(s) > 1}
    if shared:
        problems.append(
            f"{len(shared)} final target(s) shared across seeds, e.g. "
            f"{sorted(shared.items())[:2]}")
    return {"cross_seed_shared_targets": len(shared)}


def check_tools(records: dict[str, list[dict]],
                problems: list[str]) -> dict:
    representation = A2.check_tool_representation(records)
    problems.extend(f"tool: {f}" for f in representation["findings"])
    sanity = A2.check_payload_sanity(records)
    problems.extend(f"payload: {f}" for f in sanity["findings"])

    try:
        from PRIMORA.prime.capability import CapabilityRegistry
        registry = CapabilityRegistry()
        known = set(registry.get_names())
    except Exception as exc:
        problems.append(f"capability registry unavailable: {exc}")
        return {"tool_turns": representation["tool_turns"]}

    failures: collections.Counter = collections.Counter()
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
                    continue
                request = json.loads(match.group(1).strip())
                action, payload = request.get("action"), request.get("payload")
                if not isinstance(payload, dict):
                    continue
                if action not in known:
                    if (row.get("seed_id"), action) not in DELIBERATE_PAYLOADS:
                        problems.append(
                            f"{row.get('id')}: unknown capability {action!r}")
                    continue
                spec = registry.get(action)
                missing = [n for n in spec.required_fields if n not in payload]
                extra = sorted(set(payload) - set(spec.required_fields)
                               - set(spec.optional_fields))
                if missing or extra:
                    problems.append(
                        f"{row.get('id')}: {action} payload "
                        f"missing={missing} extra={extra}")
            for message in row.get("messages", []):
                if message.get("role") != "tool":
                    continue
                try:
                    if json.loads(message.get("content", "")).get("status") \
                            == "failure":
                        failures[row.get("seed_id")] += 1
                except Exception:
                    pass
    return {
        "tool_turns": representation["tool_turns"],
        "tool_success": representation["success"],
        "tool_failure": representation["failure"],
        "failure_records_by_seed": dict(failures),
    }


def check_masking(records: dict[str, list[dict]],
                  system: dataset_v2.SystemContext,
                  problems: list[str]) -> dict:
    """Real masks, verified. Token counts are reported, never gated."""
    try:
        from transformers import AutoTokenizer
        config = TrainerConfig()
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model, trust_remote_code=True)
    except Exception as exc:
        problems.append(f"masking unavailable: {exc}")
        return {}

    supervised = {"conversational": 0, "capability": 0}
    per_split_tokens = {split: 0 for split in SPLIT_ORDER}
    truncated: list[str] = []
    trained_turns = 0
    for split in SPLIT_ORDER:
        for row in records.get(split, []):
            messages = [{"role": "system", "content": system.content}]
            messages.extend(row["messages"])
            try:
                result = build_mask(tokenizer, messages, row["train_on"],
                                    max_length=config.max_seq_length)
                verify_mask(result, messages, tokenizer, row.get("id", "<rec>"))
            except MaskingError as exc:
                problems.append(f"{row.get('id')}: mask check failed: {exc}")
                continue
            if result.truncated:
                truncated.append(row.get("id"))
                problems.append(
                    f"{row.get('id')}: renders past max_seq_length "
                    f"{config.max_seq_length}")
            trained_turns += len(result.trained_turns)
            per_split_tokens[split] += result.trained_tokens
            band = ("conversational"
                    if row.get("category") in D3.CONVERSATIONAL_CATEGORIES
                    else "capability")
            supervised[band] += result.trained_tokens

    total = supervised["conversational"] + supervised["capability"]
    return {
        "trained_turns": trained_turns,
        "truncated_records": len(truncated),
        "supervised_tokens_train": supervised,
        "supervised_tokens_by_split": per_split_tokens,
        "token_ratio_conversational": (
            round(supervised["conversational"] / total, 4) if total else 0.0),
        "note": "supervised-token figures are reported, not gated",
    }


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def audit(records: dict[str, list[dict]],
          system: Optional[dataset_v2.SystemContext] = None) -> dict:
    problems: list[str] = []
    report: dict = {}

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

    report["counts"] = check_counts(records, problems)
    check_schema(records, system, problems)
    report["provenance"] = check_provenance(records, problems)
    report["leakage"] = check_leakage(records, problems)
    report["tools"] = check_tools(records, problems)

    # -- diversity: reported, with only the repeat cap enforced ------------
    targets = collections.Counter()
    openings = collections.Counter()
    for split in SPLIT_ORDER:
        for row in records.get(split, []):
            targets[_target(row)] += 1
            openings[_opening(row)] += 1
    train_targets = collections.Counter(
        _target(row) for row in records.get("train", []))
    over_cap = [t for t, n in targets.items() if n > BD.MAX_TARGET_REPEAT]
    if over_cap:
        problems.append(
            f"{len(over_cap)} final target(s) repeat more than "
            f"{BD.MAX_TARGET_REPEAT} times across the dataset, e.g. "
            f"{max(over_cap, key=lambda t: targets[t])[:60]!r}")

    by_category = collections.Counter(
        row.get("category") for row in records.get("train", []))
    conversational = sum(n for c, n in by_category.items()
                         if c in D3.CONVERSATIONAL_CATEGORIES)
    capability = sum(n for c, n in by_category.items()
                    if c in D3.CAPABILITY_CATEGORIES)
    train_total = conversational + capability

    all_rows = [row for split in SPLIT_ORDER for row in records.get(split, [])]
    domains: dict[str, set[str]] = collections.defaultdict(set)
    registers: dict[str, set[str]] = collections.defaultdict(set)
    by_id = {s.seed_id: s for s in BD.seeds()}
    for row in all_rows:
        seed = by_id.get(row.get("seed_id"))
        if seed:
            domains[seed.category].add(seed.domain)
            registers[seed.category].add(SB3.fold_register(seed.register))

    report["diversity"] = {
        "distinct_final_targets": len(targets),
        "max_final_target_repeat": max(targets.values()) if targets else 0,
        "distinct_openings": len(openings),
        "max_opening_repeat": max(openings.values()) if openings else 0,
        "target_repeat_cap": BD.MAX_TARGET_REPEAT,
        "distinct_user_turns": len({A2.normalise(A2.first_user(row))
                                    for row in all_rows}),
        "distinct_assistant_targets": len({
            A2.normalise(m["content"]) for row in all_rows
            for m in row["messages"] if m["role"] == "assistant"}),
    }
    report["composition"] = {
        "train_records": train_total,
        "conversational_records": conversational,
        "capability_records": capability,
        "conversational_share": (
            round(conversational / train_total, 4) if train_total else 0.0),
        "by_category": dict(sorted(by_category.items())),
        "train_on_lengths": dict(sorted(collections.Counter(
            str(row.get("train_on")) for row in records.get("train", [])).items(),
            key=lambda kv: -kv[1])),
    }
    report["spread"] = {
        category: {"domains": len(domains.get(category, ())),
                   "registers": len(registers.get(category, ()))}
        for category in sorted(domains)}

    report["masking"] = check_masking(records, system, problems)
    return {"problems": problems, "report": report}


def main() -> int:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    records = {split: load(split, data_dir) for split in SPLIT_ORDER}
    if not any(records.values()):
        print(f"no dataset found under {data_dir}")
        return 1
    result = audit(records)
    print(json.dumps(result["report"], indent=2, default=str))
    problems = result["problems"]
    if problems:
        print(f"\nFAILURES ({len(problems)}):")
        for problem in problems[:40]:
            print(f"  FAIL {problem}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        return 1
    print("\nOK: every v0.3 dataset gate held.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
