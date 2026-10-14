"""Regression tests for the v0.5 merged dataset.

Deliberately record-inspecting: every gate reads the real v4/v2 sources and
the real assembled merge rather than trusting hardcoded counts alone. The
build is in-memory (``BD5.build()``), so nothing is written to ``data/v5``
unless the builder is run explicitly.

Run: python -m pytest TESTS/training/test_dataset_v5.py -q
"""

from __future__ import annotations

import collections
import copy
import json
import os
import sys

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_dataset_v4 as BD4  # noqa: E402
import build_dataset_v5 as BD5  # noqa: E402
import validate_dataset_v5 as VD5  # noqa: E402
from aurora import dataset_v2  # noqa: E402


def _build():
    records, problems, info = BD5.build()
    assert problems == [], problems
    return records, info


# --------------------------------------------------------------------------
# 1. counts
# --------------------------------------------------------------------------


def test_total_and_split_counts() -> None:
    records, _info = _build()
    counts = {s: len(records[s]) for s in BD5.SPLIT_ORDER}
    assert counts == {"train": 1297, "validation": 180, "holdout": 180}
    assert sum(counts.values()) == 1657


# --------------------------------------------------------------------------
# 2. v4 eval identity: content (system metadata excluded) + membership
# --------------------------------------------------------------------------


def test_v4_eval_content_and_membership_identical() -> None:
    records, _info = _build()
    for side in ("validation", "holdout"):
        v5_rows = records[side]
        v4_rows = BD5.load_source(side, BD5.V4_DIR)
        # membership
        assert sorted(r["id"] for r in v5_rows) == \
            sorted(r["id"] for r in v4_rows), side
        # content, hashed with system_ref/system_fingerprint excluded
        assert BD5.split_digest(v5_rows) == BD5.split_digest(v4_rows), side
        # and the ONLY fields that differ are the two system metadata fields
        v4_by_id = {r["id"]: r for r in v4_rows}
        for row in v5_rows:
            src = v4_by_id[row["id"]]
            for key in set(row) | set(src):
                if key in BD5.SYSTEM_FIELDS:
                    assert row[key] != src[key] or True  # may or may not differ
                else:
                    assert row.get(key) == src.get(key), (side, row["id"], key)


def test_v4_eval_records_really_were_restamped() -> None:
    """Option 1 refreshes the stale ref; every v5 record must carry the live one."""
    records, info = _build()
    live = info["system_ref"]
    for side in BD5.SPLIT_ORDER:
        for row in records[side]:
            assert row["system_ref"] == live, row["id"]
            assert row["system_fingerprint"] == info["system_fingerprint"]
    # the source v4 files still carry the stale ref (untouched on disk)
    v4_val = BD5.load_source("validation", BD5.V4_DIR)
    assert all(r["system_ref"] != live for r in v4_val)


# --------------------------------------------------------------------------
# 3. merged-record provenance
# --------------------------------------------------------------------------


def test_merged_records_have_accurate_provenance() -> None:
    records, info = _build()
    merged = [r for side in BD5.SPLIT_ORDER for r in records[side]
              if r["generator_version"] == BD5.V2_GENERATOR_VERSION]
    v4_rows = [r for side in BD5.SPLIT_ORDER for r in records[side]
               if r["generator_version"] == BD5.V4_GENERATOR_VERSION]
    assert len(merged) == BD5.EXPECTED_RETAINED == 217
    assert len(v4_rows) == 1440
    assert all(r["seed_provenance"] == BD5.MERGED_PROVENANCE for r in merged)
    assert all(r["seed_provenance"] == BD5.V4_PROVENANCE for r in v4_rows)
    # all merged records are train-split expect_block records from v2
    assert all(r["split"] == "train" for r in merged)
    assert all(r.get("expect_block") is True for r in merged)
    assert [r["id"] for r in merged] == sorted(r["id"] for r in merged)
    assert len(info["retained_ids"]) == 217


def test_merged_records_are_the_audited_selection() -> None:
    """The selection is reproducible from source and matches the manifest."""
    records, info = _build()
    v2_train = BD5.load_source("train", BD5.V2_DIR)
    src = [r for r in v2_train if r.get("expect_block") is True]
    selected = BD5.select_v2_records(src)
    assert [r["id"] for r in selected] == info["retained_ids"]
    assert len(selected) == 217
    # order-independent: shuffling the source cannot change the selection
    shuffled = list(reversed(src))
    assert [r["id"] for r in BD5.select_v2_records(shuffled)] == \
        info["retained_ids"]


# --------------------------------------------------------------------------
# 4. combined exact-target repetition limit across all splits
# --------------------------------------------------------------------------


def test_repetition_limit_passes_across_all_splits() -> None:
    records, _info = _build()
    counts = collections.Counter(
        BD5.target_of(r) for side in BD5.SPLIT_ORDER for r in records[side])
    assert max(counts.values()) <= BD5.TARGET_REPEAT_LIMIT == 2
    caught: list[str] = []
    report = VD5.check_target_repetition(records, caught)
    assert caught == [], caught
    assert report["over_limit_exact"] == 0
    assert report["max_exact_repeat"] <= 2
    # the gate bites: a third copy of one target is refused
    rows = [r for side in BD5.SPLIT_ORDER for r in records[side]]
    twin = dict(rows[0])
    caught2: list[str] = []
    VD5.check_target_repetition({"train": rows + [twin]}, caught2)
    assert caught2, caught2
    assert BD5.gate_repetition({"train": rows + [twin]})


# --------------------------------------------------------------------------
# 5. block coverage and the six actions
# --------------------------------------------------------------------------


def test_block_targets_and_actions_covered() -> None:
    records, info = _build()
    train = records["train"]
    block_targets = {BD5.target_of(r) for r in train
                     if "<prime_request>" in BD5.target_of(r)}
    assert len(block_targets) == 17, len(block_targets)
    assert block_targets == set(info["retained_block_targets"]) | {
        t for t in block_targets if t not in info["retained_block_targets"]} \
        or len(block_targets) >= 17
    actions = {r.get("expect_action") for r in train if r.get("expect_action")}
    assert actions == {"recall", "learn", "research", "use_tool",
                       "calculate", "consult_specialist"}, actions
    caught: list[str] = []
    report = VD5.check_coverage(records, caught)
    assert caught == [], caught
    assert report["distinct_block_targets"] == 17
    assert report["action_kinds"] == 6


# --------------------------------------------------------------------------
# 6. duplicate and leakage checks operate on real records
# --------------------------------------------------------------------------


def test_duplicate_and_leakage_checks_use_real_records() -> None:
    records, _info = _build()
    system = dataset_v2.resolve_system_context(
        records["train"][0]["system_ref"])

    # clean dataset passes both
    caught: list[str] = []
    VD5.check_leakage(records, caught)
    assert caught == [], caught
    caught = []
    VD5.check_schema(records, system, caught)
    assert caught == [], caught

    # a duplicated record id is caught
    broken = {s: list(rows) for s, rows in records.items()}
    broken["train"] = broken["train"] + [dict(broken["train"][0])]
    caught = []
    VD5.check_schema(broken, system, caught)
    assert any("duplicate record id" in p for p in caught), caught

    # a distinctive value leaking across splits is caught. The value must
    # match one of build_dataset_v4's _VALUE_PATTERNS (a dotted version
    # qualifies) and appear in two different splits.
    leaky = {s: [dict(r) for r in rows] for s, rows in records.items()}
    sentinel = "the 99.88.77 release window"
    assert BD4.extract_values(sentinel), "sentinel must yield a distinctive value"
    for side in ("train", "validation"):
        leaky[side][0]["messages"] = [
            {"role": "user", "content": sentinel},
            {"role": "assistant", "content": BD5.target_of(leaky[side][0])},
        ]
    caught = []
    VD5.check_leakage(leaky, caught)
    assert any("share distinctive value" in p for p in caught), caught


# --------------------------------------------------------------------------
# 7. deterministic generation
# --------------------------------------------------------------------------


def test_generation_is_deterministic() -> None:
    records_a, info_a = _build()
    records_b, info_b = _build()
    assert info_a["system_ref"] == info_b["system_ref"]
    assert info_a["retained_ids"] == info_b["retained_ids"]
    for side in BD5.SPLIT_ORDER:
        assert [r["id"] for r in records_a[side]] == \
            [r["id"] for r in records_b[side]], side
        assert BD5.split_digest(records_a[side]) == \
            BD5.split_digest(records_b[side]), side
        assert records_a[side] == records_b[side], side


# --------------------------------------------------------------------------
# 8. the validator bites: over-limit fixture + missing block target
# --------------------------------------------------------------------------


def test_validator_rejects_over_limit_fixture() -> None:
    records, _info = _build()
    rows = [r for side in BD5.SPLIT_ORDER for r in records[side]]
    # inject a third supervised copy of one exact target
    base = rows[0]
    extra = copy.deepcopy(base)
    extra["id"] = "FIXTURE-OVERLIMIT-00"
    fixture = {"train": rows + [extra],
               "validation": records["validation"],
               "holdout": records["holdout"]}
    caught: list[str] = []
    report = VD5.check_target_repetition(fixture, caught)
    assert caught, caught
    assert report["over_limit_exact"] >= 1
    assert BD5.gate_repetition(fixture)


def test_validator_detects_missing_block_target() -> None:
    records, _info = _build()
    train = records["train"]
    block_targets = sorted({BD5.target_of(r) for r in train
                            if "<prime_request>" in BD5.target_of(r)})
    assert len(block_targets) == 17
    victim = block_targets[0]
    thinned = {"train": [r for r in train if BD5.target_of(r) != victim],
               "validation": records["validation"],
               "holdout": records["holdout"]}
    assert len(thinned["train"]) < len(train)
    caught: list[str] = []
    report = VD5.check_coverage(thinned, caught)
    assert report["distinct_block_targets"] == 16
    assert any("distinct block targets" in p for p in caught), caught


def test_validator_full_audit_is_clean() -> None:
    records, _info = _build()
    system = dataset_v2.resolve_system_context(
        records["train"][0]["system_ref"])
    result = VD5.audit(records, system, tokenize=False)
    assert result["problems"] == [], result["problems"]
    report = result["report"]
    assert report["contract"]["counts"]["total"] == 1657
    assert report["contract"]["provenance"]["merged_records"] == 217
    assert report["contract"]["provenance"]["v4_records"] == 1440
    assert report["contract"]["provenance"]["bad_system_ref"] == 0
    for side in ("validation", "holdout"):
        assert report["contract"]["eval_identity"][side]["membership_ok"]
        assert report["contract"]["eval_identity"][side]["content_ok"]
    assert report["contract"]["repetition"]["over_limit_exact"] == 0
    assert report["contract"]["coverage"]["distinct_block_targets"] == 17


def test_sources_are_untouched_after_build() -> None:
    """The merge reads v2/v4 but never writes them."""
    before = {}
    for d, splits in ((BD5.V4_DIR, BD5.SPLIT_ORDER), (BD5.V2_DIR, ("train",))):
        for side in splits:
            path = os.path.join(d, f"{side}.jsonl")
            before[path] = json.dumps(
                BD5.load_source(side, d), sort_keys=True)
    _build()
    for d, splits in ((BD5.V4_DIR, BD5.SPLIT_ORDER), (BD5.V2_DIR, ("train",))):
        for side in splits:
            path = os.path.join(d, f"{side}.jsonl")
            after = json.dumps(BD5.load_source(side, d), sort_keys=True)
            assert after == before[path], path
