"""Small build + validation test for the v0.4 behavioral dataset package.

Deliberately narrow. It proves the package works end to end: bank, grouping,
split gates, record shape, a full build to a temp directory, and the static
audit minus the tokenizer pass (the slow part, exercised by
``validate_dataset_v4.py`` on the real corpus).

Run: python -m pytest TESTS/training/test_dataset_v4.py -q
"""

from __future__ import annotations

import collections
import json
import os
import sys
import tempfile

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_dataset_v4 as BD4  # noqa: E402
import responses_v4 as RV4  # noqa: E402
import seed_bank_v4 as SB4  # noqa: E402
import validate_dataset_v4 as VD4  # noqa: E402
from aurora import dataset_v2  # noqa: E402


def test_bank_and_specs_agree() -> None:
    assert len(SB4.SEEDS) == 240
    RV4.self_check()
    assert set(RV4.SPECS) == {seed["id"] for seed in SB4.SEEDS}


def test_split_is_whole_seed_and_exact() -> None:
    _seeds, split, info = BD4.plan()
    assert info["problems"] == [], info["problems"]
    counts = {side: sum(1 for s in split.values() if s == side)
              for side in BD4.SPLIT_ORDER}
    assert counts == {"train": 180, "validation": 30, "holdout": 30}


def test_shared_value_seeds_share_a_split() -> None:
    """The grouping obligations: a shared distinctive value is atomic."""
    _seeds, split, _info = BD4.plan()
    assert split["C07"] == split["G03"]     # ./config/aurora.toml
    assert split["A24"] == split["G06"]     # helios
    assert split["A25"] == split["A35"]     # 3.14.2
    assert split["A19"] == split["A34"]     # 14 March
    assert split["A26"] == split["B24"]     # Norwex


def test_value_pools_are_disjoint_across_splits() -> None:
    _seeds, split, _info = BD4.plan()
    pools = {side: set() for side in BD4.SPLIT_ORDER}
    for sid, side in split.items():
        pools[side] |= BD4.seed_values(sid)
    for i, left in enumerate(BD4.SPLIT_ORDER):
        for right in BD4.SPLIT_ORDER[i + 1:]:
            assert not pools[left] & pools[right], (left, right)


def test_target_repetition_gate() -> None:
    """One exact final target is supervised at most twice, enforced by both
    the builder and the independent validator."""
    records, problems, _info = BD4.build()
    assert problems == [], problems
    assert BD4.target_repetition_gate(records) == []

    counts = collections.Counter(
        BD4.target_of(row) for rows in records.values() for row in rows)
    assert max(counts.values()) <= BD4.TARGET_REPEAT_LIMIT == 2

    assert VD4.TARGET_REPEAT_LIMIT == BD4.TARGET_REPEAT_LIMIT
    report = VD4.check_target_repetition(records, [])
    assert report["over_limit_exact"] == 0, report
    assert report["max_exact_repeat"] <= BD4.TARGET_REPEAT_LIMIT
    assert report["distinct_exact"] == 720

    # The gate actually bites: a third copy of one target must be refused.
    rows = [row for side_rows in records.values() for row in side_rows]
    twin = dict(rows[0])
    caught: list[str] = []
    VD4.check_target_repetition({"train": rows + [twin]}, caught)
    assert caught, caught
    assert BD4.target_repetition_gate({"train": rows + [twin]})


def test_record_shapes() -> None:
    records, problems, _info = BD4.build()
    assert problems == [], problems
    assert {s: len(records[s]) for s in BD4.SPLIT_ORDER} == \
        {"train": 1080, "validation": 180, "holdout": 180}

    by_sid = {row["seed_id"]: row for rows in records.values() for row in rows}
    # A direct seed: two turns, loss on the assistant reply.
    direct = by_sid["A01"]
    assert [m["role"] for m in direct["messages"]] == ["user", "assistant"]
    assert direct["train_on"] == [0]
    # A context seed: context, acknowledgement, ask, reply; loss on the reply.
    context = next(row for rows in records.values() for row in rows
                   if RV4.SPECS[row["seed_id"]].context)
    assert [m["role"] for m in context["messages"]] == \
        ["user", "assistant", "user", "assistant"]
    assert context["train_on"] == [1]
    # A prior seed: prior exchange, ask, reply; loss on the reply.
    prior = next(row for rows in records.values() for row in rows
                 if RV4.SPECS[row["seed_id"]].prior_ask)
    assert [m["role"] for m in prior["messages"]] == \
        ["user", "assistant", "user", "assistant"]
    assert prior["train_on"] == [1]

    # Six records per seed: ids 00..05, variants 0/1/2, bare vs framed pairs.
    ids = [row["id"] for rows in records.values() for row in rows]
    assert len(ids) == len(set(ids)) == 1440
    per_seed: dict[str, list[dict]] = {}
    for rows in records.values():
        for row in rows:
            per_seed.setdefault(row["seed_id"], []).append(row)
    for sid, rows in per_seed.items():
        assert len(rows) == 6, sid
        assert sorted(r["id"] for r in rows) == \
            [f"{sid}-{j:02d}" for j in range(6)]
        assert {r["variant"] for r in rows} == {0, 1, 2}


def test_bare_and_framed_differ() -> None:
    records, _problems, _info = BD4.build()
    per_seed: dict[str, dict[str, str]] = {}
    for rows in records.values():
        for row in rows:
            final_ask = [m for m in row["messages"]
                         if m["role"] == "user"][-1]["content"]
            per_seed.setdefault(row["seed_id"], {})[row["id"].split("-")[-1]] \
                = final_ask
    for sid, asks in per_seed.items():
        spec = RV4.SPECS[sid]
        for j in (0, 2, 4):
            assert asks[f"{j:02d}"] == spec.ask, sid
        for j in (1, 3, 5):
            assert asks[f"{j:02d}"] != spec.ask, sid
            assert asks[f"{j:02d}"].endswith(spec.ask), sid


def test_full_build_audits_clean() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        records, problems, info = BD4.build()
        assert problems == [], problems
        written = BD4.write(records, info, tmp)
        assert written == {"train": 1080, "validation": 180, "holdout": 180}
        for side in BD4.SPLIT_ORDER:
            path = os.path.join(tmp, f"{side}.jsonl")
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        json.loads(line)
            assert len(VD4.load(side, tmp)) == written[side]
        assert os.path.exists(os.path.join(tmp, "build_manifest.json"))
        assert os.path.exists(os.path.join(tmp, "HOLDOUT_PROTECTED.json"))

        system = dataset_v2.resolve_system_context(
            records["train"][0]["system_ref"])
        result = VD4.audit(records, system, tokenize=False)
        assert result["problems"] == [], result["problems"]
        report = result["report"]
        assert report["contract"]["leakage"]["cross_seed_shared_targets"] == 0
        assert report["contract"]["spec_checks"]["tallies"] == {}
        assert report["pathology"]["diversity"]["template_openings"] == 0
        assert report["pathology"]["diversity"]["near_duplicates_ge_0.9"] == 0
