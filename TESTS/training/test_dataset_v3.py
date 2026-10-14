"""Small generation + validation test for the v0.3 dataset package.

Deliberately narrow. It proves the package works end to end on a slice of the
bank - build, write, load, validate, and check the guarantees that matter most
(schema, provenance, split isolation, masking) - without tokenising the whole
corpus. The full run is ``build_dataset_v3.py`` followed by
``validate_dataset_v3.py``.

Run: python -m pytest TESTS/training/test_dataset_v3.py -q
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_dataset_v3 as BD  # noqa: E402
import validate_dataset_v3 as VD  # noqa: E402
from aurora import dataset_v2  # noqa: E402


def test_bank_and_split_are_whole_seed() -> None:
    all_seeds = BD.seeds()
    assert len(all_seeds) == BD.D3.SEED_BUDGET
    split = BD.split_of(all_seeds)
    # The review-only anchor is not training_allowed and is never assigned.
    assert len(split) == sum(1 for s in all_seeds if s.training_allowed)
    # Every evaluation seed must be able to fill the depth on its own, so the
    # split cannot be padded by repeating one target.
    counts = BD.record_counts(split, all_seeds)
    _all, split, counts, _draws = BD.plan()
    for seed_id, want in counts.items():
        if split[seed_id] == "train" or want == 0:
            continue
        assert want in (BD.EVAL_DEPTH, BD.EVAL_DEPTH + 1), \
            f"{seed_id} fills {want}"


def test_expansion_respects_target_repeat_cap() -> None:
    _all, _split, counts, draws = BD.plan()
    for seed_id, want in counts.items():
        if want == 0:
            continue
        assert want <= BD.MAX_TARGET_REPEAT * len(draws[seed_id]), seed_id


def test_failure_example_is_supervised() -> None:
    """The count_words refusal is the bank's only failure case; it must train."""
    split = BD.split_of(BD.seeds())
    assert split["C15-R01"] == "train"


def test_small_build_validates(tmp_path=None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        records, problems, meta = BD.build()
        assert problems == [], problems
        written = BD.write(records, meta, tmp)
        assert written == {s: len(records[s]) for s in BD.SPLIT_ORDER}
        assert written["train"] > 0
        assert written["validation"] > 0
        assert written["holdout"] > 0

        for side in BD.SPLIT_ORDER:
            path = os.path.join(tmp, f"{side}.jsonl")
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        json.loads(line)          # valid JSONL
            assert len(VD.load(side, tmp)) == written[side]

        # Full audit minus the tokenizer pass, which is the slow part and is
        # exercised by validate_dataset_v3 on the real corpus.
        system = dataset_v2.resolve_system_context(
            records["train"][0]["system_ref"])
        result = VD.audit(records, system)
        assert result["problems"] == [], result["problems"]
        assert result["report"]["provenance"]["seeds_emitted"] == \
            sum(1 for s in BD.seeds() if s.training_allowed)
        assert result["report"]["leakage"]["cross_seed_shared_targets"] == 0
        assert result["report"]["tools"]["tool_failure"] > 0
        composition = result["report"]["composition"]
        assert composition["conversational_records"] > \
            composition["capability_records"]
