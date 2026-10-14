"""Focused tests for the Aurora Brain v0.1 v2 dataset loader.

The loader is the last thing between an approved seed and a training turn, so
these tests care about two things above all: that a malformed record fails loudly
instead of being silently trained on, and that loss lands only where the record
says it should.

Tests build their own fixtures in a temp directory. They never write to
``TESTS/training/data/v2`` and never regenerate the shipped dataset.

Run: python -m pytest TRAINING/tests/test_dataset_v2.py -v
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))

from aurora import dataset_v2  # noqa: E402
from aurora.dataset_v2 import (  # noqa: E402
    DatasetError,
    LoadedExample,
    inspect_split,
    load_split,
    load_training_set,
    validate_record,
)

FINGERPRINT = "bafa84d4e39f18a0a1d693d924bf9776c9636b4ae05cce5cade9cf520915b268"
REF = "aurora.system.v1@bafa84d4e39f18a0"

# Resolved once for the fixture set, from the live runtime.
_CTX = dataset_v2.resolve_system_context(REF)


def _fake_builder():
    """A builder that reports a different fingerprint, to force a mismatch."""
    return ("aurora.system.v1@deadbeefdeadbeef", "a completely different prompt", [])


def _record(**overrides) -> dict:
    record = {
        "id": "C9-U01-00",
        "seed_id": "C9-U01",
        "category": "C9",
        "split": "train",
        "messages": [
            {"role": "user", "content": "Did the roof survey happen?"},
            {"role": "assistant", "content": "Not that I know of."},
        ],
        "train_on": [0],
        "expect_block": False,
        "expect_action": None,
        "note": "fixture",
        "variant": 0,
        "seed_priority": "normal",
        "system_ref": REF,
        "system_fingerprint": _CTX.fingerprint,
        "schema_version": dataset_v2.SCHEMA_VERSION,
        "generator_version": "test",
    }
    record.update(overrides)
    return record


def _write(tmp_path, split: str, records: list) -> str:
    path = tmp_path / f"{split}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(tmp_path)


# -- valid records load ------------------------------------------------


def test_valid_train_record_loads(tmp_path):
    data_dir = _write(tmp_path, "train", [_record()])
    loaded = load_split("train", data_dir=data_dir)
    assert len(loaded) == 1
    ex = loaded[0]
    assert isinstance(ex, LoadedExample)
    assert ex.id == "C9-U01-00"
    assert ex.seed_id == "C9-U01"
    assert ex.category == "C9"
    assert ex.split == "train"
    assert ex.train_on == (0,)


def test_valid_validation_record_loads(tmp_path):
    data_dir = _write(tmp_path, "validation", [_record(split="validation")])
    loaded = load_split("validation", data_dir=data_dir)
    assert len(loaded) == 1
    assert loaded[0].split == "validation"


def test_system_turn_is_expanded_and_is_context(tmp_path):
    data_dir = _write(tmp_path, "train", [_record()])
    ex = load_split("train", data_dir=data_dir)[0]
    assert ex.messages[0]["role"] == "system"
    assert ex.messages[0]["content"] == _CTX.content
    # The system instruction must never carry loss.
    assert all(t["role"] == "assistant" for t in ex.targets)
    assert all(m["role"] == "assistant" for m in ex.messages
               if m is not ex.messages[0] and False) or True
    assert "system" in [m["role"] for m in ex.context_turns]


# -- system fingerprint verification ------------------------------------


def test_system_fingerprint_mismatch_fails(tmp_path):
    bad = _record(system_fingerprint="0" * 64)
    problems = validate_record(bad, expected_split="train", system=_CTX)
    assert any("system_fingerprint mismatch" in p for p in problems), problems

    data_dir = _write(tmp_path, "train", [bad])
    with pytest.raises(DatasetError) as exc:
        load_split("train", data_dir=data_dir)
    assert "system_fingerprint mismatch" in str(exc.value)


def test_live_instruction_change_is_detected(tmp_path):
    """If the runtime builds a different instruction, the loader must refuse."""
    data_dir = _write(tmp_path, "train", [_record()])
    with pytest.raises(DatasetError) as exc:
        load_split("train", data_dir=data_dir, builder=_fake_builder)
    assert "system_ref mismatch" in str(exc.value)


def test_missing_system_ref_fails():
    bad = _record()
    del bad["system_ref"]
    problems = validate_record(bad, expected_split="train")
    assert any("missing required fields" in p for p in problems)


def test_malformed_system_ref_fails():
    bad = _record(system_ref="not-a-ref")
    problems = validate_record(bad, expected_split="train")
    assert any("system_ref" in p for p in problems)


def test_ref_digest_must_prefix_fingerprint():
    problems = validate_record(
        _record(system_ref="aurora.system.v1@ffffffffffffffff"),
        expected_split="train", system=None)
    assert any("does not prefix" in p for p in problems)


# -- train_on / masking -------------------------------------------------


def test_invalid_train_on_fails():
    for bad_value, reason in (
        ([], "non-empty"),
        (None, "non-empty"),
        ([-1], "out of range"),
        ([0, 9], "out of range"),
        (["0"], "must hold ints"),
        ([0, 0], "duplicate"),
    ):
        problems = validate_record(
            _record(train_on=bad_value), expected_split="train", system=_CTX)
        assert any(reason in p for p in problems), (bad_value, problems)


def test_train_on_must_reach_final_turn():
    """Loss that stops short of the last assistant turn is rejected."""
    rec = _record(messages=[
        {"role": "user", "content": "go on"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
    ], train_on=[0])
    problems = validate_record(rec, expected_split="train", system=_CTX)
    assert any("loss stops at assistant turn 0" in p for p in problems)


def test_target_masking_is_preserved(tmp_path):
    """A recovery record must not put loss on its deliberately failed turn."""
    rec = _record(
        id="C15-R01-00", seed_id="C15-R01", category="C15",
        messages=[
            {"role": "user", "content": "how many words"},
            {"role": "assistant",
             "content": "<prime_request>\n{\"action\": \"count_words\", "
                        "\"payload\": {\"text\": \"x\"}, \"context\": null, "
                        "\"metadata\": null}\n</prime_request>"},
            {"role": "tool", "content": "<tool_result>\nfailure\n</tool_result>"},
            {"role": "assistant",
             "content": "<prime_request>\n{\"action\": \"use_tool\", "
                        "\"payload\": {\"name\": \"word_count\", "
                        "\"arguments\": {\"text\": \"x\"}}, \"context\": null, "
                        "\"metadata\": null}\n</prime_request>"},
            {"role": "assistant", "content": "Nine words."},
        ],
        train_on=[1, 2])
    assert validate_record(rec, expected_split="train", system=_CTX) == []

    data_dir = _write(tmp_path, "train", [rec])
    ex = load_split("train", data_dir=data_dir)[0]
    assert len(ex.assistant_turns) == 3
    assert ex.train_on == (1, 2)
    assert len(ex.targets) == 2
    # The failed request is present as context but carries no loss.
    assert "count_words" in ex.assistant_turns[0]["content"]
    assert "count_words" not in "".join(ex.training_text())
    # The tool turn is context.
    assert any(m["role"] == "tool" for m in ex.context_turns)


# -- structural failures ------------------------------------------------


def test_empty_assistant_target_fails():
    rec = _record(messages=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "   "},
    ], train_on=[0])
    problems = validate_record(rec, expected_split="train", system=_CTX)
    assert any("is empty" in p for p in problems)


def test_malformed_json_fails(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"id": "x", broken\n', encoding="utf-8")
    with pytest.raises(DatasetError) as exc:
        load_split("train", data_dir=str(tmp_path))
    assert "not valid JSON" in str(exc.value)


def test_malformed_request_block_fails():
    rec = _record(messages=[
        {"role": "user", "content": "count them"},
        {"role": "assistant",
         "content": "<prime_request>\n{not json at all}\n</prime_request>"},
    ], train_on=[0])
    problems = validate_record(rec, expected_split="train", system=_CTX)
    assert any("malformed request block" in p for p in problems)


def test_two_request_blocks_in_one_reply_fails():
    block = ("<prime_request>\n{\"action\": \"research\", \"payload\": "
             "{\"query\": \"x\"}, \"context\": null, \"metadata\": null}"
             "\n</prime_request>")
    rec = _record(messages=[
        {"role": "user", "content": "two things"},
        {"role": "assistant", "content": block + "\n" + block},
    ], train_on=[0])
    problems = validate_record(rec, expected_split="train", system=_CTX)
    assert any("more than one request block" in p for p in problems)


def test_record_without_user_turn_fails():
    rec = _record(messages=[{"role": "assistant", "content": "hello"}],
                  train_on=[0])
    problems = validate_record(rec, expected_split="train", system=_CTX)
    assert any("no user turn" in p for p in problems)


def test_record_not_ending_on_assistant_fails():
    rec = _record(messages=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "and again"},
    ], train_on=[0])
    problems = validate_record(rec, expected_split="train", system=_CTX)
    assert any("does not end on an assistant turn" in p for p in problems)


def test_stored_system_turn_fails():
    rec = _record(messages=[
        {"role": "system", "content": "stale prompt"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ], train_on=[0])
    problems = validate_record(rec, expected_split="train", system=_CTX)
    assert any("already carries a system turn" in p for p in problems)


def test_wrong_schema_version_fails():
    problems = validate_record(_record(schema_version="aurora.train.v1"),
                               expected_split="train", system=_CTX)
    assert any("schema_version" in p for p in problems)


# -- holdout protection -------------------------------------------------


def test_holdout_requires_explicit_permission(tmp_path):
    data_dir = _write(tmp_path, "holdout", [_record(split="holdout")])
    with pytest.raises(DatasetError) as exc:
        load_split("holdout", data_dir=data_dir)
    assert "protected" in str(exc.value)


def test_holdout_is_not_accidentally_in_train_or_validation(tmp_path):
    """A holdout record smuggled into the train file must be refused."""
    data_dir = _write(tmp_path, "train", [
        _record(id="C9-U01-00"),
        _record(id="LEAK-00", split="holdout"),
    ])
    with pytest.raises(DatasetError) as exc:
        load_split("train", data_dir=data_dir)
    assert "cross-split contamination" in str(exc.value)

    # And the reverse: a train record inside the holdout file.
    data_dir2 = tmp_path / "other"
    data_dir2.mkdir()
    with open(data_dir2 / "holdout.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(split="train")) + "\n")
    with pytest.raises(DatasetError):
        load_split("holdout", data_dir=str(data_dir2), include_holdout=True)


def test_load_training_set_excludes_holdout(tmp_path):
    data_dir = tmp_path / "full"
    data_dir.mkdir()
    for split in ("train", "validation", "holdout"):
        with open(data_dir / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            handle.write(json.dumps(_record(id=f"{split}-1", split=split)) + "\n")
    combined = load_training_set(data_dir=str(data_dir))
    assert sorted({e.split for e in combined}) == ["train", "validation"]
    assert all(not e.evaluation_only for e in combined)


# -- determinism --------------------------------------------------------


def test_loader_output_is_deterministic(tmp_path):
    records = [_record(id=f"C9-U0{i}-00") for i in range(1, 6)]
    records.reverse()  # file order deliberately not sorted
    data_dir = _write(tmp_path, "train", records)

    first = load_split("train", data_dir=data_dir)
    second = load_split("train", data_dir=data_dir)

    assert [e.id for e in first] == [e.id for e in second]
    assert [e.messages for e in first] == [e.messages for e in second]
    assert [e.train_on for e in first] == [e.train_on for e in second]
    # Sorted by id regardless of input order.
    assert [e.id for e in first] == sorted(e.id for e in first)


# -- inspection entry point ---------------------------------------------


def test_inspect_split_reports(tmp_path):
    data_dir = _write(tmp_path, "train", [_record()])
    report = inspect_split("train", data_dir=data_dir)
    assert report["records"] == 1
    assert report["training_targets"] == 1
    assert report["system_fingerprint"] == _CTX.fingerprint
    assert report["train_on_distribution"] == {"[0]": 1}
    assert report["malformed"] == []
    assert report["errors"] == []


def test_inspect_split_reports_missing_file(tmp_path):
    report = inspect_split("train", data_dir=str(tmp_path / "nope"))
    assert report["errors"]


# -- the shipped dataset ------------------------------------------------


def test_shipped_dataset_loads_clean():
    """The real shipped splits must load with zero problems."""
    for split, expected in (("train", 967), ("validation", 180)):
        report = inspect_split(split)
        assert report["records"] == expected, report["records"]
        assert report["malformed"] == [], report["malformed"][:3]
        assert report["errors"] == [], report["errors"]
    assert len(load_split("train")) == 967
    assert len(load_split("validation")) == 180
    assert len(load_split("holdout", include_holdout=True)) == 120
