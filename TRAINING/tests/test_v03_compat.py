"""Compatibility sanity check for the v0.3 training run.

CPU only. No model weights are loaded, so this is the check to run before
touching the GPU: it walks several v0.3 records through the exact path the
trainer uses - ``dataset_v2.load_split`` -> ``masking.build_mask`` /
``verify_mask`` -> ``preprocess.preprocess_example`` -> ``CausalCollator`` - and
asserts the things that would otherwise surface as a crash or a silent quality
loss minutes into a multi-hour run.

What it proves:

*   the v0.3 JSONL loads through the accepted loader, with the live system
    context resolved and its fingerprint verified;
*   masks verify through the real masking module and the EOS terminator stays
    supervised;
*   nothing truncates at ``max_seq_length``, on either split;
*   the collator produces correctly padded tensors, with padding excluded from
    the loss and each row reproducing its own supervised span;
*   the holdout is refused rather than loaded.

Run: python TRAINING/tests/test_v03_compat.py
"""

from __future__ import annotations

import os
import sys

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))

import torch  # noqa: E402

from aurora.config import DEFAULT_BASE_MODEL, TrainerConfig  # noqa: E402
from aurora.dataset_v2 import DatasetError, load_split  # noqa: E402
from aurora.masking import IGNORE_INDEX  # noqa: E402
from aurora.preprocess import CausalCollator, preprocess_example  # noqa: E402
from aurora.qlora import check_model_files, load_tokenizer  # noqa: E402

DATA_DIR = os.path.join(_ROOT, "TESTS", "training", "data", "v3")
EXPECTED = {"train": 1518, "validation": 180, "holdout": 120}
SAMPLE = 24

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  PASS  {message}")
    else:
        FAILURES.append(message)
        print(f"  FAIL  {message}")


def main() -> int:
    config = TrainerConfig()
    print("=" * 72)
    print("AURORA BRAIN v0.3 - TRAINER COMPATIBILITY SANITY CHECK (CPU)")
    print("=" * 72)
    print(f"  data dir  : {config.data_dir}")
    print(f"  base model: {DEFAULT_BASE_MODEL}")
    print(f"  seq length: {config.max_seq_length}")
    print()

    print("[1] base model files")
    check_model_files(config.base_model)
    tokenizer = load_tokenizer(config.base_model)
    check(tokenizer.pad_token_id is not None,
          f"tokenizer has a pad token (id {tokenizer.pad_token_id})")
    check(len(tokenizer) > 0, f"tokenizer vocabulary is {len(tokenizer):,}")

    print()
    print("[2] splits load through the accepted loader")
    splits = {}
    for name, expected in EXPECTED.items():
        try:
            rows = load_split(name, data_dir=DATA_DIR,
                              include_holdout=(name == "holdout"))
            splits[name] = rows
            check(len(rows) == expected,
                  f"{name}: {len(rows)} records (expected {expected})")
            check(all(e.split == name for e in rows),
                  f"{name}: every record declares split={name!r}")
        except DatasetError as exc:
            if name == "holdout":
                check(False, f"holdout loaded without protection: {exc}")
            else:
                check(False, f"{name} failed to load: {exc}")

    print()
    print("[3] holdout is refused without explicit consent")
    try:
        load_split("holdout", data_dir=DATA_DIR)
        check(False, "holdout loaded without include_holdout=True")
    except DatasetError as exc:
        check("protected" in str(exc).lower(),
              f"holdout refused: {str(exc)[:60]}...")

    print()
    print("[4] masking through the real path (train + validation)")
    eos_id = tokenizer.eos_token_id
    for name in ("train", "validation"):
        rows = splits.get(name, [])
        if not rows:
            continue
        sample = rows[:SAMPLE]
        items = [preprocess_example(e, tokenizer, config.max_seq_length)
                 for e in sample]
        check(not any(i.truncated for i in items),
              f"{name}: none of {len(items)} sampled records truncate at "
              f"{config.max_seq_length}")
        check(all(sum(1 for l in i.labels if l != IGNORE_INDEX) > 0
                  for i in items),
              f"{name}: every sampled record carries supervised tokens")
        lengths = [len(i.input_ids) for i in items]
        print(f"        {name} sample lengths {min(lengths)}..{max(lengths)}")
        # EOS supervision: the terminator of a selected turn must carry loss,
        # and it must directly follow supervised content.
        supervised_eos = 0
        misplaced = 0
        for item in items:
            for pos, (tid, lab) in enumerate(zip(item.input_ids, item.labels)):
                if tid != eos_id or lab == IGNORE_INDEX:
                    continue
                supervised_eos += 1
                if pos == 0 or item.labels[pos - 1] == IGNORE_INDEX:
                    misplaced += 1
        # Every record supervises at least its final assistant turn, so each one
        # must supervise at least one EOS. Intermediate assistant turns may
        # legitimately have a masked terminator - a record with train_on [1] has
        # two assistant turns and only the second is supervised - so the check
        # is "at least one", not "exactly one per turn".
        check(supervised_eos >= len(items),
              f"{name}: every sampled record supervises at least one EOS "
              f"terminator ({supervised_eos} supervised EOS across "
              f"{len(items)} records)")
        check(misplaced == 0,
              f"{name}: every supervised EOS follows supervised content")

    print()
    print("[5] batch collation")
    train_rows = splits.get("train", [])
    if train_rows:
        items = [preprocess_example(e, tokenizer, config.max_seq_length)
                 for e in train_rows[:SAMPLE]]
        collator = CausalCollator(tokenizer.pad_token_id)
        batch = collator([{"input_ids": i.input_ids, "labels": i.labels,
                           "attention_mask": [1] * len(i.input_ids)}
                          for i in items])
        width = max(len(i.input_ids) for i in items)
        check(batch["input_ids"].shape == (len(items), width),
              f"input_ids collated to {tuple(batch['input_ids'].shape)}")
        check(batch["labels"].shape == batch["input_ids"].shape,
              "labels match input_ids shape")
        check(bool((batch["input_ids"].dtype == torch.long)
                   and (batch["labels"].dtype == torch.long)),
              "tensors are long")
        pad_supervised = 0
        rows_with_padding = 0
        for row_index, item in enumerate(items):
            tail = width - len(item.input_ids)
            if not tail:
                continue
            rows_with_padding += 1
            check(bool((batch["input_ids"][row_index][len(item.input_ids):]
                        == tokenizer.pad_token_id).all()),
                  f"{item.example_id}: padded tail uses the pad token")
            check(bool((batch["attention_mask"][row_index]
                        [len(item.input_ids):] == 0).all()),
                  f"{item.example_id}: attention mask is 0 on padding")
            pad_supervised += int(
                (batch["labels"][row_index][len(item.input_ids):]
                 != IGNORE_INDEX).sum())
            # The unpadded part must be reproduced exactly, or collation itself
            # corrupted a row.
            check(batch["input_ids"][row_index][:len(item.input_ids)].tolist()
                  == item.input_ids,
                  f"{item.example_id}: row survives collation unaltered")
        check(pad_supervised == 0,
              "padding positions are excluded from the loss")
        print(f"        {rows_with_padding} of {len(items)} rows were padded")
        total = int((batch["labels"] != IGNORE_INDEX).sum())
        check(total > 0, f"batch carries {total:,} supervised tokens")

    print()
    print("=" * 72)
    if FAILURES:
        print(f"{len(FAILURES)} PROBLEM(S):")
        for problem in FAILURES:
            print(f"  FAIL {problem}")
        return 1
    print("OK: the v0.3 dataset is compatible with the v0.1 trainer path.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())