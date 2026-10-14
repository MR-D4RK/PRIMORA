"""Preprocessing for Aurora Brain v0.1 QLoRA training.

Turns validated v2 records into fixed tensors. ``datasets`` is deliberately not
used: it is not installed, and the masking this dataset needs is a subset over
assistant turns rather than the contiguous completion span the library assumes.

The loader is not replaced or wrapped in another format. Records arrive through
``TRAINING/aurora/dataset_v2.py`` already validated, with the system context
resolved and its fingerprint verified; this module only tokenises and masks.

Truncation is counted and reported rather than hidden, because a record clipped
below its final assistant turn would lose all supervision. ``build_split`` fails
on those instead of silently shipping an unsupervised example.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch.utils.data import Dataset

from aurora.dataset_v2 import DEFAULT_OUTDIR, LoadedExample, load_split
from aurora.masking import (
    IGNORE_INDEX,
    MaskResult,
    build_mask,
    verify_mask,
)


@dataclass
class Preprocessed:
    example_id: str
    seed_id: str
    category: str
    input_ids: list[int]
    labels: list[int]
    truncated: bool

    def __len__(self) -> int:
        return len(self.input_ids)


class SupervisedDataset(Dataset):
    """A plain torch Dataset of pre-masked records.

    No shuffle happens here; the Trainer owns sampling so a seed fully determines
    the order.
    """

    def __init__(self, items: Sequence[Preprocessed]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        return {
            "input_ids": item.input_ids,
            "labels": item.labels,
            "attention_mask": [1] * len(item.input_ids),
        }


class CausalCollator:
    """Right-pads a batch and keeps padding out of the loss.

    ``labels`` are padded with ``IGNORE_INDEX`` so a short sequence in a batch
    contributes nothing at its pad positions. ``input_ids`` pad with the
    tokenizer's pad token, and ``attention_mask`` is 0 there so attention and
    loss both ignore them.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for f in features:
            pad = width - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [IGNORE_INDEX] * pad)
            attention.append(f["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def preprocess_example(
    example: LoadedExample,
    tokenizer,
    max_length: int,
    verify: bool = True,
) -> Preprocessed:
    """Tokenise and mask a single validated record."""
    try:
        mask: MaskResult = build_mask(
            tokenizer, example.messages, example.train_on, max_length)
    except Exception as exc:
        raise type(exc)(f"{example.id}: {exc}") from exc
    if verify:
        verify_mask(mask, example.messages, tokenizer, example.id)
    return Preprocessed(
        example_id=example.id,
        seed_id=example.seed_id,
        category=example.category,
        input_ids=mask.input_ids,
        labels=mask.labels,
        truncated=mask.truncated,
    )


def build_split(
    split: str,
    tokenizer,
    max_length: int,
    *,
    data_dir: str = DEFAULT_OUTDIR,
    include_holdout: bool = False,
    limit: Optional[int] = None,
    verify: bool = True,
) -> tuple[SupervisedDataset, dict[str, Any]]:
    """Load one split, mask it, and return the dataset plus a statistics block.

    The holdout is never built here without ``include_holdout``. Training does
    not pass it, and the loader refuses it anyway.
    """
    if split == "holdout" and not include_holdout:
        raise ValueError(
            "the holdout is protected; build_split requires include_holdout=True "
            "and training must not set it")

    examples = load_split(split, data_dir=data_dir, include_holdout=include_holdout)
    if limit is not None:
        examples = examples[:limit]

    items: list[Preprocessed] = []
    for example in examples:
        items.append(preprocess_example(example, tokenizer, max_length, verify=verify))

    lengths = [len(i) for i in items]
    trained = [sum(1 for l in i.labels if l != IGNORE_INDEX) for i in items]
    stats = {
        "split": split,
        "records": len(items),
        "truncated": sum(1 for i in items if i.truncated),
        "max_length": max_length,
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "length_mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "trained_tokens": sum(trained),
        "total_tokens": sum(lengths),
        "trained_fraction": (sum(trained) / sum(lengths)) if lengths else 0.0,
        "categories": len({i.category for i in items}),
        "seeds": len({i.seed_id for i in items}),
    }
    return SupervisedDataset(items), stats


__all__ = [
    "CausalCollator",
    "Preprocessed",
    "SupervisedDataset",
    "build_split",
    "preprocess_example",
]
