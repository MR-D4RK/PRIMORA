"""Loader for the Aurora Brain v0.1 v2 training dataset.

Reads ``TESTS/training/data/v2/{train,validation,holdout}.jsonl`` and turns each
record into a training-ready example: a system turn resolved from the live
runtime, the stored conversation, and the assistant turns that actually carry
loss.

Why the system turn is resolved rather than stored
--------------------------------------------------
The old 3,400-record corpus began every record with a user turn and supplied the
1,385-token composed instruction from an external step, so a stored record did
not describe the conversation Aurora actually has. ``RUNTIME_FORMAT`` in the seed
design fixes this: the system turn is stored as a reference key plus a recorded
fingerprint, and the loader expands the key and *verifies the fingerprint before
use*.

The instruction is built here from the same canonical sources the runtime uses -
``AuroraConfig.system_instruction`` plus ``build_capability_instruction``,
``build_specialist_instruction`` and ``build_tool_instruction`` - via
``generate_v2.build_system_context``. That is imported rather than copied so
there is exactly one definition of Aurora's system prompt. If the runtime
instruction or the capability catalogue changes, the fingerprint stops matching
and this loader fails loudly instead of training against a stale prompt.

Masking
-------
Loss is on assistant turns only, and on exactly those a record's ``train_on``
names. System, user and tool turns are context. The system turn is prepended
here for convenience but is never trained, so expanding it cannot leak the
instruction into the loss.

Holdout
-------
``load_split("holdout")`` must be called deliberately and sets
``example.evaluation_only``. The train and validation loaders refuse to return a
record whose ``split`` field is anything other than the requested one, so a
holdout record cannot ride along.

Run: python TRAINING/aurora/dataset_v2.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

sys.path.insert(0, r"C:\PRIMORA PROJECT")
sys.path.insert(0, os.path.join(r"C:\PRIMORA PROJECT", "TESTS", "training"))

from PRIMORA.aurora.aurora import extract_prime_request  # noqa: E402

# The one canonical system-context builder. Imported, not reimplemented, so the
# dataset's fingerprint and the runtime's instruction cannot drift apart.
from generate_v2 import (  # noqa: E402
    DEFAULT_OUTDIR,
    SCHEMA_VERSION,
    build_system_context,
)

SPLITS = ("train", "validation", "holdout")

REQUIRED_FIELDS = (
    "id", "seed_id", "category", "split", "messages", "train_on",
    "system_ref", "system_fingerprint", "schema_version", "generator_version",
)

VALID_ROLES = ("system", "user", "assistant", "tool")

# Matches the runtime parser in PRIMORA/aurora/aurora.py. A record that contains
# an opening tag but does not parse is malformed and must not be trained on.
PRIME_REQUEST_BLOCK = re.compile(r"<prime_request>(.*?)</prime_request>", re.DOTALL)

REF_PREFIX = "aurora.system.v1@"


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be used as-is. Never swallowed."""


@dataclass(frozen=True)
class LoadedExample:
    """One validated record, ready to be tokenised."""

    id: str
    seed_id: str
    category: str
    split: str
    messages: list[dict[str, str]]           # system turn prepended
    train_on: tuple[int, ...]                # assistant-turn indexes
    system_fingerprint: str
    schema_version: str
    generator_version: str
    expect_block: bool = False
    expect_action: Optional[str] = None
    note: str = ""
    evaluation_only: bool = False

    @property
    def assistant_turns(self) -> list[dict[str, str]]:
        return [m for m in self.messages if m["role"] == "assistant"]

    @property
    def context_turns(self) -> list[dict[str, str]]:
        """Every turn that is context: system, user and tool."""
        return [m for m in self.messages if m["role"] != "assistant"]

    @property
    def targets(self) -> list[dict[str, str]]:
        """The assistant turns that carry loss, in order."""
        assistants = self.assistant_turns
        return [assistants[i] for i in self.train_on]

    def training_text(self) -> list[str]:
        return [t["content"] for t in self.targets]


# --------------------------------------------------------------------------
# System context
# --------------------------------------------------------------------------


@dataclass
class SystemContext:
    ref: str
    fingerprint: str
    content: str


_CACHE: dict[str, SystemContext] = {}


def resolve_system_context(
    expected_ref: str,
    builder: Optional[Callable[[], tuple[str, str, list[str]]]] = None,
    verify_fingerprint: bool = True,
) -> SystemContext:
    """Resolve a ``system_ref`` against the live runtime instruction.

    Raises DatasetError if the reference is malformed, if the live instruction
    no longer produces it, or if ``verify_fingerprint`` is set and the resolved
    digest disagrees with ``expected_ref``'s embedded digest.
    """
    if not isinstance(expected_ref, str) or not expected_ref.startswith(REF_PREFIX):
        raise DatasetError(f"malformed system_ref: {expected_ref!r}")
    ref_digest = expected_ref[len(REF_PREFIX):]
    if not re.fullmatch(r"[0-9a-f]{16}", ref_digest):
        raise DatasetError(f"system_ref has no usable digest: {expected_ref!r}")

    # The cache is skipped when a builder is injected. Caching on the ref alone
    # would mean a second load in the same process reuses the first result
    # without re-resolving, so a runtime prompt that changed mid-run would be
    # served from cache and never verified - exactly the drift this check exists
    # to catch.
    if builder is None and expected_ref in _CACHE:
        return _CACHE[expected_ref]

    build = builder or build_system_context
    live_ref, content, _capabilities = build()
    live_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

    if live_ref != expected_ref:
        raise DatasetError(
            f"system_ref mismatch: dataset expects {expected_ref!r} but the "
            f"live runtime builds {live_ref!r}. The system instruction or the "
            f"capability catalogue has changed since generation; regenerate "
            f"against the current runtime or pin the old instruction."
        )
    if verify_fingerprint and not live_digest.startswith(ref_digest):
        raise DatasetError(
            f"system fingerprint mismatch for {expected_ref!r}: live digest "
            f"is {live_digest}, reference encodes {ref_digest}"
        )

    ctx = SystemContext(ref=expected_ref, fingerprint=live_digest, content=content)
    if builder is None:
        _CACHE[expected_ref] = ctx
    return ctx


# --------------------------------------------------------------------------
# Record validation
# --------------------------------------------------------------------------


def validate_record(
    raw: Any,
    *,
    expected_split: Optional[str] = None,
    system: Optional[SystemContext] = None,
    builder: Optional[Callable[[], tuple[str, str, list[str]]]] = None,
    record_fingerprint: Optional[str] = None,
    evaluation_only: bool = False,
) -> list[str]:
    """Return a list of problems. Empty means the record is usable.

    Every problem is reported rather than only the first, so a malformed file
    can be fixed in one pass instead of one error per run.
    """
    problems: list[str] = []

    if not isinstance(raw, dict):
        return ["record is not a JSON object"]

    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        problems.append(f"missing required fields: {missing}")
        return problems  # nothing further is meaningful

    rid = raw.get("id")

    if expected_split is not None and raw.get("split") != expected_split:
        # This is the holdout-leakage guard: a record whose declared split is
        # not the split being loaded must never be returned.
        problems.append(
            f"split is {raw.get('split')!r}, expected {expected_split!r} "
            f"(refusing: cross-split contamination)")

    if raw.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version is {raw.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION!r}")

    # -- messages and role ordering ------------------------------------
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        problems.append("messages must be a non-empty list")
        return problems

    for i, m in enumerate(messages):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            problems.append(f"message {i} is not a {{role, content}} object")
            continue
        if m["role"] not in VALID_ROLES:
            problems.append(f"message {i} has unknown role {m['role']!r}")

    if not any(m.get("role") == "user" for m in messages if isinstance(m, dict)):
        problems.append("record has no user turn")
    if messages[0].get("role") == "system":
        problems.append("record already carries a system turn; it is expanded "
                        "by the loader and must not be stored in the file")

    assistants = [m for m in messages
                  if isinstance(m, dict) and m.get("role") == "assistant"]
    if not assistants:
        problems.append("record has no assistant turn")
    elif messages[-1].get("role") != "assistant":
        problems.append("record does not end on an assistant turn")

    # -- train_on / masking ---------------------------------------------
    # train_on is a *subset* of assistant turns. Recovery records deliberately
    # omit the failed request turn (C15-R01 is [1, 2] across three assistant
    # turns), and prior-context records omit filler acknowledgements
    # (C13-A06 is [1] across two). So the invariant is that the indexes are
    # real, in range, and that loss reaches the final turn - not that every
    # assistant turn is trained.
    train_on = raw.get("train_on")
    if not isinstance(train_on, list) or not train_on:
        problems.append("train_on must be a non-empty list")
    elif any(not isinstance(i, int) or isinstance(i, bool) for i in train_on):
        problems.append(f"train_on must hold ints, got {train_on}")
    elif len(set(train_on)) != len(train_on):
        problems.append(f"train_on has duplicate indexes: {train_on}")
    elif any(i < 0 or i >= len(assistants) for i in train_on):
        problems.append(
            f"train_on {train_on} out of range for {len(assistants)} "
            f"assistant turn(s)")
    elif max(train_on) != len(assistants) - 1:
        problems.append(
            f"loss stops at assistant turn {max(train_on)} but the record "
            f"ends on {len(assistants) - 1}")

    # -- targets are non-empty ------------------------------------------
    for i in (train_on if isinstance(train_on, list) else []):
        if not isinstance(i, int) or isinstance(i, bool):
            continue
        if 0 <= i < len(assistants):
            content = assistants[i].get("content", "")
            if not content or not content.strip():
                problems.append(f"training target {i} is empty")

    # -- request blocks --------------------------------------------------
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        content = m.get("content", "") or ""
        if content.count("<prime_request>") != content.count("</prime_request>"):
            problems.append(f"message {i} has an unbalanced request block")
        if "<prime_request>" in content:
            if content.count("<prime_request>") > 1:
                problems.append(
                    f"message {i} contains more than one request block; the "
                    f"runtime rejects the whole reply")
            if extract_prime_request(content) is None:
                problems.append(f"message {i} has a malformed request block")

    # -- system context --------------------------------------------------
    system_ref = raw.get("system_ref")
    stored_fingerprint = raw.get("system_fingerprint")
    if system is not None:
        if system_ref != system.ref:
            problems.append(f"system_ref {system_ref!r} does not match the "
                            f"resolved reference {system.ref!r}")
        if stored_fingerprint != system.fingerprint:
            problems.append(
                f"system_fingerprint mismatch: record stores "
                f"{stored_fingerprint!r}, runtime produces "
                f"{system.fingerprint!r}")
    else:
        # Still check shape even when resolution was skipped.
        if not isinstance(stored_fingerprint, str) or \
                not re.fullmatch(r"[0-9a-f]{64}", stored_fingerprint):
            problems.append("system_fingerprint is not a sha256 digest")
        if isinstance(system_ref, str) and isinstance(stored_fingerprint, str) \
                and stored_fingerprint.startswith(
                    system_ref[len(REF_PREFIX):] if
                    system_ref.startswith(REF_PREFIX) else "\0"):
            pass
        else:
            problems.append("system_ref digest does not prefix system_fingerprint")

    return problems


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _read_jsonl(path: str) -> Iterable[tuple[int, str]]:
    """Yield (line number, raw line). Malformed JSON is raised, not skipped."""
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno} is not valid JSON: {exc}") from exc


def load_split(
    split: str,
    *,
    data_dir: str = DEFAULT_OUTDIR,
    include_holdout: bool = False,
    strict: bool = True,
    builder: Optional[Callable[[], tuple[str, str, list[str]]]] = None,
) -> list[LoadedExample]:
    """Load one split, deterministically ordered by record id.

    ``include_holdout`` must be set explicitly to read the protected holdout, and
    loading train or validation never returns a holdout record.
    """
    if split not in SPLITS:
        raise DatasetError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split == "holdout" and not include_holdout:
        raise DatasetError(
            "holdout is protected. Pass include_holdout=True to read it, and "
            "only when it is being scored - not for tuning or wording.")

    path = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        raise DatasetError(f"dataset split not found: {path}")

    # Resolve the system context once, from the first record's reference, then
    # verify every record against it.
    records = list(_read_jsonl(path))
    if not records:
        raise DatasetError(f"{path} is empty")

    ref = records[0][1].get("system_ref")
    system = resolve_system_context(ref, builder=builder)

    evaluation_only = split == "holdout"
    out: list[LoadedExample] = []
    failures: list[str] = []

    for lineno, raw in records:
        rid = raw.get("id") if isinstance(raw, dict) else f"line {lineno}"
        problems = validate_record(
            raw, expected_split=split, system=system,
            evaluation_only=evaluation_only)
        if problems:
            for problem in problems:
                failures.append(f"{path}:{lineno} [{rid}]: {problem}")
            if strict:
                raise DatasetError(
                    f"{len(failures)} problem(s) in {split}; first: {failures[0]}")
            continue

        messages = [{"role": "system", "content": system.content}]
        messages.extend({"role": m["role"], "content": m["content"]}
                        for m in raw["messages"])

        out.append(LoadedExample(
            id=raw["id"],
            seed_id=raw["seed_id"],
            category=raw["category"],
            split=raw["split"],
            messages=messages,
            train_on=tuple(raw["train_on"]),
            system_fingerprint=raw["system_fingerprint"],
            schema_version=raw["schema_version"],
            generator_version=raw["generator_version"],
            expect_block=bool(raw.get("expect_block", False)),
            expect_action=raw.get("expect_action"),
            note=raw.get("note", ""),
            evaluation_only=evaluation_only,
        ))

    # Deterministic ordering regardless of file order.
    out.sort(key=lambda e: e.id)
    return out


def load_training_set(**kwargs) -> list[LoadedExample]:
    """Train plus validation. Never includes the holdout."""
    return load_split("train", **kwargs) + load_split("validation", **kwargs)


# --------------------------------------------------------------------------
# Inspection entry point
# --------------------------------------------------------------------------


def inspect_split(split: str, *, data_dir: str = DEFAULT_OUTDIR,
                  include_holdout: bool = False) -> dict[str, Any]:
    """Report on a split without returning training examples."""
    path = os.path.join(data_dir, f"{split}.jsonl")
    report: dict[str, Any] = {
        "split": split,
        "path": path,
        "records": 0,
        "training_targets": 0,
        "malformed": [],
        "errors": [],
        "train_on_distribution": {},
        "system_fingerprint": None,
        "system_ref": None,
    }
    if not os.path.exists(path):
        report["errors"].append(f"missing file: {path}")
        return report

    try:
        rows = [(n, r) for n, r in _read_jsonl(path)]
    except DatasetError as exc:
        report["errors"].append(str(exc))
        return report

    report["records"] = len(rows)

    # Shape-only validation first, so a fingerprint mismatch cannot mask a
    # structural problem.
    system = None
    try:
        ref = rows[0][1].get("system_ref") if isinstance(rows[0][1], dict) else None
        system = resolve_system_context(ref)
    except DatasetError as exc:
        report["errors"].append(str(exc))

    for lineno, raw in rows:
        rid = raw.get("id") if isinstance(raw, dict) else f"line {lineno}"
        if not isinstance(raw, dict):
            report["malformed"].append(f"{path}:{lineno}: not a JSON object")
            continue
        problems = validate_record(raw, expected_split=split, system=system)
        if problems:
            report["malformed"].append(
                f"{path}:{lineno} [{rid}]: {problems[0]}")
            continue
        assistants = [m for m in raw["messages"] if m["role"] == "assistant"]
        trained = len(raw["train_on"])
        report["training_targets"] += trained
        key = str(raw["train_on"])
        report["train_on_distribution"][key] = \
            report["train_on_distribution"].get(key, 0) + 1
        if report["system_fingerprint"] is None:
            report["system_fingerprint"] = raw["system_fingerprint"]
            report["system_ref"] = raw["system_ref"]
        if raw.get("expect_block"):
            report.setdefault("emit_records", 0)
            report["emit_records"] += 1
        del assistants

    return report


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    include_holdout = "--holdout" in argv
    names = [a for a in argv if not a.startswith("-")] or ["train", "validation"]

    print("=" * 72)
    print("AURORA BRAIN v0.1 - v2 DATASET LOADER")
    print("=" * 72)

    total_errors = 0
    for name in names:
        report = inspect_split(name, include_holdout=include_holdout)
        print(f"  split              : {report['split']}")
        print(f"  records            : {report['records']}")
        print(f"  training targets   : {report['training_targets']}")
        print(f"  system_ref         : {report['system_ref']}")
        print(f"  system fingerprint : {report['system_fingerprint']}")
        print(f"  train_on shapes    : {report['train_on_distribution']}")
        print(f"  malformed records  : {len(report['malformed'])}")
        print(f"  validation errors  : {len(report['errors'])}")
        for err in report["errors"][:5]:
            print(f"      ERROR {err}")
        for bad in report["malformed"][:5]:
            print(f"      MALFORMED {bad}")
        print()
        total_errors += len(report["errors"]) + len(report["malformed"])

    print(f"  total problems     : {total_errors}")
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
