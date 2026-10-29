"""Read-only audit and explicit, backed-up retirement of malformed legacy states.

This migration never infers a replacement fact and never deletes source text.
Run preview before the user-authorized --apply; a SQLite backup is mandatory.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

from .store import ContextStore


def suspicious_preference(state) -> str | None:
    """Flag sentence-shaped subject fields; no topic-specific vocabulary."""
    if not state.concept.startswith("likes for "):
        return None
    subject = state.concept[len("likes for "):].strip()
    if re.search(
        r"\b(?:i|we|you|they)\s+(?:love|like|adore|enjoy|"
        r"prefer|hate|dislike|think|feel|say)\b",
        subject, re.I,
    ):
        return "subject_contains_conversational_clause"
    if len(subject) > 90 or len(subject.split()) > 12:
        return "subject_exceeds_conservative_entity_limit"
    return None


def preview(store: ContextStore) -> list[dict]:
    result = []
    for state in store.fetch_states(status="current"):
        reason = suspicious_preference(state)
        if reason:
            result.append({
                "state_id": state.state_id, "scope": state.scope,
                "concept": state.concept, "value": state.value,
                "source_ids": list(state.source_ids), "reason": reason,
            })
    return result


def apply_reviewed(db_path: Path, state_ids: list[int]) -> dict:
    """Retire only explicitly reviewed IDs with an atomic SQLite backup."""
    if not state_ids:
        return {"backed_up": None, "retired": 0}
    db_path = Path(db_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    store = ContextStore(db_path)
    flagged = {record["state_id"] for record in preview(store)}
    ids = list(dict.fromkeys(int(i) for i in state_ids))
    if not set(ids) <= flagged:
        raise ValueError("state ids changed or are not reviewable; rerun preview")
    backup = db_path.with_name(db_path.name + ".before-m13.sqlite")
    if backup.exists():
        raise FileExistsError(
            f"refusing to overwrite existing migration backup: {backup}"
        )
    # Backup through SQLite to capture committed WAL contents consistently.
    with sqlite3.connect(str(db_path)) as source:
        with sqlite3.connect(str(backup)) as destination:
            source.backup(destination)
    # The repository's existing state lifecycle, never direct SQL deletion.
    store.supersede_states(ids)
    return {"backed_up": str(backup), "retired": len(ids)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--state-ids", type=int, nargs="+")
    parser.add_argument("--confirm-reviewed", action="store_true")
    args = parser.parse_args(argv)
    if not args.db.exists():
        parser.error("database file not found")
    store = ContextStore(args.db)
    report = preview(store)
    print(json.dumps({"flagged": report, "count": len(report)}, indent=2))
    if args.apply:
        if not args.confirm_reviewed or not args.state_ids:
            parser.error("--apply requires --confirm-reviewed and --state-ids")
        print(json.dumps(apply_reviewed(args.db, args.state_ids), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
