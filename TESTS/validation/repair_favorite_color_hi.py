"""Repair the production favorite-color state overwritten by a greeting.

State 18 was created from message 6031 ("hi") because the reconciler carried
the prior slot question across an assistant turn. The user's actual statement
is message 18 ("its yellow"). This script corrects state 18 in place, backed
by a full SQLite backup and an exact rollback file.

Dry-run by default:

    python TESTS\\validation\\repair_favorite_color_hi.py
    python TESTS\\validation\\repair_favorite_color_hi.py --apply
    python TESTS\\validation\\repair_favorite_color_hi.py --db COPY.db --apply

This script is not collected by pytest (no ``test_`` prefix).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PRIMORA.prime.context.isolation import (  # noqa: E402
    MEMORY_TABLES,
    production_db_path,
    production_digest,
)

TARGET_ID = 18
TARGET_CONCEPT = "favorite color"
TARGET_BAD_VALUE = "hi"
TARGET_GOOD_VALUE = "yellow"
BAD_SOURCE_ID = 6031
GOOD_SOURCE_ID = 18
SESSION_ID = "default"


def _connect(path: Path, readonly: bool) -> sqlite3.Connection:
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        con = sqlite3.connect(str(path), timeout=10)
    con.row_factory = sqlite3.Row
    return con


def _state_rows(con: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in con.execute("SELECT * FROM states ORDER BY id")]


def _messages(con: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in con.execute("SELECT * FROM messages ORDER BY id")]


def _content_digest(path: Path) -> str:
    con = _connect(path, readonly=True)
    digest = hashlib.sha256()
    try:
        for table in MEMORY_TABLES:
            try:
                rows = con.execute(
                    f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            except sqlite3.OperationalError:
                continue
            digest.update(table.encode("utf-8"))
            for row in rows:
                digest.update(repr(tuple(row)).encode("utf-8"))
    finally:
        con.close()
    return digest.hexdigest()


def _message(con: sqlite3.Connection, message_id: int) -> dict:
    row = con.execute(
        "SELECT * FROM messages WHERE id = ?", (message_id,)
    ).fetchone()
    return dict(row) if row is not None else {}


def _verify_before(con: sqlite3.Connection) -> dict:
    problems: list[str] = []
    states = _state_rows(con)
    rows = {row["id"]: row for row in states}
    target = rows.get(TARGET_ID)
    if target is None:
        problems.append(f"state {TARGET_ID} missing")
    else:
        if (target["concept"], target["value"], target["status"]) != (
                TARGET_CONCEPT, TARGET_BAD_VALUE, "current"):
            problems.append(f"state {TARGET_ID} unexpected: {target!r}")
        target_sources = {
            int(value) for value in re.findall(
                r"\d+", str(target["source_ids"]))
        }
        if target_sources != {BAD_SOURCE_ID}:
            problems.append(
                f"state {TARGET_ID} source is not {BAD_SOURCE_ID}: {target!r}")
    current_color = [
        row for row in states
        if row["concept"] == TARGET_CONCEPT and row["status"] == "current"
    ]
    if [row["id"] for row in current_color] != [TARGET_ID]:
        problems.append(f"current color rows are not [{TARGET_ID}]: "
                        f"{current_color!r}")
    bad_source = _message(con, BAD_SOURCE_ID)
    if (bad_source.get("role"), bad_source.get("content")) != ("user", "hi"):
        problems.append(f"message {BAD_SOURCE_ID} unexpected: {bad_source!r}")
    good_source = _message(con, GOOD_SOURCE_ID)
    if (good_source.get("role"), good_source.get("content"),
            good_source.get("session_id")) != (
            "user", "its yellow", SESSION_ID):
        problems.append(f"message {GOOD_SOURCE_ID} unexpected: {good_source!r}")
    if problems:
        raise SystemExit("ABORT - preconditions failed:\n  " + "\n  ".join(
            problems))
    return {
        "target": target,
        "states": states,
        "messages": _messages(con),
        "good_source": good_source,
    }


def _backup(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = Path(tempfile.gettempdir()) / "opencode" / "memreg" / (
        f"aurora_context_pre_color_repair_{stamp}.db")
    target.parent.mkdir(parents=True, exist_ok=True)
    src = _connect(path, readonly=True)
    dst = sqlite3.connect(str(target))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return target


def _quote(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _update_sql(row: dict) -> str:
    columns = [column for column in row if column != "id"]
    assignments = ", ".join(
        f"{column} = {_quote(row[column])}" for column in columns)
    return f"UPDATE states SET {assignments} WHERE id = {row['id']};"


def _write_rollback(path: Path, before: dict) -> None:
    path.write_text(
        "\n".join([
            "-- Rollback for TESTS/validation/repair_favorite_color_hi.py",
            f"-- Restores state {before['id']} exactly as it was before repair.",
            "BEGIN IMMEDIATE;",
            _update_sql(before),
            "COMMIT;",
        ]) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="write the repair (default: dry run)")
    parser.add_argument("--db", type=Path, default=None,
                        help="database path (default: production memory)")
    args = parser.parse_args()

    path = (args.db or production_db_path()).resolve()
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"{mode}: {path}")

    con = _connect(path, readonly=not args.apply)
    try:
        verified = _verify_before(con)
        target = verified["target"]
        before_digest = _content_digest(path)
        unrelated_before = [
            row for row in verified["states"] if row["id"] != TARGET_ID
        ]
        print(f"planned update: state {TARGET_ID}: "
              f"{target['concept']} = {target['value']!r} -> "
              f"{TARGET_GOOD_VALUE!r}")
        print(f"source: message {GOOD_SOURCE_ID} "
              f"({verified['good_source']['content']!r})")
        print(f"content digest before: {before_digest}")
        if not args.apply:
            print("dry run complete; pass --apply to write")
            return 0

        backup = _backup(path)
        print(f"backup: {backup}")

        con.close()
        con = _connect(path, readonly=False)
        with con:
            con.execute("BEGIN IMMEDIATE")
            observed = verified["good_source"]["timestamp"]
            current = con.execute(
                "UPDATE states SET value = ?, source_ids = ?,"
                " observed_at = ?, valid_from = ?, valid_until = NULL,"
                " superseded_at = NULL"
                " WHERE id = ? AND status = 'current' AND value = ?",
                (
                    TARGET_GOOD_VALUE, str(GOOD_SOURCE_ID), observed, observed,
                    TARGET_ID, TARGET_BAD_VALUE,
                ),
            )
            if current.rowcount != 1:
                raise SystemExit("ABORT - target state changed during repair")
            color = con.execute(
                "SELECT id, value, status, source_ids FROM states"
                " WHERE concept = ? AND status = 'current'",
                (TARGET_CONCEPT,),
            ).fetchall()
            if len(color) != 1 or color[0]["value"] != TARGET_GOOD_VALUE:
                raise SystemExit(f"ABORT - color check failed: {color!r}")
            if str(color[0]["source_ids"]) != str(GOOD_SOURCE_ID):
                raise SystemExit(
                    f"ABORT - color source check failed: {color!r}")
        con.commit()

        after_states = _state_rows(con)
        after_messages = _messages(con)
        before_by_id = {row["id"]: row for row in verified["states"]}
        after_by_id = {row["id"]: row for row in after_states}
        unrelated_after = [
            row for row in after_states if row["id"] != TARGET_ID
        ]
        unrelated_ok = unrelated_after == unrelated_before
        messages_ok = after_messages == verified["messages"]
        target_after = after_by_id[TARGET_ID]
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        before_count = len(verified["states"])
        after_count = len(after_states)
        if not unrelated_ok:
            raise SystemExit("ABORT - unrelated states changed")
        if not messages_ok:
            raise SystemExit("ABORT - messages changed")
        if integrity != "ok":
            raise SystemExit(f"ABORT - integrity check: {integrity}")
        if before_count != after_count:
            raise SystemExit("ABORT - state count changed")

        after_digest = _content_digest(path)
        rollback = backup.with_name(backup.stem + "_rollback.sql")
        _write_rollback(rollback, before_by_id[TARGET_ID])
        manifest = {
            "db": str(path),
            "backup": str(backup),
            "rollback": str(rollback),
            "before": before_by_id[TARGET_ID],
            "after": target_after,
            "digest_before": before_digest,
            "digest_after": after_digest,
            "production_digest": (
                production_digest()
                if path == production_db_path().resolve() else None
            ),
            "states": {"before": before_count, "after": after_count},
            "unrelated_states_unchanged": unrelated_ok,
            "messages_unchanged": messages_ok,
            "integrity_check": integrity,
        }
        manifest_path = backup.with_name(backup.stem + "_manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        print(f"updated state {TARGET_ID}: "
              f"{TARGET_CONCEPT} = {TARGET_GOOD_VALUE!r}")
        print(f"content digest after: {after_digest}")
        if manifest["production_digest"]:
            print(f"production digest: {manifest['production_digest']}")
        print(f"rollback: {rollback}")
        print(f"manifest: {manifest_path}")
        print("repair applied")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
