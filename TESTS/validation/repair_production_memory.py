"""One-off, reversible repair of production memory pollution.

The anaphora-overreach defect stored "compare there engine" as the favorite
car and "you are useless" as the favorite music genre/song; the bare-
correction defect overwrote the favorite soccer team with a complaint clause.
Those rows are deleted here, and ``favorite color = yellow`` is restored from
the user's own message 18 ("its yellow"), which predates anaphora support.

Dry-run by default. ``--apply`` takes a full SQLite backup, writes the repair
in one transaction, and emits a rollback SQL file containing the exact rows
that were removed and the id of the row that was added.

    python TESTS\\validation\\repair_production_memory.py
    python TESTS\\validation\\repair_production_memory.py --apply
    python TESTS\\validation\\repair_production_memory.py --db COPY.db

This script is not collected by pytest (no ``test_`` prefix).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PRIMORA.prime.context.isolation import production_db_path  # noqa: E402

ARTIFACTS = (
    (11, "favorite soccer team", "i meant based that i have told you before"),
    (14, "favorite soccer team",
     "your info are wrong its 2026 and its has not releaseed yet"),
    (15, "favorite car", "compare there engine"),
    (16, "favorite music genre and song", "you are useless"),
)
COLOR_SOURCE_ID = 18
COLOR_CONCEPT = "favorite color"
COLOR_VALUE = "yellow"
COLOR_SESSION = "default"


def _connect(path: Path, readonly: bool) -> sqlite3.Connection:
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        con = sqlite3.connect(str(path), timeout=10)
    con.row_factory = sqlite3.Row
    return con


def _state_rows(con: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in con.execute("SELECT * FROM states ORDER BY id")]


def _verify(con: sqlite3.Connection) -> dict:
    problems: list[str] = []
    rows = {r["id"]: r for r in _state_rows(con)}
    for state_id, concept, value in ARTIFACTS:
        row = rows.get(state_id)
        if row is None:
            problems.append(f"state {state_id} missing")
        elif (row["concept"], row["value"]) != (concept, value):
            problems.append(
                f"state {state_id} changed: {(row['concept'], row['value'])!r}"
            )
    color = [
        r for r in rows.values()
        if r["concept"] == COLOR_CONCEPT
    ]
    if color:
        problems.append(f"favorite color already exists: {color!r}")
    msg = con.execute(
        "SELECT id, session_id, role, content, timestamp FROM messages"
        " WHERE id = ?", (COLOR_SOURCE_ID,)
    ).fetchone()
    if msg is None:
        problems.append(f"message {COLOR_SOURCE_ID} missing")
    elif (msg["content"], msg["role"], msg["session_id"]) != (
            "its yellow", "user", COLOR_SESSION):
        problems.append(
            f"message {COLOR_SOURCE_ID} unexpected: {dict(msg)!r}"
        )
    if problems:
        raise SystemExit("ABORT - preconditions failed:\n  " + "\n  ".join(
            problems))
    return {
        "artifacts": [rows[i] for i, _, _ in ARTIFACTS],
        "color_source": dict(msg),
        "before": rows,
    }


def _backup(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = Path(tempfile.gettempdir()) / "opencode" / "memreg" / (
        f"aurora_context_pre_repair_{stamp}.db")
    target.parent.mkdir(parents=True, exist_ok=True)
    src = _connect(path, readonly=True)
    dst = sqlite3.connect(str(target))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return target


def _insert_sql(row: dict) -> str:
    def quote(value):
        if value is None:
            return "NULL"
        if isinstance(value, (int, float)):
            return repr(value)
        return "'" + str(value).replace("'", "''") + "'"

    columns = ", ".join(row)
    values = ", ".join(quote(v) for v in row.values())
    return f"INSERT INTO states ({columns}) VALUES ({values});"


def _write_rollback(path: Path, state_ids: list[int], new_id: int,
                    before: dict[int, dict]) -> None:
    lines = [
        "-- Rollback for TESTS/validation/repair_production_memory.py",
        f"-- Restores the rows deleted by this repair and removes the"
        f" backfilled state {new_id}.",
        "BEGIN IMMEDIATE;",
        f"DELETE FROM states WHERE id = {new_id};",
    ]
    lines.extend(_insert_sql(before[i]) for i in state_ids)
    lines.append("COMMIT;")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="write the repair (default: dry run)")
    parser.add_argument("--db", type=Path, default=None,
                        help="database path (default: production memory)")
    args = parser.parse_args()

    path = args.db or production_db_path()
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"{mode}: {path}")

    con = _connect(path, readonly=not args.apply)
    try:
        verified = _verify(con)
        before_rows = verified["before"]
        print("planned deletes:")
        for row in verified["artifacts"]:
            print(f"  state {row['id']}: {row['concept']} = {row['value']!r}"
                  f" [{row['status']}]")
        print(f"planned insert: {COLOR_CONCEPT} = {COLOR_VALUE!r}"
              f" (scope global, source message {COLOR_SOURCE_ID},"
              f" observed {verified['color_source']['timestamp']})")
        if not args.apply:
            print("dry run complete; pass --apply to write")
            return 0

        backup = _backup(path)
        print(f"backup: {backup}")

        con.close()
        con = _connect(path, readonly=False)
        with con:
            con.execute("BEGIN IMMEDIATE")
            now = time.time()
            ids = [r["id"] for r in verified["artifacts"]]
            placeholders = ",".join("?" for _ in ids)
            deleted = con.execute(
                f"DELETE FROM states WHERE id IN ({placeholders})", ids
            ).rowcount
            if deleted != len(ids):
                raise SystemExit(
                    f"ABORT - deleted {deleted} of {len(ids)} rows")
            cur = con.execute(
                "INSERT INTO states"
                "(session_id, scope, concept, value, source_ids, created_at,"
                " status, supersedes_id, authority, memory_type, confidence,"
                " observed_at, valid_from, valid_until, superseded_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 'current', NULL, 'user', 'FACT',"
                " 1.0, ?, ?, NULL, NULL)",
                (
                    COLOR_SESSION, "global", COLOR_CONCEPT, COLOR_VALUE,
                    str(COLOR_SOURCE_ID), now,
                    verified["color_source"]["timestamp"],
                    verified["color_source"]["timestamp"],
                ),
            )
            new_id = int(cur.lastrowid)
            if con.execute("SELECT COUNT(*) FROM states WHERE id IN"
                           f" ({placeholders})", ids).fetchone()[0] != 0:
                raise SystemExit("ABORT - deleted rows still present")
            color = con.execute(
                "SELECT id, value, status FROM states WHERE concept = ?"
                " AND scope = 'global'", (COLOR_CONCEPT,)
            ).fetchall()
            if len(color) != 1 or color[0]["value"] != COLOR_VALUE:
                raise SystemExit(f"ABORT - color check failed: {color!r}")
        con.commit()

        after = {r["id"]: r for r in _state_rows(con)}
        rollback = backup.with_name(backup.stem + "_rollback.sql")
        _write_rollback(rollback, ids, new_id, before_rows)
        manifest = {
            "db": str(path),
            "backup": str(backup),
            "rollback": str(rollback),
            "deleted": verified["artifacts"],
            "inserted": after[new_id],
            "states_before": len(before_rows),
            "states_after": len(after),
        }
        manifest_path = backup.with_name(backup.stem + "_manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        print(f"inserted state {new_id}: {COLOR_CONCEPT} = {COLOR_VALUE!r}")
        print(f"rollback: {rollback}")
        print(f"manifest: {manifest_path}")
        print("repair applied")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
