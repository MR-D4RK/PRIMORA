"""Read-only preview and explicit import for legacy/Memory V2 stores.

The source database is opened with SQLite ``mode=ro`` and is never passed to
``ContextStore``.  Import writes only to a caller-supplied destination that
passes :func:`require_isolated`.  Database-local integer ids are remapped as
rows are copied; every source reference is rewritten through those maps and a
missing reference is reported instead of guessed.
"""

from __future__ import annotations

import json
import argparse
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .isolation import production_db_path, require_isolated
from .store import ContextStore


TABLES = (
    "sessions", "messages", "digests", "experience_events", "states",
    "candidates", "entities", "entity_aliases", "relationships",
    "observations", "observation_sources", "observation_history",
    "formation_status", "formation_jobs", "retrieval_vectors",
    "retrieval_index_meta",
)

_OPTIONAL_TABLES = {"digests", "experience_events", "candidates", "entities",
                    "entity_aliases", "relationships", "observations",
                    "observation_sources", "observation_history",
                    "formation_status", "retrieval_vectors",
                    "formation_jobs", "retrieval_index_meta"}

_REQUIRED_COLUMNS = {
    "sessions": {"session_id", "created_at", "last_activity_at"},
    "messages": {"id", "session_id", "seq", "role", "content", "timestamp", "fingerprint"},
    "states": {"id", "session_id", "scope", "concept", "value", "source_ids", "status"},
    # Identity and foreign-key anchors used by each importer. Newer optional
    # fields are copied with defaults; missing anchors are reported as
    # unsupported instead of failing halfway through an atomic import.
    "digests": {"id", "session_id", "source_ids", "seq_start", "seq_end", "created_at", "digest_type", "content"},
    "experience_events": {"id", "session_id", "source_message_id", "kind", "actor", "statement", "reported_at"},
    "candidates": {"id", "session_id", "scope", "source_ids", "status"},
    "entities": {"id", "session_id", "scope", "entity_type", "display_name", "normalized_name", "status", "source_ids"},
    "entity_aliases": {"id", "session_id", "scope", "entity_id", "alias_text", "normalized_alias", "status", "source_ids"},
    "relationships": {"id", "session_id", "scope", "relationship_type", "subject_kind", "subject_id", "object_kind", "object_id", "status", "source_ids"},
    "observations": {"id", "session_id", "scope", "concept", "normalized_concept", "value", "status"},
    "observation_sources": {"id", "observation_id"},
    "observation_history": {"id", "observation_id", "snapshot_json"},
    "formation_status": {"message_id"},
    "formation_jobs": {
        "job_id", "message_id", "session_id", "status", "attempts",
        "last_error", "created_at", "updated_at",
    },
}


@dataclass(frozen=True)
class RestoreIssue:
    table: str
    source_id: Optional[int]
    reason: str
    severity: str = "unsupported"
    target_id: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table, "source_id": self.source_id,
            "reason": self.reason, "severity": self.severity,
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class RestorePreview:
    source_path: str
    target_path: Optional[str]
    scanned: dict[str, int] = field(default_factory=dict)
    transferable: dict[str, int] = field(default_factory=dict)
    already_present: dict[str, int] = field(default_factory=dict)
    conflicts: list[RestoreIssue] = field(default_factory=list)
    unsupported: list[RestoreIssue] = field(default_factory=list)
    derived: list[RestoreIssue] = field(default_factory=list)

    @property
    def blocking_issues(self) -> list[RestoreIssue]:
        """Issues that must stop an import before a destination write."""
        return [*self.conflicts, *self.unsupported]

    @property
    def safe_to_import(self) -> bool:
        return not self.blocking_issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "target_path": self.target_path,
            "scanned": dict(self.scanned),
            "transferable": dict(self.transferable),
            "already_present": dict(self.already_present),
            "conflicts": [item.as_dict() for item in self.conflicts],
            "unsupported": [item.as_dict() for item in self.unsupported],
            "derived": [item.as_dict() for item in self.derived],
            "blocking_issues": [item.as_dict() for item in self.blocking_issues],
            "safe_to_import": self.safe_to_import,
        }


@dataclass(frozen=True)
class RestoreResult:
    source_path: str
    target_path: str
    imported: dict[str, int] = field(default_factory=dict)
    already_present: dict[str, int] = field(default_factory=dict)
    conflicts: list[RestoreIssue] = field(default_factory=list)
    unsupported: list[RestoreIssue] = field(default_factory=list)
    id_maps: dict[str, dict[int, int]] = field(default_factory=dict)
    derived: list[RestoreIssue] = field(default_factory=list)

    @property
    def blocking_issues(self) -> list[RestoreIssue]:
        return [*self.conflicts, *self.unsupported]

    @property
    def changed(self) -> bool:
        return any(self.imported.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "target_path": self.target_path,
            "imported": dict(self.imported),
            "already_present": dict(self.already_present),
            "conflicts": [item.as_dict() for item in self.conflicts],
            "unsupported": [item.as_dict() for item in self.unsupported],
            "derived": [item.as_dict() for item in self.derived],
            "blocking_issues": [item.as_dict() for item in self.blocking_issues],
            "id_maps": {
                table: {str(source): target for source, target in mapping.items()}
                for table, mapping in self.id_maps.items()
            },
            "changed": self.changed,
        }


class _ImportConflict(RuntimeError):
    """Internal signal used to roll back a race-discovered conflict."""

    def __init__(self, issue: RestoreIssue):
        super().__init__(issue.reason)
        self.issue = issue


def _read_only(path: Path) -> sqlite3.Connection:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(
        f"file:{path.as_posix()}?mode=ro", uri=True,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }


def _rows(connection: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    return list(connection.execute(f"SELECT * FROM {table}"))


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _supported(connection: sqlite3.Connection, table: str) -> bool:
    """Whether a source table has the columns needed by its importer."""
    return (
        table in _tables(connection)
        and not (_REQUIRED_COLUMNS.get(table, set()) - _columns(connection, table))
    )


def _value(row: sqlite3.Row, name: str, default: Any = None) -> Any:
    return row[name] if name in row.keys() else default


def _event_value(row: sqlite3.Row, name: str) -> Any:
    defaults = {
        "authority": "user",
        "occurred_precision": "unknown",
        "occurred_timezone": "UTC",
    }
    return _value(row, name, defaults.get(name))


def _ids(raw: Any) -> list[int]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, (list, tuple)):
        values = raw
    else:
        values = str(raw).split(",")
    return [int(value) for value in values if str(value).strip()]


def _csv(values: list[int]) -> str:
    return ",".join(str(int(value)) for value in dict.fromkeys(values))


def _map_ids(raw: Any, mapping: dict[int, int]) -> tuple[list[int], Optional[str]]:
    try:
        source_ids = _ids(raw)
    except (TypeError, ValueError):
        return [], "malformed source id list"
    missing = [value for value in source_ids if value not in mapping]
    if missing:
        return [], f"missing source message mapping: {missing[:5]}"
    return [mapping[value] for value in source_ids], None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _dedupe_issues(items: list[RestoreIssue]) -> list[RestoreIssue]:
    seen: set[RestoreIssue] = set()
    output: list[RestoreIssue] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _target_message_for_source(
    source: sqlite3.Connection,
    target_rows: dict[str, list[sqlite3.Row]],
    source_message_id: Any,
) -> Optional[int]:
    """Resolve a source message to its destination id without writing."""
    source_message = source.execute(
        "SELECT session_id,seq,fingerprint FROM messages WHERE id=?",
        (source_message_id,),
    ).fetchone()
    if source_message is None:
        return None
    for target in target_rows.get("messages", []):
        if (target["session_id"], target["seq"], target["fingerprint"]) == (
            source_message["session_id"], source_message["seq"], source_message["fingerprint"],
        ):
            return int(target["id"])
    return None


def _target_snapshot(path: Optional[Path]) -> dict[str, list[sqlite3.Row]]:
    if path is None or not path.exists():
        return {}
    connection = _read_only(path)
    try:
        available = _tables(connection)
        return {table: _rows(connection, table) for table in TABLES if table in available}
    finally:
        connection.close()


class MemoryRestoreService:
    """Preview or explicitly import one store into an isolated destination."""

    def __init__(self, source_path: Path | str, target_path: Path | str):
        self.source_path = Path(source_path).resolve()
        self.target_path = require_isolated(target_path)
        if self.source_path == self.target_path:
            raise ValueError("source and target databases must be different")
        # Path resolution does not distinguish hard links. Refuse an existing
        # destination that aliases either the source or production database,
        # so an explicit import can never write through a second pathname.
        if self.target_path.exists():
            for protected in (self.source_path, production_db_path()):
                try:
                    same_file = protected.exists() and self.target_path.samefile(protected)
                except OSError:
                    same_file = False
                if same_file:
                    raise ValueError(
                        "source, destination, and production databases must be distinct"
                    )

    def preview(self) -> RestorePreview:
        source = _read_only(self.source_path)
        target_rows = _target_snapshot(self.target_path)
        conflicts: list[RestoreIssue] = []
        unsupported: list[RestoreIssue] = []
        scanned: dict[str, int] = {}
        transferable: dict[str, int] = {}
        already: dict[str, int] = {}
        derived: list[RestoreIssue] = []
        try:
            available = _tables(source)
            for table in TABLES:
                if table not in available:
                    if table not in _OPTIONAL_TABLES:
                        unsupported.append(RestoreIssue(table, None, "required table is missing"))
                    continue
                missing = _REQUIRED_COLUMNS.get(table, set()) - _columns(source, table)
                if missing:
                    unsupported.append(RestoreIssue(
                        table, None,
                        "missing required columns: " + ", ".join(sorted(missing)),
                    ))
                    continue
                rows = _rows(source, table)
                scanned[table] = len(rows)
                if table in {"retrieval_vectors", "retrieval_index_meta"}:
                    derived.extend(
                        RestoreIssue(table, _value(row, "vector_id", _value(row, "key")),
                                     "derived retrieval index is rebuilt, not imported",
                                     "derived")
                        for row in rows
                    )
                    continue
                for row in rows:
                    source_id = _value(row, "id", _value(row, "message_id"))
                    issue = self._classify_row(table, row, target_rows, source)
                    if issue is not None:
                        (conflicts if issue.severity == "conflict" else unsupported).append(issue)
                    elif self._already_present(table, row, target_rows):
                        already[table] = already.get(table, 0) + 1
                    else:
                        transferable[table] = transferable.get(table, 0) + 1
            return RestorePreview(
                str(self.source_path), str(self.target_path), scanned,
                transferable, already, conflicts, unsupported, derived,
            )
        finally:
            source.close()

    def _classify_row(
        self, table: str, row: sqlite3.Row,
        target_rows: dict[str, list[sqlite3.Row]], source: sqlite3.Connection,
    ) -> Optional[RestoreIssue]:
        source_id = _value(row, "id", _value(row, "message_id"))
        if table == "messages":
            try:
                fp = str(_value(row, "fingerprint", ""))
                source_duplicate = source.execute(
                    "SELECT id FROM messages WHERE session_id=? AND seq=? "
                    "AND id<>? AND fingerprint<>? LIMIT 1",
                    (row["session_id"], row["seq"], row["id"], fp),
                ).fetchone()
                if source_duplicate is not None:
                    return RestoreIssue(
                        table, source_id,
                        "source contains different content at the same session/sequence",
                        "conflict", int(source_duplicate[0]),
                    )
                for target in target_rows.get("messages", []):
                    if (target["session_id"], target["seq"]) == (
                        row["session_id"], row["seq"]
                    ) and target["fingerprint"] != fp:
                        return RestoreIssue(table, source_id,
                                            "same session/sequence has different content",
                                            "conflict", target["id"])
            except (KeyError, TypeError):
                return RestoreIssue(table, source_id, "message schema is unsupported")
            return None
        if table == "sessions":
            for target in target_rows.get("sessions", []):
                if target["session_id"] != row["session_id"]:
                    continue
                try:
                    differing = (
                        _norm(target["title"]) != _norm(_value(row, "title", ""))
                        or target["status"] != _value(row, "status", "active")
                        or float(target["created_at"]) != float(row["created_at"])
                    )
                except (KeyError, TypeError, ValueError):
                    return RestoreIssue(table, None, "session schema or timestamp is unsupported")
                if differing:
                    return RestoreIssue(
                        table, None,
                        "session metadata conflicts with destination",
                        "conflict",
                    )
                break
            return None
        if table == "states":
            try:
                source_ids = _ids(_value(row, "source_ids", ""))
            except (TypeError, ValueError):
                return RestoreIssue(table, source_id, "malformed state source ids")
            for message_id in source_ids:
                if source.execute("SELECT 1 FROM messages WHERE id=?", (message_id,)).fetchone() is None:
                    return RestoreIssue(table, source_id, "state source message is missing")
            predecessor = _value(row, "supersedes_id")
            if predecessor is not None and source.execute(
                "SELECT 1 FROM states WHERE id=?", (predecessor,)
            ).fetchone() is None:
                return RestoreIssue(table, source_id, "superseded predecessor is missing")
            if _value(row, "status", "current") == "current" and _value(row, "authority", "user") != "user":
                return RestoreIssue(table, source_id,
                                    "current non-user state cannot be admitted",
                                    "unsupported")
            for target in target_rows.get("states", []):
                if (target["scope"], _norm(target["concept"])) == (
                    row["scope"], _norm(row["concept"])
                ) and target["status"] == "current" and row["status"] == "current" \
                        and not (_value(row, "memory_type", "")
                                 and _value(target, "memory_type", "")
                                 and _value(row, "memory_type", "") != _value(target, "memory_type", "")) \
                        and _norm(target["value"]) != _norm(row["value"]):
                    return RestoreIssue(table, source_id,
                                        "current state conflicts with destination",
                                        "conflict", target["id"])
            return None
        if table in {"entities", "entity_aliases", "relationships", "observations"}:
            try:
                source_ids = _ids(_value(row, "source_ids", ""))
            except (TypeError, ValueError):
                return RestoreIssue(table, source_id, "malformed source ids")
            for message_id in source_ids:
                if source.execute("SELECT 1 FROM messages WHERE id=?", (message_id,)).fetchone() is None:
                    return RestoreIssue(table, source_id, "source message is missing")
        if table == "candidates":
            try:
                source_ids = _ids(_value(row, "source_ids", ""))
            except (TypeError, ValueError):
                return RestoreIssue(table, source_id, "malformed candidate source ids")
            for message_id in source_ids:
                if source.execute("SELECT 1 FROM messages WHERE id=?", (message_id,)).fetchone() is None:
                    return RestoreIssue(table, source_id, "candidate source message is missing")
        if table == "entities":
            merged = _value(row, "merged_into_entity_id")
            if merged is not None and source.execute(
                "SELECT 1 FROM entities WHERE id=?", (merged,)
            ).fetchone() is None:
                return RestoreIssue(table, source_id, "merged entity target is missing")
        if table == "entity_aliases":
            if source.execute(
                "SELECT 1 FROM entities WHERE id=?", (row["entity_id"],)
            ).fetchone() is None:
                return RestoreIssue(table, source_id, "alias entity is missing")
        if table == "relationships":
            for kind, endpoint in (
                (row["subject_kind"], row["subject_id"]),
                (row["object_kind"], row["object_id"]),
            ):
                endpoint_table = {
                    "entity": "entities", "memory_state": "states",
                    "observation": "observations",
                }.get(kind)
                if endpoint_table is None or source.execute(
                    f"SELECT 1 FROM {endpoint_table} WHERE id=?", (endpoint,)
                ).fetchone() is None:
                    return RestoreIssue(table, source_id, "relationship endpoint is missing")
            predecessor = _value(row, "supersedes_id")
            if predecessor is not None and source.execute(
                "SELECT 1 FROM relationships WHERE id=?", (predecessor,)
            ).fetchone() is None:
                return RestoreIssue(table, source_id, "relationship predecessor is missing")
        if table == "observations":
            if not _supported(source, "observation_sources"):
                return RestoreIssue(
                    table, source_id,
                    "observation sources table is unavailable; provenance cannot be preserved",
                )
            links = source.execute(
                "SELECT source_message_id FROM observation_sources WHERE observation_id=?",
                (source_id,),
            ).fetchall()
            if not links:
                return RestoreIssue(table, source_id, "observation has no provenance links")
            for link in links:
                linked_message = link[0]
                if linked_message is not None and source.execute(
                    "SELECT 1 FROM messages WHERE id=?", (linked_message,)
                ).fetchone() is None:
                    return RestoreIssue(table, source_id, "observation source message is missing")
        if table == "entities":
            for target in target_rows.get("entities", []):
                if (target["scope"], target["normalized_name"], target["status"]) != (
                    row["scope"], row["normalized_name"], row["status"]
                ):
                    continue
                if (target["entity_type"] != row["entity_type"]
                        or _norm(target["display_name"]) != _norm(row["display_name"])):
                    return RestoreIssue(
                        table, source_id,
                        "equivalent entity has different type or display name",
                        "conflict", target["id"],
                    )
        if table == "entity_aliases":
            source_entity = source.execute(
                "SELECT normalized_name FROM entities WHERE id=?",
                (row["entity_id"],),
            ).fetchone()
            for target in target_rows.get("entity_aliases", []):
                if (target["scope"], target["normalized_alias"], target["status"]) == (
                    row["scope"], row["normalized_alias"], row["status"]
                ):
                    target_entity = next(
                        (item for item in target_rows.get("entities", [])
                         if item["id"] == target["entity_id"]), None
                    )
                    if (source_entity is not None and target_entity is not None
                            and source_entity["normalized_name"] == target_entity["normalized_name"]):
                        continue
                    return RestoreIssue(
                        table, source_id,
                        "alias resolves to a different destination entity",
                        "conflict", target["id"],
                    )
        if table == "experience_events":
            source_message_id = _value(row, "source_message_id")
            if source.execute("SELECT 1 FROM messages WHERE id = ?", (source_message_id,)).fetchone() is None:
                return RestoreIssue(table, source_id, "event source message is missing")
            target_message_id = _target_message_for_source(
                source, target_rows, source_message_id,
            )
            if target_message_id is not None:
                for target_event in target_rows.get("experience_events", []):
                    if target_event["source_message_id"] != target_message_id:
                        continue
                    differs = any(
                        _event_value(target_event, field) != _event_value(row, field)
                        for field in (
                            "session_id", "kind", "actor", "statement",
                            "reported_at", "authority", "occurred_at",
                            "occurred_precision", "occurred_timezone",
                        )
                    )
                    if differs:
                        return RestoreIssue(
                            table, source_id,
                            "event conflicts with destination evidence",
                            "conflict", target_event["id"],
                        )
                    break
        if table == "digests":
            try:
                digest_sources = _ids(_value(row, "source_ids", ""))
            except (TypeError, ValueError):
                return RestoreIssue(table, source_id, "malformed digest source ids")
            for message_id in digest_sources:
                if source.execute("SELECT 1 FROM messages WHERE id=?", (message_id,)).fetchone() is None:
                    return RestoreIssue(table, source_id, "digest source message is missing")
        if table == "formation_status":
            message_id = _value(row, "message_id")
            if source.execute("SELECT 1 FROM messages WHERE id=?", (message_id,)).fetchone() is None:
                return RestoreIssue(table, source_id, "formation source message is missing")
        if table == "formation_jobs":
            message_id = _value(row, "message_id")
            if source.execute("SELECT 1 FROM messages WHERE id=?", (message_id,)).fetchone() is None:
                return RestoreIssue(table, source_id, "formation job source message is missing")
            if _value(row, "status") not in {"pending", "processing", "complete", "failed"}:
                return RestoreIssue(table, source_id, "invalid formation job status")
            target_message = _target_message_for_source(
                source, target_rows, message_id,
            )
            if target_message is not None:
                for target_job in target_rows.get("formation_jobs", []):
                    if int(_value(target_job, "message_id")) != target_message:
                        continue
                    if _value(target_job, "status") != _value(row, "status"):
                        return RestoreIssue(
                            table, source_id,
                            "formation job state conflicts with destination",
                            "conflict", int(_value(target_job, "job_id")),
                        )
                    break
        if table == "observation_history":
            observation_id = _value(row, "observation_id")
            if source.execute(
                "SELECT 1 FROM observations WHERE id=?", (observation_id,)
            ).fetchone() is None:
                return RestoreIssue(table, source_id, "history observation is missing")
            superseded_by = _value(row, "superseded_by_id")
            if superseded_by is not None and source.execute(
                "SELECT 1 FROM observations WHERE id=?", (superseded_by,)
            ).fetchone() is None:
                return RestoreIssue(table, source_id, "history replacement observation is missing")
            try:
                snapshot = json.loads(_value(row, "snapshot_json", "{}"))
            except (TypeError, ValueError):
                return RestoreIssue(table, source_id, "history snapshot is not valid JSON")
            if not isinstance(snapshot, dict):
                return RestoreIssue(table, source_id, "history snapshot must be a JSON object")
            raw_source_ids = snapshot.get("source_ids", [])
            if raw_source_ids is None:
                raw_source_ids = []
            if not isinstance(raw_source_ids, list):
                return RestoreIssue(table, source_id, "history source_ids is not a list")
            for raw_source_id in raw_source_ids:
                try:
                    message_id = int(raw_source_id)
                except (TypeError, ValueError):
                    return RestoreIssue(table, source_id, "history contains a malformed source id")
                if source.execute("SELECT 1 FROM messages WHERE id = ?", (message_id,)).fetchone() is None:
                    return RestoreIssue(table, source_id, "history source message is missing")
        return None

    def _already_present(
        self, table: str, row: sqlite3.Row,
        target_rows: dict[str, list[sqlite3.Row]],
    ) -> bool:
        targets = target_rows.get(table, [])
        if table == "messages":
            return any(
                target["session_id"] == row["session_id"]
                and target["seq"] == row["seq"]
                and target["fingerprint"] == row["fingerprint"]
                for target in targets
            )
        if table == "sessions":
            return any(target["session_id"] == row["session_id"] for target in targets)
        if table == "states":
            return any(
                target["scope"] == row["scope"]
                and _norm(target["concept"]) == _norm(row["concept"])
                and _norm(target["value"]) == _norm(row["value"])
                and target["status"] == row["status"]
                for target in targets
            )
        if table == "candidates":
            return any(
                target["session_id"] == row["session_id"]
                and target["scope"] == row["scope"]
                and target["proposed_type"] == _value(row, "proposed_type", "")
                and _norm(target["proposed_concept"]) == _norm(row["proposed_concept"])
                and _norm(target["proposed_value"]) == _norm(row["proposed_value"])
                and target["proposed_rationale"] == _value(row, "proposed_rationale", "")
                and target["status"] == row["status"]
                for target in targets
            )
        if table == "entities":
            return any(target["scope"] == row["scope"]
                       and target["normalized_name"] == row["normalized_name"]
                       and target["status"] == row["status"] for target in targets)
        if table == "observations":
            return any(target["scope"] == row["scope"]
                       and target["normalized_concept"] == row["normalized_concept"]
                       and _norm(target["value"]) == _norm(row["value"])
                       and all(
                           _value(row, field, "") == _value(target, field, "")
                           for field in ("observation_type", "actor", "subject", "predicate", "authority")
                       )
                       and target["status"] == row["status"] for target in targets)
        if table == "formation_jobs":
            return any(
                _value(target, "message_id") == _value(row, "message_id")
                and _value(target, "session_id") == _value(row, "session_id")
                and _value(target, "status") == _value(row, "status")
                for target in targets
            )
        return False

    def import_into(self) -> RestoreResult:
        """Copy safe rows into the isolated destination, idempotently."""
        preview = self.preview()
        # A conflict or unsupported primary-memory record is a blocking issue.
        # Return before ContextStore initialization so even a new destination
        # is not created as a side effect of an unsafe import. Derived indexes
        # are explicitly non-blocking and are rebuilt by normal V2 startup.
        if preview.blocking_issues:
            return RestoreResult(
                str(self.source_path), str(self.target_path),
                already_present=dict(preview.already_present),
                conflicts=list(preview.conflicts),
                unsupported=list(preview.unsupported),
                derived=list(preview.derived),
            )
        source = _read_only(self.source_path)
        # ContextStore performs additive schema setup only on the destination.
        ContextStore(self.target_path)
        target = sqlite3.connect(str(self.target_path))
        target.row_factory = sqlite3.Row
        target.execute("PRAGMA foreign_keys=ON")
        imported: dict[str, int] = {}
        already: dict[str, int] = {}
        conflicts = list(preview.conflicts)
        unsupported = list(preview.unsupported)
        maps: dict[str, dict[int, int]] = {table: {} for table in TABLES}
        rolled_back = False
        try:
            try:
                with target:
                    # Serialize destination writers for the whole import. A
                    # preflight conflict can otherwise appear between the
                    # read-only preview and the first INSERT.
                    target.execute("BEGIN IMMEDIATE")
                    self._import_sessions(source, target, maps, imported, already)
                    self._import_messages(source, target, maps, imported, already, conflicts)
                    self._import_events(source, target, maps, imported, already, unsupported)
                    self._import_states(source, target, maps, imported, already, conflicts, unsupported)
                    self._import_candidates(source, target, maps, imported, already, unsupported)
                    self._import_entities(source, target, maps, imported, already, conflicts, unsupported)
                    self._import_aliases(source, target, maps, imported, already, unsupported)
                    self._import_relationships(source, target, maps, imported, already, unsupported)
                    self._import_observations(source, target, maps, imported, already, conflicts, unsupported)
                    self._import_observation_history(source, target, maps, imported, already, unsupported)
                    self._import_digests_and_status(source, target, maps, imported, already, unsupported)
            except _ImportConflict as exc:
                # The connection context has already rolled back every write
                # made in this import. Do not return ids that no longer exist.
                (conflicts if exc.issue.severity == "conflict" else unsupported).append(exc.issue)
                imported.clear()
                already.clear()
                for mapping in maps.values():
                    mapping.clear()
                rolled_back = True
        finally:
            target.close()
            source.close()
        return RestoreResult(
            str(self.source_path), str(self.target_path), imported, already,
            _dedupe_issues(conflicts), _dedupe_issues(unsupported),
            maps if not rolled_back else {}, _dedupe_issues(preview.derived),
        )

    @staticmethod
    def _bump(counter: dict[str, int], table: str) -> None:
        counter[table] = counter.get(table, 0) + 1

    def _import_sessions(self, source, target, maps, imported, already):
        if not _supported(source, "sessions"):
            return
        for row in _rows(source, "sessions"):
            session_id = row["session_id"]
            exists = target.execute(
                "SELECT session_id,created_at,title,status FROM sessions "
                "WHERE session_id = ?", (session_id,)
            ).fetchone()
            if exists:
                if (
                    _norm(exists["title"]) != _norm(_value(row, "title", ""))
                    or exists["status"] != _value(row, "status", "active")
                    or float(exists["created_at"]) != float(row["created_at"])
                ):
                    raise _ImportConflict(RestoreIssue(
                        "sessions", None,
                        "session metadata conflicts with destination", "conflict",
                    ))
                self._bump(already, "sessions")
                continue
            target.execute(
                "INSERT INTO sessions(session_id,created_at,last_activity_at,title,status,previous_titles) VALUES(?,?,?,?,?,?)",
                (session_id, _value(row, "created_at", 0.0), _value(row, "last_activity_at", 0.0),
                 _value(row, "title", ""), _value(row, "status", "active"), _value(row, "previous_titles", "[]")),
            )
            self._bump(imported, "sessions")

    def _import_messages(self, source, target, maps, imported, already, conflicts):
        if not _supported(source, "messages"):
            return
        for row in sorted(_rows(source, "messages"), key=lambda item: item["id"]):
            existing = target.execute(
                "SELECT id,fingerprint FROM messages WHERE session_id=? AND seq=? AND fingerprint=?",
                (row["session_id"], row["seq"], row["fingerprint"]),
            ).fetchone()
            if existing:
                maps["messages"][int(row["id"])] = int(existing["id"])
                self._bump(already, "messages")
                continue
            conflict = target.execute(
                "SELECT id FROM messages WHERE session_id=? AND seq=? LIMIT 1",
                (row["session_id"], row["seq"]),
            ).fetchone()
            if conflict:
                issue = RestoreIssue(
                    "messages", int(row["id"]),
                    "same session/sequence has different content", "conflict",
                    int(conflict["id"]),
                )
                raise _ImportConflict(issue)
            target.execute(
                "INSERT INTO messages(session_id,seq,role,content,timestamp,fingerprint,importance) VALUES(?,?,?,?,?,?,?)",
                (row["session_id"], row["seq"], row["role"], row["content"],
                 row["timestamp"], row["fingerprint"], _value(row, "importance", 0.0)),
            )
            found = target.execute("SELECT id FROM messages WHERE session_id=? AND seq=? AND fingerprint=?",
                                   (row["session_id"], row["seq"], row["fingerprint"])).fetchone()
            maps["messages"][int(row["id"])] = int(found["id"])
            self._bump(imported, "messages")

    def _import_events(self, source, target, maps, imported, already, unsupported):
        if not _supported(source, "experience_events"):
            return
        for row in _rows(source, "experience_events"):
            source_id = int(row["id"])
            mapped_message = maps["messages"].get(int(row["source_message_id"]))
            if mapped_message is None:
                raise _ImportConflict(RestoreIssue(
                    "experience_events", source_id,
                    "source message was not imported",
                ))
            existing = target.execute(
                "SELECT id,session_id,kind,actor,statement,reported_at,authority,"
                "occurred_at,occurred_precision,occurred_timezone "
                "FROM experience_events WHERE source_message_id=?",
                (mapped_message,),
            ).fetchone()
            if existing:
                differs = any(
                    _event_value(existing, field) != _event_value(row, field)
                    for field in (
                        "session_id", "kind", "actor", "statement", "reported_at",
                        "authority", "occurred_at", "occurred_precision",
                        "occurred_timezone",
                    )
                )
                if differs:
                    raise _ImportConflict(RestoreIssue(
                        "experience_events", source_id,
                        "event conflicts with destination evidence", "conflict",
                        int(existing["id"]),
                    ))
                maps["experience_events"][source_id] = int(existing["id"])
                self._bump(already, "experience_events")
                continue
            target.execute(
                "INSERT INTO experience_events(session_id,source_message_id,kind,actor,statement,reported_at,authority,occurred_at,occurred_precision,occurred_timezone) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (_value(row, "session_id"), mapped_message, _value(row, "kind"), _value(row, "actor"),
                 _value(row, "statement"), _value(row, "reported_at", 0.0), _value(row, "authority", "user"),
                 _value(row, "occurred_at"), _value(row, "occurred_precision", "unknown"),
                 _value(row, "occurred_timezone", "UTC")),
            )
            found = target.execute("SELECT id FROM experience_events WHERE source_message_id=?", (mapped_message,)).fetchone()
            maps["experience_events"][source_id] = int(found["id"])
            self._bump(imported, "experience_events")

    def _state_identity(self, target, row):
        candidates = target.execute(
            "SELECT id,source_ids,concept,value,memory_type FROM states "
            "WHERE scope=? AND status=?",
            (row["scope"], row["status"]),
        ).fetchall()
        for candidate in candidates:
            if (_norm(candidate["concept"]) == _norm(row["concept"])
                    and _norm(candidate["value"]) == _norm(row["value"])
                    and not (
                        _value(row, "memory_type", "")
                        and candidate["memory_type"]
                        and _value(row, "memory_type", "") != candidate["memory_type"]
                    )):
                return candidate
        return None

    @staticmethod
    def _merge_int_sources(
        target, table: str, row_id: int, incoming: list[int]
    ) -> None:
        if not incoming:
            return
        current = target.execute(
            f"SELECT source_ids FROM {table} WHERE id=?", (row_id,)
        ).fetchone()
        if current is None:
            raise _ImportConflict(RestoreIssue(
                table, None, "equivalent destination record disappeared", "conflict",
            ))
        existing = _ids(current[0])
        merged = list(dict.fromkeys(existing + incoming))
        if merged != existing:
            target.execute(
                f"UPDATE {table} SET source_ids=? WHERE id=?",
                (_csv(merged), row_id),
            )

    @staticmethod
    def _merge_text_sources(
        target, table: str, row_id: int, column: str, incoming: Any
    ) -> None:
        values = [str(item).strip() for item in str(incoming or "").split(",")
                  if str(item).strip()]
        if not values:
            return
        current = target.execute(
            f"SELECT {column} FROM {table} WHERE id=?", (row_id,)
        ).fetchone()
        if current is None:
            raise _ImportConflict(RestoreIssue(
                table, None, "equivalent destination record disappeared", "conflict",
            ))
        existing = [str(item).strip() for item in str(current[0] or "").split(",")
                    if str(item).strip()]
        merged = list(dict.fromkeys(existing + values))
        if merged != existing:
            target.execute(
                f"UPDATE {table} SET {column}=? WHERE id=?",
                (",".join(merged), row_id),
            )

    def _import_states(self, source, target, maps, imported, already, conflicts, unsupported):
        if not _supported(source, "states"):
            return
        for row in sorted(_rows(source, "states"), key=lambda item: item["id"]):
            source_id = int(row["id"])
            mapped_sources, error = _map_ids(_value(row, "source_ids", ""), maps["messages"])
            if error:
                raise _ImportConflict(RestoreIssue("states", source_id, error))
            predecessor = _value(row, "supersedes_id")
            mapped_predecessor = maps["states"].get(int(predecessor)) if predecessor is not None else None
            if predecessor is not None and mapped_predecessor is None:
                raise _ImportConflict(RestoreIssue(
                    "states", source_id,
                    "superseded predecessor was not imported",
                ))
            current = None
            if row["status"] == "current":
                for candidate in target.execute(
                    "SELECT id,concept,value,memory_type FROM states "
                    "WHERE scope=? AND status='current'", (row["scope"],)
                ).fetchall():
                    if _norm(candidate["concept"]) == _norm(row["concept"]):
                        source_type = _value(row, "memory_type", "")
                        if (source_type and candidate["memory_type"]
                                and source_type != candidate["memory_type"]):
                            continue
                        current = candidate
                        break
            if current and _norm(current["value"]) != _norm(row["value"]):
                raise _ImportConflict(RestoreIssue(
                    "states", source_id, "current state conflicts with destination",
                    "conflict", int(current["id"]),
                ))
            exact = self._state_identity(target, row)
            if exact:
                maps["states"][source_id] = int(exact["id"])
                self._merge_int_sources(target, "states", int(exact["id"]), mapped_sources)
                self._bump(already, "states")
                continue
            target.execute(
                "INSERT INTO states(session_id,scope,concept,value,source_ids,created_at,status,supersedes_id,authority,memory_type,confidence,observed_at,valid_from,valid_until,superseded_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (_value(row, "session_id"), row["scope"], row["concept"], row["value"], _csv(mapped_sources),
                 _value(row, "created_at", 0.0), _value(row, "status", "current"), mapped_predecessor,
                 _value(row, "authority", "user"), _value(row, "memory_type", ""), _value(row, "confidence"),
                 _value(row, "observed_at"), _value(row, "valid_from"), _value(row, "valid_until"), _value(row, "superseded_at")),
            )
            found = target.execute("SELECT last_insert_rowid() AS id").fetchone()
            maps["states"][source_id] = int(found["id"])
            self._bump(imported, "states")

    def _import_candidates(self, source, target, maps, imported, already, unsupported):
        if not _supported(source, "candidates"):
            return
        for row in _rows(source, "candidates"):
            mapped, error = _map_ids(_value(row, "source_ids", ""), maps["messages"])
            if error:
                raise _ImportConflict(RestoreIssue("candidates", int(row["id"]), error))
            exists = target.execute(
                "SELECT id FROM candidates WHERE session_id=? AND scope=? "
                "AND proposed_type=? AND proposed_concept=? AND proposed_value=? "
                "AND proposed_rationale=? AND status=?",
                (_value(row, "session_id"), _value(row, "scope"),
                 _value(row, "proposed_type", ""),
                 _value(row, "proposed_concept", ""),
                 _value(row, "proposed_value", ""),
                 _value(row, "proposed_rationale", ""),
                 _value(row, "status", "pending")),
            ).fetchone()
            if exists:
                maps["candidates"][int(row["id"])] = int(exists["id"])
                self._merge_int_sources(target, "candidates", int(exists["id"]), mapped)
                self._bump(already, "candidates")
                continue
            target.execute(
                "INSERT INTO candidates(session_id,scope,source_ids,proposed_type,proposed_concept,proposed_value,proposed_rationale,authority,confidence,status,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (_value(row, "session_id"), row["scope"], _csv(mapped), _value(row, "proposed_type", ""),
                 _value(row, "proposed_concept", ""), _value(row, "proposed_value", ""), _value(row, "proposed_rationale", ""),
                 _value(row, "authority", "user"), _value(row, "confidence", 0.3), _value(row, "status", "pending"),
                 _value(row, "created_at", 0.0), _value(row, "expires_at")),
            )
            self._bump(imported, "candidates")

    def _import_entities(self, source, target, maps, imported, already, conflicts, unsupported):
        if not _supported(source, "entities"):
            return
        for row in sorted(_rows(source, "entities"), key=lambda item: item["id"]):
            source_id = int(row["id"])
            mapped, error = _map_ids(_value(row, "source_ids", ""), maps["messages"])
            if error:
                raise _ImportConflict(RestoreIssue("entities", source_id, error))
            existing = target.execute(
                "SELECT id,entity_type,display_name,status FROM entities WHERE scope=? AND normalized_name=?",
                (row["scope"], row["normalized_name"]),
            ).fetchall()
            if any(item["status"] == "active" and row["status"] == "active"
                   and (item["entity_type"] != row["entity_type"] or _norm(item["display_name"]) != _norm(row["display_name"]))
                   for item in existing):
                raise _ImportConflict(RestoreIssue(
                    "entities", source_id,
                    "equivalent entity has different type or display name",
                    "conflict", int(existing[0]["id"]),
                ))
            same = next((item for item in existing if item["status"] == row["status"] and item["entity_type"] == row["entity_type"]), None)
            if same:
                maps["entities"][source_id] = int(same["id"])
                self._merge_int_sources(target, "entities", int(same["id"]), mapped)
                self._merge_text_sources(
                    target, "entities", int(same["id"]),
                    "source_episode_ids", _value(row, "source_episode_ids", ""),
                )
                self._bump(already, "entities")
                continue
            merged = _value(row, "merged_into_entity_id")
            merged_target = maps["entities"].get(int(merged)) if merged is not None else None
            if merged is not None and merged_target is None:
                raise _ImportConflict(RestoreIssue(
                    "entities", source_id,
                    "merged entity target was not imported",
                ))
            target.execute(
                "INSERT INTO entities(session_id,scope,entity_type,display_name,normalized_name,status,merged_into_entity_id,source_ids,authority,confidence,created_at,observed_at,merged_at,source_episode_ids,valid_from,valid_until) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (_value(row, "session_id"), row["scope"], row["entity_type"], row["display_name"], row["normalized_name"],
                 row["status"], merged_target, _csv(mapped), _value(row, "authority", "user"), _value(row, "confidence", 0.3),
                 _value(row, "created_at", 0.0), _value(row, "observed_at"), _value(row, "merged_at"), _value(row, "source_episode_ids", ""),
                 _value(row, "valid_from"), _value(row, "valid_until")),
            )
            maps["entities"][source_id] = int(target.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])
            self._bump(imported, "entities")

    def _import_aliases(self, source, target, maps, imported, already, unsupported):
        if not _supported(source, "entity_aliases"):
            return
        for row in _rows(source, "entity_aliases"):
            entity = maps["entities"].get(int(row["entity_id"]))
            mapped, error = _map_ids(_value(row, "source_ids", ""), maps["messages"])
            if entity is None or error:
                raise _ImportConflict(RestoreIssue(
                    "entity_aliases", int(row["id"]),
                    "entity or source mapping is unavailable",
                ))
            exists = target.execute(
                "SELECT id FROM entity_aliases WHERE entity_id=? AND normalized_alias=? AND status=?",
                (entity, row["normalized_alias"], row["status"]),
            ).fetchone()
            conflicting = target.execute(
                "SELECT id FROM entity_aliases WHERE scope=? AND normalized_alias=? "
                "AND status=? AND entity_id<>?",
                (row["scope"], row["normalized_alias"], row["status"], entity),
            ).fetchone()
            if conflicting:
                raise _ImportConflict(RestoreIssue(
                    "entity_aliases", int(row["id"]),
                    "alias resolves to a different destination entity",
                    "conflict", int(conflicting["id"]),
                ))
            if exists:
                maps["entity_aliases"][int(row["id"])] = int(exists["id"])
                self._merge_int_sources(target, "entity_aliases", int(exists["id"]), mapped)
                self._bump(already, "entity_aliases")
                continue
            target.execute(
                "INSERT INTO entity_aliases(session_id,scope,entity_id,alias_text,normalized_alias,status,source_ids,authority,confidence,created_at,observed_at,valid_from,valid_until,superseded_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (_value(row, "session_id"), row["scope"], entity, row["alias_text"], row["normalized_alias"], row["status"], _csv(mapped),
                 _value(row, "authority", "user"), _value(row, "confidence", 0.3), _value(row, "created_at", 0.0), _value(row, "observed_at"),
                 _value(row, "valid_from"), _value(row, "valid_until"), _value(row, "superseded_at")),
            )
            maps["entity_aliases"][int(row["id"])] = int(target.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])
            self._bump(imported, "entity_aliases")

    def _relationship_ref(self, kind: str, source_id: int, maps: dict[str, dict[int, int]]) -> Optional[int]:
        table = {"entity": "entities", "memory_state": "states", "observation": "observations"}.get(kind)
        return maps.get(table, {}).get(int(source_id)) if table else None

    def _import_relationships(self, source, target, maps, imported, already, unsupported):
        if not _supported(source, "relationships"):
            return
        for row in _rows(source, "relationships"):
            subject = self._relationship_ref(row["subject_kind"], row["subject_id"], maps)
            object_id = self._relationship_ref(row["object_kind"], row["object_id"], maps)
            mapped, error = _map_ids(_value(row, "source_ids", ""), maps["messages"])
            if subject is None or object_id is None or error:
                raise _ImportConflict(RestoreIssue(
                    "relationships", int(row["id"]),
                    "relationship endpoint or source mapping is unavailable",
                ))
            exists = target.execute(
                "SELECT id FROM relationships WHERE scope=? AND relationship_type=? AND subject_kind=? AND subject_id=? AND object_kind=? AND object_id=? AND status=?",
                (row["scope"], row["relationship_type"], row["subject_kind"], subject, row["object_kind"], object_id, row["status"]),
            ).fetchone()
            if exists:
                maps["relationships"][int(row["id"])] = int(exists["id"])
                self._merge_int_sources(target, "relationships", int(exists["id"]), mapped)
                self._merge_text_sources(
                    target, "relationships", int(exists["id"]),
                    "source_episode_ids", _value(row, "source_episode_ids", ""),
                )
                self._bump(already, "relationships")
                continue
            predecessor = _value(row, "supersedes_id")
            mapped_predecessor = (
                maps["relationships"].get(int(predecessor))
                if predecessor is not None else None
            )
            if predecessor is not None and mapped_predecessor is None:
                raise _ImportConflict(RestoreIssue(
                    "relationships", int(row["id"]),
                    "relationship predecessor was not imported",
                ))
            try:
                target.execute(
                    "INSERT INTO relationships(session_id,scope,relationship_type,subject_kind,subject_id,object_kind,object_id,status,source_ids,source_episode_ids,authority,confidence,created_at,observed_at,valid_from,valid_until,supersedes_id,superseded_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (_value(row, "session_id"), row["scope"], row["relationship_type"], row["subject_kind"], subject, row["object_kind"], object_id,
                     row["status"], _csv(mapped), _value(row, "source_episode_ids", ""), _value(row, "authority", "user"), _value(row, "confidence", 0.3),
                     _value(row, "created_at", 0.0), _value(row, "observed_at"), _value(row, "valid_from"), _value(row, "valid_until"), mapped_predecessor, _value(row, "superseded_at")),
                )
            except sqlite3.IntegrityError as exc:
                if row["status"] == "current":
                    existing_current = target.execute(
                        "SELECT id FROM relationships WHERE scope=? AND relationship_type=? "
                        "AND subject_id=? AND object_id=? AND status='current'",
                        (row["scope"], row["relationship_type"], subject, object_id),
                    ).fetchone()
                    if existing_current:
                        raise _ImportConflict(RestoreIssue(
                            "relationships", int(row["id"]),
                            "current relationship conflicts with destination",
                            "conflict", int(existing_current["id"]),
                        )) from exc
                raise
            maps["relationships"][int(row["id"])] = int(target.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])
            self._bump(imported, "relationships")

    def _import_observations(self, source, target, maps, imported, already, conflicts, unsupported):
        if not _supported(source, "observations"):
            return
        for row in sorted(_rows(source, "observations"), key=lambda item: item["id"]):
            source_id = int(row["id"])
            mapped, error = _map_ids(_value(row, "source_ids", ""), maps["messages"])
            if error:
                raise _ImportConflict(RestoreIssue("observations", source_id, error))
            current = None
            for candidate in target.execute(
                "SELECT id,value,observation_type,actor,subject,predicate,authority "
                "FROM observations WHERE scope=? AND normalized_concept=? "
                "AND status='current'",
                (row["scope"], row["normalized_concept"]),
            ).fetchall():
                if all(
                    _value(row, field, "") == candidate[field]
                    for field in ("observation_type", "actor", "subject", "predicate", "authority")
                ):
                    current = candidate
                    break
            if current and row["status"] == "current" and _norm(current["value"]) != _norm(row["value"]):
                raise _ImportConflict(RestoreIssue(
                    "observations", source_id,
                    "current observation conflicts with destination", "conflict",
                    int(current["id"]),
                ))
            exact = None
            for candidate in target.execute(
                "SELECT id,value,observation_type,actor,subject,predicate,authority "
                "FROM observations WHERE scope=? "
                "AND normalized_concept=? AND status=?",
                (row["scope"], row["normalized_concept"], row["status"]),
            ).fetchall():
                if (_norm(candidate["value"]) == _norm(row["value"])
                        and all(
                            _value(row, field, "") == candidate[field]
                            for field in ("observation_type", "actor", "subject", "predicate", "authority")
                        )):
                    exact = candidate
                    break
            if exact:
                maps["observations"][source_id] = int(exact["id"])
                self._bump(already, "observations")
                observation_target = int(exact["id"])
            else:
                predecessor = _value(row, "supersedes_id")
                mapped_predecessor = maps["observations"].get(int(predecessor)) if predecessor is not None else None
                if predecessor is not None and mapped_predecessor is None:
                    raise _ImportConflict(RestoreIssue(
                        "observations", source_id,
                        "observation predecessor was not imported",
                    ))
                target.execute(
                    "INSERT INTO observations(session_id,scope,concept,normalized_concept,value,observation_type,actor,subject,predicate,authority,status,confidence,created_at,observed_at,occurred_from,occurred_until,valid_from,valid_until,supersedes_id,superseded_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (_value(row, "session_id"), row["scope"], row["concept"], row["normalized_concept"], row["value"], _value(row, "observation_type", "observation"),
                     _value(row, "actor", "user"), _value(row, "subject", ""), _value(row, "predicate", ""), _value(row, "authority", "user"), row["status"],
                     _value(row, "confidence", 1.0), _value(row, "created_at", 0.0), _value(row, "observed_at"), _value(row, "occurred_from"), _value(row, "occurred_until"),
                     _value(row, "valid_from"), _value(row, "valid_until"), mapped_predecessor, _value(row, "superseded_at")),
                )
                observation_target = int(target.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])
                maps["observations"][source_id] = observation_target
                self._bump(imported, "observations")
            if _supported(source, "observation_sources"):
                for link in source.execute("SELECT * FROM observation_sources WHERE observation_id=?", (source_id,)):
                    mapped_message = maps["messages"].get(_value(link, "source_message_id")) if _value(link, "source_message_id") is not None else None
                    if _value(link, "source_message_id") is not None and mapped_message is None:
                        raise _ImportConflict(RestoreIssue(
                            "observation_sources", int(link["id"]),
                            "source message was not imported",
                        ))
                    target.execute(
                        "INSERT OR IGNORE INTO observation_sources(observation_id,source_message_id,source_episode_id,source_quote,source_role,linked_at) VALUES(?,?,?,?,?,?)",
                        (observation_target, mapped_message, _value(link, "source_episode_id"), _value(link, "source_quote", ""), _value(link, "source_role", "user"), _value(link, "linked_at", 0.0)),
                    )

    def _import_digests_and_status(self, source, target, maps, imported, already, unsupported):
        if _supported(source, "digests"):
            for row in _rows(source, "digests"):
                mapped, error = _map_ids(_value(row, "source_ids", ""), maps["messages"])
                if error:
                    raise _ImportConflict(RestoreIssue("digests", int(row["id"]), error))
                exists = target.execute("SELECT id FROM digests WHERE session_id=? AND source_ids=? AND content=?",
                                        (_value(row, "session_id"), _csv(mapped), _value(row, "content", ""))).fetchone()
                if exists:
                    self._bump(already, "digests")
                    continue
                target.execute(
                    "INSERT INTO digests(session_id,source_ids,seq_start,seq_end,created_at,digest_type,content) VALUES(?,?,?,?,?,?,?)",
                    (_value(row, "session_id"), _csv(mapped), _value(row, "seq_start", 0), _value(row, "seq_end", 0), _value(row, "created_at", 0.0), _value(row, "digest_type", ""), _value(row, "content", "")),
                )
                self._bump(imported, "digests")
        if _supported(source, "formation_status"):
            for row in _rows(source, "formation_status"):
                message = maps["messages"].get(int(row["message_id"]))
                if message is None:
                    raise _ImportConflict(RestoreIssue(
                        "formation_status", int(row["message_id"]),
                        "source message was not imported",
                    ))
                exists = target.execute("SELECT 1 FROM formation_status WHERE message_id=?", (message,)).fetchone()
                if exists:
                    self._bump(already, "formation_status")
                    continue
                target.execute(
                    "INSERT INTO formation_status(message_id,session_id,status,attempts,diagnostics,updated_at) VALUES(?,?,?,?,?,?)",
                    (message, _value(row, "session_id"), _value(row, "status"), _value(row, "attempts", 0), _value(row, "diagnostics", "[]"), _value(row, "updated_at", 0.0)),
                )
                self._bump(imported, "formation_status")
        if _supported(source, "formation_jobs"):
            for row in _rows(source, "formation_jobs"):
                message = maps["messages"].get(int(row["message_id"]))
                if message is None:
                    raise _ImportConflict(RestoreIssue(
                        "formation_jobs", int(row["job_id"]),
                        "formation job source message was not imported",
                    ))
                exists = target.execute(
                    "SELECT job_id FROM formation_jobs WHERE message_id=?",
                    (message,),
                ).fetchone()
                if exists:
                    maps["formation_jobs"][int(row["job_id"])] = int(exists[0])
                    self._bump(already, "formation_jobs")
                    continue
                target.execute(
                    "INSERT INTO formation_jobs(message_id,session_id,status,attempts,last_error,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                    (
                        message, _value(row, "session_id"),
                        _value(row, "status", "pending"),
                        _value(row, "attempts", 0),
                        _value(row, "last_error", ""),
                        _value(row, "created_at", 0.0),
                        _value(row, "updated_at", 0.0),
                    ),
                )
                maps["formation_jobs"][int(row["job_id"])] = int(
                    target.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
                )
                self._bump(imported, "formation_jobs")

    def _import_observation_history(self, source, target, maps, imported, already, unsupported):
        if not _supported(source, "observation_history"):
            return
        for row in _rows(source, "observation_history"):
            source_observation = int(row["observation_id"])
            observation = maps["observations"].get(source_observation)
            if observation is None:
                raise _ImportConflict(RestoreIssue(
                    "observation_history", int(row["id"]),
                    "observation was not imported",
                ))
            superseded_by = _value(row, "superseded_by_id")
            mapped_by = (
                maps["observations"].get(int(superseded_by))
                if superseded_by is not None else None
            )
            if superseded_by is not None and mapped_by is None:
                raise _ImportConflict(RestoreIssue(
                    "observation_history", int(row["id"]),
                    "replacement observation was not imported",
                ))
            snapshot = _value(row, "snapshot_json", "{}")
            try:
                decoded = json.loads(snapshot)
                if isinstance(decoded, dict):
                    decoded["observation_id"] = observation
                    if "superseded_by_id" in decoded:
                        decoded["superseded_by_id"] = mapped_by
                    if isinstance(decoded.get("source_ids"), list):
                        mapped_history_sources = []
                        for item in decoded["source_ids"]:
                            try:
                                source_message_id = int(item)
                            except (TypeError, ValueError) as exc:
                                raise _ImportConflict(RestoreIssue(
                                    "observation_history", int(row["id"]),
                                    "history contains a malformed source id",
                                )) from exc
                            mapped_message = maps["messages"].get(source_message_id)
                            if mapped_message is None:
                                raise _ImportConflict(RestoreIssue(
                                    "observation_history", int(row["id"]),
                                    "history source message was not imported",
                                ))
                            mapped_history_sources.append(mapped_message)
                        decoded["source_ids"] = mapped_history_sources
                    snapshot = json.dumps(decoded, ensure_ascii=False)
            except (TypeError, ValueError):
                raise _ImportConflict(RestoreIssue(
                    "observation_history", int(row["id"]),
                    "history snapshot is not valid JSON",
                ))
            exists = target.execute(
                "SELECT id FROM observation_history WHERE observation_id=? AND snapshot_json=? AND reason=?",
                (observation, snapshot, _value(row, "reason", "")),
            ).fetchone()
            if exists:
                maps["observation_history"][int(row["id"])] = int(exists["id"])
                self._bump(already, "observation_history")
                continue
            target.execute(
                "INSERT INTO observation_history(observation_id,snapshot_json,reason,changed_at,superseded_by_id) VALUES(?,?,?,?,?)",
                (observation, snapshot, _value(row, "reason", ""), _value(row, "changed_at", 0.0), mapped_by),
            )
            maps["observation_history"][int(row["id"])] = int(target.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])
            self._bump(imported, "observation_history")


def preview_restore(source_path: Path | str, target_path: Path | str) -> RestorePreview:
    return MemoryRestoreService(source_path, target_path).preview()


def import_restore(source_path: Path | str, target_path: Path | str) -> RestoreResult:
    return MemoryRestoreService(source_path, target_path).import_into()


def main(argv: Optional[list[str]] = None) -> int:
    """PowerShell-friendly explicit preview/import entry point."""
    parser = argparse.ArgumentParser(
        description="Preview or explicitly import PRIMORA memory into an isolated SQLite database"
    )
    parser.add_argument("action", choices=("preview", "import"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        service = MemoryRestoreService(args.source, args.target)
        result = service.preview() if args.action == "preview" else service.import_into()
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        # Derived indexes are informational and do not block. Primary-memory
        # conflicts/unsupported rows make the command visibly non-successful
        # while still returning the complete JSON report for review.
        return 2 if result.blocking_issues else 0
    except (FileNotFoundError, ValueError, OSError, sqlite3.DatabaseError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
