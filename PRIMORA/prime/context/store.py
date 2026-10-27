"""SQLite persistence for raw conversation history.

Append-only. Raw rows are never updated or deleted. Thread-safe through
per-operation connections. FTS5 when the runtime supports it, LIKE fallback
otherwise (probed, never assumed).
"""

import hashlib
import json
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager

from .entities import (
    ENTITY_AUTHORITIES,
    ENTITY_TYPES,
    normalize_entity_name,
)
from .models import (
    CandidateRecord,
    EntityAliasRecord,
    EntityRecord,
    ExperienceEvent,
    RelationshipRecord,
    ObservationRecord,
    ObservationHistoryRecord,
    SessionRecord,
    StateRecord,
    StoredMessage,
)
from .relationships import (
    RELATIONSHIP_ENDPOINTS,
    RELATIONSHIP_STATUSES,
    RELATIONSHIP_TYPES,
)


def default_db_path() -> Path:
    here = Path(__file__).resolve()
    # PRIMORA/prime/context/store.py -> PRIMORA/ = parents[2]
    root = here.parents[2]
    return root / "data" / "context" / "aurora_context.db"


def fingerprint(session_id: str, seq: int, role: str, content: str) -> str:
    material = f"{session_id}\x00{seq}\x00{role}\x00{content}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# System-row marker written when a conversation is cleared. Reloads start
# after the newest marker, so a clear persists across restarts without
# ever deleting a raw row.
CONVERSATION_CLEARED = "<conversation cleared>"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp REAL NOT NULL,
    fingerprint TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.0,
    UNIQUE(session_id, seq, fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_messages_session_seq
    ON messages(session_id, seq);
CREATE TABLE IF NOT EXISTS digests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    source_ids TEXT NOT NULL,
    seq_start INTEGER NOT NULL,
    seq_end INTEGER NOT NULL,
    created_at REAL NOT NULL,
    digest_type TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_digests_session
    ON digests(session_id);
CREATE TABLE IF NOT EXISTS experience_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    source_message_id INTEGER NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    actor TEXT NOT NULL,
    statement TEXT NOT NULL,
    reported_at REAL NOT NULL,
    authority TEXT NOT NULL DEFAULT 'user',
    occurred_at REAL,
    occurred_precision TEXT NOT NULL DEFAULT 'unknown',
    occurred_timezone TEXT NOT NULL DEFAULT 'UTC',
    FOREIGN KEY (source_message_id) REFERENCES messages(id)
);
CREATE INDEX IF NOT EXISTS idx_experience_events_reported
    ON experience_events(reported_at, id);
CREATE INDEX IF NOT EXISTS idx_experience_events_session
    ON experience_events(session_id, reported_at);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    last_activity_at REAL NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    previous_titles TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    concept TEXT NOT NULL,
    value TEXT NOT NULL,
    source_ids TEXT NOT NULL,
    created_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'current',
    supersedes_id INTEGER,
    authority TEXT NOT NULL DEFAULT 'user',
    memory_type TEXT NOT NULL DEFAULT '',
    confidence REAL,
    observed_at REAL,
    valid_from REAL,
    valid_until REAL,
    superseded_at REAL
);
CREATE INDEX IF NOT EXISTS idx_states_session_concept
    ON states(session_id, concept);
CREATE INDEX IF NOT EXISTS idx_states_scope_concept
    ON states(scope, concept);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    source_ids TEXT NOT NULL,
    proposed_type TEXT NOT NULL DEFAULT '',
    proposed_concept TEXT NOT NULL DEFAULT '',
    proposed_value TEXT NOT NULL DEFAULT '',
    proposed_rationale TEXT NOT NULL DEFAULT '',
    authority TEXT NOT NULL DEFAULT 'user',
    confidence REAL NOT NULL DEFAULT 0.3,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL,
    expires_at REAL
);
CREATE INDEX IF NOT EXISTS idx_candidates_scope_concept
    ON candidates(scope, proposed_concept);
CREATE INDEX IF NOT EXISTS idx_candidates_status
    ON candidates(status);
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    display_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'candidate',
    merged_into_entity_id INTEGER,
    source_ids TEXT NOT NULL,
    authority TEXT NOT NULL DEFAULT 'user',
    confidence REAL NOT NULL DEFAULT 0.3,
    created_at REAL NOT NULL,
    observed_at REAL,
    merged_at REAL,
    source_episode_ids TEXT NOT NULL DEFAULT '',
    valid_from REAL,
    valid_until REAL
);
CREATE INDEX IF NOT EXISTS idx_entities_scope_name_status
    ON entities(scope, normalized_name, status);
CREATE INDEX IF NOT EXISTS idx_entities_scope_status
    ON entities(scope, status);
CREATE INDEX IF NOT EXISTS idx_entities_merged_into
    ON entities(merged_into_entity_id);
CREATE TABLE IF NOT EXISTS entity_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    alias_text TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'candidate',
    source_ids TEXT NOT NULL,
    authority TEXT NOT NULL DEFAULT 'user',
    confidence REAL NOT NULL DEFAULT 0.3,
    created_at REAL NOT NULL,
    observed_at REAL,
    valid_from REAL,
    valid_until REAL,
    superseded_at REAL
);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_scope_name_status
    ON entity_aliases(scope, normalized_alias, status);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_entity_status
    ON entity_aliases(entity_id, status);
CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    subject_kind TEXT NOT NULL,
    subject_id INTEGER NOT NULL,
    object_kind TEXT NOT NULL,
    object_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'candidate',
    source_ids TEXT NOT NULL,
    source_episode_ids TEXT NOT NULL DEFAULT '',
    authority TEXT NOT NULL DEFAULT 'user',
    confidence REAL NOT NULL DEFAULT 0.3,
    created_at REAL NOT NULL,
    observed_at REAL,
    valid_from REAL,
    valid_until REAL,
    supersedes_id INTEGER,
    superseded_at REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_relationships_current_key
    ON relationships(scope, relationship_type, subject_id, object_id)
    WHERE status = 'current';
CREATE INDEX IF NOT EXISTS idx_relationships_subject
    ON relationships(scope, subject_kind, subject_id, status);
CREATE INDEX IF NOT EXISTS idx_relationships_object
    ON relationships(scope, object_kind, object_id, status);
CREATE INDEX IF NOT EXISTS idx_relationships_supersedes
    ON relationships(supersedes_id);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    concept TEXT NOT NULL,
    normalized_concept TEXT NOT NULL,
    value TEXT NOT NULL,
    observation_type TEXT NOT NULL DEFAULT 'observation',
    actor TEXT NOT NULL DEFAULT 'user',
    subject TEXT NOT NULL DEFAULT '',
    predicate TEXT NOT NULL DEFAULT '',
    authority TEXT NOT NULL DEFAULT 'user',
    status TEXT NOT NULL DEFAULT 'current',
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    observed_at REAL,
    occurred_from REAL,
    occurred_until REAL,
    valid_from REAL,
    valid_until REAL,
    supersedes_id INTEGER,
    superseded_at REAL,
    FOREIGN KEY (supersedes_id) REFERENCES observations(id)
);
CREATE INDEX IF NOT EXISTS idx_observations_scope_status
    ON observations(scope, status);
CREATE INDEX IF NOT EXISTS idx_observations_normalized_concept
    ON observations(normalized_concept);
CREATE INDEX IF NOT EXISTS idx_observations_valid_interval
    ON observations(scope, valid_from, valid_until);
CREATE TABLE IF NOT EXISTS observation_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id INTEGER NOT NULL,
    source_message_id INTEGER,
    source_episode_id TEXT,
    source_quote TEXT NOT NULL DEFAULT '',
    source_role TEXT NOT NULL DEFAULT 'user',
    linked_at REAL NOT NULL,
    FOREIGN KEY (observation_id) REFERENCES observations(id),
    FOREIGN KEY (source_message_id) REFERENCES messages(id),
    UNIQUE(observation_id, source_message_id, source_episode_id, source_quote)
);
CREATE INDEX IF NOT EXISTS idx_observation_sources_observation
    ON observation_sources(observation_id);
CREATE INDEX IF NOT EXISTS idx_observation_sources_message
    ON observation_sources(source_message_id);
CREATE TABLE IF NOT EXISTS observation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    changed_at REAL NOT NULL,
    superseded_by_id INTEGER,
    FOREIGN KEY (observation_id) REFERENCES observations(id),
    FOREIGN KEY (superseded_by_id) REFERENCES observations(id)
);
CREATE INDEX IF NOT EXISTS idx_observation_history_observation
    ON observation_history(observation_id, changed_at, id);
CREATE TABLE IF NOT EXISTS retrieval_vectors (
    vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    ref_id INTEGER NOT NULL,
    scope TEXT NOT NULL,
    session_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_retrieval_vectors_identity
    ON retrieval_vectors(kind, ref_id, model_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_vectors_kind_scope
    ON retrieval_vectors(kind, scope, model_id);
CREATE TABLE IF NOT EXISTS retrieval_index_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS formation_status (
    message_id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    diagnostics TEXT NOT NULL DEFAULT '[]',
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_formation_status_session
    ON formation_status(session_id, status);
CREATE TABLE IF NOT EXISTS formation_jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_formation_jobs_status
    ON formation_jobs(status, updated_at, job_id);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
    USING fts5(content, content='messages', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


class ContextStore:
    """Append-only conversation store with optional FTS5 search."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self._db_path = Path(db_path) if db_path else default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_available: Optional[bool] = None
        self._init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def fts_available(self) -> bool:
        if self._fts_available is None:
            self._fts_available = self._probe_fts()
        return self._fts_available

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._db_path), timeout=30.0)
        con.execute("PRAGMA foreign_keys=ON;")
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    @staticmethod
    def _validate_user_source_roles(
        con: sqlite3.Connection,
        source_ids: list[int],
        authority: str,
        session_id: str,
        scope: str,
    ) -> None:
        """Validate *new writes* against authentic, session-owned user turns.

        Never infer trust from a missing message ID. Historic/imported records
        remain readable, but new admissions cannot use missing, assistant,
        tool or cross-session rows as user authority.
        """
        if authority != "user":
            return
        if scope != "global" and scope != session_id:
            raise ValueError("user source scope must match originating session")
        if not source_ids:
            raise ValueError("user-authority memory requires a user source")
        for source_id in dict.fromkeys(int(value) for value in source_ids):
            if source_id <= 0:
                raise ValueError(f"invalid user source_id={source_id}")
            row = con.execute(
                "SELECT role, session_id FROM messages WHERE id = ?",
                (source_id,),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"user-authority memory requires an existing user source "
                    f"(source_id={source_id})"
                )
            if str(row[0]).casefold() != "user":
                raise ValueError(
                    "user-authority memory requires a user source "
                    f"(source_id={source_id}, role={row[0]!r})"
                )
            if row[1] != session_id:
                raise ValueError(
                    "user-authority source belongs to another session "
                    f"(source_id={source_id})"
                )

    @contextmanager
    def _connection(self):
        """Yield one SQLite connection and always close it.

        ``sqlite3.Connection.__exit__`` commits or rolls back a transaction,
        but it deliberately does not close the connection.  The store opens
        short-lived connections for every operation, so the ownership wrapper
        must close them on both normal and exceptional exits.
        """
        con = self._connect()
        try:
            with con:
                yield con
        finally:
            con.close()

    def _init_db(self) -> None:
        with self._connection() as con:
            con.executescript(_SCHEMA)
            # Migrate pre-existing databases that lack newer columns.
            cols = {r[1] for r in con.execute("PRAGMA table_info(sessions)").fetchall()}
            if "previous_titles" not in cols:
                con.execute(
                    "ALTER TABLE sessions ADD COLUMN previous_titles"
                    " TEXT NOT NULL DEFAULT '[]'"
                )
            state_cols = {r[1] for r in con.execute("PRAGMA table_info(states)").fetchall()}
            if "memory_type" not in state_cols:
                con.execute(
                    "ALTER TABLE states ADD COLUMN memory_type TEXT NOT NULL DEFAULT ''"
                )
            if "confidence" not in state_cols:
                # Nullable: legacy rows keep NULL and are read back with the
                # authority-based default (user 1.0, otherwise 0.3), so no
                # historical confidence is ever invented by migration.
                con.execute(
                    "ALTER TABLE states ADD COLUMN confidence REAL"
                )
            for column in (
                "observed_at", "valid_from", "valid_until", "superseded_at",
            ):
                if column not in state_cols:
                    con.execute(f"ALTER TABLE states ADD COLUMN {column} REAL")
            relationship_cols = {
                r[1] for r in con.execute(
                    "PRAGMA table_info(relationships)"
                ).fetchall()
            }
            if "source_episode_ids" not in relationship_cols:
                con.execute(
                    "ALTER TABLE relationships ADD COLUMN source_episode_ids"
                    " TEXT NOT NULL DEFAULT ''"
                )
            event_cols = {
                r[1] for r in con.execute(
                    "PRAGMA table_info(experience_events)"
                ).fetchall()
            }
            for column, definition in (
                ("occurred_at", "REAL"),
                ("occurred_precision", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("occurred_timezone", "TEXT NOT NULL DEFAULT 'UTC'"),
            ):
                if column not in event_cols:
                    con.execute(
                        f"ALTER TABLE experience_events ADD COLUMN {column} {definition}"
                    )
            entity_cols = {
                r[1] for r in con.execute("PRAGMA table_info(entities)").fetchall()
            }
            for column, definition in (
                ("source_episode_ids", "TEXT NOT NULL DEFAULT ''"),
                ("valid_from", "REAL"),
                ("valid_until", "REAL"),
            ):
                if column not in entity_cols:
                    con.execute(f"ALTER TABLE entities ADD COLUMN {column} {definition}")
            con.execute(
                "UPDATE states SET observed_at = COALESCE(observed_at, created_at),"
                " valid_from = COALESCE(valid_from, created_at)"
                " WHERE status = 'current'"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_states_temporal"
                " ON states(scope, concept, valid_from, valid_until)"
            )
        # Probe FTS5 support; never claim it without verifying.
        try:
            with self._connection() as con:
                con.executescript(_FTS_SCHEMA)
            # Backfill any rows written before the virtual table existed.
            with self._connection() as con:
                con.execute(
                    "INSERT OR IGNORE INTO messages_fts(rowid, content) "
                    "SELECT id, content FROM messages "
                    "WHERE id NOT IN (SELECT rowid FROM messages_fts)"
                )
            self._fts_available = True
        except sqlite3.OperationalError:
            self._fts_available = False

    def _probe_fts(self) -> bool:
        try:
            with self._connection() as con:
                con.execute(
                    "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH ?",
                    ("probeunlikelytermxyz",),
                ).fetchone()
            return True
        except sqlite3.OperationalError:
            return False

    def append(
        self,
        session_id: str,
        seq: int,
        role: str,
        content: str,
        importance: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> tuple[int, bool, Optional[str]]:
        """Append one row. Returns (message_id, is_new, diagnostic).

        Idempotent: re-ingesting an identical (session, seq, fingerprint)
        triple is a no-op. Same (session, seq) with different content is
        stored as a separate row plus a `seq_conflict` diagnostic — the old
        row is never overwritten.
        """
        ts = timestamp if timestamp is not None else time.time()
        fp = fingerprint(session_id, seq, role, content)
        diagnostic: Optional[str] = None
        with self._connection() as con:
            existing = con.execute(
                "SELECT id, fingerprint FROM messages"
                " WHERE session_id = ? AND seq = ?",
                (session_id, seq),
            ).fetchall()
            if any(f != fp for _, f in existing):
                # Same position, different content: keep both, flag it.
                diagnostic = (
                    f"seq_conflict: session={session_id} seq={seq} "
                    "content differs from stored row; appended separately"
                )
            cur = con.execute(
                "INSERT OR IGNORE INTO messages"
                "(session_id, seq, role, content, timestamp, fingerprint, importance)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, seq, role, content, ts, fp, importance),
            )
            if cur.lastrowid:
                return cur.lastrowid, True, diagnostic
            row = con.execute(
                "SELECT id FROM messages"
                " WHERE session_id = ? AND seq = ? AND fingerprint = ?",
                (session_id, seq, fp),
            ).fetchone()
            return row[0], False, diagnostic

    def set_formation_status(
        self, message_id: int, session_id: str, status: str,
        diagnostics: Optional[list[str]] = None,
    ) -> None:
        """Record idempotent derived-memory formation progress."""
        if status not in ("processing", "complete", "incomplete"):
            raise ValueError("invalid formation status")
        payload = json.dumps(list(diagnostics or []), ensure_ascii=False)
        with self._connection() as con:
            con.execute(
                "INSERT INTO formation_status"
                "(message_id, session_id, status, attempts, diagnostics, updated_at)"
                " VALUES (?, ?, ?, 1, ?, ?)"
                " ON CONFLICT(message_id) DO UPDATE SET"
                " session_id=excluded.session_id, status=excluded.status,"
                " attempts=formation_status.attempts + 1,"
                " diagnostics=excluded.diagnostics, updated_at=excluded.updated_at",
                (int(message_id), session_id, status, payload, time.time()),
            )

    def formation_status(
        self, message_id: Optional[int] = None,
        session_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Read derived-memory formation progress for diagnostics/replay."""
        query = (
            "SELECT message_id, session_id, status, attempts, diagnostics,"
            " updated_at FROM formation_status WHERE 1=1"
        )
        params: list[Any] = []
        if message_id is not None:
            query += " AND message_id = ?"
            params.append(int(message_id))
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY message_id ASC"
        with self._connection() as con:
            rows = con.execute(query, params).fetchall()
        out = []
        for row in rows:
            try:
                details = json.loads(row[4])
            except (TypeError, ValueError):
                details = []
            out.append({
                "message_id": row[0], "session_id": row[1],
                "status": row[2], "attempts": row[3],
                "diagnostics": details, "updated_at": row[5],
            })
        return out

    def enqueue_formation_job(self, message_id: int, session_id: str) -> bool:
        """Queue one typed-formation job idempotently."""
        now = time.time()
        with self._connection() as con:
            cur = con.execute(
                "INSERT OR IGNORE INTO formation_jobs"
                "(message_id, session_id, status, attempts, last_error, created_at, updated_at)"
                " VALUES (?, ?, 'pending', 0, '', ?, ?)",
                (int(message_id), session_id, now, now),
            )
            return bool(cur.rowcount)

    def claim_formation_jobs(self, limit: int = 8) -> list[dict[str, Any]]:
        """Claim pending/stale jobs transactionally for restart-safe draining."""
        limit = max(1, min(int(limit), 100))
        now = time.time()
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            # A crashed worker leaves processing behind. Requeue it on the
            # next drain; jobs are idempotent by message_id.
            con.execute(
                "UPDATE formation_jobs SET status='pending', updated_at=? "
                "WHERE status='processing'", (now,)
            )
            rows = con.execute(
                "SELECT job_id, message_id, session_id, attempts FROM formation_jobs "
                "WHERE status IN ('pending', 'failed') AND attempts < 5 "
                "ORDER BY job_id LIMIT ?", (limit,)
            ).fetchall()
            jobs = []
            for job_id, message_id, session_id, attempts in rows:
                con.execute(
                    "UPDATE formation_jobs SET status='processing', attempts=?, updated_at=? "
                    "WHERE job_id=? AND status IN ('pending', 'failed')",
                    (int(attempts) + 1, now, int(job_id)),
                )
                jobs.append({
                    "job_id": int(job_id), "message_id": int(message_id),
                    "session_id": session_id, "attempts": int(attempts) + 1,
                })
            return jobs

    def finish_formation_job(
        self, job_id: int, status: str, error: str = "",
    ) -> None:
        if status not in ("complete", "failed", "pending"):
            raise ValueError("invalid formation job status")
        with self._connection() as con:
            con.execute(
                "UPDATE formation_jobs SET status=?, last_error=?, updated_at=? "
                "WHERE job_id=?",
                (status, str(error or ""), time.time(), int(job_id)),
            )

    def formation_jobs(
        self, session_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT job_id, message_id, session_id, status, attempts, "
            "last_error, created_at, updated_at FROM formation_jobs WHERE 1=1"
        )
        params: list[Any] = []
        if session_id is not None:
            query += " AND session_id=?"
            params.append(session_id)
        if status is not None:
            query += " AND status=?"
            params.append(status)
        query += " ORDER BY job_id"
        with self._connection() as con:
            rows = con.execute(query, params).fetchall()
        return [
            {"job_id": r[0], "message_id": r[1], "session_id": r[2],
             "status": r[3], "attempts": r[4], "last_error": r[5],
             "created_at": r[6], "updated_at": r[7]}
            for r in rows
        ]

    def save_experience_event(
        self, session_id: str, source_message_id: int,
        kind: str, actor: str, statement: str,
        occurred_at: Optional[float] = None,
        occurred_precision: str = "unknown",
        occurred_timezone: str = "UTC",
    ) -> Optional[int]:
        """Persist verified user provenance once; never trust caller timestamps.

        Event status is *user-reported*, not external factual verification.
        Only the exact source user message is permitted as event evidence.
        """
        if kind not in (
            "decision", "completion", "resolution", "start", "attempt",
            "activity", "discussion", "milestone", "end", "setback",
        ) or actor not in ("i", "we"):
            raise ValueError("invalid event classification")
        if occurred_precision not in ("unknown", "day", "instant"):
            raise ValueError("invalid occurred_precision")
        if occurred_at is not None:
            occurred_at = self._normalize_timestamp(occurred_at, "occurred_at")
        with self._connection() as con:
            row = con.execute(
                "SELECT session_id, role, content, timestamp FROM messages WHERE id = ?",
                (source_message_id,),
            ).fetchone()
            if row is None or row[0] != session_id or row[1] != "user":
                raise ValueError("event source must be a user message in this session")
            if row[2].strip() != statement.strip():
                raise ValueError("event must preserve the original user statement")
            con.execute(
                "INSERT OR IGNORE INTO experience_events "
                "(session_id, source_message_id, kind, actor, statement, "
                "reported_at, authority, occurred_at, occurred_precision, "
                "occurred_timezone) VALUES (?, ?, ?, ?, ?, ?, 'user', ?, ?, ?)",
                (session_id, source_message_id, kind, actor, statement, row[3],
                 occurred_at, occurred_precision, occurred_timezone),
            )
            found = con.execute(
                "SELECT id FROM experience_events WHERE source_message_id = ?",
                (source_message_id,),
            ).fetchone()
            return int(found[0]) if found else None

    def fetch_experience_events(
        self, *, session_id: Optional[str] = None, kind: Optional[str] = None,
        limit: int = 100,
    ) -> list[ExperienceEvent]:
        """Retrieve user-sourced reports in chronological order, across sessions.

        This read API does not inject records into model context. Later
        episodic retrieval must select and verify evidence deliberately.
        """
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        query = (
            "SELECT e.id, e.session_id, e.source_message_id, e.kind, "
            "e.actor, e.statement, e.reported_at, e.authority, e.occurred_at,"
            " e.occurred_precision, e.occurred_timezone "
            "FROM experience_events e JOIN messages m "
            "ON m.id = e.source_message_id "
            "WHERE e.authority = 'user' AND m.role = 'user' "
            "AND m.session_id = e.session_id AND m.content = e.statement "
            "AND m.timestamp = e.reported_at"
        )
        params: list[Any] = []
        if session_id is not None:
            query += " AND e.session_id = ?"
            params.append(session_id)
        if kind is not None:
            query += " AND e.kind = ?"
            params.append(kind)
        query += " ORDER BY e.reported_at DESC, e.id DESC LIMIT ?"
        params.append(limit)
        with self._connection() as con:
            rows = con.execute(query, params).fetchall()
        return [ExperienceEvent(*row) for row in reversed(rows)]

    def fetch_legacy_experience_reports(self, *, limit: int = 5000) -> list[ExperienceEvent]:
        """Read-only fallback for user reports predating experience capture.

        The original message is the evidence; no rows are backfilled or
        rewritten. Bounded scanning avoids loading unlimited chat history.
        """
        from .episodes import extract_experience_claim
        if not 1 <= limit <= 10000:
            raise ValueError("invalid legacy experience scan limit")
        with self._connection() as con:
            rows = con.execute(
                "SELECT m.id, m.session_id, m.content, m.timestamp "
                "FROM messages m WHERE m.role = 'user' "
                "AND length(m.content) <= 1500 "
                "AND NOT EXISTS (SELECT 1 FROM experience_events e "
                "WHERE e.source_message_id = m.id) "
                "ORDER BY m.id DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for message_id, session_id, statement, timestamp in rows:
            claim = extract_experience_claim(statement, reported_at=timestamp)
            if claim is not None:
                result.append(ExperienceEvent(
                    -message_id, session_id, message_id, claim.kind,
                    claim.actor, claim.statement, timestamp, "user",
                    claim.occurred_at, claim.occurred_precision,
                    claim.occurred_timezone,
                ))
        return result

    def immediate_user_reaction(self, source_message_id: int) -> Optional[StoredMessage]:
        """Next user statement only, after a verified experience source.

        This is supporting evidence, not proof that the subject of a vague
        reaction was independently resolved. No changes to memory are made.
        """
        import re
        with self._connection() as con:
            source = con.execute(
                "SELECT session_id FROM messages "
                "WHERE id = ? AND role = 'user'", (source_message_id,)
            ).fetchone()
            if source is None:
                return None
            row = con.execute(
                "SELECT id, session_id, seq, role, content, timestamp, "
                "fingerprint, importance FROM messages WHERE session_id = ? "
                "AND id > ? AND role = 'user' ORDER BY id ASC LIMIT 1",
                (source[0], source_message_id)
            ).fetchone()
        if row is None or not re.fullmatch(
            r"\s*i (?:really )?(?:loved?|liked?|enjoyed?|hated?|disliked?) "
            r"(?:it|that|them)[.! ]*", row[4], re.I
        ):
            return None
        return StoredMessage(*row)

    def count(self, session_id: str) -> int:
        with self._connection() as con:
            row = con.execute(
                "SELECT count(*) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return int(row[0])

    def fetch_range(
        self,
        session_id: str,
        start_seq: int = 0,
        end_seq: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[StoredMessage]:
        query = (
            "SELECT id, session_id, seq, role, content, timestamp,"
            " fingerprint, importance FROM messages"
            " WHERE session_id = ? AND seq >= ?"
        )
        params: list[Any] = [session_id, start_seq]
        if end_seq is not None:
            query += " AND seq <= ?"
            params.append(end_seq)
        query += " ORDER BY seq ASC, id ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._connection() as con:
            rows = con.execute(query, params).fetchall()
        return [StoredMessage(*r) for r in rows]

    def fetch_message(self, message_id: int) -> Optional[StoredMessage]:
        """One stored message by id. Used by the live hybrid admission path."""
        with self._connection() as con:
            row = con.execute(
                "SELECT id, session_id, seq, role, content, timestamp,"
                " fingerprint, importance FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
        return StoredMessage(*row) if row is not None else None

    def has_user_content(self, content: str, *, exclude_id: Optional[int] = None) -> bool:
        """Whether an identical user source was already processed.

        Session replay can legitimately append the same immutable source to a
        new conversation session.  The source row remains distinct for
        provenance, while model-backed formation need not pay for a second
        interpretation of byte-identical text.
        """
        query = "SELECT 1 FROM messages WHERE role = 'user' AND content = ?"
        params: list[Any] = [str(content)]
        if exclude_id is not None:
            query += " AND id != ?"
            params.append(int(exclude_id))
        query += " LIMIT 1"
        with self._connection() as con:
            return con.execute(query, params).fetchone() is not None

    def fetch_messages_by_ids(
        self, message_ids: list[int]
    ) -> list[StoredMessage]:
        """Bounded bulk fetch preserving the requested id order.

        The live hybrid path resolves fused items to their evidence messages
        in one round trip instead of one query per item.
        """
        if not message_ids:
            return []
        unique = list(dict.fromkeys(int(v) for v in message_ids))
        with self._connection() as con:
            rows = con.execute(
                "SELECT id, session_id, seq, role, content, timestamp,"
                " fingerprint, importance FROM messages"
                f" WHERE id IN ({','.join('?' for _ in unique)})",
                unique,
            ).fetchall()
        found = {row[0]: StoredMessage(*row) for row in rows}
        return [found[v] for v in unique if v in found]

    def fetch_latest(
        self, session_id: str, limit: int
    ) -> list[StoredMessage]:
        with self._connection() as con:
            rows = con.execute(
                "SELECT id, session_id, seq, role, content, timestamp,"
                " fingerprint, importance FROM messages"
                " WHERE session_id = ? ORDER BY seq DESC, id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [StoredMessage(*r) for r in reversed(rows)]

    def fetch_conversation(
        self, session_id: str, limit: int = 30
    ) -> list[StoredMessage]:
        """Latest conversational turns in chronological (insertion) order.

        Row id is the chronology: legacy rows share seq positions, so
        seq-ordered fetches interleave them wrong. Protocol traffic (system
        instructions, prime_request bodies) is not conversation and is
        excluded server-side so ``limit`` counts real turns only.
        Everything at or before the newest clear marker is excluded as
        well.
        """
        if limit <= 0:
            return []
        with self._connection() as con:
            marker = con.execute(
                "SELECT id FROM messages"
                " WHERE session_id = ? AND role = 'system' AND content = ?"
                " ORDER BY id DESC LIMIT 1",
                (session_id, CONVERSATION_CLEARED),
            ).fetchone()
            after_id = marker[0] if marker is not None else -1
            rows = con.execute(
                "SELECT id, session_id, seq, role, content, timestamp,"
                " fingerprint, importance FROM messages"
                " WHERE session_id = ? AND id > ?"
                " AND role IN ('user', 'assistant')"
                " AND content NOT LIKE ?"
                " ORDER BY id DESC LIMIT ?",
                (session_id, after_id, "%<prime_request>%", int(limit)),
            ).fetchall()
        return [StoredMessage(*r) for r in reversed(rows)]

    def max_seq(self, session_id: str) -> int:
        """Highest stored sequence in this session, or -1 when there is none."""
        with self._connection() as con:
            row = con.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM messages"
                " WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row[0])

    @staticmethod
    def _fts_phrase(term: str) -> str:
        """Render one user term as a single FTS5 phrase literal.

        A raw term is injected into the `MATCH` grammar as-is, so any
        character the tokenizer can keep (an apostrophe, a hyphen, a
        question mark) is read as FTS5 syntax and raises
        `OperationalError`; the caller then drops the term silently.
        Quoting makes the whole term one literal phrase, which the
        `unicode61` index tokenizes the same way it tokenized the stored
        text. Safe words are unaffected: a one-token phrase matches
        exactly like the bare token.
        """
        return '"' + term.replace('"', '""') + '"'

    def search(
        self, session_id: Optional[str], query: str, limit: int = 10
    ) -> list[StoredMessage]:
        """Full-text search; session_id=None searches all sessions.

        LIKE fallback when no FTS5. Results carry their session_id so
        callers can separate current-session history from cross-session
        history. ORDER BY is deterministic (session, seq).

        FTS5 mode passes the term as a quoted phrase, so user punctuation
        can never escape into MATCH syntax. The LIKE fallback keeps the
        raw term: there, punctuation is an ordinary substring.
        """
        if self.fts_available:
            sql = (
                "SELECT m.id, m.session_id, m.seq, m.role, m.content,"
                " m.timestamp, m.fingerprint, m.importance"
                " FROM messages_fts f JOIN messages m ON m.id = f.rowid"
                " WHERE messages_fts MATCH ?"
            )
            params: list[Any] = [self._fts_phrase(query)]
            if session_id is not None:
                sql += " AND m.session_id = ?"
                params.append(session_id)
            sql += " ORDER BY m.session_id ASC, m.seq ASC LIMIT ?"
            params.append(limit)
        else:
            sql = (
                "SELECT id, session_id, seq, role, content, timestamp,"
                " fingerprint, importance FROM messages"
                " WHERE content LIKE ?"
            )
            params = [f"%{query}%"]
            if session_id is not None:
                sql += " AND session_id = ?"
                params.append(session_id)
            sql += " ORDER BY session_id ASC, seq ASC LIMIT ?"
            params.append(limit)
        with self._connection() as con:
            rows = con.execute(sql, params).fetchall()
        return [StoredMessage(*r) for r in rows]

    def search_user_evidence(
        self, terms: tuple[str, ...] = (), *, start: Optional[float] = None,
        end: Optional[float] = None, limit: int = 180,
        oldest_first: bool = False,
    ) -> list[StoredMessage]:
        """Bounded cross-session *raw* user evidence, even if formation missed it.

        SQL is parameterized and only original user messages qualify. Word
        matching supplies candidates; Aurora decides their meaning. No memory
        indexes or existing structured records are mutated by this query.
        """
        if not 1 <= limit <= 300:
            raise ValueError("evidence limit must be 1..300")
        parts = ["role = 'user'"]
        params: list[Any] = []
        if start is not None:
            parts.append("timestamp >= ?")
            params.append(float(start))
        if end is not None:
            parts.append("timestamp < ?")
            params.append(float(end))
        if terms:
            parts.append("(" + " OR ".join(
                "content LIKE ? ESCAPE '\\'" for _ in terms) + ")")
            params.extend("%" + term.replace("\\", "\\\\").replace("%", "\\%")
                          .replace("_", "\\_") + "%" for term in terms)
        order = "ASC" if oldest_first else "DESC"
        params.append(limit)
        with self._connection() as con:
            rows = con.execute(
                "SELECT id, session_id, seq, role, content, timestamp, "
                "fingerprint, importance FROM messages WHERE " +
                " AND ".join(parts) + f" ORDER BY timestamp {order}, id {order} LIMIT ?",
                params,
            ).fetchall()
        return [StoredMessage(*row) for row in rows]

    def neighboring_user_evidence(self, message: StoredMessage) -> list[StoredMessage]:
        """Two preceding and following user turns; may be unrelated.

        Bound to one session and a two-hour interval. These rows are *context*,
        not proof that a later reaction referred to the earlier subject.
        """
        with self._connection() as con:
            rows = con.execute(
                "SELECT id, session_id, seq, role, content, timestamp, "
                "fingerprint, importance FROM messages "
                "WHERE role = 'user' AND session_id = ? AND id != ? "
                "AND timestamp BETWEEN ? AND ? "
                "ORDER BY ABS(id - ?) ASC LIMIT 6",
                (message.session_id, message.message_id,
                 message.timestamp - 7200, message.timestamp + 7200,
                 message.message_id),
            ).fetchall()
        return [StoredMessage(*row) for row in rows[:4]]

    def save_digest(
        self,
        session_id: str,
        source_ids: list[int],
        seq_start: int,
        seq_end: int,
        digest_type: str,
        content: str,
    ) -> int:
        with self._connection() as con:
            cur = con.execute(
                "INSERT INTO digests"
                "(session_id, source_ids, seq_start, seq_end,"
                " created_at, digest_type, content)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    ",".join(str(i) for i in source_ids),
                    seq_start,
                    seq_end,
                    time.time(),
                    digest_type,
                    content,
                ),
            )
            return cur.lastrowid

    @staticmethod
    def _decode_titles(raw: Any) -> list[str]:
        import json as _json

        try:
            items = _json.loads(raw or "[]")
            return [str(x) for x in items if isinstance(x, str)]
        except Exception:
            return []

    def _row_to_session(self, row) -> SessionRecord:
        return SessionRecord(
            session_id=row[0], created_at=row[1], last_activity_at=row[2],
            title=row[3], status=row[4],
            previous_titles=self._decode_titles(row[5]),
        )

    def ensure_session(
        self,
        session_id: str,
        title: str = "",
        timestamp: Optional[float] = None,
    ) -> SessionRecord:
        """Create the session row if absent; always touch activity time."""
        ts = timestamp if timestamp is not None else time.time()
        with self._connection() as con:
            con.execute(
                "INSERT OR IGNORE INTO sessions"
                "(session_id, created_at, last_activity_at, title, status)"
                " VALUES (?, ?, ?, ?, 'active')",
                (session_id, ts, ts, title),
            )
            con.execute(
                "UPDATE sessions SET last_activity_at = ? WHERE session_id = ?",
                (ts, session_id),
            )
            row = con.execute(
                "SELECT session_id, created_at, last_activity_at, title,"
                " status, previous_titles"
                " FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._row_to_session(row)

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        with self._connection() as con:
            row = con.execute(
                "SELECT session_id, created_at, last_activity_at, title,"
                " status, previous_titles"
                " FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def list_sessions(self) -> list[SessionRecord]:
        with self._connection() as con:
            rows = con.execute(
                "SELECT session_id, created_at, last_activity_at, title,"
                " status, previous_titles"
                " FROM sessions ORDER BY last_activity_at DESC"
            ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def rename_session(self, session_id: str, title: str) -> Optional[SessionRecord]:
        """Rename a session. Identity stable; old name kept as provenance.

        A rename is metadata only: no raw row is created, modified, or
        deleted. Returns the updated record, or None if unknown.
        """
        import json as _json

        with self._connection() as con:
            row = con.execute(
                "SELECT session_id, created_at, last_activity_at, title,"
                " status, previous_titles"
                " FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            previous = self._decode_titles(row[5])
            old_title = row[3] or ""
            new_title = (title or "").strip()
            if old_title and old_title != new_title and old_title not in previous:
                previous.append(old_title)
            previous = previous[-10:]
            con.execute(
                "UPDATE sessions SET title = ?, previous_titles = ?,"
                " last_activity_at = ? WHERE session_id = ?",
                (new_title, _json.dumps(previous), time.time(), session_id),
            )
            updated = con.execute(
                "SELECT session_id, created_at, last_activity_at, title,"
                " status, previous_titles"
                " FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._row_to_session(updated)

    def set_session_status(self, session_id: str, status: str) -> None:
        with self._connection() as con:
            con.execute(
                "UPDATE sessions SET status = ?, last_activity_at = ?"
                " WHERE session_id = ?",
                (status, time.time(), session_id),
            )

    @staticmethod
    def _normalize_timestamp(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a normalized numeric timestamp")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{name} must be finite")
        return normalized

    @staticmethod
    def _validate_interval(
        valid_from: Optional[float], valid_until: Optional[float]
    ) -> None:
        if (valid_from is not None and valid_until is not None
                and valid_until < valid_from):
            raise ValueError("valid_until must not be earlier than valid_from")

    def _resolve_observed_at(
        self,
        con,
        source_ids: list[int],
        created_at: float,
        observed_at: Optional[float],
    ) -> float:
        if observed_at is not None:
            return self._normalize_timestamp(observed_at, "observed_at")
        for source_id in source_ids:
            row = con.execute(
                "SELECT timestamp FROM messages WHERE id = ?", (source_id,)
            ).fetchone()
            if row is not None:
                try:
                    return self._normalize_timestamp(row[0], "message timestamp")
                except ValueError:
                    pass
        return created_at

    def _insert_state_row(
        self,
        con,
        session_id: str,
        scope: str,
        concept: str,
        value: str,
        source_ids: list[int],
        created_at: float,
        observed_at: float,
        valid_from: float,
        valid_until: Optional[float],
        status: str,
        supersedes_id: Optional[int],
        authority: str,
        superseded_at: Optional[float],
        memory_type: str,
        confidence: Optional[float],
    ) -> int:
        cur = con.execute(
            "INSERT INTO states"
            "(session_id, scope, concept, value, source_ids,"
            " created_at, status, supersedes_id, authority, memory_type,"
            " confidence, observed_at, valid_from, valid_until, superseded_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, scope, concept, value,
                ",".join(str(i) for i in source_ids),
                created_at, status, supersedes_id, authority, memory_type,
                confidence, observed_at, valid_from, valid_until, superseded_at,
            ),
        )
        return int(cur.lastrowid)

    def create_state(
        self,
        session_id: str,
        scope: str,
        concept: str,
        value: str,
        source_ids: list[int],
        status: str = "current",
        supersedes_id: Optional[int] = None,
        authority: str = "user",
        timestamp: Optional[float] = None,
        memory_type: str = "",
        confidence: Optional[float] = None,
        observed_at: Optional[float] = None,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
        superseded_at: Optional[float] = None,
    ) -> int:
        if authority != "user" and status == "current":
            raise ValueError(
                "only user-authored states may be current "
                f"(got authority={authority!r})"
            )
        created_at = (
            self._normalize_timestamp(timestamp, "created_at")
            if timestamp is not None else time.time()
        )
        normalized_until = (
            self._normalize_timestamp(valid_until, "valid_until")
            if valid_until is not None else None
        )
        normalized_superseded_at = (
            self._normalize_timestamp(superseded_at, "superseded_at")
            if superseded_at is not None else None
        )
        with self._connection() as con:
            self._validate_user_source_roles(con, source_ids, authority, session_id, scope)
            observed = self._resolve_observed_at(
                con, source_ids, created_at, observed_at
            )
            effective_from = (
                self._normalize_timestamp(valid_from, "valid_from")
                if valid_from is not None else observed
            )
            self._validate_interval(effective_from, normalized_until)
            return self._insert_state_row(
                con, session_id, scope, concept, value, source_ids,
                created_at, observed, effective_from, normalized_until,
                status, supersedes_id, authority, normalized_superseded_at,
                memory_type, confidence,
            )

    def create_state_with_explains(
        self,
        session_id: str,
        scope: str,
        concept: str,
        value: str,
        source_ids: list[int],
        explains_state_id: int,
        memory_type: str = "RATIONALE",
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
        observed_at: Optional[float] = None,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
    ) -> tuple[int, int]:
        sources = list(dict.fromkeys(int(item) for item in source_ids))
        if not sources:
            raise ValueError("explained state requires source provenance")
        created_at = (
            self._normalize_timestamp(timestamp, "created_at")
            if timestamp is not None else time.time()
        )
        normalized_from = (
            self._normalize_timestamp(valid_from, "valid_from")
            if valid_from is not None else None
        )
        normalized_until = (
            self._normalize_timestamp(valid_until, "valid_until")
            if valid_until is not None else None
        )
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            self._validate_user_source_roles(con, sources, "user", session_id, scope)
            explained = con.execute(
                "SELECT scope, status, authority FROM states WHERE id = ?",
                (explains_state_id,),
            ).fetchone()
            if explained is None:
                raise ValueError(f"state {explains_state_id} not found")
            if explained[0] != scope:
                raise ValueError("explained state scope does not match")
            if explained[1] != "current" or explained[2] != "user":
                raise ValueError("explained state must be current user state")
            observed = self._resolve_observed_at(
                con, sources, created_at, observed_at
            )
            effective_from = normalized_from if normalized_from is not None else observed
            self._validate_interval(effective_from, normalized_until)
            state_id = self._insert_state_row(
                con, session_id, scope, concept, value, sources,
                created_at, observed, effective_from, normalized_until,
                "current", None, "user", None, memory_type, confidence,
            )
            relationship_id = self._insert_relationship_row(
                con,
                session_id,
                scope,
                "explains",
                "memory_state",
                state_id,
                "memory_state",
                explains_state_id,
                sources,
                "current",
                "user",
                confidence,
                created_at,
                observed,
                effective_from,
                normalized_until,
                None,
                None,
            )
        return state_id, relationship_id

    def create_state_successor(
        self,
        session_id: str,
        scope: str,
        concept: str,
        value: str,
        source_ids: list[int],
        predecessor_ids: list[int],
        authority: str = "user",
        timestamp: Optional[float] = None,
        memory_type: str = "",
        confidence: Optional[float] = None,
        observed_at: Optional[float] = None,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
        transition_time: Optional[float] = None,
    ) -> int:
        if authority != "user":
            raise ValueError("only user-authored states may replace current state")
        created_at = (
            self._normalize_timestamp(timestamp, "created_at")
            if timestamp is not None else time.time()
        )
        transition = (
            self._normalize_timestamp(transition_time, "transition_time")
            if transition_time is not None else created_at
        )
        normalized_until = (
            self._normalize_timestamp(valid_until, "valid_until")
            if valid_until is not None else None
        )
        expected_ids = list(dict.fromkeys(int(i) for i in predecessor_ids))
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            self._validate_user_source_roles(con, source_ids, authority, session_id, scope)
            current_rows = con.execute(
                "SELECT id FROM states"
                " WHERE scope = ? AND concept = ? AND status = 'current'"
                " ORDER BY id ASC",
                (scope, concept),
            ).fetchall()
            current_ids = [int(row[0]) for row in current_rows]
            if set(current_ids) != set(expected_ids):
                raise ValueError("current state changed during replacement")
            observed = self._resolve_observed_at(
                con, source_ids, created_at, observed_at
            )
            effective_from = (
                self._normalize_timestamp(valid_from, "valid_from")
                if valid_from is not None else observed
            )
            self._validate_interval(effective_from, normalized_until)
            predecessor_id = current_ids[0] if current_ids else None
            new_id = self._insert_state_row(
                con, session_id, scope, concept, value, source_ids,
                created_at, observed, effective_from, normalized_until,
                "current", predecessor_id, authority, None, memory_type,
                confidence,
            )
            for state_id in current_ids:
                cur = con.execute(
                    "UPDATE states SET status = 'superseded',"
                    " valid_from = CASE WHEN valid_from IS NOT NULL"
                    " AND ? < valid_from THEN NULL ELSE valid_from END,"
                    " valid_until = ?, superseded_at = ?"
                    " WHERE id = ? AND status = 'current'",
                    (effective_from, effective_from, transition, state_id),
                )
                if cur.rowcount != 1:
                    raise ValueError("predecessor changed during replacement")
            return new_id

    def supersede_states(
        self,
        state_ids: list[int],
        valid_until: Optional[float] = None,
        transition_time: Optional[float] = None,
    ) -> None:
        ids = list(dict.fromkeys(int(i) for i in state_ids))
        if not ids:
            return
        transition = (
            self._normalize_timestamp(transition_time, "transition_time")
            if transition_time is not None else time.time()
        )
        end = (
            self._normalize_timestamp(valid_until, "valid_until")
            if valid_until is not None else transition
        )
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            for state_id in ids:
                row = con.execute(
                    "SELECT status, valid_from FROM states WHERE id = ?",
                    (state_id,),
                ).fetchone()
                if row is None or row[0] != "current":
                    raise ValueError("state changed during supersession")
                cur = con.execute(
                    "UPDATE states SET status = 'superseded',"
                    " valid_from = CASE WHEN valid_from IS NOT NULL"
                    " AND ? < valid_from THEN NULL ELSE valid_from END,"
                    " valid_until = ?, superseded_at = ?"
                    " WHERE id = ? AND status = 'current'",
                    (end, end, transition, state_id),
                )
                if cur.rowcount != 1:
                    raise ValueError("state changed during supersession")

    def set_state_status(self, state_id: int, status: str) -> None:
        """Transition a state row (e.g. current -> superseded). Raw kept."""
        with self._connection() as con:
            con.execute(
                "UPDATE states SET status = ? WHERE id = ?",
                (status, state_id),
            )

    # ------------------------------------------------------------------
    # Evidence-backed observations (additive V2 surface)
    # ------------------------------------------------------------------
    @staticmethod
    def _observation_normalize(value: str) -> str:
        return " ".join(str(value).strip().casefold().split())

    def _observation_row_to_record(self, row) -> ObservationRecord:
        return ObservationRecord(
            observation_id=int(row[0]), session_id=row[1], scope=row[2],
            concept=row[3], normalized_concept=row[4], value=row[5],
            observation_type=row[6], actor=row[7], subject=row[8],
            predicate=row[9], authority=row[10], status=row[11],
            confidence=float(row[12]), created_at=row[13], observed_at=row[14],
            occurred_from=row[15], occurred_until=row[16], valid_from=row[17],
            valid_until=row[18], supersedes_id=row[19], superseded_at=row[20],
            source_ids=[int(v) for v in (row[21] or "").split(",") if v],
            source_episode_ids=[v for v in (row[22] or "").split(",") if v],
            proof_count=int(row[23] or 0),
        )

    def _observation_select(self) -> str:
        return (
            "SELECT o.id, o.session_id, o.scope, o.concept,"
            " o.normalized_concept, o.value, o.observation_type, o.actor,"
            " o.subject, o.predicate, o.authority, o.status, o.confidence,"
            " o.created_at, o.observed_at, o.occurred_from, o.occurred_until,"
            " o.valid_from, o.valid_until, o.supersedes_id, o.superseded_at,"
            " COALESCE((SELECT group_concat(DISTINCT source_message_id)"
            "   FROM observation_sources s WHERE s.observation_id=o.id"
            "     AND s.source_message_id IS NOT NULL), ''),"
            " COALESCE((SELECT group_concat(DISTINCT source_episode_id)"
            "   FROM observation_sources s WHERE s.observation_id=o.id"
            "     AND s.source_episode_id IS NOT NULL), ''),"
            " (SELECT count(*) FROM observation_sources s"
            "   WHERE s.observation_id=o.id)"
            " FROM observations o"
        )

    def create_observation(
        self,
        session_id: str,
        scope: str,
        concept: str,
        value: str,
        source_ids: list[int],
        *,
        source_quotes: Optional[dict[int, str]] = None,
        source_episode_ids: Optional[list[str]] = None,
        observation_type: str = "observation",
        actor: str = "user",
        subject: str = "",
        predicate: str = "",
        authority: str = "user",
        status: str = "current",
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
        observed_at: Optional[float] = None,
        occurred_from: Optional[float] = None,
        occurred_until: Optional[float] = None,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
        supersedes_id: Optional[int] = None,
    ) -> int:
        """Insert an observation and its deduplicated evidence links.

        This low-level store method performs no model interpretation.  The
        observation service validates authority and source ownership before
        calling it; the checks here make direct callers fail closed too.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        if scope != "global" and scope != session_id:
            raise ValueError("observation scope must be global or session_id")
        if authority != "user" and status == "current":
            raise ValueError("only user observations may be current")
        if status not in ("current", "superseded", "unresolved", "candidate", "rejected"):
            raise ValueError("invalid observation status")
        if not str(concept).strip() or not str(value).strip():
            raise ValueError("observation concept and value are required")
        ids = list(dict.fromkeys(int(v) for v in source_ids))
        episodes = list(dict.fromkeys(str(v).strip() for v in (source_episode_ids or []) if str(v).strip()))
        if not ids and not episodes:
            raise ValueError("observation requires source provenance")
        created = self._normalize_timestamp(timestamp, "created_at") if timestamp is not None else time.time()
        normalized_from = self._normalize_timestamp(valid_from, "valid_from") if valid_from is not None else None
        normalized_until = self._normalize_timestamp(valid_until, "valid_until") if valid_until is not None else None
        occurred_start = self._normalize_timestamp(occurred_from, "occurred_from") if occurred_from is not None else None
        occurred_end = self._normalize_timestamp(occurred_until, "occurred_until") if occurred_until is not None else None
        self._validate_interval(normalized_from, normalized_until)
        self._validate_interval(occurred_start, occurred_end)
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            for source_id in ids:
                row = con.execute(
                    "SELECT session_id, role, timestamp FROM messages WHERE id = ?",
                    (source_id,),
                ).fetchone()
                if row is None or row[1] != "user":
                    raise ValueError("observation sources must be user messages")
                if scope != "global" and row[0] != scope:
                    raise ValueError("observation source crosses scope")
            observed = self._resolve_observed_at(con, ids, created, observed_at)
            effective_from = normalized_from if normalized_from is not None else observed
            self._validate_interval(effective_from, normalized_until)
            cur = con.execute(
                "INSERT INTO observations"
                "(session_id, scope, concept, normalized_concept, value,"
                " observation_type, actor, subject, predicate, authority,"
                " status, confidence, created_at, observed_at, occurred_from,"
                " occurred_until, valid_from, valid_until, supersedes_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, scope, concept.strip(), self._observation_normalize(concept),
                 value.strip(), observation_type, actor, subject, predicate,
                 authority, status, float(confidence), created, observed,
                 occurred_start, occurred_end, effective_from, normalized_until,
                 supersedes_id),
            )
            observation_id = int(cur.lastrowid)
            for source_id in ids:
                con.execute(
                    "INSERT OR IGNORE INTO observation_sources"
                    "(observation_id, source_message_id, source_episode_id,"
                    " source_quote, source_role, linked_at) VALUES (?, ?, NULL, ?, 'user', ?)",
                    (observation_id, source_id,
                     (source_quotes or {}).get(source_id, "")[:1000], created),
                )
            for episode_id in episodes:
                con.execute(
                    "INSERT OR IGNORE INTO observation_sources"
                    "(observation_id, source_message_id, source_episode_id,"
                    " source_quote, source_role, linked_at) VALUES (?, NULL, ?, '', 'user', ?)",
                    (observation_id, episode_id, created),
                )
            return observation_id

    def fetch_observations(
        self, *, session_id: Optional[str] = None, scope: Optional[str] = None,
        concept: Optional[str] = None, status: Optional[str] = None,
        normalized_concept: Optional[str] = None, at_time: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list[ObservationRecord]:
        query = self._observation_select() + " WHERE 1=1"
        params: list[Any] = []
        for column, value in (("o.session_id", session_id), ("o.scope", scope),
                              ("o.concept", concept), ("o.status", status),
                              ("o.normalized_concept", normalized_concept)):
            if value is not None:
                query += f" AND {column} = ?"
                params.append(value)
        if at_time is not None:
            stamp = self._normalize_timestamp(at_time, "at_time")
            query += " AND o.valid_from <= ? AND (o.valid_until IS NULL OR ? < o.valid_until)"
            params.extend([stamp, stamp])
        query += " ORDER BY o.id ASC"
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            query += " LIMIT ?"
            params.append(limit)
        with self._connection() as con:
            rows = con.execute(query, params).fetchall()
        return [self._observation_row_to_record(row) for row in rows]

    def fetch_observation_history(self, observation_id: int) -> list[ObservationHistoryRecord]:
        with self._connection() as con:
            rows = con.execute(
                "SELECT id, observation_id, snapshot_json, reason, changed_at,"
                " superseded_by_id FROM observation_history"
                " WHERE observation_id = ? ORDER BY id ASC", (int(observation_id),)
            ).fetchall()
        return [ObservationHistoryRecord(
            history_id=int(row[0]), observation_id=int(row[1]),
            snapshot=json.loads(row[2]), reason=row[3], changed_at=row[4],
            superseded_by_id=row[5]) for row in rows]

    def fetch_observation_sources(self, observation_id: int) -> list[dict[str, Any]]:
        """Return the immutable evidence links used to derive proof_count."""
        with self._connection() as con:
            rows = con.execute(
                "SELECT id, observation_id, source_message_id, source_episode_id,"
                " source_quote, source_role, linked_at FROM observation_sources"
                " WHERE observation_id = ? ORDER BY id ASC", (int(observation_id),)
            ).fetchall()
        return [{"id": row[0], "observation_id": row[1],
                 "source_message_id": row[2], "source_episode_id": row[3],
                 "source_quote": row[4], "source_role": row[5],
                 "linked_at": row[6]} for row in rows]

    def add_observation_evidence(
        self,
        observation_id: int,
        source_ids: list[int],
        *,
        source_quotes: Optional[dict[int, str]] = None,
        source_episode_ids: Optional[list[str]] = None,
    ) -> int:
        """Add source links atomically while leaving the observation immutable.

        The observation value and lifecycle are never edited here.  Every
        message link is checked against the same authority and scope gates as
        initial admission; duplicate links are harmless and idempotent.
        """
        ids = list(dict.fromkeys(int(value) for value in (source_ids or [])))
        episodes = list(dict.fromkeys(
            str(value).strip() for value in (source_episode_ids or [])
            if str(value).strip()
        ))
        if not ids and not episodes:
            raise ValueError("observation evidence requires a source")
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT scope, status, authority FROM observations WHERE id = ?",
                (int(observation_id),),
            ).fetchone()
            if row is None:
                raise ValueError("observation not found")
            if row[1] not in ("current", "candidate", "unresolved"):
                raise ValueError("observation is not supportable")
            scope = row[0]
            linked = 0
            for source_id in ids:
                source = con.execute(
                    "SELECT session_id, role, content FROM messages WHERE id = ?",
                    (source_id,),
                ).fetchone()
                if source is None or source[1] != "user":
                    raise ValueError("observation sources must be user messages")
                if scope != "global" and source[0] != scope:
                    raise ValueError("observation source crosses scope")
                quote = (source_quotes or {}).get(source_id, "")
                if not isinstance(quote, str) or len(quote) > 1000:
                    raise ValueError("source quote is invalid")
                if quote and quote.casefold() not in source[2].casefold():
                    raise ValueError("source quote is not supported by source message")
                cur = con.execute(
                    "INSERT OR IGNORE INTO observation_sources"
                    "(observation_id, source_message_id, source_episode_id,"
                    " source_quote, source_role, linked_at)"
                    " VALUES (?, ?, NULL, ?, 'user', ?)",
                    (int(observation_id), source_id, quote[:1000], time.time()),
                )
                linked += int(cur.rowcount > 0)
            for episode_id in episodes:
                cur = con.execute(
                    "INSERT OR IGNORE INTO observation_sources"
                    "(observation_id, source_message_id, source_episode_id,"
                    " source_quote, source_role, linked_at)"
                    " VALUES (?, NULL, ?, '', 'user', ?)",
                    (int(observation_id), episode_id, time.time()),
                )
                linked += int(cur.rowcount > 0)
            return linked

    def supersede_observation(
        self, observation_id: int, *, reason: str = "superseded",
        replacement_id: Optional[int] = None, transition_time: Optional[float] = None,
        valid_until: Optional[float] = None,
    ) -> ObservationRecord:
        transition = self._normalize_timestamp(transition_time, "transition_time") if transition_time is not None else time.time()
        end = self._normalize_timestamp(valid_until, "valid_until") if valid_until is not None else transition
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(self._observation_select() + " WHERE o.id = ?", (int(observation_id),)).fetchone()
            if row is None:
                raise ValueError("observation not found")
            current = self._observation_row_to_record(row)
            if current.status != "current":
                raise ValueError("only current observations may be superseded")
            snapshot = {"observation_id": current.observation_id, "session_id": current.session_id,
                        "scope": current.scope, "concept": current.concept,
                        "normalized_concept": current.normalized_concept, "value": current.value,
                        "observation_type": current.observation_type, "actor": current.actor,
                        "subject": current.subject, "predicate": current.predicate,
                        "source_ids": current.source_ids, "source_episode_ids": current.source_episode_ids,
                        "authority": current.authority, "status": current.status,
                        "confidence": current.confidence, "created_at": current.created_at,
                        "observed_at": current.observed_at, "occurred_from": current.occurred_from,
                        "occurred_until": current.occurred_until, "valid_from": current.valid_from,
                        "valid_until": current.valid_until, "supersedes_id": current.supersedes_id,
                        "superseded_at": current.superseded_at, "proof_count": current.proof_count}
            con.execute(
                "INSERT INTO observation_history(observation_id, snapshot_json, reason, changed_at, superseded_by_id)"
                " VALUES (?, ?, ?, ?, ?)",
                (current.observation_id, json.dumps(snapshot, ensure_ascii=False), reason, transition, replacement_id),
            )
            con.execute(
                "UPDATE observations SET status='superseded', valid_until=?, superseded_at=?"
                " WHERE id=? AND status='current'", (end, transition, current.observation_id),
            )
            updated = con.execute(self._observation_select() + " WHERE o.id = ?", (current.observation_id,)).fetchone()
        return self._observation_row_to_record(updated)

    def _state_rows_to_records(self, rows) -> list[StateRecord]:
        out = []
        for r in rows:
            authority = r[9]
            raw_confidence = r[11] if len(r) > 11 else None
            if raw_confidence is None:
                # Legacy rows predate confidence: deterministic rule only.
                raw_confidence = 1.0 if authority == "user" else 0.3
            out.append(StateRecord(
                state_id=r[0], session_id=r[1], scope=r[2], concept=r[3],
                value=r[4],
                source_ids=[int(x) for x in r[5].split(",") if x],
                created_at=r[6], status=r[7], supersedes_id=r[8],
                authority=authority,
                memory_type=(r[10] or "") if len(r) > 10 else "",
                confidence=raw_confidence,
                observed_at=r[12] if len(r) > 12 else None,
                valid_from=r[13] if len(r) > 13 else None,
                valid_until=r[14] if len(r) > 14 else None,
                superseded_at=r[15] if len(r) > 15 else None,
            ))
        return out

    def add_state_evidence(self, state_id: int, message_id: int) -> bool:
        """Attach another user message to an unchanged current state.

        Does not create a successor, change its temporal validity, or edit
        raw messages. Returns False when the source was already attached.
        """
        if isinstance(state_id, bool) or isinstance(message_id, bool):
            raise ValueError("state_id and message_id must be integers")
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT source_ids, status, authority FROM states WHERE id = ?",
                (int(state_id),),
            ).fetchone()
            if row is None or row[1] != "current" or row[2] != "user":
                raise ValueError("evidence requires a current user state")
            origin = con.execute(
                "SELECT role FROM messages WHERE id = ?", (int(message_id),)
            ).fetchone()
            if origin is None or origin[0] != "user":
                raise ValueError("evidence must reference a user message")
            ids = [int(value) for value in row[0].split(",") if value]
            if int(message_id) in ids:
                return False
            ids.append(int(message_id))
            con.execute(
                "UPDATE states SET source_ids = ? WHERE id = ?",
                (",".join(str(value) for value in ids), int(state_id)),
            )
            return True

    def fetch_states(
        self,
        session_id: Optional[str] = None,
        scope: Optional[str] = None,
        concept: Optional[str] = None,
        status: Optional[str] = None,
        state_ids: Optional[list[int]] = None,
        at_time: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list[StateRecord]:
        if state_ids is not None and not state_ids:
            return []
        query = (
            "SELECT id, session_id, scope, concept, value, source_ids,"
            " created_at, status, supersedes_id, authority, memory_type,"
            " confidence, observed_at, valid_from, valid_until, superseded_at"
            " FROM states WHERE 1=1"
        )
        params: list[Any] = []
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        if scope is not None:
            query += " AND scope = ?"
            params.append(scope)
        if concept is not None:
            query += " AND concept = ?"
            params.append(concept)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if state_ids is not None:
            query += " AND id IN (%s)" % ",".join("?" for _ in state_ids)
            params.extend(int(value) for value in state_ids)
        if at_time is not None:
            evaluation_time = self._normalize_timestamp(at_time, "at_time")
            query += (
                " AND valid_from IS NOT NULL AND valid_from <= ?"
                " AND (valid_until IS NULL OR ? < valid_until)"
            )
            params.extend([evaluation_time, evaluation_time])
        query += " ORDER BY id ASC"
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            query += " LIMIT ?"
            params.append(limit)
        with self._connection() as con:
            rows = con.execute(query, params).fetchall()
        return self._state_rows_to_records(rows)

    def create_candidate(
        self,
        session_id: str,
        scope: str,
        source_ids: list[int],
        proposed_type: str,
        proposed_concept: str,
        proposed_value: str,
        proposed_rationale: str = "",
        authority: str = "user",
        confidence: float = 0.3,
        status: str = "pending",
        timestamp: Optional[float] = None,
        expires_at: Optional[float] = None,
    ) -> int:
        with self._connection() as con:
            self._validate_user_source_roles(con, source_ids, authority, session_id, scope)
            cur = con.execute(
                "INSERT INTO candidates"
                "(session_id, scope, source_ids, proposed_type,"
                " proposed_concept, proposed_value, proposed_rationale,"
                " authority, confidence, status, created_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, scope,
                    ",".join(str(i) for i in source_ids),
                    proposed_type, proposed_concept, proposed_value,
                    proposed_rationale, authority, confidence, status,
                    timestamp if timestamp is not None else time.time(),
                    expires_at,
                ),
            )
            return cur.lastrowid

    def _candidate_rows_to_records(self, rows) -> list[CandidateRecord]:
        out = []
        for r in rows:
            out.append(CandidateRecord(
                candidate_id=r[0], session_id=r[1], scope=r[2],
                source_ids=[int(x) for x in r[3].split(",") if x],
                proposed_type=r[4] or "", proposed_concept=r[5] or "",
                proposed_value=r[6] or "", proposed_rationale=r[7] or "",
                authority=r[8] or "user", confidence=r[9],
                status=r[10], created_at=r[11], expires_at=r[12],
            ))
        return out

    def fetch_candidates(
        self,
        scope: Optional[str] = None,
        status: Optional[str] = None,
        concept: Optional[str] = None,
        candidate_id: Optional[int] = None,
    ) -> list[CandidateRecord]:
        query = (
            "SELECT id, session_id, scope, source_ids, proposed_type,"
            " proposed_concept, proposed_value, proposed_rationale,"
            " authority, confidence, status, created_at, expires_at"
            " FROM candidates WHERE 1=1"
        )
        params: list[Any] = []
        if scope is not None:
            query += " AND scope = ?"
            params.append(scope)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if concept is not None:
            query += " AND proposed_concept = ?"
            params.append(concept)
        if candidate_id is not None:
            query += " AND id = ?"
            params.append(candidate_id)
        query += " ORDER BY id ASC"
        with self._connection() as con:
            rows = con.execute(query, params).fetchall()
        return self._candidate_rows_to_records(rows)

    def set_candidate_status(self, candidate_id: int, status: str) -> None:
        with self._connection() as con:
            con.execute(
                "UPDATE candidates SET status = ? WHERE id = ?",
                (status, candidate_id),
            )

    def _entity_row_to_record(self, row) -> EntityRecord:
        return EntityRecord(
            entity_id=row[0], session_id=row[1], scope=row[2],
            entity_type=row[3], display_name=row[4],
            normalized_name=row[5], status=row[6],
            merged_into_entity_id=row[7],
            source_ids=[int(value) for value in row[8].split(",") if value],
            authority=row[9], confidence=row[10], created_at=row[11],
            observed_at=row[12], merged_at=row[13],
            source_episode_ids=[value for value in (row[14] or "").split(",") if value],
            valid_from=row[15], valid_until=row[16],
        )

    def _entity_rows_to_records(self, rows) -> list[EntityRecord]:
        return [self._entity_row_to_record(row) for row in rows]

    def _entity_select(self) -> str:
        return (
            "SELECT id, session_id, scope, entity_type, display_name,"
            " normalized_name, status, merged_into_entity_id, source_ids,"
            " authority, confidence, created_at, observed_at, merged_at,"
            " source_episode_ids, valid_from, valid_until"
            " FROM entities"
        )

    def fetch_entity(self, entity_id: int) -> Optional[EntityRecord]:
        with self._connection() as con:
            row = con.execute(
                self._entity_select() + " WHERE id = ?", (entity_id,)
            ).fetchone()
        return self._entity_row_to_record(row) if row is not None else None

    def fetch_entities(
        self,
        entity_id: Optional[int] = None,
        scope: Optional[str] = None,
        status: Optional[str] = None,
        entity_type: Optional[str] = None,
        normalized_name: Optional[str] = None,
        entity_ids: Optional[list[int]] = None,
        limit: Optional[int] = None,
        at_time: Optional[float] = None,
    ) -> list[EntityRecord]:
        if entity_ids is not None and not entity_ids:
            return []
        query = self._entity_select() + " WHERE 1=1"
        params: list[Any] = []
        if entity_id is not None:
            query += " AND id = ?"
            params.append(entity_id)
        if scope is not None:
            query += " AND scope = ?"
            params.append(scope)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if entity_type is not None:
            query += " AND entity_type = ?"
            params.append(entity_type)
        if normalized_name is not None:
            query += " AND normalized_name = ?"
            params.append(normalized_name)
        if entity_ids is not None:
            query += " AND id IN (%s)" % ",".join("?" for _ in entity_ids)
            params.extend(int(value) for value in entity_ids)
        if at_time is not None:
            evaluation_time = self._normalize_timestamp(at_time, "at_time")
            query += " AND (valid_from IS NULL OR valid_from <= ?)"
            query += " AND (valid_until IS NULL OR ? < valid_until)"
            params.extend([evaluation_time, evaluation_time])
        query += " ORDER BY id ASC"
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            query += " LIMIT ?"
            params.append(limit)
        with self._connection() as con:
            rows = con.execute(query, params).fetchall()
        return self._entity_rows_to_records(rows)

    def create_entity(
        self,
        session_id: str,
        scope: str,
        entity_type: str,
        display_name: str,
        normalized_name: str,
        source_ids: list[int],
        status: str = "candidate",
        authority: str = "user",
        confidence: float = 0.3,
        timestamp: Optional[float] = None,
        observed_at: Optional[float] = None,
        merged_into_entity_id: Optional[int] = None,
        merged_at: Optional[float] = None,
        source_episode_ids: Optional[list[str]] = None,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
    ) -> int:
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"unknown entity_type={entity_type!r}")
        if status not in ("candidate", "active"):
            raise ValueError("entities must be created as candidate or active")
        if authority not in ENTITY_AUTHORITIES:
            raise ValueError(f"unknown authority={authority!r}")
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError("display_name must be a non-empty string")
        if not isinstance(normalized_name, str) or not normalized_name:
            raise ValueError("normalized_name must be a non-empty string")
        if normalized_name != normalize_entity_name(display_name):
            raise ValueError("normalized_name does not match deterministic normalization")
        sources = list(dict.fromkeys(int(value) for value in source_ids))
        if status == "active" and authority != "user":
            raise ValueError(
                "only user-authored entities may be active "
                f"(got authority={authority!r})"
            )
        if status == "active" and not sources:
            raise ValueError("active entities require source provenance")
        created_at = (
            self._normalize_timestamp(timestamp, "created_at")
            if timestamp is not None else time.time()
        )
        normalized_merged_at = (
            self._normalize_timestamp(merged_at, "merged_at")
            if merged_at is not None else None
        )
        normalized_from = (
            self._normalize_timestamp(valid_from, "valid_from")
            if valid_from is not None else None
        )
        normalized_until = (
            self._normalize_timestamp(valid_until, "valid_until")
            if valid_until is not None else None
        )
        self._validate_interval(normalized_from, normalized_until)
        with self._connection() as con:
            self._validate_user_source_roles(con, sources, authority, session_id, scope)
            observed = self._resolve_observed_at(
                con, sources, created_at, observed_at
            )
            cur = con.execute(
                "INSERT INTO entities"
                "(session_id, scope, entity_type, display_name,"
                " normalized_name, status, merged_into_entity_id, source_ids,"
                " authority, confidence, created_at, observed_at, merged_at,"
                " source_episode_ids, valid_from, valid_until)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, scope, entity_type, display_name.strip(),
                    normalized_name, status, merged_into_entity_id,
                    ",".join(str(value) for value in sources), authority,
                    confidence, created_at, observed, normalized_merged_at,
                    ",".join(str(value) for value in (source_episode_ids or [])),
                    normalized_from, normalized_until,
                ),
            )
            return int(cur.lastrowid)

    def confirm_entity(
        self, entity_id: int, confidence: float = 1.0
    ) -> Optional[EntityRecord]:
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT status, authority, source_ids FROM entities"
                " WHERE id = ?",
                (entity_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"entity {entity_id} not found")
            if row[0] != "candidate":
                raise ValueError(f"entity {entity_id} is {row[0]}, not candidate")
            if row[1] != "user":
                raise ValueError(f"entity {entity_id} authority is not user")
            if not row[2]:
                raise ValueError(f"entity {entity_id} has no provenance")
            cur = con.execute(
                "UPDATE entities SET status = 'active', confidence = ?"
                " WHERE id = ? AND status = 'candidate'",
                (confidence, entity_id),
            )
            if cur.rowcount != 1:
                raise ValueError("entity changed during confirmation")
            updated = con.execute(
                self._entity_select() + " WHERE id = ?", (entity_id,)
            ).fetchone()
        return self._entity_row_to_record(updated)

    def archive_entity(self, entity_id: int) -> Optional[EntityRecord]:
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT status FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"entity {entity_id} not found")
            if row[0] != "active":
                raise ValueError(f"entity {entity_id} is {row[0]}, not active")
            cur = con.execute(
                "UPDATE entities SET status = 'archived'"
                " WHERE id = ? AND status = 'active'",
                (entity_id,),
            )
            if cur.rowcount != 1:
                raise ValueError("entity changed during archival")
            updated = con.execute(
                self._entity_select() + " WHERE id = ?", (entity_id,)
            ).fetchone()
        return self._entity_row_to_record(updated)

    def merge_entities(
        self,
        losing_entity_id: int,
        canonical_entity_id: int,
        merged_at: float,
    ) -> Optional[EntityRecord]:
        transition = self._normalize_timestamp(merged_at, "merged_at")
        if losing_entity_id == canonical_entity_id:
            raise ValueError("entity cannot be merged into itself")
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            losing = con.execute(
                self._entity_select() + " WHERE id = ?",
                (losing_entity_id,),
            ).fetchone()
            canonical = con.execute(
                self._entity_select() + " WHERE id = ?",
                (canonical_entity_id,),
            ).fetchone()
            if losing is None:
                raise ValueError(f"entity {losing_entity_id} not found")
            if canonical is None:
                raise ValueError(f"entity {canonical_entity_id} not found")
            losing_record = self._entity_row_to_record(losing)
            canonical_record = self._entity_row_to_record(canonical)
            if losing_record.status != "active":
                raise ValueError(
                    f"entity {losing_entity_id} is {losing_record.status},"
                    " not active"
                )
            if canonical_record.status != "active":
                raise ValueError(
                    f"entity {canonical_entity_id} is {canonical_record.status},"
                    " not active"
                )
            if losing_record.scope != canonical_record.scope:
                raise ValueError("entities must share one scope to merge")
            cur = con.execute(
                "UPDATE entities SET status = 'merged',"
                " merged_into_entity_id = ?, merged_at = ?"
                " WHERE id = ? AND status = 'active'",
                (canonical_entity_id, transition, losing_entity_id),
            )
            if cur.rowcount != 1:
                raise ValueError("entity changed during merge")
            updated = con.execute(
                self._entity_select() + " WHERE id = ?",
                (losing_entity_id,),
            ).fetchone()
        return self._entity_row_to_record(updated)

    def _alias_row_to_record(self, row) -> EntityAliasRecord:
        return EntityAliasRecord(
            alias_id=row[0], session_id=row[1], scope=row[2],
            entity_id=row[3], alias_text=row[4], normalized_alias=row[5],
            status=row[6],
            source_ids=[int(value) for value in row[7].split(",") if value],
            authority=row[8], confidence=row[9], created_at=row[10],
            observed_at=row[11], valid_from=row[12], valid_until=row[13],
            superseded_at=row[14],
        )

    def _alias_select(self) -> str:
        return (
            "SELECT id, session_id, scope, entity_id, alias_text,"
            " normalized_alias, status, source_ids, authority, confidence,"
            " created_at, observed_at, valid_from, valid_until, superseded_at"
            " FROM entity_aliases"
        )

    def fetch_entity_alias(
        self, alias_id: int
    ) -> Optional[EntityAliasRecord]:
        with self._connection() as con:
            row = con.execute(
                self._alias_select() + " WHERE id = ?", (alias_id,)
            ).fetchone()
        return self._alias_row_to_record(row) if row is not None else None

    def fetch_entity_aliases(
        self,
        alias_id: Optional[int] = None,
        entity_id: Optional[int] = None,
        scope: Optional[str] = None,
        status: Optional[str] = None,
        normalized_alias: Optional[str] = None,
        at_time: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list[EntityAliasRecord]:
        query = self._alias_select() + " WHERE 1=1"
        params: list[Any] = []
        if alias_id is not None:
            query += " AND id = ?"
            params.append(alias_id)
        if entity_id is not None:
            query += " AND entity_id = ?"
            params.append(entity_id)
        if scope is not None:
            query += " AND scope = ?"
            params.append(scope)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if normalized_alias is not None:
            query += " AND normalized_alias = ?"
            params.append(normalized_alias)
        if at_time is not None:
            evaluation_time = self._normalize_timestamp(at_time, "at_time")
            query += (
                " AND valid_from IS NOT NULL AND valid_from <= ?"
                " AND (valid_until IS NULL OR ? < valid_until)"
            )
            params.extend([evaluation_time, evaluation_time])
        query += " ORDER BY id ASC"
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            query += " LIMIT ?"
            params.append(limit)
        with self._connection() as con:
            rows = con.execute(query, params).fetchall()
        return [self._alias_row_to_record(row) for row in rows]

    def create_entity_alias(
        self,
        entity_id: int,
        session_id: str,
        scope: str,
        alias_text: str,
        normalized_alias: str,
        source_ids: list[int],
        status: str = "candidate",
        authority: str = "user",
        confidence: float = 0.3,
        timestamp: Optional[float] = None,
        observed_at: Optional[float] = None,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
    ) -> int:
        if status not in ("candidate", "current"):
            raise ValueError("aliases must be created as candidate or current")
        if authority not in ENTITY_AUTHORITIES:
            raise ValueError(f"unknown authority={authority!r}")
        if not isinstance(alias_text, str) or not alias_text.strip():
            raise ValueError("alias_text must be a non-empty string")
        if not isinstance(normalized_alias, str) or not normalized_alias:
            raise ValueError("normalized_alias must be a non-empty string")
        if normalized_alias != normalize_entity_name(alias_text):
            raise ValueError("normalized_alias does not match deterministic normalization")
        sources = list(dict.fromkeys(int(value) for value in source_ids))
        created_at = (
            self._normalize_timestamp(timestamp, "created_at")
            if timestamp is not None else time.time()
        )
        normalized_from = (
            self._normalize_timestamp(valid_from, "valid_from")
            if valid_from is not None else None
        )
        normalized_until = (
            self._normalize_timestamp(valid_until, "valid_until")
            if valid_until is not None else None
        )
        with self._connection() as con:
            self._validate_user_source_roles(con, sources, authority, session_id, scope)
            entity = con.execute(
                "SELECT scope, status, source_ids FROM entities"
                " WHERE id = ?",
                (entity_id,),
            ).fetchone()
            if entity is None:
                raise ValueError(f"entity {entity_id} not found")
            if entity[0] != scope:
                raise ValueError("alias scope must match entity scope")
            if entity[1] not in ("candidate", "active"):
                raise ValueError(f"entity {entity_id} is {entity[1]}")
            if status == "current" and authority != "user":
                raise ValueError(
                    "only user-authored aliases may be current "
                    f"(got authority={authority!r})"
                )
            if status == "current" and not sources:
                raise ValueError("current aliases require source provenance")
            if status == "current" and entity[1] != "active":
                raise ValueError("aliases become current only for active entities")
            observed = self._resolve_observed_at(
                con, sources, created_at, observed_at
            )
            effective_from = (
                normalized_from if normalized_from is not None else observed
            )
            self._validate_interval(effective_from, normalized_until)
            cur = con.execute(
                "INSERT INTO entity_aliases"
                "(session_id, scope, entity_id, alias_text, normalized_alias,"
                " status, source_ids, authority, confidence, created_at,"
                " observed_at, valid_from, valid_until, superseded_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, scope, entity_id, alias_text.strip(),
                    normalized_alias, status,
                    ",".join(str(value) for value in sources), authority,
                    confidence, created_at, observed, effective_from,
                    normalized_until, None,
                ),
            )
            return int(cur.lastrowid)

    def confirm_entity_alias(
        self,
        alias_id: int,
        confidence: float = 1.0,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
    ) -> Optional[EntityAliasRecord]:
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                self._alias_select() + " WHERE id = ?", (alias_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"alias {alias_id} not found")
            alias = self._alias_row_to_record(row)
            if alias.status != "candidate":
                raise ValueError(f"alias {alias_id} is {alias.status}")
            if alias.authority != "user":
                raise ValueError(f"alias {alias_id} authority is not user")
            if not alias.source_ids:
                raise ValueError(f"alias {alias_id} has no provenance")
            entity = con.execute(
                "SELECT scope, status FROM entities WHERE id = ?",
                (alias.entity_id,),
            ).fetchone()
            if entity is None:
                raise ValueError(f"entity {alias.entity_id} not found")
            if entity[0] != alias.scope:
                raise ValueError("alias scope must match entity scope")
            if entity[1] != "active":
                raise ValueError("aliases become current only for active entities")
            effective_from = (
                self._normalize_timestamp(valid_from, "valid_from")
                if valid_from is not None else alias.valid_from
            )
            if effective_from is None:
                effective_from = alias.observed_at
            normalized_until = (
                self._normalize_timestamp(valid_until, "valid_until")
                if valid_until is not None else alias.valid_until
            )
            self._validate_interval(effective_from, normalized_until)
            cur = con.execute(
                "UPDATE entity_aliases SET status = 'current', confidence = ?,"
                " valid_from = ?, valid_until = ?, superseded_at = NULL"
                " WHERE id = ? AND status = 'candidate'",
                (confidence, effective_from, normalized_until, alias_id),
            )
            if cur.rowcount != 1:
                raise ValueError("alias changed during confirmation")
            updated = con.execute(
                self._alias_select() + " WHERE id = ?", (alias_id,)
            ).fetchone()
        return self._alias_row_to_record(updated)

    def _relationship_row_to_record(self, row) -> RelationshipRecord:
        return RelationshipRecord(
            relationship_id=row[0], session_id=row[1], scope=row[2],
            relationship_type=row[3], subject_kind=row[4],
            subject_id=row[5], object_kind=row[6], object_id=row[7],
            status=row[8],
            source_ids=[int(value) for value in row[9].split(",") if value],
            authority=row[10], confidence=row[11], created_at=row[12],
            observed_at=row[13], valid_from=row[14], valid_until=row[15],
            supersedes_id=row[16], superseded_at=row[17],
            source_episode_ids=[value for value in (row[18] or "").split(",") if value],
        )

    def _relationship_select(self) -> str:
        return (
            "SELECT id, session_id, scope, relationship_type, subject_kind,"
            " subject_id, object_kind, object_id, status, source_ids,"
            " authority, confidence, created_at, observed_at, valid_from,"
            " valid_until, supersedes_id, superseded_at, source_episode_ids"
            " FROM relationships"
        )

    def _relationship_endpoint(
        self, con, kind: str, entity_id: int
    ) -> Optional[tuple[str, str, str]]:
        if kind == "entity":
            return con.execute(
                "SELECT scope, status, authority FROM entities WHERE id = ?",
                (entity_id,),
            ).fetchone()
        if kind == "memory_state":
            return con.execute(
                "SELECT scope, status, authority FROM states WHERE id = ?",
                (entity_id,),
            ).fetchone()
        raise ValueError(f"unknown relationship endpoint kind={kind!r}")

    def _validate_relationship_endpoints(
        self,
        con,
        scope: str,
        relationship_type: str,
        subject_kind: str,
        subject_id: int,
        object_kind: str,
        object_id: int,
        require_current: bool,
    ) -> None:
        if relationship_type not in RELATIONSHIP_TYPES:
            raise ValueError(f"unknown relationship_type={relationship_type!r}")
        expected = RELATIONSHIP_ENDPOINTS[relationship_type]
        if (subject_kind, object_kind) != expected:
            raise ValueError(
                f"{relationship_type} requires endpoints {expected[0]} ->"
                f" {expected[1]}"
            )
        if subject_kind == object_kind and subject_id == object_id:
            raise ValueError("relationship endpoints must be different")
        for kind, record_id in (
            (subject_kind, subject_id), (object_kind, object_id)
        ):
            endpoint = self._relationship_endpoint(con, kind, record_id)
            if endpoint is None:
                raise ValueError(f"{kind} {record_id} not found")
            if endpoint[0] != scope:
                raise ValueError("relationship scope must match endpoint scope")
            if require_current:
                expected_status = "active" if kind == "entity" else "current"
                if endpoint[1] != expected_status or endpoint[2] != "user":
                    raise ValueError(
                        f"{kind} {record_id} must be current user"
                        f" {expected_status}"
                    )
            elif kind == "entity" and endpoint[1] not in ("candidate", "active"):
                raise ValueError(f"entity {record_id} is {endpoint[1]}")

    def _insert_relationship_row(
        self,
        con,
        session_id: str,
        scope: str,
        relationship_type: str,
        subject_kind: str,
        subject_id: int,
        object_kind: str,
        object_id: int,
        source_ids: list[int],
        status: str,
        authority: str,
        confidence: float,
        created_at: float,
        observed_at: float,
        valid_from: float,
        valid_until: Optional[float],
        supersedes_id: Optional[int],
        superseded_at: Optional[float],
        source_episode_ids: Optional[list[str]] = None,
    ) -> int:
        cur = con.execute(
            "INSERT INTO relationships"
            "(session_id, scope, relationship_type, subject_kind, subject_id,"
            " object_kind, object_id, status, source_ids, source_episode_ids,"
            " authority, confidence,"
            " created_at, observed_at, valid_from, valid_until, supersedes_id,"
            " superseded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,"
            " ?, ?, ?, ?)",
            (
                session_id, scope, relationship_type, subject_kind,
                subject_id, object_kind, object_id, status,
                ",".join(str(value) for value in source_ids),
                ",".join(str(value) for value in (source_episode_ids or [])),
                authority,
                confidence, created_at, observed_at, valid_from, valid_until,
                supersedes_id, superseded_at,
            ),
        )
        return int(cur.lastrowid)

    def fetch_relationship(
        self, relationship_id: int
    ) -> Optional[RelationshipRecord]:
        with self._connection() as con:
            row = con.execute(
                self._relationship_select() + " WHERE id = ?",
                (relationship_id,),
            ).fetchone()
        return self._relationship_row_to_record(row) if row is not None else None

    def fetch_relationships(
        self,
        scope: Optional[str] = None,
        relationship_type: Optional[str] = None,
        status: Optional[str] = None,
        subject_kind: Optional[str] = None,
        subject_id: Optional[int] = None,
        object_kind: Optional[str] = None,
        object_id: Optional[int] = None,
        at_time: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list[RelationshipRecord]:
        query = self._relationship_select() + " WHERE 1=1"
        params: list[Any] = []
        for column, value in (
            ("scope", scope),
            ("relationship_type", relationship_type),
            ("status", status),
            ("subject_kind", subject_kind),
            ("subject_id", subject_id),
            ("object_kind", object_kind),
            ("object_id", object_id),
        ):
            if value is not None:
                query += f" AND {column} = ?"
                params.append(value)
        if at_time is not None:
            evaluation_time = self._normalize_timestamp(at_time, "at_time")
            query += (
                " AND valid_from IS NOT NULL AND valid_from <= ?"
                " AND (valid_until IS NULL OR ? < valid_until)"
            )
            params.extend([evaluation_time, evaluation_time])
        query += " ORDER BY id ASC"
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            query += " LIMIT ?"
            params.append(limit)
        with self._connection() as con:
            rows = con.execute(query, params).fetchall()
        return [self._relationship_row_to_record(row) for row in rows]

    def create_relationship(
        self,
        session_id: str,
        scope: str,
        relationship_type: str,
        subject_kind: str,
        subject_id: int,
        object_kind: str,
        object_id: int,
        source_ids: list[int],
        status: str = "candidate",
        authority: str = "user",
        confidence: float = 0.3,
        timestamp: Optional[float] = None,
        observed_at: Optional[float] = None,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
        source_episode_ids: Optional[list[str]] = None,
    ) -> int:
        if status not in ("candidate", "unresolved"):
            raise ValueError("relationships must be created as candidate or unresolved")
        if status not in RELATIONSHIP_STATUSES:
            raise ValueError(f"unknown relationship status={status!r}")
        if relationship_type not in RELATIONSHIP_TYPES:
            raise ValueError(f"unknown relationship_type={relationship_type!r}")
        if authority not in ENTITY_AUTHORITIES:
            raise ValueError(f"unknown authority={authority!r}")
        if scope != "global" and session_id != scope:
            raise ValueError("session relationship must originate in its scope")
        sources = list(dict.fromkeys(int(value) for value in source_ids))
        created_at = (
            self._normalize_timestamp(timestamp, "created_at")
            if timestamp is not None else time.time()
        )
        normalized_from = (
            self._normalize_timestamp(valid_from, "valid_from")
            if valid_from is not None else None
        )
        normalized_until = (
            self._normalize_timestamp(valid_until, "valid_until")
            if valid_until is not None else None
        )
        with self._connection() as con:
            self._validate_user_source_roles(con, sources, authority, session_id, scope)
            self._validate_relationship_endpoints(
                con, scope, relationship_type, subject_kind, subject_id,
                object_kind, object_id, False,
            )
            observed = self._resolve_observed_at(
                con, sources, created_at, observed_at
            )
            effective_from = normalized_from if normalized_from is not None else observed
            self._validate_interval(effective_from, normalized_until)
            return self._insert_relationship_row(
                con, session_id, scope, relationship_type, subject_kind,
                subject_id, object_kind, object_id, sources, status,
                authority, confidence, created_at, observed, effective_from,
                normalized_until, None, None, source_episode_ids,
            )

    def confirm_relationship(
        self,
        relationship_id: int,
        confidence: float = 1.0,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
        transition_time: Optional[float] = None,
    ) -> Optional[RelationshipRecord]:
        transition = (
            self._normalize_timestamp(transition_time, "transition_time")
            if transition_time is not None else time.time()
        )
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                self._relationship_select() + " WHERE id = ?",
                (relationship_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"relationship {relationship_id} not found")
            candidate = self._relationship_row_to_record(row)
            if candidate.status != "candidate":
                raise ValueError(
                    f"relationship {relationship_id} is {candidate.status},"
                    " not candidate"
                )
            if candidate.authority != "user":
                raise ValueError(
                    f"relationship {relationship_id} authority is not user"
                )
            if not candidate.source_ids:
                raise ValueError(f"relationship {relationship_id} has no provenance")
            self._validate_relationship_endpoints(
                con, candidate.scope, candidate.relationship_type,
                candidate.subject_kind, candidate.subject_id,
                candidate.object_kind, candidate.object_id, True,
            )
            effective_from = (
                self._normalize_timestamp(valid_from, "valid_from")
                if valid_from is not None else candidate.valid_from
            )
            if effective_from is None:
                effective_from = candidate.observed_at
            normalized_until = (
                self._normalize_timestamp(valid_until, "valid_until")
                if valid_until is not None else candidate.valid_until
            )
            self._validate_interval(effective_from, normalized_until)
            current_rows = con.execute(
                self._relationship_select() +
                " WHERE scope = ? AND relationship_type = ? AND subject_id = ?"
                " AND object_id = ? AND status = 'current' ORDER BY id ASC",
                (
                    candidate.scope, candidate.relationship_type,
                    candidate.subject_id, candidate.object_id,
                ),
            ).fetchall()
            if len(current_rows) > 1:
                raise ValueError("multiple current relationships for one key")
            predecessor_id = (
                int(current_rows[0][0]) if current_rows else None
            )
            if predecessor_id is not None:
                predecessor = self._relationship_row_to_record(current_rows[0])
                cur = con.execute(
                    "UPDATE relationships SET status = 'superseded',"
                    " valid_from = CASE WHEN valid_from IS NOT NULL"
                    " AND ? < valid_from THEN NULL ELSE valid_from END,"
                    " valid_until = ?, superseded_at = ?"
                    " WHERE id = ? AND status = 'current'",
                    (
                        effective_from, effective_from, transition,
                        predecessor_id,
                    ),
                )
                if cur.rowcount != 1:
                    raise ValueError("relationship changed during confirmation")
            con.execute(
                "UPDATE relationships SET status = 'rejected'"
                " WHERE scope = ? AND relationship_type = ? AND subject_id = ?"
                " AND object_id = ? AND status = 'unresolved'",
                (
                    candidate.scope, candidate.relationship_type,
                    candidate.subject_id, candidate.object_id,
                ),
            )
            cur = con.execute(
                "UPDATE relationships SET status = 'current', confidence = ?,"
                " valid_from = ?, valid_until = ?, supersedes_id = ?,"
                " superseded_at = NULL WHERE id = ? AND status = 'candidate'",
                (
                    confidence, effective_from, normalized_until,
                    predecessor_id, relationship_id,
                ),
            )
            if cur.rowcount != 1:
                raise ValueError("relationship changed during confirmation")
            updated = con.execute(
                self._relationship_select() + " WHERE id = ?",
                (relationship_id,),
            ).fetchone()
        return self._relationship_row_to_record(updated)

    def reject_relationship(
        self, relationship_id: int
    ) -> Optional[RelationshipRecord]:
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT status FROM relationships WHERE id = ?",
                (relationship_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"relationship {relationship_id} not found")
            if row[0] not in ("candidate", "unresolved"):
                raise ValueError(
                    f"relationship {relationship_id} is {row[0]}, not rejectable"
                )
            cur = con.execute(
                "UPDATE relationships SET status = 'rejected'"
                " WHERE id = ? AND status IN ('candidate', 'unresolved')",
                (relationship_id,),
            )
            if cur.rowcount != 1:
                raise ValueError("relationship changed during rejection")
            updated = con.execute(
                self._relationship_select() + " WHERE id = ?",
                (relationship_id,),
            ).fetchone()
        return self._relationship_row_to_record(updated)

    def mark_relationship_unresolved(
        self, relationship_id: int, transition_time: float
    ) -> Optional[RelationshipRecord]:
        transition = self._normalize_timestamp(
            transition_time, "transition_time"
        )
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT status FROM relationships WHERE id = ?",
                (relationship_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"relationship {relationship_id} not found")
            if row[0] != "current":
                raise ValueError(
                    f"relationship {relationship_id} is {row[0]}, not current"
                )
            cur = con.execute(
                "UPDATE relationships SET status = 'unresolved',"
                " valid_from = CASE WHEN valid_from IS NOT NULL"
                " AND ? < valid_from THEN NULL ELSE valid_from END,"
                " valid_until = ?, superseded_at = NULL"
                " WHERE id = ? AND status = 'current'",
                (transition, transition, relationship_id),
            )
            if cur.rowcount != 1:
                raise ValueError("relationship changed during conflict")
            updated = con.execute(
                self._relationship_select() + " WHERE id = ?",
                (relationship_id,),
            ).fetchone()
        return self._relationship_row_to_record(updated)

    def supersede_relationship(
        self,
        relationship_id: int,
        valid_until: Optional[float],
        transition_time: float,
    ) -> Optional[RelationshipRecord]:
        transition = self._normalize_timestamp(
            transition_time, "transition_time"
        )
        end = (
            self._normalize_timestamp(valid_until, "valid_until")
            if valid_until is not None else transition
        )
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT status FROM relationships WHERE id = ?",
                (relationship_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"relationship {relationship_id} not found")
            if row[0] != "current":
                raise ValueError(
                    f"relationship {relationship_id} is {row[0]}, not current"
                )
            cur = con.execute(
                "UPDATE relationships SET status = 'superseded',"
                " valid_from = CASE WHEN valid_from IS NOT NULL"
                " AND ? < valid_from THEN NULL ELSE valid_from END,"
                " valid_until = ?, superseded_at = ?"
                " WHERE id = ? AND status = 'current'",
                (end, end, transition, relationship_id),
            )
            if cur.rowcount != 1:
                raise ValueError("relationship changed during supersession")
            updated = con.execute(
                self._relationship_select() + " WHERE id = ?",
                (relationship_id,),
            ).fetchone()
        return self._relationship_row_to_record(updated)

    def supersede_states_with_relationships(
        self,
        state_ids: list[int],
        explained_state_ids: Optional[list[int]] = None,
        relationship_type: Optional[str] = None,
        valid_until: Optional[float] = None,
        transition_time: Optional[float] = None,
    ) -> list[RelationshipRecord]:
        ids = list(dict.fromkeys(int(value) for value in state_ids))
        explained_ids = list(dict.fromkeys(
            int(value) for value in (explained_state_ids or [])
        ))
        if not ids:
            return []
        transition = (
            self._normalize_timestamp(transition_time, "transition_time")
            if transition_time is not None else time.time()
        )
        end = (
            self._normalize_timestamp(valid_until, "valid_until")
            if valid_until is not None else transition
        )
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            for state_id in ids:
                row = con.execute(
                    "SELECT status FROM states WHERE id = ?", (state_id,)
                ).fetchone()
                if row is None:
                    raise ValueError(f"state {state_id} not found")
                if row[0] != "current":
                    raise ValueError(
                        f"state {state_id} is {row[0]}, not current"
                    )
                cur = con.execute(
                    "UPDATE states SET status = 'superseded',"
                    " valid_from = CASE WHEN valid_from IS NOT NULL"
                    " AND ? < valid_from THEN NULL ELSE valid_from END,"
                    " valid_until = ?, superseded_at = ?"
                    " WHERE id = ? AND status = 'current'",
                    (end, end, transition, state_id),
                )
                if cur.rowcount != 1:
                    raise ValueError("state changed during supersession")
            relationship_ids: list[int] = []
            if relationship_type is not None and explained_ids:
                subject_marks = ",".join("?" for _ in ids)
                object_marks = ",".join("?" for _ in explained_ids)
                rows = con.execute(
                    "SELECT id FROM relationships"
                    " WHERE relationship_type = ? AND subject_kind = 'memory_state'"
                    f" AND subject_id IN ({subject_marks})"
                    " AND object_kind = 'memory_state'"
                    f" AND object_id IN ({object_marks}) AND status = 'current'",
                    [relationship_type, *ids, *explained_ids],
                ).fetchall()
                relationship_ids = [int(row[0]) for row in rows]
                for relationship_id in relationship_ids:
                    cur = con.execute(
                        "UPDATE relationships SET status = 'superseded',"
                        " valid_from = CASE WHEN valid_from IS NOT NULL"
                        " AND ? < valid_from THEN NULL ELSE valid_from END,"
                        " valid_until = ?, superseded_at = ?"
                        " WHERE id = ? AND status = 'current'",
                        (end, end, transition, relationship_id),
                    )
                    if cur.rowcount != 1:
                        raise ValueError(
                            "relationship changed during supersession"
                        )
            updated_rows = []
            for relationship_id in relationship_ids:
                row = con.execute(
                    self._relationship_select() + " WHERE id = ?",
                    (relationship_id,),
                ).fetchone()
            if row is not None:
                updated_rows.append(self._relationship_row_to_record(row))
        return updated_rows

    # ------------------------------------------------------------------
    # M6 semantic vector index
    #
    # A derived cache of embeddings for structured memory and canonical
    # entities. It stores no source text: only a content hash, so the row
    # can be matched against the live state/entity without duplicating it.
    # Deleting every row is lossless; `SemanticIndex.rebuild()` recreates it.
    # No ANN structure and no external vector database: candidate scoring is
    # an exact float32 dot product, which is sufficient at this scale.
    # ------------------------------------------------------------------
    VECTOR_KINDS = ("state", "entity")

    def upsert_vector(
        self,
        kind: str,
        ref_id: int,
        scope: str,
        session_id: str,
        model_id: str,
        dim: int,
        content_hash: str,
        vector: bytes,
        timestamp: Optional[float] = None,
    ) -> int:
        """Insert or refresh one embedding. Never touches memory truth."""
        if kind not in self.VECTOR_KINDS:
            raise ValueError(f"unknown vector kind={kind!r}")
        created = (
            self._normalize_timestamp(timestamp, "timestamp")
            if timestamp is not None else time.time()
        )
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "DELETE FROM retrieval_vectors"
                " WHERE kind = ? AND ref_id = ? AND model_id = ?",
                (kind, ref_id, model_id),
            )
            cur = con.execute(
                "INSERT INTO retrieval_vectors"
                "(kind, ref_id, scope, session_id, model_id, dim,"
                " content_hash, vector, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (kind, ref_id, scope, session_id, model_id, dim,
                 content_hash, vector, created),
            )
            return int(cur.lastrowid)

    def delete_vectors(
        self,
        kind: Optional[str] = None,
        ref_id: Optional[int] = None,
        model_id: Optional[str] = None,
    ) -> int:
        query = "DELETE FROM retrieval_vectors WHERE 1=1"
        params: list[Any] = []
        for column, value in (
            ("kind", kind), ("ref_id", ref_id), ("model_id", model_id),
        ):
            if value is not None:
                query += f" AND {column} = ?"
                params.append(value)
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            cur = con.execute(query, params)
            return int(cur.rowcount)

    def fetch_vector_rows(
        self,
        kind: Optional[str] = None,
        scope: Optional[str] = None,
        model_id: Optional[str] = None,
        ref_ids: Optional[list[int]] = None,
        limit: Optional[int] = None,
    ) -> list[tuple[str, int, str, str, str, int, str, bytes, float]]:
        if ref_ids is not None and not ref_ids:
            return []
        query = (
            "SELECT kind, ref_id, scope, session_id, model_id, dim,"
            " content_hash, vector, created_at FROM retrieval_vectors"
            " WHERE 1=1"
        )
        params: list[Any] = []
        for column, value in (
            ("kind", kind), ("scope", scope), ("model_id", model_id),
        ):
            if value is not None:
                query += f" AND {column} = ?"
                params.append(value)
        if ref_ids is not None:
            query += " AND ref_id IN (%s)" % ",".join("?" for _ in ref_ids)
            params.extend(int(value) for value in ref_ids)
        query += " ORDER BY vector_id ASC"
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            query += " LIMIT ?"
            params.append(limit)
        with self._connection() as con:
            return con.execute(query, params).fetchall()

    def count_vectors(
        self,
        kind: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> int:
        query = "SELECT count(*) FROM retrieval_vectors WHERE 1=1"
        params: list[Any] = []
        for column, value in (("kind", kind), ("model_id", model_id)):
            if value is not None:
                query += f" AND {column} = ?"
                params.append(value)
        with self._connection() as con:
            return int(con.execute(query, params).fetchone()[0])

    def set_index_meta(self, key: str, value: str) -> None:
        with self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "INSERT INTO retrieval_index_meta(key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )

    def get_index_meta(self, key: str) -> Optional[str]:
        with self._connection() as con:
            row = con.execute(
                "SELECT value FROM retrieval_index_meta WHERE key = ?", (key,)
            ).fetchone()
        return str(row[0]) if row is not None else None


    def fetch_digests(
        self, session_id: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        with self._connection() as con:
            rows = con.execute(
                "SELECT id, session_id, source_ids, seq_start, seq_end,"
                " created_at, digest_type, content FROM digests"
                " WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "digest_id": r[0],
                    "session_id": r[1],
                    "source_ids": [
                        int(x) for x in r[2].split(",") if x
                    ],
                    "seq_start": r[3],
                    "seq_end": r[4],
                    "created_at": r[5],
                    "digest_type": r[6],
                    "content": r[7],
                }
            )
        return out
