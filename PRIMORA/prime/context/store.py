"""SQLite persistence for raw conversation history.

Append-only. Raw rows are never updated or deleted. Thread-safe through
per-operation connections. FTS5 when the runtime supports it, LIKE fallback
otherwise (probed, never assumed).
"""

import hashlib
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from .models import SessionRecord, StateRecord, StoredMessage


def default_db_path() -> Path:
    here = Path(__file__).resolve()
    # PRIMORA/prime/context/store.py -> PRIMORA/ = parents[2]
    root = here.parents[2]
    return root / "data" / "context" / "aurora_context.db"


def fingerprint(session_id: str, seq: int, role: str, content: str) -> str:
    material = f"{session_id}\x00{seq}\x00{role}\x00{content}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


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
    authority TEXT NOT NULL DEFAULT 'user'
);
CREATE INDEX IF NOT EXISTS idx_states_session_concept
    ON states(session_id, concept);
CREATE INDEX IF NOT EXISTS idx_states_scope_concept
    ON states(scope, concept);
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
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.executescript(_SCHEMA)
            # Migrate pre-existing databases that lack newer columns.
            cols = {r[1] for r in con.execute("PRAGMA table_info(sessions)").fetchall()}
            if "previous_titles" not in cols:
                con.execute(
                    "ALTER TABLE sessions ADD COLUMN previous_titles"
                    " TEXT NOT NULL DEFAULT '[]'"
                )
        # Probe FTS5 support; never claim it without verifying.
        try:
            with self._connect() as con:
                con.executescript(_FTS_SCHEMA)
            # Backfill any rows written before the virtual table existed.
            with self._connect() as con:
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
            with self._connect() as con:
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
        with self._connect() as con:
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

    def count(self, session_id: str) -> int:
        with self._connect() as con:
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
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
        return [StoredMessage(*r) for r in rows]

    def fetch_latest(
        self, session_id: str, limit: int
    ) -> list[StoredMessage]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, session_id, seq, role, content, timestamp,"
                " fingerprint, importance FROM messages"
                " WHERE session_id = ? ORDER BY seq DESC, id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [StoredMessage(*r) for r in reversed(rows)]

    def search(
        self, session_id: Optional[str], query: str, limit: int = 10
    ) -> list[StoredMessage]:
        """Full-text search; session_id=None searches all sessions.

        LIKE fallback when no FTS5. Results carry their session_id so
        callers can separate current-session history from cross-session
        history. ORDER BY is deterministic (session, seq).
        """
        if self.fts_available:
            sql = (
                "SELECT m.id, m.session_id, m.seq, m.role, m.content,"
                " m.timestamp, m.fingerprint, m.importance"
                " FROM messages_fts f JOIN messages m ON m.id = f.rowid"
                " WHERE messages_fts MATCH ?"
            )
            params: list[Any] = [query]
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
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [StoredMessage(*r) for r in rows]

    def save_digest(
        self,
        session_id: str,
        source_ids: list[int],
        seq_start: int,
        seq_end: int,
        digest_type: str,
        content: str,
    ) -> int:
        with self._connect() as con:
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
        with self._connect() as con:
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
        with self._connect() as con:
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
        with self._connect() as con:
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

        with self._connect() as con:
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
        with self._connect() as con:
            con.execute(
                "UPDATE sessions SET status = ?, last_activity_at = ?"
                " WHERE session_id = ?",
                (status, time.time(), session_id),
            )

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
    ) -> int:
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO states"
                "(session_id, scope, concept, value, source_ids,"
                " created_at, status, supersedes_id, authority)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, scope, concept, value,
                    ",".join(str(i) for i in source_ids),
                    timestamp if timestamp is not None else time.time(),
                    status, supersedes_id, authority,
                ),
            )
            return cur.lastrowid

    def set_state_status(self, state_id: int, status: str) -> None:
        """Transition a state row (e.g. current -> superseded). Raw kept."""
        with self._connect() as con:
            con.execute(
                "UPDATE states SET status = ? WHERE id = ?",
                (status, state_id),
            )

    def _state_rows_to_records(self, rows) -> list[StateRecord]:
        out = []
        for r in rows:
            out.append(StateRecord(
                state_id=r[0], session_id=r[1], scope=r[2], concept=r[3],
                value=r[4],
                source_ids=[int(x) for x in r[5].split(",") if x],
                created_at=r[6], status=r[7], supersedes_id=r[8],
                authority=r[9],
            ))
        return out

    def fetch_states(
        self,
        session_id: Optional[str] = None,
        scope: Optional[str] = None,
        concept: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[StateRecord]:
        query = (
            "SELECT id, session_id, scope, concept, value, source_ids,"
            " created_at, status, supersedes_id, authority FROM states WHERE 1=1"
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
        query += " ORDER BY id ASC"
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
        return self._state_rows_to_records(rows)

    def fetch_digests(
        self, session_id: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        with self._connect() as con:
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
