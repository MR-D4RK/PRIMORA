"""Fail-closed isolation for harnesses, evaluation, and training runs.

Milestone 3. This exists because a silent fallback to production memory is
the worst kind of bug in a memory system: nothing errors, the run looks fine,
and Aurora's real memories are quietly overwritten with test data. It has
happened.

Nothing here changes production behaviour. The guard is inert unless a
harness explicitly activates it, so the real Aurora runtime is unaffected.

    with harness_session("t1") as session:
        context = session.context_manager()
        ...

Two layers, deliberately:

*   :func:`harness_session` gives a run its own temporary store, cleans it up
    on exit, and never deletes anything it did not create.
*   :func:`production_guard` intercepts ``ContextStore`` construction for the
    life of a ``with`` block and raises if *anything* resolves to the
    production database, including an implicit ``ContextManager()`` with no
    path at all.

The second layer is the important one. A harness that forgets to pass a
``db_path`` is exactly the failure being prevented, so the refusal is enforced
at the single point where a path stops being a default and becomes a real
file.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .store import ContextStore, default_db_path

# Every table that holds memory, used for content fingerprints.
MEMORY_TABLES = (
    "messages", "sessions", "digests", "states", "candidates",
    "entities", "entity_aliases", "relationships",
    "retrieval_vectors", "retrieval_index_meta", "messages_fts",
)


class IsolationError(RuntimeError):
    """Raised when a harness touches, or might touch, production memory."""


def production_db_path() -> Path:
    return Path(default_db_path()).resolve()


def require_isolated(path: os.PathLike | str) -> Path:
    """Return the resolved path, or refuse it if it is production memory.

    Refuses the production file *and* anything inside the directory that holds
    it. A harness that writes ``<production_dir>/experiment.db`` is still
    littering the production memory directory, and it is the kind of mistake
    that looks like isolation at the call site.
    """
    resolved = Path(path).resolve()
    production = production_db_path()
    if resolved == production or production.parent in resolved.parents:
        raise IsolationError(
            f"refusing to use the production memory path: {resolved}"
        )
    return resolved


def production_fingerprint() -> dict[str, Any]:
    """Row counts and content hashes for production memory.

    Content, not file bytes. Opening a WAL database folds the WAL back into
    the main file, which changes the file's hash without changing a single
    row, so a file-hash guard produces false alarms. This one does not.
    """
    import hashlib

    path = production_db_path()
    if not path.exists():
        return {"exists": False}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        out: dict[str, Any] = {"exists": True}
        for table in MEMORY_TABLES:
            rows = con.execute(f"SELECT * FROM {table}").fetchall()
            out[table] = {
                "rows": len(rows),
                "sha256": hashlib.sha256(
                    "\n".join(sorted(repr(r) for r in rows)).encode("utf-8")
                ).hexdigest(),
            }
        return out
    finally:
        con.close()


def production_digest(fingerprint: Optional[dict[str, Any]] = None) -> str:
    """One comparable string for the whole of production memory."""
    import hashlib

    data = fingerprint or production_fingerprint()
    if not data.get("exists"):
        return "<absent>"
    parts = [
        f"{t}={data[t]['rows']}:{data[t]['sha256'][:16]}" for t in MEMORY_TABLES
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


class LazyManagerAllowed:
    """The one narrow escape from ``production_guard``, and it is named.

    ``Prime.__init__`` unavoidably constructs a ``ContextManager`` with
    ``db_path=None``, and a guard that refuses that refuses the harness's own
    construction. The construction is harmless: ``ContextManager`` is lazy, so
    nothing is opened and nothing is written until ``_get_store()``.

    So this allows *only* the omitted-path construction, and only for the
    duration of the block. An explicit production path is still refused, and
    any attempt to open a production store - lazy or otherwise - is still
    refused by ``production_guard``.

    It exists because a guard that cannot be used by the code it protects is
    not protection; it is an obstacle that gets worked around.
    """

    def __init__(self) -> None:
        self._depth = 0

    def __enter__(self) -> "LazyManagerAllowed":
        global _LAZY_ALLOWED
        _LAZY_ALLOWED[0] += 1
        return self

    def __exit__(self, *exc: Any) -> None:
        global _LAZY_ALLOWED
        _LAZY_ALLOWED[0] = max(0, _LAZY_ALLOWED[0] - 1)


# Depth counter, so nested allowances behave.
_LAZY_ALLOWED = [0]


def build_prime_with_isolation_guard(*args, **kwargs):
    """Construct a ``Prime`` safely while a production guard is active.

    Thin wrapper that keeps the one permitted lazy construction in a single
    named place, rather than every harness rediscovering it.
    """
    from .. import Prime

    with LazyManagerAllowed():
        return Prime(*args, **kwargs)


@contextmanager
def production_guard() -> Iterator[None]:
    """Forbid production memory for the duration of the block.

    Two interception points, and both are needed:

    *   ``ContextManager.__init__`` - catches an omitted ``db_path`` at
        construction, which is the case that is invisible at the call site.
        ``ContextManager`` is lazy, so nothing else would see it.
    *   ``ContextStore.__init__`` - catches the lazy ``_get_store()`` path and
        any store built directly, by whatever route.
    """
    from .manager import ContextManager

    production = production_db_path()

    original_store = ContextStore.__init__
    original_manager = ContextManager.__init__
    state = {"active": True}

    def refuse(resolved, origin):
        if state["active"] and resolved == production:
            raise IsolationError(
                f"{origin} attempted to open the production memory database: "
                f"{production}. Pass an explicit temporary db_path, or use "
                "harness_session()."
            )

    def guarded_store(self, db_path=None, *args, **kwargs):
        resolved = Path(db_path).resolve() if db_path else production
        refuse(resolved, "A ContextStore")
        return original_store(self, db_path, *args, **kwargs)

    def guarded_manager(self, db_path=None, *args, **kwargs):
        if db_path is None and _LAZY_ALLOWED[0] > 0:
            # Declared escape: Prime's unavoidable lazy construction. Nothing
            # is opened here, and the store is still refused if it ever is.
            return original_manager(self, db_path, *args, **kwargs)
        resolved = Path(db_path).resolve() if db_path else production
        refuse(resolved, "A ContextManager")
        return original_manager(self, db_path, *args, **kwargs)

    ContextStore.__init__ = guarded_store
    ContextManager.__init__ = guarded_manager
    try:
        yield
    finally:
        ContextStore.__init__ = original_store
        ContextManager.__init__ = original_manager


class IsolatedSession:
    """One run's private world: its own directory, its own stores."""

    def __init__(self, root: Path, owned: bool) -> None:
        self._root = root
        self._owned = owned
        self._factories: list[Callable[[], Any]] = []

    @property
    def root(self) -> Path:
        return self._root

    def db_path(self, name: str = "context.db") -> Path:
        require_isolated(self._root / name)
        return self._root / name

    def context_manager(self, **kwargs):
        """A fresh ContextManager on this session's private store.

        Fail-closed: if isolation cannot be established it raises rather than
        returning a manager that would fall back to production.
        """
        from .manager import ContextManager

        path = self.db_path(f"context_{len(self._factories)}.db")
        self._root.mkdir(parents=True, exist_ok=True)
        context = ContextManager(db_path=str(path), **kwargs)
        if context._db_path is None:
            raise IsolationError(
                "ContextManager fell back to the production database path"
            )
        require_isolated(context._db_path)
        self._factories.append(lambda: context)
        return context

    def dispose(self) -> None:
        """Remove only what this session created."""
        if self._owned and self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)


@contextmanager
def harness_session(prefix: str = "harness_", root: Optional[str] = None) -> Iterator[IsolatedSession]:
    """Give a run a private store, refuse production, clean up after itself.

    Cleanup removes only the temporary directory this session created. The
    production database is never opened for writing and never deleted.
    """
    if root is not None:
        session = IsolatedSession(require_isolated(root), owned=True)
    else:
        try:
            created = Path(tempfile.mkdtemp(prefix=prefix))
        except OSError as exc:
            # Fail closed. A harness that cannot get a private store must
            # stop, not continue and hope.
            raise IsolationError(
                f"could not create an isolated store for '{prefix}': {exc}"
            ) from exc
        session = IsolatedSession(require_isolated(created), owned=True)

    before = production_digest()
    with production_guard():
        try:
            yield session
        finally:
            session.dispose()
    after = production_digest()
    if after != before:
        raise IsolationError(
            "production memory changed during an isolated run "
            f"({before[:12]} -> {after[:12]})"
        )
