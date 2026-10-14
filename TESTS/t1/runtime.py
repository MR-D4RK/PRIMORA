"""Isolated runtime for T1 interrogation.

CURRENT: this module builds a real Aurora on top of the real generation path
(`Aurora.respond`) while guaranteeing that nothing touches production memory.

Isolation strategy, in order of importance:
  1. Every ContextManager and ContextStore is constructed with an explicit
     temporary path. Nothing is ever constructed with the default path.
  2. `Prime.__init__` builds a ContextManager with the default path. Its store
     is lazy, so it is never opened, but the harness replaces the attribute
     anyway so no code path can reach production even by accident.
  3. The Prime provider refuses every tool call, so a stray prime_request
     cannot reach an external service.
  4. Each scenario gets its own database file, so cross-session retrieval
     cannot leak one scenario's memory into another's prompt.
  5. `assert_isolated()` re-checks the resolved paths at run time.

NOT IMPLEMENTED: any write to production, and any model weight change. The
brain is loaded through the existing AuroraBrain with the existing adapter.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import AuroraBrain, Brain
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.isolation import (
    IsolationError,
    production_digest,
    production_guard,
    require_isolated,
)
from PRIMORA.prime.context.store import ContextStore, default_db_path
from PRIMORA.prime.providers import Provider

from TESTS.t1.scenarios import MemorySpec, Scenario, SeedState


PRODUCTION_DB = Path(default_db_path()).resolve()


class T1IsolationError(RuntimeError):
    """Raised if anything points at the production database."""


class T1BlockedProvider(Provider):
    """Refuses every capability call.

    T1 measures conversational grounding, not tool execution. A blocked
    provider makes a stray prime_request produce an honest failure instead
    of reaching a real service, and it keeps the run hermetic.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def invoke(self, capability: str, payload: dict[str, Any]) -> PrimeResult:
        self.calls.append((capability, dict(payload or {})))
        return PrimeResult.failure(
            "t1_harness: tool execution is disabled during interrogation"
        )


def production_db_sha256() -> str:
    if not PRODUCTION_DB.exists():
        return "<absent>"
    return hashlib.sha256(PRODUCTION_DB.read_bytes()).hexdigest()


_PRODUCTION_MEMORY_TABLES = (
    "messages", "sessions", "digests", "states", "candidates",
    "entities", "entity_aliases", "relationships",
    "retrieval_vectors", "retrieval_index_meta", "messages_fts",
)


def production_data_fingerprint() -> dict[str, Any]:
    """Logical content of production memory.

    This, not the file hash, is the correct guard. Opening a WAL database
    read-write and closing it folds the WAL back into the main file, which
    changes the file's bytes without changing a single row. Guarding on the
    file hash therefore produces false alarms; guarding on content does not.

    Returns a per-table row count plus a hash over every memory row.
    """
    if not PRODUCTION_DB.exists():
        return {"exists": False}
    con = sqlite3.connect(f"file:{PRODUCTION_DB}?mode=ro", uri=True)
    try:
        out: dict[str, Any] = {"exists": True}
        for table in _PRODUCTION_MEMORY_TABLES:
            rows = con.execute(f"SELECT * FROM {table}").fetchall()
            digest = hashlib.sha256(
                "\n".join(sorted(repr(r) for r in rows)).encode("utf-8")
            ).hexdigest()
            out[table] = {"rows": len(rows), "sha256": digest}
        return out
    finally:
        con.close()


def production_data_digest(fingerprint: Optional[dict[str, Any]] = None) -> str:
    """One comparable string for the whole production memory content."""
    data = fingerprint if fingerprint is not None else production_data_fingerprint()
    if not data.get("exists"):
        return "<absent>"
    parts = [f"{t}={data[t]['rows']}:{data[t]['sha256'][:16]}"
             for t in _PRODUCTION_MEMORY_TABLES]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def assert_isolated(path: os.PathLike | str) -> Path:
    """Fail loudly if a path resolves to the production database."""
    resolved = Path(path).resolve()
    if resolved == PRODUCTION_DB:
        raise T1IsolationError(
            f"refusing to use the production database: {resolved}"
        )
    return resolved


@dataclass
class TurnResult:
    success: bool
    content: str
    error: Optional[str]
    latency_s: float
    model: str


class T1Runtime:
    """A real Aurora wired to a throwaway database.

    The model is loaded once and reused for every scenario, so a full
    interrogation costs one warmup plus one generation per turn.
    """

    def __init__(
        self,
        workdir: Optional[str] = None,
        hybrid_retrieval: bool = True,
        shadow_retrieval: bool = False,
        max_new_tokens: Optional[int] = None,
        brain: Optional[Brain] = None,
    ) -> None:
        self._owns_workdir = workdir is None
        self._workdir = Path(
            workdir or tempfile.mkdtemp(prefix="t1_runtime_")
        )
        assert_isolated(self._workdir)
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._hybrid_retrieval = hybrid_retrieval

        self.config = AuroraConfig.from_env()
        if max_new_tokens is not None:
            object.__setattr__(self.config, "max_new_tokens", max_new_tokens)

        # A caller may supply its own Brain. This exists so the harness's own
        # tests can exercise the full CLI without loading real weights; the
        # production path passes nothing and gets the real AuroraBrain.
        self._injected_brain = brain

        self.provider = T1BlockedProvider()
        # Prime.__init__ unavoidably builds a ContextManager with db_path=None,
        # and db_path=None means the PRODUCTION database. Its store is lazy so
        # it is never opened, but it must never be reachable either. Repoint it
        # at a throwaway path immediately, before anything can use it.
        self.prime = Prime(self.provider)
        self._prime_scratch_dir = self._workdir / "prime_scratch"
        self._prime_scratch_dir.mkdir(parents=True, exist_ok=True)
        self.prime._context = self._new_context(self._prime_scratch_dir)

        # Process-wide refusal for as long as this runtime lives: the
        # enforcement behind the promise this class makes, and it also catches
        # a production store built anywhere else in the process by a path that
        # forgot to pass a db_path. Installed *after* Prime, because Prime
        # unavoidably constructs a db_path=None manager and the guard would
        # otherwise refuse the harness's own construction. That window is
        # inert: ContextManager's store is lazy and is repointed above before
        # anything can open it.
        self._production_guard = production_guard()
        self._production_guard.__enter__()

        self._brain: Optional[Brain] = None
        self._aurora: Optional[Aurora] = None
        self.warmup_s: Optional[float] = None

    def _new_context(self, directory: Path) -> ContextManager:
        """A ContextManager that cannot resolve to the production path."""
        db_path = assert_isolated(directory / "context.db")
        context = ContextManager(
            db_path=str(db_path),
            hybrid_retrieval=self._hybrid_retrieval,
            shadow_retrieval=False,
        )
        # Defence in depth: ContextManager resolves db_path=None to the
        # production default inside _get_store, so assert the resolved store
        # path too rather than trusting the constructor argument.
        if context._db_path is None:
            raise T1IsolationError(
                "ContextManager fell back to the production database path"
            )
        assert_isolated(context._db_path)
        return context

    # -- lifecycle --------------------------------------------------------

    @property
    def brain(self) -> Brain:
        if self._brain is None:
            raise RuntimeError("runtime not warmed up; call warmup()")
        return self._brain

    def warmup(self, verbose: bool = False) -> Optional[str]:
        if self._brain is not None:
            return None
        if self._injected_brain is not None:
            self._brain = self._injected_brain
        else:
            err = self.config.validate()
            if err:
                raise RuntimeError(f"Aurora config invalid: {err}")
            self._brain = AuroraBrain(config=self.config)
        started = time.perf_counter()
        error = self._brain.warmup()
        self.warmup_s = time.perf_counter() - started
        if error:
            raise RuntimeError(f"brain warmup failed: {error}")
        if verbose:
            print(f"[t1] model ready in {self.warmup_s:.1f}s: "
                  f"{self._brain.get_model_name()}")
        return None

    def model_name(self) -> str:
        return self.brain.get_model_name()

    def close(self) -> None:
        self._brain = None
        try:
            self._production_guard.__exit__(None, None, None)
        except Exception:
            pass
        if self._owns_workdir and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)

    def __enter__(self) -> "T1Runtime":
        self.warmup()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- per-scenario isolation -------------------------------------------

    def new_scenario_dir(self, scenario_id: str) -> Path:
        path = assert_isolated(self._workdir / f"scenario_{scenario_id}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def build_scenario(
        self, scenario: Scenario, keep: bool = False,
    ) -> "ScenarioRuntime":
        """A fresh database, context manager, and Aurora for one scenario.

        The Aurora object is rebuilt per scenario so its context manager is
        the isolated one. The Brain is shared, so this costs nothing: the
        model is not reloaded.

        Binding the context once at warmup and reusing it for every scenario
        was the original design and it was wrong. `Prime.__init__` supplies a
        db_path=None manager, which resolves to production, so a shared
        Aurora wrote every scenario into the real database.
        """
        directory = self.new_scenario_dir(scenario.id)
        context = self._new_context(directory)
        # The store must be built from the context manager's own resolved
        # path. Passing a second filename here silently creates two separate
        # databases, so the seeded memory and the injected context diverge.
        store = ContextStore(str(context._db_path))
        # Prime must also be repointed, in case the orchestrator ever reaches
        # for its context.
        self.prime._context = context
        aurora = Aurora(
            self.prime, brain=self.brain, config=self.config,
            context_manager=context,
        )
        seed_memory(store, scenario.memory, scope=scenario.id)
        return ScenarioRuntime(
            scenario=scenario,
            aurora=aurora,
            context=context,
            store=store,
            db_path=Path(context._db_path),
            keep=keep,
        )


class ScenarioRuntime:
    """One scenario's isolated world."""

    def __init__(
        self,
        scenario: Scenario,
        aurora: Aurora,
        context: ContextManager,
        store: ContextStore,
        db_path: Path,
        keep: bool = False,
    ) -> None:
        self.scenario = scenario
        self.aurora = aurora
        self.context = context
        self.store = store
        self.db_path = assert_isolated(db_path)
        self.keep = keep

    def ask(self, prompt: str, history: tuple[dict[str, str], ...] = ()) -> TurnResult:
        messages = [dict(m) for m in history] + [{"role": "user", "content": prompt}]
        started = time.perf_counter()
        result = self.aurora.respond(messages, session_id=self.scenario.id)
        latency = time.perf_counter() - started
        return TurnResult(
            success=result.success,
            content=result.content or "",
            error=result.error,
            latency_s=latency,
            model=result.model,
        )

    def row_counts(self) -> dict[str, int]:
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            out: dict[str, int] = {}
            for table in ("messages", "sessions", "digests", "states",
                          "candidates", "entities", "retrieval_vectors"):
                out[table] = con.execute(
                    f"SELECT count(*) FROM {table}").fetchone()[0]
            return out
        finally:
            con.close()

    def dispose(self) -> None:
        if not self.keep:
            shutil.rmtree(self.db_path.parent, ignore_errors=True)


# --------------------------------------------------------------------------
# Memory seeding
# --------------------------------------------------------------------------


def seed_memory(
    store: ContextStore, spec: MemorySpec, scope: str,
    session_id: Optional[str] = None,
) -> dict[str, int]:
    """Write a scenario's memory fixture into an isolated store.

    Uses direct inserts rather than the proposal/confirm flow so the fixture
    is exactly what the scenario declares, with no candidate state to
    reconcile and no dependence on formation heuristics.
    """
    session = session_id or scope
    source_ids: list[int] = []
    inserted_ids: dict[tuple[str, str], int] = {}
    next_seq = 0
    for message in spec.messages:
        mid, _new, _conflict = store.append(
            session, next_seq, message.role, message.content,
            timestamp=message.timestamp,
        )
        source_ids.append(mid)
        next_seq += 1

    # Every state needs provenance, so synthesise a source message when the
    # scenario did not supply one. This mirrors how a real state is recorded
    # and keeps source_ids resolvable.
    for state in spec.states:
        if not source_ids:
            mid, _new, _conflict = store.append(
                session, next_seq, "user",
                f"{state.concept} is {state.value}",
                timestamp=state.timestamp,
            )
            source_ids.append(mid)
            next_seq += 1
            provenance = [mid]
        else:
            provenance = list(source_ids)

        _insert_state(store, state, scope, session, provenance,
                      inserted_ids)

    return {"messages": len(spec.messages), "states": len(spec.states),
            "source_messages": len(source_ids)}


def _insert_state(
    store: ContextStore, state: SeedState, scope: str, session: str,
    source_ids: list[int], inserted_ids: dict[str, int],
) -> int:
    effective_scope = state.scope or scope
    # Resolve supersession by concept so a fixture does not depend on
    # autoincrement values it cannot control.
    supersedes_id = state.supersedes_id
    if supersedes_id is None and state.supersedes_concept:
        supersedes_id = inserted_ids.get(
            (effective_scope, state.supersedes_concept)
        )
    with store._connect() as con:
        con.execute("BEGIN IMMEDIATE")
        cur = con.execute(
            "INSERT INTO states"
            "(session_id, scope, concept, value, source_ids, created_at,"
            " status, supersedes_id, authority, memory_type, confidence,"
            " observed_at, valid_from, valid_until, superseded_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'user', ?, 1.0, ?, ?, ?, ?)",
            (
                session, effective_scope, state.concept, state.value,
                ",".join(str(i) for i in source_ids), float(state.timestamp),
                state.status, supersedes_id, state.memory_type,
                float(state.timestamp), float(state.timestamp), None,
                float(state.timestamp) if state.status == "superseded" else None,
            ),
        )
        new_id = int(cur.lastrowid)
    inserted_ids.setdefault((effective_scope, state.concept), new_id)
    return new_id
