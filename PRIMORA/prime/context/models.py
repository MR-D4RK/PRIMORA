"""Immutable records for the Context Manager.

No database logic. No scoring logic. Shapes only.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class StoredMessage:
    """One verbatim persisted conversation row."""

    message_id: int
    session_id: str
    seq: int
    role: str
    content: str
    timestamp: float
    fingerprint: str
    importance: float = 0.0


@dataclass(frozen=True)
class ContextItem:
    """A candidate for the working view with its score breakdown."""

    message: StoredMessage
    score: float = 0.0
    relevance: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    redundancy: float = 0.0
    source: str = "recent"


@dataclass(frozen=True)
class ContextView:
    """Assembled bounded working set for one model call."""

    messages: list[dict[str, str]]
    input_tokens: int
    allowed_output: int
    session_id: str
    diagnostics: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DigestRecord:
    """Provenance + content for one extractive digest."""

    digest_id: int
    session_id: str
    source_ids: list[int]
    seq_start: int
    seq_end: int
    created_at: float
    digest_type: str
    content: str


@dataclass(frozen=True)
class SessionRecord:
    """One conversation session. Identity only — history stays in messages.

    `previous_titles` records earlier names in rename order. It is metadata
    provenance, not raw history: renames never fabricate conversation rows.
    """

    session_id: str
    created_at: float
    last_activity_at: float
    title: str = ""
    status: str = "active"
    previous_titles: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StateRecord:
    """One tracked belief with provenance and lifecycle.

    Statuses: "current" (presented as true now), "superseded" (replaced,
    kept as history), "unresolved" (conflicting, no explicit correction).
    """

    state_id: int
    session_id: str
    scope: str
    concept: str
    value: str
    source_ids: list[int]
    created_at: float
    status: str = "current"
    supersedes_id: Optional[int] = None
    authority: str = "user"


@dataclass(frozen=True)
class BudgetConfig:
    """Working-context budget. All values in tokens."""

    input_target: int = 1000
    input_ceiling: int = 1100
    output_reserve: int = 128
    total_budget: int = 1228

    @classmethod
    def from_env(cls) -> "BudgetConfig":
        import os

        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, str(default)))
            except (TypeError, ValueError):
                return default

        return cls(
            input_target=_int("PRIMORA_AURORA_CONTEXT_TARGET", 1000),
            input_ceiling=_int("PRIMORA_AURORA_CONTEXT_CEILING", 1100),
            output_reserve=_int("PRIMORA_AURORA_CONTEXT_OUTPUT_RESERVE", 128),
            total_budget=_int("PRIMORA_AURORA_CONTEXT_TOTAL", 1228),
        )


@dataclass(frozen=True)
class BuildResult:
    """Outcome of one context build, including degradation signals."""

    view: ContextView
    raw_count: int
    selected_count: int
    degraded: bool = False


class MemoryProvider(Protocol):
    """Optional future long-term-memory seam.

    Memory answers "What should Aurora remember?".
    The Context Manager answers "What should Aurora think about right now?".
    V1 ships no implementation; the manager treats a missing provider
    as an empty result set.
    """

    def retrieve(
        self,
        session_id: str,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return relevant memories as message-like dicts."""
        ...  # pragma: no cover - interface only


def noop_memory(
    session_id: str,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Default memory provider: always empty, never fails."""
    return []
