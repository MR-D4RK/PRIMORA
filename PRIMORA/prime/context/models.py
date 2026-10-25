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
class ExperienceEvent:
    """An explicitly user-reported experience, with one immutable source.

    ``reported_at`` is the time the message was recorded. ``occurred_at`` is
    optional source-backed relative-date resolution with an explicit
    precision; it is never substituted when the user did not provide one.
    ``verified`` means provenance-verified, not that the external event itself
    has been independently checked.
    """

    event_id: int
    session_id: str
    source_message_id: int
    kind: str
    actor: str
    statement: str
    reported_at: float
    authority: str = "user"
    occurred_at: Optional[float] = None
    occurred_precision: str = "unknown"
    occurred_timezone: str = "UTC"


@dataclass(frozen=True)
class EpisodeRecord:
    """Derived grouping of source-backed event reports, NOT an inferred story.

    Ids are stable for the same evidence, but may change if an older source is
    subsequently imported. "first/last" are report times, not event times.
    """

    episode_id: str
    subject: str
    events: tuple[ExperienceEvent, ...]
    source_message_ids: tuple[int, ...]
    session_ids: tuple[str, ...]
    first_reported_at: float
    last_reported_at: float


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
    memory_type is one of FACT/PREFERENCE/DECISION/INSTRUCTION/RATIONALE,
    or "" for rows written before typing existed (treated as untyped).
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
    memory_type: str = ""
    # Evidential support under the memory rules (NOT truth probability):
    # explicit user statements 1.0, uncertain/assistant-contested 0.3.
    # Never overrides lifecycle status.
    confidence: float = 1.0
    observed_at: Optional[float] = None
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None
    superseded_at: Optional[float] = None


@dataclass(frozen=True)
class ObservationRecord:
    """A source-backed observation owned by Prime.

    Observations are deliberately separate from states: they may describe a
    proposed or unresolved interpretation while retaining the exact user
    evidence that supports it.  ``proof_count`` is derived from linked
    sources and is never a model supplied confidence value.
    """

    observation_id: int
    session_id: str
    scope: str
    concept: str
    normalized_concept: str
    value: str
    observation_type: str
    actor: str
    subject: str
    predicate: str
    source_ids: list[int]
    source_episode_ids: list[str]
    authority: str
    status: str
    confidence: float
    created_at: float
    observed_at: Optional[float] = None
    occurred_from: Optional[float] = None
    occurred_until: Optional[float] = None
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None
    supersedes_id: Optional[int] = None
    superseded_at: Optional[float] = None
    proof_count: int = 0

    @property
    def id(self) -> int:
        return self.observation_id

    @property
    def source_message_ids(self) -> list[int]:
        return list(self.source_ids)


@dataclass(frozen=True)
class ObservationHistoryRecord:
    """Immutable snapshot taken before an observation lifecycle change."""

    history_id: int
    observation_id: int
    snapshot: dict[str, Any]
    reason: str
    changed_at: float
    superseded_by_id: Optional[int] = None

    @property
    def id(self) -> int:
        return self.history_id


@dataclass(frozen=True)
class CandidateRecord:
    """One proposed memory awaiting confirmation.

    A candidate is never truth: it is excluded from normal model context
    until confirmed (which creates a StateRecord). Statuses: "pending"
    (uncertain, held), "rejected" (overtaken by a confirmed state),
    "confirmed" (reserved for future confirmation flows).
    """

    candidate_id: int
    session_id: str
    scope: str
    source_ids: list[int]
    proposed_type: str
    proposed_concept: str
    proposed_value: str
    proposed_rationale: str = ""
    authority: str = "user"
    confidence: float = 0.3
    status: str = "pending"
    created_at: float = 0.0
    expires_at: Optional[float] = None


@dataclass(frozen=True)
class EntityRecord:
    entity_id: int
    session_id: str
    scope: str
    entity_type: str
    display_name: str
    normalized_name: str
    source_ids: list[int]
    authority: str
    confidence: float
    created_at: float
    observed_at: Optional[float]
    status: str = "candidate"
    merged_into_entity_id: Optional[int] = None
    merged_at: Optional[float] = None
    source_episode_ids: list[str] = field(default_factory=list)
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None


@dataclass(frozen=True)
class EntityAliasRecord:
    alias_id: int
    entity_id: int
    session_id: str
    scope: str
    alias_text: str
    normalized_alias: str
    source_ids: list[int]
    authority: str
    confidence: float
    created_at: float
    observed_at: Optional[float]
    valid_from: Optional[float]
    valid_until: Optional[float]
    status: str = "candidate"
    superseded_at: Optional[float] = None


@dataclass(frozen=True)
class RelationshipRecord:
    relationship_id: int
    session_id: str
    scope: str
    relationship_type: str
    subject_kind: str
    subject_id: int
    object_kind: str
    object_id: int
    source_ids: list[int]
    authority: str
    confidence: float
    created_at: float
    observed_at: Optional[float]
    valid_from: Optional[float]
    valid_until: Optional[float]
    status: str = "candidate"
    supersedes_id: Optional[int] = None
    superseded_at: Optional[float] = None
    source_episode_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RelatedEntity:
    """One entity reached from another entity by a single stored hop.

    `direction` is the traversal direction as read, so an inbound
    `part_of` edge reads as the derived "contains". `derived` is True for
    that inverse: the stored row is always the canonical `part_of`
    assertion and is never duplicated.
    """

    entity: EntityRecord
    relationship: RelationshipRecord
    direction: str  # "part_of" | "contains"
    derived: bool = False
    evaluation_time: Optional[float] = None


@dataclass(frozen=True)
class EntityMemoryLink:
    """One memory state linked to an entity by a stored `concerns` edge.

    Structural link only: the state is never matched by keyword.
    """

    state: StateRecord
    relationship: RelationshipRecord
    derived: bool = False
    evaluation_time: Optional[float] = None


@dataclass(frozen=True)
class ExplanationPath:
    """The fixed two-record explanation path: rationale -explains-> decision.

    This is a fixed shape, not a traversal. `concerned_entities` is the
    optional, equally fixed second segment (decision -concerns-> entity)
    used by "why did we decide this?".
    """

    rationale_state: StateRecord
    decision_state: StateRecord
    relationship: RelationshipRecord
    concerned_entities: list[EntityRecord] = field(default_factory=list)
    concern_relationships: list[RelationshipRecord] = field(
        default_factory=list
    )
    derived: bool = False
    evaluation_time: Optional[float] = None


@dataclass(frozen=True)
class RelationshipContext:
    """A small, hard-bounded selection of M5 material for one scope.

    Never a graph dump. `hop_limit` is the effective approved depth after
    clamping; `truncated` reports that a cap removed material.
    """

    scope: str
    temporal: str
    at_time: float
    hop_limit: int
    entities: list[EntityRecord] = field(default_factory=list)
    relationships: list[RelationshipRecord] = field(default_factory=list)
    memories: list[StateRecord] = field(default_factory=list)
    derived_relationships: int = 0
    truncated: bool = False
    diagnostics: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ActiveMemoryItem:
    """One entry of the Active Memory view.

    A computed, read-only projection over persistent structures — never a
    separate stored copy. Only currently-true/useful information appears:
    `current` states (user authority), session metadata, newest digests.
    Superseded, pending, rejected, and unresolved rows are excluded.
    """

    kind: str  # "state" | "session" | "digest"
    memory_type: str
    concept: str
    value: str
    content: str  # rendered line, same shape as injected context lines
    authority: str
    confidence: float
    importance: float
    scope: str
    session_id: str
    source_ids: list[int]
    status: str
    state_id: Optional[int] = None
    created_at: Optional[float] = None
    observed_at: Optional[float] = None
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None
    superseded_at: Optional[float] = None
    supersedes_id: Optional[int] = None


@dataclass(frozen=True)
class ActiveMemory:
    """The Active Memory view for one session at one moment.

    Computed on demand from the store; membership rule: current states
    (newest per scope/concept), current session metadata, newest digests.
    Nothing here is persisted separately.
    """

    session_id: str
    items: list["ActiveMemoryItem"]


@dataclass(frozen=True)
class BudgetConfig:
    """Working-context budget. All values in tokens.

    ``state_reserve`` is how much of the input target the conversation fill
    must leave untouched so the structured ``[state ...]`` facts still have
    room. The fill stops at ``input_target - state_reserve`` (never below
    the mandatory base plus current turn), so states and retrieval get at
    least that much space on a turn where history would otherwise consume
    everything.
    """

    input_target: int = 1000
    input_ceiling: int = 1100
    output_reserve: int = 128
    total_budget: int = 1228
    state_reserve: int = 100

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
            state_reserve=_int("PRIMORA_AURORA_CONTEXT_STATE_RESERVE", 100),
        )


@dataclass(frozen=True)
class RetrievalCandidate:
    """One candidate produced by one retrieval channel.

    A candidate is a *proposal* drawn from a single channel. It never
    carries truth: authority, lifecycle, and temporal eligibility are
    decided by the deterministic gates before a candidate is created, and
    `exclusion_reason` records why a fused candidate was dropped.

    `item_id` is the stable identity used for fusion, formatted as
    `"<kind>:<ref_id>"` over `message`, `state`, and `entity`.
    """

    item_id: str
    kind: str  # "message" | "state" | "entity"
    ref_id: int
    scope: str
    session_id: str
    channel: str  # "lexical" | "semantic" | "entity" | "relationship"
    channel_rank: int
    channel_score: float
    similarity: Optional[float] = None
    relationship_id: Optional[int] = None
    relationship_type: Optional[str] = None
    entity_id: Optional[int] = None
    fused_score: Optional[float] = None
    final_rank: Optional[int] = None
    included: bool = False
    exclusion_reason: str = ""


@dataclass(frozen=True)
class RetrievalTrace:
    """Internal, per-request record of how retrieval decided.

    Never rendered into a prompt and never persisted as memory. Held on
    `BuildResult` so benchmarks and diagnostics can inspect it while Aurora
    sees exactly the same context as before.
    """

    session_id: str
    scope: str
    query: str
    temporal: str
    at_time: float
    candidates: list[RetrievalCandidate] = field(default_factory=list)
    fused_order: list[str] = field(default_factory=list)
    channel_timings_ms: dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
    degraded: bool = False
    diagnostics: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ShadowComparison:
    """What the hybrid path would have returned, beside what lexical returned.

    Produced only while shadow mode is enabled. It is an internal record:
    it does not change the assembled context, Active Memory, Working Memory,
    or any Aurora output.
    """

    session_id: str
    query: str
    lexical_ids: list[int] = field(default_factory=list)
    lexical_ranks: dict[int, int] = field(default_factory=dict)
    hybrid_ids: list[int] = field(default_factory=list)
    hybrid_ranks: dict[int, int] = field(default_factory=dict)
    overlap: list[int] = field(default_factory=list)
    new_only: list[int] = field(default_factory=list)
    old_only: list[int] = field(default_factory=list)
    channel_contributions: dict[str, int] = field(default_factory=dict)
    channel_timings_ms: dict[str, float] = field(default_factory=dict)
    hybrid_latency_ms: float = 0.0
    fused_floor_applied: bool = False
    degraded: bool = False
    error: str = ""


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
