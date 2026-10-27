"""Evidence-backed observations owned by Prime.

The service is intentionally small and synchronous.  It is an admission
boundary: callers provide source message ids and a proposed interpretation;
the service checks that every source is a user message in the requested
scope, then delegates persistence to :class:`ContextStore`.  It never treats
an assistant statement or a model confidence score as user evidence.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from .models import ObservationHistoryRecord, ObservationRecord


def observations_enabled() -> bool:
    values = (
        os.environ.get("PRIMORA_MEMORY_V2_OBSERVATIONS", "0"),
        os.environ.get("PRIMORA_MEMORY_V2", "0"),
    )
    return any(value.strip().lower() in {"1", "true", "yes", "on"} for value in values)


def _scope(session_id: str, value: Optional[str]) -> str:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be non-empty")
    result = value or session_id
    if result != "global" and result != session_id:
        raise ValueError("observation scope must be global or the source session")
    return result


def _clean(value: Any, name: str, limit: int = 240) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    value = " ".join(value.strip().split())
    if len(value) > limit:
        raise ValueError(f"{name} exceeds {limit} characters")
    return value


@dataclass(frozen=True)
class ObservationProposal:
    """A typed proposal before admission; it contains no database ids."""

    concept: str
    value: str
    source_ids: list[int]
    scope: str = "session"
    observation_type: str = "observation"
    actor: str = "user"
    subject: str = ""
    predicate: str = ""
    source_quotes: dict[int, str] | None = None
    source_episode_ids: list[str] | None = None
    occurred_from: float | None = None
    occurred_until: float | None = None
    valid_from: float | None = None
    valid_until: float | None = None
    status: str = "current"
    confidence: float = 1.0
    authority: str = "user"


class ObservationService:
    """Prime admission and lifecycle API for observations.

    ``enabled`` is explicit so tests and operators can roll out the schema
    without changing the legacy ContextManager path.  Direct callers must
    opt in; the default follows the disabled-by-default feature flag.
    """

    def __init__(self, store: Any, *, enabled: Optional[bool] = None):
        self.store = store
        self.enabled = observations_enabled() if enabled is None else bool(enabled)

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError("evidence-backed observations are disabled")

    def _validate_sources(
        self, session_id: str, scope: str, source_ids: list[int],
        source_quotes: dict[int, str] | None,
    ) -> list[int]:
        ids = list(dict.fromkeys(int(value) for value in source_ids))
        if not ids:
            raise ValueError("observation requires at least one source message")
        for source_id in ids:
            message = self.store.fetch_message(source_id)
            if message is None:
                raise ValueError(f"source message {source_id} does not exist")
            if message.role != "user":
                raise ValueError("observation sources must be user-authored")
            if scope != "global" and message.session_id != scope:
                raise ValueError("observation source crosses scope")
            quote = (source_quotes or {}).get(source_id)
            if quote is not None:
                if not isinstance(quote, str) or len(quote) > 1000:
                    raise ValueError("source quote is invalid")
                if quote and quote.casefold() not in message.content.casefold():
                    raise ValueError("source quote is not supported by source message")
        return ids

    def create(
        self, session_id: str, proposal: ObservationProposal | None = None, **kwargs: Any
    ) -> ObservationRecord:
        self._require_enabled()
        if proposal is None:
            proposal = ObservationProposal(**kwargs)
        scope = _scope(session_id, proposal.scope if proposal.scope != "session" else session_id)
        concept = _clean(proposal.concept, "concept")
        value = _clean(proposal.value, "value", 1000)
        if proposal.actor not in {"user", "derived"}:
            raise ValueError("observation actor is not admitted by Prime")
        if proposal.status not in {
            "current", "candidate", "unresolved", "superseded", "rejected",
        }:
            raise ValueError("invalid observation status")
        # A derived interpretation is useful for review and retrieval
        # diagnostics, but it must never become a current personal fact
        # without an explicit Prime confirmation path.
        if proposal.actor == "derived" and proposal.status == "current":
            raise ValueError("derived observations require review before current status")
        if proposal.authority != "user" and proposal.status == "current":
            raise ValueError("only user authority may be current")
        if proposal.observation_type not in {
            "observation", "preference", "state", "event", "relationship", "derived",
        }:
            raise ValueError("unsupported observation type")
        ids = self._validate_sources(session_id, scope, proposal.source_ids, proposal.source_quotes)
        observation_id = self.store.create_observation(
            session_id, scope, concept, value, ids,
            source_quotes=proposal.source_quotes,
            source_episode_ids=proposal.source_episode_ids,
            observation_type=proposal.observation_type,
            actor=proposal.actor,
            subject=proposal.subject,
            predicate=proposal.predicate,
            authority=proposal.authority, status=proposal.status,
            confidence=proposal.confidence,
            occurred_from=proposal.occurred_from,
            occurred_until=proposal.occurred_until,
            valid_from=proposal.valid_from,
            valid_until=proposal.valid_until,
        )
        record = self.store.fetch_observations(scope=scope)
        for item in reversed(record):
            if item.observation_id == observation_id:
                return item
        raise RuntimeError("observation was inserted but could not be read")

    def add_support(
        self,
        observation_id: int,
        source_ids: list[int],
        *,
        source_quotes: dict[int, str] | None = None,
        source_episode_ids: list[str] | None = None,
    ) -> int:
        """Attach additional immutable evidence without rewriting an observation."""
        self._require_enabled()
        observation = self.store.fetch_observations(
            status=None, limit=None
        )
        match = next((item for item in observation if item.observation_id == int(observation_id)), None)
        if match is None:
            raise ValueError("observation not found")
        if match.status not in {"current", "candidate", "unresolved"}:
            raise ValueError("evidence can only support an active observation")
        return self.store.add_observation_evidence(
            int(observation_id), source_ids,
            source_quotes=source_quotes,
            source_episode_ids=source_episode_ids,
        )

    def read(self, **filters: Any) -> list[ObservationRecord]:
        self._require_enabled()
        return self.store.fetch_observations(**filters)

    def supersede(
        self, observation_id: int, *, reason: str = "superseded",
        replacement_id: int | None = None, transition_time: float | None = None,
        valid_until: float | None = None,
    ) -> ObservationRecord:
        self._require_enabled()
        reason = _clean(reason, "reason", 500)
        return self.store.supersede_observation(
            observation_id, reason=reason, replacement_id=replacement_id,
            transition_time=transition_time, valid_until=valid_until,
        )

    def replace(
        self, session_id: str, previous_id: int, proposal: ObservationProposal,
        *, reason: str = "user correction",
    ) -> ObservationRecord:
        """Admit a new assertion, then retain an immutable prior snapshot."""
        replacement = self.create(session_id, proposal)
        self.supersede(previous_id, reason=reason,
                       replacement_id=replacement.observation_id,
                       transition_time=replacement.observed_at)
        return replacement

    def history(self, observation_id: int) -> list[ObservationHistoryRecord]:
        self._require_enabled()
        return self.store.fetch_observation_history(observation_id)


def create_observation(store: Any, session_id: str, **kwargs: Any) -> ObservationRecord:
    """Convenience wrapper for callers that do not need a service object."""
    return ObservationService(store).create(session_id, **kwargs)


def normalize_concept(value: str) -> str:
    """Public canonical key used by typed retrieval and deduplication."""
    return re.sub(r"\s+", " ", value.strip().casefold())

