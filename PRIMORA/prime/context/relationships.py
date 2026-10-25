import math
import re
import time
from typing import Any, Optional

from . import entities as entities_mod
from .models import RelationshipRecord


RELATIONSHIP_TYPES = frozenset({"part_of", "concerns", "explains"})
RELATIONSHIP_STATUSES = frozenset({
    "candidate", "current", "superseded", "unresolved", "rejected"
})
RELATIONSHIP_ENDPOINTS = {
    "part_of": ("entity", "entity"),
    "concerns": ("memory_state", "entity"),
    "explains": ("memory_state", "memory_state"),
}

_PART_OF_RE = re.compile(
    r"^(?P<subject>.+?)\s+is\s+part\s+of\s+(?P<object>.+?)\.?$",
    re.IGNORECASE,
)


def _normalized_time(value: Optional[float], name: str) -> float:
    if value is None:
        return time.time()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a normalized numeric timestamp")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _scope_key(session_id: str, scope: Optional[str]) -> str:
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    if scope is None or scope == "session":
        return session_id
    if scope == "global":
        return "global"
    raise ValueError("scope must be session or global")


def _source_ids(source_ids: Optional[list[int]]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in (source_ids or [])))


def create_candidate(
    store: Any,
    session_id: str,
    relationship_type: str,
    subject_id: int,
    object_id: int,
    source_ids: Optional[list[int]] = None,
    authority: str = "user",
    scope: Optional[str] = None,
    status: str = "candidate",
    timestamp: Optional[float] = None,
    observed_at: Optional[float] = None,
    valid_from: Optional[float] = None,
    valid_until: Optional[float] = None,
    source_episode_ids: Optional[list[str]] = None,
) -> tuple[int, list[str]]:
    if relationship_type not in RELATIONSHIP_TYPES:
        raise ValueError(f"unknown relationship_type={relationship_type!r}")
    if authority not in entities_mod.ENTITY_AUTHORITIES:
        raise ValueError(f"unknown authority={authority!r}")
    if status not in ("candidate", "unresolved"):
        raise ValueError("relationships must be created as candidate or unresolved")
    scope_value = _scope_key(session_id, scope)
    subject_kind, object_kind = RELATIONSHIP_ENDPOINTS[relationship_type]
    if subject_kind == object_kind and subject_id == object_id:
        raise ValueError("relationship endpoints must be different")
    relationship_id = store.create_relationship(
        session_id,
        scope_value,
        relationship_type,
        subject_kind,
        subject_id,
        object_kind,
        object_id,
        _source_ids(source_ids),
        status=status,
        authority=authority,
        confidence=0.3,
        timestamp=timestamp,
        observed_at=observed_at,
        valid_from=valid_from,
        valid_until=valid_until,
        source_episode_ids=source_episode_ids,
    )
    return relationship_id, [
        f"relationship_candidate_created: {relationship_type}"
    ]


def confirm(
    store: Any,
    relationship_id: int,
    valid_from: Optional[float] = None,
    valid_until: Optional[float] = None,
    at_time: Optional[float] = None,
) -> tuple[RelationshipRecord, list[str]]:
    transition = _normalized_time(at_time, "at_time")
    confirmed = store.confirm_relationship(
        relationship_id,
        confidence=1.0,
        valid_from=valid_from,
        valid_until=valid_until,
        transition_time=transition,
    )
    if confirmed is None:
        raise ValueError(f"relationship {relationship_id} not found")
    return confirmed, [
        f"relationship_confirmed: {confirmed.relationship_type}"
    ]


def reject(
    store: Any, relationship_id: int
) -> tuple[RelationshipRecord, list[str]]:
    rejected = store.reject_relationship(relationship_id)
    if rejected is None:
        raise ValueError(f"relationship {relationship_id} not found")
    return rejected, [
        f"relationship_rejected: {rejected.relationship_type}"
    ]


def mark_unresolved(
    store: Any,
    relationship_id: int,
    at_time: Optional[float] = None,
) -> tuple[RelationshipRecord, list[str]]:
    transition = _normalized_time(at_time, "at_time")
    unresolved = store.mark_relationship_unresolved(
        relationship_id, transition
    )
    if unresolved is None:
        raise ValueError(f"relationship {relationship_id} not found")
    return unresolved, [
        f"relationship_unresolved: {unresolved.relationship_type}"
    ]


def supersede(
    store: Any,
    relationship_id: int,
    valid_until: Optional[float] = None,
    at_time: Optional[float] = None,
) -> tuple[RelationshipRecord, list[str]]:
    transition = _normalized_time(at_time, "at_time")
    superseded = store.supersede_relationship(
        relationship_id, valid_until, transition
    )
    if superseded is None:
        raise ValueError(f"relationship {relationship_id} not found")
    return superseded, [
        f"relationship_superseded: {superseded.relationship_type}"
    ]


def propose_part_of_from_text(
    store: Any,
    session_id: str,
    text: str,
    source_ids: Optional[list[int]] = None,
    scope: Optional[str] = None,
    timestamp: Optional[float] = None,
    observed_at: Optional[float] = None,
) -> tuple[Optional[int], list[str]]:
    scope_value = _scope_key(session_id, scope)
    if not isinstance(text, str):
        return None, ["relationship_skipped_part_of: text is not a string"]
    match = _PART_OF_RE.match(text.strip())
    if match is None:
        return None, ["relationship_skipped_part_of: pattern mismatch"]
    subject_name = entities_mod.normalize_entity_name(match.group("subject"))
    object_name = entities_mod.normalize_entity_name(match.group("object"))
    if not subject_name or not object_name:
        return None, ["relationship_skipped_part_of: empty endpoint name"]
    if subject_name == object_name:
        return None, ["relationship_skipped_part_of: self relationship"]
    subject_rows, subject_ambiguous = entities_mod.resolve(
        store, scope_value, subject_name
    )
    object_rows, object_ambiguous = entities_mod.resolve(
        store, scope_value, object_name
    )
    if subject_ambiguous or len(subject_rows) != 1:
        return None, ["relationship_skipped_part_of: ambiguous subject"]
    if object_ambiguous or len(object_rows) != 1:
        return None, ["relationship_skipped_part_of: ambiguous object"]
    return create_candidate(
        store,
        session_id,
        "part_of",
        subject_rows[0].entity_id,
        object_rows[0].entity_id,
        source_ids=source_ids,
        authority="user",
        scope=scope,
        timestamp=timestamp,
        observed_at=observed_at,
    )
