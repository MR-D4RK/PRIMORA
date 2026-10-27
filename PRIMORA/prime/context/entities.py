import math
import re
import time
import unicodedata
from typing import Any, Optional

from .models import EntityAliasRecord, EntityRecord


ENTITY_TYPES = frozenset({"PERSON", "AGENT", "PROJECT", "COMPONENT", "FILE"})
ENTITY_STATUSES = frozenset({"candidate", "active", "merged", "archived"})
ALIAS_STATUSES = frozenset({"candidate", "current", "superseded", "rejected"})
ENTITY_AUTHORITIES = frozenset({"user", "assistant", "system", "tool"})

_PUNCTUATION_TRANSLATION = str.maketrans({
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "−": "-",
    "…": "...",
    "\u00a0": " ",
    "\u2009": " ",
})
_SURROUNDING_PUNCTUATION = "'\"`()[]{}<>"
_TERMINAL_PUNCTUATION = ".,;:!?"


def normalize_entity_name(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    text = unicodedata.normalize("NFKC", raw)
    text = text.translate(_PUNCTUATION_TRANSLATION).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(_SURROUNDING_PUNCTUATION)
    text = text.rstrip(_TERMINAL_PUNCTUATION)
    return text.strip(_SURROUNDING_PUNCTUATION).strip()


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
    display_name: str,
    entity_type: str,
    source_ids: Optional[list[int]] = None,
    authority: str = "user",
    scope: Optional[str] = None,
    timestamp: Optional[float] = None,
    observed_at: Optional[float] = None,
    source_episode_ids: Optional[list[str]] = None,
    valid_from: Optional[float] = None,
    valid_until: Optional[float] = None,
) -> tuple[int, list[str]]:
    scope_value = _scope_key(session_id, scope)
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"unknown entity_type={entity_type!r}")
    if authority not in ENTITY_AUTHORITIES:
        raise ValueError(f"unknown authority={authority!r}")
    if not isinstance(display_name, str) or not display_name.strip():
        raise ValueError("display_name must be a non-empty string")
    normalized = normalize_entity_name(display_name)
    if not normalized:
        raise ValueError("display_name has no resolvable normalized form")
    entity_id = store.create_entity(
        session_id,
        scope_value,
        entity_type,
        display_name.strip(),
        normalized,
        _source_ids(source_ids),
        status="candidate",
        authority=authority,
        confidence=0.3,
        timestamp=timestamp,
        observed_at=observed_at,
        source_episode_ids=source_episode_ids,
        valid_from=valid_from,
        valid_until=valid_until,
    )
    return entity_id, [f"entity_candidate_created: {display_name.strip()}"]


def confirm(store: Any, entity_id: int) -> tuple[EntityRecord, list[str]]:
    entity = store.fetch_entity(entity_id)
    if entity is None:
        raise ValueError(f"entity {entity_id} not found")
    if entity.status != "candidate":
        raise ValueError(f"entity {entity_id} is {entity.status}, not candidate")
    if entity.authority != "user":
        raise ValueError(f"entity {entity_id} authority is not user")
    if not entity.source_ids:
        raise ValueError(f"entity {entity_id} has no provenance")
    confirmed = store.confirm_entity(entity_id, confidence=1.0)
    if confirmed is None:
        raise ValueError(f"entity {entity_id} not found")
    return confirmed, [f"entity_confirmed: {confirmed.display_name}"]


def archive(
    store: Any,
    entity_id: int,
    authority: str = "user",
) -> tuple[EntityRecord, list[str]]:
    if authority != "user":
        raise ValueError("only user authority may archive an entity")
    entity = store.fetch_entity(entity_id)
    if entity is None:
        raise ValueError(f"entity {entity_id} not found")
    if entity.status != "active":
        raise ValueError(f"entity {entity_id} is {entity.status}, not active")
    archived = store.archive_entity(entity_id)
    if archived is None:
        raise ValueError(f"entity {entity_id} not found")
    return archived, [f"entity_archived: {archived.display_name}"]


def merge(
    store: Any,
    losing_entity_id: int,
    canonical_entity_id: int,
    authority: str = "user",
    at_time: Optional[float] = None,
) -> tuple[EntityRecord, list[str]]:
    if authority != "user":
        raise ValueError("only user authority may merge entities")
    transition = _normalized_time(at_time, "at_time")
    merged = store.merge_entities(
        losing_entity_id, canonical_entity_id, transition
    )
    if merged is None:
        raise ValueError(f"entity {losing_entity_id} not found")
    return merged, [
        f"entity_merged: {losing_entity_id} -> {canonical_entity_id}"
    ]


def canonical_entity(
    store: Any,
    scope: str,
    entity_id: int,
    at_time: Optional[float] = None,
    max_depth: int = 16,
) -> Optional[EntityRecord]:
    evaluation_time = _normalized_time(at_time, "at_time")
    current_id = entity_id
    seen: set[int] = set()
    entity: Optional[EntityRecord] = None
    for _ in range(max_depth):
        if current_id in seen:
            break
        seen.add(current_id)
        entity = store.fetch_entity(current_id)
        if entity is None or entity.scope != scope:
            return None
        if entity.status != "merged" or entity.merged_into_entity_id is None:
            return entity
        if entity.merged_at is not None and entity.merged_at > evaluation_time:
            return entity
        current_id = entity.merged_into_entity_id
    return entity


def resolve(
    store: Any,
    scope: str,
    name: str,
    at_time: Optional[float] = None,
    limit: int = 5,
) -> tuple[list[EntityRecord], bool]:
    if not isinstance(scope, str) or not scope:
        return [], False
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    bounded_limit = min(limit, 20)
    normalized = normalize_entity_name(name)
    if not normalized:
        return [], False
    evaluation_time = _normalized_time(at_time, "at_time")
    query_limit = min(bounded_limit * 2, 40)
    candidates = store.fetch_entities(
        scope=scope,
        normalized_name=normalized,
        limit=query_limit,
        at_time=evaluation_time,
    )
    aliases = store.fetch_entity_aliases(
        scope=scope,
        status="current",
        normalized_alias=normalized,
        at_time=evaluation_time,
        limit=query_limit,
    )
    candidate_ids = [entity.entity_id for entity in candidates]
    candidate_ids.extend(alias.entity_id for alias in aliases)
    canonical_ids: list[int] = []
    for candidate_id in candidate_ids:
        canonical = canonical_entity(
            store, scope, candidate_id, evaluation_time
        )
        if canonical is not None and canonical.status == "active":
            if canonical.entity_id not in canonical_ids:
                canonical_ids.append(canonical.entity_id)
    resolved = store.fetch_entities(
        scope=scope,
        status="active",
        entity_ids=canonical_ids,
        limit=bounded_limit + 1,
        at_time=evaluation_time,
    )
    ambiguous = len(resolved) > 1
    return resolved[:bounded_limit], ambiguous


def add_alias_candidate(
    store: Any,
    entity_id: int,
    alias_text: str,
    session_id: str,
    source_ids: Optional[list[int]] = None,
    authority: str = "user",
    scope: Optional[str] = None,
    timestamp: Optional[float] = None,
    observed_at: Optional[float] = None,
    valid_from: Optional[float] = None,
    valid_until: Optional[float] = None,
) -> tuple[int, list[str]]:
    if authority not in ENTITY_AUTHORITIES:
        raise ValueError(f"unknown authority={authority!r}")
    entity = store.fetch_entity(entity_id)
    if entity is None:
        raise ValueError(f"entity {entity_id} not found")
    if entity.status not in ("candidate", "active"):
        raise ValueError(f"entity {entity_id} is {entity.status}")
    scope_value = entity.scope if scope is None else _scope_key(session_id, scope)
    if scope_value != entity.scope:
        raise ValueError("alias scope must match entity scope")
    if entity.scope != "global" and session_id != entity.scope:
        raise ValueError("session alias must originate in the entity session")
    if not isinstance(alias_text, str) or not alias_text.strip():
        raise ValueError("alias_text must be a non-empty string")
    normalized = normalize_entity_name(alias_text)
    if not normalized:
        raise ValueError("alias_text has no resolvable normalized form")
    alias_id = store.create_entity_alias(
        entity.entity_id,
        session_id,
        scope_value,
        alias_text.strip(),
        normalized,
        _source_ids(source_ids),
        status="candidate",
        authority=authority,
        confidence=0.3,
        timestamp=timestamp,
        observed_at=observed_at,
        valid_from=valid_from,
        valid_until=valid_until,
    )
    return alias_id, [f"alias_candidate_created: {alias_text.strip()}"]


def confirm_alias(
    store: Any,
    alias_id: int,
    valid_from: Optional[float] = None,
    valid_until: Optional[float] = None,
) -> tuple[EntityAliasRecord, list[str]]:
    confirmed = store.confirm_entity_alias(
        alias_id,
        confidence=1.0,
        valid_from=valid_from,
        valid_until=valid_until,
    )
    if confirmed is None:
        raise ValueError(f"alias {alias_id} not found")
    return confirmed, [f"alias_confirmed: {confirmed.alias_text}"]
