"""M5 Stage C: bounded, scope-explicit relationship queries.

Read-only. Nothing here creates, confirms, or mutates an entity, alias, or
relationship; every write path stays in `entities.py` / `relationships.py`.

Bounds and gates implemented here, exactly as the M5 architecture requires:

* one hop only (`MAX_TRAVERSAL_HOPS = 1`) — no recursion, no global walk,
  no shortest path, no connected components;
* a mandatory result limit on every operation, hard-clamped;
* canonical stored direction only, with `contains` derived at read time;
* shared query gates: `authority = user`, accepted lifecycle status, and
  M4 half-open validity `valid_from <= t < valid_until`;
* explicit scope, with global scope opt-in only.
"""

import math
import time
from typing import Any, Optional

from . import entities as entities_mod
from .models import (
    EntityMemoryLink,
    EntityRecord,
    ExplanationPath,
    RelatedEntity,
    RelationshipContext,
    RelationshipRecord,
    StateRecord,
)
from .relationships import RELATIONSHIP_ENDPOINTS, RELATIONSHIP_TYPES

# One hop is the approved ceiling. There is no multi-hop traversal.
MAX_TRAVERSAL_HOPS = 1
DEFAULT_LIMIT = 20
MAX_LIMIT = 50
MAX_CONTEXT_ENTITIES = 12
MAX_CONTEXT_RELATIONSHIPS = 24
MAX_CONTEXT_MEMORIES = 12

TEMPORAL_MODES = ("current", "as_of", "historical")
DIRECTIONS = ("outbound", "inbound", "both")
EXPLANATION_DIRECTIONS = ("inbound", "outbound")

# Accepted lifecycle statuses per temporal mode. `candidate`, `rejected`,
# and `unresolved` are never accepted: a proposal is not an assertion.
_TEMPORAL_STATUSES = {
    "current": ("current",),
    "as_of": ("current", "superseded"),
    "historical": ("current", "superseded"),
}


# ----------------------------------------------------------------------
# Bounds and normalization
# ----------------------------------------------------------------------
def _bounded_limit(limit: Any, maximum: int = MAX_LIMIT) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    return min(limit, maximum)


def _normalized_time(value: Optional[float], name: str) -> float:
    if value is None:
        return time.time()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a normalized numeric timestamp")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _row_id(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _scope_list(scope: str, include_global: bool) -> list[str]:
    if not isinstance(scope, str) or not scope:
        raise ValueError("scope must be a non-empty string")
    if include_global and scope != "global":
        return [scope, "global"]
    return [scope]


def _window(
    temporal: str, at_time: Optional[float]
) -> tuple[tuple[str, ...], Optional[float]]:
    """Return (accepted statuses, validity filter time) for a mode."""
    if temporal not in TEMPORAL_MODES:
        raise ValueError(
            f"temporal must be one of {TEMPORAL_MODES}, got {temporal!r}"
        )
    if temporal == "as_of" and at_time is None:
        raise ValueError("temporal='as_of' requires at_time")
    if temporal == "historical" and at_time is not None:
        raise ValueError("temporal='historical' does not accept at_time")
    if temporal == "historical":
        return _TEMPORAL_STATUSES[temporal], None
    return _TEMPORAL_STATUSES[temporal], _normalized_time(at_time, "at_time")


# ----------------------------------------------------------------------
# Endpoint gates
# ----------------------------------------------------------------------
def _find_state(store: Any, scope: str, state_id: int) -> Optional[StateRecord]:
    rows = store.fetch_states(scope=scope, state_ids=[state_id], limit=1)
    return rows[0] if rows else None


def _state_accepted(state: StateRecord, temporal: str) -> bool:
    if state.authority != "user":
        return False
    if temporal == "current":
        return state.status == "current"
    return state.status in ("current", "superseded")


def _endpoint_usable(
    store: Any, scope: str, kind: str, record_id: int, temporal: str
) -> bool:
    """Current queries need current/active endpoints; history may keep them."""
    if kind == "entity":
        entity = store.fetch_entity(record_id)
        if entity is None or entity.scope != scope:
            return False
        return entity.status == "active" if temporal == "current" else (
            entity.status != "archived"
        )
    state = _find_state(store, scope, record_id)
    if state is None:
        return False
    return _state_accepted(state, temporal)


def _gate(store: Any, row: RelationshipRecord, temporal: str) -> bool:
    if row.authority != "user":
        return False
    if not _endpoint_usable(store, row.scope, row.subject_kind, row.subject_id,
                            temporal):
        return False
    return _endpoint_usable(
        store, row.scope, row.object_kind, row.object_id, temporal
    )


def _fetch_gated(
    store: Any,
    scopes: list[str],
    statuses: tuple[str, ...],
    window: Optional[float],
    temporal: str,
    limit: int,
    **filters: Any,
) -> list[RelationshipRecord]:
    """Bounded row read across explicit scopes, with the shared gates."""
    out: list[RelationshipRecord] = []
    seen: set[int] = set()
    for one_scope in scopes:
        for status in statuses:
            rows = store.fetch_relationships(
                scope=one_scope, status=status, at_time=window,
                limit=limit, **filters,
            )
            for row in rows:
                if row.relationship_id in seen:
                    continue
                if not _gate(store, row, temporal):
                    continue
                seen.add(row.relationship_id)
                out.append(row)
                if len(out) >= limit:
                    return out
    return out


def _target_entity_id(
    store: Any,
    scope: str,
    entity_id: int,
    follow_merge: bool,
    at_time: Optional[float],
) -> Optional[int]:
    """Resolve a caller's entity id, following merges to the canonical row."""
    entity = store.fetch_entity(entity_id)
    if entity is None or entity.scope != scope:
        return None
    if not follow_merge:
        return entity.entity_id
    canonical = entities_mod.canonical_entity(
        store, scope, entity_id, _normalized_time(at_time, "at_time")
    )
    return canonical.entity_id if canonical is not None else None


# ----------------------------------------------------------------------
# One-hop traversal
# ----------------------------------------------------------------------
def entity_relationships(
    store: Any,
    scope: str,
    entity_id: int,
    direction: str = "outbound",
    relationship_type: Optional[str] = None,
    temporal: str = "current",
    at_time: Optional[float] = None,
    limit: int = DEFAULT_LIMIT,
    include_global: bool = False,
    follow_merge: bool = True,
) -> list[RelationshipRecord]:
    """Bounded one-hop traversal of stored relationship rows.

    Exactly one hop. Never chains: the returned rows name `entity_id` as an
    endpoint, they are not the neighbourhood of its neighbours.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    if relationship_type is not None and (
        relationship_type not in RELATIONSHIP_TYPES
    ):
        raise ValueError(f"unknown relationship_type={relationship_type!r}")
    target = _row_id(entity_id, "entity_id")
    bounded = _bounded_limit(limit)
    statuses, window = _window(temporal, at_time)
    resolved = _target_entity_id(store, scope, target, follow_merge, at_time)
    if resolved is None:
        return []
    scopes = _scope_list(scope, include_global)
    rows: list[RelationshipRecord] = []
    if direction in ("outbound", "both"):
        rows = _fetch_gated(
            store, scopes, statuses, window, temporal, bounded,
            subject_kind="entity", subject_id=resolved,
            relationship_type=relationship_type,
        )
    if direction in ("inbound", "both") and len(rows) < bounded:
        rows += _fetch_gated(
            store, scopes, statuses, window, temporal, bounded - len(rows),
            object_kind="entity", object_id=resolved,
            relationship_type=relationship_type,
        )
    return rows


def related_entities(
    store: Any,
    scope: str,
    entity_id: int,
    direction: str = "both",
    relationship_type: Optional[str] = None,
    temporal: str = "current",
    at_time: Optional[float] = None,
    limit: int = DEFAULT_LIMIT,
    include_global: bool = False,
    follow_merge: bool = True,
) -> list[RelatedEntity]:
    """Entities reachable in one hop through entity->entity edges only.

    An inbound `part_of` row reads as the derived inverse `contains`. The
    stored row is never rewritten, duplicated, or re-typed.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    target = _row_id(entity_id, "entity_id")
    resolved = _target_entity_id(store, scope, target, follow_merge, at_time)
    if resolved is None:
        return []
    rows = entity_relationships(
        store, scope, target, direction=direction,
        relationship_type=relationship_type, temporal=temporal,
        at_time=at_time, limit=limit, include_global=include_global,
        follow_merge=follow_merge,
    )
    evaluation = (
        _normalized_time(at_time, "at_time")
        if temporal != "historical" else None
    )
    out: list[RelatedEntity] = []
    for row in rows:
        kinds = RELATIONSHIP_ENDPOINTS[row.relationship_type]
        if (row.subject_kind, row.object_kind) != kinds:
            continue
        if kinds != ("entity", "entity"):
            continue
        if row.object_id == resolved:
            other_id, derived, name = row.subject_id, True, "contains"
        elif row.subject_id == resolved:
            other_id, derived, name = row.object_id, False, "part_of"
        else:
            continue
        if other_id == resolved:
            continue
        entity = store.fetch_entity(other_id)
        if entity is None:
            continue
        out.append(RelatedEntity(
            entity=entity, relationship=row, direction=name, derived=derived,
            evaluation_time=evaluation,
        ))
    return out


def contained_entities(
    store: Any,
    scope: str,
    entity_id: int,
    temporal: str = "current",
    at_time: Optional[float] = None,
    limit: int = DEFAULT_LIMIT,
    include_global: bool = False,
    follow_merge: bool = True,
) -> list[RelatedEntity]:
    """Derived inverse of inbound `part_of`. Read-time only, never stored."""
    return related_entities(
        store, scope, entity_id, direction="inbound",
        relationship_type="part_of", temporal=temporal, at_time=at_time,
        limit=limit, include_global=include_global, follow_merge=follow_merge,
    )


# ----------------------------------------------------------------------
# Entity -> memory
# ----------------------------------------------------------------------
def entity_memories(
    store: Any,
    scope: str,
    entity_id: int,
    temporal: str = "current",
    at_time: Optional[float] = None,
    limit: int = DEFAULT_LIMIT,
    include_global: bool = False,
    follow_merge: bool = True,
) -> list[EntityMemoryLink]:
    """Memory states linked to this entity by stored `concerns` edges only.

    Structural link, never keyword matching.
    """
    target = _row_id(entity_id, "entity_id")
    resolved = _target_entity_id(store, scope, target, follow_merge, at_time)
    if resolved is None:
        return []
    bounded = _bounded_limit(limit)
    statuses, window = _window(temporal, at_time)
    rows = _fetch_gated(
        store, _scope_list(scope, include_global), statuses, window, temporal,
        bounded, object_kind="entity", object_id=resolved,
        relationship_type="concerns",
    )
    evaluation = (
        _normalized_time(at_time, "at_time")
        if temporal != "historical" else None
    )
    out: list[EntityMemoryLink] = []
    for row in rows:
        state = _find_state(store, row.scope, row.subject_id)
        if state is None or not _state_accepted(state, temporal):
            continue
        out.append(EntityMemoryLink(
            state=state, relationship=row, derived=False,
            evaluation_time=evaluation,
        ))
    return out


# ----------------------------------------------------------------------
# Explanation path (fixed two-record shape)
# ----------------------------------------------------------------------
def explanations(
    store: Any,
    scope: str,
    state_id: int,
    direction: str = "inbound",
    temporal: str = "current",
    at_time: Optional[float] = None,
    limit: int = 10,
    include_global: bool = False,
    concern_limit: int = 4,
) -> list[ExplanationPath]:
    """The fixed path `rationale -explains-> decision`.

    `direction="inbound"` answers "why was this decided?" (the given state is
    the decision). `direction="outbound"` answers "what does this rationale
    explain?". A rationale is read only through a stored `explains` row; the
    optional second fixed segment (decision -concerns-> entity) is included
    for context. Neither segment recurses.
    """
    if direction not in EXPLANATION_DIRECTIONS:
        raise ValueError("direction must be 'inbound' or 'outbound'")
    target = _row_id(state_id, "state_id")
    bounded = min(_bounded_limit(limit), 20)
    statuses, window = _window(temporal, at_time)
    scopes = _scope_list(scope, include_global)
    # An entity id and a state id can share a number. Only a real memory
    # state may start an explanation path.
    if not any(_find_state(store, one_scope, target) for one_scope in scopes):
        return []
    filters = (
        {"object_kind": "memory_state", "object_id": target}
        if direction == "inbound"
        else {"subject_kind": "memory_state", "subject_id": target}
    )
    rows = _fetch_gated(
        store, scopes, statuses, window, temporal, bounded,
        relationship_type="explains", **filters,
    )
    evaluation = (
        _normalized_time(at_time, "at_time")
        if temporal != "historical" else None
    )
    paths: list[ExplanationPath] = []
    for row in rows:
        rationale = _find_state(store, row.scope, row.subject_id)
        decision = _find_state(store, row.scope, row.object_id)
        if rationale is None or decision is None:
            continue
        if not _state_accepted(rationale, temporal):
            continue
        if not _state_accepted(decision, temporal):
            continue
        concerned: list[EntityRecord] = []
        concerns: list[RelationshipRecord] = []
        for concern_status in statuses:
            for concern in store.fetch_relationships(
                scope=row.scope, relationship_type="concerns",
                status=concern_status, subject_kind="memory_state",
                subject_id=decision.state_id, at_time=window,
                limit=min(_bounded_limit(concern_limit), MAX_LIMIT),
            ):
                entity = store.fetch_entity(concern.object_id)
                if entity is None or concern.authority != "user":
                    continue
                concerns.append(concern)
                concerned.append(entity)
        paths.append(ExplanationPath(
            rationale_state=rationale, decision_state=decision,
            relationship=row, concerned_entities=concerned,
            concern_relationships=concerns, derived=False,
            evaluation_time=evaluation,
        ))
    return paths


# ----------------------------------------------------------------------
# Selected relationship context
# ----------------------------------------------------------------------
def relationship_context(
    store: Any,
    scope: str,
    entity_ids: Optional[list[int]] = None,
    state_ids: Optional[list[int]] = None,
    relationship_limit: int = MAX_CONTEXT_RELATIONSHIPS,
    entity_limit: int = MAX_CONTEXT_ENTITIES,
    memory_limit: int = MAX_CONTEXT_MEMORIES,
    hop_limit: int = MAX_TRAVERSAL_HOPS,
    temporal: str = "current",
    at_time: Optional[float] = None,
    include_global: bool = False,
) -> RelationshipContext:
    """A small selected set of entities, relationships, and memories.

    Selected input only — an empty selection is not "everything". `hop_limit`
    is clamped to the approved one hop and reported back, so an over-deep
    request can never widen the result.
    """
    diagnostics: list[str] = []
    if isinstance(hop_limit, bool) or not isinstance(hop_limit, int):
        hop_limit = MAX_TRAVERSAL_HOPS
    if hop_limit != MAX_TRAVERSAL_HOPS:
        diagnostics.append(
            f"hop_limit_clamped: requested={hop_limit} "
            f"approved={MAX_TRAVERSAL_HOPS}"
        )
        hop_limit = MAX_TRAVERSAL_HOPS
    try:
        max_relationships = _bounded_limit(
            relationship_limit, MAX_CONTEXT_RELATIONSHIPS
        )
        max_entities = _bounded_limit(entity_limit, MAX_CONTEXT_ENTITIES)
        max_memories = _bounded_limit(memory_limit, MAX_CONTEXT_MEMORIES)
        statuses, window = _window(temporal, at_time)
        scopes = _scope_list(scope, include_global)
        evaluation = (
            _normalized_time(at_time, "at_time")
            if temporal != "historical" else time.time()
        )
    except ValueError as exc:
        return RelationshipContext(
            scope=scope if isinstance(scope, str) else "",
            temporal=temporal if temporal in TEMPORAL_MODES else "current",
            at_time=0.0, hop_limit=MAX_TRAVERSAL_HOPS,
            diagnostics=[f"relationship_query_rejected: {exc}"],
        )

    seeds = _clean_ids(entity_ids)[:max_entities]
    seed_states = _clean_ids(state_ids)[:max_memories]

    entities: list[EntityRecord] = []
    relationships: list[RelationshipRecord] = []
    memories: list[StateRecord] = []
    entity_index: set[int] = set()
    relationship_index: set[int] = set()
    memory_index: set[int] = set()
    derived_count = 0
    truncated = False

    def add_entity(entity: Optional[EntityRecord]) -> None:
        nonlocal truncated
        if entity is None or entity.entity_id in entity_index:
            return
        if len(entities) >= max_entities:
            truncated = True
            return
        entity_index.add(entity.entity_id)
        entities.append(entity)

    def add_relationship(row: RelationshipRecord, derived: bool) -> bool:
        nonlocal truncated, derived_count
        if row.relationship_id in relationship_index:
            return True
        if len(relationships) >= max_relationships:
            truncated = True
            return False
        relationship_index.add(row.relationship_id)
        relationships.append(row)
        if derived:
            derived_count += 1
        return True

    def add_memory(state: Optional[StateRecord]) -> None:
        nonlocal truncated
        if state is None or state.state_id in memory_index:
            return
        if len(memories) >= max_memories:
            truncated = True
            return
        memory_index.add(state.state_id)
        memories.append(state)

    def add_path_relationships(path: ExplanationPath) -> None:
        if add_relationship(path.relationship, False):
            add_memory(path.rationale_state)
            add_memory(path.decision_state)
        for entity in path.concerned_entities:
            add_entity(entity)

    def bounded_rows(rows: list[Any], maximum: int) -> list[Any]:
        """Cap a fetched list, reporting whether material was dropped."""
        nonlocal truncated
        if len(rows) > maximum:
            truncated = True
        return rows[:maximum]

    # Selected seeds only, resolved through the same gates as every read.
    for one_scope in scopes:
        for seed in seeds:
            canonical = entities_mod.canonical_entity(
                store, one_scope, seed, evaluation
            )
            if canonical is not None and canonical.status == "active":
                add_entity(canonical)
        for seed in seed_states:
            state = _find_state(store, one_scope, seed)
            if state is not None and _state_accepted(state, temporal):
                add_memory(state)

    # One hop out from the selected entities, then one hop in. Never a
    # second expansion: the loop walks the seed list, not its own output.
    for seed in seeds:
        for row in bounded_rows(entity_relationships(
            store, scope, seed, direction="outbound", temporal=temporal,
            at_time=at_time, limit=max_relationships + 1,
            include_global=include_global,
        ), max_relationships):
            if not add_relationship(row, False):
                break
            if row.object_kind == "entity":
                add_entity(store.fetch_entity(row.object_id))
        for link in bounded_rows(contained_entities(
            store, scope, seed, temporal=temporal, at_time=at_time,
            limit=max_relationships + 1, include_global=include_global,
        ), max_relationships):
            if not add_relationship(link.relationship, True):
                break
            add_entity(link.entity)
        for link in bounded_rows(entity_memories(
            store, scope, seed, temporal=temporal, at_time=at_time,
            limit=max_memories + 1, include_global=include_global,
        ), max_memories):
            if not add_relationship(link.relationship, False):
                break
            add_memory(link.state)

    for seed in seed_states:
        for one_scope in scopes:
            for path in explanations(
                store, one_scope, seed, direction="outbound",
                temporal=temporal, at_time=at_time,
                limit=min(max_relationships, 8),
            ):
                add_path_relationships(path)
            for path in explanations(
                store, one_scope, seed, direction="inbound",
                temporal=temporal, at_time=at_time,
                limit=min(max_relationships, 8),
            ):
                add_path_relationships(path)
            for row in bounded_rows(_fetch_gated(
                store, [one_scope], statuses, window, temporal,
                max_relationships + 1, relationship_type="concerns",
                subject_kind="memory_state", subject_id=seed,
            ), max_relationships):
                if not add_relationship(row, False):
                    break
                add_entity(store.fetch_entity(row.object_id))
    return RelationshipContext(
        scope=scope, temporal=temporal, at_time=evaluation,
        hop_limit=hop_limit, entities=entities, relationships=relationships,
        memories=memories, derived_relationships=derived_count,
        truncated=truncated, diagnostics=diagnostics,
    )


def _clean_ids(values: Optional[list[Any]]) -> list[int]:
    if not values:
        return []
    return [
        int(value) for value in values
        if isinstance(value, int) and not isinstance(value, bool)
    ]
