"""Bounded relational and associative retrieval owned by Prime (M7).

The graph is the existing SQLite entity/alias/relationship lifecycle.  This
reader performs no writes and never creates an edge from semantic similarity.
Every packet carries the relationship's source ids, scope, lifecycle and a
plain reason for the connection; inverse traversal is marked as inferred so
Aurora cannot present it as a user-stated fact.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Optional

from . import entities
from .models import RelationEvidencePacket, RelationalRetrievalResult


def relational_retrieval_enabled() -> bool:
    value = os.environ.get("PRIMORA_MEMORY_V2_RELATIONAL", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


class RelationalRetriever:
    """Read-only, authority-gated graph traversal with a hard hop limit."""

    def __init__(self, store: Any, *, enabled: Optional[bool] = None):
        self.store = store
        self.enabled = relational_retrieval_enabled() if enabled is None else bool(enabled)

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError("relational retrieval is disabled")

    def _valid_sources(self, relation: Any) -> bool:
        if relation.authority != "user" or not relation.source_ids:
            return False
        messages = self.store.fetch_messages_by_ids(relation.source_ids)
        return len(messages) == len(set(relation.source_ids)) and all(
            message.role == "user" for message in messages
        )

    def _entity(self, entity_id: int, scope: str, at_time: Optional[float]) -> Any:
        item = entities.canonical_entity(self.store, scope, entity_id, at_time)
        if item is None or item.status != "active" or item.authority != "user":
            return None
        return item

    def _state(self, state_id: int, scope: str, at_time: Optional[float]) -> Any:
        rows = self.store.fetch_states(
            scope=scope, state_ids=[state_id], status="current", at_time=at_time
        )
        if not rows:
            return None
        row = rows[0]
        return row if row.authority == "user" and row.source_ids else None

    def _packet_for_edge(
        self, relation: Any, *, current_id: int, direction: str,
        scope: str, hop: int, at_time: Optional[float],
    ) -> tuple[Optional[RelationEvidencePacket], Optional[int]]:
        if relation.scope != scope or relation.status != "current":
            return None, None
        if not self._valid_sources(relation):
            return None, None
        if at_time is not None and not (
            relation.valid_from is not None
            and relation.valid_from <= at_time
            and (relation.valid_until is None or at_time < relation.valid_until)
        ):
            return None, None
        inferred = direction != "out"
        prefix = "user-stated"
        reason = f"{prefix} {relation.relationship_type}"
        if inferred:
            reason += "; inverse traversal of a user-stated edge"
        source_ids = list(dict.fromkeys(relation.source_ids))
        if relation.relationship_type == "part_of":
            subject = self._entity(relation.subject_id, scope, at_time)
            obj = self._entity(relation.object_id, scope, at_time)
            if subject is None or obj is None:
                return None, None
            if direction == "out":
                target = obj
                left, right = subject.display_name, obj.display_name
            else:
                target = subject
                left, right = obj.display_name, subject.display_name
            return RelationEvidencePacket(
                packet_type="entity", scope=scope,
                relationship_id=relation.relationship_id,
                relationship_type=relation.relationship_type,
                subject=left, object=right, source_ids=source_ids,
                source_episode_ids=list(relation.source_episode_ids),
                status=relation.status, authority=relation.authority,
                valid_from=relation.valid_from, valid_until=relation.valid_until,
                reason=reason, inferred=inferred, hop=hop,
                related_id=target.entity_id,
            ), target.entity_id
        if relation.relationship_type == "concerns":
            state = self._state(relation.subject_id, scope, at_time)
            obj = self._entity(relation.object_id, scope, at_time)
            if state is None or obj is None or relation.object_id != current_id:
                return None, None
            return RelationEvidencePacket(
                packet_type="state", scope=scope,
                relationship_id=relation.relationship_id,
                relationship_type=relation.relationship_type,
                subject=f"{state.concept} = {state.value}",
                object=obj.display_name, source_ids=source_ids,
                source_episode_ids=list(relation.source_episode_ids),
                status=relation.status, authority=relation.authority,
                valid_from=relation.valid_from, valid_until=relation.valid_until,
                reason=reason, inferred=False, hop=hop,
                related_id=state.state_id,
            ), None
        # explains edges are traversable only when a state is explicitly
        # requested in a future API; silently omitting them avoids fan-out.
        return None, None

    def retrieve(
        self,
        session_id: str,
        query: str = "",
        *,
        scope: Optional[str] = None,
        entity_id: Optional[int] = None,
        at_time: Optional[float] = None,
        hop_limit: int = 2,
        limit: int = 20,
    ) -> RelationalRetrievalResult:
        self._require_enabled()
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        graph_scope = session_id if scope in (None, "session") else scope
        if graph_scope not in {session_id, "global"}:
            raise ValueError("scope must be global or session_id")
        if isinstance(hop_limit, bool) or not isinstance(hop_limit, int) or not 0 <= hop_limit <= 4:
            raise ValueError("hop_limit must be between 0 and 4")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        diagnostics: list[str] = []
        roots: list[Any] = []
        if entity_id is not None:
            root = self._entity(int(entity_id), graph_scope, at_time)
            if root is not None:
                roots = [root]
            else:
                diagnostics.append("entity_not_found_or_not_user_active")
        else:
            if not isinstance(query, str) or not query.strip():
                return RelationalRetrievalResult(query="", scope=graph_scope, diagnostics=["query_required"])
            roots, ambiguous = entities.resolve(
                self.store, graph_scope, query.strip(), at_time=at_time, limit=5
            )
            if ambiguous:
                diagnostics.append("ambiguous_entity_alias")
                return RelationalRetrievalResult(
                    query=query, scope=graph_scope, diagnostics=diagnostics
                )
            roots = [item for item in roots if item.authority == "user" and item.status == "active"]
            if not roots:
                diagnostics.append("entity_not_found_or_not_user_active")
        if not roots or hop_limit == 0:
            return RelationalRetrievalResult(query=query, scope=graph_scope, diagnostics=diagnostics)

        relations = self.store.fetch_relationships(scope=graph_scope, status="current", limit=500)
        queue = deque((root.entity_id, 0) for root in roots)
        visited = {(root.entity_id, 0) for root in roots}
        packets: list[RelationEvidencePacket] = []
        packet_keys: set[tuple[int, int, bool]] = set()
        while queue and len(packets) < limit:
            current_id, hop = queue.popleft()
            for relation in relations:
                if relation.relationship_type != "part_of" and relation.relationship_type != "concerns":
                    continue
                endpoints: list[tuple[str, bool]] = []
                if relation.relationship_type == "part_of":
                    if relation.subject_id == current_id:
                        endpoints.append(("out", False))
                    if relation.object_id == current_id:
                        endpoints.append(("in", True))
                elif relation.object_id == current_id:
                    endpoints.append(("out", False))
                for direction, _ in endpoints:
                    packet, target_id = self._packet_for_edge(
                        relation, current_id=current_id, direction=direction,
                        scope=graph_scope, hop=hop + 1, at_time=at_time,
                    )
                    if packet is None or packet.hop > hop_limit:
                        continue
                    key = (packet.relationship_id, packet.related_id or -1, packet.inferred)
                    if key in packet_keys:
                        continue
                    packet_keys.add(key)
                    packets.append(packet)
                    if target_id is not None and packet.hop < hop_limit:
                        next_key = (target_id, packet.hop)
                        if next_key not in visited:
                            visited.add(next_key)
                            queue.append(next_key)
                    if len(packets) >= limit:
                        break
                if len(packets) >= limit:
                    break
        truncated = bool(queue) or len(packets) >= limit
        if truncated:
            diagnostics.append("bounded_traversal_limit_reached")
        return RelationalRetrievalResult(
            query=query, scope=graph_scope, packets=packets,
            diagnostics=diagnostics, truncated=truncated,
        )

    def related_entities(self, session_id: str, entity_name: str, **kwargs: Any) -> RelationalRetrievalResult:
        return self.retrieve(session_id, entity_name, **kwargs)


AssociativeRetriever = RelationalRetriever
retrieve_related = RelationalRetriever.retrieve

