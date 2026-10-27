"""Prime's single durable-memory admission gateway.

Aurora and formation providers propose meaning. This gateway is the shared
write seam for deterministic extraction, typed formation, learning, and
working-memory promotion. It delegates the existing lifecycle rules rather
than introducing another schema or authority model.
"""
from __future__ import annotations

from typing import Any


class MemoryAdmissionGateway:
    """Prime-owned adapter around candidate and state admission."""

    def __init__(self, store: Any):
        self.store = store

    def create_pending(
        self,
        session_id: str,
        scope: str,
        source_ids: list[int],
        proposed_type: str,
        proposed_concept: str,
        proposed_value: str,
        proposed_rationale: str = "",
        authority: str = "user",
        confidence: float = 0.3,
    ) -> int:
        """Create one reviewable candidate through the trusted store seam."""
        return self.store.create_candidate(
            session_id, scope, list(source_ids), proposed_type,
            proposed_concept, proposed_value, proposed_rationale,
            authority=authority, confidence=confidence, status="pending",
        )

    def confirm_pending(self, session_id: str, candidate_id: int) -> list[str]:
        from . import state as state_mod
        return state_mod.confirm_pending(self.store, session_id, candidate_id)

    def create_state(self, *args: Any, **kwargs: Any) -> int:
        """Admit a validated state while retaining the store's invariants."""
        return self.store.create_state(*args, **kwargs)

    def create_state_successor(self, *args: Any, **kwargs: Any) -> int:
        """Admit a correction/supersession through the same write seam."""
        return self.store.create_state_successor(*args, **kwargs)

    def create_state_with_explains(self, *args: Any, **kwargs: Any) -> tuple[int, int]:
        """Admit a source-backed rationale and its explains edge."""
        return self.store.create_state_with_explains(*args, **kwargs)

    def create_entity(self, *args: Any, **kwargs: Any) -> int:
        return self.store.create_entity(*args, **kwargs)

    def create_entity_alias(self, *args: Any, **kwargs: Any) -> int:
        return self.store.create_entity_alias(*args, **kwargs)

    def create_relationship(self, *args: Any, **kwargs: Any) -> int:
        return self.store.create_relationship(*args, **kwargs)

    def create_observation(self, *args: Any, **kwargs: Any) -> int:
        return self.store.create_observation(*args, **kwargs)

    def admit_user_candidates(
        self, session_id: str, message_id: int, candidates: list[Any]
    ) -> list[str]:
        from . import state as state_mod
        return state_mod.apply_user_candidates(
            self.store, session_id, message_id, candidates,
        )

