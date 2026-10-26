"""Bounded evidence-backed consolidation owned by Prime (Memory V2 M6).

This module deliberately produces *reviewable proposals*.  It does not run
from ordinary recall, does not backfill a database, and never infers a causal
or emotional relationship from co-occurrence.  The only values it groups are
already source-backed user assertions/events with the same canonical concept
and value.
"""

from __future__ import annotations

import os
import uuid
from collections import defaultdict
from typing import Any, Iterable, Optional

from .models import ConsolidationProposal, ConsolidationReport
from .observation import ObservationProposal, ObservationService, normalize_concept


def consolidation_enabled() -> bool:
    value = os.environ.get("PRIMORA_MEMORY_V2_CONSOLIDATION", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _canonical(value: str) -> str:
    """Normalize a referent without introducing subject-specific aliases."""
    text = normalize_concept(value)
    for prefix in ("preferred ", "preference for "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip()


def _within_time(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        return True
    # A month is a bounded compatibility window, not a claim that events
    # happened together.  Values outside it remain separate proposals.
    return abs(float(left) - float(right)) <= 31 * 86400


class ConsolidationService:
    """Opt-in, bounded M6 consolidation and dry-run/replay service."""

    def __init__(self, store: Any, *, enabled: Optional[bool] = None):
        self.store = store
        self.enabled = consolidation_enabled() if enabled is None else bool(enabled)

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError("memory consolidation is disabled")

    def _sources(self, ids: Iterable[int]) -> tuple[list[int], dict[int, str]]:
        unique = list(dict.fromkeys(int(item) for item in ids))
        messages = self.store.fetch_messages_by_ids(unique)
        found = {message.message_id: message for message in messages}
        valid: list[int] = []
        quotes: dict[int, str] = {}
        for source_id in unique:
            message = found.get(source_id)
            if message is None or message.role != "user":
                continue
            valid.append(source_id)
            # Full source text is an exact, bounded quote.  Prime validates it
            # again at admission, so this cannot create provenance.
            quotes[source_id] = message.content[:1000]
        return valid, quotes

    def _records(self, scope: str, source_filter: Optional[set[int]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        states = self.store.fetch_states(scope=scope, status="current")
        for state in states:
            ids, quotes = self._sources(state.source_ids)
            if source_filter is not None and not source_filter.intersection(ids):
                continue
            if not ids or state.authority != "user":
                continue
            records.append({
                "concept": state.concept, "value": state.value,
                "source_ids": ids, "quotes": quotes,
                "source_episode_ids": [], "observed_at": state.observed_at or state.created_at,
                "occurred_from": None, "occurred_until": None,
                "type": state.memory_type.casefold() or "state",
            })
        observations = self.store.fetch_observations(scope=scope)
        for item in observations:
            ids, quotes = self._sources(item.source_ids)
            if source_filter is not None and not source_filter.intersection(ids):
                continue
            if not ids or item.authority != "user" or item.status not in {"current", "candidate"}:
                continue
            records.append({
                "concept": item.concept, "value": item.value,
                "source_ids": ids, "quotes": quotes,
                "source_episode_ids": item.source_episode_ids,
                "observed_at": item.observed_at or item.created_at,
                "occurred_from": item.occurred_from,
                "occurred_until": item.occurred_until,
                "type": item.observation_type,
            })
        # Experience rows are included only when there is a repeated exact
        # statement.  We never summarize two unrelated activities together.
        for event in self.store.fetch_experience_events(limit=1000):
            if scope != "global" and event.session_id != scope:
                continue
            if source_filter is not None and event.source_message_id not in source_filter:
                continue
            if event.authority != "user" or event.actor != "user":
                continue
            ids, quotes = self._sources([event.source_message_id])
            if not ids:
                continue
            records.append({
                "concept": event.kind, "value": event.statement,
                "source_ids": ids, "quotes": quotes,
                "source_episode_ids": [], "observed_at": event.reported_at,
                "occurred_from": event.occurred_at,
                "occurred_until": event.occurred_at,
                "type": "event",
            })
        return records

    def propose(
        self,
        session_id: str,
        *,
        scope: str = "global",
        source_ids: Optional[Iterable[int]] = None,
        limit: int = 100,
    ) -> tuple[list[ConsolidationProposal], int, int, int, list[str]]:
        self._require_enabled()
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        if scope not in {"global", session_id}:
            raise ValueError("scope must be global or the source session")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        source_filter = None if source_ids is None else {int(item) for item in source_ids}
        records = self._records(scope, source_filter)
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            key = (_canonical(record["concept"]), _canonical(record["value"]), record["type"])
            if not key[0] or not key[1]:
                continue
            # Keep incompatible event dates separate; a repeated preference or
            # state may consolidate across sessions when its value is stable.
            if groups[key] and not any(
                _within_time(record["observed_at"], existing["observed_at"])
                for existing in groups[key]
            ):
                continue
            groups[key].append(record)

        proposals: list[ConsolidationProposal] = []
        skipped = 0
        existing = self.store.fetch_observations(scope=scope)
        for (concept, value, kind), items in groups.items():
            source_union: list[int] = []
            quotes: dict[int, str] = {}
            episode_union: list[str] = []
            observed: list[float] = []
            occurred: list[float] = []
            for item in items:
                for source_id in item["source_ids"]:
                    if source_id not in source_union:
                        source_union.append(source_id)
                        quotes[source_id] = item["quotes"].get(source_id, "")
                for episode_id in item["source_episode_ids"]:
                    if episode_id not in episode_union:
                        episode_union.append(episode_id)
                if item["observed_at"] is not None:
                    observed.append(float(item["observed_at"]))
                for value_at in (item["occurred_from"], item["occurred_until"]):
                    if value_at is not None:
                        occurred.append(float(value_at))
            if len(source_union) < 2:
                # One source remains a verified fact, not a derived
                # consolidation.  Holding it avoids manufactured proof.
                skipped += 1
                continue
            duplicate = any(
                _canonical(item.concept) == concept
                and _canonical(item.value) == value
                and set(item.source_ids).issuperset(source_union)
                and item.observation_type == "derived"
                for item in existing
            )
            if duplicate:
                skipped += 1
                continue
            proposals.append(ConsolidationProposal(
                scope=scope, session_id=session_id, concept=concept,
                value=value, source_ids=source_union, source_quotes=quotes,
                source_episode_ids=episode_union, proof_count=len(source_union),
                observation_type="derived", predicate=kind,
                observed_at=max(observed) if observed else None,
                occurred_from=min(occurred) if occurred else None,
                occurred_until=max(occurred) if occurred else None,
                confidence=1.0, uncertainty="derived from repeated user evidence",
                reason="same canonical referent and value across verified sources",
                status="candidate",
            ))
            if len(proposals) >= limit:
                break
        return proposals, len(records), len(groups), skipped, []

    def run(
        self,
        session_id: str,
        *,
        scope: str = "global",
        source_ids: Optional[Iterable[int]] = None,
        dry_run: bool = True,
        limit: int = 100,
    ) -> ConsolidationReport:
        """Run a bounded proposal/replay pass with explicit progress output."""
        self._require_enabled()
        proposals, scanned, grouped, skipped, diagnostics = self.propose(
            session_id, scope=scope, source_ids=source_ids, limit=limit
        )
        applied: list[int] = []
        if not dry_run:
            service = ObservationService(self.store, enabled=True)
            for proposal in proposals:
                try:
                    row = service.create(session_id, ObservationProposal(
                        proposal.concept, proposal.value, proposal.source_ids,
                        scope=proposal.scope, observation_type="derived",
                        actor="derived", subject=proposal.concept,
                        predicate=proposal.predicate,
                        source_quotes=proposal.source_quotes,
                        source_episode_ids=proposal.source_episode_ids,
                        occurred_from=proposal.occurred_from,
                        occurred_until=proposal.occurred_until,
                        valid_from=proposal.observed_at,
                        status="candidate", confidence=proposal.confidence,
                        authority="user",
                    ))
                    applied.append(row.observation_id)
                except (ValueError, RuntimeError) as exc:
                    diagnostics.append(f"consolidation_skipped:{type(exc).__name__}:{exc}")
        return ConsolidationReport(
            run_id=uuid.uuid4().hex, dry_run=bool(dry_run), scanned=scanned,
            grouped=grouped, proposals=proposals, applied_ids=applied,
            skipped=skipped, diagnostics=[
                f"scanned={scanned}", f"groups={grouped}",
                f"proposals={len(proposals)}", f"applied={len(applied)}",
                *diagnostics,
            ],
        )

    def replay(self, session_id: str, source_ids: Iterable[int], **kwargs: Any) -> ConsolidationReport:
        """Replay only explicitly named evidence; no implicit backfill occurs."""
        return self.run(session_id, source_ids=list(source_ids), **kwargs)


# Small functional aliases keep the service easy to discover from Prime code.
consolidate = ConsolidationService.run

