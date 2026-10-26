"""Read-only metamemory integrity audit (Memory V2 M8).

The audit is explicit and bounded.  It reports reviewable findings and
proposals; it never changes a state, observation, relationship, source row,
Aurora prompt, identity, policy, or runtime control.  No model call is made.
"""

from __future__ import annotations

import os
import uuid
from collections import defaultdict
from typing import Any, Optional

from .models import AuditFinding, AuditReport


def audit_enabled() -> bool:
    value = os.environ.get("PRIMORA_MEMORY_V2_AUDIT", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _key(value: str) -> str:
    return " ".join(str(value).casefold().split())


class MemoryAuditService:
    """Prime-owned, source-aware integrity and reflection job."""

    def __init__(self, store: Any, *, enabled: Optional[bool] = None):
        self.store = store
        self.enabled = audit_enabled() if enabled is None else bool(enabled)

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError("memory audit is disabled")

    def _finding(
        self, findings: list[AuditFinding], *, kind: str, severity: str,
        record_kind: str, record_id: int, concept: str = "", value: str = "",
        source_ids: Optional[list[int]] = None, message: str,
        recommendation: str = "review",
    ) -> None:
        findings.append(AuditFinding(
            kind=kind, severity=severity, record_kind=record_kind,
            record_id=int(record_id), concept=concept, value=value,
            source_ids=list(source_ids or []), message=message,
            recommendation=recommendation,
        ))

    def _source_gate(
        self, findings: list[AuditFinding], *, record_kind: str, record_id: int,
        source_ids: list[int], scope: str, concept: str, value: str,
    ) -> None:
        if not source_ids:
            self._finding(
                findings, kind="unsupported_claim", severity="error",
                record_kind=record_kind, record_id=record_id,
                concept=concept, value=value,
                message="record has no immutable source message links",
                recommendation="hold or reject until user evidence is linked",
            )
            return
        messages = self.store.fetch_messages_by_ids(source_ids)
        found = {item.message_id: item for item in messages}
        if len(found) != len(set(source_ids)):
            self._finding(
                findings, kind="unsupported_claim", severity="error",
                record_kind=record_kind, record_id=record_id,
                concept=concept, value=value, source_ids=source_ids,
                message="one or more source ids do not exist",
                recommendation="hold for Prime review",
            )
            return
        invalid = [
            item.message_id for item in messages
            if item.role != "user" or (scope != "global" and item.session_id != scope)
        ]
        if invalid:
            self._finding(
                findings, kind="unsupported_claim", severity="error",
                record_kind=record_kind, record_id=record_id,
                concept=concept, value=value, source_ids=source_ids,
                message=f"source authority or scope failed for ids {invalid}",
                recommendation="reject or correct provenance links",
            )

    def run(self, *, scope: Optional[str] = None, limit: int = 1000) -> AuditReport:
        self._require_enabled()
        if scope is not None and (not isinstance(scope, str) or not scope.strip()):
            raise ValueError("scope must be non-empty when supplied")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5000:
            raise ValueError("limit must be between 1 and 5000")
        findings: list[AuditFinding] = []
        scanned = 0

        state_rows = self.store.fetch_states(scope=scope)
        current_states = [
            row for row in state_rows
            if row.status == "current" and (scope is None or row.scope == scope)
        ]
        by_concept: dict[str, list[Any]] = defaultdict(list)
        by_fact: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for row in state_rows[:limit]:
            scanned += 1
            if row.status == "current":
                self._source_gate(
                    findings, record_kind="state", record_id=row.state_id,
                    source_ids=row.source_ids, scope=row.scope,
                    concept=row.concept, value=row.value,
                )
                by_concept[_key(row.concept)].append(row)
                by_fact[(_key(row.concept), _key(row.value))].append(row)
            elif row.status == "superseded":
                self._finding(
                    findings, kind="outdated_observation", severity="info",
                    record_kind="state", record_id=row.state_id,
                    concept=row.concept, value=row.value,
                    source_ids=row.source_ids,
                    message="superseded state is retained for historical recall",
                    recommendation="exclude from current answers; retain provenance",
                )
        for concept, rows in by_concept.items():
            values = {_key(row.value) for row in rows}
            if len(values) > 1:
                for row in rows:
                    self._finding(
                        findings, kind="conflict", severity="warning",
                        record_kind="state", record_id=row.state_id,
                        concept=row.concept, value=row.value,
                        source_ids=row.source_ids,
                        message="multiple current values share one concept",
                        recommendation="review correction/supersession lifecycle",
                    )
        for _, rows in by_fact.items():
            if len(rows) > 1:
                for row in rows[1:]:
                    self._finding(
                        findings, kind="duplicate_fact", severity="warning",
                        record_kind="state", record_id=row.state_id,
                        concept=row.concept, value=row.value,
                        source_ids=row.source_ids,
                        message="duplicate current fact for canonical concept/value",
                        recommendation="merge evidence links or supersede duplicate",
                    )

        observation_rows = self.store.fetch_observations(scope=scope)
        current_observations: dict[str, list[Any]] = defaultdict(list)
        for row in observation_rows[:limit]:
            scanned += 1
            if row.status in {"current", "candidate", "unresolved"}:
                self._source_gate(
                    findings, record_kind="observation", record_id=row.observation_id,
                    source_ids=row.source_ids, scope=row.scope,
                    concept=row.concept, value=row.value,
                )
                if row.observation_type == "derived" and row.proof_count < 2:
                    self._finding(
                        findings, kind="insufficient_evidence", severity="warning",
                        record_kind="observation", record_id=row.observation_id,
                        concept=row.concept, value=row.value,
                        source_ids=row.source_ids,
                        message="derived observation has fewer than two distinct proofs",
                        recommendation="keep candidate pending until corroborated",
                    )
                if row.actor not in {"user", "derived"} or (
                    row.actor == "derived" and row.status == "current"
                ):
                    self._finding(
                        findings, kind="unsupported_claim", severity="error",
                        record_kind="observation", record_id=row.observation_id,
                        concept=row.concept, value=row.value,
                        source_ids=row.source_ids,
                        message="observation actor is not admissible as a current claim",
                        recommendation="hold for Prime review",
                    )
                current_observations[(_key(row.concept), _key(row.value))].append(row)
            elif row.status == "superseded":
                self._finding(
                    findings, kind="outdated_observation", severity="info",
                    record_kind="observation", record_id=row.observation_id,
                    concept=row.concept, value=row.value,
                    source_ids=row.source_ids,
                    message="superseded observation remains available for history",
                    recommendation="exclude from current answers",
                )
        for _, rows in current_observations.items():
            for row in rows[1:]:
                self._finding(
                    findings, kind="duplicate_fact", severity="warning",
                    record_kind="observation", record_id=row.observation_id,
                    concept=row.concept, value=row.value,
                    source_ids=row.source_ids,
                    message="duplicate active observation for canonical concept/value",
                    recommendation="merge support or review duplicate",
                )

        for candidate in self.store.fetch_candidates(scope=scope, status="pending")[:limit]:
            scanned += 1
            self._finding(
                findings, kind="unresolved_candidate", severity="info",
                record_kind="candidate", record_id=candidate.candidate_id,
                concept=candidate.proposed_concept, value=candidate.proposed_value,
                source_ids=candidate.source_ids,
                message="candidate is awaiting explicit Prime confirmation",
                recommendation="review evidence before confirmation",
            )
        for relation in self.store.fetch_relationships(scope=scope, limit=limit):
            scanned += 1
            if relation.status == "current":
                self._source_gate(
                    findings, record_kind="relationship",
                    record_id=relation.relationship_id,
                    source_ids=relation.source_ids, scope=relation.scope,
                    concept=relation.relationship_type, value="relationship",
                )
            if relation.status in {"candidate", "unresolved"}:
                self._finding(
                    findings, kind="unresolved_candidate", severity="info",
                    record_kind="relationship", record_id=relation.relationship_id,
                    source_ids=relation.source_ids,
                    message="relationship is not admitted as current",
                    recommendation="review endpoints and source evidence",
                )
            if relation.status == "current" and not relation.source_ids:
                self._finding(
                    findings, kind="unsupported_claim", severity="error",
                    record_kind="relationship", record_id=relation.relationship_id,
                    message="current relationship has no source ids",
                    recommendation="hold or reject relationship",
                )

        proposals = [
            {
                "kind": item.kind, "record_kind": item.record_kind,
                "record_id": item.record_id, "recommendation": item.recommendation,
                "source_ids": list(item.source_ids),
            }
            for item in findings
            if item.kind in {
                "conflict", "duplicate_fact", "unsupported_claim",
                "insufficient_evidence", "unresolved_candidate",
            }
        ]
        return AuditReport(
            run_id=uuid.uuid4().hex, scanned=scanned, findings=findings,
            proposals=proposals,
            diagnostics=[
                f"scanned={scanned}", f"findings={len(findings)}",
                "read_only=true", "llm_calls=0",
            ],
        )

    audit = run


MetamemoryAudit = MemoryAuditService
run_audit = MemoryAuditService.run

