"""Domain-independent, proposal-only memory formation.

This module is deliberately independent of the legacy slot extractor.  A
provider may describe a previously unseen preference or event, but Prime
validates every field against the immutable user message before any derived
row is written.  A provider failure returns diagnostics and leaves the
existing deterministic extractor as the only admission path.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from PRIMORA.prime.contract import RequestStatus

MEMORY_KINDS = frozenset({"preference", "state", "event", "relationship", "observation"})
CLAIM_KINDS = frozenset({
    "assertion", "hypothesis", "question", "fiction", "negation",
    "third_party", "assistant",
})
ADMITTABLE_CLAIMS = frozenset({"assertion"})
MAX_CANDIDATES = 8
MAX_TEXT = 1000


@dataclass(frozen=True)
class TypedMemoryCandidate:
    kind: str
    concept: str = ""
    value: str = ""
    claim_kind: str = "assertion"
    actor: str = "user"
    subject: str = ""
    predicate: str = ""
    scope: str = "session"
    source_ids: list[int] = field(default_factory=list)
    source_quote: str = ""
    confidence: float = 1.0
    occurred_from: Optional[float] = None
    occurred_until: Optional[float] = None
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None
    rationale: str = ""


@dataclass(frozen=True)
class FormationResult:
    candidates: list[TypedMemoryCandidate] = field(default_factory=list)
    admitted_ids: list[int] = field(default_factory=list)
    held_ids: list[int] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    model_calls: int = 0


def _string(value: Any, limit: int = MAX_TEXT) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())[:limit]


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_candidates(payload: Any) -> list[TypedMemoryCandidate]:
    """Parse strict JSON-like provider output without accepting prose."""
    if isinstance(payload, dict):
        data = payload.get("candidates", payload.get("proposals", []))
    elif isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            return []
        if isinstance(data, dict):
            data = data.get("candidates", data.get("proposals", []))
    else:
        return []
    if not isinstance(data, list):
        return []
    out: list[TypedMemoryCandidate] = []
    for item in data[:MAX_CANDIDATES]:
        if not isinstance(item, dict):
            continue
        out.append(TypedMemoryCandidate(
            kind=_string(item.get("kind") or item.get("memory_kind"), 40).casefold(),
            concept=_string(item.get("concept"), 240),
            value=_string(item.get("value"), MAX_TEXT),
            claim_kind=_string(item.get("claim_kind") or item.get("claim"), 40).casefold() or "assertion",
            actor=_string(item.get("actor"), 40).casefold() or "user",
            subject=_string(item.get("subject"), 240),
            predicate=_string(item.get("predicate"), 120).casefold(),
            scope=_string(item.get("scope"), 40).casefold() or "session",
            source_ids=[int(v) for v in item.get("source_ids", [])
                       if isinstance(v, int) and not isinstance(v, bool)],
            source_quote=_string(item.get("source_quote"), 1000),
            confidence=max(0.0, min(1.0, _number(item.get("confidence")) or 0.0)),
            occurred_from=_number(item.get("occurred_from")),
            occurred_until=_number(item.get("occurred_until")),
            valid_from=_number(item.get("valid_from")),
            valid_until=_number(item.get("valid_until")),
            rationale=_string(item.get("rationale"), 500),
        ))
    return out


def validate_candidate(candidate: TypedMemoryCandidate, *, session_id: str,
                       message: Any, source_lookup: Any) -> tuple[bool, str]:
    """Return whether a proposal is safe for Prime admission."""
    if candidate.kind not in MEMORY_KINDS:
        return False, "unsupported_kind"
    if candidate.claim_kind not in CLAIM_KINDS:
        return False, "unsupported_claim_kind"
    if candidate.claim_kind not in ADMITTABLE_CLAIMS:
        return False, f"non_assertion:{candidate.claim_kind}"
    if candidate.actor != "user":
        return False, "authority_actor_not_user"
    if candidate.scope not in {"session", "global"}:
        return False, "invalid_scope"
    if not candidate.concept or not candidate.value:
        return False, "missing_concept_or_value"
    if len(candidate.concept) > 240 or len(candidate.value) > MAX_TEXT:
        return False, "proposal_too_long"
    if not candidate.source_ids or any(i <= 0 for i in candidate.source_ids):
        return False, "missing_source_ids"
    message_ids = set(candidate.source_ids)
    sources = []
    for source_id in message_ids:
        source = source_lookup(source_id)
        if source is None or source.role != "user":
            return False, "source_not_user"
        if candidate.scope == "session" and source.session_id != session_id:
            return False, "cross_scope_source"
        sources.append(source)
    if not candidate.source_quote or not any(
        candidate.source_quote.casefold() in source.content.casefold()
        for source in sources
    ):
        return False, "source_quote_not_supported"
    for start, end in ((candidate.occurred_from, candidate.occurred_until),
                       (candidate.valid_from, candidate.valid_until)):
        if start is not None and (not math.isfinite(start)):
            return False, "invalid_time"
        if end is not None and (not math.isfinite(end)):
            return False, "invalid_time"
        if start is not None and end is not None and end < start:
            return False, "invalid_interval"
    if candidate.kind in {"relationship", "observation"} and not candidate.predicate:
        return False, "missing_predicate"
    return True, "ok"


def provider_proposals(provider: Any, message: Any, session_id: str) -> tuple[list[TypedMemoryCandidate], str]:
    """Make one bounded provider call and parse its strict response."""
    if provider is None:
        return [], "provider_not_configured"
    payload = {
        "messages": [
            {"role": "system", "content": (
                "Extract only explicit user-authored memory proposals. Return JSON only: "
                '{"candidates":[{"kind":"preference|state|event|relationship|observation",'
                '"concept":"...","value":"...","claim_kind":"assertion|hypothesis|question|fiction|negation|third_party|assistant",'
                '"actor":"user","scope":"session|global","source_ids":[ID],"source_quote":"exact substring",'
                '"predicate":"..."}]} . Never infer from an assistant message. Questions, fiction, negation, and hypotheses are not assertions.'
            )},
            {"role": "user", "content": message.content},
        ],
        "session_id": session_id,
        "source_message_id": message.message_id,
        "options": {"temperature": 0, "max_tokens": 600},
    }
    try:
        result = provider.invoke("memory_formation", payload)
    except Exception as exc:
        return [], f"provider_error:{exc}"
    if getattr(result, "status", None) != RequestStatus.SUCCESS:
        return [], f"provider_failure:{getattr(result, 'error', 'unknown')}"
    output = getattr(result, "output", None)
    if isinstance(output, dict) and "content" in output:
        output = output["content"]
    candidates = parse_candidates(output)
    if not candidates and output not in ({"candidates": []}, "{\"candidates\": []}"):
        return [], "malformed_json_or_empty"
    return candidates, "ok"


def apply_proposals(store: Any, session_id: str, message: Any, provider: Any) -> FormationResult:
    """Validate and admit proposals; deterministic extraction remains separate."""
    candidates, diagnostic = provider_proposals(provider, message, session_id)
    if diagnostic != "ok":
        return FormationResult(diagnostics=[diagnostic], model_calls=1)
    admitted: list[int] = []
    held: list[int] = []
    diagnostics: list[str] = []
    for candidate in candidates:
        valid, reason = validate_candidate(
            candidate, session_id=session_id, message=message,
            source_lookup=store.fetch_message,
        )
        if not valid:
            diagnostics.append(f"proposal_rejected:{reason}")
            continue
        # Relationships and observations are reviewable proposals until an
        # entity endpoint or explicit observation admission is available.
        if candidate.kind in {"relationship", "observation"}:
            held.append(store.create_candidate(
                session_id, "global" if candidate.scope == "global" else session_id,
                candidate.source_ids, candidate.kind, candidate.concept,
                candidate.value, candidate.rationale, "user", candidate.confidence,
            ))
            continue
        if candidate.kind == "event":
            event_kind = candidate.predicate or "activity"
            try:
                event_id = store.save_experience_event(
                    session_id, candidate.source_ids[0], event_kind, "i",
                    store.fetch_message(candidate.source_ids[0]).content,
                )
            except Exception as exc:
                diagnostics.append(f"event_proposal_held:{exc}")
                held.append(store.create_candidate(
                    session_id, session_id, candidate.source_ids, candidate.kind,
                    candidate.concept, candidate.value, candidate.rationale,
                    "user", candidate.confidence,
                ))
            else:
                if event_id is not None:
                    admitted.append(event_id)
            continue
        scope = "global" if candidate.scope == "global" else session_id
        existing = [state for state in store.fetch_states(scope=scope, concept=candidate.concept, status="current")]
        predecessor_ids = [state.state_id for state in existing]
        try:
            same = next((state for state in existing if state.value.casefold() == candidate.value.casefold()), None)
            if same is not None:
                for source_id in candidate.source_ids:
                    if source_id not in same.source_ids:
                        store.add_state_evidence(same.state_id, source_id)
                admitted.append(same.state_id)
                continue
            if predecessor_ids:
                state_id = store.create_state_successor(
                    session_id, scope, candidate.concept, candidate.value,
                    candidate.source_ids, predecessor_ids, memory_type=(
                        "PREFERENCE" if candidate.kind == "preference" else "FACT"
                    ), confidence=candidate.confidence, observed_at=message.timestamp,
                    valid_from=candidate.valid_from,
                    valid_until=candidate.valid_until,
                )
            else:
                state_id = store.create_state(
                    session_id, scope, candidate.concept, candidate.value,
                    candidate.source_ids, memory_type=(
                        "PREFERENCE" if candidate.kind == "preference" else "FACT"
                    ), confidence=candidate.confidence, observed_at=message.timestamp,
                    valid_from=candidate.valid_from,
                    valid_until=candidate.valid_until,
                )
            admitted.append(state_id)
        except Exception as exc:
            diagnostics.append(f"proposal_write_failed:{exc}")
    return FormationResult(candidates=candidates, admitted_ids=admitted,
                           held_ids=held, diagnostics=diagnostics, model_calls=1)

