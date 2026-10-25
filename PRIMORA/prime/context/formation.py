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
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from PRIMORA.prime.contract import RequestStatus
from .episodes import extract_experience_claim

MEMORY_KINDS = frozenset({"preference", "state", "event", "relationship", "observation"})
CLAIM_KINDS = frozenset({
    "assertion", "hypothesis", "question", "fiction", "negation",
    "third_party", "assistant",
})
ADMITTABLE_CLAIMS = frozenset({"assertion"})
MAX_CANDIDATES = 8
MAX_TEXT = 1000
PREFERENCE_PREDICATES = frozenset({
    "like", "likes", "liked", "love", "loves", "loved", "prefer",
    "prefers", "preferred", "enjoy", "enjoys", "enjoyed", "favorite",
    "favourite", "dislike", "dislikes", "disliked", "hate", "hates",
    "hated", "don't like", "do not like",
})
_EVENT_KIND_MAP = {
    "visit": "activity", "visited": "activity", "attend": "activity",
    "attended": "activity", "watch": "activity", "watched": "activity",
    "read": "activity", "play": "activity", "played": "activity",
    "listen": "activity", "listened": "activity", "activity": "activity",
    "complete": "completion", "completed": "completion",
    "finish": "completion", "finished": "completion",
    "resolve": "resolution", "resolved": "resolution",
    "start": "start", "started": "start", "begin": "start", "began": "start",
}
_PREFERENCE_FILLER_WORDS = frozenset({
    "a", "an", "and", "for", "i", "my", "the", "thing", "things",
    "user", "preferred", "preference", "preferences", "like", "likes",
    "liked", "love", "loves", "loved", "prefer", "prefers", "preferred",
    "enjoy", "enjoys", "enjoyed", "favorite", "favourite",
    "dislike", "dislikes", "disliked", "hate", "hates", "hated", "don",
    "dont", "do", "not",
})


def _preference_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", text.casefold()):
        if raw in _PREFERENCE_FILLER_WORDS or len(raw) < 3:
            continue
        if raw.endswith("ies") and len(raw) > 4:
            raw = raw[:-3] + "y"
        elif raw.endswith("s") and len(raw) > 4:
            raw = raw[:-1]
        terms.add(raw)
    return terms


def _preference_fingerprint(concept: str, value: str) -> set[str]:
    return _preference_terms(f"{concept} {value}")


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
    timings_ms: dict[str, float] = field(default_factory=dict)


class _TimingStore:
    """Thin timing proxy for the SQLite calls made during proposal admission."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self.elapsed_ms = 0.0

    def __getattr__(self, name: str):
        target = getattr(self._store, name)
        if not callable(target):
            return target

        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return target(*args, **kwargs)
            finally:
                self.elapsed_ms += (time.perf_counter() - started) * 1000

        return timed


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
            # llama.cpp/Qwen can return a short reasoning preamble followed
            # by the requested JSON object. Decode only an actual JSON object;
            # field and provenance validation still happen below.
            data = None
            decoder = json.JSONDecoder()
            for position, character in enumerate(text[:4000]):
                if character != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(text[position:])
                except (TypeError, ValueError):
                    continue
                if isinstance(candidate, dict):
                    data = candidate
                    break
            if data is None:
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
    if candidate.claim_kind not in ADMITTABLE_CLAIMS and not (
        candidate.kind == "preference" and candidate.claim_kind == "negation"
    ):
        return False, f"non_assertion:{candidate.claim_kind}"
    if candidate.actor != "user":
        return False, "authority_actor_not_user"
    if candidate.scope not in {"session", "global"}:
        return False, "invalid_scope"
    if not candidate.concept or not candidate.value:
        return False, "missing_concept_or_value"
    if candidate.kind == "preference":
        if candidate.value.casefold().strip(" .!?\"'") in PREFERENCE_PREDICATES:
            return False, "preference_value_is_predicate"
        if re.match(r"^(?:i|we)\s+(?:like|love|prefer|enjoy)\b", candidate.value, re.I):
            return False, "preference_value_is_statement"
        if candidate.concept.casefold().strip(" .!?\"'") in PREFERENCE_PREDICATES:
            return False, "preference_concept_is_predicate"
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
    if candidate.kind == "preference":
        candidate_terms = _preference_fingerprint(
            candidate.concept, candidate.value
        )
        source_terms = set().union(*(
            _preference_terms(source.content) for source in sources
        ))
        if not candidate_terms or not candidate_terms.issubset(source_terms):
            return False, "preference_generalization_not_source_supported"
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
                f'"predicate":"..."}}]}} . Use global for a personal preference unless the user explicitly makes it temporary or session-only. The only admissible source is source_message_id={message.message_id}; copy that integer and an exact substring from the user message as source_quote. Never infer from an assistant message. Questions, fiction, negation, and hypotheses are not assertions.'
            )},
            {"role": "user", "content": message.content},
        ],
        "session_id": session_id,
        "source_message_id": message.message_id,
        "options": {
            "temperature": 0,
            "max_tokens": 256,
            # Qwen3 chat templates otherwise spend a local model's budget in
            # reasoning_content before emitting the requested JSON.
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "response_format": {"type": "json_object"},
    }
    try:
        result = provider.invoke("memory_formation", payload)
    except Exception as exc:
        return [], f"provider_error:{exc}"
    if getattr(result, "status", None) != RequestStatus.SUCCESS:
        return [], f"provider_failure:{getattr(result, 'error', 'unknown')}"
    output = getattr(result, "output", None)
    if isinstance(output, dict):
        output = output.get("content") or output.get("reasoning_content") or output.get("reasoning")
    candidates = parse_candidates(output)
    if not candidates and output not in ({"candidates": []}, "{\"candidates\": []}"):
        if isinstance(output, str) and _empty_candidate_object(output):
            return [], "ok"
        return [], "malformed_json_or_empty"
    return candidates, "ok"


def _empty_candidate_object(text: str) -> bool:
    decoder = json.JSONDecoder()
    for position, character in enumerate(text[:4000]):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[position:])
        except (TypeError, ValueError):
            continue
        if isinstance(candidate, dict) and candidate.get("candidates") == []:
            return True
    return False


def apply_proposals(store: Any, session_id: str, message: Any, provider: Any) -> FormationResult:
    """Validate and admit proposals; deterministic extraction remains separate."""
    timed_store = _TimingStore(store)
    store = timed_store
    started = time.perf_counter()
    provider_started = time.perf_counter()
    candidates, diagnostic = provider_proposals(provider, message, session_id)
    provider_ms = (time.perf_counter() - provider_started) * 1000
    timings = {"provider_ms": provider_ms}
    if diagnostic != "ok":
        timings["db_ms"] = timed_store.elapsed_ms
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        return FormationResult(
            diagnostics=[diagnostic, _timing_diagnostic(timings)],
            model_calls=1, timings_ms=timings,
        )
    admitted: list[int] = []
    held: list[int] = []
    diagnostics: list[str] = []
    processing_started = time.perf_counter()
    for candidate in candidates:
        candidate, provenance_bound = _bind_current_provenance(candidate, message)
        if provenance_bound:
            diagnostics.append("provenance_bound_to_current_user_source")
        session_bound = bool(re.search(
            r"\b(?:this session|this conversation|for now|temporarily|only here)\b",
            message.content, re.IGNORECASE,
        ))
        if candidate.kind == "preference" and (
            (candidate.scope == "session" and not session_bound)
            or (candidate.scope == "global" and session_bound)
        ):
            # Personal preferences are cross-session by default. Explicit
            # temporary/session wording remains session-scoped. Provenance
            # validation still runs after this normalization.
            from dataclasses import replace
            candidate = replace(
                candidate, scope="session" if session_bound else "global"
            )
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
            source = store.fetch_message(candidate.source_ids[0])
            claim = extract_experience_claim(
                source.content, "user", reported_at=source.timestamp
            ) if source is not None else None
            event_kind = _EVENT_KIND_MAP.get(
                (candidate.predicate or "").casefold(),
                claim.kind if claim is not None else "activity",
            )
            actor = claim.actor if claim is not None else (
                "we" if source is not None and source.content.casefold().startswith("we ") else "i"
            )
            statement = claim.statement if claim is not None else source.content
            try:
                event_id = store.save_experience_event(
                    session_id, candidate.source_ids[0], event_kind, actor,
                    statement,
                    claim.occurred_at if claim is not None else None,
                    claim.occurred_precision if claim is not None else "unknown",
                    claim.occurred_timezone if claim is not None else "UTC",
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
        if candidate.kind == "preference":
            candidate_terms = _preference_fingerprint(
                candidate.concept, candidate.value
            )
            deduplicated = False
            for state in store.fetch_states(scope=scope, status="current"):
                if not set(candidate.source_ids).intersection(state.source_ids):
                    continue
                state_terms = _preference_fingerprint(state.concept, state.value)
                if candidate_terms and candidate_terms.issubset(state_terms):
                    for source_id in candidate.source_ids:
                        if source_id not in state.source_ids:
                            store.add_state_evidence(state.state_id, source_id)
                    admitted.append(state.state_id)
                    diagnostics.append(
                        f"proposal_deduplicated: preference state {state.state_id}"
                    )
                    deduplicated = True
                    break
            if deduplicated:
                continue
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
    processing_ms = (time.perf_counter() - processing_started) * 1000
    timings["admission_ms"] = processing_ms
    timings["db_ms"] = timed_store.elapsed_ms
    timings["total_ms"] = (time.perf_counter() - started) * 1000
    diagnostics.append(_timing_diagnostic(timings))
    return FormationResult(candidates=candidates, admitted_ids=admitted,
                           held_ids=held, diagnostics=diagnostics, model_calls=1,
                           timings_ms=timings)


def _timing_diagnostic(timings: dict[str, float]) -> str:
    return "typed_formation_timing: " + " ".join(
        f"{key}={value:.2f}ms" for key, value in sorted(timings.items())
    )


def _bind_current_provenance(
    candidate: TypedMemoryCandidate, message: Any,
) -> tuple[TypedMemoryCandidate, bool]:
    """Fill only omitted provenance from the exact message Prime is forming.

    A model may omit the repeated source fields even after being instructed to
    emit them. Binding an omission to the current user message is safe and
    deterministic; an explicit wrong id or unsupported quote still reaches
    the normal rejection gates unchanged.
    """
    from dataclasses import replace

    if candidate.source_ids:
        if candidate.source_ids == [message.message_id] and not candidate.source_quote:
            return replace(candidate, source_quote=message.content), True
        return candidate, False
    if candidate.source_quote and candidate.source_quote.casefold() not in message.content.casefold():
        return candidate, False
    return replace(
        candidate, source_ids=[message.message_id],
        source_quote=candidate.source_quote or message.content,
    ), True

