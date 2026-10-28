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
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from PRIMORA.prime.contract import RequestStatus
from .episodes import extract_experience_claim
from .admission import MemoryAdmissionGateway

MEMORY_KINDS = frozenset({"preference", "state", "event", "relationship", "observation"})
CLAIM_KINDS = frozenset({
    "assertion", "hypothesis", "question", "fiction", "negation",
    "third_party", "assistant",
})
ADMITTABLE_CLAIMS = frozenset({"assertion"})
MAX_CANDIDATES = 8
MAX_TEXT = 1000


def _positive_int_env(name: str, default: int, maximum: int = 2048) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(1, min(value, maximum))


def _temperature_env(name: str, default: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(0.0, min(value, 2.0))


def _thinking_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")
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
    "dont", "do", "not", "no", "longer", "anymore", "currently", "now",
})

_NEGATIVE_PREFERENCE_RE = re.compile(
    r"\b(?:dislike|dislikes|disliked|hate|hates|hated)\b|"
    r"\b(?:no longer|not anymore|don't|do not|never)\s+(?:really\s+)?"
    r"(?:like|love|enjoy|prefer)\b", re.I,
)
_POSITIVE_PREFERENCE_RE = re.compile(
    r"\b(?:like|likes|liked|love|loves|loved|enjoy|enjoys|enjoyed|"
    r"prefer|prefers|preferred|favorite|favourite)\b", re.I,
)


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


# This is linguistic evidence checking, not an object/topic vocabulary.
# A model cannot promote "enjoy" to "love", or "like" to "favorite",
# merely because both have positive sentiment.
_PREFERENCE_VERBS = {
    "like": r"\b(?:like|likes|liked|liking)\b",
    "love": r"\b(?:love|loves|loved|loving)\b",
    "adore": r"\b(?:adore|adores|adored|adoring)\b",
    "enjoy": r"\b(?:enjoy|enjoys|enjoyed|enjoying)\b",
    "prefer": r"\b(?:prefer|prefers|preferred|preferring)\b",
    "favorite": r"\b(?:favorite|favourite|favorites|favourites)\b",
    "hate": r"\b(?:hate|hates|hated|hating)\b",
    "dislike": r"\b(?:dislike|dislikes|disliked|disliking)\b",
    "admire": r"\b(?:admire|admires|admired|admiring)\b",
}
_NEGATED_LIKE_CLAUSE = re.compile(
    r"\b(?:do not|don't|no longer|never)\s+(?:really\s+)?like\b",
    re.IGNORECASE,
)


def _source_preference_verbs(text: str) -> set[str]:
    """Normalize inflection without conflating different preference strengths."""
    text = text or ""
    has_negated_like = bool(_NEGATED_LIKE_CLAUSE.search(text))
    text = _NEGATED_LIKE_CLAUSE.sub(" ", text)
    modes = {
        label for label, pattern in _PREFERENCE_VERBS.items()
        if re.search(pattern, text, flags=re.IGNORECASE)
    }
    if has_negated_like:
        modes.add("dislike")
    return modes


def _preference_fingerprint(concept: str, value: str) -> set[str]:
    return _preference_terms(f"{concept} {value}")


def _preference_polarity(concept: str, value: str, source: str = "") -> Optional[str]:
    """Return a typed preference polarity, independent of subject vocabulary."""
    corpus = f"{concept} {value} {source}"
    if _NEGATIVE_PREFERENCE_RE.search(corpus):
        return "negative"
    if _POSITIVE_PREFERENCE_RE.search(corpus):
        return "positive"
    return None


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
    if candidate.rationale:
        rationale = " ".join(candidate.rationale.split()).casefold()
        if len(rationale) < 12:
            return False, "rationale_too_short"
        if not any(rationale in " ".join(source.content.split()).casefold()
                   for source in sources):
            return False, "rationale_not_source_supported"
        if re.match(r"^(?:you|someone|people|they)\b", rationale, re.I):
            return False, "rationale_not_user_authored"
    if candidate.kind in {"preference", "state"}:
        # Derivative statements with explicit affinity verbs must preserve the
        # user's stated predicate. Neutral summaries without a verb continue
        # through the existing evidence gates unchanged.
        proposed_verbs = _source_preference_verbs(candidate.value)
        if proposed_verbs:
            source_verbs = set().union(*(
                _source_preference_verbs(source.content)
                for source in sources
            ))
            if not proposed_verbs.issubset(source_verbs):
                return False, "preference_strength_not_source_supported"
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



# Strict, domain-independent admission shape for the model-first path.
# This is a semantic schema check, not an entity-specific extraction regex.
_PREFERENCE_WORDING = {
    "love": "loves", "like": "likes", "adore": "adores",
    "enjoy": "enjoys", "prefer": "prefers",
    "hate": "hates", "dislike": "dislikes",
}
_NON_REFERENTIAL = frozenset({
    "it", "this", "that", "them", "everything", "anything", "something",
    "someone", "anyone", "everyone", "nothing",
})


def validate_semantic_candidate(candidate: TypedMemoryCandidate, message: Any) -> tuple[bool, str]:
    """Check canonical meaning and literal user evidence before durable write."""
    quote = candidate.source_quote.strip()
    if candidate.source_ids != [message.message_id]:
        return False, "semantic_source_must_be_current_user"
    if not quote or quote.casefold() not in message.content.casefold():
        return False, "semantic_missing_exact_quote"
    negative_preference = (
        candidate.kind == "preference"
        and candidate.claim_kind == "negation"
        and candidate.predicate in ("dislike", "hate")
    )
    if candidate.claim_kind != "assertion" and not negative_preference:
        return False, "semantic_non_assertion"
    if candidate.kind != "preference":
        # Other types remain constrained by the existing provenance validator.
        return True, "ok"
    subject = " ".join(candidate.subject.split())
    if (not subject or len(subject) > 90 or len(subject.split()) > 9
            or subject.casefold() in _NON_REFERENTIAL):
        return False, "semantic_invalid_subject"
    if re.search(r"\b(?:i|we|you|they)\s+(?:love|like|adore|enjoy|"
                 r"prefer|hate|dislike|think|feel|say)\b", subject, re.I):
        return False, "semantic_subject_is_sentence"
    if not re.search(r"(?<!\w)" + re.escape(subject) + r"(?!\w)", quote, re.I):
        return False, "semantic_subject_not_in_quote"
    if candidate.concept.casefold() != "likes for " + subject.casefold():
        return False, "semantic_bad_concept"
    predicate = candidate.predicate.casefold()
    verb = _PREFERENCE_WORDING.get(predicate)
    if verb is None:
        return False, "semantic_unknown_attitude"
    if candidate.value.casefold().strip(" .!") != f"{verb} {subject}".casefold():
        return False, "semantic_bad_value"
    if predicate not in _source_preference_verbs(quote):
        return False, "semantic_attitude_unproven"
    # Naming an object and an attitude in someone else's report is not
    # evidence of the user's own preference.
    if not re.search(
        r"\b(?:i|we)\s+(?:(?:really|truly|absolutely)\s+)?"
        r"(?:love|like|adore|enjoy|prefer|hate|dislike|"
        r"don't\s+like|do\s+not\s+like)\b"
        r"|\bmy\s+(?:fav|favorite|favourite)\b",
        quote, re.I,
    ):
        return False, "semantic_attitude_not_first_person"
    # Reject clear hypothetical framing even if the model mislabeled it.
    if re.search(
        r"\b(?:hypothetical|fictional|pretend(?:ing)?|imaginary|"
        r"imagine\s+(?:that|if)|if\s+i\s+(?:were|had|could|would))\b",
        message.content, re.I,
    ):
        return False, "semantic_nonfactual_source"
    return True, "ok"


def provider_proposals(
    provider: Any, message: Any, session_id: str, *,
    context: Optional[list[dict[str, str]]] = None,
    semantic_first: bool = False,
) -> tuple[list[TypedMemoryCandidate], str]:
    """Make one bounded provider call and parse its strict response."""
    if provider is None:
        return [], "provider_not_configured"
    prior = []
    if semantic_first:
        for turn in (context or [])[-4:]:
            role = str(turn.get("role", ""))
            content = str(turn.get("content", ""))[:320]
            if role in ("user", "assistant") and content:
                prior.append({"role": role, "content": content})
    payload = {
        "messages": [
            {"role": "system", "content": (
                "Extract only explicit user-authored memory proposals. Return JSON only: "
                '{"candidates":[{"kind":"preference|state|event|relationship|observation",'
                '"concept":"...","value":"...","claim_kind":"assertion|hypothesis|question|fiction|negation|third_party|assistant",'
                '"actor":"user","scope":"session|global","source_ids":[ID],"source_quote":"exact substring",'
                f'"predicate":"...","rationale":"exact user-stated reason"}}]}} . Use global for a personal preference unless the user explicitly makes it temporary or session-only. The only admissible source is source_message_id={message.message_id}; copy that integer and an exact substring from the user message as source_quote. If a reason is present, copy an exact user-authored causal clause into rationale. Never infer from an assistant message. Questions, fiction, and hypotheses are not assertions; direct dislike or hate is a real negative preference, for which claim_kind=negation is acceptable. In the preference kind, provide a short exact named subject, the attitude predicate (love/like/adore/enjoy/prefer/hate/dislike), concept as likes for <subject>, and third-person value as loves/likes/adores/enjoys/prefers/hates/dislikes <subject>. Separate subject from descriptive attributes. For an explicit personal fact or favorite slot (my favorite <category> is <value>), propose kind=state, concept=favorite <category>, and value as the exact stated <value>. Do not treat it as an attitude toward the category. If no reliable claim return an empty candidates array. Earlier dialogue resolves references only, never supplies personal evidence.'
            )},
            *prior,
            {"role": "user", "content": message.content},
        ],
        "session_id": session_id,
        "source_message_id": message.message_id,
        "options": {
            "temperature": _temperature_env(
                "PRIMORA_MEMORY_FORMATION_TEMPERATURE"
            ),
            "max_tokens": _positive_int_env(
                "PRIMORA_MEMORY_FORMATION_MAX_TOKENS", 256
            ),
            # Qwen3 chat templates otherwise spend a local model's budget in
            # reasoning_content before emitting the requested JSON.
            "chat_template_kwargs": {"enable_thinking": _thinking_env(
                "PRIMORA_MEMORY_FORMATION_ENABLE_THINKING", False
            )},
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


def _admit_rationale(
    store: Any,
    session_id: str,
    scope: str,
    preference_state_id: int,
    candidate: TypedMemoryCandidate,
    message: Any,
    admission: Optional[MemoryAdmissionGateway] = None,
) -> list[str]:
    """Persist a provider rationale only after source validation.

    Typed formation has one admission boundary. A rationale is a second
    derived state linked to the admitted preference; it is never accepted as
    a free-standing explanation and it retains the same user source ids.
    """
    rationale = " ".join(candidate.rationale.split()).strip()
    if not rationale:
        return []
    concept = "rationale for " + candidate.concept
    current = store.fetch_states(scope=scope, concept=concept, status="current")
    for row in current:
        if row.value.casefold().strip() == rationale.casefold():
            for source_id in candidate.source_ids:
                if source_id not in row.source_ids:
                    store.add_state_evidence(row.state_id, source_id)
            return [f"rationale_evidence_merged: {row.state_id}"]
    if current:
        transition = getattr(message, "timestamp", None)
        store.supersede_states(
            [row.state_id for row in current],
            transition_time=transition,
        )
    admission = admission or MemoryAdmissionGateway(store)
    rationale_id, relationship_id = admission.create_state_with_explains(
        session_id, scope, concept, rationale, candidate.source_ids,
        preference_state_id, memory_type="RATIONALE", confidence=candidate.confidence,
        observed_at=getattr(message, "timestamp", None),
    )
    return [
        f"rationale_admitted: {rationale_id}",
        f"rationale_relationship_admitted: {relationship_id}",
    ]


_EXPLICIT_CAUSAL_SOURCE_RE = re.compile(
    r"(?:^|[.!?;])\s*((?:it|this|that)\s+"
    r"(?:helps?|makes?|keeps?|lets?|allows?|gives?|supports?|"
    r"improves?|calms?|relaxes?|focuses?|grounds?|energizes?)\b.+)$",
    re.IGNORECASE | re.DOTALL,
)


def _source_rationale(store: Any, source_ids: list[int]) -> str:
    """Return an explicitly causal source clause, never a paraphrase."""
    for source_id in source_ids:
        source = store.fetch_message(source_id)
        if source is None or source.role != "user":
            continue
        match = _EXPLICIT_CAUSAL_SOURCE_RE.search(source.content.strip())
        if match:
            value = " ".join(match.group(1).split()).strip(" .!?;")
            if len(value) >= 12:
                return value
    return ""


_EXPLICIT_ATTRIBUTE_CAUSE_RE = re.compile(
    r"\bfor\s+((?:its|their|the way|how)\b[^.!?;]+)",
    re.IGNORECASE,
)


def _rationale_supported_by_user_source(
    store: Any, source_ids: list[int], rationale: str,
) -> bool:
    """Require a reason within an explicit causal clause, not just any quote.

    A matching source sentence alone does not prove that it gives a *reason*.
    Interpretation belongs to Aurora; Prime conservatively verifies an
    explicit user-authored causal span before admitting a derived explanation.
    Unrecognized causal phrasing stays in raw history rather than becoming
    an invented rationale.
    """
    from .state import _split_rationale

    claimed = " ".join((rationale or "").split()).strip().casefold()
    if len(claimed) < 12:
        return False
    for source_id in source_ids:
        source = store.fetch_message(source_id)
        if source is None or source.role != "user":
            continue
        content = " ".join(source.content.split())
        _head, because_reason = _split_rationale(content)
        reasons = [because_reason, _source_rationale(store, [source_id])]
        # "I admire this for its unusual shape" explicitly states an
        # attribute as a reason. "I use this for travel notes" is a
        # purpose/context and is not necessarily why it is preferred.
        attribute = _EXPLICIT_ATTRIBUTE_CAUSE_RE.search(content)
        if attribute:
            reasons.append(attribute.group(1))
        if any(
            claimed in " ".join(reason.split()).casefold()
            for reason in reasons if reason
        ):
            return True
    return False


def apply_proposals(
    store: Any, session_id: str, message: Any, provider: Any, *,
    context: Optional[list[dict[str, str]]] = None,
    semantic_first: bool = False,
) -> FormationResult:
    """Validate and admit proposals; deterministic extraction remains separate."""
    timed_store = _TimingStore(store)
    store = timed_store
    admission = MemoryAdmissionGateway(store)
    started = time.perf_counter()
    provider_started = time.perf_counter()
    candidates, diagnostic = provider_proposals(
        provider, message, session_id, context=context,
        semantic_first=semantic_first,
    )
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
        # A provider may quote the whole source sentence as a "reason".
        # A quote proves attribution, not causality. Keep the supported
        # primary memory but drop an explanation without an explicit user
        # causal clause. Reject invented standalone rationale states.
        if candidate.rationale and not _rationale_supported_by_user_source(
            store, candidate.source_ids, candidate.rationale,
        ):
            from dataclasses import replace
            candidate = replace(candidate, rationale="")
            diagnostics.append("rationale_dropped:no_explicit_user_reason")
        if (candidate.concept.casefold().startswith("rationale for ")
                and not _rationale_supported_by_user_source(
                    store, candidate.source_ids, candidate.value,
                )):
            diagnostics.append("proposal_rejected:unsupported_rationale_value")
            continue
        valid, reason = validate_candidate(
            candidate, session_id=session_id, message=message,
            source_lookup=store.fetch_message,
        )
        if valid and semantic_first:
            valid, reason = validate_semantic_candidate(candidate, message)
        if not valid:
            diagnostics.append(f"proposal_rejected:{reason}")
            continue
        if candidate.kind == "preference" and not candidate.rationale:
            explicit_rationale = _source_rationale(store, candidate.source_ids)
            if explicit_rationale:
                from dataclasses import replace
                candidate = replace(candidate, rationale=explicit_rationale)
                diagnostics.append("rationale_bound_to_explicit_user_clause")
        # Relationships and observations are reviewable proposals until an
        # entity endpoint or explicit observation admission is available.
        if candidate.kind in {"relationship", "observation"}:
            held.append(admission.create_pending(
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
                held.append(admission.create_pending(
                    session_id, session_id, candidate.source_ids, candidate.kind,
                    candidate.concept, candidate.value, candidate.rationale,
                    "user", candidate.confidence,
                ))
            else:
                if event_id is not None:
                    admitted.append(event_id)
            continue
        scope = "global" if candidate.scope == "global" else session_id
        if semantic_first and candidate.kind == "state":
            # A paraphrased generic assertion is not automatically a user
            # fact. Without literal evidence for its value, keep it pending
            # for review rather than promote speculative model prose.
            if candidate.value.casefold() not in candidate.source_quote.casefold():
                held.append(admission.create_pending(
                    session_id, scope, candidate.source_ids, candidate.kind,
                    candidate.concept, candidate.value, candidate.rationale,
                    "user", candidate.confidence,
                ))
                diagnostics.append("state_proposal_held:not_literal_source_value")
                continue
        existing = [state for state in store.fetch_states(scope=scope, concept=candidate.concept, status="current")]
        if candidate.kind == "preference":
            candidate_terms = _preference_fingerprint(
                candidate.concept, candidate.value
            )
            def _source_text(source_ids: list[int]) -> str:
                texts = []
                for source_id in source_ids:
                    source = store.fetch_message(source_id)
                    if source is not None:
                        texts.append(source.content)
                return " ".join(texts)

            candidate_source = _source_text(candidate.source_ids)
            candidate_polarity = _preference_polarity(
                candidate.concept, candidate.value, candidate_source
            )
            deduplicated = False
            for state in store.fetch_states(scope=scope, status="current"):
                if not set(candidate.source_ids).intersection(state.source_ids):
                    continue
                state_terms = _preference_fingerprint(state.concept, state.value)
                state_sources = _source_text(state.source_ids)
                state_polarity = _preference_polarity(
                    state.concept, state.value, state_sources
                )
                # Different wording for the same subject is one semantic
                # claim only when polarity also agrees.  Exact subject terms
                # avoid collapsing distinct claims such as "ceramics" and
                # "ceramics workshops" while still merging "no longer like"
                # with the canonical dislike state from the same source.
                if (candidate_terms and candidate_terms == state_terms
                        and candidate_polarity is not None
                        and candidate_polarity == state_polarity):
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
                diagnostics.extend(_admit_rationale(
                    store, session_id, scope, same.state_id, candidate, message,
                    admission=admission,
                ))
                continue
            if predecessor_ids:
                state_id = admission.create_state_successor(
                    session_id, scope, candidate.concept, candidate.value,
                    candidate.source_ids, predecessor_ids, memory_type=(
                        "PREFERENCE" if candidate.kind == "preference" else "FACT"
                    ), confidence=candidate.confidence, observed_at=message.timestamp,
                    valid_from=candidate.valid_from,
                    valid_until=candidate.valid_until,
                )
            else:
                state_id = admission.create_state(
                    session_id, scope, candidate.concept, candidate.value,
                    candidate.source_ids, memory_type=(
                        "PREFERENCE" if candidate.kind == "preference" else "FACT"
                    ), confidence=candidate.confidence, observed_at=message.timestamp,
                    valid_from=candidate.valid_from,
                    valid_until=candidate.valid_until,
                )
            admitted.append(state_id)
            diagnostics.extend(_admit_rationale(
                store, session_id, scope, state_id, candidate, message,
                admission=admission,
            ))
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

