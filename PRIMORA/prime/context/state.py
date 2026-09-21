"""Deterministic user-state extraction and transitions. No LLM.

Only user-authored messages can create or change current state.
Assistant statements never become current state on their own; an
assistant claim that contradicts tracked current state is recorded as
`unresolved` for visibility, never as truth.

Raw messages are never modified. Supersession links rows; old rows stay.
"""

import re
from dataclasses import dataclass
from typing import Any, Optional

GLOBAL_MARKERS = (
    "for all conversations",
    "for every session",
    "globally",
    "across sessions",
    "in every session",
)

_ARTICLES_RE = re.compile(r"^(?:the|my|a|an)\s+", re.IGNORECASE)

_INVALIDATE_RE = re.compile(
    r"^(.+?)\s+is\s+no\s+longer\s+(?:correct|true|valid|right)\.?$", re.IGNORECASE
)
_CORRECT_NEW_RE = re.compile(
    r"^(?:the\s+)?new\s+(?:and\s+correct\s+)?(.+?)\s+is\s+(.+?)\.?$",
    re.IGNORECASE,
)
_CORRECT_WORD_RE = re.compile(
    r"^(?:the\s+)?correct\s+(.+?)\s+is\s+(.+?)\.?$", re.IGNORECASE
)
_FROM_NOW_ON_RE = re.compile(
    r"^from\s+now\s+on[,:]?\s+(.+)$", re.IGNORECASE
)
_CHANGE_RE = re.compile(
    r"^change\s+(.+?)\s+to\s+(.+?)\.?$", re.IGNORECASE
)
_INSTEAD_RE = re.compile(
    r"^instead[,:]?\s+(?:use\s+)?(.+)$", re.IGNORECASE
)
_ESTABLISH_RE = re.compile(
    r"^(?:the|my)\s+(.+?)\s+is\s+(.+?)\.?$", re.IGNORECASE
)
_CLEAN_GLOBAL_RE = re.compile(
    r"\b(?:for all conversations|for every session|globally|"
    r"across sessions|in every session)\b[,:]?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StateCandidate:
    concept: str
    value: str = ""
    kind: str = "establish"  # establish | correct | invalidate
    scope: str = "session"  # session | global


def normalize_concept(raw: str) -> str:
    text = _ARTICLES_RE.sub("", raw.strip()).strip().lower()
    return re.sub(r"\s+", " ", text)


def normalize_value(raw: str) -> str:
    return raw.strip().rstrip(".")


def _scope_of(text: str) -> str:
    lowered = text.lower()
    for marker in GLOBAL_MARKERS:
        if marker in lowered:
            return "global"
    return "session"


def _parse_establishment(text: str) -> Optional[tuple[str, str]]:
    match = _ESTABLISH_RE.match(text.strip())
    if not match:
        return None
    concept = normalize_concept(match.group(1))
    value = normalize_value(match.group(2))
    if not concept or not value or len(concept.split()) > 6:
        return None
    return concept, value


def extract_candidates(text: str, role: str) -> list[StateCandidate]:
    """Extract state candidates from user-authored text only."""
    if role != "user" or not isinstance(text, str):
        return []
    stripped = text.strip()
    if not stripped:
        return []
    scope = _scope_of(stripped)
    # A global marker ("Globally, the motto is X") is routing metadata,
    # not part of the statement itself.
    work = (
        _CLEAN_GLOBAL_RE.sub(" ", stripped).strip(" ,")
        if scope == "global" else stripped
    )
    if not work:
        return []

    match = _INVALIDATE_RE.match(work)
    if match:
        concept = normalize_concept(match.group(1))
        if concept:
            return [StateCandidate(concept, "", "invalidate", scope)]
        return []

    for pattern in (_CORRECT_NEW_RE, _CORRECT_WORD_RE):
        match = pattern.match(work)
        if match:
            concept = normalize_concept(match.group(1))
            value = normalize_value(match.group(2))
            if concept and value:
                return [StateCandidate(concept, value, "correct", scope)]
            return []

    match = _CHANGE_RE.match(work)
    if match:
        concept = normalize_concept(match.group(1))
        value = normalize_value(match.group(2))
        if concept and value:
            return [StateCandidate(concept, value, "correct", scope)]
        return []

    match = _FROM_NOW_ON_RE.match(work)
    if match:
        parsed = _parse_establishment(match.group(1))
        if parsed:
            return [StateCandidate(parsed[0], parsed[1], "correct", scope)]
        return []

    match = _INSTEAD_RE.match(work)
    if match:
        parsed = _parse_establishment(match.group(1))
        if parsed:
            return [StateCandidate(parsed[0], parsed[1], "correct", scope)]
        return []

    parsed = _parse_establishment(work)
    if parsed:
        return [StateCandidate(parsed[0], parsed[1], "establish", scope)]
    return []


def _scope_key(scope: str, session_id: str) -> str:
    return "global" if scope == "global" else session_id


def resolve_concept(tracked: list[str], candidate: str) -> str:
    """Map a candidate concept onto an already-tracked one, if related.

    Exact match wins. Otherwise the longest tracked concept that appears
    as a whole word inside the candidate (or vice versa) wins, so "test
    phrase" and "phrase" refer to the same tracked concept while
    "passphrase" does not. Deterministic; unknown concepts pass through.
    """
    if candidate in tracked:
        return candidate
    for known in sorted(set(tracked), key=len, reverse=True):
        if re.search(r"\b" + re.escape(known) + r"\b", candidate):
            return known
        if re.search(r"\b" + re.escape(candidate) + r"\b", known):
            return known
    return candidate


def apply_user_candidates(
    store: Any,
    session_id: str,
    message_id: int,
    candidates: list[StateCandidate],
) -> list[str]:
    """Apply user state changes. Newest user value wins; old rows kept."""
    diagnostics: list[str] = []
    for cand in candidates:
        scope = _scope_key(cand.scope, session_id)
        try:
            known = [s.concept for s in store.fetch_states(scope=scope)]
            concept = resolve_concept(known, cand.concept)
            currents = store.fetch_states(
                scope=scope, concept=concept, status="current"
            )
        except Exception as e:
            diagnostics.append(f"state_failed: state read failed: {e}")
            continue
        if cand.kind == "invalidate":
            for cur in currents:
                store.set_state_status(cur.state_id, "superseded")
            diagnostics.append(f"state_invalidated: {concept}")
            continue
        if currents and currents[0].value.lower() == cand.value.lower():
            continue
        try:
            prev_id = currents[0].state_id if currents else None
            new_id = store.create_state(
                session_id, scope, concept, cand.value, [message_id],
                status="current", supersedes_id=prev_id, authority="user",
            )
            for cur in currents:
                store.set_state_status(cur.state_id, "superseded")
            diagnostics.append(
                f"state_updated: {concept} -> {cand.value} (state {new_id})"
            )
        except Exception as e:
            diagnostics.append(f"state_failed: state write failed: {e}")
    return diagnostics


def check_assistant_conflict(
    store: Any,
    session_id: str,
    message_id: int,
    text: str,
) -> list[str]:
    """Record (never resolve) assistant claims contradicting current state."""
    diagnostics: list[str] = []
    if not isinstance(text, str) or not text.strip():
        return diagnostics
    parsed = _parse_establishment(text.strip())
    if parsed is None:
        return diagnostics
    concept, value = parsed
    try:
        known = [s.concept for s in store.fetch_states(scope=session_id)]
        concept = resolve_concept(known, concept)
        currents = store.fetch_states(
            scope=session_id, concept=concept, status="current"
        )
        existing_unresolved = store.fetch_states(
            scope=session_id, concept=concept, status="unresolved"
        )
    except Exception as e:
        return [f"state_failed: state read failed: {e}"]
    if not currents:
        return diagnostics
    if currents[0].value.lower() == value.lower():
        return diagnostics
    if any(u.value.lower() == value.lower() for u in existing_unresolved):
        return diagnostics
    try:
        new_id = store.create_state(
            session_id, session_id, concept, value, [message_id],
            status="unresolved", supersedes_id=None, authority="assistant",
        )
        diagnostics.append(
            f"state_conflict: {concept} current={currents[0].value!r} "
            f"claimed={value!r} (state {new_id})"
        )
    except Exception as e:
        diagnostics.append(f"state_failed: state write failed: {e}")
    return diagnostics
