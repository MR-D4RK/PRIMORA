"""Deterministic typed memory-query interpretation and evidence ranking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .observation import normalize_concept

_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_PREFERENCE_WORDS = re.compile(
    r"\b(?:like|liked|love|loved|enjoy|enjoyed|prefer|preferred|favorite|favourite|dislike|hate)\b",
    re.I,
)
_PREDICATE_WORDS = re.compile(
    r"\b(liked|loved|enjoyed|preferred|like|love|enjoy|prefer|favorite|favourite|dislike|hate)s?\b", re.I,
)
_EXPLICIT_USER_PREFERENCE = re.compile(
    r"\b(?:i|my)\s+(?:really\s+|usually\s+|always\s+|currently\s+)?"
    r"(?:like|love|enjoy|prefer|favor|favour|hate|dislike|favorite|favourite)\b",
    re.I,
)


@dataclass(frozen=True)
class QueryIntent:
    fact_kind: str
    predicate: str
    scope: str = "global"
    count: Optional[int] = None
    order: str = "relevance"
    temporal_scope: str = "current"
    subject: Optional[str] = None
    positive_only: bool = False


def interpret(text: str, plan: Optional[dict[str, Any]] = None) -> Optional[QueryIntent]:
    text = " ".join(str(text or "").strip().casefold().split())
    if not text or not _PREFERENCE_WORDS.search(text):
        return None
    plan = plan or {}
    count = plan.get("result_count")
    if count is None:
        match = re.search(r"\b(?:last|first|top)\s+(\d+|[a-z]+)\b", text)
        if match:
            raw = match.group(1)
            count = int(raw) if raw.isdigit() else _COUNT_WORDS.get(raw)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 30:
        count = None
    order = plan.get("order") or ("oldest" if re.search(r"\bfirst\b|earliest|oldest", text) else "recent" if re.search(r"\blast\b|latest|recent", text) else "relevance")
    if order not in {"relevance", "recent", "oldest"}:
        order = "relevance"
    temporal = plan.get("temporal_scope") or ("recent" if re.search(r"\brecent|last\b", text) else "current")
    if temporal not in {"current", "historical", "recent"}:
        temporal = "current"
    match = _PREDICATE_WORDS.search(text)
    predicate = match.group(1).casefold() if match else "preference"
    if predicate in {"liked", "likes"}:
        predicate = "like"
    elif predicate in {"loved", "loves"}:
        predicate = "love"
    elif predicate in {"enjoyed", "enjoys"}:
        predicate = "enjoy"
    elif predicate in {"preferred", "prefers"}:
        predicate = "prefer"
    if predicate in {"favorite", "favourite"}:
        predicate = "favorite"
    subject = plan.get("subject")
    if not subject:
        # A named object is a filter after the predicate, never a dictionary
        # of domains.  Inventory queries leave it unset.
        match = re.search(r"\b(?:about|for|in)\s+([a-z][a-z0-9 _-]{1,80})", text)
        if match and not re.search(r"\b(?:the last|recent)", match.group(1)):
            subject = match.group(1).strip(" ?.! ")
    positive_only = bool(re.search(
        r"\b(?:like|liked|love|loved|enjoy|enjoyed|prefer|preferred)\b", text
    )) and "preference" not in text
    return QueryIntent("preference", predicate, "global", count, order, temporal, subject, positive_only)


def _predicate_match(intent: QueryIntent, text: str, concept: str, value: str) -> bool:
    corpus = f"{text} {concept} {value}".casefold()
    if intent.positive_only and re.search(
        r"\b(?:do not|don't|never)\s+(?:really\s+)?(?:like|love|enjoy|prefer)\b"
        r"|\b(?:dislike|hate)\b", text.casefold()
    ):
        return False
    if intent.predicate == "preference":
        return bool(_PREFERENCE_WORDS.search(corpus))
    if intent.predicate == "favorite":
        return bool(re.search(r"\bfavo(?:u)?rite\b", corpus)) or concept.casefold().startswith("favorite ")
    return bool(re.search(rf"\b{re.escape(intent.predicate)}(?:s|ed)?\b", corpus))


def preference_evidence(store: Any, intent: QueryIntent, *, session_id: str,
                        now: Optional[float] = None) -> str:
    """Build a gated evidence packet; relevance is checked before count/order."""
    now = now if now is not None else __import__("time").time()
    candidates: list[dict[str, Any]] = []
    states = store.fetch_states(scope="global") + store.fetch_states(scope=session_id)
    # A raw message that was the source of a superseded preference is
    # historical evidence, not a current answer. Keep its immutable row, but
    # do not let the lexical fallback re-admit it after the lifecycle gate.
    invalid_preference_sources = {
        source_id
        for state in states
        if state.memory_type == "PREFERENCE" and state.status != "current"
        for source_id in state.source_ids
    }
    for state in states:
        allowed_status = ("current", "superseded") if intent.temporal_scope == "historical" else ("current",)
        if state.authority != "user" or state.status not in allowed_status or not state.source_ids:
            continue
        if intent.temporal_scope != "historical" and state.valid_from is not None and not (state.valid_from <= now and (state.valid_until is None or now < state.valid_until)):
            continue
        if state.memory_type != "PREFERENCE" and not state.concept.casefold().startswith(("likes for ", "favorite ")):
            continue
        messages = [m for m in store.fetch_messages_by_ids(state.source_ids)
                    if m.role == "user" and 0 <= m.timestamp <= now + 300]
        for msg in messages:
            if not _predicate_match(intent, msg.content, state.concept, state.value):
                continue
            if intent.subject and intent.subject.casefold() not in f"{state.concept} {state.value} {msg.content}".casefold():
                continue
            candidates.append({"id": msg.message_id, "timestamp": msg.timestamp,
                               "text": msg.content[:500], "concept": state.concept,
                               "value": state.value, "source_ids": state.source_ids,
                               "score": 1.0 + (0.25 if intent.subject else 0.0)})
    # Observations are an additive channel and obey the same source gates.
    try:
        observations = store.fetch_observations(scope="global") + store.fetch_observations(scope=session_id)
    except (AttributeError, RuntimeError):
        observations = []
    for observation in observations:
        allowed_observation_status = ("current", "superseded") if intent.temporal_scope == "historical" else ("current",)
        if observation.authority != "user" or observation.status not in allowed_observation_status or not observation.source_ids:
            continue
        if intent.temporal_scope != "historical" and observation.valid_from is not None and not (
            observation.valid_from <= now
            and (observation.valid_until is None or now < observation.valid_until)
        ):
            continue
        for msg in store.fetch_messages_by_ids(observation.source_ids):
            if msg.role != "user" or not _predicate_match(intent, msg.content, observation.concept, observation.value):
                continue
            if intent.subject and intent.subject.casefold() not in f"{observation.concept} {observation.value} {msg.content}".casefold():
                continue
            candidates.append({"id": msg.message_id, "timestamp": msg.timestamp,
                               "text": msg.content[:500], "concept": observation.concept,
                               "value": observation.value, "source_ids": observation.source_ids,
                               "score": 1.1})
    # Lexical-only evidence remains eligible when formation has not produced
    # a typed row.  The first-person gate prevents an assistant suggestion or
    # a third-party statement from becoming a personal preference.
    try:
        raw_messages = store.search_user_evidence(
            ("like", "love", "enjoy", "prefer", "favorite", "favourite"),
            limit=240,
        )
    except (AttributeError, RuntimeError):
        raw_messages = []
    for msg in raw_messages:
        if msg.role != "user" or not _EXPLICIT_USER_PREFERENCE.search(msg.content):
            continue
        if msg.message_id in invalid_preference_sources:
            continue
        if not _predicate_match(intent, msg.content, "", ""):
            continue
        if intent.subject and intent.subject.casefold() not in msg.content.casefold():
            continue
        candidates.append({"id": msg.message_id, "timestamp": msg.timestamp,
                           "text": msg.content[:500], "concept": "lexical preference",
                           "value": msg.content[:240], "source_ids": [msg.message_id],
                           "score": 0.5})
    # Deduplicate by canonical concept/value and shared source evidence.
    unique: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    for candidate in candidates:
        key = (normalize_concept(candidate["concept"]), normalize_concept(candidate["value"]),
               tuple(sorted(set(candidate["source_ids"]))))
        old = unique.get(key)
        if old is None or candidate["score"] > old["score"]:
            unique[key] = candidate
    candidates = list(unique.values())
    # A typed row and its lexical source are the same evidence packet when
    # they point at the same message. Keep the typed representation, which
    # carries the canonical concept/value, and discard the duplicate.
    deduped: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: row["score"], reverse=True):
        source_set = set(candidate["source_ids"])
        if any(source_set & set(existing["source_ids"]) for existing in deduped):
            continue
        deduped.append(candidate)
    candidates = deduped
    candidates.sort(key=lambda row: (row["timestamp"], row["id"]), reverse=intent.order != "oldest")
    if intent.count is not None:
        candidates = candidates[:intent.count]
    if not candidates:
        return ""
    lines = [
        "Verified preference evidence. Only direct user-source records are included.",
        f"Predicate: {intent.predicate}; order: {intent.order}; requested count: {intent.count or 'all'}.",
    ]
    for row in candidates:
        stamp = datetime.fromtimestamp(row["timestamp"], timezone.utc)
        source_ids = ",".join(str(i) for i in row["source_ids"])
        lines.append(f"[message {row['id']}; source_ids={source_ids}; {stamp:%Y-%m-%d %H:%M} UTC] {row['text']!r} | stored: {row['concept']}: {row['value']}")
    return "\n".join(lines)

