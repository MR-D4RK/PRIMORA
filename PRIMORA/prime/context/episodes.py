"""Conservative, domain-independent capture of explicitly reported experiences.

An episode event is an attributable *report of an action*, not independent
proof that it happened. This is intentionally a narrow Batch 1 extractor:
no semantic guessing, model summarization, or subject-specific vocabulary.
Later batches can introduce richer episode organization without changing the
immutable source evidence contract.
"""

from dataclasses import dataclass
from datetime import datetime, time as datetime_time, timedelta, timezone
from hashlib import sha256
import os
import re
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .models import EpisodeRecord, ExperienceEvent


@dataclass(frozen=True)
class ExperienceClaim:
    kind: str
    actor: str
    statement: str
    occurred_at: Optional[float] = None
    occurred_precision: str = "unknown"
    occurred_timezone: str = "UTC"


# Category tags describe the action, not the subject.  "We finished the
# remix" and "We finished the UI" take exactly the same path.
_ACTIONS = {
    "decided": "decision", "chose": "decision", "agreed": "decision",
    "finished": "completion", "completed": "completion",
    "fixed": "resolution", "resolved": "resolution", "solved": "resolution",
    "started": "start", "began": "start",
    "tried": "attempt", "tested": "attempt", "attempted": "attempt",
    "worked on": "activity", "discussed": "discussion",
    "launched": "milestone", "published": "milestone",
    "stopped": "end", "cancelled": "end", "canceled": "end",
    "failed": "setback",
    # Past-tense reports of activities are episodes, even when the user
    # does not claim to have finished an entire book, series, or project.
    "watched": "activity", "visited": "activity",
    "attended": "activity", "read": "activity",
    "listened to": "activity", "played": "activity",
}

_ACTION_RE = "|".join(re.escape(word) for word in sorted(_ACTIONS, key=len, reverse=True))
_CLAIM_RE = re.compile(
    rf"^\s*(?:(?P<relative>today|yesterday|earlier today|last night|recently|just)\s*[,;:]?\s+)?"
    rf"(?P<actor>I|we)\s+(?:(?:have|had)\s+)?(?:(?:recently|just)\s+)?"
    rf"(?P<action>{_ACTION_RE})\b(?:\s+(?P<object>.+?))?"
    rf"(?:\s+(?P<relative_after>today|yesterday|earlier today|last night|recently|just))?$",
    re.IGNORECASE,
)


def _resolve_relative_date(
    relative: Optional[str], reported_at: Optional[float],
    timezone_name: Optional[str] = None,
) -> tuple[Optional[float], str, str]:
    """Resolve explicit relative dates at local midnight, retaining day precision."""
    if not relative or reported_at is None:
        return None, "unknown", timezone_name or "UTC"
    zone_name = timezone_name or os.environ.get("PRIMORA_USER_TIMEZONE", "UTC")
    try:
        zone = ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, ValueError):
        zone_name = "UTC"
        zone = timezone.utc
    local_reported = datetime.fromtimestamp(float(reported_at), zone)
    relative = relative.casefold()
    if relative in {"yesterday", "last night"}:
        date = local_reported.date() - timedelta(days=1)
    else:
        date = local_reported.date()
    local_midnight = datetime.combine(date, datetime_time.min, tzinfo=zone)
    return local_midnight.timestamp(), "day", zone_name


def extract_experience_claim(
    text: str, role: str = "user", *, reported_at: Optional[float] = None,
    timezone_name: Optional[str] = None,
) -> Optional[ExperienceClaim]:
    """Return one supported, explicit first-person event or None.

    Only a new user message can create a claim. Quoted instructions, questions,
    future plans, negations, assistant turns and vague statements are excluded.
    The verbatim statement remains the evidence; no cause or outcome is inferred.
    """
    if role != "user" or not isinstance(text, str):
        return None
    statement = text.strip()
    if not statement or len(statement) > 1500 or "?" in statement:
        return None
    # Avoid extracting from pasted multi-line materials, quotations or code.
    if "\n" in statement or statement.startswith(("\"", "'", "`", ">", "- ")):
        return None
    match = _CLAIM_RE.fullmatch(statement.rstrip(".! "))
    if not match:
        return None
    obj = (match.group("object") or "").strip()
    if len(obj) < 3 or not re.search(r"[\w]", obj):
        return None
    # Never infer that a quoted or conditional future task was completed.
    if obj.casefold().startswith(("if ", "maybe ", "perhaps ", "not ", "never ")):
        return None
    occurred_at, occurred_precision, occurred_timezone = _resolve_relative_date(
        match.group("relative") or match.group("relative_after"),
        reported_at, timezone_name
    )
    return ExperienceClaim(
        kind=_ACTIONS[match.group("action").lower()],
        actor=match.group("actor").lower(),
        statement=statement,
        occurred_at=occurred_at,
        occurred_precision=occurred_precision,
        occurred_timezone=occurred_timezone,
    )


# Episode organization is a conservative *view* over source-backed events.
# Identical activity objects are grouped; we do not infer two differently
# named projects are the same, or that completing one caused another event.
_LEADING_DETERMINERS = frozenset({
    "a", "an", "the", "my", "our", "this", "that", "these", "those",
})
_OBJECT_WORDS = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def experience_subject(statement: str) -> Optional[tuple[str, str]]:
    """Extract a subject label and conservative, domain-independent key.

    A simple exact normalized object match is safer than merging episodes on
    shared tokens or assumed meanings. More sophisticated entity linking is
    intentionally deferred until source-backed relationship modeling.
    """
    match = _CLAIM_RE.fullmatch(statement.strip().rstrip(".! "))
    if match is None:
        return None
    obj = (match.group("object") or "").strip()
    words = _OBJECT_WORDS.findall(obj.casefold())
    # Strip a common grammatical activity wrapper, never topical words.
    # "started working on a remix" and "tested the remix" can share a
    # concrete subject without any music-specific extraction rule.
    if words[:2] in (["working", "on"], ["work", "on"]):
        words = words[2:]
    elif words[:3] == ["to", "work", "on"]:
        words = words[3:]
    while words and words[0] in _LEADING_DETERMINERS:
        words.pop(0)
    # Single pronouns and deictic references cannot identify a subject.
    if not words or (len(words) == 1 and words[0] in {
        "it", "them", "him", "her", "something", "anything", "everything",
    }):
        return None
    # Generic grammatical normalization, not per-domain aliases/stems.
    key = " ".join(words)
    return (" ".join(words), key)


def organize_experience_events(
    events: list[ExperienceEvent], *, max_gap_days: int = 45,
) -> list[EpisodeRecord]:
    """Organize verified user event reports into bounded evidence groups.

    Each event remains unchanged. Only *exact* normalized subjects are merged,
    and a gap greater than max_gap_days splits separate episodes. A missing
    or ambiguous subject stays as a one-event episode. Dates represent message
    reporting times, not necessarily event occurrence times.
    """
    if not 1 <= max_gap_days <= 365:
        raise ValueError("max_gap_days must be 1..365")
    gap_seconds = max_gap_days * 86400
    groups: list[tuple[str, str, list[ExperienceEvent]]] = []
    last_by_key: dict[str, int] = {}
    seen: set[int] = set()
    for event in sorted(events, key=lambda e: (e.reported_at, e.source_message_id)):
        if event.source_message_id in seen or event.authority != "user":
            continue
        claim = extract_experience_claim(event.statement)
        if claim is None or claim.kind != event.kind or claim.actor != event.actor:
            continue
        seen.add(event.source_message_id)
        subject = experience_subject(event.statement)
        label, key = subject if subject else (event.statement, "")
        index = last_by_key.get(key) if key else None
        if index is None or (event.reported_at - groups[index][2][-1].reported_at) > gap_seconds:
            groups.append((label, key, [event]))
            if key:
                last_by_key[key] = len(groups) - 1
        else:
            groups[index][2].append(event)

    result = []
    for label, key, members in groups:
        original = members[0]
        # Identifier is a stable function of the current earliest verified
        # message, not a separately invented persistent episode entity.
        ident = sha256(f"{key}\0{original.source_message_id}".encode()).hexdigest()[:16]
        result.append(EpisodeRecord(
            episode_id=f"observed-{ident}",
            subject=label,
            events=tuple(members),
            source_message_ids=tuple(e.source_message_id for e in members),
            session_ids=tuple(dict.fromkeys(e.session_id for e in members)),
            first_reported_at=original.reported_at,
            last_reported_at=members[-1].reported_at,
        ))
    return sorted(result, key=lambda g: (g.last_reported_at, g.episode_id), reverse=True)
