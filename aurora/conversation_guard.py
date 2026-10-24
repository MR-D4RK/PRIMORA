"""Narrow conversation guards for observed local-model mistakes.

These rules do not interpret general user intent or replace Aurora's model.
They only catch two concrete failures: outside research for an unambiguous
personal-preference statement, and an unchanged answer after a correction.
"""

from __future__ import annotations

import re
from typing import Any


_PREFERENCE_STATEMENT = re.compile(
    r"^(?:i\s+(?:(?:really|also|just)\s+)?(?:love|like|adore|hate|"
    r"dislike|prefer)\b|(?:no[,!]?\s+)?(?:the feeling is|what i meant is)\b)",
    re.IGNORECASE,
)
_EXPLICIT_LOOKUP = re.compile(
    r"\b(?:research|search|look up|find out|google|browse|check online|"
    r"look into)\b", re.IGNORECASE,
)
_ACCOUNTABILITY_CORRECTION = re.compile(
    r"\b(?:your assumptions?|you (?:assumed|invented|made (?:that|it) up)|"
    r"you were (?:assuming|guessing))\b", re.IGNORECASE,
)
_DENIES_ASSUMPTION = re.compile(
    r"\b(?:i (?:did not|didn't|never) (?:assume|guess|invent)|"
    r"i (?:was not|wasn't) (?:assuming|guessing))\b", re.IGNORECASE,
)

_CORRECTION = re.compile(
    r"^(?:no\b|nope\b|actually\b|not quite\b|i meant\b|"
    r"what i meant\b|that's not what i meant\b)", re.IGNORECASE,
)


def latest_user_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def reject_unrequested_lookup(request: Any, user_text: str) -> str | None:
    """Refuse a *research* request triggered solely by a personal statement.

    A user who explicitly asks for a lookup can always trigger research.
    Other actions (including approved memory tools) remain untouched.
    """
    action = getattr(request, "action", None)
    if action not in ("research", "research_read"):
        return None
    # A user's private conversational history cannot be discovered by an
    # external research search. Keep explicit research requests allowed.
    if (re.search(r"\b(?:when\s+did\s+i|what\s+did\s+i|"
                  r"what\s+have\s+i)\s+(?:first\s+)?"
                  r"(?:tell|say|mention|told|said|mentioned)\b", user_text,
                  re.I) and not _EXPLICIT_LOOKUP.search(user_text)):
        return ("not_requested: the user asked about their own statements. "
                "Use internal memory only, do not research the web.")
    if getattr(request, "action", None) != "research":
        return None
    statement = user_text.strip()
    if re.fullmatch(r"\s*(?:just\s+)?retrieve\s+it\s+(?:from|in)\s+"
                    r"(?:your\s+)?memory[.!?]*\s*", user_text, re.I):
        return "not_requested: use stored memory, not outside research."
    if (_PREFERENCE_STATEMENT.search(statement)
            and not _EXPLICIT_LOOKUP.search(statement)
            and "?" not in statement):
        return (
            "not_requested: the user expressed or clarified a personal "
            "preference, and did not ask for outside research. Do not search "
            "for an explanation of their feelings. Respond to their latest "
            "statement directly, without attributing unspoken reasons."
        )
    return None


def repeats_answer_after_correction(
    original_messages: list[dict[str, str]], new_answer: str
) -> bool:
    """An exact replay of the preceding answer cannot satisfy a correction.

    Intentionally exact after whitespace normalization: similar answers or
    ordinary follow-up questions are left to Aurora, not heuristics.
    """
    user_text = latest_user_text(original_messages)
    if not _CORRECTION.search(user_text.strip()):
        return False
    earlier = next((str(msg.get("content", "")) for msg in reversed(
        original_messages) if msg.get("role") == "assistant"), "")
    normalize = lambda s: " ".join(s.split()).casefold()
    return bool(earlier.strip() and new_answer.strip()
                and normalize(earlier) == normalize(new_answer))


def denies_assumption_after_correction(
    original_messages: list[dict[str, str]], new_answer: str
) -> bool:
    """Flag a narrow dishonest denial when the user names our assumption.

    The model still composes its own acknowledgement after bounded retry.
    This guard never infers who was right about the underlying fact.
    """
    user_text = latest_user_text(original_messages)
    return bool(_ACCOUNTABILITY_CORRECTION.search(user_text)
                and _DENIES_ASSUMPTION.search(new_answer))
