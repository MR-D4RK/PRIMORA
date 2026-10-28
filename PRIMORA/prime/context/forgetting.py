"""Durable topic suppression for PRIMORA's append-only memory.

This is retrieval suppression, NOT physical erasure. Raw SQLite rows, FTS,
backups, and exported copies retain their original bytes. Suppression is
stored in the same DB and checked on every model-facing read.
"""
from __future__ import annotations

import re


def normalize_topic(topic: str) -> str:
    if not isinstance(topic, str):
        raise ValueError("topic must be text")
    topic = " ".join(topic.strip(" \t\r\n.?!'\"").split()).casefold()
    if not (2 <= len(topic) <= 80) or len(topic.split()) > 7:
        raise ValueError("topic must be a specific name or short phrase")
    if not re.fullmatch(r"[\w][\w '\-]*[\w]", topic, re.UNICODE):
        raise ValueError("topic must contain ordinary words")
    if topic in {"me", "you", "everything", "anything", "all",
                 "this", "that", "something", "memory", "memories"}:
        raise ValueError("please name a specific topic")
    return topic


def matches_topic(text: str, topic: str) -> bool:
    """Whole-word phrase match, without subject-specific keyword tables."""
    words = re.findall(r"\w+", topic, flags=re.UNICODE)
    if not words:
        return False
    expression = r"(?<!\w)" + r"[\W_]+".join(
        re.escape(word) for word in words) + r"(?!\w)"
    return bool(re.search(expression, str(text or ""), re.IGNORECASE))


_FORGET = re.compile(
    r"^\s*(?:hey\s+aurora\s*[,!.]?\s*)?"
    r"(?:please\s+)?forget\s+"
    r"(?:(?:everything|anything|all|what\s+you\s+know)\s+)?"
    r"(?:you\s+know\s+)?about\s+"
    r"(?P<topic>.+?)\s*[.?!]*\s*$",
    re.IGNORECASE,
)


def parse_forget_request(text: str) -> str | None:
    """Only explicit commands; questions and negations are not actions."""
    match = _FORGET.fullmatch(str(text or ""))
    if match is None:
        return None
    try:
        return normalize_topic(match.group("topic"))
    except ValueError:
        return None
