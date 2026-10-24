"""Conservative cross-turn reference resolution, independent of subject domain.

Only user statements may supply referents. A reference without a unique,
explicit name is never promoted to long-term personal memory.
"""
import re
from typing import Optional

_PRONOUN_PREFERENCE = re.compile(
    r"^i\s+(?:really\s+)?(?P<verb>love|like|adore|hate|dislike)\s+"
    r"(?:it|that|them)(?:\s+too)?[.!]?\s*$", re.IGNORECASE,
)
# These are grammatical *naming* constructions, not a vocabulary of music,
# products, projects, locations, or other allowed subjects.
_NAMING = re.compile(
    r"\b(?:called|named|titled|which\s+is)\s+"
    r"(?P<referent>[\w][\w '\-]{0,75})\s*[.!]?\s*$", re.IGNORECASE,
)
_INVALID = frozenset({"it", "that", "them", "something", "anything", "everything", "this"})


def is_bare_reference_preference(text: str) -> bool:
    return bool(_PRONOUN_PREFERENCE.fullmatch((text or '').strip()))


def resolve_preference_reference(text: str, previous_user: str) -> Optional[str]:
    """Return a self-contained USER preference or None if uncertain.

    This does not use assistant text, arbitrary previous sessions, or an LLM.
    The caller must also ensure the turns are adjacent and retain both sources.
    """
    match = _PRONOUN_PREFERENCE.fullmatch((text or '').strip())
    if match is None or not previous_user:
        return None
    names = list(_NAMING.finditer(previous_user.strip()))
    if len(names) != 1:
        return None
    name = " ".join(names[0].group("referent").split()).strip(" .!\"')")
    if not name or name.casefold() in _INVALID or len(name.split()) > 8:
        return None
    # Multiple candidate objects in a list are ambiguous.
    if re.search(r"\b(?:and|or)\b|[,;:]", name, re.I):
        return None
    return f"I {match.group('verb')} {name}"
