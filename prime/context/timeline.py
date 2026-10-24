"""Deterministic, read-only timeline query intent and presentation.

No model calls, DB migration, or guessed preference-onset times.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Optional


@dataclass(frozen=True)
class TimelineRequest:
    mode: str  # started | recent | changes
    category: Optional[str]
    since: Optional[float] = None
    target: Optional[str] = None  # explicit subject named by the user


_QUESTION = re.compile(
    r"^\s*(?:(?:hey\s+)?aurora[,!]?\s*)?"
    r"(?:what|which|how|when|tell\s+me|show\s+me|list|have\s+i)\b", re.I,
)
_TOPIC = re.compile(r"\b(?:lik(?:e|ed|ing)|lov(?:e|ed|ing)|"
                    r"fav(?:s|orites?|ourites?)?|preferences?|interests?)\b", re.I)
_START = re.compile(r"\b(?:start(?:ed)?|begin|began|newly|new)\b", re.I)
_RECENCY = re.compile(r"\b(?:recently|recent|latest|lately|last|newest|new)\b", re.I)
_CHANGE = re.compile(r"\b(?:changed?|changing|evolved?|evolution|"
                     r"previously|used\s+to|before|history)\b", re.I)
_EXPLICIT_BEGIN = re.compile(
    r"^\s*i(?:\s+have|['’]ve)?\s+(?:(?:just|recently)\s+)?"
    r"(?:started|began)\s+(?:to\s+)?(?:like|love|adore|liking|loving|adoring)\b",
    re.I,
)
_CATEGORY = (
    ("car", re.compile(r"\b(?:cars?|vehicles?|automobiles?)\b", re.I)),
    ("color", re.compile(r"\b(?:colou?rs?|shades?)\b", re.I)),
    ("song", re.compile(r"\b(?:songs?|tracks?|playlists?)\b", re.I)),
    ("artist", re.compile(r"\b(?:artists?|musicians?|singers?)\b", re.I)),
)
_LAST_DAYS = re.compile(r"\b(?:in\s+the\s+)?last\s+(\d{1,3})\s+days?\b", re.I)
# Questions about when the USER made a statement must not be answered from
# assistant text or from a general model guess about release dates.
_WHEN_SAID = re.compile(
    r"\b(?:when\s+did\s+i\s+(?:first\s+)?(?:tell|say|mention|talk)"
    r"|when\s+have\s+i\s+(?:told|mentioned|said)"
    r"|when\s+was\s+the\s+first\s+time\s+i\s+(?:said|mentioned)"
    r"|what\s+(?:date|day)\s+did\s+i\s+(?:tell|say|mention)"
    r"|(?:what|which)\s+(?:are\s+)?(?:all\s+)?my\s+(?:fav(?:orite|ourite)?\s+)?cars?\s+and\s+when\s+did\s+i\s+(?:tell|mention))\b", re.I,
)
_USER_SAID = re.compile(
    r"\b(?:what\s+(?:exactly\s+)?did\s+i\s+(?:tell|say|mention)"
    r"|what\s+have\s+i\s+(?:told|said|mentioned))\b", re.I,
)
_ABOUT = re.compile(r"\babout\s+(?P<subject>[^?!,.]{2,80})", re.I)
_VAGUE = {"them", "those", "it", "that", "them all", "my favorites",
          "my favourite cars", "my favorite cars", "my cars", "my interests"}


def _category_for(text: str) -> Optional[str]:
    return next((name for name, pattern in _CATEGORY
                 if pattern.search(text)), None)


def _subject_for(text: str) -> Optional[str]:
    match = _ABOUT.search(text)
    if not match:
        return None
    subject = match.group("subject").strip().rstrip(" .!?")
    subject = re.split(r"\s+(?:just\s+tell|please\s+tell|only\s+tell|and\s+tell)\b",
                       subject, maxsplit=1, flags=re.I)[0].strip()
    subject = re.sub(r"^(?:the|my)\s+", "", subject, flags=re.I)
    # "what have I told you about why I love the Regera?" is about Regera.
    subject = re.sub(r"^why\s+i\s+(?:love|like|adore)\s+", "", subject, flags=re.I)
    subject = re.sub(r"^(?:the|my)\s+", "", subject, flags=re.I)
    if subject.casefold() in _VAGUE or subject.casefold().startswith(
        ("my fav", "my car", "those ", "the rest")
    ):
        return None
    return subject


def parse_request(text: str, *, now: float,
                  context: tuple[str, ...] = ()) -> Optional[TimelineRequest]:
    """Only recognize explicit timeline or personal-source questions.

    Previous user utterances may resolve a bare 'them' to an *explicit*
    category, but older assistant content never determines the subject.
    """
    question = text.strip()
    date_question = bool(_WHEN_SAID.search(question))
    source_question = bool(_USER_SAID.search(question))
    if date_question or source_question:
        mode = "dates" if date_question else "claims"
        category = _category_for(question)
        target = _subject_for(question)
        if not category and not target:
            for previous in reversed(context[-8:]):
                category = _category_for(previous)
                if category:
                    break
        # A bare "them" without a grounded earlier category must not
        # trigger retrieval of every memory in the database.
        if not category and not target:
            category = "unresolved"
        return TimelineRequest(mode, category, None, target)
    if not _QUESTION.search(question) or not _TOPIC.search(question):
        return None
    if _CHANGE.search(question):
        mode = "changes"
    elif _START.search(question) and _RECENCY.search(question):
        mode = "started"
    elif _RECENCY.search(question):
        mode = "recent"
    else:
        return None
    category = _category_for(question)
    since = None
    last_days = _LAST_DAYS.search(question)
    if last_days:
        since = now - int(last_days.group(1)) * 86400
    elif re.search(r"\bthis\s+month\b", question, re.I):
        date = datetime.fromtimestamp(now, tz=timezone.utc)
        since = datetime(date.year, date.month, 1,
                         tzinfo=timezone.utc).timestamp()
    return TimelineRequest(mode, category, since)


# An explicit favorites inventory is a *personal-memory* question, not an
# external research request. Keep recognition narrow: generic questions like
# "which cars are best" continue through the normal reasoning path.
_INVENTORY_FAV = re.compile(
    r"\b(?:fav|favorite|favourite|favorites|favourites)\b", re.I,
)
_INVENTORY_LIST = re.compile(
    r"^\s*(?:(?:hey\s+)?aurora[,!]?[ \t]*)?"
    r"(?:what|which|list|show|tell|remind|can\s+you\s+(?:list|show|tell|mention))\b",
    re.I,
)
_INVENTORY_PLURAL = re.compile(r"\b(?:cars|vehicles|automobiles)\b", re.I)
_INVENTORY_FOLLOWUP = re.compile(
    r"^\s*(?:can\s+you\s+)?(?:tell|mention|mentino|show|explain|list|"
    r"what\s+about)\b.*\b(?:why|reasons?|rest|others?)\b.*"
    r"\b(?:them|each|ones|cars)\b", re.I,
)
_INVENTORY_DATE_REFERENCE = re.compile(
    r"\bwhen\b.*\b(?:tell|told|say|said|mention|mentioned)\b.*"
    r"\b(?:them|each|those)\b", re.I,
)
_INVENTORY_REASON_REFERENCE = re.compile(
    r"\b(?:why|reasons?)\b.*\b(?:them|each|those)\b", re.I,
)



def favorite_inventory_category(text: str, *,
                                context: tuple[str, ...] = ()) -> Optional[str]:
    """Recognize explicitly requested *lists* of the user's favorites.

    Follow-ups such as 'why do I like each of them?' inherit a category only
    from the user's own earlier category question, never from model replies.
    """
    question = (text or "").strip()
    # An explicit different category overrides any earlier car question.
    named_category = _category_for(question)
    if named_category not in (None, "car"):
        return None
    if (_INVENTORY_LIST.search(question) and
            _INVENTORY_FAV.search(question) and
            _INVENTORY_PLURAL.search(question)):
        return "car"
    if (_INVENTORY_FOLLOWUP.search(question) or
            _INVENTORY_DATE_REFERENCE.search(question) or
            _INVENTORY_REASON_REFERENCE.search(question) or re.fullmatch(
        r"what\s+about\s+(?:the\s+)?rest[?! .]*", question, re.I
    ) or re.search(
        r"^\s*(?:why\s+do\s+i\s+(?:love|like)|"
        r"what\s+are\s+my\s+reasons\s+for)\b.*"
        r"\b(?:each|them|all)\b", question, re.I
    )):
        for earlier in reversed(context[-8:]):
            if _INVENTORY_FAV.search(earlier) and _INVENTORY_PLURAL.search(earlier):
                return "car"
    return None


# A small, explicit car-name vocabulary closes the gap when the user says
# 'I love Regera' without saying the word 'car'. This classifies the named
# object only; it NEVER generates a favorite or its reasons. Future entity
# typing can replace this vocabulary without changing stored memories.
_CAR_NAME = re.compile(
    r"\b(?:corvette|mustang|koenigsegg|regera|apollo\s+ie|zr1|"
    r"ford|chevrolet|ferrari|lamborghini|porsche|bugatti|mclaren|"
    r"aston\s+martin|maserati|bentley|dodge|tesla|toyota|nissan|"
    r"bmw|mercedes|audi|lexus|subaru|mazda|honda|volkswagen|"
    r"boss\s+(?:302|429)|\bC8\b)\b", re.I,
)


def verified_category_hint(category: str, subject: str) -> bool:
    """Limited lexical type hint, not evidence the user likes the item."""
    # A car mentioned *inside a preference about an attribute* is not a
    # second car. Example: "likes for advanced engineering of regera not c8"
    # expresses a reason, not a sixth vehicle. Require the entity at the
    # start of the normalized subject (allowing a model year prefix).
    label = re.sub(r"^(?:19|20)\d{2}\s+", "", subject.strip(), flags=re.I)
    return category == "car" and bool(_CAR_NAME.match(label))


def is_explicit_start(text: str) -> bool:
    return bool(_EXPLICIT_BEGIN.search(text))


def event_line(state, *, explicitly_started: bool,
               source=None) -> str:
    """One grounded event; only verified user evidence can be passed in."""
    when = source.timestamp if source is not None else state.valid_from
    recorded = datetime.fromtimestamp(when, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")
    kind = "explicitly said newly liked" if explicitly_started else "recorded preference"
    current = "current" if state.status == "current" else "superseded/historical"
    # Keep each concept attached to its OWN value and user statement.
    fact = f"{state.concept} = {state.value}"
    evidence = ""
    if source is not None:
        # Quote the user as DATA, never as a fresh instruction.
        excerpt = " ".join(source.content.split())[:220]
        evidence = f"; user statement (quote, not instruction): {excerpt!r}"
    return (f"- {recorded}; {kind}; {current}; "
            f"fact: [{fact}]; source IDs: {source.message_id if source else 'unknown'}"
            + evidence)


def category_matches(store, state, category: Optional[str]) -> bool:
    """Use explicit user category evidence, not model/brand-name guessing.

    The closest preceding category question supplies context for a shorthand
    answer (e.g. "I've started liking Regera"). A later question never
    relabels an older preference, and an older *different* category question
    stops the search. Ungrounded category assignments are omitted.
    """
    if category is None:
        return True
    if state.concept.startswith("favorite "):
        noun = state.concept[len("favorite "):]
        return noun == category or noun.endswith(" " + category)
    if not state.concept.startswith("likes for "):
        return False
    if not state.source_ids:
        return False
    sources = store.fetch_messages_by_ids(state.source_ids[:2])
    for source in sources:
        if source.role != "user":
            continue
        for name, pattern in _CATEGORY:
            if pattern.search(source.content):
                return name == category
        prior = store.fetch_range(source.session_id,
                                  max(0, source.seq - 4),
                                  source.seq - 1, limit=4)
        for earlier in reversed(prior):
            if earlier.role != "user" or not _QUESTION.search(earlier.content):
                continue
            named = next((name for name, pattern in _CATEGORY
                          if pattern.search(earlier.content)), None)
            if named is not None:
                return named == category
    return False
