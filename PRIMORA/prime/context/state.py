"""Deterministic user-state extraction and transitions. No LLM.

Only user-authored messages can create or change current state.
Assistant statements never become current state on their own; an
assistant claim that contradicts tracked current state is recorded as
`unresolved` for visibility, never as truth.

Raw messages are never modified. Supersession links rows; old rows stay.
"""

import hashlib
import re
import time
from dataclasses import dataclass, replace as dataclass_replace
from typing import Any, Optional

from . import slots as slots_mod

# Memory types for deterministic M1 formation.
FACT = "FACT"
PREFERENCE = "PREFERENCE"
DECISION = "DECISION"
INSTRUCTION = "INSTRUCTION"
RATIONALE = "RATIONALE"
UNCERTAIN = "UNCERTAIN"

# Words that name a memory type inside correction phrasing, longest first
# so "preference" wins over its substring "prefer".
_TYPE_CUES = (
    ("instruction", INSTRUCTION),
    ("preference", PREFERENCE),
    ("decided", DECISION),
    ("decision", DECISION),
    ("prefer", PREFERENCE),
    ("decide", DECISION),
    ("like", PREFERENCE),
    ("love", PREFERENCE),
    ("rule", INSTRUCTION),
    ("fact", FACT),
)

# Bare type words (lowercase, as produced by normalize_concept).
_TYPE_WORDS = frozenset({"preference", "decision", "instruction", "fact"})

_TYPE_BY_WORD = {"preference": PREFERENCE, "decision": DECISION,
                 "instruction": INSTRUCTION, "fact": FACT}
_WORD_BY_TYPE = {PREFERENCE: "preference", DECISION: "decision",
                 INSTRUCTION: "instruction", FACT: "fact"}

# A type-word statement ("I prefer dark mode") becomes its own fact under a
# topic-bearing concept derived from its value, the way Mem0 keys each
# extracted fact by content instead of one generic slot per type. Long
# values are capped; a digest keeps two long values from collapsing back
# into one slot.
_TOPIC_CONCEPT_MAX = 80

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

# --- M1 deterministic formation patterns (all anchored, user-only) ---
_CORRECTION_PREFIX_RE = re.compile(
    r"^(?:actually|well|no|correction|i\s+said)[,:]?\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_THATS_WRONG_RE = re.compile(
    r"^that'?s wrong[,:]?\s*(?:—|-)?\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_PREFER_RE = re.compile(
    r"^i\s+(?:really\s+)?(?:prefer|like|love)\s+(.+?)\.?$",
    re.IGNORECASE | re.DOTALL,
)
_NEG_PREFER_RE = re.compile(
    r"^i\s+do\s+not\s+(?:like|love|prefer|use)\s+(.+?)(?:\s+anymore)?\.?$"
    r"|^i\s+don'?t\s+(?:like|love|prefer|use)\s+(.+?)(?:\s+anymore)?\.?$",
    re.IGNORECASE | re.DOTALL,
)
_DECIDE_RE = re.compile(
    r"^we\s+(?:decided|chose|choose|have\s+decided|have\s+chosen|kept)\s+"
    r"(?:to\s+|that\s+|on\s+)?(.+?)\.?$",
    re.IGNORECASE | re.DOTALL,
)
_KEEPING_RE = re.compile(
    r"^we\s+(?:are|were)\s+keeping\s+(.+?)\.?$",
    re.IGNORECASE | re.DOTALL,
)
_LETS_KEEP_RE = re.compile(
    r"^let'?s\s+keep\s+(.+?)\s+as\s+(.+?)\.?$",
    re.IGNORECASE | re.DOTALL,
)
_DECIDE_NEG_RE = re.compile(
    r"^we'?ve\s+decided\s+not\s+to\s+use\s+(.+?)\.?$",
    re.IGNORECASE | re.DOTALL,
)
_ALWAYS_RE = re.compile(
    r"^always\s+(.+?)\.?$", re.IGNORECASE | re.DOTALL,
)
_NEVER_RE = re.compile(
    r"^never\s+(.+?)\.?$", re.IGNORECASE | re.DOTALL,
)
_DONT_RE = re.compile(
    r"^(?:don't|do not)\s+(.+?)\s+anymore\.?$",
    re.IGNORECASE | re.DOTALL,
)
_WE_CHANGED_RE = re.compile(
    r"^we\s+changed(?:\s+the)?\s+(.+?)\s+to\s+(.+?)\.?$",
    re.IGNORECASE | re.DOTALL,
)
_HEDGE_RE = re.compile(
    r"\b(maybe|perhaps|might|could|probably|possibly|likely)\b"
    r"|\bi\s+(?:think|guess|suppose)\b",
    re.IGNORECASE,
)
_HEDGE_PREFIX_RE = re.compile(
    r"^(?:maybe|perhaps|i\s+think(?:\s+that)?|i\s+guess|i\s+suppose)"
    r"[,:]?\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_NOW_SUFFIX_RE = re.compile(r"\s+now\.?$", re.IGNORECASE)

# Normalize only explicit *shade-independent* likes. Without this, each
# natural paraphrase becomes a different concept ("likes for purple whether
# ..."), so Aurora has multiple overlapping current preferences.
# Deliberately anchored: song titles and unrelated trailing clauses must not
# be silently truncated into a different assertion.
_ALL_SHADES_RE = re.compile(
    r"^i\s+(?:really\s+)?(?:love|like|adore)\s+"
    r"(?P<object>[a-z][a-z0-9 '\-]{0,75}?)\s+"
    r"(?:regardless\s+of\s+(?:its|the)\s+shade|"
    r"no\s+matter\s+(?:what|which)\s+shade|"
    r"whether\s+it(?:'s|\s+is|s)\s+bright\s+or\s+not)"
    r"(?:,\s*and\s+it\s+gives\s+me\s+a\s+similar\s+feeling"
    r"\s+to\s+(?P<target>[a-z][a-z -]{1,35}))?"
    r"[.!]?$", re.IGNORECASE,
)


def _extract_all_shades(text: str) -> list["StateCandidate"]:
    """Canonicalize equivalent statements without guessing a preference."""
    match = _ALL_SHADES_RE.fullmatch(text.strip())
    if not match:
        return []
    obj = match.group("object").strip()
    # Reuse the slot registry, including its scope and provenance policy.
    base = _extract_slots("I love " + obj)
    if len(base) != 1 or base[0].slot != "likes":
        return []
    normalized = dataclass_replace(base[0], value=base[0].value + " in any shade")
    target = match.group("target")
    if not target:
        return [normalized]
    # Do not discard a trailing clause. Only split when an existing slot
    # explicitly supports the user's comparison, preserving both claims.
    relationship = _extract_slots(f"{obj} feels like {target} to me")
    if len(relationship) != 1 or relationship[0].slot != "color feeling":
        return []
    return [normalized, relationship[0]]

# In speech, punctuation is frequently missing: "I love Aurora I love
# every aspect of it, especially the design". The generic likes slot would
# otherwise treat the entire repeated clause as the *name of the thing*.
# Accept only a repeated FIRST-PERSON preference with a pronoun referring
# back to one clearly named subject. The remainder is preserved verbatim in
# its user source, not reinterpreted as a new fact or invented motive.
_REPEATED_SUBJECT_LIKE_RE = re.compile(
    r"^i\s+(?:really\s+)?(?P<verb>love|like|adore|enjoy)\s+"
    r"(?P<subject>[a-z][a-z0-9 '&\-]{1,74}?)"
    r"\s*[,;.]?\s+i\s+(?:really\s+)?(?:love|like|adore|enjoy)\s+"
    r"(?:(?:every|each)\s+(?:aspect|part|bit)\s+of|"
    r"all\s+(?:the\s+)?(?:aspects|parts|bits)\s+of)\s+"
    r"(?:it|them)\b(?P<details>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_REPEATED_SUBJECT_JUNK = re.compile(
    r"\b(?:i|you|we|they|he|she|my|your|our|their|his|her|"
    r"this|that|it|them|anything|everything|something|everyone|"
    r"because|whether|but|if|when|which|what)\b",
    re.IGNORECASE,
)


def _extract_repeated_subject_preference(text: str) -> list["StateCandidate"]:
    """Preserve the actual item rather than making its details part of its name.

    No domain-specific classification (music, games, books, etc.) is inferred.
    It remains an ordinary, source-backed liked-item preference; the entire
    text is retained as evidence for its explicitly described dimensions.
    """
    match = _REPEATED_SUBJECT_LIKE_RE.fullmatch(text.strip())
    if not match:
        return []
    subject = " ".join(match.group("subject").split()).strip(" .,!;")
    if (not subject or len(subject.split()) > 7
            or _REPEATED_SUBJECT_JUNK.search(subject)):
        return []
    first_clause = f"I {match.group('verb')} {subject}"
    candidates = _extract_slots(first_clause)
    if (len(candidates) != 1 or candidates[0].slot != "likes"
            or not candidates[0].auto_confirm):
        return []
    return candidates


# High-confidence natural-language car preferences that the generic likes
# slot cannot parse correctly. Keep these anchored: speculative noun/brand
# guessing would create false personal memories.
_MODEL_YEAR_CAR_RE = re.compile(
    r"^(?:hey\s+)?(?:aurora[,!]?\s+)?i\s+(?:really\s+)?"
    r"(?:love|like|adore)\s+(?P<car>ford\s+mustan(?:g|d))\s+"
    r"regardless\s+of\s+(?:(?:their|its|the)\s+)?"
    r"(?:production|produced|producted|model)\s+years?|years?"
    r"[.!]?$", re.IGNORECASE,
)
_CLASSIC_FAVORITE_RE = re.compile(
    r"^(?:(?:for\s+instance|for\s+example)[, ]+)?"
    r"(?:the\s+)?(?P<car>\d{4}\s+[a-z][a-z0-9 -]{1,42}?)\s+"
    r"is\s+my\s+(?:fav|favorite|favourite)\s+classic\s+car[.!]?$",
    re.IGNORECASE,
)


def _extract_explicit_car_preferences(text: str) -> list["StateCandidate"]:
    """Only explicit, grounded statements; never infer a particular Boss model."""
    statement = text.strip()
    match = _MODEL_YEAR_CAR_RE.fullmatch(statement)
    if match:
        # A narrowly-scoped known typo correction. Preserve the user's exact
        # wording in raw history, but key the preference to the real model.
        car = "ford mustang"
        base = _extract_slots("I love " + car)
        if len(base) == 1 and base[0].slot == "likes":
            return [dataclass_replace(
                base[0], value="loves ford mustang across model years"
            )]
    match = _CLASSIC_FAVORITE_RE.fullmatch(statement)
    if match:
        model = " ".join(match.group("car").strip().split())
        return _extract_slots("my favorite classic car is " + model)
    return []


# M2 / Batch 3: explicit preference beginnings, endings, and "now"
# updates. They are recorded with the SOURCE MESSAGE observation time, not
# a fabricated date on which the user's taste actually began.
# Keep these patterns anchored and narrow: ambiguous past-tense remarks
# should not silently overwrite a current preference.
_NEW_LIKE_ALT_RE = re.compile(
    r"^i(?:\s+have|['’]ve)?\s+(?:(?:just|recently)\s+)?"
    r"(?:started|began)\s+(?:to\s+)?"
    r"(?P<verb>like|love|adore|liking|loving|adoring)\s+"
    r"(?P<obj>.+?)[.!]?$", re.IGNORECASE,
)
_NO_LONGER_LIKE_RE = re.compile(
    r"^i\s+(?:no\s+longer|don'?t\s+anymore)\s+"
    r"(?:like|love|adore)\s+(?P<obj>.+?)[.!]?$", re.IGNORECASE,
)
_STOPPED_LIKE_RE = re.compile(
    r"^i\s+(?:have\s+|'ve\s+)?"
    r"(?:stopped|quit)\s+(?:liking|loving|adoring)\s+"
    r"(?P<obj>.+?)[.!]?$", re.IGNORECASE,
)
_FAVORITE_NOW_RE = re.compile(
    r"^(?P<prefix>my\s+(?:fav|favorite|favourite)\s+"
    r"[a-z][a-z ]{0,30}?\s+is\s+)(?P<value>.+?)\s+now[.!]?$",
    re.IGNORECASE,
)


# A reason in the preceding USER statement can be connected to a named
# preference only by a direct follow-up. No assistant explanation is evidence.
_FOLLOWUP_WHY_LOVE_RE = re.compile(
    r"^(?:that['’]?s|that is)\s+why\s+i\s+(?:love|like|adore)\s+"
    r"(?P<object>(?:the\s+)?[a-z][a-z0-9 -]{1,65})[.!]?$", re.IGNORECASE,
)
_PRIOR_REASON_RE = re.compile(
    r"^(?:it|this|that)\s+(?:has|had|features|uses|is|was)\s+"
    r"[a-z0-9 ,'-]{10,150}[.!]?$", re.IGNORECASE,
)


def extract_followup_preference_reason(text: str, previous_user: str):
    """Return (preference, reason) only for a tightly linked user pair.

    Both statements must come from the user in the same session. The caller
    additionally checks recency and attaches both source-message IDs.
    """
    match = _FOLLOWUP_WHY_LOVE_RE.fullmatch((text or "").strip())
    reason = (previous_user or "").strip()
    if not match or not _PRIOR_REASON_RE.fullmatch(reason):
        return None
    candidates = _extract_slots("I love " + match.group("object"))
    if len(candidates) != 1 or candidates[0].slot != "likes":
        return None
    like = candidates[0]
    rationale = StateCandidate(
        concept="rationale for " + like.concept,
        value=reason.rstrip(".! "), scope="global", mtype=RATIONALE,
    )
    return like, rationale


def _extract_preference_evolution(text: str) -> list["StateCandidate"]:
    """Route high-confidence change language into existing slots.

    No invented onset timestamp, no implied deletion, and no new tables:
    normal successor/evidence operations retain the actual historical chain.
    """
    work = text.strip()
    match = _NEW_LIKE_ALT_RE.fullmatch(work)
    if match:
        obj = match.group("obj").strip().rstrip(".! ")
        if not obj or obj.lower() in ("it", "that", "them"):
            return []
        base = _extract_slots("I love " + obj)
        if len(base) != 1 or base[0].slot != "likes":
            return []
        # The generic likes slot historically merges 'like' and 'love'.
        # Preserve the stated intensity for *new* timeline entries.
        verb = match.group("verb").casefold()
        if verb in ("like", "liking"):
            return [dataclass_replace(
                base[0], value=re.sub(r"^loves\b", "likes", base[0].value)
            )]
        return base
    match = _NO_LONGER_LIKE_RE.fullmatch(work) or _STOPPED_LIKE_RE.fullmatch(work)
    if match:
        obj = match.group("obj").strip().rstrip(".! ")
        if not obj or obj.lower() in ("it", "that", "them"):
            return []
        base = _extract_slots("I don't like " + obj)
        return base if len(base) == 1 and base[0].slot == "likes" else []
    match = _FAVORITE_NOW_RE.fullmatch(work)
    if match:
        # "now" marks the correction, but is not part of the car/color name.
        base = _extract_slots(match.group("prefix") + match.group("value"))
        if len(base) == 1 and base[0].concept.startswith("favorite "):
            return [dataclass_replace(base[0], kind="correct")]
    return []


# Context-keyed choices have a stable topic independent of the option.
# These are general language structures, not item-specific facts.
_CONTEXT_FIRST_PREFERENCE_RE = re.compile(
    r"^for\s+(?P<context>[^,?.!]{2,90}),?\s+i\s+"
    r"(?P<now>now\s+)?prefer\s+(?P<choice>[^?.!]{2,130})[.!]?$",
    re.IGNORECASE,
)
_OPTION_FIRST_PREFERENCE_RE = re.compile(
    r"^i\s+(?P<now>now\s+)?prefer\s+"
    r"(?P<choice>[^?.!]{2,130}?)\s+for\s+"
    r"(?P<context>[^,?.!]{2,90})[.!]?$",
    re.IGNORECASE,
)
_EXPLICIT_CHANGE_PREFIX_RE = re.compile(
    r"^(?:i(?:'ve| have)\s+changed\s+my\s+mind|"
    r"i\s+changed\s+my\s+mind)[.!]\s*(?P<rest>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_OLD_CHOICE_REPORT_RE = re.compile(
    r"^(?:the\s+)?[a-z0-9][a-z0-9 '’-]{1,110}\s+"
    r"was\s+my\s+(?:old|previous)\s+choice[.!]?$",
    re.IGNORECASE,
)


def _extract_contextual_preference(text: str) -> list["StateCandidate"]:
    """Create one source-backed preference per explicitly named use context.

    The user must actually assert a choice and its context. We do not infer
    a replacement merely because an unrelated new option is mentioned.
    """
    statement = text.strip()
    explicit_change = False
    change = _EXPLICIT_CHANGE_PREFIX_RE.fullmatch(statement)
    if change:
        statement = change.group("rest").strip()
        explicit_change = True
    # An additional sentence is allowed only when it reports an old choice,
    # not when it adds an unexamined assertion to the current value.
    if "." in statement.rstrip("."):
        main, rest = statement.split(".", 1)
        if not _OLD_CHOICE_REPORT_RE.fullmatch(rest.strip()):
            return []
        statement = main.strip()
    match = (_CONTEXT_FIRST_PREFERENCE_RE.fullmatch(statement)
             or _OPTION_FIRST_PREFERENCE_RE.fullmatch(statement))
    if match is None:
        return []
    context = normalize_concept(match.group("context").strip())
    choice = _ARTICLES_RE.sub("", match.group("choice").strip()).strip()
    choice = choice.rstrip(".!? ")
    if (not context or not choice or context == choice
            or len(context.split()) > 12 or len(choice.split()) > 18):
        return []
    if _HEDGE_RE.search(statement) or re.search(
        r"\b(?:if|would|should|might|could|maybe|perhaps)\b",
        statement, re.IGNORECASE,
    ):
        return []
    return [StateCandidate(
        concept="preferred option for " + context,
        value=choice,
        kind="correct" if (explicit_change or match.group("now")) else "establish",
        scope="global",
        mtype=PREFERENCE,
        slot="contextual preference",
    )]


# Minimum rationale length: avoids "because yes" becoming memory.
_RATIONALE_MIN_LEN = 12

# "I don't love X because Y" is ambiguous: the user may reject Y as a
# motive while still loving X. It is NOT proof that they dislike X. Store
# the raw turn, but do not replace any existing preference state.
_NEGATED_REASON_RE = re.compile(
    r"^i\s+(?:do\s+not|don'?t|never)\s+"
    r"(?:love|like|adore|prefer)\b.+\bbecause\b",
    re.IGNORECASE | re.DOTALL,
)
# Second-hand claims are not the user's stated motive. "Because you said
# it means creativity" may establish the liking, but must not create a
# personal rationale saying creativity is why they love the thing.
_ATTRIBUTED_REASON_RE = re.compile(
    r"^(?:you\s+(?:said|told\s+me|think|suggested|claimed)|"
    r"(?:someone|people|they)\s+(?:said|say|think)|"
    r"i\s+(?:heard|read)\s+that)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StateCandidate:
    concept: str
    value: str = ""
    kind: str = "establish"  # establish | correct | invalidate
    scope: str = "session"  # session | global
    mtype: str = ""  # FACT | PREFERENCE | DECISION | INSTRUCTION | RATIONALE
    auto_confirm: bool = True  # False -> pending candidate, never truth
    rationale: str = ""  # attached user-stated reason, if any
    confidence: float = 1.0
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None
    transition_time: Optional[float] = None
    # Where the claim came from. Defaults to "user" because fresh
    # extraction only ever runs on user messages; a candidate admitted from
    # another origin (research, a specialist, an observation) carries its own
    # authority so provenance is not flattened at confirmation.
    authority: str = "user"
    # True when the concept was named by the user's own words ("we changed
    # the decision", "my preference is no longer correct") rather than
    # synthesised by an extraction pattern. A correction that names the
    # slot may adopt the single tracked fact of that type; a re-assertion
    # ("I prefer light mode") may not — type match alone is not a target.
    explicit_slot: bool = False
    # Non-empty when the slot registry routed this candidate to a named
    # personal-fact slot (favorite color, likes, instruction, ...). The
    # concept is already resolved, so apply must not substring-match it
    # onto other tracked concepts, and the scope comes from the registry
    # (personal facts are global).
    slot: str = ""


def normalize_concept(raw: str) -> str:
    text = _ARTICLES_RE.sub("", raw.strip()).strip().lower()
    return re.sub(r"\s+", " ", text)


def normalize_value(raw: str) -> str:
    return raw.strip().rstrip(".")


def _topic_concept(type_word: str, value: str) -> str:
    """Distinct concept for a type-word statement, keyed by its value."""
    base = normalize_concept("%s for %s" % (type_word, value))
    if len(base) <= _TOPIC_CONCEPT_MAX:
        return base
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    return base[:_TOPIC_CONCEPT_MAX].rstrip() + " " + digest


# Function words plus the extraction verbs themselves: a correction shares
# a topic with a tracked fact only through content words ("mode" in
# "light mode" vs "dark mode"), never through these.
_LIGHT_STOPWORDS = frozenset((
    "a an the and or but if of at by for with on in to is are was were "
    "be been it its this that these those i you he she we they me him her "
    "us them my your his our their what when where which who how why not no "
    "do does did can could should would will just very really there here "
    "prefer prefers preferred like likes liked love loves loved keep keeps "
    "kept keeping always never actually said correction well"
).split())


def _content_words(text: str) -> set:
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return {w for w in words
            if w not in _LIGHT_STOPWORDS and w not in _TYPE_WORDS}


def _correction_target(store: Any, scope: str, mtype: str,
                       value: str) -> list[str]:
    """Tracked concepts of one type that share a topic word with a correction.

    A correction may supersede a fact only when the new message shares its
    topic ("light mode" vs "dark mode") or explicitly names its slot. Type
    match alone is never enough.
    """
    words = _content_words(value)
    if not words:
        return []
    return sorted({
        s.concept for s in _inherit_pool(store, scope)
        if (s.memory_type or "") == mtype
        and words & (_content_words(s.value) | _content_words(s.concept))
    })


def _is_user_named(concept: str) -> bool:
    """True when the concept words were the user's own topic ("favorite
    color"), not a synthesised type slot ("preference",
    "preference for pizza") or a registry-keyed slot concept
    ("likes for pizza", "instruction for be concise")."""
    if concept in _TYPE_WORDS:
        return False
    for word in _TYPE_WORDS:
        if concept.startswith(word + " for "):
            return False
    if " for " in concept and slots_mod.is_slot_concept(concept):
        return False
    return True


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
    if concept == "name":
        # The generic establisher no longer produces a bare "name"
        # concept: "my name is X" routes through the first-name slot,
        # and nothing may collide with it.
        return None
    return concept, value


_CAUSAL_SENTENCE_RE = re.compile(
    r"^(?P<head>.+?[.!?;])\s+"
    r"(?P<rationale>(?:it|this|that)\s+"
    r"(?:helps?|makes?|keeps?|lets?|allows?|gives?|supports?|"
    r"improves?|calms?|relaxes?|focuses?|grounds?|energizes?)\b.+)$",
    re.IGNORECASE | re.DOTALL,
)


def _split_rationale(text: str) -> tuple[str, str]:
    """Split an explicit causal clause from a preference statement.

    The direct ``because`` form remains the primary path. A second sentence
    is accepted only when it is explicitly linked by a demonstrative pronoun
    and a causal verb (``It helps...``, ``This keeps...``). That preserves
    user meaning without treating an unrelated follow-up sentence as a
    rationale.
    """
    parts = re.split(r"\s+[Bb]ecause\s+", text, maxsplit=1)
    if len(parts) == 2:
        head, rationale = parts[0].strip(), parts[1].strip().rstrip(".")
    else:
        causal = _CAUSAL_SENTENCE_RE.match(text.strip())
        if causal is None:
            return text, ""
        head = causal.group("head").strip()
        rationale = causal.group("rationale").strip().rstrip(".")
    if _ATTRIBUTED_REASON_RE.match(rationale):
        return head, ""
    if len(rationale) < _RATIONALE_MIN_LEN:
        return text, ""
    return head, rationale


def _strip_now(value: str) -> str:
    return _NOW_SUFFIX_RE.sub("", value.strip()).strip()


def _extract_legacy(work: str, scope: str) -> list[StateCandidate]:
    """Original narrow patterns, unchanged behavior and ordering."""
    match = _INVALIDATE_RE.match(work)
    if match:
        concept = normalize_concept(match.group(1))
        if concept:
            return [StateCandidate(concept, "", "invalidate", scope,
                                   explicit_slot=True)]
        return []

    for pattern in (_CORRECT_NEW_RE, _CORRECT_WORD_RE):
        match = pattern.match(work)
        if match:
            concept = normalize_concept(match.group(1))
            value = normalize_value(match.group(2))
            if concept and value:
                return [StateCandidate(concept, value, "correct", scope,
                                       explicit_slot=True)]
            return []

    match = _CHANGE_RE.match(work)
    if match:
        concept = normalize_concept(match.group(1))
        value = normalize_value(match.group(2))
        if concept and value:
            return [StateCandidate(concept, value, "correct", scope,
                                   explicit_slot=True)]
        return []

    match = _FROM_NOW_ON_RE.match(work)
    if match:
        parsed = _parse_establishment(match.group(1))
        if parsed:
            return [StateCandidate(parsed[0], parsed[1], "correct", scope,
                                   explicit_slot=True)]
        # Fall through to the extended instruction fallback below.

    match = _INSTEAD_RE.match(work)
    if match:
        parsed = _parse_establishment(match.group(1))
        if parsed:
            return [StateCandidate(parsed[0], parsed[1], "correct", scope,
                                   explicit_slot=True)]
        # Fall through to the extended patterns below.

    parsed = _parse_establishment(work)
    if parsed:
        return [StateCandidate(parsed[0], parsed[1], "establish", scope)]
    return []


def _extract_extended(work: str, scope: str) -> list[StateCandidate]:
    """M1 natural-phrasing patterns. Deterministic, conservative."""
    match = _PREFER_RE.match(work)
    if match:
        value = normalize_value(_strip_now(match.group(1)))
        if value:
            return [StateCandidate("preference", value, "establish", scope,
                                   PREFERENCE)]
        return []

    match = _NEG_PREFER_RE.match(work)
    if match:
        obj = next(g for g in match.groups() if g is not None)
        value = normalize_value(_strip_now(obj))
        if value:
            return [StateCandidate("preference", "not " + value, "establish",
                                   scope, PREFERENCE)]
        return []

    for pattern, verb in (
        (_DECIDE_RE, None),
        (_KEEPING_RE, "keeping "),
        (_DECIDE_NEG_RE, "not use "),
    ):
        match = pattern.match(work)
        if match:
            obj = match.group(1).strip()
            obj = re.sub(r"^(?:the|a|an)\s+", "", obj, flags=re.IGNORECASE)
            value = normalize_value((verb or "") + obj)
            if value:
                return [StateCandidate("decision", value, "establish",
                                       scope, DECISION)]
            return []
    match = _LETS_KEEP_RE.match(work)
    if match:
        value = normalize_value(
            "keep %s as %s" % (match.group(1).strip(),
                               match.group(2).strip()))
        if value:
            return [StateCandidate("decision", value, "establish",
                                   scope, DECISION)]
        return []

    match = _FROM_NOW_ON_RE.match(work)
    if match:
        value = normalize_value(match.group(1))
        if value:
            # A topic-less standing directive is its own instruction, not
            # an implicit replacement of whatever came before.
            return [StateCandidate("instruction", value, "establish", scope,
                                   INSTRUCTION)]
        return []

    for pattern, prefix in (
        (_ALWAYS_RE, ""),
        (_NEVER_RE, "never "),
        (_DONT_RE, "don't "),
    ):
        match = pattern.match(work)
        if match:
            value = normalize_value(prefix + match.group(1))
            if value:
                return [StateCandidate("instruction", value, "establish",
                                       scope, INSTRUCTION)]
            return []

    match = _WE_CHANGED_RE.match(work)
    if match:
        concept = normalize_concept(match.group(1))
        value = normalize_value(match.group(2))
        if concept and value:
            return [StateCandidate(concept, value, "correct", scope,
                                   DECISION, explicit_slot=True)]
        return []

    return []


def _extract_slots(text: str, include_notes: bool = False
                   ) -> list[StateCandidate]:
    """Route one statement through the slot registry, if it names a slot.

    Slot-routed candidates always carry the registry's scope: personal
    facts (names, favorites, likes, rules, ...) are global, so they are
    visible from every session. Revocations arrive with an empty concept;
    apply resolves them against the protected rule entries.
    """
    routed = slots_mod.route_statement(text, include_fallback=include_notes)
    out: list[StateCandidate] = []
    for item in routed:
        out.append(StateCandidate(
            concept=item["concept"], value=item["value"],
            kind=item["kind"], scope=item["scope"], mtype=item["mtype"],
            slot=item["slot"],
        ))
    return out


# --- Elliptical replies resolved against the prior question --------------
# "It's yellow." names no slot by itself; the slot lives in the question
# that preceded it. The equivalent direct statement is synthesised below
# and routed through the slot registry, so an unsupported or ambiguous
# pairing still yields nothing: the registry stays the only authority.
_ANAPHORA_RE = re.compile(
    r"^(?:it'?s|its|that'?s|thats|this is|they'?re|he'?s|she'?s)\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_QUESTION_LIKE_RE = re.compile(
    r"\?\s*$"
    r"|^(?:what|whats|which|who|whose|where|when|why|how|do you|did you|"
    r"can you|could you|tell me|remember|remind me|any idea|any clue)\b",
    re.IGNORECASE,
)
_SLOT_PHRASE_RE = re.compile(
    r"\bmy\s+([a-z][a-z'\-]*(?:\s+[a-z][a-z'\-]*){0,5})\s*$",
    re.IGNORECASE,
)
_SLOT_PHRASE_JUNK_RE = re.compile(
    r"\b(?:is|are|what|what's|whats|who|whose|where|when|why|how|which|"
    r"please|again|then|actually|really)\b",
    re.IGNORECASE,
)
_ANSWER_VOID = frozenset({
    "yes", "no", "yeah", "yep", "nope", "nah", "ok", "okay", "sure",
    "thanks", "thank you", "maybe", "idk", "dunno", "nothing", "none",
    "not sure", "dont know", "don't know", "hi", "hello", "hey", "hiya",
    "howdy", "yo", "sup", "greetings", "good morning", "good afternoon",
    "good evening", "whats up", "what's up",
})
_GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|hiya|howdy|yo|sup|greetings|"
    r"good\s+(?:morning|afternoon|evening)|what'?s\s+up)"
    r"(?:\s+(?:there|again|aurora))?[!.,]*$",
    re.IGNORECASE,
)
# A bare elliptical answer is a noun phrase, never a question or an
# instruction aimed at the assistant ("List my previous favorite
# colors." must not become a favorite-color value).
_BARE_VERB_RE = re.compile(
    r"^(?:what|when|where|why|who|whose|which|how|is|are|was|were|do|does|"
    r"did|can|could|would|should|will|please|list|tell|show|give|say|"
    r"describe|explain|remind|remember|name|check|find|search|look|read|"
    r"repeat|compare|use|stop|wait|listen|try|make|go|come|keep|run|set|"
    r"let|help|think|know|want|need)\b",
    re.IGNORECASE,
)
# A noun-phrase value must not contain a finite clause ("your info are
# wrong"), a second-person address ("you are useless"), or a first-person
# statement ("i meant that").
_CLAUSE_WORD_RE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being|do|does|did|have|has|had|"
    r"will|would|shall|should|can|could|may|might|must)\b",
    re.IGNORECASE,
)
_SECOND_PERSON_RE = re.compile(
    r"\b(?:you|your|yours|you're|youre)\b", re.IGNORECASE
)
# A change the user announces but has not yet valued: "i wanna change my fav
# color to something else". The slot is named ("my fav color"); the value is
# whatever the next elliptical turn supplies.
_CHANGE_INTENT_RE = re.compile(
    r"\b(?:change|update|set|switch|pick|choose)\s+(?:my|the)\s+"
    r"([a-z][a-z'\-]*(?:\s+[a-z][a-z'\-]*){0,5}?)"
    r"\s*(?=\s+(?:to|into|back|again)\b|[,.;!?]|$)",
    re.IGNORECASE,
)
# The assistant side of the same exchange: a question or an invitation to
# supply a value ("What would you like your new favorite color to be?",
# "please let me know your new favorite color").
_ASSISTANT_ASK_RE = re.compile(
    r"\?\s*$|\b(?:let me know|tell me|remind me|what(?:'s| is)|which)\b",
    re.IGNORECASE,
)
# Second-person slot reference in an assistant turn. Only "favorite X"
# phrasings qualify, because the question is about a preference the user is
# being invited to state, and the registry (not this regex) decides whether
# the noun names a real slot.
_SECOND_PERSON_SLOT_RE = re.compile(
    r"\b(?:my|your)\s+(?:(?:new|current|latest)\s+)?"
    r"((?:fav|favorite|favourite)\s+[a-z][a-z'\-]*)"
    r"\s*(?=\s+(?:to|be|is|are|now|and|or)\b|[,.;!?]|$)",
    re.IGNORECASE,
)


def _plausible_answer_value(text: str, max_words: int) -> bool:
    raw = (text or "").strip()
    if not raw or len(raw.split()) > max_words:
        return False
    if _GREETING_RE.match(raw):
        return False
    if any(mark in raw for mark in (",", "?", ":")):
        return False
    if _CLAUSE_WORD_RE.search(raw):
        return False
    if _SECOND_PERSON_RE.search(raw):
        return False
    if re.match(r"^(?:i|we|my|our|me|us)\b", raw, re.IGNORECASE):
        return False
    if normalize_value(raw).lower() in _ANSWER_VOID:
        return False
    return True


def _plausible_bare_answer(text: str, max_words: int) -> bool:
    if not _plausible_answer_value(text, max_words):
        return False
    if _BARE_VERB_RE.match((text or "").strip()):
        return False
    return True


def _clean_slot_phrase(phrase: str) -> Optional[str]:
    """Validate one candidate slot phrase; the registry gates the rest."""
    phrase = (phrase or "").strip().strip(" .,;:!?")
    if not phrase or len(phrase.split()) > 6:
        return None
    if _SLOT_PHRASE_JUNK_RE.search(phrase):
        return None
    return phrase.lower()


def _question_slot_phrase(
    question, allow_second_person: bool = False
) -> Optional[str]:
    """The "my X" subject a prior turn named, else None.

    Deterministic. A question-like message maps through its trailing
    subject ("what is my favorite color?" -> "favorite color"). A user
    turn announcing a change maps through the slot it names ("i wanna
    change my fav color to something else" -> "fav color"). With
    ``allow_second_person``, an assistant question or invitation may name
    the slot in the second person ("your new favorite color"). Anything
    else yields None, so no fact is invented.
    """
    if not isinstance(question, str):
        return None
    q = question.strip()
    if not q:
        return None
    question_like = _QUESTION_LIKE_RE.search(q) is not None
    if not question_like:
        if allow_second_person:
            if not _ASSISTANT_ASK_RE.search(q):
                return None
        else:
            match = _CHANGE_INTENT_RE.search(q)
            return _clean_slot_phrase(match.group(1)) if match else None
    q = q.split("?", 1)[0].strip().lower()
    match = _SLOT_PHRASE_RE.search(q)
    if match:
        phrase = _clean_slot_phrase(match.group(1))
        if phrase:
            return phrase
    if allow_second_person:
        match = _SECOND_PERSON_SLOT_RE.search(question.strip())
        if match:
            return _clean_slot_phrase(match.group(1))
    return None


def _anaphora_value(text: str) -> str:
    """Value an elliptical reply asserts, or "" when it states none.

    Accepts copular anaphora ("It's yellow.", "That's yellow") and short
    bare answers ("yellow.") that follow a slot-mapping question.
    Questions, acknowledgements, and first-person statements are not
    answers and yield "".
    """
    raw = (text or "").strip()
    if not raw or raw.endswith("?"):
        return ""
    match = _ANAPHORA_RE.match(raw)
    if match:
        value = normalize_value(match.group(1).strip())
    else:
        # Bare answers are short noun phrases: at most three words, no
        # comma-separated lists, no questions, no instructions, and no
        # finite clause ("compare there engine", "you are useless").
        if not _plausible_bare_answer(raw, 3):
            return ""
        value = normalize_value(raw)
    if not value or len(value) > 60:
        return ""
    if _GREETING_RE.match(value):
        return ""
    if value.lower() in _ANSWER_VOID:
        return ""
    return value


def _expand_with_context(
    text: str, context_question: str, allow_second_person: bool = False
) -> list[StateCandidate]:
    """Resolve an elliptical reply against the turn it answers.

    Synthesises the equivalent direct statement ("my favorite color is
    yellow") and routes it through the slot registry. No slot match means
    no candidate: never a note, never a guess.
    """
    phrase = _question_slot_phrase(
        context_question, allow_second_person=allow_second_person)
    if not phrase:
        return []
    value = _anaphora_value(text)
    if not value:
        return []
    return _extract_slots(f"my {phrase} is {value}")


def extract_candidates(
    text: str,
    role: str,
    context_question: Optional[str] = None,
    assistant_question: Optional[str] = None,
) -> list[StateCandidate]:
    """Extract state candidates from user-authored text only.

    ``context_question`` is the user's own immediately preceding turn when
    this message answers it elliptically ("It's yellow." after "what is my
    favorite color?", or "violet" after "i wanna change my fav color to
    something else"). ``assistant_question`` is the assistant turn
    immediately before this message, consulted only when the user context
    resolved nothing and only when that turn asked for or invited the
    value. Both are last resorts, mapped only through the slot registry —
    without them extraction behaves exactly as before.
    """
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
    # A pronoun is not a stable personal preference. Context resolution,
    # if independently verified, happens at the ContextManager boundary.
    from .discourse import is_bare_reference_preference
    if is_bare_reference_preference(work):
        return []
    if _NEGATED_REASON_RE.match(work):
        return []

    contextual = _extract_contextual_preference(work)
    if contextual:
        return contextual

    head, rationale = _split_rationale(work)

    # Correction prefixes ("actually, ...", "no, ...") unwrap to the real
    # statement, which then parses normally and supersedes via the standard
    # newest-wins path.
    corrected = False
    match = _CORRECTION_PREFIX_RE.match(head)
    if match:
        head = match.group(1).strip()
        corrected = True
        if not head:
            return []
    else:
        match = _THATS_WRONG_RE.match(head)
        if match:
            head = match.group(1).strip()
            corrected = True
            if not head:
                return []

    # Slot phrasings first: a statement that names a personal-fact slot
    # ("my first name is X", "I love pizza", "always Y") lands there
    # instead of the generic legacy patterns.
    cands = _extract_preference_evolution(head)
    if not cands:
        cands = _extract_explicit_car_preferences(head)
    if not cands:
        cands = _extract_all_shades(head)
    if not cands and _REPEATED_SUBJECT_LIKE_RE.fullmatch(head.strip()):
        cands = _extract_repeated_subject_preference(head)
        if not cands:
            # An ambiguous subject ("this"/"anything") is not a named
            # preference. Do not let the generic greedy likes slot treat
            # the whole utterance as an item name instead.
            return []
    if not cands:
        cands = _extract_slots(head)
    if not cands:
        cands = _extract_legacy(head, scope)
    if not cands:
        cands = _extract_extended(head, scope)
    if not cands and corrected:
        # Bare correction ("Actually, green."): a value with no parseable
        # concept. Resolve by inheritance in apply, else hold pending. The
        # remainder must still be a value, not an unrelated clause
        # ("no your info are wrong") that would overwrite the tracked fact.
        if _plausible_answer_value(head, 12):
            cands = [StateCandidate(
                "", normalize_value(head), "correct", scope)]
    if not cands and not corrected:
        # Meaningful personal statements that fit no slot and no legacy
        # pattern ("I'm allergic to penicillin") become capped, deduped
        # notes instead of being dropped.
        cands = _extract_slots(head, include_notes=True)
    if corrected:
        # An explicit correction prefix marks a re-asserted statement as a
        # correction of the tracked fact, so "Actually, I prefer green."
        # updates the single tracked concept of that type instead of
        # establishing an additional one. A re-asserted protected rule is
        # exempt: an ordinary statement may never re-target a rule.
        cands = [
            dataclass_replace(c, kind="correct")
            if (c.kind == "establish"
                and not slots_mod.is_protected_concept(c.concept))
            else c
            for c in cands
        ]
    if not cands:
        # Ambiguous phrasing with no extractable structure: try the
        # hedge-stripped remainder once ("I think I prefer blue" keeps
        # its preference shape but stays pending); otherwise hold an
        # uncertain pending candidate, or leave raw experience alone.
        stripped_head = head
        match = _HEDGE_PREFIX_RE.match(head)
        if match and match.group(1).strip():
            stripped_head = match.group(1).strip()
            inner = _extract_slots(stripped_head, include_notes=True)
            if not inner:
                inner = _extract_legacy(stripped_head, scope)
            if not inner:
                inner = _extract_extended(stripped_head, scope)
            if inner:
                return [dataclass_replace(c, auto_confirm=False,
                                          confidence=0.3) for c in inner]
        # Elliptical reply ("It's yellow.") resolved against the prior
        # question. The slot registry gates the result, so an unsupported
        # pairing falls through to the ordinary empty outcome below. The
        # assistant turn is consulted only when the user's own turn named
        # nothing, so "violet" after an invitation still lands in the slot
        # the invitation asked about.
        if context_question:
            cands = _expand_with_context(stripped_head, context_question)
        if not cands and assistant_question:
            cands = _expand_with_context(
                stripped_head, assistant_question,
                allow_second_person=True)
        if not cands and _HEDGE_RE.search(work):
            trimmed = work[:200].strip()
            if trimmed:
                return [StateCandidate(
                    "", trimmed, "establish", scope, UNCERTAIN,
                    auto_confirm=False, confidence=0.3)]
        if not cands:
            return []

    if rationale:
        cands = [dataclass_replace(c, rationale=rationale)
                 if i == 0 else c
                 for i, c in enumerate(cands)]
    if _HEDGE_RE.search(work):
        # Hedged explicit statements ("I think I prefer blue") must not
        # auto-confirm; they wait as pending candidates.
        cands = [dataclass_replace(c, auto_confirm=False, confidence=0.3)
                 for c in cands]
    return cands


def _scope_key(scope: str, session_id: str) -> str:
    return "global" if scope == "global" else session_id


def _tracked_concepts(rows) -> list[str]:
    """Concepts eligible for resolution/inheritance.

    Rationale rows ("rationale for X") are excluded so ordinary
    statements can never resolve onto them.
    """
    return [s.concept for s in rows
            if not s.concept.startswith("rationale for ")]


def _cue_type(text: str) -> Optional[str]:
    lowered = text.lower()
    for word, mtype in _TYPE_CUES:
        if re.search(r"\b" + re.escape(word) + r"\b", lowered):
            return mtype
    return None


def _inherit_pool(store: Any, scope: str) -> list[Any]:
    """Current non-rationale states visible to inheritance (scope + global)."""
    try:
        pool = [s for s in store.fetch_states(scope=scope, status="current")
                if not s.concept.startswith("rationale for ")]
        if scope != "global":
            pool += [s for s in store.fetch_states(
                scope="global", status="current")
                if not s.concept.startswith("rationale for ")]
    except Exception:
        return []
    return pool


def _typed_concepts(store: Any, scope: str, cue: str) -> list[str]:
    """Distinct current concepts of one memory type, in the inherit pool."""
    return sorted({s.concept for s in _inherit_pool(store, scope)
                   if (s.memory_type or "") == cue})


def inherit_concept(store: Any, scope: str, cue: Optional[str] = None):
    """Resolve a bare/type-word correction to a tracked concept.

    Returns (concept, memory_type) or (None, None) when nothing can be
    identified — the caller must then hold a pending candidate instead
    of guessing. Type cues resolve only when exactly one concept of that
    type is tracked; with several, the correction is ambiguous and is
    refused rather than silently aimed at the newest. Untyped legacy rows
    participate only through the unique-current fallback, never by cue.
    """
    pool = _inherit_pool(store, scope)
    if not pool:
        return None, None
    if cue is not None:
        matches = [s for s in pool if (s.memory_type or "") == cue]
        concepts = {s.concept for s in matches}
        if len(concepts) == 1:
            only = max(matches, key=lambda s: s.state_id)
            return only.concept, only.memory_type or cue
        return None, None
    concepts = {s.concept for s in pool}
    if len(concepts) == 1:
        only = max(pool, key=lambda s: s.state_id)
        return only.concept, only.memory_type or ""
    return None, None


def _apply_type_word_concept(store: Any, scope: str, cand, concept: str,
                             mtype: str, allow_pending: bool):
    """Resolve a bare type-word concept to a tracked or topic-bearing one.

    Each type-word statement ("I prefer dark mode") is its own fact under a
    topic-bearing concept, so unrelated preferences/decisions/instructions
    never share one newest-wins slot. A correction supersedes a tracked fact
    only when its value shares a topic word with that fact ("light mode" vs
    "dark mode") or when the sentence explicitly names the slot itself ("we
    changed the decision"); with several candidates it is held pending
    rather than aimed at any of them. Returns (concept, mtype, hold_pending).
    """
    pool_concepts = _typed_concepts(store, scope, mtype)
    if cand.kind == "invalidate":
        # "My preference is no longer correct" names the slot; with a
        # single occupant the reference is definite, with several it is
        # ambiguous and held.
        if len(pool_concepts) == 1:
            return pool_concepts[0], mtype, False
        if len(pool_concepts) > 1 and allow_pending:
            return concept, mtype, True
        return _topic_concept(concept, cand.value), mtype, False
    if cand.kind != "correct":
        # Establish (including confirmed hedged pendings): its own fact.
        return _topic_concept(concept, cand.value), mtype, False
    targets = _correction_target(store, scope, mtype, cand.value)
    if len(targets) == 1:
        return targets[0], mtype, False
    if len(targets) > 1:
        if allow_pending:
            return concept, mtype, True
        return _topic_concept(concept, cand.value), mtype, False
    if getattr(cand, "explicit_slot", False):
        # The user named the slot ("we changed the decision to X"): the
        # single occupant is the definite referent; several are ambiguous.
        if len(pool_concepts) == 1:
            return pool_concepts[0], mtype, False
        if allow_pending:
            return concept, mtype, True
    return _topic_concept(concept, cand.value), mtype, False


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


def _reject_pending(store: Any, scope: str, concept: str,
                    type_word: Optional[str] = None) -> list[str]:
    """Mark pending candidates for a concept rejected once it is superseded.

    Hedged type-word pendings ("I think I prefer blue") keep the bare type
    word as their proposed concept while the confirmed fact lives under a
    topic-bearing concept, so the bare type word is rejected alongside.
    """
    diagnostics: list[str] = []
    targets = [concept]
    if type_word and type_word != concept:
        targets.append(type_word)
    seen: set[int] = set()
    pendings = []
    try:
        for target in targets:
            for row in store.fetch_candidates(
                    scope=scope, status="pending", concept=target):
                if row.candidate_id not in seen:
                    seen.add(row.candidate_id)
                    pendings.append(row)
    except Exception as e:
        return [f"state_failed: pending read failed: {e}"]
    for pend in pendings:
        try:
            store.set_candidate_status(pend.candidate_id, "rejected")
            diagnostics.append(f"candidate_rejected: {pend.proposed_concept}")
        except Exception as e:
            diagnostics.append(f"state_failed: pending reject failed: {e}")
    return diagnostics


def _record_pending(store: Any, session_id: str, scope: str, message_id: int,
                    cand, reason: str = "") -> list[str]:
    """Hold an uncertain candidate. Dedup-guarded; never truth.

    When `reason` is given it is emitted as a second diagnostic naming why
    the statement was held, so an ambiguous correction is visible instead
    of silently disappearing.
    """
    try:
        existing = store.fetch_candidates(scope=scope, status="pending")
        for row in existing:
            if (row.proposed_concept == cand.concept
                    and row.proposed_value == cand.value
                    and row.source_ids == [message_id]):
                return []
        store.create_candidate(
            session_id, scope, [message_id],
            proposed_type=cand.mtype or UNCERTAIN,
            proposed_concept=cand.concept, proposed_value=cand.value,
            proposed_rationale=cand.rationale, authority="user",
            confidence=cand.confidence, status="pending",
        )
        out = [f"candidate_pending: {cand.concept or cand.value or 'uncertain'}"]
        if reason:
            out.append(
                f"candidate_pending_reason: {cand.value or 'uncertain'};"
                f" {reason}")
        return out
    except Exception as e:
        return [f"state_failed: pending write failed: {e}"]


def _supersede_related(
    store: Any,
    scope: str,
    concept: str,
    valid_until: Optional[float] = None,
    transition_time: Optional[float] = None,
    predecessor_state_ids: Optional[list[int]] = None,
) -> list[str]:
    """Supersede current rationale rows tied to a superseded concept."""
    diagnostics: list[str] = []
    try:
        related = store.fetch_states(
            scope=scope, concept="rationale for " + concept,
            status="current")
        relationships = store.supersede_states_with_relationships(
            [row.state_id for row in related],
            explained_state_ids=predecessor_state_ids or [],
            relationship_type="explains",
            valid_until=valid_until,
            transition_time=transition_time,
        )
        for _ in related:
            diagnostics.append(
                f"state_superseded_related: rationale for {concept}")
        for relationship in relationships:
            diagnostics.append(
                f"relationship_superseded_related: {relationship.relationship_id}"
            )
    except Exception as e:
        return [f"state_failed: state write failed: {e}"]
    return diagnostics


def _apply_confirmed(store: Any, session_id: str, scope: str,
                     message_id: int, cand,
                     concept: str, mtype: str,
                     resolve: bool = True) -> list[str]:
    """Apply one resolved auto-confirm candidate as structured memory.

    Shared by fresh extraction and pending confirmation, so both paths
    enforce identical supersession, rationale, and provenance behavior.
    `resolve` is False for concepts already produced by the type-word
    resolver: their value-keyed form must not be substring-matched onto a
    shorter sibling ("preference for blue" must not absorb "preference
    for blue car").
    """
    diagnostics: list[str] = []
    try:
        known = _tracked_concepts(store.fetch_states(scope=scope))
        if resolve and cand.mtype != RATIONALE and not concept.startswith(
                "rationale for "):
            concept = resolve_concept(known, concept)
        if (cand.kind in ("correct", "invalidate")
                and slots_mod.is_protected_concept(concept)
                and getattr(cand, "slot", "") != "rules revoke"):
            # Ordinary corrections never retire a standing rule; only an
            # explicit revocation ("stop doing X", "forget that rule")
            # may. Hold the statement visible instead of overwriting.
            diagnostics.append(f"rule_protected: {concept}")
            diagnostics.extend(_record_pending(
                store, session_id, scope, message_id,
                dataclass_replace(cand, concept=concept,
                                  mtype=mtype or UNCERTAIN, confidence=0.3),
                reason="protected rule; only an explicit revocation"
                       " may retire it"))
            return diagnostics
        currents = store.fetch_states(
            scope=scope, concept=concept, status="current"
        )
        if not currents and scope != "global":
            # A correction may name a fact that lives in the global scope
            # ("Actually, green." onto a global favorite color); adopt the
            # fact's own scope so the successor supersedes in place
            # instead of forking a session duplicate.
            in_global = store.fetch_states(
                scope="global", concept=concept, status="current")
            if in_global:
                scope = "global"
                currents = in_global
    except Exception as e:
        return [f"state_failed: state read failed: {e}"]
    if cand.kind == "invalidate":
        transition = (
            cand.transition_time if cand.transition_time is not None else time.time()
        )
        try:
            store.supersede_states(
                [cur.state_id for cur in currents],
                valid_until=cand.valid_until,
                transition_time=transition,
            )
        except Exception as e:
            return [f"state_failed: state write failed: {e}"]
        diagnostics.append(f"state_invalidated: {concept}")
        diagnostics.extend(_supersede_related(
            store, scope, concept,
            valid_until=cand.valid_until,
            transition_time=transition,
            predecessor_state_ids=[cur.state_id for cur in currents],
        ))
        diagnostics.extend(_reject_pending(
            store, scope, concept, type_word=_WORD_BY_TYPE.get(mtype)))
        return diagnostics
    # Repeated user statements are evidence for the *same* current fact,
    # not a new historical preference change. Preserve original timestamps
    # and attach the additional user message as provenance. A generic "I
    # love purple" is NOT evidence for the stronger "in any shade" clause.
    def comparable(value: str) -> str:
        cleaned = " ".join(value.strip().rstrip(".! ").split()).casefold()
        # "the Regera" and "Regera" refer to the same keyed preference.
        # Preserve exact wording in the stored value and raw messages.
        if concept.startswith("likes for "):
            cleaned = re.sub(r"^(loves|likes|hates|dislikes) the ", r"\1 ", cleaned)
        return cleaned
    same_value = (currents and len(currents) == 1 and
                  comparable(currents[0].value) == comparable(cand.value))
    # A softer rephrasing of an existing 'love' is not a retraction.
    # Keep the stronger preference and attach the new statement as evidence.
    softer_rephrase = (currents and len(currents) == 1
                       and cand.kind == "establish"
                       and concept.startswith("likes for ")
                       and comparable(currents[0].value).startswith("loves ")
                       and comparable(cand.value).startswith("likes ")
                       and comparable(currents[0].value)[6:]
                       == comparable(cand.value)[6:])
    if same_value or softer_rephrase:
        if (mtype == PREFERENCE
                and getattr(cand, "authority", "user") == "user"
                and currents[0].authority == "user"):
            try:
                store.add_state_evidence(currents[0].state_id, message_id)
            except Exception as e:
                return [f"state_failed: evidence update failed: {e}"]
        return diagnostics
    # A later generic "I love purple" is not a retraction of an earlier
    # explicit "purple in any shade". Keep the more informative value
    # unless the user marks the new message as a correction.
    if (currents and cand.kind == "establish"
            and concept.startswith("likes for ")
            and currents[0].value.casefold()
            == cand.value.casefold() + " in any shade"):
        return diagnostics
    try:
        transition = (
            cand.transition_time if cand.transition_time is not None else time.time()
        )
        new_id = store.create_state_successor(
            session_id, scope, concept, cand.value, [message_id],
            predecessor_ids=[cur.state_id for cur in currents],
            authority=getattr(cand, "authority", None) or "user",
            # Confirmation is the act that upgrades a held candidate, so
            # confidence is 1.0 here regardless of the pending row's own
            # support value. Do not propagate it: pending is deliberately
            # stored low.
            memory_type=mtype, confidence=1.0,
            valid_from=cand.valid_from, valid_until=cand.valid_until,
            transition_time=transition,
        )
        new_state = next(
            state for state in store.fetch_states(
                scope=scope, concept=concept, status="current")
            if state.state_id == new_id
        )
        diagnostics.append(
            f"state_updated: {concept} -> {cand.value} (state {new_id})"
        )
        diagnostics.extend(_supersede_related(
            store, scope, concept,
            valid_until=new_state.valid_from,
            transition_time=transition,
            predecessor_state_ids=[cur.state_id for cur in currents],
        ))
        diagnostics.extend(_reject_pending(
            store, scope, concept, type_word=_WORD_BY_TYPE.get(mtype)))
        if cand.rationale:
            rationale_id, relationship_id = (
                store.create_state_with_explains(
                    session_id, scope, "rationale for " + concept,
                    cand.rationale, [message_id], new_id,
                    memory_type=RATIONALE, confidence=1.0,
                    valid_from=new_state.valid_from,
                )
            )
            diagnostics.append(
                f"state_updated: rationale for {concept}"
                f" (state {rationale_id})"
            )
            diagnostics.append(
                f"relationship_updated: explains {rationale_id} -> {new_id}"
                f" (relationship {relationship_id})"
            )
    except Exception as e:
        diagnostics.append(f"state_failed: state write failed: {e}")
    return diagnostics


def confirm_pending(store: Any, session_id: str,
                    candidate_id: int) -> list[str]:
    """Promote a pending candidate to confirmed structured memory.

    Only pending rows with a real concept and a concrete (non-UNCERTAIN)
    type may confirm; anything else is refused with a diagnostic rather
    than guessed. Confirmation runs the identical application path as
    fresh extraction, so provenance, supersession, and rationale linkage
    behave the same. Never touches raw messages.
    """
    try:
        rows = store.fetch_candidates(candidate_id=candidate_id)
    except Exception as e:
        return [f"state_failed: candidate read failed: {e}"]
    if not rows:
        return [f"state_failed: candidate not found: {candidate_id}"]
    row = rows[0]
    if row.status != "pending":
        return [f"state_failed: candidate {candidate_id} is {row.status},"
                " not pending"]
    if not row.proposed_concept or (row.proposed_type or "") == UNCERTAIN:
        return [f"state_failed: candidate {candidate_id} has no"
                " confirmable concept"]
    if not row.source_ids:
        return [f"state_failed: candidate {candidate_id} has no provenance"]
    scope = "global" if row.scope == "global" else session_id
    cand = StateCandidate(
        concept=row.proposed_concept, value=row.proposed_value,
        # Confirmation establishes the held statement as a fact. A hedged
        # type-word candidate ("I think I prefer blue") is a new statement,
        # not a correction of some other tracked fact, so it never inherits
        # a slot on confirm.
        kind="establish", scope=row.scope, mtype=row.proposed_type,
        auto_confirm=True, rationale=row.proposed_rationale or "",
        confidence=1.0,
        # Carry the recorded origin through, so a research- or
        # specialist-sourced candidate is not confirmed as if the user had
        # said it. The candidate row already holds it.
        authority=row.authority or "user",
    )
    concept = row.proposed_concept
    if slots_mod.is_slot_concept(concept):
        # Slot-routed personal facts are global wherever they are
        # confirmed from: a learned "favorite movie" candidate and the
        # user's own "my favorite movie is X" are one memory, not two.
        scope = "global"
    mtype = row.proposed_type
    slot_resolved = False
    if concept in _TYPE_WORDS:
        # A held type-word candidate ("I think I prefer blue") confirms
        # onto the single tracked concept of that type, or else becomes a
        # topic-bearing fact — never a newest-slot overwrite and never a
        # substring guess.
        mtype = _TYPE_BY_WORD.get(concept, mtype)
        concept, mtype, _hold = _apply_type_word_concept(
            store, scope, cand, concept, mtype, allow_pending=False)
        slot_resolved = True
    elif slots_mod.is_slot_concept(concept):
        # A held slot statement ("I think my favorite color is blue")
        # confirms onto its own slot, never substring-matched elsewhere.
        slot_resolved = True
    diagnostics = _apply_confirmed(
        store, session_id, scope, row.source_ids[0], cand, concept, mtype,
        resolve=not slot_resolved)
    if any(d.startswith("state_failed") for d in diagnostics):
        return diagnostics
    try:
        store.set_candidate_status(candidate_id, "confirmed")
    except Exception as e:
        return diagnostics + [f"state_failed: confirm write failed: {e}"]
    return diagnostics + [f"candidate_confirmed: {row.proposed_concept}"]


def _notes_full(store: Any, scope: str) -> bool:
    """True when the append-only notes slot has reached its cap."""
    cap = slots_mod.notes_cap()
    try:
        rows = store.fetch_states(scope=scope, status="current")
    except Exception:
        return False
    return sum(1 for s in rows
               if s.concept.startswith("notes for ")) >= cap


def _apply_rule_revocation(
    store: Any, session_id: str, scope: str, message_id: int, cand,
) -> list[str]:
    """Retire protected rules, but only via an explicit revocation.

    "stop doing X" targets the single rule whose words match X; "forget
    that rule" targets the single tracked rule. Zero or several matches
    are held pending with a reason, never guessed.
    """
    try:
        pool = _inherit_pool(store, scope)
    except Exception:
        pool = []
    rules = sorted({s.concept for s in pool
                    if slots_mod.is_protected_concept(s.concept)})
    concept: Optional[str] = None
    if cand.value:
        words = _content_words(cand.value)
        targets = [c for c in rules
                   if words & (_content_words(c))]
        if len(targets) == 1:
            concept = targets[0]
        elif len(targets) > 1:
            return _record_pending(
                store, session_id, scope, message_id,
                dataclass_replace(cand, concept="", mtype=UNCERTAIN,
                                  confidence=0.3),
                reason="ambiguous rule revocation; candidates: %s"
                       % (", ".join(targets),))
        else:
            return _record_pending(
                store, session_id, scope, message_id,
                dataclass_replace(cand, concept="", mtype=UNCERTAIN,
                                  confidence=0.3),
                reason="no tracked rule matches the revocation")
    elif len(rules) == 1:
        concept = rules[0]
    elif len(rules) > 1:
        return _record_pending(
            store, session_id, scope, message_id,
            dataclass_replace(cand, concept="", mtype=UNCERTAIN,
                              confidence=0.3),
            reason="ambiguous rule revocation; candidates: %s"
                   % (", ".join(rules),))
    else:
        return _record_pending(
            store, session_id, scope, message_id,
            dataclass_replace(cand, concept="", mtype=UNCERTAIN,
                              confidence=0.3),
            reason="no tracked rule to revoke")
    revoke = dataclass_replace(cand, concept=concept, kind="invalidate")
    return _apply_confirmed(
        store, session_id, scope, message_id, revoke, concept, INSTRUCTION,
        resolve=False)


def _apply_slot_retire(
    store: Any, session_id: str, scope: str, message_id: int, cand,
) -> list[str]:
    """Retire a slot fact, but only when the negation names the value
    actually stored: "I don't live in Karaj" retires home location if
    and only if the current value is Karaj. A mismatch is held pending
    with a reason, never guessed."""
    diagnostics: list[str] = []
    concept = cand.concept
    try:
        currents = store.fetch_states(
            scope=scope, concept=concept, status="current")
    except Exception as e:
        return [f"state_failed: state read failed: {e}"]
    match = [c for c in currents
             if (c.value or "").lower() == (cand.value or "").lower()]
    if not match:
        stored = currents[0].value if currents else None
        return _record_pending(
            store, session_id, scope, message_id,
            dataclass_replace(cand, kind="establish", mtype=UNCERTAIN,
                              confidence=0.3),
            reason="retirement names %r but stored %s is %r" % (
                cand.value, concept, stored))
    transition = (cand.transition_time if cand.transition_time is not None
                  else time.time())
    try:
        store.supersede_states(
            [c.state_id for c in match],
            valid_until=cand.valid_until, transition_time=transition)
    except Exception as e:
        return [f"state_failed: state write failed: {e}"]
    diagnostics.append(f"state_invalidated: {concept}")
    diagnostics.extend(_supersede_related(
        store, scope, concept, valid_until=cand.valid_until,
        transition_time=transition,
        predecessor_state_ids=[c.state_id for c in match]))
    return diagnostics


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
        if getattr(cand, "slot", "") == "rules revoke":
            diagnostics.extend(_apply_rule_revocation(
                store, session_id, scope, message_id, cand))
            continue
        if getattr(cand, "slot", ""):
            # Slot-routed personal fact: the registry already named the
            # concept and the global scope, so no substring resolution.
            if not cand.auto_confirm:
                # Hedged slot statement ("I think my favorite color is
                # blue"): a pending candidate until confirmed, same as
                # every other hedged statement.
                diagnostics.extend(
                    _record_pending(store, session_id, scope, message_id,
                                    cand))
                continue
            if cand.kind == "retire_slot":
                diagnostics.extend(_apply_slot_retire(
                    store, session_id, scope, message_id, cand))
                continue
            if cand.slot == "notes" and _notes_full(store, scope):
                diagnostics.append(
                    "notes_full: capacity reached; statement not stored")
                continue
            diagnostics.extend(_apply_confirmed(
                store, session_id, scope, message_id, cand,
                cand.concept, cand.mtype, resolve=False))
            continue
        if not cand.auto_confirm:
            diagnostics.extend(
                _record_pending(store, session_id, scope, message_id, cand))
            continue
        concept = cand.concept
        mtype = cand.mtype
        slot_resolved = False
        if not concept:
            # Bare correction ("Actually, green."): resolve only to a fact
            # the statement can actually refer to — a typed fact sharing
            # its topic, or the scope's single user-named slot. Anything
            # else is held pending, never guessed.
            cue = _cue_type(cand.value)
            resolved, resolved_type = None, None
            if cand.kind == "correct" and cue:
                targets = _correction_target(store, scope, cue, cand.value)
                if len(targets) == 1:
                    resolved, resolved_type = targets[0], cue
                else:
                    why = (
                        "targets %d tracked %s facts: %s" % (
                            len(targets), cue, ", ".join(targets))
                        if len(targets) > 1 else
                        "no tracked %s fact shares its topic" % cue)
                    diagnostics.extend(_record_pending(
                        store, session_id, scope, message_id,
                        dataclass_replace(cand, mtype=cand.mtype or UNCERTAIN,
                                          confidence=0.3),
                        reason=why))
                    continue
            else:
                try:
                    pool = _inherit_pool(store, scope)
                except Exception:
                    pool = []
                named = [s for s in pool if _is_user_named(s.concept)]
                if len(named) == 1:
                    only = max(named, key=lambda s: s.state_id)
                    resolved = only.concept
                    resolved_type = only.memory_type or ""
            if resolved is None:
                diagnostics.extend(_record_pending(
                    store, session_id, scope, message_id,
                    dataclass_replace(cand, mtype=cand.mtype or UNCERTAIN,
                                      confidence=0.3),
                    reason="no single referent for a bare correction"))
                continue
            concept, mtype = resolved, resolved_type or mtype
        elif concept in _TYPE_WORDS:
            # Bare type word ("we changed the decision"): adopt a tracked
            # concept only when the statement shares its topic or names
            # the slot itself; else establish a topic-bearing fact of its
            # own so unrelated statements never share one newest-wins slot.
            mtype = _TYPE_BY_WORD.get(concept, mtype)
            concept, mtype, hold = _apply_type_word_concept(
                store, scope, cand, concept, mtype, allow_pending=True)
            if hold:
                diagnostics.extend(_record_pending(
                    store, session_id, scope, message_id,
                    dataclass_replace(cand, mtype=mtype, confidence=0.3),
                    reason="ambiguous %s correction; candidates: %s" % (
                        mtype, ", ".join(_typed_concepts(store, scope, mtype)))))
                continue
            slot_resolved = True
        diagnostics.extend(_apply_confirmed(
            store, session_id, scope, message_id, cand, concept, mtype,
            resolve=not slot_resolved))
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
        # Personal facts live in the global scope; an assistant claim
        # contradicting one is a conflict just the same.
        known = _tracked_concepts(_inherit_pool(store, session_id))
        concept = resolve_concept(known, concept)
        currents = store.fetch_states(
            scope=session_id, concept=concept, status="current"
        )
        if not currents:
            currents = store.fetch_states(
                scope="global", concept=concept, status="current")
        existing_unresolved = store.fetch_states(
            scope=session_id, concept=concept, status="unresolved"
        )
        if not existing_unresolved:
            existing_unresolved = store.fetch_states(
                scope="global", concept=concept, status="unresolved")
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
            confidence=0.3,
        )
        diagnostics.append(
            f"state_conflict: {concept} current={currents[0].value!r} "
            f"claimed={value!r} (state {new_id})"
        )
    except Exception as e:
        diagnostics.append(f"state_failed: state write failed: {e}")
    return diagnostics


# --- Update claims must match what was stored ----------------------------
# "Your favorite color has been updated to violet!" is a claim about a
# write, not about the world. Persistence happens in the Mind and can fail
# or refuse an ambiguous value, so the loop checks the claim against the
# stored current value before the reply reaches the user. The patterns are
# deliberately narrow: only completed updates, never a future or
# conditional statement ("I'll update it when you tell me").
_UPDATE_CLAIM_PATTERNS = (
    re.compile(
        r"\b(?:i(?:'ve| have|'d)?\s+)?(?:updated|changed|set|saved|recorded"
        r"|noted)\s+(?:your|the user'?s)\s+(?:new\s+)?"
        r"((?:fav|favorite|favourite)\s+[a-z][a-z'\-]*)"
        r"\s+to\s+([^.!?\n,]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:your|the user'?s)\s+(?:new\s+)?"
        r"((?:fav|favorite|favourite)\s+[a-z][a-z'\-]*)"
        r"\s+(?:(?:has|have|was|were)\s+been\s+(?:updated|changed|set)"
        r"\s+to|is\s+now|is\s+(?:updated|changed|set)\s+to)\s*"
        r"([^.!?\n,]+)",
        re.IGNORECASE,
    ),
)


def _favorite_concept(noun: str) -> Optional[str]:
    """Canonical stored concept for a claimed "favorite X" update."""
    words = (noun or "").lower().split()
    if not words or words[0] not in ("fav", "favorite", "favourite"):
        return None
    rest = " ".join(words[1:]).strip()
    if not rest:
        return None
    canonical = slots_mod.favorite_noun(rest)
    if not canonical:
        return None
    return "favorite " + canonical


def preference_update_claim_error(store: Any, text: str) -> Optional[str]:
    """Correction when a completed update claim has no stored change.

    Returns None for anything that is not a completed "favorite X" update
    claim, for a claim that matches the stored current value, and when the
    store cannot be read (the caller stays silent rather than risk a false
    correction). When the claim is unbacked, the returned text tells the
    model exactly what is stored so it can answer honestly.
    """
    if not isinstance(text, str) or not text:
        return None
    claimed = None
    for pattern in _UPDATE_CLAIM_PATTERNS:
        match = pattern.search(text)
        if match:
            claimed = (match.group(1), match.group(2))
            break
    if claimed is None:
        return None
    concept = _favorite_concept(claimed[0])
    if concept is None:
        return None
    value = normalize_value(claimed[1].strip())
    if not value:
        return None
    try:
        rows = store.fetch_states(scope="global", status="current")
    except Exception:
        return None
    current = [s for s in rows if (s.concept or "") == concept]
    if current and (current[0].value or "").strip().lower() == value.lower():
        return None
    if not current:
        return (
            "unbacked_memory_change: your reply says the user's %s was "
            "updated to %r, but no such fact is stored and nothing was "
            "saved. Do not claim it was saved; say you do not have it, or "
            "ask the user to state it." % (concept, value)
        )
    return (
        "unbacked_memory_change: your reply says the user's %s was updated "
        "to %r, but the stored current value is %r. That change was not "
        "saved. Do not claim it happened; use the stored value or ask the "
        "user to restate the change." % (concept, value, current[0].value)
    )
