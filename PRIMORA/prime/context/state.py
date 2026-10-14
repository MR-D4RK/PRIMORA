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

# Minimum rationale length: avoids "because yes" becoming memory.
_RATIONALE_MIN_LEN = 12


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
    "preference for pizza")."""
    if concept in _TYPE_WORDS:
        return False
    for word in _TYPE_WORDS:
        if concept.startswith(word + " for "):
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
    return concept, value


def _split_rationale(text: str) -> tuple[str, str]:
    """Split 'head because rationale'. Returns (head, rationale or '')."""
    parts = re.split(r"\s+[Bb]ecause\s+", text, maxsplit=1)
    if len(parts) != 2:
        return text, ""
    head, rationale = parts[0].strip(), parts[1].strip().rstrip(".")
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

    cands = _extract_legacy(head, scope)
    if not cands:
        cands = _extract_extended(head, scope)
    if not cands and corrected:
        # Bare correction ("Actually, green."): a value with no parseable
        # concept. Resolve by inheritance in apply, else hold pending.
        val = normalize_value(head)
        if val and len(val.split()) <= 12:
            cands = [StateCandidate("", val, "correct", scope)]
    if corrected:
        # An explicit correction prefix marks a re-asserted statement as a
        # correction of the tracked fact, so "Actually, I prefer green."
        # updates the single tracked concept of that type instead of
        # establishing an additional one.
        cands = [
            dataclass_replace(c, kind="correct") if c.kind == "establish"
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
            inner = _extract_legacy(stripped_head, scope)
            if not inner:
                inner = _extract_extended(stripped_head, scope)
            if inner:
                return [dataclass_replace(c, auto_confirm=False,
                                          confidence=0.3) for c in inner]
        if _HEDGE_RE.search(work):
            trimmed = work[:200].strip()
            if trimmed:
                return [StateCandidate(
                    "", trimmed, "establish", scope, UNCERTAIN,
                    auto_confirm=False, confidence=0.3)]
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
        currents = store.fetch_states(
            scope=scope, concept=concept, status="current"
        )
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
    if currents and currents[0].value.lower() == cand.value.lower():
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
        known = _tracked_concepts(store.fetch_states(scope=session_id))
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
            confidence=0.3,
        )
        diagnostics.append(
            f"state_conflict: {concept} current={currents[0].value!r} "
            f"claimed={value!r} (state {new_id})"
        )
    except Exception as e:
        diagnostics.append(f"state_failed: state write failed: {e}")
    return diagnostics
