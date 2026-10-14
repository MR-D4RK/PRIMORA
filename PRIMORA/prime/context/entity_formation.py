"""M5 deterministic formation from user conversation messages.

This module connects the existing M5 infrastructure (entities, aliases,
relationships) to the live message path. It contains **extraction and
orchestration only**: every write goes through the existing
``entities_mod`` / ``relationships_mod`` APIs, so provenance, authority,
scope, temporal validity, and the candidate/current lifecycle remain owned
by the existing implementation.

Design constraints honoured here:

* User messages only. Assistant, system, and tool turns never establish an
  M5 fact.
* Explicit wording only. Every entity pattern requires the user to name the
  type, either as ``my <type> is <NAME>`` or ``<NAME> is a <type>``. Nothing
  is inferred from capitalisation, from a bare noun, or from context.
* Exact identity only. An existing entity is reused on an exact normalized
  name match inside the same scope. Similar names are never merged, and this
  module never calls the merge API.
* Session scope by default. No automatic promotion to global.
* Part-of is confirmed, because it comes from an explicit user assertion.
  Memory-to-entity ``concerns`` links are left as candidates, because
  containment between a state and an entity is an inference rather than an
  assertion. Both use the existing create-then-confirm lifecycle.

NOT IMPLEMENTED: semantic entity inference, automatic merging, session to
global promotion, and any model or LLM involvement of any kind.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from . import entities as entities_mod
from . import relationships as relationships_mod


# Type words the user must say explicitly, mapped to the M5 entity type.
# Nothing outside this table can be formed automatically.
_EXPLICIT_TYPE_WORDS: dict[str, str] = {
    "name": "PERSON",
    "person": "PERSON",
    "agent": "AGENT",
    "project": "PROJECT",
    "component": "COMPONENT",
    "file": "FILE",
}

# "my project is PRIMORA"  /  "my agent is called X" is NOT matched
_MY_TYPE_IS = re.compile(
    r"^\s*my\s+(" + "|".join(sorted(_EXPLICIT_TYPE_WORDS)) + r")s?"
    r"(?:\s+(?:is|are|'s))(?:\s+called)?\s+(?P<name>[^,.;!?]+)",
    re.IGNORECASE,
)

# "Aurora is an agent"  /  "Verse is a component"
_NAME_IS_TYPE = re.compile(
    r"^\s*(?P<name>[^,.;!?]+?)\s+(?:is|are)\s+(?:an?\s+|the\s+)?"
    r"(?P<type>" + "|".join(sorted(_EXPLICIT_TYPE_WORDS)) + r")s?\s*$",
    re.IGNORECASE,
)

# Explicit aliasing only. "the Verse interface" / "Aurora's world" are
# deliberately NOT patterns: those need semantic interpretation.
_ALSO_CALLED = re.compile(
    r"^\s*(?P<name>[^,.;!?]+?)\s+(?:is\s+also\s+(?:called|known\s+as)"
    r"|(?:is\s+)?also\s+called|aka)\s+(?P<alias>[^,.;!?]+)",
    re.IGNORECASE,
)
_I_CALL_HEAD = re.compile(r"^\s*i\s+call\s+(?P<rest>[^,.;!?]+?)\s*$", re.IGNORECASE)
_DETERMINER = re.compile(r"\b(?:the|a|an)\b", re.IGNORECASE)

# Determiners split "I call <name> the <alias>". Splitting on the LAST
# determiner keeps the whole leading noun phrase as the name, so
# "I call the Verse interface the Verse world" resolves to
# name="the Verse interface", alias="the Verse world".

# A filesystem path is syntactically unambiguous, so a pasted path is an
# explicit reference to a file. Requires a separator or a drive letter.
# Kept as the behavioural oracle for _scan_path_tokens: the scanner below
# reproduces this pattern's exact finditer semantics without the engine's
# quadratic re-scan of every start position inside long separator-free runs.
_PATH_TOKEN = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s\"'<>|]+"
    r"|(?:[\w.\-]+[\\/])+[\w.\-]+\.[A-Za-z0-9]{1,8})"
)

# Single-character probes for the scanner. They exist so the scanner's
# character classes are the regex engine's classes exactly, Unicode \w
# semantics included, rather than an approximation.
_PROBE_WORD = re.compile(r"[\w.\-]")          # alt-2 segment character
_PROBE_EXT = re.compile(r"[A-Za-z0-9]")       # file-extension character
_PROBE_DRIVE = re.compile(r"[A-Za-z]")        # alt-1 drive letter
_PROBE_TAIL = re.compile(r"[^\s\"'<>|]")      # alt-1 continuation character
_ALT1_SEP = frozenset("\\/")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MAX_NAME = 80
_MAX_ALIAS = 80

# Never a real entity name. "This is an agent." is a grammatical statement
# about something already known, not an assertion that an entity called
# "This" exists, so demonstratives, pronouns, and generic nouns are refused.
_NOT_A_NAME = frozenset({
    "this", "that", "these", "those", "it", "they", "them", "he", "she",
    "we", "us", "you", "i", "me", "my", "our", "your", "their", "his",
    "her", "its", "something", "anything", "everything", "nothing", "none",
    "one", "ones", "thing", "things", "stuff", "something else", "there",
    "here", "what", "which", "who", "whom", "whose", "everything else",
    "all", "any", "some", "each", "every", "other", "another", "same",
})


@dataclass
class FormationResult:
    diagnostics: list[str]
    entity_ids: dict[str, int]      # normalized name -> entity id
    alias_ids: list[int]
    relationship_ids: list[int]


def _sentences(text: str) -> list[str]:
    """Split into sentences with the terminal delimiter removed.

    The delimiter must be dropped, not just trimmed of surrounding space:
    `_NAME_IS_TYPE` and `_ALSO_CALLED` are whole-sentence patterns anchored
    with ``\\s*$``, so a retained trailing period would make every sentence
    fail to match. Path extraction is unaffected because it reads the raw
    text rather than the segment list.
    """
    parts = [
        p.strip().rstrip(".!?").strip()
        for p in _SENTENCE_SPLIT.split(text.strip())
    ]
    parts = [p for p in parts if p]
    if not parts and text.strip():
        stripped = text.strip().rstrip(".!?").strip()
        return [stripped] if stripped else []
    return parts


def _clean_name(raw: str) -> str:
    text = " ".join(raw.split()).strip()
    return text[:_MAX_NAME].strip()


def _scan_path_tokens(text: str) -> list[str]:
    r"""Every _PATH_TOKEN match, in order, in O(len(text)) time.

    Equivalent to ``[m.group(0) for m in _PATH_TOKEN.finditer(text)]``,
    produced by walking the pattern's structure directly instead of letting
    the engine retry every start position (MEMORY_AUDIT C1: a 25k-char
    separator-free run took seconds, and longer inputs hung).

    Why this is exact, alternative by alternative:

    *   Alt 1 (``[A-Za-z]:[\\/]tail``) has no backtracking that can change
        the outcome: the tail class greedily consumes until a forbidden
        character, and nothing follows it in the pattern.
    *   Alt 2 (``(?:[\\w.\\-]+[\\\\/])+[\\w.\\-]+\\.[A-Za-z0-9]{1,8}``):
        separators are outside ``[\\w.\\-]``, so shortening any single
        segment run can never land on a separator -- only the *number* of
        leading ``(run + separator)`` iterations is backtrackable. The
        engine therefore tries the greedy chain of iterations first, then
        the extension at each chain position from longest chain to
        shortest; each extension attempt is resolved in O(1) from arrays
        precomputed right-to-left.
    *   A failed start rules out every later start up to and including the
        end of the final chain run: those starts attempt the same or a
        subset of the extension positions just tried. The only alternative
        that can still fire inside that span is alt 1 starting at the
        character before a colon that ends the final run (``b:/c`` inside
        ``ab:/c``), so that single position is checked before skipping.

    The original pattern is kept above as the oracle; the differential
    tests compare this scanner against it.
    """
    n = len(text)
    if n == 0:
        return []

    # wend[i]: end (exclusive) of the maximal [\w.\-] run starting at i.
    wend = [0] * (n + 1)
    wend[n] = n
    for i in range(n - 1, -1, -1):
        wend[i] = i if _PROBE_WORD.match(text, i) is None else wend[i + 1]

    def _valid_dot(j: int) -> bool:
        # A dot the final \.[A-Za-z0-9]{1,8} can use, at position j.
        return (text[j] == "." and j + 1 < n
                and _PROBE_EXT.match(text, j + 1) is not None)

    # last_dot_before[i]: largest usable dot position strictly before i.
    last_dot_before: list[Optional[int]] = [None] * (n + 1)
    for i in range(1, n + 1):
        last_dot_before[i] = (
            i - 1 if _valid_dot(i - 1) else last_dot_before[i - 1]
        )

    # arun[i]: length of the ASCII-alnum run starting at i (for the
    # greedy {1,8} extension, which ends the match -- nothing follows it).
    arun = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        arun[i] = (0 if _PROBE_EXT.match(text, i) is None
                   else arun[i + 1] + 1)

    def _alt1_end(i: int) -> Optional[int]:
        # [A-Za-z]:[\\/] with at least one [^\s"'<>|] after the separator.
        if i + 3 >= n or _PROBE_DRIVE.match(text, i) is None:
            return None
        if text[i + 1] != ":" or text[i + 2] not in _ALT1_SEP:
            return None
        j = i + 3
        while j < n and _PROBE_TAIL.match(text, j) is not None:
            j += 1
        return j if j > i + 3 else None

    def _extension_end(p: int) -> Optional[int]:
        # [\w.\-]+\.[A-Za-z0-9]{1,8} starting at p, engine semantics:
        # longest run first, so the largest usable dot inside the run.
        if p >= n or _PROBE_WORD.match(text, p) is None:
            return None
        r = last_dot_before[wend[p]]
        if r is None or r < p + 1:
            return None
        return r + 1 + min(8, arun[r + 1])

    out: list[str] = []
    i = 0
    while i < n:
        end = _alt1_end(i)
        if end is not None:
            out.append(text[i:end])
            i = end
            continue

        if _PROBE_WORD.match(text, i) is not None:
            # Greedy segment chain: each (run + separator) iteration is
            # forced, so the chain is unique; the loop stops where the
            # next iteration could not start.
            chain = [i]
            p = i
            while True:
                e = wend[p]
                if (e < n and text[e] in _ALT1_SEP
                        and e + 1 < n
                        and _PROBE_WORD.match(text, e + 1) is not None):
                    p = e + 1
                    chain.append(p)
                else:
                    break

            # Alt 2 needs at least one leading segment (the group is +),
            # so the extension is tried at chain positions 1..G only,
            # longest chain first, exactly as the engine backtracks.
            end = None
            for k in range(len(chain) - 1, 0, -1):
                end = _extension_end(chain[k])
                if end is not None:
                    break
            if end is not None:
                out.append(text[i:end])
                i = end
                continue

            # Alt 2 failed at i, so it fails at every start up to the end
            # of the final chain run (same or subset of extensions). The
            # only alternative left inside that span is alt 1 at the
            # character before a colon ending that run ("b:/c" in
            # "ab:/c"); jump there so the next iteration tries it first.
            # The exact alt-1 predicate is reused so this jump can never
            # land on a start that fails and loops back to itself.
            g_end = wend[chain[-1]]
            j = g_end - 1
            if (g_end < n and text[g_end] == ":"
                    and _alt1_end(j) is not None):
                i = j
            else:
                i = g_end + 1
            continue

        # Neither alternative can start here: alt 1 needs a drive letter
        # (a word character), alt 2 needs a segment character.
        i += 1
    return out


def _path_entities(text: str) -> list[str]:
    """Basenames of path-like tokens, de-duplicated, order preserved."""
    out: list[str] = []
    for token in _scan_path_tokens(text):
        token = token.rstrip(".,;:!?)")
        if len(token) > 260 or not token:
            continue
        base = re.split(r"[\\/]", token)[-1]
        base = base.strip()
        if base and "." in base and len(base) <= 120 and base not in out:
            out.append(base)
    return out


def _find_existing(store: Any, scope: str, normalized: str) -> list[Any]:
    """Every non-archived entity in this scope with this exact name.

    `entities_mod.resolve` only surfaces *active* entities, which is the
    right rule for relationship endpoints but the wrong rule for formation:
    a repeat of the same assertion must not create a second candidate.
    """
    try:
        rows = store.fetch_entities(
            scope=scope, normalized_name=normalized, limit=10
        )
    except Exception:
        return []
    return [r for r in rows if r.status in ("active", "candidate")]


def _ensure_entity(
    store: Any, session_id: str, name: str, entity_type: str,
    source_ids: list[int], timestamp: Optional[float],
) -> tuple[Optional[int], list[str]]:
    """Reuse an exact match, otherwise create and confirm one candidate.

    Returns the entity id, or None when the name is ambiguous. An ambiguous
    name is never merged or guessed at: the caller records a diagnostic and
    formation for that name stops.
    """
    diagnostics: list[str] = []
    cleaned = _clean_name(name)
    normalized = entities_mod.normalize_entity_name(cleaned)
    if not normalized:
        return None, ["entity_skipped: empty name"]

    existing = _find_existing(store, session_id, normalized)
    if existing:
        if len({r.entity_id for r in existing}) > 1:
            return None, [
                f"entity_skipped: ambiguous existing name {normalized!r}"
            ]
        return existing[0].entity_id, [
            f"entity_reused: {cleaned} (entity {existing[0].entity_id})"
        ]

    try:
        entity_id, diags = entities_mod.create_candidate(
            store, session_id, cleaned, entity_type, source_ids,
            authority="user", scope=None, timestamp=timestamp,
        )
    except Exception as exc:
        return None, [f"entity_failed: {exc}"]
    diagnostics.extend(diags)
    # Explicit user assertion of a named type takes the existing approved
    # create-then-confirm path. confirm() itself re-checks authority and
    # provenance, so a candidate that fails either check is refused here too.
    try:
        _record, confirm_diags = entities_mod.confirm(store, entity_id)
        diagnostics.extend(confirm_diags)
    except Exception as exc:
        diagnostics.append(f"entity_not_confirmed: {exc}")
    return entity_id, diagnostics


def _resolve_active(store: Any, scope: str, name: str) -> Optional[int]:
    """Resolve an already-established entity by exact name, or None."""
    rows, ambiguous = entities_mod.resolve(store, scope, name)
    if ambiguous or len(rows) != 1:
        return None
    return rows[0].entity_id


def _extract_entity_assertions(
    text: str,
) -> list[tuple[str, str]]:
    """(name, entity_type) pairs from explicit user wording, in order."""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(raw_name: str, entity_type: str) -> None:
        cleaned = _clean_name(raw_name)
        normalized = entities_mod.normalize_entity_name(cleaned)
        if not normalized or len(normalized) > _MAX_NAME:
            return
        # A name that is itself a type word is not an entity name.
        if normalized in _EXPLICIT_TYPE_WORDS:
            return
        # Demonstratives, pronouns, and generic nouns are not identities.
        if normalized in _NOT_A_NAME:
            return
        key = (normalized, entity_type)
        if key in seen:
            return
        seen.add(key)
        found.append((cleaned, entity_type))

    for sentence in _sentences(text):
        match = _MY_TYPE_IS.match(sentence)
        if match:
            word = match.group(1).lower()
            add(match.group("name"), _EXPLICIT_TYPE_WORDS[word])
            continue
        match = _NAME_IS_TYPE.match(sentence)
        if match:
            word = match.group("type").lower()
            add(match.group("name"), _EXPLICIT_TYPE_WORDS[word])

    for base in _path_entities(text):
        add(base, "FILE")
    return found


def _split_i_call(rest: str) -> Optional[tuple[str, str]]:
    """Split "I call <rest>" into (name, alias), or None if ambiguous.

    Two accepted shapes, both explicit:
      * a determiner introduces the alias -- "the Verse interface the Verse
        world" splits at the LAST determiner, keeping the whole leading noun
        phrase as the name;
      * a bare two-token remainder -- "Bob The Builder" is too ambiguous and
        is refused rather than guessed at.
    """
    matches = list(_DETERMINER.finditer(rest))
    if matches:
        cut = matches[-1].start()
        name, alias = rest[:cut].strip(), rest[cut:].strip()
        if name and alias:
            return name, alias
        return None
    tokens = rest.split()
    if len(tokens) == 2:
        return tokens[0], tokens[1]
    return None


def _extract_alias_assertions(text: str) -> list[tuple[str, str]]:
    """(entity_name, alias_text) pairs from explicit aliasing only."""
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sentence in _sentences(text):
        pair: Optional[tuple[str, str]] = None
        match = _ALSO_CALLED.match(sentence)
        if match:
            pair = (match.group("name"), match.group("alias"))
        else:
            head = _I_CALL_HEAD.match(sentence)
            if head:
                pair = _split_i_call(head.group("rest"))
        if not pair:
            continue
        name = _clean_name(pair[0])
        alias = _clean_name(pair[1])
        if not name or not alias or len(alias) > _MAX_ALIAS:
            continue
        key = (entities_mod.normalize_entity_name(name),
               entities_mod.normalize_entity_name(alias))
        if key[0] == key[1] or key in seen:
            continue
        seen.add(key)
        found.append((name, alias))
    return found


def _state_ids_for_message(
    store: Any, session_id: str, message_id: int,
) -> list[int]:
    try:
        rows = store.fetch_states(scope=session_id, limit=500)
    except Exception:
        return []
    out: list[int] = []
    for row in rows:
        if message_id in (row.source_ids or []):
            out.append(row.state_id)
    return out


def _mentions(text: str, name: str) -> bool:
    """Deterministic containment test between a state and an entity name.

    Punctuation is flattened to spaces first. Without that, a state value of
    "PRIMORA. The deployment window is Friday" would fail a whole-token test
    against "primora" purely because of the full stop, which is the same
    terminator class of bug as the sentence segmenter.
    """
    normalized_name = _word_key(name)
    if len(normalized_name) < 2:
        return False
    haystack = _word_key(text)
    if not haystack:
        return False
    return f" {normalized_name} " in f" {haystack} "


def _word_key(text: str) -> str:
    """Lowercase, alphanumeric-token form used for containment tests."""
    return " ".join(
        re.sub(r"[^0-9a-z]+", " ", entities_mod.normalize_entity_name(text)).split()
    )


_LIVE_RELATIONSHIP_STATUSES = ("candidate", "current", "unresolved")


def _relationship_exists(
    store: Any, scope: str, relationship_type: str,
    subject_kind: str, subject_id: int,
    object_kind: str, object_id: int,
) -> bool:
    """True if an equivalent live edge already exists in this scope.

    Re-processing a message must not accumulate identical edges. The unique
    index in the schema covers (kind, ref, model) style identity for vectors,
    not relationship endpoints, so this check is what makes formation
    idempotent for relationships.
    """
    try:
        rows = store.fetch_relationships(
            scope=scope,
            relationship_type=relationship_type,
            subject_kind=subject_kind,
            subject_id=subject_id,
            object_kind=object_kind,
            object_id=object_id,
            limit=20,
        )
    except Exception:
        return False
    return any(r.status in _LIVE_RELATIONSHIP_STATUSES for r in rows)


def form_from_user_message(
    store: Any,
    session_id: str,
    message_id: int,
    text: str,
    timestamp: Optional[float] = None,
) -> FormationResult:
    """Form M5 entities, aliases, and relationships from one user message.

    ``store`` is an existing ``ContextStore``. Nothing here writes outside
    the M5 APIs. Every step is wrapped so that a failure can never prevent
    M4 state formation, which has already run by the time this is called.
    """
    diagnostics: list[str] = []
    entity_ids: dict[str, int] = {}
    alias_ids: list[int] = []
    relationship_ids: list[int] = []

    if not isinstance(text, str) or not text.strip():
        return FormationResult(["m5_skipped: empty message"], {}, [], [])

    stamp = timestamp if isinstance(timestamp, (int, float)) else time.time()
    source_ids = [int(message_id)]
    scope = session_id  # _scope_key maps None to session_id; explicit is clearer.

    # 1. Entities, from explicit wording only.
    try:
        for name, entity_type in _extract_entity_assertions(text):
            entity_id, diags = _ensure_entity(
                store, session_id, name, entity_type, source_ids, stamp
            )
            diagnostics.extend(diags)
            if entity_id is not None:
                entity_ids.setdefault(
                    entities_mod.normalize_entity_name(name), entity_id
                )
    except Exception as exc:
        diagnostics.append(f"entity_failed: formation aborted: {exc}")

    # 2. Aliases, candidates only, and only against an established entity.
    try:
        for name, alias in _extract_alias_assertions(text):
            entity_id = entity_ids.get(
                entities_mod.normalize_entity_name(name)
            ) or _resolve_active(store, scope, name)
            if entity_id is None:
                diagnostics.append(
                    f"alias_skipped: no established entity for {name!r}"
                )
                continue
            alias_id, diags = entities_mod.add_alias_candidate(
                store, entity_id, alias, session_id, source_ids,
                authority="user", scope=None, timestamp=stamp,
            )
            diagnostics.extend(diags)
            alias_ids.append(alias_id)
    except Exception as exc:
        diagnostics.append(f"alias_failed: formation aborted: {exc}")

    # 3. part_of, via the existing text proposer, then the existing confirm.
    #    The proposer resolves endpoints, so entities must already be active.
    #    Existence is checked first, using the module's own grammar so the two
    #    can never disagree about what a part_of sentence means.
    try:
        for sentence in _sentences(text):
            match = relationships_mod._PART_OF_RE.match(sentence)
            if match is None:
                continue
            subject = entities_mod.resolve(
                store, scope,
                entities_mod.normalize_entity_name(match.group("subject")),
            )[0]
            obj = entities_mod.resolve(
                store, scope,
                entities_mod.normalize_entity_name(match.group("object")),
            )[0]
            if len(subject) == 1 and len(obj) == 1 and _relationship_exists(
                store, scope, "part_of", "entity", subject[0].entity_id,
                "entity", obj[0].entity_id,
            ):
                diagnostics.append(
                    "relationship_reused: part_of "
                    f"{subject[0].entity_id}->{obj[0].entity_id}"
                )
                continue
            rel_id, diags = relationships_mod.propose_part_of_from_text(
                store, session_id, sentence, source_ids, scope=None,
                timestamp=stamp,
            )
            diagnostics.extend(diags)
            if rel_id is None:
                continue
            try:
                _record, confirm_diags = relationships_mod.confirm(
                    store, rel_id
                )
                diagnostics.extend(confirm_diags)
                relationship_ids.append(rel_id)
            except Exception as exc:
                diagnostics.append(f"relationship_not_confirmed: {exc}")
    except Exception as exc:
        diagnostics.append(f"relationship_failed: formation aborted: {exc}")

    # 4. concerns: an accepted state from THIS message that clearly refers to
    #    an entity established by THIS message. Left as a candidate, because
    #    containment is an inference and not a user assertion.
    try:
        state_ids = _state_ids_for_message(store, session_id, message_id)
        for state_id in state_ids:
            try:
                states = store.fetch_states(
                    scope=session_id, state_ids=[state_id], limit=1
                )
            except Exception:
                continue
            if not states:
                continue
            state = states[0]
            if state.status != "current" or state.authority != "user":
                continue
            haystack = f"{state.concept} {state.value}"
            for normalized_name, entity_id in entity_ids.items():
                if not _mentions(haystack, normalized_name):
                    continue
                if _relationship_exists(
                    store, scope, "concerns", "memory_state", state_id,
                    "entity", entity_id,
                ):
                    continue
                try:
                    rel_id, diags = relationships_mod.create_candidate(
                        store, session_id, "concerns", state_id, entity_id,
                        source_ids=source_ids, authority="user", scope=None,
                        timestamp=stamp,
                    )
                    diagnostics.extend(diags)
                    relationship_ids.append(rel_id)
                except ValueError:
                    continue
                except Exception as exc:
                    diagnostics.append(f"relationship_failed: concerns: {exc}")
    except Exception as exc:
        diagnostics.append(f"relationship_failed: concerns aborted: {exc}")

    return FormationResult(diagnostics, entity_ids, alias_ids, relationship_ids)
