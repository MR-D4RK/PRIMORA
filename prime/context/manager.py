"""ContextManager: Prime-owned working-context orchestration.

Raw history (SQLite, append-only) is the permanent source of truth.
`build()` assembles a bounded, ordered, deduplicated working view for one
model call. Never deletes raw rows. Degrades safely, never invents facts.
"""

import math
import os
import re
import time
from typing import Any, Callable, Optional

from . import entities as entities_mod
from . import entity_formation as entity_formation_mod
from . import hybrid as hybrid_mod
from . import queries as queries_mod
from . import relationships as relationships_mod
from . import scoring as scoring_mod
from . import semantic as semantic_mod
from . import slots as slots_mod
from . import state as state_mod
from . import timeline as timeline_mod
from .discourse import resolve_preference_reference
from .digest import build_extractive_digest, maybe_compact
from .episodes import extract_experience_claim, organize_experience_events
from .episode_retrieval import render_episode_answer
from .models import (
    ActiveMemory,
    ActiveMemoryItem,
    BudgetConfig,
    BuildResult,
    ContextItem,
    ContextView,
    EntityAliasRecord,
    EntityMemoryLink,
    EntityRecord,
    ExplanationPath,
    RelatedEntity,
    RelationshipContext,
    RelationshipRecord,
    RetrievalCandidate,
    RetrievalTrace,
    ShadowComparison,
    StateRecord,
    StoredMessage,
    noop_memory,
)
from .store import ContextStore

DEFAULT_SESSION_ID = "default"

HISTORY_MARKERS = (
    "previous",
    "previously",
    "prior",
    "before",
    "old",
    "used to",
    "history",
    "historical",
    "was",
    "were",
    "earlier",
    "formerly",
    "back then",
    "originally",
)

# Explicit cross-session requests name more than the current conversation.
CROSS_SESSION_PREFIX = "[from another conversation]\n"

CROSS_SESSION_MARKERS = (
    "other sessions",
    "each session",
    "all sessions",
    "every session",
    "across sessions",
    "per session",
    "both sessions",
    "all of our sessions",
)

# "What was it previously called?" — previous_titles provenance.
SESSION_HISTORY_MARKERS = (
    "previously called",
    "used to be called",
    "old name",
    "former name",
    "originally called",
    "prior name",
)

# A broad self-knowledge question ("what do you know about me?") has no
# term to overlap a stored fact with, so the overlap gate alone can never
# answer it. These markers open the session retention rule to global facts
# for that question only; a slot-naming question ("what are my favorite
# colors?") does not match and stays overlap-only.
SELF_KNOWLEDGE_RE = re.compile(
    r"\b(?:about me|about myself)\b"
    r"|\bmy (?:preferences|favorites|favourites|likes|interests|profile|"
    r"details)\b",
    re.IGNORECASE,
)

_FAVORITE_QUERY_RE = re.compile(
    r"\b(?:my\s+)?(?:fav|favorite|favourite)\s+"
    r"([a-z][a-z'\-]*(?:\s+[a-z][a-z'\-]*){0,3})",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"\?\s*$|^\s*(?:what|whats|which|who|whose|where|when|why|how|"
    r"do|did|can|could|would|tell|remind|recall|list)\b",
    re.IGNORECASE,
)
_CATEGORY_END_RE = re.compile(
    r"\b(?:is|are|was|were|do|does|did|i|you|we|they|and)\b",
    re.IGNORECASE,
)

# A request about "what we were talking about" ordinarily means this chat,
# not an open-ended search of old sessions. "Previously"/"last time"
# explicitly requests a different scope and is not intercepted.
_LOCAL_DISCUSSION_RE = re.compile(
    r"^\s*(?:hey\s+aurora[,!]?\s+)?(?:what\s+(?:were|have)\s+we\s+"
    r"(?:talking|speaking|chatting)\s+about|what\s+did\s+we\s+"
    r"(?:talk|discuss)\s+about|what\s+were\s+we\s+discussing)"
    r"[?!. ]*$", re.IGNORECASE,
)


CROSS_SESSION_CAP = 3

# Roles that carry internal control flow rather than conversation.
#
# A stored turn is either something a person said, something Aurora said, or
# the loop's own plumbing: the system instruction, and the JSON of a capability
# result. The first two are memory. The last two are not, and they are the
# reason recall is filtered below.
#
# The loop appends every capability result to its live message list as a
# ``tool`` turn, and that same list is what gets reconciled into the store. So
# a capability result becomes a stored message, becomes retrievable evidence,
# and a later recall of that evidence hands the loop its own previous output.
# Each round wraps the last, which is exponential rather than merely linear,
# and the wrapper is what ends up being tokenized for the next segment.
#
# Nothing about storing them changes: they stay in the store, stay visible to
# the ordinary context builder, and stay budgeted there. They are simply not
# evidence that a recall query may return, because a recall answer is inserted
# into the conversation and an answer must be evidence, never the machinery
# that produced a previous answer.
INTERNAL_ROLES = frozenset({"tool", "system"})

# Per-result character cap for recall output.
#
# Recall answers are pasted into a tool turn, so the cap is what keeps a
# remembered passage proportionate to the question that retrieved it. This is
# not the input ceiling doing the work: the ceiling is applied later, during
# context assembly, and by then the text has already been tokenized to measure
# it. Bounding it at the point of recall keeps the measurement itself cheap.
#
# Sized against the retrieval path's own bounds: the default limit of 5 stays
# inside a normal tool turn, and even the hard maximum of 12 remains two orders
# of magnitude below a tokenizer's context length, so no combination of limit
# and passage size can produce an oversized encode.
RECALL_MAX_CONTENT_CHARS = 1200

# Appended when a passage is cut, so a shortened answer is visibly shortened
# rather than silently passing for the whole of what was stored.
RECALL_TRUNCATION_MARKER = "\n[truncated]"


def cross_session_cap() -> int:
    """The approved cross-session admission cap.

    Policy, not a tuning knob: the cap has always been 3 and the M6
    calibration measured no case where a foreign item outranks relevant
    local evidence, so the live switch keeps it unchanged.
    """
    return CROSS_SESSION_CAP


# Start compacting once uncovered history passes this many messages.
COMPACT_AFTER_MESSAGES = 50


def _newest_per_scope_concept(states):
    """Deduplicate current states to newest row per (scope, concept).

    Returns (deduped, [(scope, concept, kept_id), ...]). Shared by the
    assembly path and the Active Memory view so both agree on currentness.
    """
    newest: dict[tuple[str, str], object] = {}
    dupes: list[tuple[str, str, int]] = []
    for st in states:
        key = (st.scope, st.concept)
        if key not in newest or st.state_id > newest[key].state_id:
            if key in newest:
                dupes.append((st.scope, st.concept,
                              max(st.state_id, newest[key].state_id)))
            newest[key] = st
    return list(newest.values()), dupes


class ContextTooLargeError(Exception):
    """The current user message alone exceeds the hard input ceiling."""


def estimate_tokens(text: str) -> int:
    """Conservative fallback estimator used only when no real counter works."""
    return max(1, len(text) // 4)


# Startup transcript restore: latest messages re-injected after a restart
# (~15 exchanges) and the estimator budget they may occupy. Older history
# stays stored and retrievable, just not injected into every request.
TRANSCRIPT_RELOAD_LIMIT = 30
TRANSCRIPT_TOKEN_BUDGET = 600


class ContextManager:
    """Prime-owned context infrastructure."""

    def __init__(
        self,
        db_path=None,
        budget: Optional[BudgetConfig] = None,
        memory_provider=None,
        shadow_retrieval: Optional[bool] = None,
        semantic_index=None,
        hybrid_retrieval: Optional[bool] = None,
        include_foreign_history: bool = True,
    ):
        self._db_path = db_path
        self._budget = budget or BudgetConfig.from_env()
        self._memory_provider = memory_provider
        self._store: Optional[ContextStore] = None
        # M6 LIVE. Hybrid retrieval is the returned result. Defaults on;
        # set hybrid_retrieval=False (or PRIMORA_HYBRID_RETRIEVAL=0) to roll
        # back to the lexical-only path, which is retained unchanged.
        if hybrid_retrieval is None:
            hybrid_retrieval = os.environ.get(
                "PRIMORA_HYBRID_RETRIEVAL", ""
            ).strip().lower() not in ("0", "false", "no", "off")
        self._hybrid_retrieval = bool(hybrid_retrieval)
        # M6 shadow/comparison. Off unless explicitly enabled, by constructor
        # argument or by PRIMORA_HYBRID_SHADOW=1. It never changes what
        # Aurora receives: it only records the lexical-vs-hybrid comparison.
        if shadow_retrieval is None:
            shadow_retrieval = os.environ.get(
                "PRIMORA_HYBRID_SHADOW", ""
            ).strip().lower() in ("1", "true", "yes", "on")
        self._shadow_retrieval = bool(shadow_retrieval)
        # Isolation escape hatch for history-free harnesses. Stored facts and
        # their sources still flow; only other sessions' verbatim conversation
        # is withheld. The live runtime never turns this off, so the normal
        # cross-session provenance tag is unchanged.
        self._include_foreign_history = bool(include_foreign_history)
        self._semantic_index = semantic_index
        self._shadow_records: list[ShadowComparison] = []
        self._retrieval_traces: list[RetrievalTrace] = []

    # ------------------------------------------------------------------
    # M6 live-switch surface
    # ------------------------------------------------------------------
    @property
    def hybrid_retrieval(self) -> bool:
        return self._hybrid_retrieval

    def set_hybrid_retrieval(self, enabled: bool) -> None:
        """Turn live hybrid retrieval on or off at runtime.

        The rollback switch. Turning it off restores the lexical-only
        admission path, which is retained verbatim.
        """
        self._hybrid_retrieval = bool(enabled)

    @property
    def include_foreign_history(self) -> bool:
        return self._include_foreign_history

    def set_include_foreign_history(self, enabled: bool) -> None:
        """Withhold (or restore) other sessions' verbatim messages.

        An explicit isolated-test switch, not a production toggle. Stored
        states, their provenance, and this session's own turns are unaffected.
        """
        self._include_foreign_history = bool(enabled)

    @property
    def shadow_retrieval(self) -> bool:
        return self._shadow_retrieval

    def set_shadow_retrieval(self, enabled: bool) -> None:
        """Turn shadow comparison on or off at runtime.

        Deliberately a first-class switch rather than an environment-only
        hack, so it can be enabled per manager without an architectural
        change. Turning it on cannot alter assembled context.
        """
        self._shadow_retrieval = bool(enabled)

    def shadow_records(self) -> list[ShadowComparison]:
        return list(self._shadow_records)

    def clear_shadow_records(self) -> None:
        self._shadow_records.clear()

    def retrieval_traces(self) -> list[RetrievalTrace]:
        return list(self._retrieval_traces)

    def learn(
        self,
        text: str,
        provenance: str,
        jev,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> dict[str, Any]:
        """Admit an Aurora-proposed lesson through the existing lifecycle.

        Milestone 3. This is orchestration only; it stores nothing new and
        invents no schema of its own. It calls three existing mechanisms, in
        the order the memory system already uses:

        1. ``store.append`` persists Aurora's sentence as an *assistant*
           message. Assistant-authored statements are not auto-admitted by
           formation, so the sentence cannot become memory merely by being
           written down, and its id is the provenance anchor.
        2. ``store.create_candidate`` records Jev's structured reading as a
           pending candidate carrying the provenance as its ``authority``.
        3. ``state.confirm_pending`` promotes it, and only if the lifecycle
           agrees. That is the same gate fresh extraction uses, so
           supersession and conflict handling behave identically.

        Returns a summary for Aurora. Never raises for a bad request.
        """
        if jev is None:
            return {"status": "rejected", "reason": "no_jev_configured"}

        request_provenance = provenance
        from PRIMORA.learning.jev import LearningRequest, PROVENANCE

        if provenance not in PROVENANCE:
            return {
                "status": "rejected",
                "reason": (
                    f"unknown_provenance: {provenance!r}. "
                    f"Use one of: {', '.join(PROVENANCE)}"
                ),
            }
        if not isinstance(text, str) or not text.strip():
            return {"status": "rejected", "reason": "empty_text"}

        statement = text.strip()
        try:
            store = self._get_store()
            seq = store.count(session_id) + 1
            message_id, _is_new, _diag = store.append(
                session_id, seq, "assistant", statement
            )
        except Exception as exc:
            return {"status": "rejected", "reason": f"persist_failed: {exc}"}

        try:
            outcome = jev.evaluate(
                LearningRequest(
                    text=statement,
                    provenance=request_provenance,
                    session_id=session_id,
                )
            )
        except Exception as exc:
            return {
                "status": "rejected",
                "reason": f"jev_failed: {exc}",
                "source_id": message_id,
            }

        if outcome.disposition == "reject":
            return {
                "status": "rejected",
                "reason": outcome.reason or "jev_rejected",
                "diagnostics": list(outcome.diagnostics),
                "source_id": message_id,
            }

        # The provenance policy is enforced HERE, not trusted from Jev.
        # Jev is a pluggable, model-driven component and a buggy or
        # third-party one could return "confirm" for an inference; the
        # admission layer is the trust boundary, so it decides for itself.
        from PRIMORA.learning.jev import HOLDS

        disposition = (
            "hold" if request_provenance in HOLDS else outcome.disposition
        )
        if disposition != outcome.disposition:
            diagnostics_note = [
                f"learn_held_by_policy: provenance={request_provenance} "
                "is never auto-admitted"
            ]
        else:
            diagnostics_note = []

        learned: list[dict[str, Any]] = []
        held: list[dict[str, Any]] = []
        diagnostics: list[str] = (
            list(outcome.diagnostics) + diagnostics_note
        )

        for candidate in outcome.candidates:
            if not candidate.is_wellformed():
                diagnostics.append("learn_rejected: candidate not wellformed")
                continue
            scope = "global" if candidate.scope == "global" else "session"
            try:
                candidate_id = store.create_candidate(
                    session_id,
                    scope,
                    [message_id],
                    proposed_type=candidate.memory_type or "FACT",
                    proposed_concept=candidate.concept,
                    proposed_value=candidate.value,
                    proposed_rationale=candidate.rationale,
                    authority=request_provenance,
                    confidence=candidate.confidence,
                    status="pending",
                )
            except Exception as exc:
                diagnostics.append(f"learn_failed: candidate write: {exc}")
                continue

            record = {
                "concept": candidate.concept,
                "candidate_id": candidate_id,
                "provenance": request_provenance,
            }
            if disposition == "hold":
                held.append(record)
                continue

            # The existing admission gate. It re-validates the candidate and
            # refuses anything without a concept, a type, or provenance.
            from PRIMORA.prime.context import state as state_mod

            confirm = state_mod.confirm_pending(store, session_id, candidate_id)
            diagnostics.extend(confirm)
            if any(d.startswith("state_failed") for d in confirm):
                held.append(record)
            else:
                record["memory_type"] = candidate.memory_type
                learned.append(record)

        if learned:
            status = "learned"
        elif held:
            status = "held"
        else:
            status = "rejected"
        return {
            "status": status,
            "reason": outcome.reason,
            "learned": learned,
            "held": held,
            "diagnostics": diagnostics,
            "source_id": message_id,
        }

    def recall(
        self,
        query: str,
        session_id: str = DEFAULT_SESSION_ID,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Run the existing M6 hybrid retrieval for an explicit query.

        This is the retrieval half of ``build``, exposed on its own so a
        capability can ask a question without assembling a whole context.
        The retrieval decision is not reimplemented here: the same
        ``hybrid_retrieve`` call, the same channels, the same fusion, the same
        gates, and the same evidence resolution that ``build`` uses.

        A fused ``state:`` or ``entity:`` item resolves to the raw stored
        messages that support it, exactly as in ``_admit_hybrid``, so what is
        returned is always verbatim stored text with its original role rather
        than an internal id or a score.

        Internal turns are excluded. A ``tool`` or ``system`` row is the
        loop's own plumbing, not something a person said, and this result is
        about to become a tool turn itself. Returning one would let a recall
        answer carry a previous recall answer inside it, which compounds every
        round instead of adding to it. The filter runs before ``limit`` is
        spent, so a request for 5 results still returns up to 5 pieces of
        conversation rather than 5 rows of which some are plumbing.

        Each passage is capped at ``RECALL_MAX_CONTENT_CHARS``, so the answer
        stays a tool turn rather than a memory dump. Ranking is untouched: this
        filters and truncates what retrieval already chose.
        """
        if not isinstance(query, str) or not query.strip():
            return []
        limit = max(1, int(limit))
        store = self._get_store()
        query_terms = scoring_mod.build_query(query, "")
        if not query_terms:
            return []

        collected = self.collect_lexical_candidates(
            session_id, query_terms, set(), [], []
        )
        trace = hybrid_mod.hybrid_retrieve(
            self, store, self.semantic_index(), session_id, session_id, query,
            collected=collected,
            temporal=("historical"
                      if collected["history_requested"] else "current"),
        )

        # Same resolution and cross-session policy as the live build path.
        message_ids: list[int] = []
        for key in trace.fused_order:
            kind, _, raw = key.partition(":")
            try:
                ref = int(raw)
            except ValueError:
                continue
            if kind == "message":
                message_ids.append(ref)
            else:
                message_ids.extend(self._evidence_message_ids(kind, ref))
        if not message_ids:
            return []
        ordered: list[StoredMessage] = []
        seen: set[int] = set()
        for message in store.fetch_messages_by_ids(message_ids[:256]):
            if message.message_id in seen:
                continue
            if message.role in INTERNAL_ROLES:
                continue
            seen.add(message.message_id)
            ordered.append(message)

        same = [m for m in ordered if m.session_id == session_id]
        foreign = (
            [m for m in ordered if m.session_id != session_id]
            if self._include_foreign_history else []
        )
        out: list[dict[str, Any]] = []
        for message, is_foreign in (
            [(m, False) for m in same]
            + [(m, True) for m in foreign[:cross_session_cap()]]
        ):
            content = message.content
            if is_foreign:
                content = f"{CROSS_SESSION_PREFIX}{content}"
            if len(content) > RECALL_MAX_CONTENT_CHARS:
                content = (content[:RECALL_MAX_CONTENT_CHARS]
                           + RECALL_TRUNCATION_MARKER)
            out.append({"role": message.role, "content": content})
            if len(out) >= limit:
                break
        return out

    def semantic_index(self):
        if self._semantic_index is None:
            self._semantic_index = semantic_mod.SemanticIndex(
                self._get_store()
            )
        return self._semantic_index

    def semantic_index_status(self) -> dict[str, Any]:
        """Index health. Read-only, never triggers a model download."""
        try:
            return self.semantic_index().status()
        except Exception as exc:
            return {
                "available": False,
                "diagnostics": [f"semantic_status_failed: {exc}"],
            }

    def sync_semantic_index(self, scope: Optional[str] = None):
        return self.semantic_index().sync(scope)

    def maybe_sync_semantic_index(self, scope: Optional[str] = None):
        """Cheap, incremental, opportunistic index refresh.

        Called from the live retrieval path so the semantic channel is
        genuinely populated. Skipped entirely when structured memory has not
        changed since the last sync, and a no-op when the model is
        unavailable. Never raises.
        """
        try:
            return self.semantic_index().maybe_sync(scope)
        except Exception:
            return {"states": 0, "entities": 0, "skipped": 1}

    def rebuild_semantic_index(self, scope: Optional[str] = None):
        return self.semantic_index().rebuild(scope)


    @property
    def budget(self) -> BudgetConfig:
        return self._budget

    @staticmethod
    def _normalized_time(value: Optional[float], name: str) -> float:
        if value is None:
            return time.time()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a normalized numeric timestamp")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{name} must be finite")
        return normalized

    @staticmethod
    def _accepted_state(state: StateRecord) -> bool:
        return (
            state.authority == "user"
            and state.status in ("current", "superseded")
        )

    @staticmethod
    def _valid_at(state: StateRecord, at_time: float) -> bool:
        return (
            state.valid_from is not None
            and state.valid_from <= at_time
            and (
                state.valid_until is None
                or at_time < state.valid_until
            )
        )

    def _scope_states(self, scope: str, concept: str) -> list[StateRecord]:
        return self._get_store().fetch_states(scope=scope, concept=concept)

    def current(
        self,
        scope: str,
        concept: str,
        at_time: Optional[float] = None,
    ) -> Optional[StateRecord]:
        evaluation_time = self._normalized_time(at_time, "at_time")
        matches = [
            state for state in self._scope_states(scope, concept)
            if state.status == "current"
            and state.authority == "user"
            and self._valid_at(state, evaluation_time)
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda state: (state.valid_from or 0.0, state.state_id),
        )

    def as_of(
        self,
        scope: str,
        concept: str,
        at_time: float,
    ) -> Optional[StateRecord]:
        evaluation_time = self._normalized_time(at_time, "at_time")
        matches = [
            state for state in self._scope_states(scope, concept)
            if self._accepted_state(state)
            and self._valid_at(state, evaluation_time)
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda state: (state.valid_from or 0.0, state.state_id),
        )

    def history(self, scope: str, concept: str) -> list[StateRecord]:
        states = [
            state for state in self._scope_states(scope, concept)
            if self._accepted_state(state)
        ]
        return sorted(
            states,
            key=lambda state: (
                state.valid_from is None,
                state.valid_from if state.valid_from is not None else 0.0,
                state.created_at,
                state.state_id,
            ),
        )

    def previous(
        self, scope: str, concept: str
    ) -> Optional[StateRecord]:
        current_state = self.current(scope, concept)
        if current_state is None:
            return None
        states = self._scope_states(scope, concept)
        accepted = [state for state in states if self._accepted_state(state)]
        if current_state.supersedes_id is not None:
            for state in accepted:
                if state.state_id == current_state.supersedes_id:
                    return state
        if current_state.valid_from is None:
            return None
        earlier = [
            state for state in accepted
            if state.valid_from is not None
            and state.valid_from < current_state.valid_from
        ]
        if not earlier:
            return None
        nearest_time = max(state.valid_from for state in earlier)
        nearest = [
            state for state in earlier
            if state.valid_from == nearest_time
        ]
        return nearest[0] if len(nearest) == 1 else None

    def changes(
        self,
        scope: str,
        concept: str,
        since: float,
        until: float,
    ) -> list[StateRecord]:
        start = self._normalized_time(since, "since")
        end = self._normalized_time(until, "until")
        if start > end:
            raise ValueError("since must not be later than until")
        return sorted(
            [
                state for state in self._scope_states(scope, concept)
                if self._accepted_state(state)
                and state.valid_from is not None
                and start <= state.valid_from < end
            ],
            key=lambda state: (state.valid_from or 0.0, state.state_id),
        )

    def preference_events(
        self,
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> list[StateRecord]:
        """Chronological user preference changes for the future timeline UI.

        Reuses confirmed global state history; repeated equivalent claims
        only add evidence to the original event. Historical timestamps are
        first observation times, NOT inferred onset dates. No raw messages,
        assistant guesses, pending candidates, or unrelated states.
        """
        start = (self._normalized_time(since, "since")
                 if since is not None else float("-inf"))
        end = (self._normalized_time(until, "until")
               if until is not None else float("inf"))
        if start > end:
            raise ValueError("since must not be later than until")
        states = self._get_store().fetch_states(scope="global")
        return sorted(
            (state for state in states
             if self._accepted_state(state)
             and (state.concept.startswith("likes for ")
                  or state.concept.startswith("favorite "))
             and state.valid_from is not None
             and start <= state.valid_from < end),
            key=lambda state: (state.valid_from, state.state_id),
        )

    def _temporal_preference_context(
        self, request: "timeline_mod.TimelineRequest",
        *,
        max_events: int = 8,
    ) -> str:
        """Evidence-bounded timeline for a user question (read only)."""
        store = self._get_store()
        events = self.preference_events(since=request.since)
        category = request.category

        def matches(state: StateRecord) -> bool:
            return timeline_mod.category_matches(store, state, category)

        # Filter *before* limiting so unrelated newer preferences cannot
        # crowd out requested entries. Their source evidence is not assumed
        # to establish when the user's taste actually began.
        # Do not label legacy source-less rows as verified user evidence.
        eligible = [st for st in events if matches(st) and st.source_ids]
        source_ids = list(dict.fromkeys(
            st.source_ids[0] for st in eligible if st.source_ids
        ))
        evidence = {m.message_id: m for m in
                    store.fetch_messages_by_ids(source_ids)}
        eligible = [st for st in eligible
                    if st.source_ids and st.source_ids[0] in evidence
                    and evidence[st.source_ids[0]].role == "user"
                    and 0 <= evidence[st.source_ids[0]].timestamp
                    <= time.time() + 300]
        if request.mode == "started":
            eligible = [st for st in eligible
                        if st.concept.startswith("likes for ")
                        and st.source_ids
                        and st.source_ids[0] in evidence
                        and evidence[st.source_ids[0]].role == "user"
                        and timeline_mod.is_explicit_start(
                            evidence[st.source_ids[0]].content)]
        elif request.mode == "recent":
            eligible = [st for st in eligible
                        if not st.value.casefold().startswith("dislikes ")]
        # For change queries, retain both current and superseded preference
        # records so an actual before/after sequence remains visible.

        eligible.sort(key=lambda st: (st.valid_from, st.state_id),
                      reverse=request.mode != "changes")
        eligible = eligible[:max_events]
        lines = [
            "Verified user preference timeline (structured memory only):",
            "Times are when Aurora first RECORDED the statement (UTC),",
            "not necessarily when the user's interest actually began.",
            "Use only these dated records to answer the timeline question.",
            "Do not call an old favorite newly discovered, mix unrelated",
            "categories, infer motives, or use assistant guesses as evidence.",
        ]
        if not eligible:
            lines.append("No matching verified timeline events were found."
                         " Say so without inventing entries or dates.")
        for st in eligible:
            lines.append(timeline_mod.event_line(
                st, explicitly_started=(st.source_ids and
                    st.source_ids[0] in evidence and
                    evidence[st.source_ids[0]].role == "user" and
                    timeline_mod.is_explicit_start(
                        evidence[st.source_ids[0]].content))))
        return "\n".join(lines)

    def _verified_new_likes(self, request) -> str:
        """Answer explicit 'started liking' questions using user evidence only."""
        from datetime import datetime, timezone

        store = self._get_store()
        now = time.time()
        results = []
        for state in self.preference_events(since=request.since):
            if not state.concept.startswith("likes for ") or not state.source_ids:
                continue
            if request.category and not timeline_mod.category_matches(
                    store, state, request.category):
                continue
            originals = store.fetch_messages_by_ids(state.source_ids)
            starts = [m for m in originals
                      if m.role == "user" and 0 <= m.timestamp <= now + 300
                      and timeline_mod.is_explicit_start(m.content)]
            if not starts:
                continue
            source = min(starts, key=lambda m: (m.timestamp, m.message_id))
            reasons = []
            for reason in store.fetch_states(
                    scope="global", concept="rationale for " + state.concept,
                    status="current"):
                if not self._accepted_state(reason) or not reason.source_ids:
                    continue
                sources = store.fetch_messages_by_ids(reason.source_ids)
                if any(m.role == "user" and
                       (m.content.strip().rstrip(".! ") == reason.value
                        or ("because " + reason.value.casefold())
                        in m.content.casefold())
                       for m in sources):
                    reasons.append(reason.value)
            results.append((source.timestamp, source.message_id, state,
                            source.content, reasons))
        results.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if not results:
            return ("I couldn't verify any explicitly new interests in "
                    "my saved user statements for that category.")
        lines = ["Newly expressed interests (times are when you told me, "
                 "not necessarily when your interest began):"]
        for timestamp, mid, state, quote, reasons in results[:10]:
            date = datetime.fromtimestamp(timestamp, timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC")
            label = state.concept.removeprefix("likes for ")
            lines.append(f'- {label}: {date}; you said: "{quote}" '
                         f'(source message {mid}).')
            if reasons:
                lines.append("  Your stated reason: " + "; ".join(reasons[:2]))
            else:
                lines.append("  I don't have a verified reason stored.")
        return "\n".join(lines)

    def _verified_favorite_inventory(self, category: str, *,
                                     dates: bool = False,
                                     reasons: bool = False) -> str:
        """One consistent, user-source-backed view for favorite-list questions.

        The model does not supply membership, dates or motives. Unknown
        reasons remain unknown; unavailable category evidence cannot become
        a favorite merely because an assistant once mentioned it.
        """
        from datetime import datetime, timezone

        store = self._get_store()
        now = time.time()
        records = []
        # Preserve the first verified positive statement even if a later
        # clarification superseded the current value. This is a READ-only
        # history lookup; only current states establish current favorites.
        historical_by_concept = {}
        for old in store.fetch_states(scope="global"):
            if not self._accepted_state(old) or not old.source_ids:
                continue
            if old.value.casefold().startswith(("dislikes ", "no longer ")):
                continue
            historical_by_concept.setdefault(old.concept, []).append(old)
        current_states = store.fetch_states(scope="global", status="current")
        for state in current_states:
            if not self._accepted_state(state) or not self._valid_at(state, now):
                continue
            if state.concept.startswith("favorite "):
                suffix = state.concept[len("favorite "):]
                if not (suffix == category or suffix.endswith(" " + category)):
                    continue
                subject = state.value.strip()
            elif state.concept.startswith("likes for "):
                subject = state.concept[len("likes for "):]
                subject = re.split(
                    r"\s+(?:the\s+most|regardless\s+of|whether|"
                    r"too(?:\s+like)?|no\s+matter)\b", subject,
                    maxsplit=1, flags=re.I)[0].strip()
                subject = re.sub(r"^(?:the|my)\s+", "", subject, flags=re.I)
                # Category inherited from a nearby question is not proof
                # that an unrelated noun is a car. Require a recognized
                # vehicle name or an explicit category assertion in the
                # user's own cited statement (checked below).
                if not timeline_mod.verified_category_hint(category, subject):
                    explicit_sources = store.fetch_messages_by_ids(state.source_ids)
                    if not any(m.role == "user" and re.search(
                        r"\b(?:cars?|vehicles?|automobiles?)\b",
                        m.content, re.I) for m in explicit_sources):
                        continue
            else:
                continue
            if not subject or not state.source_ids:
                continue
            # Verify the *user* actually named this object. Derived state
            # labels or assistant-generated text alone are insufficient.
            tokens = re.findall(r"[\w]+", subject.casefold())
            key_word = next((t for t in reversed(tokens)
                             if len(t) >= 2 and t not in
                             ("car", "classic", "the", "most")), None)
            if not key_word:
                continue
            historical_ids = list(dict.fromkeys(
                mid for version in historical_by_concept.get(state.concept, ())
                for mid in version.source_ids
            ))
            sources = [m for m in store.fetch_messages_by_ids(historical_ids)
                       if m.role == "user" and
                       0 <= m.timestamp <= now + 300 and
                       key_word in re.findall(r"[\w]+", m.content.casefold())]
            if not sources:
                continue
            source = min(sources, key=lambda m: (m.timestamp, m.message_id))
            canonical = " ".join(tokens)
            if not canonical:
                continue
            # A direct user-stated preference dimension is evidence, not a
            # generic description of the car or Aurora's interpretation.
            stated = re.search(
                r"\b(?:in\s+terms\s+of|because\s+of|due\s+to)\s+"
                r"([a-z][a-z0-9 ,&-]{2,80})", source.content, re.I)
            actual_reasons = []
            if stated:
                actual_reasons.append(stated.group(1).strip(" .!"))
            for rationale in store.fetch_states(
                    scope="global", concept="rationale for " + state.concept,
                    status="current"):
                if not self._accepted_state(rationale) or not rationale.source_ids:
                    continue
                evidence = store.fetch_messages_by_ids(rationale.source_ids)
                if any(m.role == "user" and
                       (m.content.strip().rstrip(".! ").casefold()
                        == rationale.value.casefold()
                        or ("because " + rationale.value.casefold())
                        in m.content.casefold()) for m in evidence):
                    if rationale.value not in actual_reasons:
                        actual_reasons.append(rationale.value)
            records.append((canonical, subject, state, source, actual_reasons))

        # Group duplicate representations of the same favorite. A primary
        # favorite outranks a like, but the earliest evidence is retained.
        grouped = {}
        for key, subject, state, source, why in records:
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = (subject, state, source, list(why))
                continue
            old_subject, old_state, old_source, old_why = existing
            primary = (state if state.concept.startswith("favorite ")
                       and not old_state.concept.startswith("favorite ")
                       else old_state)
            earlier = min((source, old_source),
                          key=lambda m: (m.timestamp, m.message_id))
            grouped[key] = (old_subject, primary, earlier,
                            list(dict.fromkeys(old_why + why)))
        if not grouped:
            return ("I couldn't verify any favorite " + category +
                    " preferences in your stored user statements. "
                    "I won't guess or use my earlier replies as evidence.")
        ordered = sorted(grouped.values(), key=lambda x:
                         (x[2].timestamp, x[2].message_id))
        if dates and not reasons:
            lines = ["Here are the earliest stored user-statement dates I "
                     "can verify for this category (UTC):"]
            for subject, state, source, _why in ordered[:30]:
                date = datetime.fromtimestamp(source.timestamp, timezone.utc)
                lines.append(f'- {date:%Y-%m-%d %H:%M} UTC: "{source.content}"')
            return "\n".join(lines)
        lines = ["Your verified favorite " + category + " preferences "
                 "(from your own statements):"]
        for subject, state, source, why in ordered[:30]:
            # Preserve acronyms and model identifiers rather than changing
            # 'ZR1' to 'Zr1'. The source quote is the primary evidence.
            label = " ".join(word.upper() if (any(c.isdigit() for c in word)
                              and not word.isdigit()) or word == "ie"
                             else word.capitalize()
                             for word in subject.split())
            role = ("favorite " + state.concept.removeprefix("favorite ")
                    if state.concept.startswith("favorite ") else "liked")
            line = (f'- {label} ({role}); you said: "{source.content}" '
                    f'(source message {source.message_id})')
            if dates:
                date = datetime.fromtimestamp(source.timestamp, timezone.utc)
                line += f'; first verified record: {date:%Y-%m-%d %H:%M} UTC'
            lines.append(line)
            if reasons:
                lines.append("  Your stated reason: " + "; ".join(why[:2])
                             if why else "  No verified reason stored.")
        lines.append("These are the preferences I could verify, not "
                     "necessarily every car you've ever mentioned.")
        return "\n".join(lines)

    def verified_personal_history(self, question: str, *,
                                  context: tuple[str, ...] = ()) -> Optional[str]:
        """Give a source-bound answer to explicit personal-history questions.

        Only confirmed USER preference states backed by stored USER messages
        qualify. Dates are those of the evidence messages (not invented onset
        times), and unknown dates fail closed. This read-only path deliberately
        avoids sending factual timestamp lookups to an outside research tool.
        """
        from datetime import datetime, timezone

        inventory_category = timeline_mod.favorite_inventory_category(
            question, context=context)
        if inventory_category:
            return self._verified_favorite_inventory(
                inventory_category,
                dates=bool(re.search(r"\b(?:when|dates?|times?)\b", question, re.I)),
                reasons=bool(re.search(r"\b(?:why|reasons?)\b", question, re.I)),
            )
        request = timeline_mod.parse_request(question, now=time.time(),
                                             context=context)
        # "Just retrieve it from memory" after a dated question is a
        # clarification, not an unrelated request for outside research.
        if request is None and re.fullmatch(
                r"\s*(?:just\s+)?(?:retrieve|check|look\s+up|find)\s+it\s+"
                r"(?:from|in)\s+(?:your\s+)?memory[.!?]*\s*",
                question, re.I):
            for previous in reversed(context[-8:]):
                candidate = timeline_mod.parse_request(
                    previous, now=time.time(), context=context)
                if candidate is not None and candidate.mode in ("dates", "claims"):
                    request = candidate
                    break
        if request is None:
            return None
        if request.mode == "started":
            return self._verified_new_likes(request)
        if request.mode not in ("dates", "claims"):
            return None
        # An ungrounded "them" is ambiguous. Never print arbitrary memories.
        if request.category == "unresolved":
            return "Which interest do you mean? I don't want to guess."
        store = self._get_store()
        now = time.time()
        target = request.target
        def normalize(value: str) -> str:
            return " ".join(re.findall(r"[\w]+", value.casefold()))
        target_text = normalize(target or "")

        # Reuse the existing state history (including superseded preferences)
        # and its message provenance. No new tables or memory extraction.
        events = self.preference_events()
        matches = []
        seen_ids = set()
        for state in events:
            if request.category and not timeline_mod.category_matches(
                    store, state, request.category):
                continue
            for original in store.fetch_messages_by_ids(state.source_ids[:32]):
                if original.role != "user" or original.message_id in seen_ids:
                    continue
                if not (0 <= original.timestamp <= now + 300):
                    continue
                # Query about Regera must not borrow Apollo's statements.
                original_words = normalize(original.content)
                # The target must be in the user's evidence, not merely in
                # the state's derived concept/value (which could be wrong).
                if target_text and target_text not in original_words:
                    subject_tail = target_text.split()[-1]
                    if len(target_text.split()) < 2 or subject_tail not in original_words.split():
                        continue
                seen_ids.add(original.message_id)
                matches.append((original.timestamp, original.message_id,
                                original.content, state.concept, state.value))
        matches.sort(key=lambda item: (item[0], item[1]))
        if not matches:
            return ("I couldn't verify that from my stored user statements. "
                    "I won't guess a date or attribute words to you.")
        if request.mode == "dates":
            earliest = matches[0]
            # strftime %-d is not portable on Windows; format it explicitly.
            when = datetime.fromtimestamp(earliest[0], timezone.utc)
            stamp = f"{when:%B} {when.day}, {when:%Y} at {when:%H:%M} UTC"
            if target:
                return (f"The earliest verified preference statement I found "
                        f"about {target} is from {stamp}. "
                        f"You said: \"{earliest[2]}\". "
                        "This is the earliest stored evidence I found, not "
                        "necessarily the first time you ever mentioned it.")
            lines = ["Here are the earliest stored user-statement dates I can "
                     "verify for this category (UTC):"]
            listed = set()
            for timestamp, _mid, content, concept, value in matches:
                # Show each distinct favorite/preference once, at its
                # earliest verified source time, not once per repeated chat.
                key = normalize(concept + (" " + value if concept.startswith("favorite ") else ""))
                if key in listed:
                    continue
                listed.add(key)
                when = datetime.fromtimestamp(timestamp, timezone.utc)
                lines.append(f"- {when:%Y-%m-%d %H:%M} UTC: \"{content}\"")
                if len(listed) >= 12:
                    break
            return "\n".join(lines)
        # Source-only question: quote what the USER said, never the model.
        lines = ["These are your verified statements from stored memory:"]
        for _when, _mid, content, _concept, _value in matches[:8]:
            lines.append(f"- \"{content}\"")
        # Reasons must have user-authored evidence, never an assistant claim.
        for _when, _mid, _content, concept, _value in matches:
            for rationale in store.fetch_states(
                    scope="global", concept="rationale for " + concept,
                    status="current"):
                if not self._accepted_state(rationale) or not rationale.source_ids:
                    continue
                sources = store.fetch_messages_by_ids(rationale.source_ids)
                if any(s.role == "user" and
                       (s.content.strip().rstrip(".! ") == rationale.value
                        or ("because " + rationale.value.casefold())
                        in s.content.casefold())
                       for s in sources):
                    reason = f'- Your stated reason: "{rationale.value}"'
                    if reason not in lines:
                        lines.append(reason)
        return "\n".join(lines)

    def when_changed(
        self, scope: str, concept: str
    ) -> Optional[float]:
        state = self.current(scope, concept)
        return state.valid_from if state is not None else None

    def what_replaced(self, state_id: int) -> Optional[StateRecord]:
        successors = sorted(
            [
                state for state in self._get_store().fetch_states()
                if state.supersedes_id == state_id
                and self._accepted_state(state)
            ],
            key=lambda state: state.state_id,
        )
        return successors[0] if successors else None

    def create_entity_candidate(
        self,
        session_id: str,
        display_name: str,
        entity_type: str,
        source_ids: Optional[list[int]] = None,
        authority: str = "user",
        scope: Optional[str] = None,
        timestamp: Optional[float] = None,
        observed_at: Optional[float] = None,
    ) -> tuple[Optional[int], list[str]]:
        try:
            return entities_mod.create_candidate(
                self._get_store(), session_id, display_name, entity_type,
                source_ids, authority, scope, timestamp, observed_at,
            )
        except Exception as exc:
            return None, [f"entity_failed: {exc}"]

    def confirm_entity(
        self, entity_id: int
    ) -> tuple[Optional[EntityRecord], list[str]]:
        try:
            return entities_mod.confirm(self._get_store(), entity_id)
        except Exception as exc:
            return None, [f"entity_failed: {exc}"]

    def archive_entity(
        self, entity_id: int, authority: str = "user"
    ) -> tuple[Optional[EntityRecord], list[str]]:
        try:
            return entities_mod.archive(
                self._get_store(), entity_id, authority
            )
        except Exception as exc:
            return None, [f"entity_failed: {exc}"]

    def merge_entity(
        self,
        losing_entity_id: int,
        canonical_entity_id: int,
        authority: str = "user",
        at_time: Optional[float] = None,
    ) -> tuple[Optional[EntityRecord], list[str]]:
        try:
            return entities_mod.merge(
                self._get_store(), losing_entity_id, canonical_entity_id,
                authority, at_time,
            )
        except Exception as exc:
            return None, [f"entity_failed: {exc}"]

    def get_entity(
        self,
        scope: str,
        entity_id: int,
        follow_merge: bool = False,
        at_time: Optional[float] = None,
    ) -> Optional[EntityRecord]:
        try:
            entity = self._get_store().fetch_entity(entity_id)
            if entity is None or entity.scope != scope:
                return None
            if follow_merge:
                return entities_mod.canonical_entity(
                    self._get_store(), scope, entity_id, at_time
                )
            return entity
        except Exception:
            return None

    def resolve_canonical_entity(
        self,
        scope: str,
        entity_id: int,
        at_time: Optional[float] = None,
    ) -> Optional[EntityRecord]:
        try:
            return entities_mod.canonical_entity(
                self._get_store(), scope, entity_id, at_time
            )
        except Exception:
            return None

    def resolve_entity(
        self,
        scope: str,
        name: str,
        at_time: Optional[float] = None,
        limit: int = 5,
    ) -> tuple[list[EntityRecord], bool]:
        try:
            return entities_mod.resolve(
                self._get_store(), scope, name, at_time, limit
            )
        except Exception:
            return [], False

    def list_aliases(
        self,
        scope: str,
        entity_id: Optional[int] = None,
        alias_text: Optional[str] = None,
        status: Optional[str] = None,
        at_time: Optional[float] = None,
        limit: int = 20,
    ) -> list[EntityAliasRecord]:
        try:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            normalized = (
                entities_mod.normalize_entity_name(alias_text)
                if alias_text is not None else None
            )
            return self._get_store().fetch_entity_aliases(
                scope=scope,
                entity_id=entity_id,
                status=status,
                normalized_alias=normalized,
                at_time=at_time,
                limit=min(limit, 100),
            )
        except Exception:
            return []

    def add_alias_candidate(
        self,
        entity_id: int,
        alias_text: str,
        session_id: str,
        source_ids: Optional[list[int]] = None,
        authority: str = "user",
        scope: Optional[str] = None,
        timestamp: Optional[float] = None,
        observed_at: Optional[float] = None,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
    ) -> tuple[Optional[int], list[str]]:
        try:
            return entities_mod.add_alias_candidate(
                self._get_store(), entity_id, alias_text, session_id,
                source_ids, authority, scope, timestamp, observed_at,
                valid_from, valid_until,
            )
        except Exception as exc:
            return None, [f"alias_failed: {exc}"]

    def confirm_alias(
        self,
        alias_id: int,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
    ) -> tuple[Optional[EntityAliasRecord], list[str]]:
        try:
            return entities_mod.confirm_alias(
                self._get_store(), alias_id, valid_from, valid_until
            )
        except Exception as exc:
            return None, [f"alias_failed: {exc}"]

    def create_relationship_candidate(
        self,
        session_id: str,
        relationship_type: str,
        subject_id: int,
        object_id: int,
        source_ids: Optional[list[int]] = None,
        authority: str = "user",
        scope: Optional[str] = None,
        timestamp: Optional[float] = None,
        observed_at: Optional[float] = None,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
    ) -> tuple[Optional[int], list[str]]:
        try:
            return relationships_mod.create_candidate(
                self._get_store(), session_id, relationship_type,
                subject_id, object_id, source_ids, authority, scope,
                "candidate", timestamp, observed_at, valid_from, valid_until,
            )
        except Exception as exc:
            return None, [f"relationship_failed: {exc}"]

    def record_relationship_conflict(
        self,
        session_id: str,
        relationship_type: str,
        subject_id: int,
        object_id: int,
        source_ids: Optional[list[int]] = None,
        authority: str = "assistant",
        scope: Optional[str] = None,
        timestamp: Optional[float] = None,
        observed_at: Optional[float] = None,
    ) -> tuple[Optional[int], list[str]]:
        try:
            return relationships_mod.create_candidate(
                self._get_store(), session_id, relationship_type,
                subject_id, object_id, source_ids, authority, scope,
                "unresolved", timestamp, observed_at,
            )
        except Exception as exc:
            return None, [f"relationship_failed: {exc}"]

    def confirm_relationship(
        self,
        relationship_id: int,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
        at_time: Optional[float] = None,
    ) -> tuple[Optional[RelationshipRecord], list[str]]:
        try:
            return relationships_mod.confirm(
                self._get_store(), relationship_id,
                valid_from, valid_until, at_time,
            )
        except Exception as exc:
            return None, [f"relationship_failed: {exc}"]

    def reject_relationship(
        self, relationship_id: int
    ) -> tuple[Optional[RelationshipRecord], list[str]]:
        try:
            return relationships_mod.reject(
                self._get_store(), relationship_id
            )
        except Exception as exc:
            return None, [f"relationship_failed: {exc}"]

    def mark_relationship_unresolved(
        self,
        relationship_id: int,
        at_time: Optional[float] = None,
    ) -> tuple[Optional[RelationshipRecord], list[str]]:
        try:
            return relationships_mod.mark_unresolved(
                self._get_store(), relationship_id, at_time
            )
        except Exception as exc:
            return None, [f"relationship_failed: {exc}"]

    def supersede_relationship(
        self,
        relationship_id: int,
        valid_until: Optional[float] = None,
        at_time: Optional[float] = None,
    ) -> tuple[Optional[RelationshipRecord], list[str]]:
        try:
            return relationships_mod.supersede(
                self._get_store(), relationship_id, valid_until, at_time
            )
        except Exception as exc:
            return None, [f"relationship_failed: {exc}"]

    def get_relationship(
        self, relationship_id: int
    ) -> Optional[RelationshipRecord]:
        try:
            return self._get_store().fetch_relationship(relationship_id)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # M5 Stage C: bounded relationship queries (read-only)
    # ------------------------------------------------------------------
    def get_entity_relationships(
        self,
        scope: str,
        entity_id: int,
        direction: str = "outbound",
        relationship_type: Optional[str] = None,
        temporal: str = "current",
        at_time: Optional[float] = None,
        limit: int = queries_mod.DEFAULT_LIMIT,
        include_global: bool = False,
        follow_merge: bool = True,
    ) -> list[RelationshipRecord]:
        """One-hop read of the stored rows naming this entity as an endpoint.

        Bounded by construction: single hop, mandatory limit, explicit
        scope, accepted status/authority/validity gates only. Never raises;
        an invalid request yields no rows.
        """
        try:
            return queries_mod.entity_relationships(
                self._get_store(), scope, entity_id, direction,
                relationship_type, temporal, at_time, limit,
                include_global, follow_merge,
            )
        except Exception:
            return []

    def get_related_entities(
        self,
        scope: str,
        entity_id: int,
        direction: str = "both",
        relationship_type: Optional[str] = None,
        temporal: str = "current",
        at_time: Optional[float] = None,
        limit: int = queries_mod.DEFAULT_LIMIT,
        include_global: bool = False,
        follow_merge: bool = True,
    ) -> list[RelatedEntity]:
        """Entities one hop away through explicit entity->entity edges.

        Inbound `part_of` is reported as the derived inverse `contains`;
        the stored row stays canonical.
        """
        try:
            return queries_mod.related_entities(
                self._get_store(), scope, entity_id, direction,
                relationship_type, temporal, at_time, limit,
                include_global, follow_merge,
            )
        except Exception:
            return []

    def get_contained_entities(
        self,
        scope: str,
        entity_id: int,
        temporal: str = "current",
        at_time: Optional[float] = None,
        limit: int = queries_mod.DEFAULT_LIMIT,
        include_global: bool = False,
        follow_merge: bool = True,
    ) -> list[RelatedEntity]:
        """Derived inverse of inbound `part_of`. Computed at read time only."""
        try:
            return queries_mod.contained_entities(
                self._get_store(), scope, entity_id, temporal, at_time, limit,
                include_global, follow_merge,
            )
        except Exception:
            return []

    def get_entity_memories(
        self,
        scope: str,
        entity_id: int,
        temporal: str = "current",
        at_time: Optional[float] = None,
        limit: int = queries_mod.DEFAULT_LIMIT,
        include_global: bool = False,
        follow_merge: bool = True,
    ) -> list[EntityMemoryLink]:
        """Memory states linked to this entity by stored `concerns` edges."""
        try:
            return queries_mod.entity_memories(
                self._get_store(), scope, entity_id, temporal, at_time, limit,
                include_global, follow_merge,
            )
        except Exception:
            return []

    def get_relationship_explanation(
        self,
        scope: str,
        state_id: int,
        direction: str = "inbound",
        temporal: str = "current",
        at_time: Optional[float] = None,
        limit: int = 10,
        include_global: bool = False,
    ) -> list[ExplanationPath]:
        """The fixed `rationale -explains-> decision` path for one state.

        Follows stored relationship rows only; it never re-derives the link
        from the legacy `"rationale for " + concept` naming convention.
        """
        try:
            return queries_mod.explanations(
                self._get_store(), scope, state_id, direction, temporal,
                at_time, limit, include_global,
            )
        except Exception:
            return []

    def get_selected_relationship_context(
        self,
        scope: str,
        entity_ids: Optional[list[int]] = None,
        state_ids: Optional[list[int]] = None,
        relationship_limit: int = queries_mod.MAX_CONTEXT_RELATIONSHIPS,
        entity_limit: int = queries_mod.MAX_CONTEXT_ENTITIES,
        memory_limit: int = queries_mod.MAX_CONTEXT_MEMORIES,
        hop_limit: int = queries_mod.MAX_TRAVERSAL_HOPS,
        temporal: str = "current",
        at_time: Optional[float] = None,
        include_global: bool = False,
    ) -> RelationshipContext:
        """A hard-bounded selection of M5 material for context assembly.

        Selected input only. `hop_limit` is clamped to the approved single
        hop and reported on the result, so an over-deep request can never
        widen the output. Never injected into Active Memory or the working
        view automatically.
        """
        try:
            return queries_mod.relationship_context(
                self._get_store(), scope, entity_ids, state_ids,
                relationship_limit, entity_limit, memory_limit, hop_limit,
                temporal, at_time, include_global,
            )
        except Exception as exc:
            return RelationshipContext(
                scope=scope if isinstance(scope, str) else "",
                temporal=temporal if temporal in queries_mod.TEMPORAL_MODES
                else "current",
                at_time=0.0, hop_limit=queries_mod.MAX_TRAVERSAL_HOPS,
                diagnostics=[f"relationship_query_failed: {exc}"],
            )

    def propose_relationship_from_text(
        self,
        session_id: str,
        text: str,
        source_ids: Optional[list[int]] = None,
        scope: Optional[str] = None,
        timestamp: Optional[float] = None,
        observed_at: Optional[float] = None,
    ) -> tuple[Optional[int], list[str]]:
        try:
            return relationships_mod.propose_part_of_from_text(
                self._get_store(), session_id, text, source_ids,
                scope, timestamp, observed_at,
            )
        except Exception as exc:
            return None, [f"relationship_failed: {exc}"]

    def get_active_memory(
        self,
        session_id: str = DEFAULT_SESSION_ID,
        include_global: bool = True,
        digest_limit: int = 3,
        at_time: Optional[float] = None,
    ) -> ActiveMemory:
        """Compute the Active Memory view: what is currently true/useful.

        Read-only projection over persistent structures — no separate
        storage, no writes. Membership: current states (newest per
        scope/concept), current session metadata, newest digests.
        Superseded, pending, rejected, and unresolved rows never appear.
        """
        store = self._get_store()
        evaluation_time = self._normalized_time(at_time, "at_time")
        items: list[ActiveMemoryItem] = []
        try:
            currents = store.fetch_states(
                scope=session_id, status="current")
            if include_global:
                currents += store.fetch_states(
                    scope="global", status="current")
        except Exception:
            currents = []
        currents = [
            state for state in currents
            if state.authority == "user" and self._valid_at(state, evaluation_time)
        ]
        currents, _ = _newest_per_scope_concept(currents)
        for st in sorted(currents, key=lambda s: (s.scope, s.concept)):
            content = slots_mod.render_sentence(st.concept, st.value)
            items.append(ActiveMemoryItem(
                kind="state", memory_type=st.memory_type or "",
                concept=st.concept, value=st.value, content=content,
                authority=st.authority, confidence=st.confidence,
                importance=scoring_mod.importance_of(
                    f"{st.concept} {st.value}", "user"),
                scope=st.scope, session_id=st.session_id,
                source_ids=list(st.source_ids), status="current",
                state_id=st.state_id, created_at=st.created_at,
                observed_at=st.observed_at, valid_from=st.valid_from,
                valid_until=st.valid_until, superseded_at=st.superseded_at,
                supersedes_id=st.supersedes_id,
            ))
        try:
            rec = store.get_session(session_id)
        except Exception:
            rec = None
        if rec is not None:
            label = rec.title if rec.title else "(untitled)"
            content = (f"[session current] id={rec.session_id}"
                       f" name={label!r} status={rec.status}")
            items.append(ActiveMemoryItem(
                kind="session", memory_type="SESSION", concept="session",
                value=label, content=content, authority="system",
                confidence=1.0,
                importance=scoring_mod.importance_of(content, "system"),
                scope=session_id, session_id=rec.session_id,
                source_ids=[], status=rec.status, created_at=rec.created_at,
            ))
        try:
            digests = store.fetch_digests(session_id, limit=digest_limit)
        except Exception:
            digests = []
        for d in digests:
            content = (f"[history digest {d['seq_start']}-{d['seq_end']}]"
                       f"\n{d['content']}")
            items.append(ActiveMemoryItem(
                kind="digest", memory_type="DIGEST",
                concept="digest %d-%d" % (d["seq_start"], d["seq_end"]),
                value=d["content"], content=content, authority="system",
                confidence=1.0,
                importance=scoring_mod.importance_of(content, "system"),
                scope=session_id, session_id=d["session_id"],
                source_ids=list(d["source_ids"]), status="current",
                created_at=d["created_at"],
            ))
        return ActiveMemory(session_id=session_id, items=items)

    def select_active_memory(
        self,
        session_id: str,
        query_text: str,
        limit: int = 5,
    ) -> list[ActiveMemoryItem]:
        """Rank Active Memory for a task using the existing relevance rule.

        Deterministic: relevance desc, concept asc. Used to select entries
        into Working Memory. No semantic retrieval.
        """
        terms = scoring_mod.build_query(query_text or "")
        if not terms:
            return []
        ranked = []
        for item in self.get_active_memory(session_id).items:
            overlap = scoring_mod.relevance(
                scoring_mod.tokenize(f"{item.concept} {item.value}"), terms)
            ranked.append((overlap, item.concept, item))
        ranked.sort(key=lambda r: (-r[0], r[1]))
        return [item for overlap, _, item in ranked[:limit] if overlap > 0]

    def promote_to_candidate(
        self,
        session_id: str,
        text: str,
        source_ids: Optional[list[int]] = None,
        authority: str = "user",
    ) -> tuple[list[int], list[str]]:
        """Explicit handoff from Working Memory into Memory Formation.

        Creates pending candidate rows only — never states, never truth.
        Non-user authority is refused: assistant-derived content cannot
        become user memory through promotion. Returns (ids, diagnostics).
        """
        diagnostics: list[str] = []
        if authority != "user":
            return [], ["promotion_refused: authority=%r is not user"
                        % (authority,)]
        if not isinstance(text, str) or not text.strip():
            return [], ["promotion_refused: empty text"]
        candidates = state_mod.extract_candidates(text.strip(), "user")
        store = self._get_store()
        ids: list[int] = []
        scope = state_mod._scope_of(text.strip())
        scope_key = "global" if scope == "global" else session_id
        if not candidates:
            try:
                cid = store.create_candidate(
                    session_id, scope_key, list(source_ids or []),
                    proposed_type="UNCERTAIN", proposed_concept="",
                    proposed_value=text.strip()[:200], authority="user",
                    confidence=0.3, status="pending",
                )
                ids.append(cid)
                diagnostics.append("candidate_pending: uncertain promotion")
            except Exception as e:
                diagnostics.append(f"state_failed: promote failed: {e}")
            return ids, diagnostics
        for cand in candidates:
            # Slot-routed personal facts promote with the candidate's own
            # global scope; marker-based scope applies to everything else.
            cand_scope_key = (
                "global" if cand.scope == "global" else session_id)
            try:
                existing = store.fetch_candidates(
                    scope=cand_scope_key, status="pending")
                if any(r.proposed_concept == cand.concept
                       and r.proposed_value == cand.value
                       and r.source_ids == list(source_ids or [])
                       for r in existing):
                    continue
                # Legacy X-is-Y shapes carry no type; they assert a plain
                # fact. Genuinely structureless promotions stay UNCERTAIN
                # via the branch above, never here.
                cid = store.create_candidate(
                    session_id, cand_scope_key, list(source_ids or []),
                    proposed_type=cand.mtype or "FACT",
                    proposed_concept=cand.concept,
                    proposed_value=cand.value,
                    proposed_rationale=cand.rationale, authority="user",
                    confidence=0.3, status="pending",
                )
                ids.append(cid)
                diagnostics.append(
                    f"candidate_pending: {cand.concept or cand.value}")
            except Exception as e:
                diagnostics.append(f"state_failed: promote failed: {e}")
        return ids, diagnostics

    def _get_store(self) -> ContextStore:
        if self._store is None:
            self._store = ContextStore(self._db_path)
        return self._store

    def ensure_session(
        self, session_id: str = DEFAULT_SESSION_ID
    ) -> tuple[str, list[str]]:
        """Ensure the session row exists. Never loses the request on failure.

        Returns (effective_session_id, diagnostics). If the store is
        unavailable, the requested id is kept for the in-memory request
        lifetime and a diagnostic is recorded.
        """
        if not isinstance(session_id, str) or not session_id:
            session_id = DEFAULT_SESSION_ID
        try:
            self._get_store().ensure_session(session_id)
            return session_id, []
        except Exception as e:
            return session_id, [f"store_unavailable: ensure_session: {e}"]

    def rename_session(self, session_id: str, title: str):
        """Rename a session's authoritative metadata. Raw history untouched.

        Returns the updated SessionRecord, or None when the store is
        unavailable or the session is unknown. Never raises.
        """
        try:
            return self._get_store().rename_session(session_id, title)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def experience_events(self, *, session_id: Optional[str] = None,
                          kind: Optional[str] = None, limit: int = 100):
        """Evidence-backed user-reported events, independent of subject.

        This is an inspection API; episodes aren't yet injected into Aurora's
        prompt. Organizing and retrieving related episodes are later batches.
        """
        return self._get_store().fetch_experience_events(
            session_id=session_id, kind=kind, limit=limit)

    def experience_episodes(self, *, session_id: Optional[str] = None,
                            limit: int = 100, max_gap_days: int = 45):
        """Read-only, conservative event groups spanning multiple sessions.

        This is an inspection API; no episodes are injected into model prompts
        until M3 Batch 3 verifies episodic recall and relevance.
        """
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be 1..1000")
        events = self._get_store().fetch_experience_events(
            session_id=session_id, limit=1000)
        return organize_experience_events(
            events, max_gap_days=max_gap_days)[:limit]

    @staticmethod
    def _self_contained_like(subject: str, value: str) -> bool:
        """Reject legacy unresolved references at READ time, without deletion.

        These are grammatical ambiguity markers, not lists of allowed domains.
        A meaningful named preference such as Big Room remains available.
        """
        subject = subject.casefold().strip(" .!?\"'")
        value = value.casefold().strip()
        if not subject or not value.startswith(("loves ", "likes ")):
            return False
        if subject in {"it", "this", "that", "them", "these", "those", "something",
                       "anything", "everything"}:
            return False
        # A dangling referent cannot establish an identifiable liked item.
        if re.search(r"\b(?:of|about|for|with|from|as|to)\s+"
                     r"(?:it|this|that|them|these|those)\s*$", subject):
            return False
        return True

    @staticmethod
    def _like_identity(subject: str) -> str:
        """Conservatively group qualified mentions of the same named subject.

        This is a *read-time* display key, not an update to stored memory.
        Only grammatical trailing modifiers are stripped; e.g. "Atlas Kite
        regardless of its size" shares a key with "Atlas Kite", while
        distinct named items and aspects retain separate identities.
        """
        subject = re.sub(r"^(?:the|a|an)\s+", "", subject.casefold().strip())
        subject = re.split(
            r"\s+(?:regardless\s+of|whether|even\s+if|no\s+matter\s+"
            r"(?:what|which|how)|too\s+like)\b|,\s+and\s+it\b",
            subject, maxsplit=1,
        )[0]
        return " ".join(subject.strip(" .,!?").split())

    def memory_query_evidence(self, plan: dict, *, session_id: str = "default") -> str:
        """Read-only evidence for a validated model-produced memory query plan.

        The plan determines meaning; this method enforces source, time, and
        bounded-output constraints. It never infers a preference or event.
        """
        from datetime import datetime, timezone
        from . import scoring as scoring_mod

        sources = set(plan["sources"])
        subject = (plan.get("subject") or "").casefold().strip()
        days = plan.get("recent_days")
        first = plan.get("first_mention", False)
        now = time.time()
        start = plan.get("from_utc")
        end = plan.get("to_utc")
        if start is None:
            start = now - days * 86400 if days is not None else None
        store = self._get_store()
        # Query matching requires a named subject to occur in the actual
        # memory/source. Domain categories are NOT dictionary-mapped here:
        # when a topic is broad, the model selects from diverse raw evidence.
        terms = tuple(scoring_mod.singular(t) for t in
                      scoring_mod.tokenize(subject)) if subject else ()
        def matches(*texts):
            if not terms:
                return True
            words = set()
            for text in texts:
                words.update(scoring_mod.singular(t) for t in
                             scoring_mod.tokenize(text))
            return all(term in words for term in terms)
        def permitted(ts):
            return (0 <= ts <= now + 300 and (start is None or ts >= start)
                    and (end is None or ts < end))
        items = {}
        # Current stated memories (or historical versions for first-mention
        # queries); unresolved/assistant-created rows never qualify.
        if sources.intersection({"preferences", "facts"}):
            for state in store.fetch_states(scope="global"):
                if not self._accepted_state(state) or not state.source_ids:
                    continue
                preference = (state.memory_type == "PREFERENCE" or
                              state.concept.startswith(("likes for ", "favorite ",
                                                        "rationale for ")))
                if "facts" not in sources and not preference:
                    continue
                if not first and state.status != "current":
                    continue
                if preference and state.concept.startswith("likes for "):
                    if not self._self_contained_like(
                            state.concept[len("likes for "):], state.value):
                        continue
                for msg in store.fetch_messages_by_ids(state.source_ids[:24]):
                    if msg.role != "user" or not permitted(msg.timestamp):
                        continue
                    if not matches(state.concept, state.value, msg.content):
                        continue
                    key = msg.message_id
                    row = items.setdefault(key, {"id": key, "timestamp": msg.timestamp,
                        "text": msg.content[:500], "from": "user_state", "concepts": []})
                    label = f"{state.concept}: {state.value}"
                    if label not in row["concepts"]:
                        row["concepts"].append(label)
        if "experiences" in sources:
            events = (self.experience_events(limit=1000) +
                      store.fetch_legacy_experience_reports(limit=5000))
            for event in events:
                if event.authority != "user" or not permitted(event.reported_at):
                    continue
                if not matches(event.statement):
                    continue
                original = store.fetch_messages_by_ids([event.source_message_id])
                if not original or original[0].role != "user":
                    continue
                msg = original[0]
                items.setdefault(msg.message_id, {
                    "id": msg.message_id, "timestamp": msg.timestamp,
                    "text": msg.content[:500], "from": "user_experience",
                    "concepts": [event.kind]})
                # The immediately subsequent user's remark is a *possible*
                # reaction, not independent proof of what its referent was.
                reaction = store.immediate_user_reaction(msg.message_id)
                if (reaction and permitted(reaction.timestamp) and
                        0 <= reaction.timestamp - event.reported_at <= 3600):
                    items.setdefault(reaction.message_id, {
                        "id": reaction.message_id, "timestamp": reaction.timestamp,
                        "text": reaction.content[:500], "from":
                        f"possible_followup_to_{msg.message_id}", "concepts": []})
        # Keep source-indexed facts visible even when the newest raw messages
        # are numerous; a recency-only cap would hide old favorites again.
        structured = sorted(items.values(), key=lambda r: (r["timestamp"], r["id"]),
                            reverse=not first)
        structured_ids = set(items)
        # Structured memory is an index, not the whole person's history.
        # Retrieve bounded verbatim *user* messages from every session as
        # candidate evidence, including activities that old extractors never
        # converted into episodes. No language/topic-specific extraction here.
        raw_terms = tuple(dict.fromkeys(t for t in terms if len(t) >= 2))[:8]
        raw_hits = store.search_user_evidence(
            raw_terms, start=start, end=end, limit=240,
            oldest_first=first)
        for msg in raw_hits:
            if not permitted(msg.timestamp):
                continue
            if msg.message_id not in items:
                items[msg.message_id] = {
                    "id": msg.message_id, "timestamp": msg.timestamp,
                    "text": msg.content[:500], "from": "raw_user_statement",
                    "concepts": [],
                }
            if raw_terms:
                # Nearby *user* messages may give useful conversational
                # context (e.g. an unnamed reaction after describing a
                # video). They are NOT asserted as related facts.
                for neighbor in store.neighboring_user_evidence(msg):
                    if not permitted(neighbor.timestamp):
                        continue
                    items.setdefault(neighbor.message_id, {
                        "id": neighbor.message_id,
                        "timestamp": neighbor.timestamp,
                        "text": neighbor.content[:500],
                        "from": f"possible_context_for_{msg.message_id}",
                        "concepts": [],
                    })
        raw_only = sorted((r for key, r in items.items()
                           if key not in structured_ids),
                          key=lambda r: (r["timestamp"], r["id"]),
                          reverse=not first)
        # 60 sourced records + 30 raw-only candidates. Keep both channels in
        # the evidence window instead of letting one evict the other.
        ordered = sorted(structured[:60] + raw_only[:30],
                         key=lambda r: (r["timestamp"], r["id"]),
                         reverse=not first)
        omitted_count = max(0, len(structured) - 60) + max(0, len(raw_only) - 30)
        # Cap output, disclose omissions; do not claim the answer is complete.
        limit = 90
        output = ["User-stated evidence (UTC timestamps are dates of statements, "
                  "not inferred real-world onset/event dates). "
                  "Only direct user-source records are included."]
        for row in ordered[:limit]:
            moment = datetime.fromtimestamp(row["timestamp"], timezone.utc)
            concepts = "; ".join(row["concepts"][:4])
            output.append(f'[message {row["id"]}, {moment:%Y-%m-%d %H:%M} UTC, '
                          f'{row["from"]}] {row["text"]!r}' +
                          (f' | stored: {concepts}' if concepts else ''))
        if omitted_count:
            output.append(f"Additional user-source candidates omitted: {omitted_count}. "
                          "The full inventory was not included.")
        return "\n".join(output) if ordered else ""

    def verified_likes_history(self, question: str) -> Optional[str]:
        """Source-backed, time-bounded, domain-neutral stated likes.

        Explicit broad inventory requests are recognized independently of
        tense. Records stay immutable; ambiguous legacy likes are hidden and
        related phrasing is grouped for display with its original evidence.
        """
        q = re.sub(r"^\s*(?:hey\s+)?aurora\s*[,!:.]?\s*", "", question,
                   flags=re.I).strip().rstrip("?!. ]").casefold()
        category = None
        window_days = None
        if re.search(r"\brecent(?:ly)?\b", q):
            window_days = 7
        explicit_window = re.search(
            r"\b(?:in|over|within|during) (?:the )?last (\d{1,3}) "
            r"(day|week|month)s?\b", q)
        if explicit_window:
            size = int(explicit_window.group(1))
            if size < 1 or size > 365:
                return "Please request a time window between 1 and 365 units."
            window_days = size * {"day": 1, "week": 7, "month": 30}[
                explicit_window.group(2)]
        stripped = re.sub(r"\s+(?:in|over|within|during) (?:the )?last "
                          r"\d{1,3} (?:days?|weeks?|months?)$", "", q)
        verb = r"(?:like|liked|love|loved|enjoy|enjoyed)"
        # A broad preference inventory includes explicit favorites as well as
        # likes. This is a query-plan distinction, not a subject whitelist.
        preference_inventory = bool(re.fullmatch(
            r"(?:what (?:are|were) my (?:known |current )?preferences|"
            r"(?:list|show|summarize) (?:all )?my (?:known |current )?preferences|"
            r"what do you (?:know|remember) about my preferences)", stripped))
        broad = preference_inventory or bool(re.fullmatch(
            rf"(?:what (?:have i (?:told you|said)(?: that)? i|do i) "
            rf"{verb}(?: recently)?|what have i {verb} recently|"
            rf"what (?:are )?(?:(?:all|every) )?my (?:recent )?"
            rf"(?:likes|interests)|(?:all|every) (?:my )?"
            rf"(?:likes|interests?))", stripped))
        if not broad:
            matched = re.fullmatch(
                r"(?:what|which) ([a-z][a-z '\-]{1,60}?) "
                r"(?:that )?(?:do )?i (?:like|love)", stripped)
            if matched:
                category = matched.group(1).strip()
            else:
                return None
        store = self._get_store()
        now = time.time()
        earliest = now - window_days * 86400 if window_days is not None else None
        records = {}
        for state in store.fetch_states(scope="global", status="current"):
            if not (self._accepted_state(state) and state.source_ids
                    and self._valid_at(state, now)):
                continue
            if state.concept.startswith("likes for "):
                subject = state.concept[len("likes for "):]
                if not self._self_contained_like(subject, state.value):
                    continue
                value = state.value
            elif preference_inventory and state.concept.startswith("favorite "):
                # Keep favorite dimensions distinct: "favorite color" and
                # "favorite classic car" are not interchangeable claims.
                dimension = state.concept[len("favorite "):].strip()
                if not dimension or not state.value.strip():
                    continue
                subject = f"favorite {dimension}"
                value = f"favorite {dimension}: {state.value}"
            else:
                continue
            # Filter the individual USER evidence timestamps before combining
            # versions of a preference. A recent statement about an old like
            # must not vanish, and an old statement must not become recent.
            messages = [m for m in store.fetch_messages_by_ids(state.source_ids)
                        if m.role == "user" and 0 <= m.timestamp <= now + 300]
            if earliest is not None:
                messages = [m for m in messages if m.timestamp >= earliest]
            if not messages:
                continue
            if category:
                cat_words = {scoring_mod.singular(w) for w in
                             scoring_mod.tokenize(category)}
                evidence_words = {scoring_mod.singular(w) for m in messages
                                  for w in scoring_mod.tokenize(m.content)}
                if not cat_words or not cat_words <= evidence_words:
                    continue
            key = ("favorite:" + subject.casefold() if
                   state.concept.startswith("favorite ") else
                   "like:" + self._like_identity(subject))
            if not key:
                continue
            existing = records.setdefault(key, {"messages": {}, "values": [],
                                                "latest": -1.0})
            existing["values"].append(value)
            for m in messages:
                existing["messages"][m.message_id] = m
                existing["latest"] = max(existing["latest"], m.timestamp)
        ordered = sorted(records.items(), key=lambda row: (-row[1]["latest"], row[0]))
        if not ordered:
            return ("I couldn't verify any user-stated likes in that "
                    "time window or category. That doesn't mean you never "
                    "mentioned them; I won't guess from unrelated memories.")
        prefix = (f"Here are your verified likes mentioned in the last "
                  f"{window_days} days (UTC; time of statement):" if
                  window_days is not None else
                  ("Here are your current verified preferences and favorites "
                   "from your own statements (not necessarily complete):" if
                   preference_inventory else
                   "Here are the stored likes I could verify from your own "
                   "statements (not necessarily a complete list):"))
        lines = [prefix]
        # A bounded answer must disclose omissions rather than silently imply
        # the result is complete. Grouping reduces duplicates before the cap.
        limit = 30
        for identity, record in ordered[:limit]:
            subject = identity.split(":", 1)[1]
            evidence = sorted(record["messages"].values(),
                              key=lambda m: (m.timestamp, m.message_id))
            latest = record["latest"]
            # Preserve an unmodified value for a single preference. Multiple
            # qualified phrasings are presented under their shared named item,
            # with verbatim evidence retaining every nuance.
            values = list(dict.fromkeys(record["values"]))
            if len(values) > 1:
                strength = "loves" if any(v.casefold().startswith("loves ")
                                           for v in values) else "likes"
                label = f"{strength} {subject}"
            else:
                label = values[0]
            shown = evidence[:3]
            quotes = [f'"{m.content}" (message {m.message_id})' for m in shown]
            when = (time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(latest))
                    if window_days is not None else None)
            suffix = (f"; {len(evidence)} verified statements total"
                      if len(evidence) > len(shown) else "")
            lines.append(f'- {label}' + (f' ({when})' if when else '') +
                         '; user evidence: ' + '; '.join(quotes) + suffix)
        if len(ordered) > limit:
            lines.append(f"Showing {limit} of {len(ordered)} distinct likes; "
                         "ask for a narrower category or time range to see more.")
        return "\n".join(lines)

    def verified_episode_history(self, question: str, *,
                                 context: tuple[str, ...] = ()) -> Optional[str]:
        """Evidence-only, domain-neutral episodic lookup across sessions.

        A read-only view over the Batch 1/2 event store; no automatic event
        inference, training, or mutation takes place during retrieval.
        """
        from .episode_retrieval import parse_episode_question
        if parse_episode_question(question, context=context) is None:
            return None
        # Select matching episodes BEFORE applying the small response budget.
        # Otherwise a valid older activity disappears after 100 newer ones.
        store = self._get_store()
        # Read-only legacy evidence supplements structured events; no migration
        # or guessing is needed to recall older reports made before M3 existed.
        structured = self.experience_events(limit=1000)
        legacy = store.fetch_legacy_experience_reports(limit=5000)
        episodes = organize_experience_events(structured + legacy)
        parsed = parse_episode_question(question, context=context)
        reactions = {}
        if parsed and parsed.include_feelings:
            for episode in episodes:
                # The latest *user* reaction can only be attached when the
                # episode report has a single named subject.
                for event in episode.events:
                    if " and " not in episode.subject.casefold():
                        response = store.immediate_user_reaction(
                            event.source_message_id)
                        if response is not None:
                            reactions[event.source_message_id] = response
        return render_episode_answer(
            question, episodes, context=context, reactions=reactions)

    def reconcile(
        self,
        messages: list[dict[str, Any]],
        session_id: str = DEFAULT_SESSION_ID,
    ) -> list[str]:
        """Persist a conversation snapshot idempotently.

        Sequence identity = position in the snapshot, except where a row
        carries its own ``seq``: restored transcripts re-persist with the
        exact sequence they were stored under, so reloading a transcript
        is a no-op. A row without ``seq`` continues past the highest
        sequence seen so far, so a partly-restored snapshot can never
        collide with the rows it was restored from. Same position with
        different content is stored as a separate row plus a diagnostic;
        nothing is overwritten.
        Returns diagnostics (empty on the happy path).
        """
        diagnostics: list[str] = []
        try:
            store = self._get_store()
        except Exception as e:
            return [f"store_unavailable: {e}"]
        _, session_diags = self.ensure_session(session_id)
        diagnostics.extend(session_diags)
        next_seq = 0
        prior_user: Optional[str] = None
        prior_assistant: Optional[str] = None
        for pos, msg in enumerate(messages):
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            raw_seq = msg.get("seq")
            if (
                isinstance(raw_seq, int)
                and not isinstance(raw_seq, bool)
                and raw_seq >= 0
            ):
                seq = raw_seq
            else:
                seq = max(pos, next_seq)
            next_seq = max(next_seq, seq + 1)
            raw_timestamp = msg.get("timestamp")
            timestamp = None
            if (isinstance(raw_timestamp, (int, float))
                    and not isinstance(raw_timestamp, bool)):
                candidate_timestamp = float(raw_timestamp)
                if math.isfinite(candidate_timestamp):
                    timestamp = candidate_timestamp
            try:
                importance = scoring_mod.importance_of(content, role)
                message_id, is_new, conflict = store.append(
                    session_id, seq, role, content, importance,
                    timestamp=timestamp,
                )
                if conflict:
                    diagnostics.append(conflict)
                if is_new:
                    # M3 episodic evidence is independent of semantic memory.
                    # Only new explicit user reports are admitted, and the
                    # original message is the source of both text and time.
                    if role == "user":
                        try:
                            claim = extract_experience_claim(content, role)
                            if claim is not None:
                                store.save_experience_event(
                                    session_id, message_id, claim.kind,
                                    claim.actor, claim.statement,
                                )
                        except Exception as e:
                            diagnostics.append(f"episode_capture_failed: {e}")
                    # M4 structured-memory formation first. M5 entity,
                    # alias, and relationship formation runs after it, from
                    # the same persisted message id, and is independently
                    # guarded so it can never suppress or alter M4.
                    # An elliptical user reply is resolved against the
                    # turn that preceded it: the user's own question or
                    # change announcement first, then the assistant's
                    # question/invitation. Snapshot first, then the store
                    # for legacy single-message callers.
                    context_question = None
                    assistant_question = None
                    if role == "user":
                        context_question = prior_user
                        assistant_question = prior_assistant
                        if context_question is None:
                            context_question = self._prior_user_text(
                                store, session_id, message_id
                            )
                        if assistant_question is None:
                            assistant_question = self._prior_assistant_text(
                                store, session_id, message_id
                            )
                    diagnostics.extend(
                        self._extract_states(
                            store, session_id, message_id, role, content,
                            context_question, assistant_question,
                        )
                    )
                    if role == "user":
                        diagnostics.extend(
                            self._extract_entities(
                                store, session_id, message_id, content,
                                timestamp,
                            )
                        )
            except Exception as e:
                diagnostics.append(f"store_unavailable: {e}")
                break
            if role == "user":
                prior_user = content
            elif role == "assistant":
                prior_assistant = content
        return diagnostics

    @staticmethod
    def _extract_entities(
        store,
        session_id: str,
        message_id: int,
        content: str,
        timestamp: Optional[float] = None,
    ) -> list[str]:
        """M5 formation from one user message. Never raises.

        Runs after M4 so the state ids it needs already exist. Every write
        goes through the existing entities/relationships APIs, so provenance,
        authority, scope, and the candidate lifecycle stay owned there.
        """
        try:
            return entity_formation_mod.form_from_user_message(
                store, session_id, message_id, content, timestamp,
            ).diagnostics
        except Exception as e:
            return [f"m5_failed: {e}"]

    @staticmethod
    def _extract_states(
        store,
        session_id: str,
        message_id: int,
        role: str,
        content: str,
        context_question: Optional[str] = None,
        assistant_question: Optional[str] = None,
    ) -> list[str]:
        """Deterministic state extraction. Never invents on failure."""
        try:
            if role == "user":
                candidates = state_mod.extract_candidates(
                    content, role, context_question, assistant_question
                )
                reference_message = None
                # Never store ``likes for it``. Resolve only a user-to-user
                # reference with a uniquely named antecedent in this session.
                if not candidates:
                    recent = [m for m in store.fetch_latest(session_id, 6)
                              if m.role == "user" and m.message_id != message_id]
                    previous = recent[-1] if recent else None
                    current = store.fetch_messages_by_ids([message_id])
                    if (previous and current and
                            0 < current[0].seq - previous.seq <= 3):
                        explicit = resolve_preference_reference(
                            content, previous.content)
                        if explicit:
                            candidates = state_mod.extract_candidates(
                                explicit, "user")
                            reference_message = previous.message_id
                # A consecutive USER statement can explain "that's why
                # I love X". Assistant statements never supply the motive.
                prior = [m for m in store.fetch_latest(session_id, 4)
                         if m.role == "user" and m.message_id != message_id]
                previous = prior[-1] if prior else None
                followup = None
                if previous:
                    current = store.fetch_messages_by_ids([message_id])
                    if current and 0 < current[0].seq - previous.seq <= 3:
                        followup = state_mod.extract_followup_preference_reason(
                            content, previous.content)
                if followup:
                    # "Regera" and "Koenigsegg Regera" may refer to the
                    # same preference. Resolve a short name only if there is
                    # exactly ONE matching current preference; never guess.
                    from dataclasses import replace as dc_replace
                    shorthand = followup[0].concept.removeprefix("likes for ")
                    candidates_pool = [st.concept for st in store.fetch_states(
                        scope="global", status="current")
                        if st.concept.startswith("likes for ") and (
                            st.concept == followup[0].concept or
                            st.concept.endswith(" " + shorthand))]
                    candidates_pool = sorted(set(candidates_pool))
                    if len(candidates_pool) == 1:
                        canonical = candidates_pool[0]
                        followup = (
                            dc_replace(followup[0], concept=canonical),
                            dc_replace(followup[1],
                                       concept="rationale for " + canonical),
                        )
                    candidates.extend(followup)
                if not candidates:
                    return []
                diagnostics = state_mod.apply_user_candidates(
                    store, session_id, message_id, candidates
                )
                if reference_message is not None:
                    for candidate in candidates:
                        for saved in store.fetch_states(
                                scope="global", concept=candidate.concept,
                                status="current"):
                            if saved.value == candidate.value:
                                store.add_state_evidence(saved.state_id,
                                                         reference_message)
                                break
                if followup and previous is not None:
                    for reason in store.fetch_states(
                            scope="global", concept=followup[1].concept,
                            status="current"):
                        if reason.value == followup[1].value:
                            store.add_state_evidence(reason.state_id,
                                                     previous.message_id)
                            break
                return diagnostics
            if role == "assistant":
                return state_mod.check_assistant_conflict(
                    store, session_id, message_id, content
                )
        except Exception as e:
            return [f"state_failed: extraction failed: {e}"]
        return []

    @staticmethod
    def _prior_user_text(
        store, session_id: str, exclude_id: int
    ) -> Optional[str]:
        """Most recent stored user message other than the one being processed.

        Context for legacy callers that reconcile a single-message snapshot:
        the question that triggered this turn still lives in the store.
        Bounded and failure-silent.
        """
        try:
            rows = store.fetch_latest(session_id, 8)
        except Exception:
            return None
        for row in reversed(rows):
            if row.role == "user" and row.message_id != exclude_id:
                return row.content
        return None

    @staticmethod
    def _prior_assistant_text(
        store, session_id: str, exclude_id: int
    ) -> Optional[str]:
        """Most recent stored assistant message other than the current row.

        The assistant's own question or invitation is legitimate context
        for an elliptical reply ("violet" after "please let me know your
        new favorite color"), and a legacy single-message caller has no
        snapshot to read it from. Bounded and failure-silent.
        """
        try:
            rows = store.fetch_latest(session_id, 8)
        except Exception:
            return None
        for row in reversed(rows):
            if row.role == "assistant" and row.message_id != exclude_id:
                return row.content
        return None

    def preference_update_claim_error(self, text: str) -> Optional[str]:
        """Store-backed check for a claimed preference update.

        Returns a correction text when the reply says a "favorite X" was
        updated but the stored current value does not match, and None when
        the claim is backed or the store cannot be read. Never raises.
        """
        try:
            store = self._get_store()
        except Exception:
            return None
        return state_mod.preference_update_claim_error(store, text)

    # ------------------------------------------------------------------
    # Transcript reload
    # ------------------------------------------------------------------
    def load_transcript(
        self,
        session_id: str = DEFAULT_SESSION_ID,
        limit: int = TRANSCRIPT_RELOAD_LIMIT,
        token_budget: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Reload the latest conversational turns of one session.

        Startup continuity for restarted entry points: bounded to ``limit``
        messages and to a token budget so a few very long messages cannot
        crowd the rest of the request out of context. Older history stays
        stored and retrievable, just not injected. Chronological,
        session-scoped, and failure-silent: any store problem returns an
        empty transcript instead of blocking startup.
        """
        try:
            rows = self._get_store().fetch_conversation(session_id, limit)
        except Exception:
            return []
        entries: list[dict[str, Any]] = [
            {"role": row.role, "content": row.content, "seq": row.seq}
            for row in rows
            if row.content
        ]
        if not entries:
            return []
        budget = (
            token_budget
            if token_budget is not None
            else self._transcript_budget()
        )
        while len(entries) > 1:
            total = sum(
                estimate_tokens(str(entry.get("content", "")))
                for entry in entries
            )
            if total <= budget:
                break
            entries.pop(0)
        return entries

    @staticmethod
    def _transcript_budget() -> int:
        try:
            import os

            raw = os.environ.get("PRIMORA_AURORA_TRANSCRIPT_BUDGET")
            if raw:
                return max(1, int(raw))
        except (TypeError, ValueError):
            pass
        return TRANSCRIPT_TOKEN_BUDGET

    def next_seq(self, session_id: str = DEFAULT_SESSION_ID) -> int:
        """Sequence the next new row of this session must take.

        Used where a caller must mint a row itself (the clear marker, or
        post-clear messages): one past the true stored maximum, never a
        guess from a bounded window.
        """
        try:
            return self._get_store().max_seq(session_id) + 1
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(
        self,
        messages: list[dict[str, str]],
        session_id: str = DEFAULT_SESSION_ID,
        token_counter: Optional[Callable[[list[dict[str, str]]], int]] = None,
        max_new_tokens: int = 256,
        budget: Optional[BudgetConfig] = None,
        working=None,
    ) -> BuildResult:
        """Assemble a bounded working view for one model call.

        When a Working Memory object is supplied, its rendered task state
        is layered directly after the system instruction (budget permitting)
        so the final view reads as working task + conversation + selected
        active memory + evidence. Omitted working changes nothing.
        """
        cfg = budget or self._budget
        counter = self._counter_or_fallback(token_counter)
        diagnostics: list[str] = []

        # 0. Session row first (safe fallback keeps the request alive).
        session_id, session_diags = self.ensure_session(session_id)
        diagnostics.extend(session_diags)

        # 1. Persist the snapshot first; raw truth before selection.
        diagnostics.extend(self.reconcile(messages, session_id))

        # 2. Split system instruction (mandatory) from history.
        system_msg: Optional[dict[str, str]] = None
        history = list(messages)
        if history and history[0].get("role") == "system":
            system_msg = history[0]
            history = history[1:]

        # 3. Current user turn is mandatory and never truncated.
        current: Optional[dict[str, str]] = None
        for msg in reversed(history):
            if msg.get("role") == "user":
                current = msg
                break
        if current is None and history:
            current = history[-1]

        base = [system_msg] if system_msg is not None else []
        try:
            base_tokens = counter(base) if base else 0
        except Exception:
            base_tokens = self._estimate_messages(base)
            diagnostics.append("counter_fallback: system section")
        try:
            current_tokens = counter([current]) if current is not None else 0
        except Exception:
            current_tokens = self._estimate_messages([current]) if current else 0
            diagnostics.append("counter_fallback: current turn")

        if base_tokens + current_tokens > cfg.input_ceiling:
            if current is not None:
                try:
                    self._get_store()
                except Exception as e:
                    diagnostics.append(f"store_unavailable: {e}")
            raise ContextTooLargeError(
                "Current user message alone exceeds the hard input ceiling; "
                "raw message preserved, no inference attempted."
            )

        selected: list[dict[str, str]] = list(base)
        if current is not None:
            # Current turn appended last; newest turns fill before it.
            pass
        selected_fps = {self._fp(m) for m in selected}
        if current is not None:
            selected_fps.add(self._fp(current))
        selected_texts = [m.get("content", "") for m in selected]
        if current is not None:
            selected_texts.append(current.get("content", ""))
        total = base_tokens + current_tokens

        # 3b. Working Memory layer (optional, runtime-only, never persisted).
        # Task state frames the request right after the system instruction.
        if working is not None:
            try:
                for wmsg in working.to_messages():
                    role = str(wmsg.get("role", "system"))
                    content = str(wmsg.get("content", ""))
                    if not content:
                        continue
                    fp = f"{role}\x00{content}"
                    if fp in selected_fps:
                        continue
                    cost = self._safe_count(
                        counter, [{"role": role, "content": content}],
                        diagnostics)
                    if total + cost > cfg.input_ceiling:
                        diagnostics.append(
                            "working_truncated: working set exceeds budget")
                        break
                    selected.append({"role": role, "content": content})
                    selected_fps.add(fp)
                    selected_texts.append(content)
                    total += cost
            except Exception as e:
                diagnostics.append(f"working_failed: {e}")

        # Explicit timeline questions need reserved space for dated user
        # evidence, not just the most recent conversation turns.
        timeline_request = timeline_mod.parse_request(
            current.get("content", "") if current else "", now=time.time()
        )
        # Narrow verified history questions are answered by the dedicated
        # read-only Mind path. Do not steal ordinary slot retrieval while
        # assembling model context (the historical "what did I say about
        # yellow" slot behavior must keep working).
        if timeline_request is not None and timeline_request.mode in ("dates", "claims"):
            timeline_request = None

        # 4. Newest conversation turns up to the input target, less the
        # reserve the structured [state ...] facts still need. Without it
        # the fill runs all the way to the target and a later fact has only
        # the slack between target and ceiling to land in - which is zero on
        # a full turn, so a stored user fact is silently dropped on exactly
        # the turns that most need it. The reserve is halved when the
        # mandatory base plus current turn would otherwise own the target,
        # so a small budget keeps its newest messages instead of failing.
        rest = [m for m in history if m is not current]
        reserved = max(0, int(getattr(cfg, "state_reserve", 0)))
        headroom = max(0, cfg.input_target - total)
        history_limit = max(
            total, cfg.input_target - min(reserved, headroom // 2)
        )
        if timeline_request is not None:
            evidence_reserve = min(300, max(0, cfg.input_ceiling - total))
            history_limit = max(
                total, min(history_limit, cfg.input_ceiling - evidence_reserve)
            )
        for msg in reversed(rest):
            if total >= history_limit:
                break
            fp = self._fp(msg)
            if fp in selected_fps:
                continue
            cost = self._safe_count(counter, [msg], diagnostics)
            if total + cost > history_limit:
                continue
            selected.append(msg)
            selected_fps.add(fp)
            selected_texts.append(msg.get("content", ""))
            total += cost

        raw_count = self._raw_count(session_id)
        degraded = False
        fact_msgs: list[dict] = []

        local_discussion_query = bool(
            current and _LOCAL_DISCUSSION_RE.fullmatch(
                current.get("content", "")))
        if local_discussion_query:
            note = (
                "Session scope: the user asks what was discussed in THIS "
                "conversation. Use only earlier messages in this session; "
                "do not describe memories from other sessions as this "
                "session's conversation. If there are no earlier user "
                "messages here, say we have not discussed a topic yet."
            )
            note_msg = {"role": "system", "content": note}
            if total + self._safe_count(counter, [note_msg], diagnostics) \
                    <= cfg.input_ceiling:
                selected.append(note_msg)
                selected_fps.add(self._fp(note_msg))
                selected_texts.append(note)
                total += self._safe_count(counter, [note_msg], diagnostics)

        # 5. Structured authority first: current session metadata, then
        # current scoped state. Both outrank free-form retrieval below.
        try:
            total = self._add_session_info(
                session_id, current, selected, selected_fps,
                selected_texts, counter, cfg, total, diagnostics,
            )
        except Exception as e:
            degraded = True
            diagnostics.append(f"state_failed: session info failed: {e}")
        try:
            if timeline_request is not None:
                content = self._temporal_preference_context(timeline_request)
                fact = {"role": "system", "content": content}
                cost = self._safe_count(counter, [fact], diagnostics)
                if total + cost <= cfg.input_ceiling:
                    fact_msgs.append(fact)
                    total += cost
                else:
                    diagnostics.append("timeline_truncated: no room for evidence")
            else:
                total = self._add_current_state(
                    session_id, current, selected, selected_fps,
                    selected_texts, counter, cfg, total, diagnostics,
                    facts_out=fact_msgs,
                )
        except Exception as e:
            degraded = True
            diagnostics.append(f"state_failed: assembly failed: {e}")

        # 6. Retrieval over older material (only what still fits).
        try:
            admitted = (([], total) if (local_discussion_query or timeline_request)
                        else self._retrieve(
                session_id, history, current, selected,
                selected_fps, selected_texts, counter, cfg, total,
                diagnostics,
            ))
            selected.extend(admitted[0])
            total = admitted[1]
        except Exception as e:
            degraded = True
            diagnostics.append(f"retrieval_failed: {e}")

        # 6b. Explicit cross-session state questions get per-session
        # scoped values with identity; never flattened, never silent.
        try:
            if not local_discussion_query and not timeline_request:
                total = self._add_cross_session_states(
                session_id, current, selected, selected_fps,
                selected_texts, counter, cfg, total, diagnostics,
            )
        except Exception as e:
            degraded = True
            diagnostics.append(f"state_failed: cross-session failed: {e}")

        # 7. Digests (stored first, then freshly compacted if warranted).
        try:
            d_total = (total if (local_discussion_query or timeline_request)
                       else self._add_digests(
                session_id, current, history, selected, selected_fps,
                selected_texts, counter, cfg, total, diagnostics,
            ))
            total = d_total
        except Exception as e:
            degraded = True
            diagnostics.append(f"digest_failed: {e}")

        # 8. Optional memory hook (no-op in V1).
        try:
            if not local_discussion_query and not timeline_request:
                total = self._add_memories(
                    session_id, current, history, selected, selected_fps,
                    selected_texts, counter, cfg, total, diagnostics,
                )
        except Exception as e:
            degraded = True
            diagnostics.append(f"memory_failed: {e}")

        # 9. Reassembly: system instruction, then synthesized context
        # (working memory, session info, retrieval adjuncts), then the
        # conversation in original chronological order. Current facts are
        # placed immediately before the current turn so they follow, and
        # outrank, any older assistant claim or denial the same turn admits.
        history_pos = {id(m): i for i, m in enumerate(history)}
        convo = [m for m in selected if id(m) in history_pos]
        if current is not None and all(m is not current for m in convo):
            convo.append(current)
        convo.sort(key=lambda m: history_pos[id(m)])
        # NOTE: system_msg lives in `selected` (as the protected base) but
        # is prepended explicitly below, so it must be excluded here to
        # avoid emitting the system instruction twice.
        fact_ids = {id(m) for m in fact_msgs}
        synth = [
            m for m in selected
            if id(m) not in history_pos
            and not (system_msg is not None and m is system_msg)
            and id(m) not in fact_ids
        ]
        before_current: list[dict] = []
        after_current: list[dict] = []
        seen_current = False
        for m in convo:
            if m is current:
                seen_current = True
                continue
            if seen_current:
                after_current.append(m)
            else:
                before_current.append(m)
        ordered = (
            ([system_msg] if system_msg is not None else [])
            + synth
            + before_current
            + fact_msgs
            + ([current] if current is not None else [])
            + after_current
        )

        allowed = max(
            0, min(max_new_tokens, cfg.total_budget - total)
        )
        view = ContextView(
            messages=ordered,
            input_tokens=total,
            allowed_output=allowed,
            session_id=session_id,
            diagnostics=diagnostics,
        )
        degraded = degraded or any(
            d.startswith(
                ("store_unavailable", "retrieval_failed",
                 "digest_failed", "memory_failed", "counter_fallback")
            )
            for d in diagnostics
        )
        return BuildResult(
            view=view,
            raw_count=raw_count,
            selected_count=len(ordered),
            degraded=degraded,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fp(msg: dict[str, str]) -> str:
        return f"{msg.get('role', '')}\x00{msg.get('content', '')}"

    @staticmethod
    def _estimate_messages(messages: list[dict[str, str]]) -> int:
        return sum(estimate_tokens(m.get("content", "")) + 4 for m in messages)

    def _counter_or_fallback(
        self, token_counter: Optional[Callable[[list[dict[str, str]]], int]]
    ) -> Callable[[list[dict[str, str]]], int]:
        if token_counter is not None:
            return token_counter
        return lambda msgs: self._estimate_messages(msgs)

    def _safe_count(
        self,
        counter: Callable[[list[dict[str, str]]], int],
        messages: list[dict[str, str]],
        diagnostics: list[str],
    ) -> int:
        try:
            return int(counter(messages))
        except Exception:
            diagnostics.append("counter_fallback: selection")
            return self._estimate_messages(messages)

    def _raw_count(self, session_id: str) -> int:
        try:
            return self._get_store().count(session_id)
        except Exception:
            return 0

    def _recent_user_text(self, history: list[dict]) -> str:
        for msg in reversed(history):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    @staticmethod
    def _superseded_source_ids(store, scopes: list[str]) -> set[int]:
        """Message rows that source a superseded state in these scopes.

        Used to keep stale conflicting rows out of normal retrieval.
        History queries bypass this filter; raw rows are never deleted.
        """
        ids: set[int] = set()
        try:
            for scope in scopes:
                for st in store.fetch_states(
                    scope=scope, status="superseded"
                ):
                    ids.update(st.source_ids)
        except Exception:
            pass
        return ids

    # ------------------------------------------------------------------
    # M6: lexical candidate collection
    #
    # Extracted from `_retrieve` with no behaviour change so that the hybrid
    # and shadow paths rank candidates with exactly the same production code
    # that feeds Aurora today. One implementation, no mirror, no drift.
    # ------------------------------------------------------------------
    def collect_lexical_candidates(
        self,
        session_id: str,
        query_terms: list[str],
        selected_fps: Optional[set] = None,
        selected_texts: Optional[list[str]] = None,
        diagnostics: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Same-session and cross-session lexical candidates, ranked.

        Returns `{"same": [...], "cross": [...], "now_seq": int,
        "history_requested": bool}` where each entry is
        `(score, message, breakdown)`. Never admits anything: budget
        admission stays in `_retrieve`.
        """
        store = self._get_store()
        diags = diagnostics if diagnostics is not None else []
        fps = selected_fps if selected_fps is not None else set()
        texts = selected_texts if selected_texts is not None else []
        if not query_terms:
            return {
                "same": [], "cross": [], "now_seq": 0,
                "history_requested": False,
            }
        latest = store.fetch_latest(session_id, 200)
        now_seq = max((m.seq for m in latest), default=0)
        history_requested = self._wants_history(" ".join(query_terms))
        stale_ids: set[int] = set()
        if not history_requested:
            stale_ids = self._superseded_source_ids(
                store, [session_id, "global"])

        candidates: list[StoredMessage] = []
        seen_ids: set[int] = set()
        suppressed = 0
        for term in query_terms[:12]:
            if not term or not term.strip():
                diags.append("retrieval_skipped_term: empty term")
                continue
            try:
                hits = store.search(session_id, term, limit=10)
            except Exception as e:
                diags.append(f"retrieval_skipped_term: {term!r}: {e}")
                continue
            for hit in hits:
                if hit.message_id in seen_ids:
                    continue
                seen_ids.add(hit.message_id)
                if hit.message_id in stale_ids:
                    suppressed += 1
                    continue
                candidates.append(hit)
        if suppressed:
            diags.append(
                f"retrieval_suppressed_superseded: {suppressed} message(s)"
            )
        scored = []
        for hit in candidates:
            if f"{hit.role}\x00{hit.content}" in fps:
                continue
            breakdown = scoring_mod.score(
                text=hit.content, role=hit.role, seq=hit.seq,
                now_seq=now_seq, query_terms=query_terms,
                selected_texts=texts, stored_importance=hit.importance,
            )
            scored.append((breakdown["score"], hit.seq, hit, breakdown))
        scored.sort(key=lambda s: (-s[0], s[1]))

        cross = self._cross_session_candidates(
            session_id, query_terms, texts, diags,
            history_requested=history_requested,
        )
        return {
            "same": scored, "cross": cross, "now_seq": now_seq,
            "history_requested": history_requested,
        }

    def _cross_session_candidates(
        self,
        session_id: str,
        query_terms: list[str],
        selected_texts: list[str],
        diagnostics: list[str],
        history_requested: bool = False,
    ) -> list[tuple[float, str, int, StoredMessage, dict[str, float]]]:
        if not self._include_foreign_history:
            return []
        store = self._get_store()
        candidates: list[StoredMessage] = []
        seen_ids: set[int] = set()
        for term in query_terms[:12]:
            if not term or not term.strip():
                continue
            try:
                hits = store.search(None, term, limit=20)
            except Exception as e:
                diagnostics.append(f"retrieval_skipped_term: {term!r}: {e}")
                continue
            for hit in hits:
                if hit.session_id == session_id:
                    continue
                if hit.message_id not in seen_ids:
                    seen_ids.add(hit.message_id)
                    candidates.append(hit)
        if not candidates:
            return []
        if not history_requested:
            scopes = {h.session_id for h in candidates} | {"global"}
            stale = self._superseded_source_ids(store, sorted(scopes))
            if stale:
                before = len(candidates)
                candidates = [h for h in candidates
                              if h.message_id not in stale]
                dropped = before - len(candidates)
                if dropped:
                    diagnostics.append(
                        f"retrieval_suppressed_superseded:"
                        f" {dropped} cross-session message(s)"
                    )
        if not candidates:
            return []
        session_max: dict[str, int] = {}
        for hit in candidates:
            if hit.session_id not in session_max:
                latest = store.fetch_latest(hit.session_id, 1)
                session_max[hit.session_id] = (
                    latest[0].seq if latest else hit.seq
                )
        scored = []
        for hit in candidates:
            breakdown = scoring_mod.score(
                text=hit.content, role=hit.role, seq=hit.seq,
                now_seq=session_max.get(hit.session_id, hit.seq),
                query_terms=query_terms, selected_texts=selected_texts,
                stored_importance=hit.importance,
            )
            scored.append(
                (breakdown["score"], hit.session_id, hit.seq, hit, breakdown)
            )
        scored.sort(key=lambda s: (-s[0], s[1], s[2]))
        return scored

    def _focused_favorite_terms(
        self, query_text: str, previous_user_text: str = ""
    ) -> Optional[set[str]]:
        """Ground a focused favorite lookup in its category/current value.

        Deliberately not used for broad profile/history questions. A short
        anaphoric follow-up ('how do I feel about it?') may inherit the
        category only from the immediately preceding user question.
        """
        if self._wants_history(query_text):
            return None
        category_query = query_text
        category = self._favorite_category(query_text)
        if (category is None and re.search(r"\b(?:it|that)\b", query_text, re.I)
                and _QUESTION_RE.search(query_text)):
            category_query = previous_user_text
            category = self._favorite_category(previous_user_text)
        if not category:
            return None
        # Multiple favorites and historical rankings require broad recall,
        # rather than constraining evidence to a single stored favorite.
        if re.search(r"\b(?:fav|favorite|favourite)\s+[a-z]+s\b",
                     category_query, re.I):
            return None
        try:
            states = self._get_store().fetch_states(
                scope="global", concept=f"favorite {category}",
                status="current",
            )
        except Exception:
            return None
        values = [st for st in states if st.authority == "user"]
        if not values:
            return None
        terms = set(scoring_mod.tokenize(category))
        for st in values:
            # An entire sentence is not a safe topical expansion.
            value_terms = scoring_mod.tokenize(st.value)
            if len(value_terms) <= 4:
                terms.update(value_terms)
        return terms or None

    @staticmethod
    def _focused_evidence_matches(message: StoredMessage,
                                  focus_terms: Optional[set[str]]) -> bool:
        if focus_terms is None:
            return True
        # Aurora's previous interpretations are not user preference evidence.
        if message.role != "user":
            return False
        candidate = {scoring_mod.singular(t) for t in
                     scoring_mod.tokenize(message.content)}
        focus = {scoring_mod.singular(t) for t in focus_terms}
        return bool(candidate & focus)

    def _retrieve(
        self,
        session_id: str,
        history: list[dict],
        current: Optional[dict],
        selected: list[dict],
        selected_fps: set,
        selected_texts: list[str],
        counter: Callable[[list[dict]], int],
        cfg: BudgetConfig,
        total: int,
        diagnostics: list[str],
    ) -> tuple[list[dict], int]:
        store = self._get_store()
        query_text = current.get("content", "") if current else ""
        query_terms = scoring_mod.build_query(
            query_text,
            self._recent_user_text([m for m in history if m is not current]),
        )
        if not query_terms:
            return [], total
        previous_user_text = self._recent_user_text(
            [m for m in history if m is not current]
        )
        focus_terms = self._focused_favorite_terms(
            query_text, previous_user_text
        )
        # Include the actual current favorite in the evidence search;
        # a user's feelings about yellow may not literally say "color".
        if focus_terms:
            query_terms.extend(
                sorted(term for term in focus_terms
                       if term not in query_terms)
            )
        collected = self.collect_lexical_candidates(
            session_id, query_terms, selected_fps, selected_texts,
            diagnostics,
        )
        # The full lexical ranking, kept for the shadow comparison and for the
        # documented fallback decision.
        lexical_ids = [
            hit.message_id for _s, _q, hit, _b in collected["same"]
        ] + [
            hit.message_id for _s, _q, _x, hit, _b in collected["cross"][:3]
        ]

        # M6 LIVE: hybrid retrieval is the returned result. The lexical
        # candidates are one of its channels, not a replacement, and the
        # lexical admission path below remains the fallback so the switch can
        # be rolled back by flipping one flag.
        if self._hybrid_retrieval:
            try:
                # Opportunistic index refresh: one token read when fresh.
                self.maybe_sync_semantic_index(session_id)
                trace = hybrid_mod.hybrid_retrieve(
                    self, store, self.semantic_index(), session_id,
                    session_id, query_text, collected=collected,
                    temporal=("historical"
                              if collected["history_requested"] else "current"),
                )
                self._retrieval_traces.append(trace)
                admitted, total = self._admit_hybrid(
                    session_id, trace, selected_fps, selected_texts, counter,
                    cfg, total, diagnostics, focus_terms=focus_terms,
                )
                if self._shadow_retrieval:
                    self._record_shadow(
                        session_id, current, lexical_ids, trace
                    )
                if admitted:
                    return admitted, total
                if lexical_ids:
                    diagnostics.append(
                        "retrieval_hybrid_empty: falling back to lexical"
                    )
            except Exception as e:
                diagnostics.append(f"retrieval_failed: hybrid: {e}")
        elif self._shadow_retrieval:
            # Hybrid is off but shadow is on: still compute and record what
            # hybrid would have returned, so a rollback can be evaluated.
            self._record_shadow(session_id, current, lexical_ids, None)

        return self._admit_lexical(
            session_id, collected, selected_fps, selected_texts, counter,
            cfg, total, diagnostics, focus_terms=focus_terms,
        )

    def _admit_lexical(
        self,
        session_id: str,
        collected: dict[str, Any],
        selected_fps: set,
        selected_texts: list[str],
        counter: Callable[[list[dict]], int],
        cfg: BudgetConfig,
        total: int,
        diagnostics: list[str],
        focus_terms: Optional[set[str]] = None,
    ) -> tuple[list[dict], int]:
        """The pre-M6 admission path with focused-evidence filtering."""
        admitted: list[dict] = []
        for _score, _seq, hit, _breakdown in collected["same"]:
            if not self._focused_evidence_matches(hit, focus_terms):
                continue
            if total >= cfg.input_ceiling:
                break
            msg = {"role": hit.role, "content": hit.content}
            fp = f"{hit.role}\x00{hit.content}"
            if fp in selected_fps:
                continue
            cost = self._safe_count(counter, [msg], diagnostics)
            if total + cost > cfg.input_ceiling:
                continue
            admitted.append(msg)
            selected_fps.add(fp)
            selected_texts.append(hit.content)
            total += cost
        # Cross-session pass: other sessions' history may inform the current
        # request but is never treated as current-session recency. Capped and
        # provenance-tagged so the model can tell it apart.
        try:
            cross = self._admit_cross_session(
                session_id,
                [item for item in collected["cross"]
                 if self._focused_evidence_matches(item[3], focus_terms)],
                selected_fps,
                selected_texts, counter, cfg, total, diagnostics,
            )
            admitted.extend(cross[0])
            total = cross[1]
        except Exception as e:
            diagnostics.append(f"retrieval_failed: cross-session: {e}")
        return admitted, total

    def _admit_hybrid(
        self,
        session_id: str,
        trace: RetrievalTrace,
        selected_fps: set,
        selected_texts: list[str],
        counter: Callable[[list[dict]], int],
        cfg: BudgetConfig,
        total: int,
        diagnostics: list[str],
        focus_terms: Optional[set[str]] = None,
    ) -> tuple[list[dict], int]:
        """Admit the fused hybrid ranking into the working view.

        A fused `message:` item is admitted directly. A fused `state:` or
        `entity:` item is admitted as the raw evidence messages that support
        it, so what reaches Aurora is always a stored verbatim message with
        its original role. No new prompt format is introduced.

        The current cross-session policy is preserved exactly: same-session
        evidence first in fused order, then at most CROSS_SESSION_CAP foreign
        messages, still tagged `[from another conversation]`.
        """
        store = self._get_store()
        message_ids: list[int] = []
        for key in trace.fused_order:
            kind, _, raw = key.partition(":")
            try:
                ref = int(raw)
            except ValueError:
                continue
            if kind == "message":
                message_ids.append(ref)
            else:
                message_ids.extend(self._evidence_message_ids(kind, ref))
        if not message_ids:
            return [], total
        ordered: list[StoredMessage] = []
        seen: set[int] = set()
        for message in store.fetch_messages_by_ids(message_ids[:256]):
            if not self._focused_evidence_matches(message, focus_terms):
                continue
            if message.message_id in seen:
                continue
            seen.add(message.message_id)
            ordered.append(message)

        same = [m for m in ordered if m.session_id == session_id]
        foreign = (
            [m for m in ordered if m.session_id != session_id]
            if self._include_foreign_history else []
        )
        admitted: list[dict] = []
        limit = hybrid_mod.MAX_CANDIDATES
        emitted = 0
        for message, is_foreign in (
            [(m, False) for m in same]
            + [(m, True) for m in foreign[:cross_session_cap()]]
        ):
            if total >= cfg.input_ceiling or emitted >= limit:
                break
            content = message.content
            if is_foreign:
                content = f"{CROSS_SESSION_PREFIX}{content}"
            fp = f"{message.role}\x00{content}"
            if fp in selected_fps:
                continue
            msg = {"role": message.role, "content": content}
            cost = self._safe_count(counter, [msg], diagnostics)
            if total + cost > cfg.input_ceiling:
                continue
            admitted.append(msg)
            selected_fps.add(fp)
            selected_texts.append(content)
            total += cost
            emitted += 1
        return admitted, total

    def _evidence_message_ids(self, kind: str, ref: int) -> list[int]:
        """Source message ids behind an indexed state or entity."""
        store = self._get_store()
        try:
            if kind == "state":
                rows = store.fetch_states(state_ids=[ref], limit=1)
                return list(rows[0].source_ids) if rows else []
            entity = store.fetch_entity(ref)
            return list(entity.source_ids) if entity else []
        except Exception:
            return []

    def _record_shadow(
        self,
        session_id: str,
        current: Optional[dict],
        lexical_ids: list[int],
        trace: Optional[RetrievalTrace],
    ) -> None:
        """Record the lexical-vs-hybrid comparison for later analysis.

        Never affects the assembled context. Recorded whether or not hybrid
        produced anything, because "hybrid found nothing" is itself a result
        worth comparing against the lexical ranking.
        """
        query_text = current.get("content", "") if current else ""
        try:
            if trace is None:
                store = self._get_store()
                query_terms = scoring_mod.build_query(query_text, "")
                collected = self.collect_lexical_candidates(
                    session_id, query_terms
                )
                trace = hybrid_mod.hybrid_retrieve(
                    self, store, self.semantic_index(), session_id, session_id,
                    query_text, collected=collected,
                )
            self._retrieval_traces.append(trace)
            self._shadow_records.append(hybrid_mod.shadow_comparison(
                lexical_ids, trace, session_id, query_text
            ))
        except Exception as exc:
            self._shadow_records.append(ShadowComparison(
                session_id=session_id, query=query_text,
                lexical_ids=lexical_ids,
                error=f"shadow_failed: {type(exc).__name__}: {exc}",
                degraded=True,
            ))

    def _admit_cross_session(
        self,
        session_id: str,
        scored: list[tuple],
        selected_fps: set,
        selected_texts: list[str],
        counter: Callable[[list[dict]], int],
        cfg: BudgetConfig,
        total: int,
        diagnostics: list[str],
        cap: int = 3,
    ) -> tuple[list[dict], int]:
        if not self._include_foreign_history:
            return [], total
        admitted: list[dict] = []
        for _, _, _, hit, _breakdown in scored[:cap]:
            if total >= cfg.input_ceiling:
                break
            content = f"{CROSS_SESSION_PREFIX}{hit.content}"
            fp = f"{hit.role}\x00{content}"
            if fp in selected_fps:
                continue
            msg = {"role": hit.role, "content": content}
            cost = self._safe_count(counter, [msg], diagnostics)
            if total + cost > cfg.input_ceiling:
                continue
            admitted.append(msg)
            selected_fps.add(fp)
            selected_texts.append(content)
            total += cost
        return admitted, total

    def _retrieve_cross_session(
        self,
        session_id: str,
        query_terms: list[str],
        now_seq: int,
        selected: list[dict],
        selected_fps: set,
        selected_texts: list[str],
        counter: Callable[[list[dict]], int],
        cfg: BudgetConfig,
        total: int,
        diagnostics: list[str],
        cap: int = 3,
        history_requested: bool = False,
    ) -> tuple[list[dict], int]:
        """Cross-session gather + admit.

        Kept as a public seam for existing callers. Candidate gathering now
        lives in `_cross_session_candidates` so the hybrid path can rank the
        same rows without duplicating the logic.
        """
        store = self._get_store()
        if not query_terms:
            return [], total
        scored = self._cross_session_candidates(
            session_id, query_terms, selected_texts, diagnostics,
            history_requested=history_requested,
        )
        return self._admit_cross_session(
            session_id, scored, selected_fps, selected_texts, counter, cfg,
            total, diagnostics, cap=cap,
        )


    @staticmethod
    def _wants_history(text: str) -> bool:
        lowered = (text or "").lower()
        return any(m in lowered for m in HISTORY_MARKERS)

    @staticmethod
    def _wants_self_knowledge(text: str) -> bool:
        return bool(SELF_KNOWLEDGE_RE.search(text or ""))

    @staticmethod
    def _normalize_ident(text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _referenced_sessions(self, store, session_id: str, query: str) -> list[str]:
        """Session ids explicitly named in the query (id or title).

        Matching is normalized case-insensitive substring on identifiers
        and titles. Never invents: unknown names simply match nothing.
        """
        norm_q = self._normalize_ident(query)
        if not norm_q:
            return []
        try:
            sessions = store.list_sessions()
        except Exception:
            return []
        refs: list[str] = []
        for rec in sessions:
            for cand in (rec.session_id, rec.title):
                if not cand:
                    continue
                norm_cand = self._normalize_ident(cand)
                if norm_cand and norm_cand in norm_q and rec.session_id not in refs:
                    refs.append(rec.session_id)
                    break
        return refs

    @staticmethod
    def _wants_cross_session(text: str) -> bool:
        lowered = (text or "").lower()
        return any(m in lowered for m in CROSS_SESSION_MARKERS)

    def _add_session_info(
        self,
        session_id: str,
        current: Optional[dict],
        selected: list[dict],
        selected_fps: set,
        selected_texts: list[str],
        counter: Callable[[list[dict]], int],
        cfg: BudgetConfig,
        total: int,
        diagnostics: list[str],
    ) -> int:
        """Authoritative session metadata outranks historical mentions."""
        query_text = current.get("content", "") if current else ""
        lowered = query_text.lower()
        has_session_word = any(
            w in lowered for w in ("session", "conversation", "chat")
        )
        has_name_word = any(
            w in lowered
            for w in ("called", "name", "named", "titled", "title")
        )
        wants_name = has_session_word and has_name_word
        wants_old_name = any(m in lowered for m in SESSION_HISTORY_MARKERS)
        if not wants_name and not wants_old_name:
            return total
        try:
            store = self._get_store()
            refs = self._referenced_sessions(store, session_id, query_text)
            targets = refs or [session_id]
            lines = []
            for target in targets:
                rec = store.get_session(target)
                if rec is None:
                    continue
                label = rec.title if rec.title else "(untitled)"
                if wants_name:
                    lines.append(
                        f"[session current] id={rec.session_id}"
                        f" name={label!r} status={rec.status}"
                    )
                if wants_old_name:
                    for old in rec.previous_titles:
                        lines.append(
                            f"[session history] {rec.session_id}"
                            f" was previously called {old!r}"
                        )
            for line in lines:
                if total >= cfg.input_ceiling:
                    break
                msg = {"role": "system", "content": line}
                fp = f"system\x00{line}"
                if fp in selected_fps:
                    continue
                cost = self._safe_count(counter, [msg], diagnostics)
                if total + cost > cfg.input_ceiling:
                    continue
                selected.append(msg)
                selected_fps.add(fp)
                selected_texts.append(line)
                total += cost
        except Exception as e:
            diagnostics.append(f"state_failed: session info failed: {e}")
        return total

    def _add_cross_session_states(
        self,
        session_id: str,
        current: Optional[dict],
        selected: list[dict],
        selected_fps: set,
        selected_texts: list[str],
        counter: Callable[[list[dict]], int],
        cfg: BudgetConfig,
        total: int,
        diagnostics: list[str],
    ) -> int:
        """Explicit cross-session questions get per-session scoped values.

        Each value keeps its session identity, lifecycle status, and source
        provenance. Never flattens sessions into one generic answer.
        """
        query_text = current.get("content", "") if current else ""
        if not query_text:
            return total
        try:
            store = self._get_store()
            refs = self._referenced_sessions(store, session_id, query_text)
            explicit = bool(refs) or self._wants_cross_session(query_text)
            if not explicit:
                return total
            wants_history = self._wants_history(query_text)
            query_terms = scoring_mod.build_query(query_text)
            if refs:
                # Referenced other sessions only; the current session is
                # already covered by the scoped sections above.
                targets = [t for t in refs if t != session_id]
            else:
                targets = [
                    s.session_id for s in store.list_sessions()
                    if s.session_id != session_id
                ][:8]
            emitted = 0
            for target in targets:
                if emitted >= 6 or total >= cfg.input_ceiling:
                    break
                try:
                    rows = (
                        store.fetch_states(scope=target, status="current")
                        + store.fetch_states(scope=target, status="unresolved")
                    )
                    if wants_history:
                        rows = rows + store.fetch_states(
                            scope=target, status="superseded")
                except Exception:
                    continue
                ranked = []
                for st in rows:
                    overlap = scoring_mod.relevance(
                        scoring_mod.tokenize(f"{st.concept} {st.value}"),
                        query_terms,
                    )
                    ranked.append((overlap, st.concept, st))
                ranked.sort(key=lambda r: (-r[0], r[1]))
                matched = [r for r in ranked if r[0] > 0]
                if not matched and refs:
                    # Explicitly named session with no term overlap:
                    # show up to 2 of its current values rather than silence.
                    matched = [(0, c, s) for _, c, s in ranked
                               if s.status == "current"][:2]
                for _, _, st in matched:
                    if emitted >= 6 or total >= cfg.input_ceiling:
                        break
                    tag = ("previous"
                           if st.status == "superseded" else st.status)
                    content = (
                        f"[states session {st.session_id} — {tag}"
                        f" src:{','.join(str(i) for i in st.source_ids)}] "
                        f"{st.concept} = {st.value}"
                    )
                    msg = {"role": "system", "content": content}
                    fp = f"system\x00{content}"
                    if fp in selected_fps:
                        continue
                    cost = self._safe_count(counter, [msg], diagnostics)
                    if total + cost > cfg.input_ceiling:
                        continue
                    selected.append(msg)
                    selected_fps.add(fp)
                    selected_texts.append(content)
                    total += cost
                    emitted += 1
        except Exception as e:
            diagnostics.append(f"state_failed: cross-session failed: {e}")
        return total

    @staticmethod
    def _favorite_category(text: str) -> Optional[str]:
        q = (text or "").strip()
        if not q or not _QUESTION_RE.search(q):
            return None
        match = _FAVORITE_QUERY_RE.search(q)
        if not match:
            # Explicit request for the omitted members of a car list.
            if re.search(r"\bother\s+cars?\b", q, re.I):
                return "car"
            return None
        phrase = _CATEGORY_END_RE.split(match.group(1), 1)[0].strip()
        if not phrase:
            return None
        category = slots_mod.favorite_noun(phrase)
        if not category or len(category.split()) > 4:
            return None
        return category

    @staticmethod
    def _current_message_id(store, session_id: str, current: Optional[dict]):
        if not current:
            return None
        content = current.get("content", "")
        if not content:
            return None
        seq = current.get("seq")
        try:
            if isinstance(seq, int) and not isinstance(seq, bool):
                for row in store.fetch_range(session_id, seq, seq):
                    if row.role == "user" and row.content == content:
                        return row.message_id
            for row in reversed(store.fetch_latest(session_id, 4)):
                if row.role == "user" and row.content == content:
                    return row.message_id
        except Exception:
            return None
        return None

    @staticmethod
    def _state_context_matches(store, state, category: str,
                               exclude_id: Optional[int] = None) -> bool:
        terms = {
            scoring_mod.singular(term)
            for term in scoring_mod.tokenize(category)
        }
        if not terms:
            return False
        try:
            sources = store.fetch_messages_by_ids(list(state.source_ids)[:2])
        except Exception:
            return False
        for source in sources:
            if source.role != "user":
                continue
            try:
                window = store.fetch_range(
                    source.session_id, max(0, source.seq - 3),
                    source.seq + 3, limit=12,
                )
            except Exception:
                window = [source]
            found = {
                scoring_mod.singular(term)
                for message in window
                if message.role == "user"
                and message.message_id != exclude_id
                and (
                    _QUESTION_RE.search(message.content)
                    or re.search(r"\b(?:fav|favorite|favourite)\s+"
                                 r"(?:[a-z]+\s+){0,2}"
                                 + re.escape(category) + r"s?\b",
                                 message.content, re.I)
                )
                for term in scoring_mod.tokenize(message.content)
            }
            if terms <= found:
                return True
        return False

    def _add_current_state(
        self,
        session_id: str,
        current: Optional[dict],
        selected: list[dict],
        selected_fps: set,
        selected_texts: list[str],
        counter: Callable[[list[dict]], int],
        cfg: BudgetConfig,
        total: int,
        diagnostics: list[str],
        facts_out: Optional[list[dict]] = None,
    ) -> int:
        store = self._get_store()
        evaluation_time = time.time()
        query_text = current.get("content", "") if current else ""
        query_terms = scoring_mod.build_query(query_text)
        current_id = self._current_message_id(store, session_id, current)
        if _LOCAL_DISCUSSION_RE.fullmatch(query_text):
            # No global personal facts should masquerade as what was said
            # in *this* conversation.
            return total

        # Session scope only: another session's current value must never
        # answer for this session. Cross-session questions are handled
        # explicitly by _add_cross_session_states.
        current_states = (
            store.fetch_states(scope=session_id, status="current")
            + store.fetch_states(scope="global", status="current")
        )
        current_states = [
            state for state in current_states
            if state.authority == "user"
            and self._valid_at(state, evaluation_time)
        ]
        # Single-current invariant (read side): exactly one current value
        # per scope/concept may compete. Newest row wins; older duplicates
        # are skipped loudly so legacy duplicates can never both inject.
        current_states, dupes = _newest_per_scope_concept(current_states)
        for scope, concept, kept_id in dupes:
            diagnostics.append(
                f"state_duplicate_current: {scope}/{concept}"
                f" keeping state {kept_id}"
            )
        # Relevance first so a fact the question actually names lands first;
        # recency then concept, so a broad question ("what do you know about
        # me?") surfaces the newest stored facts instead of an alphabetical
        # one. The admission rule below is per fact, not per rank: a state
        # with no term overlap is still this session's own current fact, and
        # skipping it outright is why "what about my fav car?" could not
        # reach the stored preference at all. An overlapping fact may use the
        # full ceiling whatever its scope - the question named it. A
        # non-overlapping one is admitted only inside the state reserve, so
        # it can never displace history or a fact the question named, and
        # only from this session or on a broad self-knowledge question: an
        # unrelated global must still earn its place by overlapping, so it
        # cannot crowd a turn out of the session's own memory.
        wants_self_knowledge = self._wants_self_knowledge(query_text)
        bridge_category = (
            self._favorite_category(query_text) if query_text else None
        )
        # Do not suppress other liked items merely because a primary
        # favorite is stored. "My favorite car" and "cars I also love"
        # are separate facts and must both be available for list queries.
        ranked = []
        for st in current_states:
            overlap = scoring_mod.relevance(
                scoring_mod.tokenize(f"{st.concept} {st.value}"), query_terms
            )
            ranked.append((overlap, st))
        ranked.sort(key=lambda r: (-r[0], -r[1].state_id, r[1].concept))
        bridge_ids: set[int] = set()
        if bridge_category:
            for overlap, st in ranked:
                # A narrower favorite category ("classic car") is also a
                # valid car favorite. It should be rendered as a fact,
                # rather than treated as an inferred preference.
                if st.concept.startswith("favorite "):
                    suffix = st.concept[len("favorite "):]
                    if suffix == bridge_category or suffix.endswith(
                            " " + bridge_category):
                        bridge_ids.add(st.state_id)
                    continue
                if not st.concept.startswith("likes for "):
                    continue
                if self._state_context_matches(
                        store, st, bridge_category, current_id):
                    bridge_ids.add(st.state_id)
        reserve_limit = min(cfg.input_ceiling, cfg.input_target)
        # Facts render as plain sentences under one heading, never as a
        # machine format. The heading carries the usage guidance and is
        # admitted on the ceiling (it is fixed cost); the facts keep the
        # existing reserve/overlap admission rules unchanged.
        heading = slots_mod.block_heading()
        heading_msg = {"role": "system", "content": heading}
        heading_fp = f"system\x00{heading}"
        heading_added = False

        def _emit(content: str, limit: int) -> None:
            nonlocal total, heading_added
            msg = {"role": "system", "content": content}
            fp = f"system\x00{content}"
            if fp in selected_fps:
                return
            cost = self._safe_count(counter, [msg], diagnostics)
            if total + cost > limit:
                return
            if not heading_added:
                if heading_fp in selected_fps:
                    heading_added = True
                else:
                    heading_cost = self._safe_count(
                        counter, [heading_msg], diagnostics)
                    if total + heading_cost + cost <= cfg.input_ceiling:
                        selected.append(heading_msg)
                        selected_fps.add(heading_fp)
                        selected_texts.append(heading)
                        total += heading_cost
                        if facts_out is not None:
                            facts_out.append(heading_msg)
                    heading_added = True
            selected.append(msg)
            selected_fps.add(fp)
            selected_texts.append(content)
            total += cost
            if facts_out is not None:
                facts_out.append(msg)

        for overlap, st in ranked:
            if total >= cfg.input_ceiling:
                break
            if st.state_id in bridge_ids:
                continue
            if overlap > 0:
                limit = cfg.input_ceiling
            elif st.scope == session_id or wants_self_knowledge:
                limit = reserve_limit
            else:
                continue
            if total >= limit:
                continue
            _emit(slots_mod.render_sentence(st.concept, st.value), limit)

        for _, st in ranked:
            if total >= cfg.input_ceiling:
                break
            if st.state_id not in bridge_ids:
                continue
            # The canonical state must be visible alongside its raw source;
            # a user may have misspelled the name in the source message.
            _emit(slots_mod.render_sentence(st.concept, st.value),
                  cfg.input_ceiling)
            if st.concept.startswith("favorite "):
                continue
            try:
                sources = store.fetch_messages_by_ids(
                    list(st.source_ids)[:2])
            except Exception:
                continue
            for source in sources:
                if source.role != "user" or not source.content:
                    continue
                text = re.sub(r"\s+", " ", source.content).strip()
                if len(text) > 300:
                    text = text[:297].rstrip() + "..."
                content = (
                    f'The user said about their favorite {bridge_category}: '
                    f'"{text}".'
                )
                _emit(content, cfg.input_ceiling)

        # Unresolved conflicts stay visible when relevant, never as truth.
        try:
            open_conflicts = (
                store.fetch_states(scope=session_id, status="unresolved")
                + store.fetch_states(scope="global", status="unresolved")
            )
        except Exception:
            open_conflicts = []
        for st in open_conflicts:
            overlap = scoring_mod.relevance(
                scoring_mod.tokenize(f"{st.concept} {st.value}"), query_terms
            )
            if overlap <= 0 or total >= cfg.input_ceiling:
                continue
            content = (
                f"Possible conflict (not established): {st.concept}"
                f" may be {st.value}."
            )
            msg = {"role": "system", "content": content}
            fp = f"system\x00{content}"
            if fp in selected_fps:
                continue
            cost = self._safe_count(counter, [msg], diagnostics)
            if total + cost > cfg.input_ceiling:
                continue
            selected.append(msg)
            selected_fps.add(fp)
            selected_texts.append(content)
            total += cost
        return total

    def _add_digests(
        self,
        session_id: str,
        current: Optional[dict],
        history: list[dict],
        selected: list[dict],
        selected_fps: set,
        selected_texts: list[str],
        counter: Callable[[list[dict]], int],
        cfg: BudgetConfig,
        total: int,
        diagnostics: list[str],
    ) -> int:
        store = self._get_store()
        query_terms = scoring_mod.build_query(
            current.get("content", "") if current else ""
        )

        def _str_counter(text: str) -> int:
            return self._safe_count(
                counter, [{"role": "user", "content": text}], diagnostics
            )

        # Opportunistic compaction of old uncovered history. Only rows
        # not already represented by a digest are processed, so repeated
        # builds never re-digest the same material.
        try:
            raw_total = store.count(session_id)
            if raw_total >= COMPACT_AFTER_MESSAGES:
                oldest = store.fetch_range(session_id, 0, raw_total - 20, limit=60)
                if oldest:
                    existing = store.fetch_digests(session_id, limit=10)
                    covered = set()
                    for d in existing:
                        covered.update(d["source_ids"])
                    uncovered = [m for m in oldest if m.message_id not in covered]
                    if len(uncovered) >= 20:
                        maybe_compact(
                            store,
                            session_id,
                            max(m.seq for m in uncovered),
                            query_terms=query_terms or None,
                            token_counter=_str_counter,
                            messages=uncovered,
                        )
        except Exception as e:
            diagnostics.append(f"digest_failed: compaction: {e}")

        # Include newest stored digests while budget allows.
        try:
            digests = store.fetch_digests(session_id, limit=3)
        except Exception as e:
            raise RuntimeError(f"digest read failed: {e}")
        for d in digests:
            if total >= cfg.input_ceiling:
                break
            msg = {
                "role": "system",
                "content": f"[history digest {d['seq_start']}-{d['seq_end']}]"
                f"\n{d['content']}",
            }
            fp = f"system\x00{msg['content']}"
            if fp in selected_fps:
                continue
            cost = self._safe_count(counter, [msg], diagnostics)
            if total + cost > cfg.input_ceiling:
                continue
            selected.append(msg)
            selected_fps.add(fp)
            selected_texts.append(msg["content"])
            total += cost
        return total

    def _add_memories(
        self,
        session_id: str,
        current: Optional[dict],
        history: list[dict],
        selected: list[dict],
        selected_fps: set,
        selected_texts: list[str],
        counter: Callable[[list[dict]], int],
        cfg: BudgetConfig,
        total: int,
        diagnostics: list[str],
    ) -> int:
        provider = self._memory_provider
        if provider is None:
            return total
        query = current.get("content", "") if current else ""
        try:
            if callable(provider):
                memories = provider(session_id, query, 5)
            else:
                memories = provider.retrieve(session_id, query, 5)
        except Exception as e:
            raise RuntimeError(f"memory provider failed: {e}")
        for mem in memories or []:
            if total >= cfg.input_ceiling:
                break
            role = str(mem.get("role", "system"))
            content = str(mem.get("content", ""))
            if not content:
                continue
            msg = {"role": role, "content": f"[memory]\n{content}"}
            fp = f"{role}\x00{msg['content']}"
            if fp in selected_fps:
                continue
            cost = self._safe_count(counter, [msg], diagnostics)
            if total + cost > cfg.input_ceiling:
                continue
            selected.append(msg)
            selected_fps.add(fp)
            selected_texts.append(msg["content"])
            total += cost
        return total
