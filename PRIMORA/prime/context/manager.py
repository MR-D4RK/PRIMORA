"""ContextManager: Prime-owned working-context orchestration.

Raw history (SQLite, append-only) is the permanent source of truth.
`build()` assembles a bounded, ordered, deduplicated working view for one
model call. Never deletes raw rows. Degrades safely, never invents facts.
"""

import re
import time
from typing import Any, Callable, Optional

from . import scoring as scoring_mod
from . import state as state_mod
from .digest import build_extractive_digest, maybe_compact
from .models import (
    BudgetConfig,
    BuildResult,
    ContextItem,
    ContextView,
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

# Start compacting once uncovered history passes this many messages.
COMPACT_AFTER_MESSAGES = 50


class ContextTooLargeError(Exception):
    """The current user message alone exceeds the hard input ceiling."""


def estimate_tokens(text: str) -> int:
    """Conservative fallback estimator used only when no real counter works."""
    return max(1, len(text) // 4)


class ContextManager:
    """Prime-owned context infrastructure."""

    def __init__(
        self,
        db_path=None,
        budget: Optional[BudgetConfig] = None,
        memory_provider=None,
    ):
        self._db_path = db_path
        self._budget = budget or BudgetConfig.from_env()
        self._memory_provider = memory_provider
        self._store: Optional[ContextStore] = None

    @property
    def budget(self) -> BudgetConfig:
        return self._budget

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
    def reconcile(
        self,
        messages: list[dict[str, str]],
        session_id: str = DEFAULT_SESSION_ID,
    ) -> list[str]:
        """Persist a conversation snapshot idempotently.

        Sequence identity = position in the snapshot. Re-sending the same
        snapshot is a no-op. Same position with different content is stored
        as a separate row plus a diagnostic; nothing is overwritten.
        Returns diagnostics (empty on the happy path).
        """
        diagnostics: list[str] = []
        try:
            store = self._get_store()
        except Exception as e:
            return [f"store_unavailable: {e}"]
        _, session_diags = self.ensure_session(session_id)
        diagnostics.extend(session_diags)
        for seq, msg in enumerate(messages):
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            try:
                importance = scoring_mod.importance_of(content, role)
                message_id, is_new, conflict = store.append(
                    session_id, seq, role, content, importance
                )
                if conflict:
                    diagnostics.append(conflict)
                if is_new:
                    diagnostics.extend(
                        self._extract_states(store, session_id, message_id, role, content)
                    )
            except Exception as e:
                diagnostics.append(f"store_unavailable: {e}")
                break
        return diagnostics

    @staticmethod
    def _extract_states(
        store,
        session_id: str,
        message_id: int,
        role: str,
        content: str,
    ) -> list[str]:
        """Deterministic state extraction. Never invents on failure."""
        try:
            if role == "user":
                candidates = state_mod.extract_candidates(content, role)
                if not candidates:
                    return []
                return state_mod.apply_user_candidates(
                    store, session_id, message_id, candidates
                )
            if role == "assistant":
                return state_mod.check_assistant_conflict(
                    store, session_id, message_id, content
                )
        except Exception as e:
            return [f"state_failed: extraction failed: {e}"]
        return []

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
    ) -> BuildResult:
        """Assemble a budgeted working view for one model call."""
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

        # 4. Newest conversation turns up to the input target.
        rest = [m for m in history if m is not current]
        for msg in reversed(rest):
            if total >= cfg.input_target:
                break
            fp = self._fp(msg)
            if fp in selected_fps:
                continue
            cost = self._safe_count(counter, [msg], diagnostics)
            if total + cost > cfg.input_target:
                continue
            selected.append(msg)
            selected_fps.add(fp)
            selected_texts.append(msg.get("content", ""))
            total += cost

        raw_count = self._raw_count(session_id)
        degraded = False

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
            state_total = self._add_current_state(
                session_id, current, selected, selected_fps,
                selected_texts, counter, cfg, total, diagnostics,
            )
            total = state_total
        except Exception as e:
            degraded = True
            diagnostics.append(f"state_failed: assembly failed: {e}")

        # 6. Retrieval over older material (only what still fits).
        try:
            admitted = self._retrieve(
                session_id, history, current, selected,
                selected_fps, selected_texts, counter, cfg, total,
                diagnostics,
            )
            selected.extend(admitted[0])
            total = admitted[1]
        except Exception as e:
            degraded = True
            diagnostics.append(f"retrieval_failed: {e}")

        # 6b. Explicit cross-session state questions get per-session
        # scoped values with identity; never flattened, never silent.
        try:
            total = self._add_cross_session_states(
                session_id, current, selected, selected_fps,
                selected_texts, counter, cfg, total, diagnostics,
            )
        except Exception as e:
            degraded = True
            diagnostics.append(f"state_failed: cross-session failed: {e}")

        # 7. Digests (stored first, then freshly compacted if warranted).
        try:
            d_total = self._add_digests(
                session_id, current, history, selected, selected_fps,
                selected_texts, counter, cfg, total, diagnostics,
            )
            total = d_total
        except Exception as e:
            degraded = True
            diagnostics.append(f"digest_failed: {e}")

        # 8. Optional memory hook (no-op in V1).
        try:
            total = self._add_memories(
                session_id, current, history, selected, selected_fps,
                selected_texts, counter, cfg, total, diagnostics,
            )
        except Exception as e:
            degraded = True
            diagnostics.append(f"memory_failed: {e}")

        # 9. Reassembly: system instruction, then synthesized context
        # (digests/memory, oldest first), then conversation in original
        # chronological order with the current turn in its true position.
        history_pos = {id(m): i for i, m in enumerate(history)}
        convo = [m for m in selected if id(m) in history_pos]
        if current is not None and all(m is not current for m in convo):
            convo.append(current)
        convo.sort(key=lambda m: history_pos[id(m)])
        # NOTE: system_msg lives in `selected` (as the protected base) but
        # is prepended explicitly below, so it must be excluded here to
        # avoid emitting the system instruction twice.
        synth = [
            m for m in selected
            if id(m) not in history_pos
            and not (system_msg is not None and m is system_msg)
        ]
        ordered = (
            ([system_msg] if system_msg is not None else [])
            + synth
            + [m for m in convo if not (system_msg is not None and m is system_msg)]
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
        query_terms = scoring_mod.build_query(
            current.get("content", "") if current else "",
            self._recent_user_text([m for m in history if m is not current]),
        )
        if not query_terms:
            return [], total
        try:
            latest = store.fetch_latest(session_id, 200)
        except Exception as e:
            raise RuntimeError(f"store read failed: {e}")
        now_seq = max((m.seq for m in latest), default=0)
        admitted: list[dict] = []
        # Gather unique candidates across query terms, then rank globally.
        candidates: list[StoredMessage] = []
        seen_ids: set[int] = set()
        for term in query_terms[:12]:
            if not term or not term.strip():
                diagnostics.append("retrieval_skipped_term: empty term")
                continue
            try:
                hits = store.search(session_id, term, limit=10)
            except Exception as e:
                # One bad term must not abandon the whole retrieval pass.
                diagnostics.append(f"retrieval_skipped_term: {term!r}: {e}")
                continue
            for hit in hits:
                if hit.message_id not in seen_ids:
                    seen_ids.add(hit.message_id)
                    candidates.append(hit)
        scored = []
        for hit in candidates:
            text = hit.content
            if f"{hit.role}\x00{text}" in selected_fps:
                continue
            breakdown = scoring_mod.score(
                text=text,
                role=hit.role,
                seq=hit.seq,
                now_seq=now_seq,
                query_terms=query_terms,
                selected_texts=selected_texts,
                stored_importance=hit.importance,
            )
            scored.append((breakdown["score"], hit.seq, hit, breakdown))
        scored.sort(key=lambda s: (-s[0], s[1]))
        for _, _, hit, _ in scored:
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

        # Cross-session pass: other sessions' history may inform the
        # current request but is never treated as current-session recency.
        # Capped and provenance-tagged so the model can tell it apart.
        try:
            cross = self._retrieve_cross_session(
                session_id, query_terms, now_seq, selected,
                selected_fps, selected_texts, counter, cfg, total,
                diagnostics,
            )
            admitted.extend(cross[0])
            total = cross[1]
        except Exception as e:
            diagnostics.append(f"retrieval_failed: cross-session: {e}")
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
    ) -> tuple[list[dict], int]:
        store = self._get_store()
        candidates: list[StoredMessage] = []
        seen_ids: set[int] = set()
        for term in query_terms[:12]:
            if not term or not term.strip():
                diagnostics.append("retrieval_skipped_term: empty term")
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
            return [], total
        # Recency is per-session: newest of each foreign session first.
        session_max: dict[str, int] = {}
        for hit in candidates:
            if hit.session_id not in session_max:
                latest = store.fetch_latest(hit.session_id, 1)
                session_max[hit.session_id] = latest[0].seq if latest else hit.seq
        scored = []
        for hit in candidates:
            breakdown = scoring_mod.score(
                text=hit.content,
                role=hit.role,
                seq=hit.seq,
                now_seq=session_max.get(hit.session_id, hit.seq),
                query_terms=query_terms,
                selected_texts=selected_texts,
                stored_importance=hit.importance,
            )
            scored.append((breakdown["score"], hit.session_id, hit.seq, hit))
        scored.sort(key=lambda s: (-s[0], s[1], s[2]))
        admitted: list[dict] = []
        for _, _, _, hit in scored[:cap]:
            if total >= cfg.input_ceiling:
                break
            content = f"[from another conversation]\n{hit.content}"
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

    @staticmethod
    def _wants_history(text: str) -> bool:
        lowered = (text or "").lower()
        return any(m in lowered for m in HISTORY_MARKERS)

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
    ) -> int:
        store = self._get_store()
        query_text = current.get("content", "") if current else ""
        query_terms = scoring_mod.build_query(query_text)
        wants_history = self._wants_history(query_text)

        # Session scope only: another session's current value must never
        # answer for this session. Cross-session questions are handled
        # explicitly by _add_cross_session_states.
        current_states = (
            store.fetch_states(scope=session_id, status="current")
            + store.fetch_states(scope="global", status="current")
        )
        ranked = []
        for st in current_states:
            overlap = scoring_mod.relevance(
                scoring_mod.tokenize(f"{st.concept} {st.value}"), query_terms
            )
            ranked.append((overlap, st.concept, st))
        ranked.sort(key=lambda r: (-r[0], r[1]))
        for overlap, _, st in ranked:
            if overlap <= 0:
                continue
            if total >= cfg.input_ceiling:
                break
            content = (
                f"[state current src:{','.join(str(i) for i in st.source_ids)}] "
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
                f"[state unresolved src:{','.join(str(i) for i in st.source_ids)}] "
                f"{st.concept}: {st.value} (conflicting, not established)"
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

        # Superseded values surface only on explicit historical queries.
        if wants_history:
            try:
                old_states = (
                    store.fetch_states(scope=session_id, status="superseded")
                    + store.fetch_states(scope="global", status="superseded")
                )
            except Exception:
                old_states = []
            ranked_old = []
            for st in old_states:
                overlap = scoring_mod.relevance(
                    scoring_mod.tokenize(f"{st.concept} {st.value}"), query_terms
                )
                if overlap > 0:
                    ranked_old.append((overlap, st.concept, st))
            ranked_old.sort(key=lambda r: (-r[0], r[1]))
            for _, _, st in ranked_old[:3]:
                if total >= cfg.input_ceiling:
                    break
                content = (
                    f"[state previous src:{','.join(str(i) for i in st.source_ids)}] "
                    f"{st.concept} was: {st.value} (superseded)"
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
