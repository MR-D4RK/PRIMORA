"""Mind: the cognitive boundary between an event and Aurora.

Milestone 1. This module introduces the Mind abstraction at the point where
PRIMORA currently hands a raw ``ContextManager`` to Aurora.

    Amir / Event
        -> Mind
             |- current conversation context
             |- Working Memory (runtime, task-local)
             |- Active Memory access
             |- M6 retrieval (as a mechanism, unchanged)
             |- strict assembly / filtering / budgets / fallbacks
        -> one unified CognitiveContext
        -> Aurora

What this is
------------
A *façade plus a unified result type*. Mind owns no new storage and no new
retrieval logic. It composes capabilities that already exist:

* conversation context and strict assembly -> ``ContextManager.build``
* M6 retrieval                            -> ``ContextManager`` internally
* Active Memory                           -> ``ContextManager.get_active_memory``
* Working Memory                          -> ``WorkingMemory``, held per Mind

The mechanisms (M4 formation, M5 entities, M6 hybrid retrieval, the store)
stay underneath, unchanged and individually testable. M4/M5/M6 remain
technical mechanisms; they are not architectural layers, and nothing in this
module treats them as such.

What this deliberately does NOT do
----------------------------------
* No cognitive loop. ``build`` remains a pure request-scoped assembly.
* No Jev, and no decision hook of any kind.
* No new model, no new retrieval channel, no threshold or calibration change.
* No change to what Aurora receives. ``CognitiveContext.messages`` is exactly
  the message list ``ContextManager.build`` produced before, byte for byte.

The unified context
-------------------
``CognitiveContext`` is the single object handed to Aurora. It carries the
rendered messages plus the sections that produced them (Active Memory, the
task-local Working Memory, and a read-only projection of the M6 trace), so a
caller can reason about *why* a context looked the way it did without
re-deriving it.

Compatibility
-------------
``Mind`` accepts a ``ContextManager`` and Aurora still accepts either a
``ContextManager`` or a ``Mind``. Every existing call site keeps working
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from .manager import ContextManager
from .models import (
    ActiveMemory,
    BudgetConfig,
    BuildResult,
    ContextView,
    RetrievalTrace,
)
from .working import WorkingMemory


# --------------------------------------------------------------------------
# Read-only projections
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalSummary:
    """Read-only projection of the M6 trace for one request.

    Deliberately a summary, not the trace itself: a caller inspecting the
    mind should not be able to mutate retrieval state. Channel counts and
    the fused length are the parts that explain what Aurora was given.
    """

    attempted: bool
    hybrid: bool
    degraded: bool
    channel_counts: dict[str, int] = field(default_factory=dict)
    fused_count: int = 0
    included_count: int = 0
    admitted_count: int = 0
    total_ms: float = 0.0
    diagnostics: tuple[str, ...] = ()

    @classmethod
    def from_trace(
        cls,
        trace: Optional[RetrievalTrace],
        *,
        hybrid: bool,
        admitted_count: int,
    ) -> "RetrievalSummary":
        if trace is None:
            return cls(attempted=False, hybrid=hybrid, degraded=False,
                       admitted_count=admitted_count)
        counts: dict[str, int] = {}
        for candidate in trace.candidates:
            counts[candidate.channel] = counts.get(candidate.channel, 0) + 1
        return cls(
            attempted=True,
            hybrid=hybrid,
            degraded=trace.degraded,
            channel_counts=counts,
            fused_count=len(trace.fused_order),
            included_count=sum(1 for c in trace.candidates if c.included),
            admitted_count=admitted_count,
            total_ms=trace.total_ms,
            diagnostics=tuple(trace.diagnostics),
        )


# --------------------------------------------------------------------------
# The unified context
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CognitiveContext:
    """One unified context for Aurora.

    ``messages`` is the contract: it is exactly what the strict assembly
    produced, and it is what Aurora generates from. The remaining fields are
    the sections that produced it, exposed for inspection.
    """

    messages: list[dict[str, str]]
    session_id: str
    input_tokens: int
    allowed_output: int
    raw_count: int
    selected_count: int
    diagnostics: list[str] = field(default_factory=list)
    degraded: bool = False
    active_memory: Optional[ActiveMemory] = None
    working: Optional[WorkingMemory] = None
    retrieval: RetrievalSummary = field(
        default_factory=lambda: RetrievalSummary(
            attempted=False, hybrid=False, degraded=False)
    )

    @property
    def view(self) -> ContextView:
        """The underlying ContextView, for existing callers.

        Kept so code that already consumed ``BuildResult.view`` keeps
        working unchanged.
        """
        return ContextView(
            messages=self.messages,
            input_tokens=self.input_tokens,
            allowed_output=self.allowed_output,
            session_id=self.session_id,
            diagnostics=list(self.diagnostics),
        )

    @classmethod
    def from_build(
        cls,
        result: BuildResult,
        *,
        working: Optional[WorkingMemory],
        retrieval: RetrievalSummary,
        active_memory: Optional[ActiveMemory] = None,
    ) -> "CognitiveContext":
        view = result.view
        return cls(
            messages=view.messages,
            session_id=view.session_id,
            input_tokens=view.input_tokens,
            allowed_output=view.allowed_output,
            raw_count=result.raw_count,
            selected_count=result.selected_count,
            diagnostics=list(view.diagnostics),
            degraded=result.degraded,
            active_memory=active_memory,
            working=working,
            retrieval=retrieval,
        )

    def with_appended(
        self, messages: list[dict[str, str]], diagnostics: list[str],
    ) -> "CognitiveContext":
        """The same context with a tool turn appended and re-assembled.

        Used for the second leg of a Prime request. Returns a new context so
        the first leg stays available for inspection.
        """
        return CognitiveContext(
            messages=messages,
            session_id=self.session_id,
            input_tokens=self.input_tokens,
            allowed_output=self.allowed_output,
            raw_count=self.raw_count,
            selected_count=self.selected_count,
            diagnostics=list(self.diagnostics) + list(diagnostics),
            degraded=self.degraded,
            active_memory=self.active_memory,
            working=self.working,
            retrieval=self.retrieval,
        )


# --------------------------------------------------------------------------
# Mind
# --------------------------------------------------------------------------


class Mind:
    """The cognitive boundary for one conversation.

    Owns the runtime Working Memory for this mind, orchestrates the existing
    context assembly, and hands back one unified ``CognitiveContext``.

    The Mind is a façade. It deliberately does not re-implement, mirror, or
    second-guess assembly, gating, budgeting, or retrieval: it calls the
    existing ``ContextManager`` for all of that, so M4/M5/M6 keep exactly one
    implementation and one set of tests.
    """

    def __init__(
        self,
        context_manager: ContextManager,
        working: Optional[WorkingMemory] = None,
    ) -> None:
        if context_manager is None:
            raise ValueError("Mind requires a ContextManager")
        self._mechanisms = context_manager
        self._working = working
        self._last: Optional[CognitiveContext] = None
        self._traces_seen = 0

    # -- composition ------------------------------------------------------

    @property
    def mechanisms(self) -> ContextManager:
        """The memory mechanisms underneath. M4/M5/M6 live here."""
        return self._mechanisms

    @property
    def working(self) -> Optional[WorkingMemory]:
        """Task-local Working Memory. Runtime only, never persisted."""
        return self._working

    @working.setter
    def working(self, value: Optional[WorkingMemory]) -> None:
        self._working = value

    def begin_task(self, session_id: str, task: str = "", **kwargs: Any) -> WorkingMemory:
        """Start a task with a fresh, bounded, non-persistent working set."""
        self._working = WorkingMemory(session_id, task, **kwargs)
        return self._working

    def end_task(self) -> None:
        """Drop the working set. Nothing durable is affected."""
        if self._working is not None:
            self._working.clear()
        self._working = None

    @property
    def last_context(self) -> Optional[CognitiveContext]:
        return self._last

    @property
    def current_session_id(self) -> str:
        """The session of the task in progress, or ``"default"``.

        A capability must record into the session the conversation is
        actually in. Guessing ``"default"`` writes memory into a different
        scope than the dialogue, which then looks like the same memory
        appearing twice: one row from the capability, one from ordinary
        formation of the user's own words. The Mind already knows, so
        capabilities ask instead of assuming.
        """
        session_id = getattr(self._working, "session_id", None)
        if isinstance(session_id, str) and session_id:
            return session_id
        last = self._last
        session_id = getattr(last, "session_id", None)
        if isinstance(session_id, str) and session_id:
            return session_id
        return "default"

    # -- Active Memory ----------------------------------------------------

    def active_memory(
        self, session_id: str = "default", at_time: Optional[float] = None,
        include_global: bool = True, digest_limit: int = 3,
    ) -> ActiveMemory:
        """What is currently true and useful. Read-only projection."""
        return self._mechanisms.get_active_memory(
            session_id, include_global=include_global,
            digest_limit=digest_limit, at_time=at_time,
        )

    def select_active(
        self, session_id: str, query_text: str, limit: int = 5,
    ) -> list[Any]:
        """Rank Active Memory for a task using the existing relevance rule."""
        return self._mechanisms.select_active_memory(
            session_id, query_text, limit
        )

    def learn(
        self,
        text: str,
        provenance: str,
        jev: Any = None,
        session_id: str = "default",
    ) -> dict[str, Any]:
        """Admit an Aurora-proposed lesson through the existing lifecycle.

        Milestone 3. A façade over ``ContextManager.learn``, exactly as
        :meth:`recall` is a façade over the existing retrieval. The Mind owns
        no storage, no schema, and no admission policy of its own.
        """
        summary = self._mechanisms.learn(
            text, provenance, jev, session_id=session_id
        )
        self._flag_scope_mismatch(session_id, summary)
        return summary

    def _flag_scope_mismatch(
        self, session_id: str, summary: dict[str, Any]
    ) -> None:
        """Report a lesson recorded outside the live conversation.

        An earlier bug wrote to ``"default"`` while the conversation lived
        elsewhere, which looked like one memory stored twice and was not
        caught by any test. This is the detector that was missing: a write
        into a different scope than the active session is noted, so the
        mismatch is visible instead of silent.
        """
        if not isinstance(summary, dict) or summary.get("status") == "rejected":
            return
        active = self.current_session_id
        if active == "default" or session_id == active:
            return
        note = (
            f"scope_mismatch: lesson recorded in session {session_id!r} while "
            f"the active conversation is {active!r}. The memory will not be "
            f"retrieved as part of this conversation."
        )
        diagnostics = summary.setdefault("diagnostics", [])
        if note not in diagnostics:
            diagnostics.append(note)

    @property
    def user_texts(self) -> list[str]:
        """What Amir has said in the current session, verbatim.

        A neutral read. The Mind exposes what the user said; it does not
        decide what that means for provenance, and it does not import the
        learning package. The capability that needs the judgement applies it.
        """
        try:
            store = self._mechanisms._get_store()
            rows = store.fetch_range(self.current_session_id, 0, 500)
        except Exception:
            return []
        return [r.content for r in rows if getattr(r, "role", "") == "user"]

    def recall(
        self, query: str, session_id: str = "default", limit: int = 5,
    ) -> list[Any]:
        """Ask the existing M6 retrieval a question directly.

        Milestone 3. Delegates to ``ContextManager.recall``, which runs the
        same hybrid retrieval the context builder uses. The Mind stays a
        façade: no new ranking, no new thresholds, no second retrieval
        implementation.
        """
        return self._mechanisms.recall(query, session_id=session_id, limit=limit)

    # -- conversation -----------------------------------------------------

    def reconcile(
        self, messages: list[dict[str, Any]], session_id: str = "default",
    ) -> list[str]:
        """Persist a conversation snapshot and run memory formation.

        Delegates to the existing formation path, so M4 and M5 keep exactly
        one implementation.
        """
        return self._mechanisms.reconcile(messages, session_id)

    # -- the unified context ---------------------------------------------

    def build(
        self,
        messages: list[dict[str, str]],
        session_id: str = "default",
        token_counter: Optional[Callable[[list[dict]], int]] = None,
        max_new_tokens: int = 256,
        budget: Optional[BudgetConfig] = None,
        working: Optional[WorkingMemory] = None,
        include_active_memory: bool = True,
    ) -> CognitiveContext:
        """Assemble the one unified context for Aurora.

        ``working`` overrides the task-local working set for this call only.
        Assembly, gating, budgets, and fallbacks are the existing ones.
        """
        effective_working = working if working is not None else self._working

        before_traces = len(self._mechanisms.retrieval_traces())
        result = self._mechanisms.build(
            messages,
            session_id,
            token_counter=token_counter,
            max_new_tokens=max_new_tokens,
            budget=budget,
            working=effective_working,
        )

        traces = self._mechanisms.retrieval_traces()
        trace = traces[-1] if len(traces) > before_traces else None
        summary = RetrievalSummary.from_trace(
            trace,
            hybrid=self._mechanisms.hybrid_retrieval,
            admitted_count=result.selected_count,
        )

        active = None
        if include_active_memory:
            try:
                active = self.active_memory(result.view.session_id)
            except Exception:
                active = None

        context = CognitiveContext.from_build(
            result, working=effective_working, retrieval=summary,
            active_memory=active,
        )
        self._last = context
        return context

    def context_for(
        self,
        messages: list[dict[str, str]],
        session_id: str = "default",
        **kwargs: Any,
    ) -> CognitiveContext:
        """Alias for :meth:`build`; reads better at call sites."""
        return self.build(messages, session_id, **kwargs)

    # -- introspection ----------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Read-only health of the mind and its mechanisms."""
        out: dict[str, Any] = {
            "working": self._working is not None,
            "working_items": (
                self._working.entry_count() if self._working else 0
            ),
            "has_context": self._last is not None,
        }
        try:
            out["hybrid_retrieval"] = self._mechanisms.hybrid_retrieval
            out["shadow_retrieval"] = self._mechanisms.shadow_retrieval
            out["semantic_index"] = self._mechanisms.semantic_index_status()
        except Exception as exc:
            out["status_failed"] = str(exc)
        return out
