"""Aurora cognitive loop: one logical response, many cognition segments.

Milestone 2 introduced the loop; Milestone 3 made it usable. This is the
v0.3 runtime, and it is the *runtime mechanism* that lets Aurora sustain a
genuine multi-step cognitive process while still producing a single logical
response.

    EVENT
      -> Mind builds one CognitiveContext          (M1 boundary, unchanged)
      -> Aurora enters the cognitive loop
      -> Aurora generates one cognition segment
      -> the loop decides structurally whether cognition continues
           continue -> next segment
           finish   -> that segment is the response

    A typical v0.3 turn, where Aurora draws on several capabilities and then
    stops when she has enough:

        think -> recall -> think -> research -> think -> tool -> think -> answer

What this is
------------
Hardcoded *control flow*, not hardcoded *cognition*. The loop contains no
statement of the form "this means Aurora should research" or "this means
Aurora should act". It only knows one structural question:

    did this segment emit a capability request?

and one resource question:

    are cycles left?

The answer to "should cognition continue" is a *policy*, injected. The
default policy is structural and is what the pre-loop code did implicitly:
continue while a capability request is present and cycles remain.

When Aurora is finished is her decision, not the loop's. She signals it by
emitting no request; the loop has no opinion about whether she has enough.

What this deliberately does NOT do
----------------------------------
* No research, no memory-search tools, no specialist calls, no arbitrary tool
  calling, no proactive triggers, no goals, no Jev. Those are Milestone 3+.
* No second brain and no second agent. The loop reuses the existing
  ``AuroraBrain`` instance and the existing ``Mind``.
* No new model, and no separate cognitive decision model.
* No re-implementation of assembly, gating, budgets, or retrieval: every
  segment goes through the same ``Mind.build`` the pre-loop code called.

Bounded and deterministic
-------------------------
Three independent structural guarantees, in order of how hard they bite:

1. ``max_cycles`` is a hard ceiling on generations. Unreachable code is not
   relied upon; the ``for`` range is the bound.
2. A capability signature is executed at most ``max_repeats`` times in a
   session, so a model that keeps re-issuing the same request cannot spin.
   The allowance is two, not one: a legitimate second request is honoured and
   the third identical one stops the loop.
3. Any exception inside a segment ends the loop by returning, never by
   retrying.

The Milestone 3 seam
--------------------
Continuation is a replaceable policy and capability execution is a replaceable
callable. Milestone 3 can insert research, memory search, and specialists
behind the same seam without touching this file's control flow. Today the only
executor is the Prime request path that already existed.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.claims import unsupported_claim_error
from PRIMORA.aurora.citations import dangling_citation_error
from PRIMORA.aurora.conversation_guard import (
    reject_unrequested_lookup, latest_user_text, repeats_answer_after_correction,
    denies_assumption_after_correction,
)
from PRIMORA.aurora import request_shape
from PRIMORA.prime.context import BudgetConfig, ContextTooLargeError
from PRIMORA.prime.context.mind import Mind

# Loop runtime version. v0.2 was the first working loop: correct structure,
# but a two-cycle default that allowed at most one capability per turn, so
# genuine multi-step cognition was not reachable in practice. v0.3 raises the
# ceiling; the control flow is otherwise identical.
LOOP_VERSION = "0.3"

# How many cognition segments one turn may contain.
#
# v0.2 used 2, which was behaviour-preserving for the pre-loop code: a routing
# segment and a final segment. That made the loop structurally correct but
# practically useless for anything multi-step, because two segments permit at
# most one capability call.
#
# v0.3 uses 8, which is enough for a real chain - recall, then research, then
# a specialist, then a tool, then an answer - while remaining a hard ceiling.
#
# This is deliberately *not* the limiting safety mechanism. At 8 cycles a loop
# could attempt 7 capability executions, so the binding constraint is now
# DEFAULT_CAPABILITY_BUDGET, which is independent and set lower. The two
# bounds are separate on purpose: cycles bound how long Aurora may think,
# budget bounds how much she may do.
DEFAULT_MAX_CYCLES = 8

# A run that has taken longer than this stops continuing. Generous enough for
# a slow local model, tight enough that a hang cannot look like cognition.
DEFAULT_DEADLINE_S = 300.0

# How many times one identical request may be executed in a single session.
# Two is a deliberate allowance rather than a hard once-only rule: a model
# that legitimately needs the same computation twice should not be refused.
# The third identical request stops the loop, so a re-issuing model still
# cannot spin.
DEFAULT_MAX_REPEATS = 2

# How many capabilities one cognitive turn may execute in total. This is a
# different bound from ``max_cycles``: ``max_cycles`` limits generations, this
# limits side effects. The repeat guard stops one request repeating; this
# stops many different requests from accumulating.
#
# At the default ``max_cycles = 2`` a turn can execute at most one capability,
# so this budget is not the binding constraint yet. It becomes binding the
# moment ``max_cycles`` is raised, which is precisely when a total bound is
# needed and least remembered.
DEFAULT_CAPABILITY_BUDGET = 4

# Claims that some action has already been completed.
#
# Aurora's grounding covers not inventing facts, and it covers not inventing a
# result. Neither covers announcing an action she never took: told "remember
# this", she replied "I've recorded that preference" with no request behind it.
# That is a truthfulness failure, and prompt wording does not stop it - adding
# the instruction left the false-claim rate essentially unchanged.
#
# So the loop checks instead, because the loop is the only thing that knows
# whether anything actually ran.
#
# Capability-agnostic on purpose. The loop is not permitted to know which
# capabilities exist - that is Prime's catalogue and the Mind's boundary - so
# this asks the narrower question: did she assert that something was already
# done? If nothing succeeded this turn, any such claim is unbacked.
#
# Narrow matching, because a false correction is worse than a missed one: it
# would tell Aurora she is wrong when she is right. The object must not be the
# person, so "I've noted that you said Tuesday" - an acknowledgement of what
# was just said - is not a claim about an action she could have taken.
_COMPLETION_VERB = (
    r"learned|learnt|recorded|stored|saved|noted|logged|remembered|recalled"
    r"|looked up|found|converted|counted|consulted|checked"
    r"|submitted|filed|booked|ordered|sent|arranged|scheduled"
)

_COMPLETION_CLAIM = re.compile(
    r"\b(?:i|i've|i've got|i have|i'll|i will)\s+"
    r"(?:have\s+)?(?:"
    + _COMPLETION_VERB
    + r")\s+(?:that|it|this|the)\b(?!\s*you\b)",
    re.IGNORECASE,
)


def _claims_completed_action(text: str) -> bool:
    """Whether Aurora asserts that some action has already happened.

    True only for a clear first-person completion. Ambiguity returns False,
    which is the safe outcome: the caller does nothing rather than
    contradicting a statement that may well be true.
    """
    return _COMPLETION_CLAIM.search(text or "") is not None

# How much of each capability's output is kept as the turn's evidence log.
# The executor records every call without knowing which capabilities are
# external; deciding what counts as evidence is the consumer's job, so the
# loop stays free of any capability's name.
EVIDENCE_CHARS = 2000
EVIDENCE_MAX_ENTRIES = 24


def _fingerprint(text: str):
    """Content words of a capability result, for later provenance checks.

    A small, dependency-free reduction. The word list is intentionally short
    so the evidence stays a fingerprint rather than a copy of the output.
    """
    from PRIMORA.learning.provenance import content_tokens

    return content_tokens(text)


def _request_signature(action: str, payload: Any, context: Any,
                       metadata: Any) -> str:
    """Stable identity for a capability request, for the repeat guard."""
    try:
        blob = json.dumps(
            {"action": action, "payload": payload, "context": context,
             "metadata": metadata},
            sort_keys=True, default=str,
        )
    except Exception:
        blob = f"{action}|{payload!r}"
    return blob


def _payload_of(request: Any) -> tuple[Any, Any, Any]:
    return (getattr(request, "payload", None),
            getattr(request, "context", None),
            getattr(request, "metadata", None))


def _protected_evidence_tail(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the latest assistant-request/tool-result pair.

    Capability evidence is the input to the next cognition segment.  Keeping
    this small pair mandatory avoids treating an old transcript as equally
    important while preserving valid chat-message ordering.
    """
    last_user = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user = index
    tail = messages[last_user + 1:]
    if not tail:
        return []
    tool_index = next(
        (index for index in range(len(tail) - 1, -1, -1)
         if tail[index].get("role") == "tool"),
        None,
    )
    if tool_index is None:
        return [tail[-1]]
    start = tool_index
    if start > 0 and tail[start - 1].get("role") == "assistant":
        start -= 1
    return tail[start:tool_index + 1]


def _answer_result(
    result: Optional[BrainResult], extract: Callable[[str], Any], model: str
) -> Optional[BrainResult]:
    """Return a generation result only when it is an answer.

    A request block is an internal segment. When the loop stops because a
    cycle or repeat bound is reached, the last generation is still that
    request; returning it made the CLI print protocol markup as though it
    were the reply (and as though the request had answered the question).
    No request payload ever reaches the user: the bounded stop becomes an
    honest failure instead.
    """
    if result is None:
        return None
    content = getattr(result, "content", "") or ""
    if not content:
        return result
    try:
        internal = extract(content) is not None
    except Exception:
        internal = False
    if not internal:
        internal = (
            request_shape.is_request_attempt(content)
            or "<prime_request>" in content
        )
    if not internal:
        return result
    return BrainResult.failure(
        "Aurora emitted an internal capability request instead of an "
        "answer; the request payload was withheld.",
        model,
    )


@dataclass
class Segment:
    """One generation within a cognitive session. Runtime only."""

    index: int
    content: str
    requested: bool = False
    capability: Optional[str] = None
    executed: bool = False
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "requested": self.requested,
            "capability": self.capability,
            "executed": self.executed,
            "latency_s": round(self.latency_s, 3),
            "chars": len(self.content or ""),
        }


@dataclass
class LoopTrace:
    """What the loop did this session. Observability, not control."""

    segments: list[Segment] = field(default_factory=list)
    stop_reason: str = ""
    max_cycles: int = DEFAULT_MAX_CYCLES
    total_s: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "stop_reason": self.stop_reason,
            "max_cycles": self.max_cycles,
            "segment_count": self.count,
            "total_s": round(self.total_s, 3),
            "timings_ms": {
                key: round(value, 3) for key, value in self.timings_ms.items()
            },
        }


class CapabilityExecutor:
    """The Milestone 3 seam, as a callable.

    Today it forwards a capability request to Prime, which is exactly what the
    pre-loop code did inline. Milestone 3 replaces this object with one that
    can also run research, memory search, and specialists. The loop does not
    care: it only calls ``execute``.

    The executor is also where the per-session call budget is enforced. It is
    the single point every capability passes through, so recall, research,
    consult, learn, and any future tool are all bounded without each of them
    knowing the budget exists, and a new tool cannot forget to count itself.
    """

    def __init__(self, prime: Any, max_calls: int = DEFAULT_CAPABILITY_BUDGET) -> None:
        self._prime = prime
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        self.max_calls = int(max_calls)
        self.calls: list[tuple[str, Any]] = []
        self.evidence: list[tuple[str, Any]] = []
        self.timings_ms: dict[str, float] = {}
        self._evidence_session: str = ""
        self.exhausted = False

    def begin_session(self, session_id: str = "") -> None:
        """Reset the per-turn budget; roll the evidence over per session.

        The call budget is per turn. The evidence log is per *session*, so a
        fact gathered in one turn can still be recognised as external in a
        later one. A change of session drops the previous evidence, because
        provenance must not leak across conversations.

        What is kept is a content fingerprint, not the tool's output, so
        nothing unbounded is retained merely to support a later check.

        Citable sources are per *interaction*: a new one starts empty, so a
        number from an earlier, unrelated search can neither be read nor
        cited. The registry is Prime's; the executor already reads it for
        the citation check, and this is the per-turn reset seam. Guarded,
        so a replaceable prime without a registry still works.
        """
        self._used = 0
        self.exhausted = False
        self.timings_ms = {}
        # Actions that actually ran, as opposed to those merely attempted.
        # A refused attempt is not evidence that anything happened, so the
        # loop needs the difference before it can tell Aurora that a claim of
        # action is unbacked.
        self.succeeded: list[str] = []
        # A new interaction begins with no citable sources. Without this the
        # registry lived for the whole process, so a citation could reach a
        # page from a conversation that had nothing to do with the current
        # question, and research_read would happily fetch it again.
        sources = getattr(self._prime, "sources", None)
        reset = getattr(sources, "clear", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass
        if session_id and session_id != self._evidence_session:
            self._evidence_session = session_id
            self.evidence = []
        elif not session_id:
            self.evidence = []

    @property
    def used(self) -> int:
        return getattr(self, "_used", 0)

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.used)

    @property
    def citation_numbers(self) -> list[int]:
        """Source numbers currently citable, read from Prime's registry.

        An empty list when nothing has been searched, which makes every
        citation look dangling - the safe direction, because the correction
        then tells Aurora the truth instead of letting an invented number
        through. Read lazily so a registry wired after construction is
        still found.
        """
        sources = getattr(self._prime, "sources", None)
        if sources is None:
            return []
        try:
            return sources.numbers()
        except Exception:
            return []

    def execute(self, request: Any) -> str:
        """Run one capability and return the tool-turn content."""
        action = getattr(request, "action", "")
        payload, context, metadata = _payload_of(request)
        self.calls.append((action, payload))

        if self.used >= self.max_calls:
            # Refuse rather than execute. Reported as a normal failure so the
            # session can finish on what it already has instead of crashing.
            self.exhausted = True
            return json.dumps({
                "status": "failure",
                "output": None,
                "error": (
                    f"capability_budget_exhausted: this turn may execute at "
                    f"most {self.max_calls} capabilities, and that budget is "
                    f"spent. Answer with what you already know."
                ),
            })

        self._used = self.used + 1
        operation_started = time.perf_counter()
        try:
            result = self._prime.handle(request)
        finally:
            elapsed_ms = (time.perf_counter() - operation_started) * 1000
            self.timings_ms[f"capability_{action}_ms"] = (
                self.timings_ms.get(f"capability_{action}_ms", 0.0) + elapsed_ms
            )
        # Prime capabilities may expose finer-grained timings (for example
        # learn's Jev provider/parse/admission phases). Carry those samples to
        # the loop trace without making the loop depend on any capability's
        # schema.
        nested = getattr(result, "output", None)
        nested_timings = nested.get("timings_ms") if isinstance(nested, dict) else None
        if isinstance(nested_timings, dict):
            for key, value in nested_timings.items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                self.timings_ms[key] = self.timings_ms.get(key, 0.0) + number
        rendered = json.dumps(
            {
                "status": result.status.value,
                "output": result.output,
                "error": result.error,
            },
            default=str,
        )
        if result.status.value == "success":
            self.succeeded.append(action)
            # Only successful capability results can support a later learning
            # claim. Failed searches and rejected writes are not evidence.
            self.evidence.append((action, _fingerprint(rendered[:EVIDENCE_CHARS])))
            if len(self.evidence) > EVIDENCE_MAX_ENTRIES:
                del self.evidence[:-EVIDENCE_MAX_ENTRIES]
        return rendered


class NoCapabilityExecutor(CapabilityExecutor):
    """Refuses every capability. Useful for a pure-thinking session."""

    def __init__(self) -> None:
        super().__init__(prime=None)
        self.max_calls = 1

    def execute(self, request: Any) -> str:
        self.calls.append((getattr(request, "action", ""), None))
        return json.dumps(
            {"status": "failure", "output": None,
             "error": "no capability executor available"},
            default=str,
        )


class CognitiveLoop:
    """Runs one logical cognitive session over the existing Aurora brain.

    Holds no per-turn state, so one instance can be reused across turns. All
    session state lives in :meth:`run` locals or in the caller's Mind.
    """

    def __init__(
        self,
        mind: Mind,
        brain: Brain,
        *,
        executor: Optional[CapabilityExecutor] = None,
        max_cycles: int = DEFAULT_MAX_CYCLES,
        max_new_tokens: int = 256,
        budget: Optional[BudgetConfig] = None,
        token_counter: Optional[Callable[[list[dict]], int]] = None,
        deadline_s: float = DEFAULT_DEADLINE_S,
        max_repeats: int = DEFAULT_MAX_REPEATS,
        request_extractor: Optional[Callable[[str], Any]] = None,
        enable_memory_interpreter: Optional[bool] = None,
    ) -> None:
        if max_cycles < 1:
            raise ValueError("max_cycles must be at least 1")
        if max_repeats < 1:
            raise ValueError("max_repeats must be at least 1")
        self._mind = mind
        self._brain = brain
        self._executor = executor
        self.max_cycles = int(max_cycles)
        self.max_new_tokens = int(max_new_tokens)
        self._budget = budget
        self._token_counter = token_counter
        self.deadline_s = float(deadline_s)
        # How many times one identical request may be executed in a session.
        self.max_repeats = int(max_repeats)
        # Injected by Aurora so this module never imports aurora at import
        # time. Resolved once, not per segment.
        self._request_extractor = request_extractor or _default_extractor()
        self._last_trace: Optional[LoopTrace] = None
        # A pre-turn interpreter is retained only as an explicit compatibility
        # seam for older embedders. Aurora's production composition leaves it
        # disabled: the same Qwen generation that answers a turn decides
        # whether to request Prime recall, so ordinary turns have no hidden
        # semantic router or compulsory planning generation.
        self._enable_memory_interpreter = bool(enable_memory_interpreter)

    # -- accessors --------------------------------------------------------

    @property
    def mind(self) -> Mind:
        """The one Mind this loop uses for every segment."""
        return self._mind

    @property
    def executor(self) -> Optional[CapabilityExecutor]:
        return self._executor

    @executor.setter
    def executor(self, value: Optional[CapabilityExecutor]) -> None:
        self._executor = value

    @property
    def last_trace(self) -> Optional[LoopTrace]:
        return self._last_trace

    def _counter(self) -> Callable[[list[dict]], int]:
        if self._token_counter is not None:
            return self._token_counter
        brain = self._brain

        def count(messages: list[dict]) -> int:
            return brain.count_tokens(messages)

        return count

    def _record_formation_timings(self, trace: LoopTrace) -> None:
        """Move Prime's latest formation sample into this turn's trace."""
        consume_formation = getattr(
            getattr(self._mind, "mechanisms", None),
            "consume_formation_timings", None,
        )
        if not callable(consume_formation):
            return
        try:
            formation_timings = consume_formation()
        except Exception:
            formation_timings = {}
        if not formation_timings:
            return
        for key, value in formation_timings.items():
            name = "formation_ms" if key == "total_ms" else f"formation_{key}"
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            trace.timings_ms[name] = trace.timings_ms.get(name, 0.0) + number

    # -- the loop ---------------------------------------------------------

    def run(
        self,
        instructed: list[dict[str, str]],
        session_id: str = "default",
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> BrainResult:
        """Cognate until finished, then return one logical response.

        ``instructed`` is the message list with the system instruction
        already applied. The returned ``BrainResult`` is exactly what the
        pre-loop code returned, so callers cannot tell the difference.
        """
        extract = self._request_extractor
        trace = LoopTrace(max_cycles=self.max_cycles)
        started = time.perf_counter()
        trace.timings_ms.update({
            "planning_ms": 0.0,
            "response_generation_ms": 0.0,
            "formation_ms": 0.0,
        })
        self._last_trace = trace
        # The capability budget is per turn. The executor owns the counter and
        # resets itself; the loop does not track it. Guarded, because any
        # callable is a valid executor and the seam must stay replaceable.
        begin = getattr(self._executor, "begin_session", None)
        if callable(begin):
            try:
                begin(session_id)
            except TypeError:
                # A replaceable executor may accept no argument at all.
                begin()
        bind_session = getattr(self._mind, "bind_execution_session", None)
        if callable(bind_session):
            try:
                bind_session(session_id)
            except Exception:
                pass

        # The same Mind, and therefore the same task-local Working Memory,
        # for the whole session. Never rebuilt per segment.
        messages = list(instructed)
        seen: dict[str, int] = {}
        # Signatures that already ran successfully this turn. A repeat after
        # success is redundant and is refused; a repeat after failure is a
        # legitimate retry and still passes through ``seen``.
        result: Optional[BrainResult] = None
        # A rejected capability has to be given one more chance.
        #
        # Aurora is told what was wrong in the tool result, and the next thing
        # she does is usually say so - "let me correct that" - without yet
        # emitting the corrected request. Stopping on her first reply after a
        # failure threw that away: the correction was delivered and never used.
        # Measured on the real brain, every rejected `count_words` request ended
        # the session there, with Aurora explicitly saying she would fix it.
        #
        # One extra cycle per failure. The repeat guard and the budget are
        # untouched: this grants a generation, never a call, and an executed
        # request still passes the same budget and repeat checks as any other.
        failure_pending = False
        recovery_granted = False
        # Separate from recovery_granted: a rejected capability and an unbacked
        # claim are different faults, and each deserves its own correction. One
        # shared flag meant a failure recovery silently suppressed the claim
        # check that same turn, which is exactly the case where Aurora is most
        # likely to assert something that never happened.
        claim_granted = False
        # Separate again: a citation that points at a source number which
        # does not exist is a third, independent fault. It gets its own
        # correction so neither a failure recovery nor a claim correction
        # can silently swallow it, and so a genuine citation problem is
        # never mistaken for either.
        citation_granted = False
        # Fourth, and deliberately last so it cannot pre-empt the
        # corrections above: a JSON reply that names a capability request
        # but does not parse as one is a protocol failure, not an answer.
        # A wrapper-less request object now executes through the normal
        # path, so a JSON reply that still parses as nothing is Aurora
        # asking for something and nothing having run - which must never
        # reach the user as though it were an answer, or a recall that
        # succeeded.
        protocol_granted = False
        # Fifth, independent of capabilities: a claimed memory update is
        # checked against what was actually stored. Persistence happens in
        # the Mind and can silently refuse or fail, so the loop will not
        # hand the user a success that was never written.
        update_granted = False
        stale_answer_granted = False
        accountability_granted = False
        user_text = latest_user_text(instructed)

        # An explicitly enabled legacy embedding may still use the validated
        # JSON planning seam. Production Aurora never enables it; her normal
        # generation requests Prime capabilities directly.
        plan = None
        planning_started = time.perf_counter()
        if self._enable_memory_interpreter:
            try:
                from PRIMORA.aurora import memory_coordinator
                prior_dialogue = tuple({"role": str(m.get("role")),
                                        "content": str(m.get("content", ""))[:700]}
                                       for m in instructed[:-1]
                                       if m.get("role") in ("user", "assistant"))
                plan = memory_coordinator.interpret(
                    self._brain, user_text, prior_dialogue)
            except Exception:
                plan = None  # Ordinary cognition if the live planner fails.
        trace.timings_ms["planning_ms"] = (
            time.perf_counter() - planning_started
        ) * 1000
        if (self._enable_memory_interpreter
                and isinstance(plan, memory_coordinator.MemoryPlan)
                and plan.operation == "recall"):
            if plan.clarify:
                response = BrainResult.success(
                    "Which specific memory or experience do you mean?",
                    self._brain.get_model_name())
            else:
                try:
                    evidence = self._mind.memory_query_evidence(
                        {
                            **plan.as_dict(),
                            # Contextual follow-ups keep their literal
                            # wording for Aurora's answer but use a bounded,
                            # predicate-bearing query for typed retrieval.
                            "query": plan.retrieval_query or user_text,
                        },
                        session_id=session_id)
                except Exception:
                    response = BrainResult.success(
                        "I couldn't safely retrieve verified memory evidence "
                        "right now. I don't want to guess.",
                        self._brain.get_model_name())
                else:
                    answer_started = time.perf_counter()
                    response = memory_coordinator.grounded_answer(
                        self._brain, user_text, plan, evidence)
                    trace.timings_ms["response_generation_ms"] += (
                        time.perf_counter() - answer_started
                    ) * 1000
            trace.stop_reason = "finished"
            self._persist_turn(messages, 0, response, trace, session_id)
            self._record_formation_timings(trace)
            trace.total_s = time.perf_counter() - started
            trace.timings_ms["total_ms"] = trace.total_s * 1000
            if on_delta is not None:
                on_delta(response.content)
            return response

        # Length of `messages` as of the last successful build. The
        # end-of-turn persist compares against it to see rows a correction
        # cycle appended after that build (their own build never ran when
        # the loop stops on a deadline).
        built_len = 0

        for cycle in range(self.max_cycles):
            if time.perf_counter() - started > self.deadline_s:
                trace.stop_reason = "deadline"
                break

            try:
                context = self._mind.build(
                    messages,
                    session_id,
                    token_counter=self._counter(),
                    max_new_tokens=self.max_new_tokens,
                    budget=self._budget,
                    protected_messages=_protected_evidence_tail(messages),
                )
            except ContextTooLargeError as exc:
                trace.stop_reason = "context_too_large"
                if result is not None:
                    return _answer_result(
                        result, extract, self._brain.get_model_name())
                return BrainResult.failure(
                    str(exc), self._brain.get_model_name()
                )
            except Exception as exc:
                # A broken segment must end the session, never retry.
                trace.stop_reason = f"build_failed: {type(exc).__name__}"
                if result is not None:
                    return _answer_result(
                        result, extract, self._brain.get_model_name())
                return BrainResult.failure(
                    f"context_build_failed: {exc}",
                    self._brain.get_model_name(),
                )

            # Formation is owned by Prime's ContextManager. Consume its
            # latest sample once so repeated cognitive segments do not count
            # the same provider call twice.
            self._record_formation_timings(trace)

            built_len = len(messages)

            seg_started = time.perf_counter()
            try:
                generation_kwargs = {
                    "max_output_tokens": context.allowed_output,
                }
                # Keep the optional streaming keyword out of the call when no
                # callback is requested. This preserves compatibility with
                # existing lightweight Brain implementations while retaining
                # streaming for production brains that support it.
                if on_delta is not None:
                    generation_kwargs["on_delta"] = on_delta
                try:
                    generated = self._brain.generate(
                        context.messages, **generation_kwargs)
                except TypeError as exc:
                    if on_delta is None or "on_delta" not in str(exc):
                        raise
                    generation_kwargs.pop("on_delta", None)
                    generated = self._brain.generate(
                        context.messages, **generation_kwargs)
            except Exception as exc:
                trace.stop_reason = f"generate_failed: {type(exc).__name__}"
                if result is not None:
                    return _answer_result(
                        result, extract, self._brain.get_model_name())
                return BrainResult.failure(
                    f"generation_failed: {exc}", self._brain.get_model_name()
                )
            segment = Segment(
                index=cycle,
                content=generated.content or "",
                latency_s=time.perf_counter() - seg_started,
            )
            trace.segments.append(segment)
            trace.timings_ms["response_generation_ms"] += segment.latency_s * 1000
            result = generated

            if not generated.success:
                trace.stop_reason = "generation_failed"
                break

            request = extract(generated.content or "")
            if request is None:
                # No single request was parsed. Distinguish "the model asked for
                # nothing" from "the model asked for more than one thing",
                # because the second is a protocol failure that must be
                # visible: a request the model really did make is being
                # discarded, and reporting that as a finished answer hides it.
                content = generated.content or ""
                if content.count("<prime_request>") > 1:
                    segment.requested = False
                    trace.stop_reason = "multiple_blocks"
                    break
                # Exactly one block was emitted and it still did not parse,
                # so Aurora asked for something and nothing ran. Finishing
                # here would report the turn as a normal answer and hide
                # that, the same way reporting a discarded request as
                # finished hid it. No corrective turn: the request was
                # already rejected by the extractor after repair, and a
                # second generation would spend the budget retrying a shape
                # the model cannot produce.
                if content.count("<prime_request>") == 1:
                    segment.requested = False
                    trace.stop_reason = "protocol_error"
                    break
                # A rejected capability was just explained to her and she has
                # not had a turn to act on it. Give her exactly one, then take
                # this segment as the answer if she still asks for nothing.
                #
                # The loop deliberately does not consult the executor's budget
                # to make this decision: the executor owns the budget, and this
                # grants a generation, never a call. If she does emit a
                # corrected request it still passes the budget and the repeat
                # guard exactly as any other request does, and max_cycles
                # bounds the session either way.
                if failure_pending and not recovery_granted:
                    recovery_granted = True
                    failure_pending = False
                    continue
                # She has closed the turn by claiming an action completed. If
                # that action never ran, the claim is false and she is about to
                # hand it to Amir as settled fact. Correct it the same way a
                # failed capability is corrected - with the truth, in a tool
                # turn, and one more chance to act on it.
                #
                # State-aware by necessity: only the loop knows whether
                # anything executed, and the executor records success
                # separately from attempts because a refused request is not
                # evidence that anything happened.
                claimed = _claims_completed_action(content)
                succeeded = getattr(self._executor, "succeeded", [])
                # A research-shaped claim ("I've searched the web") is
                # backed only by a research request actually succeeding, so
                # an unrelated capability success must not silence the check
                # and a genuine success must not be contradicted. Deciding
                # that needs the capability's name, which this module's
                # scope tests forbid it from holding, so the decision lives
                # in PRIMORA.aurora.claims and is called from here.
                unsupported = unsupported_claim_error(content, succeeded)
                if unsupported is not None and not claim_granted:
                    claim_granted = True
                    messages = messages + [
                        {"role": "assistant", "content": content},
                        {"role": "tool", "content": json.dumps({
                            "status": "failure", "output": None,
                            "error": unsupported,
                        })},
                    ]
                    continue
                if (claimed
                        and not succeeded
                        and not claim_granted):
                    claim_granted = True
                    messages = messages + [
                        {"role": "assistant", "content": content},
                        {"role": "tool", "content": json.dumps({
                            "status": "failure", "output": None,
                            "error": (
                                "unbacked_claim: you stated that something had "
                                "already been done. No capability request has "
                                "run, so it did not happen. If it should "
                                "have, request it now; if not, correct "
                                "yourself."
                            ),
                        })},
                    ]
                    continue
                # A claimed memory update ("your favorite color has been
                # updated to violet") is a claim about a write. The Mind
                # owns the stored value and the loop owns the answer path,
                # so the Mind is asked whether the claim is backed. One
                # correction turn, then the claim is never printed as a
                # success it was not.
                update_error = None
                verifier = getattr(
                    self._mind, "preference_update_claim_error", None)
                if verifier is not None:
                    try:
                        update_error = verifier(content)
                    except Exception:
                        update_error = None
                if update_error is not None:
                    if update_granted:
                        segment.requested = False
                        trace.stop_reason = "unbacked_memory_change"
                        return BrainResult.failure(
                            "Aurora claimed a memory update that was not "
                            "saved, then repeated the claim after being "
                            "corrected.",
                            self._brain.get_model_name())
                    update_granted = True
                    messages = messages + [
                        {"role": "assistant", "content": content},
                        {"role": "tool", "content": json.dumps({
                            "status": "failure", "output": None,
                            "error": update_error,
                        })},
                    ]
                    continue
                # A citation that points at a source number no search in this
                # process ever produced is corrected the same way: a tool
                # turn naming the real numbers, and one chance to fix it.
                # Citations are never required - only impossible ones are
                # corrected - and the decision needs the numbers, which only
                # the executor can read, so it lives in PRIMORA.aurora.
                # citations and is called from here.
                available = getattr(self._executor, "citation_numbers", [])
                dangling = dangling_citation_error(content, available)
                if dangling is not None and not citation_granted:
                    citation_granted = True
                    messages = messages + [
                        {"role": "assistant", "content": content},
                        {"role": "tool", "content": json.dumps({
                            "status": "failure", "output": None,
                            "error": dangling,
                        })},
                    ]
                    continue
                # A reply shaped like a capability request that could not
                # be run. Emitting nothing was what let raw JSON reach
                # the user as an answer: the trace said "finished", no
                # capability ran, and the JSON string looked like a recall
                # that had. Correct it the same way a rejected capability
                # is corrected - one tool turn naming what was wrong, and
                # no second attempt - so she can say so honestly instead.
                if request_shape.is_request_attempt(content):
                    if protocol_granted:
                        # The correction was given and the same shape came
                        # back. Stop rather than print it as an answer.
                        segment.requested = False
                        trace.stop_reason = "protocol_error"
                        return BrainResult.failure(
                            "Aurora emitted a capability request that could "
                            "not be executed, then did not answer after "
                            "being told it did not run.",
                            self._brain.get_model_name())
                    protocol_granted = True
                    messages = messages + [
                        {"role": "assistant", "content": content},
                        {"role": "tool", "content": json.dumps({
                            "status": "failure", "output": None,
                            "error": (
                                "not_a_request: this was a JSON object "
                                "naming a capability request, so it was not "
                                "executed and nothing was answered. A "
                                "request must be exactly one request block "
                                "containing one JSON object, or the same "
                                "object alone as your whole reply, with a "
                                "registered action and a payload object. "
                                "Nothing ran. Emit one request, or answer "
                                "the question directly."
                            ),
                        })},
                    ]
                    continue
                # A verbatim replay after an explicit correction is not an
                # answer to that correction. Give one bounded retry, then
                # fail rather than delivering the same stale answer again.
                if repeats_answer_after_correction(instructed, content):
                    if stale_answer_granted or cycle >= self.max_cycles - 1:
                        trace.stop_reason = "stale_answer"
                        return BrainResult.failure(
                            "Aurora repeated a previous answer instead of "
                            "addressing the user's correction.",
                            self._brain.get_model_name())
                    stale_answer_granted = True
                    messages = messages + [
                        {"role": "assistant", "content": content},
                        {"role": "tool", "content": json.dumps({
                            "status": "failure", "output": None,
                            "error": (
                                "stale_reply: your answer repeated the previous "
                                "reply verbatim after a user correction. "
                                "Address the latest user message directly; "
                                "do not reuse the prior tool response."
                            ),
                        })},
                    ]
                    continue
                # When the user explicitly points out our invented
                # assumptions, denying that any assumption was made is
                # misleading. Require a bounded correction, not a canned
                # apology or a silent rewrite of the user's stored facts.
                if denies_assumption_after_correction(instructed, content):
                    if accountability_granted or cycle >= self.max_cycles - 1:
                        trace.stop_reason = "unacknowledged_assumption"
                        return BrainResult.failure(
                            "Aurora denied an assumption the user explicitly "
                            "challenged instead of addressing the correction.",
                            self._brain.get_model_name())
                    accountability_granted = True
                    messages = messages + [
                        {"role": "assistant", "content": content},
                        {"role": "tool", "content": json.dumps({
                            "status": "failure", "output": None,
                            "error": (
                                "accountability: the user identified an "
                                "unsupported assumption. Do not deny having "
                                "made it. Acknowledge the unsupported claim "
                                "and stick to the user's exact words."
                            ),
                        })},
                    ]
                    continue
                # Nothing further was asked for. This segment is the answer.
                segment.requested = False
                trace.stop_reason = "finished"
                break
            segment.requested = True
            segment.capability = getattr(request, "action", None)

            if self._executor is None:
                trace.stop_reason = "no_executor"
                break
            if cycle >= self.max_cycles - 1:
                # Cycles exhausted. The last segment is the response, exactly
                # as the pre-loop code returned it.
                trace.stop_reason = "max_cycles"
                break

            # An unambiguous personal preference is not a request for an
            # external explanation. Policy is separate from this structural
            # loop, so other capabilities and actual research questions run.
            policy_error = reject_unrequested_lookup(request, user_text)
            if policy_error is not None:
                messages = messages + [
                    {"role": "assistant", "content": generated.content},
                    {"role": "tool", "content": json.dumps({
                        "status": "failure", "output": None,
                        "error": policy_error,
                    })},
                ]
                continue
            payload, context_data, metadata = _payload_of(request)
            signature = _request_signature(
                getattr(request, "action", ""), payload, context_data, metadata
            )
            # A signature may be executed at most `self.max_repeats` times per
            # session. Once the allowance is spent, a further identical request
            # stops the loop rather than repeating forever.
            if seen.get(signature, 0) >= self.max_repeats:
                trace.stop_reason = "repeated_request"
                break
            seen[signature] = seen.get(signature, 0) + 1

            try:
                tool_content = self._executor.execute(request)
            except Exception as exc:
                tool_content = json.dumps(
                    {"status": "failure", "output": None,
                     "error": f"capability_failed: {exc}"},
                    default=str,
                )
            segment.executed = True

            # Whether this result told Aurora she was wrong. A successful
            # result is something to answer from, so a reply with no request
            # after one is genuinely the answer. A failure is an instruction
            # she has not acted on yet, which is the case that gets the extra
            # cycle.
            failure_pending = '"status": "failure"' in tool_content
            recovery_granted = False

            # Internal segments continue as context, never as separate
            # assistant turns. Reconciliation semantics are unchanged: the
            # same snapshot shape the pre-loop code passed is passed here.
            messages = messages + [
                {"role": "assistant", "content": generated.content},
                {"role": "tool", "content": tool_content},
            ]
        else:
            trace.stop_reason = trace.stop_reason or "max_cycles"

        executor_timings = getattr(self._executor, "timings_ms", {})
        if isinstance(executor_timings, dict):
            for key, value in executor_timings.items():
                trace.timings_ms[key] = trace.timings_ms.get(key, 0.0) + float(value)
        # The turn is over: write its closing state before returning so a
        # caller that exits right after respond() cannot lose it.
        self._persist_turn(messages, built_len, result, trace, session_id)
        # The closing persistence may form the just-finished user message, so
        # consume that sample after persistence as well as the samples from
        # cognitive segments above.
        self._record_formation_timings(trace)
        trace.total_s = time.perf_counter() - started
        trace.timings_ms["total_ms"] = trace.total_s * 1000
        if result is None:
            # Unreachable while max_cycles >= 1, kept so the return type is
            # total rather than Optional.
            return BrainResult.failure(
                "cognitive_loop_produced_no_segment",
                self._brain.get_model_name(),
            )
        return _answer_result(result, extract, self._brain.get_model_name())

    def _persist_turn(
        self,
        messages: list[dict[str, str]],
        built_len: int,
        result: Optional[BrainResult],
        trace: LoopTrace,
        session_id: str,
    ) -> None:
        """Write the turn's closing state before ``run`` returns.

        A finished answer joins the exact snapshot the last build already
        reconciled, so every existing row stays a fingerprint no-op while
        the reply gains its row immediately: the previous behaviour of
        waiting for the *next* turn's payload lost the last reply whenever
        the process exited first. Rows a correction cycle appended after
        its own build (the deadline case) are written by the same call,
        with the same snapshot shape ``build`` would have passed. Only
        ``finished`` appends content; protocol- and failure-shaped stops
        persist nothing they would not otherwise. Never raises:
        persistence stays best-effort exactly like the build path.
        """
        payload = messages
        if (
            trace.stop_reason == "finished"
            and result is not None
            and getattr(result, "success", False)
            and getattr(result, "content", None)
        ):
            content = str(result.content)
            last = messages[-1] if messages else None
            already_last = (
                last is not None
                and last.get("role") == "assistant"
                and str(last.get("content", "")) == content
            )
            if not already_last:
                payload = messages + [
                    {"role": "assistant", "content": content}
                ]
        if payload is messages and len(messages) == built_len:
            return
        try:
            self._mind.reconcile(payload, session_id)
        except Exception:
            pass

    def run_segments(
        self,
        instructed: list[dict[str, str]],
        session_id: str = "default",
    ) -> list[Segment]:
        """Run a session and return only the segment records.

        Convenience for tests and diagnostics. The response itself is
        discarded, so this must not be used where the answer matters.
        """
        self.run(instructed, session_id)
        trace = self._last_trace
        return list(trace.segments) if trace else []


# Resolved on first use, not at import time, so this module has no import-time
# dependency on PRIMORA.aurora.aurora. Aurora injects the extractor explicitly.
_DEFAULT_EXTRACTOR = None


def _default_extractor() -> Callable[[str], Any]:
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        from PRIMORA.aurora.aurora import extract_prime_request

        _DEFAULT_EXTRACTOR = extract_prime_request
    return _DEFAULT_EXTRACTOR
