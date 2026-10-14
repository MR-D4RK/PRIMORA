"""The Jev seam: turning "this is worth remembering" into structured memory.

Milestone 3.

What this is
------------
Jev sits between Aurora's judgement and the memory lifecycle:

    Aurora decides WHAT is worth learning      -> natural language + provenance
    Jev decides HOW to structure and evaluate -> structured candidates
    Memory lifecycle decides admission         -> existing mechanisms only

Aurora never sees or writes the structured form. She says a sentence and
states where it came from. The concept, value, memory type, and scope are
Jev's job, and this module is the only place that decides them.

Provenance, and the line it draws
---------------------------------
A learning request says where its content came from:

    user          Amir said it
    research      an external source said it
    specialist    a consulted specialist said it
    observation   Aurora witnessed it directly
    inference     Aurora concluded it
    unknown       she cannot say

The distinction is load-bearing. An inference is not a weaker fact, it is a
different kind of claim, and the policy in :data:`HOLDS` refuses to let one
become a fact by accident. Provenance maps onto the existing ``authority``
field on state rows, so it is not a new parallel notion of trust: it reuses
the one the memory system already had.

Nothing here writes to memory. Jev only ever returns a decision.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from PRIMORA.prime.contract import RequestStatus

# The provenance vocabulary Aurora may use.
USER = "user"
RESEARCH = "research"
SPECIALIST = "specialist"
OBSERVATION = "observation"
INFERENCE = "inference"
UNKNOWN = "unknown"
TOOL = "tool"

PROVENANCE = (USER, RESEARCH, SPECIALIST, TOOL, OBSERVATION, INFERENCE, UNKNOWN)

# Only what Amir himself said may become a current fact. Everything else is
# held as a pending candidate.
#
# This is not a local invention: ``ContextStore`` already refuses to make a
# non-user-authored state, entity, or alias current, at the schema level.
# Learning could not weaken that without rewriting a safety property of the
# memory system, so it honours it instead. An inference, a research result, a
# specialist's answer, a tool's output, and Aurora's own observation are all
# recorded and all wait; none of them is silently promoted to fact.
AUTO_ADMIT = frozenset({USER})

# Every provenance that is never auto-admitted.
HOLDS = frozenset(p for p in PROVENANCE if p not in AUTO_ADMIT)

# Evidential support, following the convention already used by state rows:
# explicit user statements 1.0, anything less direct lower. This is support,
# not truth probability, and it never overrides lifecycle status.
CONFIDENCE = {
    USER: 1.0,
    RESEARCH: 0.6,
    SPECIALIST: 0.6,
    TOOL: 0.6,
    OBSERVATION: 0.6,
    INFERENCE: 0.3,
    UNKNOWN: 0.3,
}

# Bounds. Learning is not an essay.
MAX_TEXT_CHARS = 2000
MAX_CANDIDATES = 4

MEMORY_TYPES = ("FACT", "PREFERENCE", "DECISION", "INSTRUCTION", "RATIONALE")


@dataclass(frozen=True)
class LearningRequest:
    """Aurora's request, in her own words."""

    text: str
    provenance: str
    session_id: str = "default"

    def validate(self) -> Optional[str]:
        """Return a reason string if the request is unusable, else None."""
        if not isinstance(self.text, str) or not self.text.strip():
            return "Missing or invalid 'text'"
        if len(self.text) > MAX_TEXT_CHARS:
            return f"'text' is too long (max {MAX_TEXT_CHARS} characters)"
        if self.provenance not in PROVENANCE:
            return (
                f"Unknown provenance: {self.provenance!r}. "
                f"Use one of: {', '.join(PROVENANCE)}"
            )
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            return "Invalid session"
        return None


@dataclass(frozen=True)
class StructuredCandidate:
    """One structured memory proposal, as Jev read it.

    Frozen and deliberately small. There is nowhere to put a truth claim: the
    lifecycle decides admission, not this object.
    """

    concept: str
    value: str = ""
    memory_type: str = ""
    scope: str = "session"
    rationale: str = ""
    confidence: float = 1.0

    def is_wellformed(self) -> bool:
        return bool(self.concept.strip()) and (
            bool(self.value.strip()) or self.memory_type != "FACT"
        )


@dataclass(frozen=True)
class LearningOutcome:
    """Jev's decision. No memory has been touched when this is returned."""

    disposition: str  # "confirm" | "hold" | "reject"
    candidates: list[StructuredCandidate] = field(default_factory=list)
    reason: str = ""
    diagnostics: list[str] = field(default_factory=list)

    @classmethod
    def confirmed(cls, candidates, reason="", diagnostics=None):
        return cls("confirm", list(candidates), reason, list(diagnostics or []))

    @classmethod
    def held(cls, candidates, reason, diagnostics=None):
        return cls("hold", list(candidates), reason, list(diagnostics or []))

    @classmethod
    def rejected(cls, reason, diagnostics=None):
        return cls("reject", [], reason, list(diagnostics or []))


class Jev(ABC):
    """Turns a natural-language learning request into structured candidates.

    The seam. Implement it to change how learning is understood without
    changing the capability, the lifecycle, or Aurora's interface.
    """

    name = "jev"

    @abstractmethod
    def evaluate(self, request: LearningRequest) -> LearningOutcome:
        """Return a decision. Must not write to memory."""


class UnconfiguredJev(Jev):
    """The default: no Jev wired up.

    Fails closed. An unconfigured learning path must not invent memory
    structure, so it says so instead of guessing.
    """

    name = "unconfigured-jev"

    def evaluate(self, request: LearningRequest) -> LearningOutcome:
        raise RuntimeError(
            "No Jev is configured. Pass jev=... to Prime, for example "
            "ModelJev(provider)."
        )


class ModelJev(Jev):
    """A Jev backed by a model, through the existing Provider abstraction.

    Reuses ``PRIMORA.prime.providers.Provider``, so the same backend serves
    Aurora, her specialists, and Jev, and any of them can be replaced without
    touching the others.

    The model is asked for strict JSON and the reply is validated field by
    field. A malformed, truncated, or unexpected reply is *rejected*, never
    partially believed: the worst case is a failed learn, not a corrupted
    memory. This is the one place where a model's output is allowed to shape
    what memory contains, so it is deliberately the most defensive code here.
    """

    name = "model-jev"

    SYSTEM_PROMPT = (
        "You convert one statement into structured memory proposals. "
        "Reply with JSON only, no prose and no code fence, shaped as:\n"
        '{"candidates": [{"concept": "...", "value": "...",\n'
        '  "memory_type": "FACT|PREFERENCE|DECISION|INSTRUCTION|RATIONALE",\n'
        '  "scope": "session|global"}]}\n'
        "concept is a short lowercase noun phrase naming what is known. "
        "value is the concrete content. scope is 'global' only when the "
        "statement is true for every conversation. "
        'Return {"candidates": []} when the statement is not something worth '
        "remembering, such as a greeting or a question."
    )

    def __init__(self, provider: Any) -> None:
        if provider is None:
            raise ValueError("ModelJev requires a Provider")
        self._provider = provider

    def evaluate(self, request: LearningRequest) -> LearningOutcome:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": request.text},
        ]
        try:
            result = self._provider.invoke("jev", {"messages": messages})
        except Exception as exc:
            return LearningOutcome.rejected(f"jev_failed: {exc}")
        # A provider that reports a failure never yields memory structure.
        if result is None:
            return LearningOutcome.rejected("jev_failed: no result")
        if result.status is not RequestStatus.SUCCESS:
            return LearningOutcome.rejected(
                f"jev_failed: {getattr(result, 'error', None) or 'provider error'}"
            )
        output = getattr(result, "output", None)
        text = output.get("content") if isinstance(output, dict) else output
        if not isinstance(text, str) or not text.strip():
            return LearningOutcome.rejected("jev_returned_nothing")

        candidates, diagnostics = self._parse(
            text, CONFIDENCE.get(request.provenance, 0.3))
        if not candidates:
            return LearningOutcome.rejected(
                "jev_found_nothing_structurable", diagnostics)
        if request.provenance in HOLDS:
            return LearningOutcome.held(
                candidates,
                f"provenance={request.provenance} is never auto-admitted",
                diagnostics,
            )
        return LearningOutcome.confirmed(
            candidates, f"provenance={request.provenance}", diagnostics)

    def _parse(
        self, text: str, confidence: float
    ) -> tuple[list[StructuredCandidate], list[str]]:
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            _, _, raw = raw.partition("\n")
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return [], ["jev_parse_failed: no JSON object"]
        try:
            payload = json.loads(raw[start:end + 1])
        except json.JSONDecodeError as exc:
            return [], [f"jev_parse_failed: {exc}"]
        if not isinstance(payload, dict):
            return [], ["jev_parse_failed: not an object"]
        items = payload.get("candidates")
        if not isinstance(items, list):
            return [], ["jev_parse_failed: 'candidates' is not a list"]

        out: list[StructuredCandidate] = []
        diagnostics: list[str] = []
        for item in items[:MAX_CANDIDATES]:
            if not isinstance(item, dict):
                diagnostics.append("jev_parse_failed: entry is not an object")
                continue
            concept = item.get("concept")
            value = item.get("value", "")
            if not isinstance(concept, str) or not concept.strip():
                diagnostics.append("jev_parse_failed: bad concept")
                continue
            if not isinstance(value, str):
                diagnostics.append("jev_parse_failed: bad value")
                continue
            memory_type = item.get("memory_type", "")
            if memory_type not in MEMORY_TYPES:
                diagnostics.append(
                    f"jev_normalized: memory_type {memory_type!r} not known")
                memory_type = ""
            scope = item.get("scope", "session")
            if scope != "global":
                scope = "session"
            candidate = StructuredCandidate(
                concept=concept.strip().lower(),
                value=value.strip(),
                memory_type=memory_type,
                scope=scope,
                confidence=confidence,
            )
            if not candidate.is_wellformed():
                diagnostics.append("jev_rejected: entry has no concrete value")
                continue
            out.append(candidate)
        return out, diagnostics
