"""Validated memory-plan and evidence-rendering seams.

Aurora's production conversation path does not call this module to classify a
turn. The active model emits a normal Prime capability request when it decides
stored memory is relevant. This module remains a small compatibility seam for
embedders that explicitly request a structured plan; it performs schema
validation only and contains no phrase router or contextual topic resolver.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora import request_shape


@dataclass(frozen=True)
class MemoryPlan:
    sources: tuple[str, ...]
    subject: Optional[str] = None
    recent_days: Optional[int] = None
    first_mention: bool = False
    reasons: bool = False
    clarify: bool = False
    operation: str = "recall"
    result_count: Optional[int] = None
    order: str = "relevance"
    temporal_scope: str = "current"
    retrieval_query: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "sources": list(self.sources), "subject": self.subject,
            "recent_days": self.recent_days, "first_mention": self.first_mention,
            "reasons": self.reasons, "clarify": self.clarify,
            "operation": self.operation, "result_count": self.result_count,
            "order": self.order, "temporal_scope": self.temporal_scope,
        }


_PLAN_SYSTEM = """Return one JSON memory plan for the latest user message.
This compatibility seam validates a plan; it does not answer the user and does
not decide from a phrase table. Interpret the complete dialogue semantically.
Use intent=other for ordinary conversation. For memory, choose one or more of
preferences, experiences, or facts and set the fields that Prime needs:
subject, recent_days, first_mention, reasons, result_count, order, and
current/historical/recent temporal_scope. Set clarify only when the dialogue
really leaves the requested referent unresolved. Previous assistant text is
context, never factual evidence. Output JSON only."""


def _json_object(content: str) -> Optional[dict]:
    """Accept JSON, fenced JSON, or a Qwen thinking preamble plus JSON."""
    decoder = json.JSONDecoder()
    for pos, char in enumerate((content or "")[:2500]):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(content[pos:])
        except (ValueError, TypeError):
            continue
        if isinstance(candidate, dict) and "intent" in candidate:
            return candidate
    return None


def validate_plan(raw: object) -> Optional[MemoryPlan | str]:
    """Return a safe plan, ``other``, or ``None`` for malformed output."""
    if not isinstance(raw, dict) or raw.get("intent") not in ("memory", "other"):
        return None
    if raw["intent"] == "other":
        return "other"
    sources = raw.get("sources")
    allowed = {"preferences", "experiences", "facts"}
    if (not isinstance(sources, list) or not sources or len(sources) > 3
            or any(not isinstance(s, str) or s not in allowed for s in sources)):
        return None
    operation = raw.get("operation", "recall")
    if operation not in {"recall", "current_turn", "correction", "confirmation",
                         "learn", "ordinary"}:
        return None
    subject = raw.get("subject")
    if subject is not None and (not isinstance(subject, str)
                                or not subject.strip() or len(subject) > 100):
        return None
    days = raw.get("recent_days")
    if (days is not None and (isinstance(days, bool) or not isinstance(days, int)
                              or not 1 <= days <= 365)):
        return None
    for key in ("first_mention", "reasons", "clarify"):
        if not isinstance(raw.get(key), bool):
            return None
    result_count = raw.get("result_count")
    if (result_count is not None and
            (isinstance(result_count, bool) or not isinstance(result_count, int)
             or not 1 <= result_count <= 30)):
        return None
    if raw.get("order", "relevance") not in ("relevance", "recent", "oldest"):
        return None
    temporal_scope = raw.get("temporal_scope", "current")
    if temporal_scope not in ("current", "historical", "recent"):
        return None
    return MemoryPlan(
        tuple(dict.fromkeys(sources)), subject.strip() if subject else None, days,
        raw["first_mention"], raw["reasons"], raw["clarify"], operation,
        result_count, raw.get("order", "relevance"), temporal_scope,
    )


def interpret(brain: Brain, user_text: str,
               previous: tuple[dict[str, str], ...]):
    """Explicit compatibility call for a model-generated structured plan.

    Production Aurora does not invoke this function. There is deliberately no
    deterministic fallback: if the model cannot interpret the turn, the caller
    must continue the ordinary cognitive path.
    """
    prompt = {
        "recent_dialogue_reference_only": list(previous[-8:]),
        "latest_user_request": str(user_text or "")[:2000],
    }
    try:
        response = brain.generate([
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ], max_output_tokens=300)
    except Exception:
        return None
    if not response.success:
        return None
    return validate_plan(_json_object(response.content or ""))


def grounded_answer(brain: Brain, question: str, plan: MemoryPlan,
                    evidence: str) -> BrainResult:
    """Phrase verified evidence for explicit compatibility callers."""
    if not evidence:
        return BrainResult.success(
            "I couldn't find verified user statements supporting that answer. "
            "I don't want to guess.", brain.get_model_name())
    if evidence.startswith("I couldn't verify a user-stated reason"):
        return BrainResult.success(evidence, brain.get_model_name())
    instruction = (
        "Answer the personal-memory question using ONLY the verified evidence. "
        "Evidence contains verbatim user reports with source IDs and UTC times "
        "when the user spoke. Never treat assistant text as a user fact or add "
        "a reason, date, or relationship absent from the evidence. If support "
        "is missing, say so plainly. Do not emit capability requests or JSON."
    )
    reply = brain.generate([
        {"role": "system", "content": instruction},
        {"role": "user", "content": json.dumps({
            "question": question, "plan": plan.as_dict(),
            "verified_evidence": evidence,
        }, ensure_ascii=False)},
    ], max_output_tokens=540)
    if (not reply.success or not (reply.content or "").strip()
            or request_shape.is_request_attempt(reply.content or "")):
        return BrainResult.success(evidence, brain.get_model_name())
    dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", evidence))
    claimed_dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", reply.content))
    ids = set(re.findall(r"\[message (\d+),", evidence))
    claimed_ids = set(re.findall(r"\b(?:source\s+)?message\s+(\d+)\b",
                                 reply.content, re.I))
    if not claimed_dates.issubset(dates) or not claimed_ids.issubset(ids):
        return BrainResult.success(evidence, brain.get_model_name())
    return reply
