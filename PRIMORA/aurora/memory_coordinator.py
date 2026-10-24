"""Aurora-led memory intent planning, Prime-validated retrieval and grounded answering.

Natural-language interpretation belongs to Aurora's existing brain. This module
only validates a small operation schema; it never classifies user words by a
list of subjects, phrases, or spelling variants. No writes and no external I/O.
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
    # ``recall`` is the only operation that may take the early, evidence-only
    # path. Current disclosures, corrections, acknowledgements, and ordinary
    # conversation return to the normal Aurora loop with this operation set to
    # their semantic class.
    operation: str = "recall"
    result_count: Optional[int] = None
    order: str = "relevance"
    temporal_scope: str = "current"

    def as_dict(self) -> dict:
        return {
            "sources": list(self.sources), "subject": self.subject,
            "recent_days": self.recent_days, "first_mention": self.first_mention,
            "reasons": self.reasons, "clarify": self.clarify,
            "operation": self.operation, "result_count": self.result_count,
            "order": self.order, "temporal_scope": self.temporal_scope,
        }


_SYSTEM = """You are Aurora interpreting the LATEST USER MESSAGE, not answering it.
First decide whether the latest message is a request to retrieve existing
personal memory or a current conversational act. A disclosure, a correction,
an acknowledgement/confirmation, a request to remember something, and ordinary
conversation must not be routed into historical lookup merely because earlier
dialogue mentions memory. Historical evidence never overrides explicit content
in the latest user message.
Understand meaning, ordinary typos, paraphrases, and immediate references. Do
not rely on subject-specific keywords or claim that you retrieved anything.
Do not treat public facts, research, or general advice as personal memory.
Output ONLY one JSON object with EXACTLY these keys:
{"intent":"memory", "operation":"recall", "sources":["preferences"],
 "subject":null, "recent_days":null, "first_mention":false,
 "reasons":false, "clarify":false, "result_count":null,
 "order":"relevance", "temporal_scope":"current"}
Set intent to "memory" or "other". For "other", use an empty sources list.
For a current act, keep intent="memory" only when useful memory context is
needed, and set operation to one of current_turn, correction, confirmation,
learn, or ordinary. These operations return to normal conversation; they are
not historical answers. Use operation="recall" only for an explicit request
about what the user previously said, stored preferences, past experiences, or
historical facts.
Fill every field from the latest request; do not blindly copy example values.
- sources contains zero or more distinct values: preferences (likes, favorites,
  reasons), experiences (actions/events), facts (other personal user statements).
- For memory intent, choose the useful sources, sometimes more than one.
- subject: a specific named item to search, or null for ANY inventory,
  including inventories narrowed by a broad category like vehicles, books or
  colors. Let the answering stage select the category from evidence; resolve
  'it/that/them' ONLY from a clear nearby referent. Previous assistant
  replies may help identify the topic but are NOT evidence of personal facts.
  Do not invent a subject.
- recent_days: integer 1..365 if the requested window is clear; interpret
  'recently' as 7 days, or null if there is no time restriction.
- result_count: integer 1..30 when the user requests a number of results,
  otherwise null. order is relevance, recent, or oldest. temporal_scope is
  current, historical, or recent.
- first_mention: true when the user asks when they told you/first mentioned it,
  NOT when the preference or event began in the real world.
- reasons: true for questions asking why the user likes/does something.
- clarify: true if the user's memory question needs a specific referent but
  the provided conversation doesn't identify one safely.
Do not invent evidence, dates, activities or user preferences. No commentary."""


def _json_object(content: str) -> Optional[dict]:
    """Accept JSON, fenced JSON, or a Qwen thinking preamble plus JSON."""
    decoder = json.JSONDecoder()
    for pos, char in enumerate(content[:2500]):
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
    """Return a safe plan, 'other', or None for malformed model output."""
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
    allowed_operations = {
        "recall", "current_turn", "correction", "confirmation",
        "learn", "ordinary",
    }
    if not isinstance(operation, str) or operation not in allowed_operations:
        return None
    subject = raw.get("subject")
    if subject is not None and (not isinstance(subject, str)
                                or not subject.strip() or len(subject) > 100):
        return None
    days = raw.get("recent_days")
    if days is not None and (isinstance(days, bool)
                             or not isinstance(days, int) or not 1 <= days <= 365):
        return None
    for key in ("first_mention", "reasons", "clarify"):
        if not isinstance(raw.get(key), bool):
            return None
    result_count = raw.get("result_count")
    if (result_count is not None and
            (isinstance(result_count, bool) or not isinstance(result_count, int)
             or not 1 <= result_count <= 30)):
        return None
    order = raw.get("order", "relevance")
    if order not in ("relevance", "recent", "oldest"):
        return None
    temporal_scope = raw.get("temporal_scope", "current")
    if temporal_scope not in ("current", "historical", "recent"):
        return None
    return MemoryPlan(tuple(dict.fromkeys(sources)),
                      subject.strip() if subject else None, days,
                      raw["first_mention"], raw["reasons"], raw["clarify"],
                      operation, result_count, order, temporal_scope)


def interpret(brain: Brain, user_text: str, previous: tuple[dict[str, str], ...]):
    """One bounded call to the user's *existing* Aurora brain."""
    prompt = {
        # Dialogue helps with references, but is NOT factual evidence.
        "recent_dialogue_reference_only": list(previous[-6:]),
        "latest_user_request": user_text[:2000],
    }
    response = brain.generate([
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ], max_output_tokens=300)
    if not response.success:
        return None
    return validate_plan(_json_object(response.content or ""))


def grounded_answer(brain: Brain, question: str, plan: MemoryPlan,
                    evidence: str) -> BrainResult:
    """Aurora phrases verified evidence; Prime supplies the only facts."""
    if not evidence:
        return BrainResult.success(
            "I couldn't find verified user statements supporting that answer. "
            "I don't want to guess.", brain.get_model_name())
    instruction = (
        "You are Aurora. Answer the user's personal memory question using ONLY "
        "the evidence below. Evidence consists of verbatim user reports with "
        "source IDs and UTC times WHEN USER SPOKE, NOT WHEN AN EVENT OCCURRED. "
        "Do not infer when someone started liking something from the timestamp "
        "of a later report. Never treat an assistant's text as a user fact. "
        "Never add facts, reasons, dates, or relationships absent from evidence. "
        "If the answer isn't supported, explicitly say so. "
        "If evidence is truncated, do not claim it is exhaustive. "
        "Differentiate an immediate possible reaction from a verified link. "
        "Use YYYY-MM-DD format when giving source dates. "
        "Index labels ('stored:') are metadata; verbatim quotes are evidence. "
        "Keep the reply natural, concise, and directly responsive. "
        "Do NOT emit capability requests, JSON, or action claims."
    )
    reply = brain.generate([
        {"role": "system", "content": instruction},
        {"role": "user", "content":
            json.dumps({"question": question, "plan": plan.as_dict(),
                        "verified_evidence": evidence}, ensure_ascii=False)},
    ], max_output_tokens=540)
    if (not reply.success or not (reply.content or "").strip()
            or request_shape.is_request_attempt(reply.content or "")):
        return BrainResult.success(evidence, brain.get_model_name())
    # Structural provenance guard: a newly invented timestamp or citation
    # is unsafe regardless of how fluent the answer sounds. A semantic claim
    # still requires live evaluation; this is not an entailment prover.
    dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", evidence))
    claimed_dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", reply.content))
    ids = set(re.findall(r"\[message (\d+),", evidence))
    claimed_ids = set(re.findall(r"\b(?:source\s+)?message\s+(\d+)\b",
                                 reply.content, re.I))
    if not claimed_dates.issubset(dates) or not claimed_ids.issubset(ids):
        return BrainResult.success(evidence, brain.get_model_name())
    return reply
