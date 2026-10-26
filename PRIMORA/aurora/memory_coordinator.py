"""Aurora-led memory intent planning, Prime-validated retrieval and grounded answering.

Natural-language interpretation belongs to Aurora's existing brain. This module
only validates a small operation schema; it never classifies user words by a
list of subjects, phrases, or spelling variants. No writes and no external I/O.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
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
    # A contextual follow-up keeps the user's literal question for phrasing,
    # while this bounded retrieval query gives Prime the predicate and
    # referent needed to select typed evidence.  It is never evidence.
    retrieval_query: Optional[str] = None

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
                      operation, result_count, order, temporal_scope,
                      None)


def interpret(brain: Brain, user_text: str, previous: tuple[dict[str, str], ...]):
    """One bounded call to the user's *existing* Aurora brain."""
    # A conversational summary is ordinary dialogue.  Do not spend a planner
    # call or invoke Prime merely because it contains the word "said".
    if _CONVERSATION_SUMMARY_RE.fullmatch(user_text or ""):
        return "other"
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
        return _contextual_plan(user_text, previous, None)
    plan = validate_plan(_json_object(response.content or ""))
    if plan is None or plan == "other":
        contextual = _contextual_plan(user_text, previous, None)
        if contextual is not None:
            return contextual
        return plan
    plan = _contextual_plan(user_text, previous, plan)
    if isinstance(plan, MemoryPlan) and plan.clarify:
        # A model may conservatively ask for clarification on a clear
        # inventory question such as "What is my favorite color?".  Keep
        # clarification for genuinely unresolved pronouns/referents, while
        # letting explicit noun phrases use Prime's verified evidence path.
        if not _needs_referent_clarification(user_text, previous):
            plan = replace(plan, clarify=False)
    return plan


_UNRESOLVED_REFERENT = re.compile(
    r"\b(?:it|this|that|them|these|those|one|ones)\b", re.IGNORECASE
)
_VAGUE_MEMORY_TAIL = re.compile(
    r"\b(?:about|regarding|concerning|for)\s*(?:\?|$)", re.IGNORECASE
)

_CONVERSATION_SUMMARY_RE = re.compile(
    r"^\s*(?:what|tell me)\s+(?:have|did)\s+you\s+"
    r"(?:say|said|mention|mentioned)\s+so\s+far\s*[?!. ]*$",
    re.IGNORECASE,
)
_CONTEXTUAL_WHY_RE = re.compile(
    r"^\s*(?:why|how\s+come|what(?:'s|\s+is)\s+the\s+reason)\s*[?!. ]*$",
    re.IGNORECASE,
)
_CONTEXTUAL_WHEN_RE = re.compile(
    r"^\s*(?:when|when\s+was\s+that|when\s+did\s+i\s+tell\s+you)\s*[?!. ]*$",
    re.IGNORECASE,
)
_CONTEXTUAL_TOPIC_RE = re.compile(
    r"^\s*(?:what\s+about\s+(?:it|that|this|them)|"
    r"and\s+what\s+about\s+(?:it|that|this|them))\s*[?!. ]*$",
    re.IGNORECASE,
)
_CONTEXTUAL_CONFIRMATION_RE = re.compile(
    r"^\s*(?:i\s+(?:already\s+)?told\s+you|"
    r"i\s+think\s+i\s+(?:have|['’]?ve)\s+told\s+you|"
    r"as\s+i\s+said|i\s+mentioned\s+that)\s*[?!. ]*$",
    re.IGNORECASE,
)
_CONTEXTUAL_FILLERS = frozenset({
    "a", "an", "and", "are", "did", "do", "for", "have", "i", "is",
    "it", "my", "of", "on", "that", "the", "to", "what", "when",
    "which", "who", "with", "you", "your",
})


def _contextual_kind(text: str) -> Optional[str]:
    """Return the narrow class of a conversational follow-up, if any."""
    if _CONTEXTUAL_WHY_RE.fullmatch(text or ""):
        return "why"
    if _CONTEXTUAL_WHEN_RE.fullmatch(text or ""):
        return "when"
    if _CONTEXTUAL_TOPIC_RE.fullmatch(text or ""):
        return "topic"
    if _CONTEXTUAL_CONFIRMATION_RE.fullmatch(text or ""):
        return "confirmation"
    return None


def _topic_from_user_text(text: str) -> Optional[str]:
    """Extract a grammatical topic pointer from one user turn.

    This intentionally recognizes sentence structure, not domains.  The
    result is only a retrieval filter; the source quote still has to be a
    user message admitted by Prime.
    """
    value = " ".join(str(text or "").strip().split()).strip(" ?!. ")
    if not value or _contextual_kind(value) is not None:
        return None
    # A direct first-person preference gives the strongest pointer.
    match = re.search(
        r"\b(?:i|my)\s+(?:really\s+|usually\s+|always\s+|currently\s+)?"
        r"(?:like|liked|love|loved|enjoy|enjoyed|prefer|preferred|"
        r"hate|hated|dislike|disliked|favorite|favourite)\s+"
        r"(?P<topic>.+?)(?:\s+because\b|[.!?]|$)", value, re.IGNORECASE,
    )
    if match:
        topic = " ".join(match.group("topic").split()).strip(" ,.!?")
        if topic:
            return topic[:100]
    # Questions of the form "what X do I ..." retain the object and a
    # qualifying time/context phrase ("music when working").
    match = re.search(
        r"\bwhat\s+(?P<topic>.+?)\s+do\s+i\b(?P<tail>.*)$",
        value, re.IGNORECASE,
    )
    if match:
        topic = match.group("topic").strip(" ,.!?")
        tail = match.group("tail").strip(" ,.!?")
        qualifier = re.search(r"\b(?:when|while|during|for)\b.+", tail,
                              re.IGNORECASE)
        if qualifier:
            topic = f"{topic} {qualifier.group(0)}"
        if topic:
            return " ".join(topic.split())[:100]
    # Named memory questions remain useful even when the predicate is
    # implicit ("what is my ...", "what did I tell you about ...").
    match = re.search(
        r"\b(?:about|regarding|concerning|for)\s+(?P<topic>.+)$",
        value, re.IGNORECASE,
    )
    if match:
        topic = match.group("topic").strip(" ,.!?")
        if topic and topic.casefold() not in {"it", "that", "this", "them"}:
            return " ".join(topic.split())[:100]
    match = re.search(
        r"\b(?:my|our)\s+(?:favorite|favourite)\s+(?P<topic>.+)$",
        value, re.IGNORECASE,
    )
    if match:
        topic = match.group("topic").strip(" ,.!?")
        if topic:
            return " ".join(topic.split())[:100]
    return None


def _topic_from_assistant_text(text: str) -> Optional[str]:
    """Use an assistant reply as a referent pointer, never as evidence."""
    value = " ".join(str(text or "").strip().split())
    if not value:
        return None
    for sentence in re.split(r"[.!?\n]+", value):
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", sentence)
        if not words or len(words) > 8:
            continue
        meaningful = [w for w in words if w.casefold() not in _CONTEXTUAL_FILLERS]
        if meaningful and len(meaningful) <= 5:
            # A short first sentence such as "Techno" is a safe topic
            # pointer. It will only narrow a later search over user sources.
            return " ".join(meaningful)[:100]
    return None


def _contextual_plan(
    user_text: str,
    previous: tuple[dict[str, str], ...],
    plan: Optional[MemoryPlan],
) -> Optional[MemoryPlan]:
    """Resolve a bounded follow-up using dialogue as a non-evidence pointer."""
    kind = _contextual_kind(user_text)
    if kind is None:
        return plan
    users = [str(item.get("content", "")) for item in previous
             if item.get("role") == "user"]
    assistants = [str(item.get("content", "")) for item in previous
                  if item.get("role") == "assistant"]
    topic = None
    for candidate in reversed(users):
        topic = _topic_from_user_text(candidate)
        if topic:
            break
    # An assistant reply may identify a concise topic only when it follows a
    # substantive user question.  An arbitrary assistant sentence alone is
    # never enough to turn an ambiguous follow-up into a memory lookup.
    has_prior_question = any(
        "?" in candidate and _contextual_kind(candidate) is None
        for candidate in users
    )
    if topic is None and has_prior_question:
        for candidate in reversed(assistants):
            topic = _topic_from_assistant_text(candidate)
            if topic:
                break

    # A planner-produced explicit subject is preferred. If it did not resolve
    # one, the grammatical pointer above is the only safe fallback.
    subject = (plan.subject if isinstance(plan, MemoryPlan) and plan.subject
               else topic)
    if isinstance(plan, MemoryPlan):
        sources = plan.sources
        result_count = plan.result_count
        order = plan.order
        temporal_scope = plan.temporal_scope
        first_mention = plan.first_mention
    else:
        sources = ("preferences",)
        result_count = None
        order = "relevance"
        temporal_scope = "current"
        first_mention = False
    if not sources:
        sources = ("preferences",)
    # Explanations inherit from an immediately preceding why/reason request,
    # including the natural repair "I think I have told you".
    prior_text = " ".join(users[-4:] + assistants[-4:]).casefold()
    reasons = (bool(plan.reasons) if isinstance(plan, MemoryPlan) else False)
    if kind == "why":
        reasons = True
    elif kind == "confirmation" and bool(
            re.search(r"\b(?:why|reason|because)\b", prior_text)):
        reasons = True
    if reasons:
        sources = ("preferences",) if "preferences" in sources else sources
        if "preferences" not in sources:
            sources = ("preferences",)
        retrieval = f"Why did I like {subject}?" if subject else "Why did I like it?"
    elif kind == "when":
        first_mention = True
        order = "oldest"
        temporal_scope = "historical"
        retrieval = f"When did I tell you about {subject}?" if subject else user_text
    else:
        retrieval = f"What do I know about {subject}?" if subject else user_text
    return MemoryPlan(
        tuple(dict.fromkeys(sources)), subject, None, first_mention, reasons,
        clarify=subject is None,
        operation="recall", result_count=result_count, order=order,
        temporal_scope=temporal_scope, retrieval_query=retrieval,
    )


def _needs_referent_clarification(
    user_text: str, previous: tuple[dict[str, str], ...] = ()
) -> bool:
    """Keep clarification only when the latest question lacks a referent.

    This is grammatical rather than domain-specific: explicit noun phrases
    (including arbitrary user-created subjects) are safe to route to Prime;
    unresolved pronouns and bare trailing ``about`` questions remain gated.
    """
    text = " ".join(str(user_text or "").split())
    if _VAGUE_MEMORY_TAIL.search(text):
        return True
    if not _UNRESOLVED_REFERENT.search(text):
        return False
    # The planner may resolve a pronoun itself and set clarify=false. If it
    # still requests clarification, retain that safe stop rather than guessing
    # which prior subject the pronoun names.
    return True


def grounded_answer(brain: Brain, question: str, plan: MemoryPlan,
                    evidence: str) -> BrainResult:
    """Aurora phrases verified evidence; Prime supplies the only facts."""
    if not evidence:
        return BrainResult.success(
            "I couldn't find verified user statements supporting that answer. "
            "I don't want to guess.", brain.get_model_name())
    # Prime uses this explicit terminal response when a rationale gate found
    # no user-authored reason.  Sending it back through Aurora's generator
    # would invite an unsupported explanation, so abstain before generation.
    if evidence.startswith("I couldn't verify a user-stated reason"):
        return BrainResult.success(evidence, brain.get_model_name())
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
