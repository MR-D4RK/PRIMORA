"""Evidence access for Aurora's normal cognition, not a language router.

The existing model decides *when* to request evidence and supplies a bounded
structured plan. Prime validates parameters and never treats model claims as
user evidence. Read-only; it never mutates a memory or calls external services.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import time
from typing import Any, Optional

from PRIMORA.prime.contract import PrimeResult
from PRIMORA.specialists.specialist import Specialist


class MemoryEvidenceSpecialist(Specialist):
    name = "memory_evidence"

    def __init__(self, mind: Optional[Any] = None) -> None:
        self._mind = mind

    def bind(self, mind: Any) -> None:
        self._mind = mind

    def handle(self, payload: dict[str, Any]) -> PrimeResult:
        if self._mind is None:
            return PrimeResult.failure("memory_evidence is not bound to Mind")
        if not isinstance(payload, dict):
            return PrimeResult.failure("Payload must be an object")
        allowed = {"sources", "subject", "date_from", "date_to", "first_mention", "reasons"}
        if set(payload) - allowed:
            return PrimeResult.failure("Unexpected memory evidence fields")
        sources = payload.get("sources")
        kinds = {"preferences", "experiences", "facts"}
        if (not isinstance(sources, list) or not 1 <= len(sources) <= 3
                or any(not isinstance(v, str) or v not in kinds for v in sources)):
            return PrimeResult.failure("sources must be a list of preferences, experiences or facts")
        subject = payload.get("subject")
        if subject is not None and (not isinstance(subject, str) or
                                    not 1 <= len(subject.strip()) <= 120):
            return PrimeResult.failure("subject must be a short string or null")
        for key in ("first_mention", "reasons"):
            if not isinstance(payload.get(key, False), bool):
                return PrimeResult.failure(f"{key} must be boolean")
        start_text, end_text = payload.get("date_from"), payload.get("date_to")
        if (start_text is None) != (end_text is None):
            return PrimeResult.failure("Both date_from and date_to are required for a date window")
        bounds = {}
        if start_text is not None:
            if (not isinstance(start_text, str) or not isinstance(end_text, str)
                    or len(start_text) != 10 or len(end_text) != 10):
                return PrimeResult.failure("Dates must be ISO YYYY-MM-DD")
            try:
                first = date.fromisoformat(start_text)
                last = date.fromisoformat(end_text)
            except ValueError:
                return PrimeResult.failure("Invalid calendar date")
            if last < first or (last - first).days > 365:
                return PrimeResult.failure("Date window must be ordered and at most 366 days")
            # The dates are local calendar days, not arbitrary rolling 24-hour
            # intervals. Convert to UTC timestamps at the boundary, never by
            # asking the language model to guess offsets or daylight saving.
            # mktime consults the OS timezone rules for *each target date*,
            # including DST transitions. datetime.now().astimezone().tzinfo
            # is commonly just today's fixed offset, which would be wrong.
            def local_midnight(day: date) -> float:
                return time.mktime((day.year, day.month, day.day, 0, 0, 0,
                                    0, 0, -1))
            bounds["from_utc"] = local_midnight(first)
            bounds["to_utc"] = local_midnight(last + timedelta(days=1))
        plan = {
            "sources": list(dict.fromkeys(sources)), "subject": subject,
            "first_mention": payload.get("first_mention", False),
            "reasons": payload.get("reasons", False), **bounds,
        }
        try:
            evidence = self._mind.memory_query_evidence(plan)
        except Exception as exc:
            return PrimeResult.failure(f"memory_evidence_failed: {type(exc).__name__}")
        return PrimeResult.success({
            "evidence": evidence or "No matching user-source evidence found.",
            "note": "Statement timestamps show when the user spoke, not when events occurred. "
                    "Raw user messages can be relevant even without a structured memory record. "
                    "Do not infer feelings, motives, or causal links without support.",
        })
