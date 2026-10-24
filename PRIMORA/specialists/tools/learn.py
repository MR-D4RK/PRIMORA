"""The ``learn`` capability: Aurora deciding something is worth keeping.

Milestone 3, and the first capability that *writes*.

    Aurora decides WHAT is worth learning, and where it came from.
    Jev decides HOW to structure it.
    The memory lifecycle decides whether it is admitted.

The division is the safety property. Aurora sends a sentence and a provenance
label. She does not send a concept, a value, a scope, a memory type, or any
part of the internal representation, and this capability rejects a payload
that tries to. If Aurora could write the structured form, she would be
constructing her own memory schema, and the provenance guarantee would mean
nothing: an inference could arrive pre-labelled as something the user said.

Learning is not retraining
---------------------------
Nothing here touches model weights, adapters, or the brain. Learning through
memory is writing a structured fact into the existing lifecycle. Retraining
is a different thing entirely and is not this capability.
"""

from typing import Any, Optional

from PRIMORA.learning.jev import MAX_TEXT_CHARS, PROVENANCE, UnconfiguredJev
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.specialists.specialist import Specialist

# Keys that would let Aurora hand-write the internal representation. Kept
# internal-memory shaped on purpose: if someone adds a field to the schema,
# this list is where the refusal has to be updated too.
_RESERVED_KEYS = frozenset({
    "concept", "value", "scope", "memory_type", "authority", "confidence",
    "status", "candidate_id", "state_id", "kind", "auto_confirm",
})


class LearnSpecialist(Specialist):
    """Propose something Aurora has decided is worth remembering."""

    name = "learn"

    def __init__(
        self,
        mind: Optional[Any] = None,
        jev: Optional[Any] = None,
        evidence_source: Optional[Any] = None,
    ) -> None:
        self._mind = mind
        self._jev = jev or UnconfiguredJev()
        # Whatever recorded what this turn's capabilities returned. Only
        # Aurora knows the loop, so it is attached there.
        self._evidence_source = evidence_source

    def set_evidence_source(self, source: Any) -> None:
        self._evidence_source = source

    def bind(self, mind: Any) -> None:
        self._mind = mind

    def set_jev(self, jev: Any) -> None:
        if jev is None:
            raise ValueError("jev cannot be None")
        self._jev = jev

    @property
    def jev(self) -> Any:
        return self._jev

    @property
    def is_configured(self) -> bool:
        """Whether a real Jev is wired for structuring lessons.

        Left unset this capability fails closed with ``jev_failed`` on every
        request. Advertising it would only invite a call that is guaranteed
        to fail, so the capability instruction reads this and omits the entry,
        exactly as research does when no provider is wired.
        """
        return not isinstance(self._jev, UnconfiguredJev)

    def handle(self, payload: dict[str, Any]) -> PrimeResult:
        if not isinstance(payload, dict):
            return PrimeResult.failure("Invalid payload type")

        offending = sorted(set(payload) & _RESERVED_KEYS)
        if offending:
            return PrimeResult.failure(
                "Aurora must not supply the internal memory structure "
                f"(refused: {', '.join(offending)}). Describe the lesson in "
                "'text' and name its origin in 'provenance'."
            )

        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return PrimeResult.failure("Missing or invalid 'text' in payload")
        if len(text) > MAX_TEXT_CHARS:
            return PrimeResult.failure(
                f"'text' is too long ({len(text)} characters, "
                f"maximum {MAX_TEXT_CHARS})"
            )

        provenance = payload.get("provenance")
        if not isinstance(provenance, str) or provenance not in PROVENANCE:
            return PrimeResult.failure(
                f"'provenance' must be one of: {', '.join(PROVENANCE)}"
            )

        if self._mind is None:
            return PrimeResult.failure("learn is not bound to a Mind")

        # During cognition the loop binds a trusted execution session. A
        # model-supplied session id may match that binding for compatibility,
        # but it can never redirect a write into another conversation.
        trusted_session = getattr(self._mind, "execution_session_id", None)
        requested_session = payload.get("session_id")
        if trusted_session:
            if requested_session is not None and requested_session != trusted_session:
                return PrimeResult.failure(
                    "'session_id' is controlled by the active execution session"
                )
            session_id = trusted_session
        else:
            # Direct programmatic callers retain the historical API. The live
            # cognitive path always has a trusted binding above.
            session_id = (
                requested_session
                or getattr(self._mind, "current_session_id", None)
                or "default"
            )
        if not isinstance(session_id, str) or not session_id.strip():
            return PrimeResult.failure("'session_id' must be a non-empty string")

        try:
            effective, diagnostic = self._attest(text, provenance)
        except Exception as exc:
            # A provenance check is part of the memory trust boundary. If it
            # cannot run, do not promote a model-provided label by default.
            return PrimeResult.failure(
                f"provenance_attestation_failed: {type(exc).__name__}"
            )

        try:
            summary = self._mind.learn(
                text, effective, self._jev, session_id=session_id
            )
        except Exception as exc:
            # Learning must never take Aurora down with it.
            return PrimeResult.failure(f"learn_failed: {exc}")

        if not isinstance(summary, dict) or "status" not in summary:
            return PrimeResult.failure("learn_returned_nothing")

        if diagnostic:
            # Tell Aurora plainly that her label was not accepted, so she is
            # not misled into thinking she recorded something as a fact.
            summary.setdefault("diagnostics", []).insert(0, diagnostic)

        # A rejected lesson is a failure, not a success payload carrying a
        # sad face inside it. The loop executor records anything with an
        # outer success status as a succeeded action, so wrapping a
        # rejection in success let Aurora claim she had learned something
        # the admission layer refused. "held" stays a success: the lifecycle
        # deliberately parked the candidate, which is a real outcome.
        if summary.get("status") == "rejected":
            reason = summary.get("reason") or "rejected"
            detail = f"{reason}; {diagnostic}" if diagnostic else reason
            return PrimeResult.failure(f"learn_rejected: {detail}")
        return PrimeResult.success(summary)

    def _attest(self, text: str, provenance: str) -> tuple[str, str]:
        """Check a self-reported provenance against the session.

        Aurora chooses the label, so the label is verified rather than
        trusted. The policy lives here, in the capability that uses it; the
        Mind is only asked what Amir actually said.
        """
        from PRIMORA.learning.provenance import attest

        user_texts = getattr(self._mind, "user_texts", None) or []
        return attest(text, provenance, self._evidence(), user_texts)

    def _evidence(self) -> list:
        """External results gathered this turn, if anything recorded them."""
        source = self._evidence_source
        if source is None:
            return []
        try:
            return list(getattr(source, "evidence", []) or [])
        except Exception:
            return []
