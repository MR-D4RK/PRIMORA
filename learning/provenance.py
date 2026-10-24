"""Provenance attestation: checking a learning claim against what happened.

Milestone 3.

The problem this exists to solve
-------------------------------
Provenance is supposed to keep an inference from silently becoming a fact. But
provenance is *self-reported*: Aurora writes the label. Nothing stops her
researching something, then recording it as ``"user"`` and having it admitted
as something Amir said. The authority invariant downstream only knows the
label, so the label has to be checked somewhere.

The check
---------
The session already knows what happened. The executor records what research
and specialist calls returned. This module compares the claim against:

* the external results gathered **this turn**, and
* what Amir has actually said in this session,

and downgrades a ``"user"`` claim when it is better supported by an external
result than by anything Amir said. A downgraded claim is not rejected: it is
re-labelled with the source that does support it, which means it is recorded
as pending rather than admitted as fact.

What this is not
-----------------
Not a proof. It is a deterministic, auditable guard that closes the trivial
bypass, and it will not catch a determined paraphrase. It is deliberately
biased toward *not* interfering: if the claim matches user speech at least as
well as any external result, it is left alone, so ordinary learning from
something Amir said is untouched.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

# Provenance a claim is downgraded to, given the capability that produced the
# supporting external result.
SOURCE_FOR_CAPABILITY = {
    "research": "research",
    "consult_specialist": "specialist",
    "use_tool": "tool",
}

_STOP = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of",
    "in", "on", "for", "and", "or", "that", "this", "it", "as", "at", "by",
    "with", "from", "has", "have", "had", "he", "she", "they", "his", "her",
    "their", "my", "your", "our", "i", "you", "me", "we", "us", "not", "now",
    "am", "s", "t",
})


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return {w for w in words if w not in _STOP and len(w) > 1}


def _overlap(claim: set[str], other: set[str]) -> float:
    """Fraction of the claim's content words supported by ``other``."""
    if not claim:
        return 0.0
    return len(claim & other) / len(claim)


def content_tokens(text: str) -> frozenset:
    """The content words of a text, as a reusable fingerprint.

    Session evidence is stored in this form rather than as text. Comparing
    two claims only ever needs to know which words they share, so retaining
    the words themselves would keep tool output alive for no reason.
    """
    return frozenset(_tokens(text))


def attest(
    text: str,
    claimed: str,
    evidence: Iterable[tuple[str, Any]],
    user_texts: Iterable[str] = (),
) -> tuple[str, str]:
    """Check a claimed provenance against the session.

    ``evidence`` is an iterable of ``(capability, tokens)`` where ``tokens``
    is a content fingerprint as produced by :func:`content_tokens`.

    Returns ``(provenance, diagnostic)``. The provenance is unchanged unless
    the claim is a ``"user"`` claim that the evidence contradicts, in which
    case it is downgraded to the source that does support it and the
    diagnostic explains why.
    """
    if claimed != "user":
        return claimed, ""

    external = [
        (capability, tokens) for capability, tokens in evidence
        if capability in SOURCE_FOR_CAPABILITY and tokens
    ]
    if not external:
        return claimed, ""

    claim = _tokens(text)
    if not claim:
        return claimed, ""

    user_support = 0.0
    for said in user_texts:
        user_support = max(user_support, _overlap(claim, _tokens(said)))

    best_capability, best_support = "", 0.0
    for capability, tokens in external:
        support = _overlap(claim, tokens)
        if support > best_support:
            best_capability, best_support = capability, support

    # Downgrade only when the external result explains nearly the whole claim
    # *and* explains it better than anything Amir said. A question that shares
    # vocabulary with the answer ("what is the latest Kafka?") is weak
    # corroboration and must not count as the user having stated the fact.
    if (
        best_capability
        and best_support >= EXTERNAL_EXPLAINS
        and best_support > user_support
    ):
        actual = SOURCE_FOR_CAPABILITY[best_capability]
        return actual, (
            f"provenance_downgraded: claimed 'user' but the statement is "
            f"supported by {best_capability} output "
            f"({best_support:.0%} of it) rather than by anything said in this "
            f"session ({user_support:.0%}). Recorded as '{actual}' and held."
        )
    return claimed, ""


# How much of the claim an external result must explain before it is allowed
# to override a self-reported "user" label.
EXTERNAL_EXPLAINS = 0.8
