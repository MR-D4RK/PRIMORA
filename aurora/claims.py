"""Completion claims that need a capability name - which the loop may not hold.

The loop's generic claim rule (``loop.py``) asks only: did she assert that
something was already done, and did nothing run? It is deliberately
capability-free; the scope tests that guard the loop reject the name of any
capability appearing in that file's code.

Research is the case where the generic rule is wrong in both directions.
"I've searched the web" is backed by exactly one thing: a successful
research request. An unrelated success (a calculation, a recall) must not
silence the check, and a genuine research success must never be
contradicted. Deciding either way requires the capability's name, so the
decision lives here, next to the verbs, and the loop calls it without ever
naming anything.

The verbs use the same first-person frame as the loop's matcher: a claim
("I", "I've", "I will" ...) plus a past-tense action plus an object
(that/it/this/the). Ambiguity stays a non-claim, because a false
correction tells Aurora she is wrong when she is right.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

# The registered capability name that backs a research-shaped claim. Kept
# here so the loop never contains it: its source tests ban capability names
# in that module, which is exactly the boundary this file exists to hold.
RESEARCH_ACTION = "research"

_RESEARCH_VERB = r"searched|researched|browsed|fetched|queried"

_RESEARCH_CLAIM = re.compile(
    r"\b(?:i|i've|i've got|i have|i'll|i will)\s+"
    r"(?:have\s+)?(?:"
    + _RESEARCH_VERB
    + r")\s+(?:that|it|this|the)\b(?!\s*you\b)",
    re.IGNORECASE,
)

# Kept word-for-word identical to the loop's historical correction, so a
# research claim in an empty turn reads exactly like any other unbacked
# claim. The loop still emits its own copy for non-research verbs; this
# copy is here only because this module returns the whole message.
_GENERIC = (
    "unbacked_claim: you stated that something had already been done. "
    "No capability request has run, so it did not happen. If it should "
    "have, request it now; if not, correct yourself."
)

# Used when something ran, just not research. The generic wording would be
# false in that situation, and a correction that misstates the evidence is
# the kind of wrongness this whole mechanism exists to prevent.
_UNRELATED_RAN = (
    "unbacked_claim: you stated that something had already been done. "
    "No research request has run this turn, so it did not happen; "
    "unrelated capabilities running does not change that. If it should "
    "have, request it now; if not, correct yourself."
)


def _claims_research_action(text: str) -> bool:
    """Whether Aurora asserts that a search or research action already happened.

    True only for a clear first-person completion in the loop's claim
    frame. Anything ambiguous is False, so the caller stays silent rather
    than contradicting a statement that may well be true.
    """
    return _RESEARCH_CLAIM.search(text or "") is not None


def unsupported_claim_error(
    content: str, succeeded: Sequence[str]
) -> Optional[str]:
    """The correction to send when a research claim has no research behind it.

    Returns the error text when ``content`` closes the turn with a
    research-shaped completion claim and no research request succeeded
    this turn; returns None otherwise. Non-research claims are the loop's
    rule to apply, so they always return None here and the loop's own
    check decides them exactly as before.
    """
    if not _claims_research_action(content):
        return None
    if RESEARCH_ACTION in succeeded:
        # She says she searched, and she did. Nothing to correct.
        return None
    if succeeded:
        # Something ran, just not this. The unrelated success must not
        # suppress the check, and the message must not misstate the facts.
        return _UNRELATED_RAN
    return _GENERIC
