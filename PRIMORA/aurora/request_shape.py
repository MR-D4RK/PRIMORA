"""Shape rules for a Prime capability request, shared by two callers.

A reply either *is* a capability request or it is *not*, and two modules
need that answer for different reasons:

* :mod:`PRIMORA.aurora.aurora` extracts the request to execute it.
* :mod:`PRIMORA.aurora.loop` must recognise the shape so a JSON reply
  that is not a valid request is never handed to the user as though it
  were an answer - or, worse, as though a recall had run.

The rules live here, and not in either module, because both need them
and aurora imports the loop: a shared dependency in this direction would
be a cycle. Nothing in this module imports from the rest of Aurora, so
either side can hold it.

The format itself is registered nowhere. A request is a JSON object with
a registered action string and a payload object; everything else about a
JSON-looking reply is a guess, and the functions below refuse to make
it.
"""

import json
from typing import Any, Optional

from PRIMORA.prime.contract import PrimeRequest

# Keys that distinguish "the model is asking for a capability" from "the
# model is quoting or explaining a JSON object". Presence of one of these
# is the only evidence available before the object parses, and it is what
# makes a truncated request still recognisable as an attempt.
_REQUEST_KEYS = ("action", "payload")

# raw_decode, not loads: the difference is the whole point. ``loads``
# reports "Extra data" for a reply holding two objects, which is exactly
# the ambiguous case that has to be recognised rather than dropped.
_DECODER = json.JSONDecoder()


def request_from_data(data: Any) -> Optional[PrimeRequest]:
    """Build a request from an already-parsed JSON value, or None.

    The single definition of a valid request shape. ``action`` must be a
    non-empty string, ``payload`` an object, and ``context`` /
    ``metadata`` absent, null, or objects. A dict that fails any of
    these is not a request, whatever else it contains - so an unknown
    action still becomes a request (Prime rejects it by name, with the
    registered list) while a malformed shape never reaches Prime at all.
    """
    if not isinstance(data, dict):
        return None
    action = data.get("action")
    if not isinstance(action, str) or not action:
        return None
    payload = data.get("payload")
    if not isinstance(payload, dict):
        return None
    context = data.get("context")
    if context is not None and not isinstance(context, dict):
        return None
    metadata = data.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return None
    return PrimeRequest(action=action, payload=payload,
                        context=context, metadata=metadata)


def bare_request_text(text: Any) -> Optional[str]:
    """The whole reply, but only when it is exactly one JSON object.

    This is what a request block with its wrapper missing looks like: the
    entire reply *is* the object the model would have wrapped. Anything
    else - prose around JSON, two objects, an array, a truncated object -
    returns None, because a request is one whole object and the other
    shapes are far more likely to be Aurora quoting or explaining.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if len(stripped) < 2:
        return None
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        json.loads(stripped, strict=False)
    except (ValueError, TypeError):
        return None
    return stripped


def _names_request_key(obj: Any) -> bool:
    return isinstance(obj, dict) and any(
        key in obj for key in _REQUEST_KEYS)


def is_request_attempt(text: Any) -> bool:
    """Whether the reply is an object *attempting* to be a request.

    True for a whole-reply JSON object whether or not it validates: a
    well-formed request the extractor refused, an object missing
    ``payload``, one truncated by ``max_tokens`` mid-key, and a reply
    holding two objects are all the model asking for something. An object
    that names neither request key is a JSON answer or a quotation, not
    an attempt, and a JSON object inside a sentence is prose. This is
    deliberately the narrow question "did she try to ask for a
    capability", never "could I run it" - execution is the extractor's
    decision and Prime's authorization, not this one.

    A truncated object is the case that matters most in practice and the
    one a closed-brace test would miss: it is still the model asking for
    something, and it must not be mistaken for an answer. The cost is
    that a reply which opens with a JSON *example* is also read as an
    attempt and earns one correction turn; that is the cheap error, and
    it is taken deliberately, because the expensive error is a JSON blob
    reaching the user looking like a recall that ran.
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    if stripped.endswith("}"):
        # Whole object, or several. Decode the first one: an object that
        # names a request key is an attempt, and so is a second object
        # after it - two requests in one reply is an attempt that must not
        # be silently discarded as an answer.
        try:
            obj, end = _DECODER.raw_decode(stripped)
        except ValueError:
            obj, end = None, 0
        if _names_request_key(obj):
            return True
        # Parsed, named no request key, and left text behind: a JSON
        # answer followed by something else is not a request either.
        return False
    # Never closed: a truncation, or JSON followed by prose. Only an
    # object naming a request key is an attempt; a stray brace is not.
    lowered = stripped.lower()
    return any('"%s"' % key in lowered for key in _REQUEST_KEYS)


# Internal markers whose content is never part of Aurora's user-facing
# answer. Kept here (rather than in the loop or the brain) so the streaming
# path and the answer-cleaning path agree on what is internal.
_THINK_FENCE_OPEN = "```think"
_THINK_FENCE_CLOSE = "```"
_PRIME_REQUEST_OPEN = "<prime_request>"
_PRIME_REQUEST_CLOSE = "</prime_request>"


def _find_earliest_open(buffer: str, pos: int):
    """Return (index, marker) of the earliest internal-block opening at or
    after ``pos``, or (-1, None) when none remains."""
    best = -1
    best_marker = None
    for marker in (_THINK_FENCE_OPEN, _PRIME_REQUEST_OPEN):
        idx = buffer.find(marker, pos)
        if idx != -1 and (best == -1 or idx < best):
            best = idx
            best_marker = marker
    return best, best_marker


def displayable_text(buffer: str) -> str:
    """Return the user-facing-safe portion of ``buffer``.

    Complete internal blocks (thinking fences and ``<prime_request>``
    blocks) are removed from the output. If an internal block opens but
    never closes within ``buffer``, everything from its opening to the end
    of ``buffer`` is withheld: it may be incomplete reasoning or an
    incomplete internal request that must never reach the user.

    Earlier prefixes of the output are always prefixes of later outputs as
    ``buffer`` grows, which makes this safe for progressive delta emission
    during streaming.
    """
    out = []
    pos = 0
    n = len(buffer)
    while pos < n:
        next_open, marker = _find_earliest_open(buffer, pos)
        if next_open == -1:
            out.append(buffer[pos:])
            break
        out.append(buffer[pos:next_open])
        close = (
            _THINK_FENCE_CLOSE if marker == _THINK_FENCE_OPEN
            else _PRIME_REQUEST_CLOSE)
        close_idx = buffer.find(close, next_open + len(marker))
        if close_idx == -1:
            # Unclosed internal block: withhold from the opening onward.
            break
        pos = close_idx + len(close)
    return "".join(out)


def displayable_len(buffer: str) -> int:
    """Number of characters safe to display from ``buffer`` (see
    :func:`displayable_text`)."""
    return len(displayable_text(buffer))
