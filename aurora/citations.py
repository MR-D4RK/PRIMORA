"""Dangling citation checks - which need the source numbers, which the loop may not hold.

The loop's generic rules ask only about claims and repeats; they are
deliberately source-free, and the scope tests that guard the loop reject
the name of any capability appearing in that file's code. Deciding whether
a ``[3]`` in an answer refers to a real source needs the set of numbers the
research machinery currently holds, so the decision lives here, next to
where the numbers are produced, and the loop calls it without ever naming
anything - exactly the pattern :mod:`PRIMORA.aurora.claims` established.

Lenient on purpose
------------------
An answer with no citations at all is fine: most answers need none, and
requiring a citation on every reply would teach Aurora to sprinkle numbers
she does not mean. Every citation that points at a number the registry
holds is also fine. Only an impossible citation - a number the current
interaction's search never produced - is corrected, and the correction is
one tool turn telling her which numbers are real, not a refusal of the
answer. The registry is cleared between interactions, so a number carried
over from an earlier turn is impossible by construction.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

# A bracketed positive integer, the shape Aurora is told to write. Three
# digits is more than any plausible registry; larger numbers are simply
# dangling like any other.
_CITATION = re.compile(r"\[(\d{1,3})\]")


def cited_numbers(content: str) -> list[int]:
    """Every ``[N]`` marker in ``content``, ascending and de-duplicated."""
    return sorted({int(m) for m in _CITATION.findall(content or "")})


def dangling_citation_error(
    content: str, available: Sequence[int]
) -> Optional[str]:
    """The correction to send when an answer cites a source that does not exist.

    Returns the error text when ``content`` contains at least one ``[N]``
    whose ``N`` is not in ``available``; returns None when there are no
    citations at all, or when every citation is valid. The caller injects
    the text as a tool turn and grants one corrective cycle, exactly as an
    unbacked claim is corrected.
    """
    cited = cited_numbers(content)
    if not cited:
        return None
    valid = set(available)
    dangling = [n for n in cited if n not in valid]
    if not dangling:
        return None
    listed = ", ".join(str(n) for n in sorted(valid)) if valid else "none"
    shown = ", ".join(f"[{n}]" for n in dangling)
    return (
        f"invalid_citation: you cited {shown}, but no such source is "
        f"available. Available source numbers: {listed}. Cite only the "
        "numbers listed here, or remove the citation and answer from what "
        "you already have."
    )
