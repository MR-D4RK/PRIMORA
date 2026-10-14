"""The ``recall`` capability: Aurora asking its own memory a question.

Milestone 3, first capability. Deliberately thin. It holds a reference to the
``Mind`` and forwards to ``Mind.recall``, which delegates to
``ContextManager.recall`` and therefore to the existing M6 hybrid retrieval.

What this is not: not a search engine, not a new index, not a new ranking, and
not a second retrieval implementation. The channels, fusion, gates, and
evidence resolution are all the ones the context builder already uses. If M6
changes, recall changes with it, because it is the same code.
"""

import re
from typing import Any, Optional

from PRIMORA.prime.contract import PrimeResult
from PRIMORA.specialists.specialist import Specialist


# Bounded so a model cannot ask for an unbounded dump and blow the context.
DEFAULT_LIMIT = 5
MAX_LIMIT = 12

# Query-echo filter. The live loop persists the current user turn *before*
# recall runs, so retrieval's own words come back as the top hit. That is not
# a stored fact - it is the question being asked. Filtering it here, at the
# capability boundary, keeps ContextManager.recall (the M6 hybrid path) and
# its ranking untouched.
_ECHO_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "am", "i", "me", "my", "mine", "you", "your", "yours", "we", "our",
    "us", "ours", "what", "which", "who", "whom", "whose", "when", "where",
    "why", "how", "do", "does", "did", "done", "have", "has", "had", "can",
    "could", "will", "would", "shall", "should", "may", "might", "must",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "about",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "not", "so", "than", "too", "very",
    "just", "because", "if", "that", "this", "these", "those", "it", "its",
    "he", "she", "him", "her", "his", "hers", "they", "them", "their",
    "there", "here", "s", "t", "don", "now", "also", "really", "please",
})
_ECHO_WORD_RE = re.compile(r"[a-z0-9']+")


def _echo_key(text: str) -> frozenset[str]:
    """Stopword-stripped token set for echo comparison."""
    return frozenset(
        w for w in _ECHO_WORD_RE.findall(text.lower()) if w not in _ECHO_STOPWORDS
    )


def _is_query_echo(query: str, content: str) -> bool:
    """True when the content is the query itself (the just-persisted turn).

    Token-set equality after stopword removal, so ``"what is my fav car"``
    echoes query ``"fav car"`` (both reduce to ``{fav, car}``) while a real
    stored fact like ``"My favourite car is a Porsche 911"`` keeps extra
    tokens and is not dropped. Empty queries never echo.

    A foreign passage is compared on its body, so an older echo of the same
    question is caught too; the M6 cross-session prefix is not a new fact.
    """
    q = _echo_key(query)
    if not q:
        return False
    body = content.split("\n", 1)[-1] if content.startswith(
        "[from another conversation]") else content
    return q == _echo_key(body)


class RecallSpecialist(Specialist):
    """Retrieve relevant stored conversation via the existing M6 path."""

    name = "recall"

    def __init__(self, mind: Optional[Any] = None) -> None:
        # Set by Prime after construction, because the registry is built before
        # the Mind exists. Kept as an attribute rather than a constructor
        # argument so registration order stays unchanged.
        self._mind = mind

    def bind(self, mind: Any) -> None:
        """Attach the Mind this capability answers from."""
        self._mind = mind

    def _current_session(self) -> str:
        """The session of the task in progress.

        Aurora is not told her own session id in the manifest, and a model
        that omits ``session_id`` would otherwise be answered from
        ``"default"`` while her conversation lives under a different name.
        That is not cosmetic: M6 treats the searched session as current and
        everything else as cross-session, so the wrong default both caps the
        result set and mislabels every passage as foreign.

        Degrades to ``"default"`` for a minimal Mind-like collaborator that
        does not expose the property, rather than failing the request.
        """
        return getattr(self._mind, "current_session_id", None) or "default"

    def handle(self, payload: dict[str, Any]) -> PrimeResult:
        if not isinstance(payload, dict):
            return PrimeResult.failure("Invalid payload type")
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            return PrimeResult.failure("Missing or invalid 'query' in payload")

        raw_limit = payload.get("limit", DEFAULT_LIMIT)
        if raw_limit is None:
            limit = DEFAULT_LIMIT
        elif isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
            return PrimeResult.failure("'limit' must be an integer")
        else:
            limit = max(1, min(raw_limit, MAX_LIMIT))

        if self._mind is None:
            return PrimeResult.failure("recall is not bound to a Mind")

        session_id = payload.get("session_id") or self._current_session()
        if not isinstance(session_id, str):
            return PrimeResult.failure("'session_id' must be a string")

        try:
            results = self._mind.recall(query, session_id=session_id, limit=limit)
        except Exception as exc:
            # A retrieval failure must be reportable to Aurora, not fatal.
            return PrimeResult.failure(f"recall_failed: {exc}")

        # Drop the query's own echo before it reaches Aurora as a "hit".
        filtered = [
            item for item in results
            if not _is_query_echo(query, str(item.get("content") or ""))
        ]

        return PrimeResult.success(
            {
                "query": query,
                "results": [
                    {"role": item.get("role"), "content": item.get("content")}
                    for item in filtered
                ],
            }
        )
