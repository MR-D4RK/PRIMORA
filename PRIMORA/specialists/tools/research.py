"""The ``research`` capability: Aurora asking the outside world a question.

Milestone 3. The division of responsibility is the point of this file:

    Aurora decides WHAT she wants to know.
    This capability decides HOW to obtain it.

Aurora sends a natural-language query and nothing else. She does not choose a
backend, pass a request parameter, name a format, or see a URL template. All
of that lives behind :class:`PRIMORA.research.provider.ResearchProvider`, so
swapping search engines cannot change the interface Aurora uses.

Deliberately absent: multi-step investigation, page fetching, crawling, and
recursion. One query, one ranked result set, returned to Aurora, who then
continues thinking. That continuation is the loop's job, not this file's.
"""

from typing import Any, Optional

from PRIMORA.prime.contract import PrimeResult
from PRIMORA.research.provider import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    ResearchProvider,
    UnconfiguredResearchProvider,
)
from PRIMORA.research.sources import SourceRegistry
from PRIMORA.specialists.specialist import Specialist

# Shipped with every successful search so Aurora knows, in the same turn the
# numbers appear, what they are for. Kept here rather than in the static
# system instruction because it is only true when sources actually exist.
CITE_INSTRUCTION = (
    "Numbered sources are available. When your answer uses one, cite it "
    "inline as [1], [2] matching its 'ref'. Cite only these numbers; to "
    "read a page in full, request research_read with the same numbers."
)


class ResearchSpecialist(Specialist):
    """Gather external information through a replaceable search provider."""

    name = "research"

    def __init__(
        self,
        provider: Optional[ResearchProvider] = None,
        sources: Optional[SourceRegistry] = None,
    ) -> None:
        self._provider = provider or UnconfiguredResearchProvider()
        self._sources = sources

    @property
    def provider(self) -> ResearchProvider:
        return self._provider

    @property
    def is_configured(self) -> bool:
        """Whether a real search backend is wired.

        Left unset this capability fails closed, so advertising it would
        only invite a request that is guaranteed to fail. The capability
        instruction reads this and omits the entry rather than promising
        a reach that is not there.
        """
        return not isinstance(self._provider, UnconfiguredResearchProvider)

    @property
    def sources(self) -> Optional[SourceRegistry]:
        return self._sources

    def set_provider(self, provider: ResearchProvider) -> None:
        """Swap the backend without touching the capability interface."""
        if provider is None:
            raise ValueError("research provider cannot be None")
        self._provider = provider

    def set_sources(self, registry: SourceRegistry) -> None:
        """Bind the shared registry that turns results into citable numbers.

        Left unset, results carry no ``ref`` and no citation instruction:
        the capability still works exactly as it did before numbers existed,
        which is why every pre-Step-2 test keeps passing unchanged.
        """
        if registry is None:
            raise ValueError("source registry cannot be None")
        self._sources = registry

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

        try:
            results = self._provider.search(query, limit=limit)
        except Exception as exc:
            # Transport, DNS, timeout, and backend errors all land here. The
            # model is told the search failed so she can say so or try a
            # different way; the loop is never crashed by a failed search.
            return PrimeResult.failure(f"research_failed: {exc}")

        found = [self._number(r.to_dict(), query) for r in results[:limit]]
        if not found:
            # An empty result set is not usable grounding. Returning it as
            # success taught Aurora that a search which found nothing had
            # succeeded, so she would either invent an answer or treat the
            # empty set as verified. A clear failure lets her say the search
            # found nothing and stop - the distinction the launch checklist
            # requires between "genuine results" and "a clear error".
            return PrimeResult.failure(
                "research_empty: no sources found for this query"
            )

        output: dict[str, Any] = {"query": query, "results": found}
        if self._sources is not None:
            output["instructions"] = CITE_INSTRUCTION
        return PrimeResult.success(output)

    def _number(self, result: dict[str, str], query: str) -> dict[str, Any]:
        """Attach a stable ``ref`` when a registry is bound; else pass through.

        The registry keeps the first number it gave a URL, so re-running a
        search does not renumber what Aurora may already have cited.
        """
        if self._sources is None:
            return result
        entry = self._sources.register(
            url=result.get("url", ""),
            title=result.get("title", ""),
            snippet=result.get("snippet", ""),
            source=result.get("source", ""),
            query=query,
        )
        numbered = dict(result)
        numbered["ref"] = entry.ref
        return numbered
