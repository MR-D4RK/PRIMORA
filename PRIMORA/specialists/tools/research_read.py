"""The ``research_read`` capability: reading one page Aurora was already pointed at.

Milestone 3, Phase 3 Step 2. A search returns ranked snippets; sometimes a
snippet is not enough to answer honestly, and the page itself has to be
read. That is this capability, and it is deliberately narrow:

    Aurora passes SOURCE NUMBERS, never URLs.
    Each number is looked up in the shared :class:`SourceRegistry`.
    At most three pages per request.
    Every page reports its own honest status; one failure never aborts the
    rest of the batch.

Why numbers, not URLs
---------------------
A URL the model writes is an SSRF surface and a provenance lie - it may
point anywhere, including back at this machine, and nothing guarantees it
is a page a search actually returned. A number copied out of a search
result can only name something this process received a moment ago. The
registry lookup is the whole security boundary, and it is why this file
never has to re-implement any of the fetcher's protections: it only ever
hands the fetcher a URL the registry already vetted by receiving it from
the provider.

Untrusted by construction
-------------------------
The text that comes back is evidence to read, never instructions to obey.
This module does not execute it, interpret it as a command, or let it
authorize anything. A future caller that pastes it into a prompt owns the
prompt-injection surface; this module owns only the lookup and the fetch.

Deliberately absent
-------------------
No crawling beyond the three named pages, no following links found in the
page, no "read everything from this search", and no way to widen the
allowlist from inside a request. The only URLs reachable are the ones the
current interaction's search already produced; the loop clears the registry
between interactions, so a number from an earlier turn reads as
``unknown_source``.
"""

from typing import Any, Optional

from PRIMORA.prime.contract import PrimeResult
from PRIMORA.research.fetcher import PageFetcher, STATUS_OK
from PRIMORA.research.sources import SourceRegistry
from PRIMORA.specialists.specialist import Specialist

# The hard ceiling on pages one request may open. Three is enough to
# cross-check a claim against a couple of independent sources and small
# enough that the extracted evidence stays well inside the context budget.
MAX_PAGES = 3


class ResearchReadSpecialist(Specialist):
    """Fetch and extract the pages behind already-numbered search sources."""

    name = "research_read"

    def __init__(
        self,
        sources: Optional[SourceRegistry] = None,
        fetcher: Optional[PageFetcher] = None,
    ) -> None:
        self._sources = sources
        self._fetcher = fetcher or PageFetcher()

    @property
    def sources(self) -> Optional[SourceRegistry]:
        return self._sources

    @property
    def fetcher(self) -> PageFetcher:
        return self._fetcher

    def set_sources(self, registry: SourceRegistry) -> None:
        """Bind the shared registry this capability looks numbers up in."""
        if registry is None:
            raise ValueError("source registry cannot be None")
        self._sources = registry

    def set_fetcher(self, fetcher: PageFetcher) -> None:
        """Swap the page fetcher. Tests inject; production uses the real one."""
        if fetcher is None:
            raise ValueError("fetcher cannot be None")
        self._fetcher = fetcher

    def handle(self, payload: dict[str, Any]) -> PrimeResult:
        if not isinstance(payload, dict):
            return PrimeResult.failure("Invalid payload type")

        raw = payload.get("sources")
        if not isinstance(raw, list) or not raw:
            return PrimeResult.failure(
                "Missing or invalid 'sources' in payload: pass a non-empty "
                "list of source numbers from your latest research"
            )
        if len(raw) > MAX_PAGES:
            return PrimeResult.failure(
                f"At most {MAX_PAGES} pages per request; "
                f"{len(raw)} were requested"
            )

        refs: list[int] = []
        for item in raw:
            # bool is a subclass of int in Python; a True/False "source
            # number" is a payload mistake, not a ref, and is refused here
            # rather than silently becoming 1 or 0.
            if isinstance(item, bool) or not isinstance(item, int):
                return PrimeResult.failure(
                    "'sources' must be a list of whole numbers, one per "
                    "source you want to read"
                )
            refs.append(item)

        if self._sources is None:
            # Nothing has been searched in this process, so no number can
            # name a real page. Failing closed is the point: an unbound
            # registry must never be treated as "allow anything".
            return PrimeResult.failure(
                "research_read: no sources are available yet. Run a "
                "research search first, then read the numbers it returned."
            )

        pages: list[dict[str, Any]] = []
        for ref in refs:
            entry = self._sources.get(ref)
            if entry is None:
                # An unknown number is a per-page failure, not a batch
                # failure: the other requested pages are still real and can
                # still be read.
                pages.append({
                    "ref": ref,
                    "url": "",
                    "status": "unknown_source",
                    "error": (
                        f"source {ref} is not available. Use only the "
                        "numbers your most recent research returned."
                    ),
                })
                continue
            pages.append(self._read_one(ref, entry))

        return PrimeResult.success({"pages": pages})

    def _read_one(self, ref: int, entry) -> dict[str, Any]:
        """Fetch one vetted URL and report its outcome honestly."""
        try:
            result = self._fetcher.fetch(entry.url, query=entry.query)
        except Exception as exc:
            # The fetcher is documented never to raise, but a swapped-in
            # fetcher might. One page's crash must not abort the batch.
            return {
                "ref": ref,
                "url": entry.url,
                "status": "fetch_error",
                "error": str(exc),
            }
        page: dict[str, Any] = {
            "ref": ref,
            "url": entry.url,
            "status": result.status,
        }
        if result.status == STATUS_OK:
            # Only a successful page carries text. A failed page saying
            # "text": "" would waste budget and blur the difference between
            # "nothing found" and "could not read".
            page["text"] = result.text
        else:
            page["error"] = result.error
        return page
