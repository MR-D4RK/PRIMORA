"""The self-hosted research backend: SearXNG's JSON search endpoint.

Milestone 3, Phase 2C. Same seam as :class:`PRIMORA.research.provider.
DuckDuckGoResearchProvider`, different transport: one local SearXNG instance
answering ``GET /search?q=...&format=json`` instead of a public API.

Why SearXNG
-----------
It keeps the research capability on infrastructure the operator controls:
no account, no API key, no bill, and no third-party rate limit. The cost is
running a local instance, which is why this module is paired with
``PRIMORA.research.service`` (lifecycle) rather than being responsible for
starting anything itself.

Deliberately absent
-------------------
No construction-time I/O (a provider that probes the network on ``__init__``
would make selection tests and imports side-effectful), no fallback endpoint,
no memory persistence, and no orchestration of the service that backs it.
Endpoint validation lives at the composition root; this class only talks to
the endpoint it is given, and reports transport, HTTP, timeout, and
malformed-response failures honestly rather than fabricating results.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Optional

from PRIMORA.research.provider import (
    DEFAULT_LIMIT,
    DEFAULT_TIMEOUT_S,
    MAX_LIMIT,
    MAX_SNIPPET_CHARS,
    ResearchProvider,
    ResearchResult,
    _clean,
    _host_of,
)

# The only endpoint the composition layer ever accepts for SearXNG. The
# lifecycle manager probes this same fixed local address, so provider and
# service can never disagree about where the instance lives.
SEARXNG_ENDPOINT = "http://127.0.0.1:8888"


class SearXNGResearchProvider(ResearchProvider):
    """Web search through a local SearXNG instance's JSON endpoint.

    Construction stores configuration and performs no I/O; the first network
    touch happens inside :meth:`search`, exactly like the DuckDuckGo backend.
    Failures are raised, never converted into empty results: the research
    capability distinguishes "the search failed" from "the search found
    nothing", and that distinction must survive this transport too.
    """

    name = "searxng"

    SEARCH_PATH = "/search"

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        # Deliberately no environment lookup here: an unvalidated endpoint
        # must never reach the provider. Composition passes the one accepted
        # value; direct construction falls back to the fixed local address.
        self._endpoint = str(endpoint or SEARXNG_ENDPOINT).rstrip("/")
        self._timeout_s = int(timeout_s)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def timeout_s(self) -> int:
        return self._timeout_s

    def search(self, query: str, limit: int = DEFAULT_LIMIT) -> list[ResearchResult]:
        limit = max(1, min(int(limit), MAX_LIMIT))
        params = urllib.parse.urlencode({"q": query, "format": "json"})
        request = urllib.request.Request(
            f"{self._endpoint}{self.SEARCH_PATH}?{params}",
            headers={"User-Agent": "PRIMORA/1.0 (+research capability)"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
            raw = response.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
        return self._to_results(payload, limit)

    def _to_results(
        self, payload: Any, limit: int
    ) -> list[ResearchResult]:
        """Convert a SearXNG JSON payload into ranked sources.

        A payload without a usable ``results`` collection is a malformed
        response, not an empty search: silently returning [] here would tell
        Aurora "no sources exist" when the truth is "the backend answered
        nonsense".
        """
        if not isinstance(payload, dict):
            raise RuntimeError(
                "malformed SearXNG response: expected a JSON object, got "
                f"{type(payload).__name__}"
            )
        if "results" not in payload:
            raise RuntimeError(
                "malformed SearXNG response: missing the 'results' field"
            )
        entries = payload["results"]
        if not isinstance(entries, list):
            raise RuntimeError(
                "malformed SearXNG response: 'results' is not a list"
            )

        results: list[ResearchResult] = []
        seen_urls: set[str] = set()
        for entry in entries:
            if len(results) >= limit:
                break
            if not isinstance(entry, dict):
                continue
            clean_url = str(entry.get("url") or "").strip()
            title_text = _clean(str(entry.get("title") or ""))
            if not clean_url or not title_text:
                continue
            if clean_url in seen_urls:
                continue
            seen_urls.add(clean_url)
            snippet_text = _clean(str(entry.get("content") or ""))[
                :MAX_SNIPPET_CHARS
            ]
            results.append(
                ResearchResult(
                    title=title_text,
                    url=clean_url,
                    snippet=snippet_text,
                    source=_host_of(clean_url),
                )
            )
        return results[:limit]
