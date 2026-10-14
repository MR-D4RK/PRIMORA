"""The research seam: how external information is obtained.

Milestone 3. One narrow, replaceable interface:

    provider.search(query, limit) -> list[ResearchResult]

Why this is separate from ``PRIMORA.prime.providers.Provider``
--------------------------------------------------------------
That abstraction exists to *invoke a language model* for a capability. Web
research is a different concern with a different shape: one search method in,
ranked sources out. Folding it into ``Provider.invoke(capability, payload)``
would make every future research backend look like a model call, and would
drag HTTP concerns into the object that also talks to the local model.

The separation is what makes the provider replaceable. Aurora's capability
interface is ``research(query)`` and does not change when the backend does.

Deliberately absent
-------------------
No page fetching, no crawling, no recursion, no autonomous multi-step
investigation, and no paid service. This is a single search returning
snippets. A future full research agent can grow a richer interface behind the
same seam without changing anything above it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

# Hard ceilings. A capability that can ask the network for anything is not a
# capability, it is an outage.
DEFAULT_LIMIT = 5
MAX_LIMIT = 10
DEFAULT_TIMEOUT_S = 10
MAX_SNIPPET_CHARS = 600


@dataclass(frozen=True)
class ResearchResult:
    """One source, as Aurora sees it.

    ``url`` and ``source`` are included deliberately: Aurora is told where an
    answer came from, so she can say so or judge whether to trust it. This is
    not provenance for an audit log; it is context the model needs.
    """

    title: str
    url: str
    snippet: str
    source: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
        }


class ResearchProvider(ABC):
    """A replaceable web-search backend.

    Implement this to swap search engines. Nothing above this line changes.
    """

    name = "research"

    @abstractmethod
    def search(self, query: str, limit: int = DEFAULT_LIMIT) -> list[ResearchResult]:
        """Return ranked sources for ``query``, or raise on transport failure.

        Raising is acceptable and expected for network or backend errors; the
        capability converts it into a reported failure rather than a crash.
        An empty list means "the search succeeded and found nothing", which is
        a different outcome and is reported differently.
        """


class UnconfiguredResearchProvider(ResearchProvider):
    """The default: no backend wired up.

    Chosen over a silent live default on purpose. A capability that reaches
    the network without being configured is a surprise in a test suite and a
    privacy surprise in a running system. Failing closed makes the absence
    obvious, and the message says exactly what to do about it.
    """

    name = "unconfigured"

    def search(self, query: str, limit: int = DEFAULT_LIMIT) -> list[ResearchResult]:
        raise RuntimeError(
            "No research provider is configured. Pass research_provider=... "
            "to Prime, for example DuckDuckGoResearchProvider()."
        )


class DuckDuckGoResearchProvider(ResearchProvider):
    """Free, keyless web search via DuckDuckGo's public JSON endpoint.

    Chosen because it requires no account, no API key, and no payment. The
    instruction was not to introduce a paid service, so this is the default
    suggestion rather than anything that bills or needs a secret.

    This is the *replaceable* part. Swapping in a commercial or self-hosted
    backend means implementing :class:`ResearchProvider` and passing it to
    Prime; no capability, loop, or Aurora code changes.
    """

    name = "duckduckgo"

    ENDPOINT = "https://api.duckduckgo.com/"

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._endpoint = endpoint or os.environ.get(
            "PRIMORA_RESEARCH_ENDPOINT", self.ENDPOINT
        )
        self._timeout_s = int(timeout_s)

    def search(self, query: str, limit: int = DEFAULT_LIMIT) -> list[ResearchResult]:
        limit = max(1, min(int(limit), MAX_LIMIT))
        params = urllib.parse.urlencode(
            {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            }
        )
        request = urllib.request.Request(
            f"{self._endpoint}?{params}",
            headers={"User-Agent": "PRIMORA/1.0 (+research capability)"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
            raw = response.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
        return self._to_results(payload, limit)

    def _to_results(
        self, payload: dict[str, Any], limit: int
    ) -> list[ResearchResult]:
        results: list[ResearchResult] = []
        seen_urls: set[str] = set()

        def add(title: Any, url: Any, snippet: Any) -> None:
            if len(results) >= limit:
                return
            clean_url = str(url or "").strip()
            title_text = _clean(str(title or ""))
            snippet_text = _clean(str(snippet or ""))[:MAX_SNIPPET_CHARS]
            if not clean_url or not title_text:
                return
            if clean_url in seen_urls:
                return
            seen_urls.add(clean_url)
            results.append(
                ResearchResult(
                    title=title_text,
                    url=clean_url,
                    snippet=snippet_text,
                    source=_host_of(clean_url),
                )
            )

        # The Abstract is the single best direct answer, when one exists.
        add(
            payload.get("Heading"),
            payload.get("AbstractURL"),
            payload.get("AbstractText"),
        )
        for group in ("RelatedTopics", "Results"):
            for topic in payload.get(group) or []:
                if isinstance(topic, dict):
                    add(
                        topic.get("Text"),
                        topic.get("FirstURL"),
                        topic.get("Text"),
                    )
                elif isinstance(topic, list):
                    # DuckDuckGo nests RelatedTopics one level deep.
                    for nested in topic:
                        if isinstance(nested, dict):
                            add(
                                nested.get("Text"),
                                nested.get("FirstURL"),
                                nested.get("Text"),
                            )
        return results[:limit]


def _clean(text: str) -> str:
    return " ".join(text.split())


def _host_of(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc
    except ValueError:
        return ""
