"""Numbered references to recent search results, bounded and stable.

Milestone 3, Phase 3 Step 2. When Aurora runs a search, the URLs that come
back become citable sources: each gets a small stable number (its ``ref``),
and that number is what she later passes to ``research_read`` and what she
writes inline as ``[1]``, ``[2]`` when answering. The registry is the single
place that remembers which numbers exist, so a model-supplied URL never has
to be trusted - the number is looked up here, and an unknown number is
refused.

Why a registry instead of letting Aurora pass URLs
--------------------------------------------------
A URL the model invents is an SSRF surface and a provenance lie. A number
the model copies out of a search result is neither: it can only name
something this process actually received from the provider a moment ago.
The registry is what makes that guarantee checkable.

Scoped to one interaction
-------------------------
The registry lives for one research interaction, not for the process. The
cognitive loop's executor calls :meth:`SourceRegistry.clear` when a new
interaction begins, so a number from an earlier, unrelated search can
neither be read nor cited: it is simply not in the set. Within one
interaction numbering is stable and the same URL keeps its first number,
so a multi-search turn still deduplicates correctly.

Bounded on purpose
------------------
Within an interaction the set is capped and oldest-first evicted, so a
long turn cannot grow an unbounded list of remembered URLs. Numbers are
never reused *within* an interaction: once a ref is evicted, citing it
later in the same interaction is a dangling citation, which the loop
corrects rather than silently re-pointing at a different page.

Deliberately absent
-------------------
No persistence across processes, no network access, and no deduplication
policy beyond "same URL keeps its first number". The bound and the
per-interaction reset are the safety mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# The bound. Twenty sources is far more than one turn's search returns (the
# provider caps at 10) and far less than anything that could crowd the
# context window if every number were cited at once.
DEFAULT_MAX_SOURCES = 20


@dataclass(frozen=True)
class SourceEntry:
    """One citable source, as the registry remembers it.

    ``query`` is kept because the fetcher's sentence selection is query-
    aware: re-reading a page with the query that found it yields more
    relevant evidence than re-reading it with nothing.
    """

    ref: int
    url: str
    title: str
    snippet: str
    source: str
    query: str

    def to_dict(self) -> dict:
        return {
            "ref": self.ref,
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "source": self.source,
        }


class SourceRegistry:
    """A bounded, stable mapping from small integer refs to source URLs.

    Single-threaded by design: the cognitive loop runs one capability at a
    time, so no lock is needed and none is taken.
    """

    def __init__(self, max_sources: int = DEFAULT_MAX_SOURCES) -> None:
        if max_sources < 1:
            raise ValueError("max_sources must be at least 1")
        self._max = int(max_sources)
        self._entries: dict[int, SourceEntry] = {}
        self._by_url: dict[str, int] = {}
        self._next_ref = 1

    @property
    def max_sources(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._entries)

    def register(
        self,
        *,
        url: str,
        title: str = "",
        snippet: str = "",
        source: str = "",
        query: str = "",
    ) -> SourceEntry:
        """Remember ``url`` and return its stable entry.

        The same URL always keeps its first number, so re-running a search
        does not silently renumber what Aurora already cited. When the
        registry is full, the lowest-numbered entry is evicted first - the
        oldest research is the least likely to still be what she is talking
        about, and dropping it keeps the bound honest.
        """
        clean_url = str(url or "").strip()
        if not clean_url:
            raise ValueError("a source URL must be non-empty")

        existing = self._by_url.get(clean_url)
        if existing is not None:
            return self._entries[existing]

        ref = self._next_ref
        self._next_ref += 1
        entry = SourceEntry(
            ref=ref,
            url=clean_url,
            title=str(title or ""),
            snippet=str(snippet or ""),
            source=str(source or ""),
            query=str(query or ""),
        )
        self._entries[ref] = entry
        self._by_url[clean_url] = ref
        self._evict_if_needed()
        return entry

    def get(self, ref: int) -> Optional[SourceEntry]:
        """Return the entry for ``ref``, or None when it is not available."""
        if isinstance(ref, bool) or not isinstance(ref, int):
            return None
        return self._entries.get(ref)

    def numbers(self) -> list[int]:
        """The refs currently available, in ascending order."""
        return sorted(self._entries.keys())

    def urls(self) -> list[str]:
        """The URLs currently available, in ref order."""
        return [self._entries[n].url for n in self.numbers()]

    def clear(self) -> None:
        """Forget everything; the next source is numbered 1 again.

        Called by the cognitive loop at the start of each interaction, so a
        number only ever names a page the current turn's search produced.
        """
        self._entries.clear()
        self._by_url.clear()
        self._next_ref = 1

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._max:
            oldest = min(self._entries)
            evicted = self._entries.pop(oldest)
            self._by_url.pop(evicted.url, None)
