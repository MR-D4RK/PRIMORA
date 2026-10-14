"""Phase 2C: the SearXNG research backend, parsed without the network.

Same properties as the DuckDuckGo parsing tests: recorded payloads in,
:class:`ResearchResult` objects out, and every failure mode (transport, HTTP,
timeout, malformed response) raised honestly instead of laundered into an
empty result. Construction must never touch the network, because provider
selection happens during startup.
"""

import json
import os
import sys
import urllib.error
import urllib.parse

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.contract import RequestStatus
from PRIMORA.research import (
    MAX_LIMIT,
    ResearchProvider,
    ResearchResult,
    SEARXNG_ENDPOINT,
    SearXNGResearchProvider,
)
from PRIMORA.research.provider import DEFAULT_TIMEOUT_S, MAX_SNIPPET_CHARS
from PRIMORA.research.searxng import SearXNGResearchProvider as FromModule
from PRIMORA.specialists.tools.research import ResearchSpecialist


PAYLOAD = {
    "query": "python 3.13",
    "number_of_results": 3,
    "results": [
        {
            "url": "https://python.org/downloads/",
            "title": "Python 3.13 released",
            "content": "Python 3.13 is the latest stable release.",
            "engine": "duckduckgo",
        },
        {
            "url": "https://docs.python.org/3/",
            "title": "Python documentation",
            "content": "Reference and tutorial documentation.",
        },
        {
            "url": "https://python.org/downloads/",
            "title": "duplicate url",
            "content": "must be deduplicated",
        },
        {"url": "", "title": "no url", "content": "skipped"},
        {"url": "https://no-title.test/", "content": "skipped"},
        "not a dict",
        {
            "url": "https://long.test/",
            "title": "long snippet",
            "content": "x" * 5000,
        },
    ],
}


class _FakeResponse:
    def __init__(self, body):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self._body = body

    def read(self, *args, **kwargs):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Recorder:
    """Stands in for ``urllib.request.urlopen`` and records every call."""

    def __init__(self, body=None, error=None):
        self.calls = []
        self._body = PAYLOAD if body is None else body
        self._error = error

    def __call__(self, request, timeout=None, **kwargs):
        self.calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "timeout": timeout,
            }
        )
        if self._error is not None:
            raise self._error
        return _FakeResponse(self._body)


def _search(monkeypatch, query="python", limit=5, **kwargs):
    recorder = _Recorder(**kwargs)
    monkeypatch.setattr("urllib.request.urlopen", recorder)
    provider = SearXNGResearchProvider()
    results = provider.search(query, limit=limit)
    return provider, recorder, results


# ======================================================================
# Construction: configuration only, no I/O
# ======================================================================


class TestConstructionIsIOFree:
    def test_construction_performs_no_io(self, monkeypatch):
        def _no_io(*args, **kwargs):
            raise AssertionError("provider construction reached the network")

        monkeypatch.setattr("urllib.request.urlopen", _no_io)
        provider = SearXNGResearchProvider()
        assert provider.endpoint == SEARXNG_ENDPOINT

    def test_endpoint_defaults_to_the_fixed_local_address(self):
        assert SearXNGResearchProvider().endpoint == "http://127.0.0.1:8888"

    def test_an_explicit_endpoint_is_honoured(self):
        provider = SearXNGResearchProvider(endpoint="http://127.0.0.1:8888/")
        assert provider.endpoint == "http://127.0.0.1:8888"

    def test_timeout_is_configurable(self):
        provider = SearXNGResearchProvider(timeout_s=3)
        assert provider.timeout_s == 3

    def test_it_is_a_research_provider(self):
        provider = SearXNGResearchProvider()
        assert isinstance(provider, ResearchProvider)
        assert provider.name == "searxng"
        assert FromModule is SearXNGResearchProvider


# ======================================================================
# The request: JSON endpoint, URL-encoded parameters
# ======================================================================


class TestRequestShape:
    def test_it_queries_the_local_json_endpoint(self, monkeypatch):
        _, recorder, _ = _search(monkeypatch, query="latest python")
        url = recorder.calls[0]["url"]
        assert url.startswith(f"{SEARXNG_ENDPOINT}/search?")
        parsed = urllib.parse.urlparse(url)
        assert parsed.netloc == "127.0.0.1:8888"

    def test_query_parameters_are_url_encoded(self, monkeypatch):
        _, recorder, _ = _search(monkeypatch, query="python & torch ? 100%")
        url = recorder.calls[0]["url"]
        query = urllib.parse.urlparse(url).query
        assert " " not in query
        # Everything up to the separator is the encoded q value itself.
        assert "&" not in query.split("&format=")[0]
        params = urllib.parse.parse_qs(query)
        assert params["q"] == ["python & torch ? 100%"]
        assert params["format"] == ["json"]

    def test_it_is_a_get_request_with_a_timeout(self, monkeypatch):
        _, recorder, _ = _search(monkeypatch)
        assert recorder.calls[0]["method"] == "GET"
        assert recorder.calls[0]["timeout"] == DEFAULT_TIMEOUT_S

    def test_the_configured_timeout_is_forwarded(self, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr("urllib.request.urlopen", recorder)
        SearXNGResearchProvider(timeout_s=7).search("q")
        assert recorder.calls[0]["timeout"] == 7


# ======================================================================
# Parsing: valid payload in, ResearchResult objects out
# ======================================================================


class TestParsing:
    def test_valid_results_become_research_results(self, monkeypatch):
        _, _, results = _search(monkeypatch, limit=5)
        assert all(isinstance(r, ResearchResult) for r in results)
        assert [r.title for r in results] == [
            "Python 3.13 released",
            "Python documentation",
            "long snippet",
        ]
        assert results[0].url == "https://python.org/downloads/"
        assert results[0].snippet == "Python 3.13 is the latest stable release."

    def test_the_host_is_reported_as_the_source(self, monkeypatch):
        _, _, results = _search(monkeypatch, limit=5)
        assert [r.source for r in results] == [
            "python.org",
            "docs.python.org",
            "long.test",
        ]

    def test_urls_are_deduplicated(self, monkeypatch):
        _, _, results = _search(monkeypatch, limit=10)
        urls = [r.url for r in results]
        assert len(urls) == len(set(urls))

    def test_malformed_entries_are_skipped_not_fatal(self, monkeypatch):
        _, _, results = _search(monkeypatch, limit=10)
        assert all(r.url.startswith("https://") for r in results)

    def test_empty_results_are_an_empty_list(self, monkeypatch):
        recorder = _Recorder(body={"results": [], "query": "nothing"})
        monkeypatch.setattr("urllib.request.urlopen", recorder)
        assert SearXNGResearchProvider().search("nothing") == []

    def test_the_limit_is_clamped_to_the_maximum(self, monkeypatch):
        many = {
            "results": [
                {"url": f"https://site{i}.test/", "title": f"T{i}",
                 "content": "c"}
                for i in range(50)
            ]
        }
        recorder = _Recorder(body=many)
        monkeypatch.setattr("urllib.request.urlopen", recorder)
        results = SearXNGResearchProvider().search("q", limit=9999)
        assert len(results) == MAX_LIMIT

    def test_a_limit_below_one_still_returns_a_result(self, monkeypatch):
        _, _, results = _search(monkeypatch, limit=0)
        assert len(results) == 1

    def test_huge_snippets_are_truncated(self, monkeypatch):
        _, _, results = _search(monkeypatch, limit=5)
        long_one = next(r for r in results if r.url == "https://long.test/")
        assert len(long_one.snippet) <= MAX_SNIPPET_CHARS


# ======================================================================
# Failures: reported honestly, never fabricated
# ======================================================================


class TestHonestFailures:
    def test_malformed_json_raises(self, monkeypatch):
        recorder = _Recorder(body="<html>502 Bad Gateway</html>")
        monkeypatch.setattr("urllib.request.urlopen", recorder)
        with pytest.raises(json.JSONDecodeError):
            SearXNGResearchProvider().search("q")

    def test_a_non_object_payload_raises(self, monkeypatch):
        recorder = _Recorder(body="[1, 2, 3]")
        monkeypatch.setattr("urllib.request.urlopen", recorder)
        with pytest.raises(RuntimeError, match="malformed SearXNG response"):
            SearXNGResearchProvider().search("q")

    def test_a_missing_results_field_raises(self, monkeypatch):
        recorder = _Recorder(body={"error": "backend exploded"})
        monkeypatch.setattr("urllib.request.urlopen", recorder)
        with pytest.raises(RuntimeError, match="malformed SearXNG response"):
            SearXNGResearchProvider().search("q")

    def test_a_non_list_results_field_raises(self, monkeypatch):
        recorder = _Recorder(body={"results": {"oops": True}})
        monkeypatch.setattr("urllib.request.urlopen", recorder)
        with pytest.raises(RuntimeError, match="malformed SearXNG response"):
            SearXNGResearchProvider().search("q")

    def test_transport_errors_propagate(self, monkeypatch):
        recorder = _Recorder(error=urllib.error.URLError("connection refused"))
        monkeypatch.setattr("urllib.request.urlopen", recorder)
        provider = SearXNGResearchProvider()
        with pytest.raises(urllib.error.URLError):
            provider.search("q")

    def test_http_errors_propagate(self, monkeypatch):
        error = urllib.error.HTTPError(
            f"{SEARXNG_ENDPOINT}/search", 503, "Service Unavailable", None, None
        )
        recorder = _Recorder(error=error)
        monkeypatch.setattr("urllib.request.urlopen", recorder)
        with pytest.raises(urllib.error.HTTPError):
            SearXNGResearchProvider().search("q")

    def test_a_failure_is_never_reported_as_empty_results(self, monkeypatch):
        """An empty list means the search worked; a failure must raise."""
        recorder = _Recorder(error=urllib.error.URLError("timed out"))
        monkeypatch.setattr("urllib.request.urlopen", recorder)
        with pytest.raises(urllib.error.URLError):
            SearXNGResearchProvider().search("q")


# ======================================================================
# Wiring: it slots behind the existing capability unchanged
# ======================================================================


class TestCapabilityWiring:
    def test_the_specialist_uses_it_without_any_capability_change(
            self, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr("urllib.request.urlopen", recorder)
        specialist = ResearchSpecialist(SearXNGResearchProvider())
        out = specialist.handle({"query": "latest python", "limit": 2})
        assert out.status is RequestStatus.SUCCESS
        assert len(out.output["results"]) == 2
        assert recorder.calls, "the provider was never consulted"

    def test_a_dead_instance_is_a_reported_failure_not_a_crash(
            self, monkeypatch):
        recorder = _Recorder(error=urllib.error.URLError("refused"))
        monkeypatch.setattr("urllib.request.urlopen", recorder)
        specialist = ResearchSpecialist(SearXNGResearchProvider())
        out = specialist.handle({"query": "q"})
        assert out.status is RequestStatus.FAILURE
        assert "research_failed" in out.error
        assert "refused" in out.error

    def test_the_provider_module_never_manages_the_service(self):
        """Lifecycle belongs to PRIMORA.research.service, not this client."""
        import ast
        path = os.path.join("PRIMORA", "research", "searxng.py")
        src = open(path, encoding="utf-8").read()
        for banned in ("subprocess", "docker", "podman", "atexit", "shutil"):
            assert banned not in src.lower(), f"lifecycle leaked: {banned}"
        tree = ast.parse(src)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                imported.update(
                    (n.name or "").split(".")[0] for n in node.names
                )
        assert "service" not in imported
