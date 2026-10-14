"""Milestone 3: the second capability, ``research()``.

The properties that matter:

1. Aurora asks a plain-language question and gets ranked sources back.
2. The provider seam is replaceable, and no transport detail leaks upward.
3. The capability is separate from the Cognitive Loop, which is unmodified.
4. Failure, empty results, and unbounded requests are all handled explicitly.
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeRequest, PrimeResult, RequestStatus
from PRIMORA.prime.providers import Provider
from PRIMORA.research import (
    DuckDuckGoResearchProvider,
    ResearchProvider,
    ResearchResult,
    SearXNGResearchProvider,
    UnconfiguredResearchProvider,
)
from PRIMORA.specialists.tools.research import ResearchSpecialist


def _request(action, payload):
    return (
        "<prime_request>\n"
        + json.dumps({"action": action, "payload": payload,
                      "context": None, "metadata": None})
        + "\n</prime_request>"
    )


class _FakeProvider(ResearchProvider):
    """Records what it was asked and returns a scripted answer."""

    name = "fake"

    def __init__(self, results=None, error=None):
        self.calls = []
        self._results = results if results is not None else [
            ResearchResult(
                title="Python 3.13 released",
                url="https://python.org/downloads/",
                snippet="Python 3.13 is the latest stable release.",
                source="python.org",
            )
        ]
        self._error = error

    def search(self, query, limit=5):
        self.calls.append({"query": query, "limit": limit})
        if self._error:
            raise self._error
        return list(self._results)[:limit]


class _NoNetworkProvider(ResearchProvider):
    """Fails the test if anything reaches a backend."""

    name = "none"

    def search(self, query, limit=5):
        raise AssertionError("the network was reached in a unit test")


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.success({"value": 4})


class _Scripted(Brain):
    def __init__(self, *replies):
        self.replies = list(replies) or ["done"]
        self.calls = []

    def generate(self, messages, max_output_tokens=None):
        self.calls.append([dict(m) for m in messages])
        r = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(r, BrainResult):
            return r
        return BrainResult.success(r, "scripted")

    def get_model_name(self):
        return "scripted"

    def count_tokens(self, messages):
        return sum(len(m.get("content", "").split()) + 2 for m in messages)


_DB_SEQ = [0]


def _prime(research_provider=None):
    """A Prime on an isolated database.

    ``Prime()`` defaults to the *production* ContextManager, and
    ``Aurora.respond`` persists the conversation snapshot. A test that builds
    one of these and then calls respond() writes to production memory, so
    every Prime made here is redirected to a temp store before use.
    """
    _DB_SEQ[0] += 1
    prime = Prime(_Provider(), research_provider=research_provider)
    _isolate(prime, f"r{_DB_SEQ[0]}")
    return prime


def _isolate(prime, name):
    """Point a Prime's Mind at a throwaway database."""
    prime._context = prime._context.__class__(
        db_path=os.path.join(tempfile.mkdtemp(prefix="research_"), name + ".db"))
    prime._mind = prime._mind.__class__(prime._context)
    recall = prime.registry.get_specialist("recall")
    if recall is not None:
        recall.bind(prime._mind)
    return prime


def _research_src():
    """Executable source of the research capability, docstrings removed.

    Scope tests scan for banned words. Scanning raw text would match the
    docstring that explains *why* crawling and recursion are absent, and fail
    for exactly the wrong reason.
    """
    import ast
    path = ResearchSpecialist.__module__.replace(".", os.sep) + ".py"
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree).lower()


def _aurora(prime, brain):
    cfg = AuroraConfig(model_path="m", adapter_path="a")
    return Aurora(prime, brain=brain, config=cfg)


# ======================================================================
# 1. The capability exists and is well-formed
# ======================================================================


class TestCapabilityContract:
    def test_research_is_registered(self):
        assert _prime().registry.has("research")

    def test_manifest_shape(self):
        spec = _prime().registry.get("research")
        assert spec.required_fields == ["query"]
        assert spec.optional_fields == ["limit"]
        assert spec.payload_schema["additionalProperties"] is False

    def test_runs_through_prime(self):
        out = _prime(_FakeProvider()).handle(
            PrimeRequest("research", {"query": "latest Python release"})
        )
        assert out.status is RequestStatus.SUCCESS
        assert out.output["results"]

    def test_works_without_a_provider_configured(self):
        """Fails closed, never silently reaching the network."""
        out = _prime().handle(PrimeRequest("research", {"query": "anything"}))
        assert out.status is RequestStatus.FAILURE
        assert "research provider" in out.error

    @pytest.mark.parametrize("bad", [None, "", "   ", 5, [], {}])
    def test_bad_query_rejected(self, bad):
        out = _prime(_FakeProvider()).handle(PrimeRequest("research", bad))
        assert out.status is RequestStatus.FAILURE

    def test_limit_is_bounded(self):
        p = _FakeProvider()
        out = _prime(p).handle(
            PrimeRequest("research", {"query": "python", "limit": 9999})
        )
        assert out.status is RequestStatus.SUCCESS
        assert p.calls[0]["limit"] == 10

    @pytest.mark.parametrize("bad", ["5", 2.5, True, [1]])
    def test_bad_limit_rejected(self, bad):
        out = _prime(_FakeProvider()).handle(
            PrimeRequest("research", {"query": "x", "limit": bad})
        )
        assert out.status is RequestStatus.FAILURE


# ======================================================================
# 2. The provider seam is replaceable
# ======================================================================


class TestProviderIsReplaceable:
    def test_default_provider_never_reaches_network(self):
        provider = UnconfiguredResearchProvider()
        with pytest.raises(RuntimeError):
            provider.search("anything")

    def test_a_second_backend_needs_no_capability_change(self):
        """Swapping the backend is one constructor argument."""
        first = _FakeProvider([ResearchResult("A", "https://a.test", "a")])
        second = _FakeProvider([ResearchResult("B", "https://b.test", "b")])

        out_a = _prime(first).handle(PrimeRequest("research", {"query": "q"}))
        out_b = _prime(second).handle(PrimeRequest("research", {"query": "q"}))
        assert out_a.output["results"][0]["title"] == "A"
        assert out_b.output["results"][0]["title"] == "B"

    def test_provider_can_be_swapped_after_construction(self):
        prime = _prime(_FakeProvider())
        prime.registry.get_specialist("research").set_provider(
            _FakeProvider([ResearchResult("C", "https://c.test", "c")])
        )
        out = prime.handle(PrimeRequest("research", {"query": "q"}))
        assert out.output["results"][0]["title"] == "C"

    def test_specialist_rejects_a_none_provider(self):
        with pytest.raises(ValueError):
            ResearchSpecialist().set_provider(None)

    def test_capability_does_not_know_transport(self):
        """Aurora's capability must contain no HTTP or backend detail."""
        src = open(ResearchSpecialist.__module__.replace(".", os.sep) + ".py",
                   encoding="utf-8").read()
        for banned in ("urllib", "http", "requests", "endpoint", "api_key",
                       "duckduckgo", "json.loads", "urlopen"):
            assert banned not in src, f"transport detail leaked: {banned}"

    def test_manifest_does_not_expose_transport(self):
        desc = _prime().registry.get("research").description.lower()
        for banned in ("http", "api", "endpoint", "url", "request parameter"):
            assert banned not in desc, f"manifest leaks: {banned}"

    def test_specialist_depends_only_on_the_abstract_base(self):
        import ast
        path = ResearchSpecialist.__module__.replace(".", os.sep) + ".py"
        tree = ast.parse(open(path, encoding="utf-8").read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    imported.add(a.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    imported.add((a.name or "").split(".")[0])
        # It must not import any concrete backend.
        assert "DuckDuckGoResearchProvider" not in imported
        assert "requests" not in imported


# ======================================================================
# 3. Aurora can ask, and results reach her next segment
# ======================================================================


class TestInsideTheLoop:
    def test_aurora_can_request_research_during_cognition(self):
        p = _FakeProvider()
        brain = _Scripted(
            _request("research", {"query": "latest Python release"}),
            "Python 3.13 is the latest stable release.",
        )
        a = _aurora(_prime(p), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "check the release")
        result = a.respond(
            [{"role": "user", "content": "what is the latest Python?"}], "s"
        )
        assert result.content.startswith("Python 3.13")
        assert len(brain.calls) == 2

    def test_request_reaches_the_capability(self):
        p = _FakeProvider()
        brain = _Scripted(
            _request("research", {"query": "latest Python release"}), "done")
        a = _aurora(_prime(p), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "latest Python?"}], "s")
        assert [name for name, _ in a.loop.executor.calls] == ["research"]

    def test_provider_is_called_with_the_exact_query(self):
        p = _FakeProvider()
        brain = _Scripted(
            _request("research", {"query": "latest Python release"}), "done")
        a = _aurora(_prime(p), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "latest Python?"}], "s")
        assert len(p.calls) == 1
        assert p.calls[0]["query"] == "latest Python release"

    def test_results_reach_the_next_segment(self):
        brain = _Scripted(
            _request("research", {"query": "latest Python release"}), "done")
        a = _aurora(_prime(_FakeProvider()), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "latest Python?"}], "s")
        second = " ".join(m["content"] for m in brain.calls[1])
        assert "Python 3.13" in second
        assert "python.org" in second

    def test_source_attribution_reaches_aurora(self):
        """She can say where the information came from."""
        brain = _Scripted(
            _request("research", {"query": "latest Python release"}), "done")
        a = _aurora(_prime(_FakeProvider()), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "latest Python?"}], "s")
        tool = [m for m in brain.calls[1] if m["role"] == "tool"][0]["content"]
        parsed = json.loads(tool)["output"]["results"][0]
        assert parsed["url"] == "https://python.org/downloads/"
        assert parsed["source"] == "python.org"
        assert parsed["title"]

    def test_aurora_is_told_research_exists(self):
        brain = _Scripted("done")
        a = _aurora(_prime(_FakeProvider()), brain)
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "hi"}], "s")
        system = brain.calls[0][0]["content"]
        assert "research" in system
        assert "Look up current or external information" in system

    def test_answer_can_be_based_only_on_research(self):
        brain = _Scripted(
            _request("research", {"query": "current aurora release"}),
            "According to the source, the current release is 2.1.",
        )
        a = _aurora(_prime(_FakeProvider()), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "t")
        result = a.respond(
            [{"role": "user", "content": "what is the current release?"}], "s")
        assert "2.1" in result.content


# ======================================================================
# 4. Failure and empty results
# ======================================================================


class TestFailureHandling:
    def test_provider_exception_becomes_a_reported_failure(self):
        p = _FakeProvider(error=RuntimeError("connection refused"))
        out = _prime(p).handle(PrimeRequest("research", {"query": "q"}))
        assert out.status is RequestStatus.FAILURE
        assert "research_failed" in out.error
        assert "connection refused" in out.error

    def test_provider_failure_does_not_break_the_loop(self):
        brain = _Scripted(
            _request("research", {"query": "q"}),
            "I could not reach the internet to check that.",
        )
        a = _aurora(_prime(_FakeProvider(error=RuntimeError("dns"))), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "t")
        result = a.respond([{"role": "user", "content": "look it up"}], "s")
        assert result.success
        assert result.content.startswith("I could not reach")
        tool = [m for m in brain.calls[1] if m["role"] == "tool"][0]["content"]
        assert "failure" in tool

    def test_empty_results_is_a_failure_not_a_success(self):
        out = _prime(_FakeProvider(results=[])).handle(
            PrimeRequest("research", {"query": "q"})
        )
        assert out.status is RequestStatus.FAILURE
        assert "no sources" in out.error.lower()

    def test_empty_results_reaches_aurora_cleanly(self):
        brain = _Scripted(
            _request("research", {"query": "q"}),
            "I searched and found no sources on that.",
        )
        a = _aurora(_prime(_FakeProvider(results=[])), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "t")
        result = a.respond([{"role": "user", "content": "look it up"}], "s")
        assert result.success
        assert result.content.startswith("I searched")
        # The tool result Aurora saw must be a failure, not an empty success,
        # so she cannot treat "no sources" as verified grounding.
        tool = [m for m in brain.calls[1] if m["role"] == "tool"][0]["content"]
        assert '"status": "failure"' in tool

    def test_provider_crash_never_escapes_as_an_exception(self):
        class Explode(ResearchProvider):
            name = "boom"

            def search(self, query, limit=5):
                raise MemoryError("something catastrophic")

        out = _prime(Explode()).handle(PrimeRequest("research", {"query": "q"}))
        assert out.status is RequestStatus.FAILURE


# ======================================================================
# 5. Bounded by the existing loop protections
# ======================================================================


class TestBoundedByLoop:
    def test_research_is_bounded_by_max_cycles(self):
        p = _FakeProvider()
        brain = _Scripted(_request("research", {"query": "1"}),
                          _request("research", {"query": "2"}),
                          _request("research", {"query": "3"}), "done")
        a = _aurora(_prime(p), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "go"}], "s")
        assert len(brain.calls) == 2
        assert len(p.calls) <= 2

    def test_repeated_identical_research_is_blocked(self):
        """The M2 repeat guard applies to research like any capability."""
        p = _FakeProvider()
        brain = _Scripted(_request("research", {"query": "same"}))
        a = _aurora(_prime(p), brain)
        a.loop.max_cycles = 20
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "go"}], "s")
        assert a.loop.last_trace.stop_reason == "repeated_request"
        assert len(p.calls) == 2, "allowed twice, then blocked"

    def test_result_size_is_capped(self):
        many = [ResearchResult(f"T{i}", f"https://x.test/{i}", "s")
                for i in range(50)]
        p = _FakeProvider(results=many)
        brain = _Scripted(
            _request("research", {"query": "q", "limit": 3}), "done")
        a = _aurora(_prime(p), brain)
        a.loop.max_cycles = 2
        a.mind.begin_task("s", "t")
        a.respond([{"role": "user", "content": "go"}], "s")
        tool = [m for m in brain.calls[1] if m["role"] == "tool"][0]["content"]
        assert len(json.loads(tool)["output"]["results"]) == 3

    def test_research_does_not_persist_to_memory(self):
        import sqlite3
        prime = _prime(_FakeProvider())
        path = str(prime._context._get_store()._db_path)

        def count():
            with sqlite3.connect(path) as con:
                return con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

        before = count()
        prime.handle(PrimeRequest("research", {"query": "q"}))
        assert count() == before


class TestNeverTouchesProductionMemory:
    """Guard: no test in this file may reach production memory.

    Added after this file wrote 22 rows into the production database, because
    ``Prime()`` defaults to the production ContextManager and ``respond``
    persists. Every Prime used above goes through ``_prime``, which isolates
    it; this proves the whole module is clean.
    """

    def test_research_tests_did_not_write_to_production(self):
        """Content equality via production_digest(), not row emptiness.

        A used install legitimately holds real memories (MEMORY_AUDIT
        C2), so "did not write" means the digest is unchanged, never
        that the tables are empty.
        """
        from PRIMORA.prime.context.isolation import (
            harness_session,
            production_db_path,
            production_digest,
        )
        if not production_db_path().exists():
            pytest.skip("no production database")
        before = production_digest()
        with harness_session("c2_research_") as session:
            store = session.context_manager()._get_store()
            assert store.fetch_states(limit=5) == []
        assert production_digest() == before, (
            "production memory changed during this test"
        )

    def test_only_one_prime_is_constructed_and_it_is_isolated(self):
        """Structural check: the single Prime() must be isolated at birth."""
        import ast
        tree = ast.parse(open(os.path.abspath(__file__),
                              encoding="utf-8").read())
        helpers = [
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("_prime", "_isolate")
        ]
        constructions = [
            n for h in helpers for n in ast.walk(h)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "Prime"
        ]
        assert len(constructions) == 1, "construct Prime() in exactly one place"

        prime_helper = next(h for h in helpers if h.name == "_prime")
        # The very next statement after Prime(...) must redirect its store.
        for index, stmt in enumerate(prime_helper.body):
            if (isinstance(stmt, ast.Assign) and
                    any(getattr(n, "id", "") == "Prime" for n in ast.walk(stmt))):
                following = prime_helper.body[index + 1]
                assert "_isolate" in ast.unparse(following), (
                    "Prime() is not isolated before use"
                )
                return
        raise AssertionError("no Prime assignment found in _prime")


# ======================================================================
# 6. The real provider: parsing, without the network
# ======================================================================


class TestDuckDuckGoParsing:
    """Parsing is unit-tested with recorded payloads; no live calls."""

    PAYLOAD = {
        "Heading": "Python (programming language)",
        "AbstractURL": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "AbstractText": "Python is a high-level programming language.",
        "RelatedTopics": [
            {"Text": "Python 3.13 released", "FirstURL": "https://python.org/313/"},
            {"Text": "Python docs", "FirstURL": "https://docs.python.org/3/"},
        ],
        "Results": [
            {"Text": "Python.org", "FirstURL": "https://python.org/"}
        ],
    }

    def test_parses_abstract_and_topics(self):
        p = DuckDuckGoResearchProvider()
        results = p._to_results(self.PAYLOAD, 5)
        titles = [r.title for r in results]
        assert "Python (programming language)" in titles
        assert "Python 3.13 released" in titles

    def test_respects_the_limit(self):
        p = DuckDuckGoResearchProvider()
        assert len(p._to_results(self.PAYLOAD, 2)) == 2

    def test_deduplicates_urls(self):
        p = DuckDuckGoResearchProvider()
        results = p._to_results(self.PAYLOAD, 10)
        urls = [r.url for r in results]
        assert len(urls) == len(set(urls))

    def test_extracts_the_host_as_source(self):
        p = DuckDuckGoResearchProvider()
        results = p._to_results(self.PAYLOAD, 5)
        assert all("." in r.source for r in results)

    def test_handles_an_empty_payload(self):
        p = DuckDuckGoResearchProvider()
        assert p._to_results({}, 5) == []

    def test_handles_malformed_entries(self):
        p = DuckDuckGoResearchProvider()
        payload = {
            "RelatedTopics": [
                {"Text": "no url", "FirstURL": ""},
                {"not": "a dict"},
                "a bare string",
            ]
        }
        assert p._to_results(payload, 5) == []

    def test_truncates_a_huge_snippet(self):
        p = DuckDuckGoResearchProvider()
        payload = {"RelatedTopics": [
            {"Text": "x" * 5000, "FirstURL": "https://x.test/"}]}
        assert len(p._to_results(payload, 5)[0].snippet) <= 600

    def test_uses_a_timeout(self):
        p = DuckDuckGoResearchProvider(timeout_s=3)
        assert p._timeout_s == 3


# ======================================================================
# 7. Scope: only research was added
# ======================================================================


class TestScope:
    def test_capabilities_registered_so_far(self):
        """Milestone 3 adds capabilities one at a time."""
        assert sorted(_prime().registry.get_names()) == [
            "calculate", "consult_specialist", "learn", "recall", "research",
            "research_read", "use_tool"]

    def test_cognitive_loop_is_unmodified(self):
        """research() required no loop changes."""
        import ast
        from PRIMORA.aurora import loop as loopmod
        path = loopmod.__file__
        tree = ast.parse(open(path, encoding="utf-8").read())
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "research" not in called
        assert "search" not in called

    def test_no_multi_step_or_agent_loop(self):
        src = _research_src()
        for banned in ("crawl", "spider", "recurs", "scrape",
                       "subagent", "autonomous", "while true"):
            assert banned not in src, banned

    def test_no_page_fetching(self):
        """Search only: it must not open result pages."""
        src = open(ResearchSpecialist.__module__.replace(".", os.sep) + ".py",
                   encoding="utf-8").read().lower()
        assert "fetch_page" not in src
        assert "read_page" not in src

    def test_no_new_paid_service(self):
        for banned in ("serpapi", "bing", "google", "openai", "anthropic",
                       "api_key", "apikey"):
            src = open(ResearchSpecialist.__module__.replace(".", os.sep)
                       + ".py", encoding="utf-8").read().lower()
            assert banned not in src, banned

    def test_mind_is_untouched_by_research(self):
        """Recall and research are capabilities, not a Mind concern.

        The Mind boundary must stay free of network and search concerns, or
        M6 and memory would inherit them.
        """
        src = open("PRIMORA" + os.sep + "prime" + os.sep + "context" +
                   os.sep + "mind.py", encoding="utf-8").read().lower()
        for banned in ("research", "urllib", "http", "web search"):
            assert banned not in src, banned

    def test_memory_mechanisms_have_no_http(self):
        for name in ("manager.py", "hybrid.py", "store.py", "semantic.py"):
            src = open("PRIMORA" + os.sep + "prime" + os.sep + "context" +
                       os.sep + name, encoding="utf-8").read().lower()
            for banned in ("urllib", "http://", "https://api"):
                assert banned not in src, f"{name}: {banned}"


# ======================================================================
# 8. No network in the unit suite
# ======================================================================


class TestNoNetworkAccess:
    def test_default_prime_never_touches_the_network(self):
        """Every test above used this Prime; none of them reached out."""
        out = _prime().handle(PrimeRequest("research", {"query": "python"}))
        assert out.status is RequestStatus.FAILURE
        assert "research provider" in out.error

    def test_the_fake_provider_is_never_actually_used_by_surprise(self):
        prime = _prime(_NoNetworkProvider())
        out = prime.handle(PrimeRequest("research", {"query": "q"}))
        assert out.status is RequestStatus.FAILURE


# ======================================================================
# 9. Production wiring: PRIMORA_RESEARCH_PROVIDER at the composition root
# ======================================================================


def _forbid_io(monkeypatch):
    """Selection must be pure: any network or subprocess call fails here."""

    def _boom(*args, **kwargs):
        raise AssertionError("provider selection performed I/O")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    monkeypatch.setattr("subprocess.run", _boom)
    monkeypatch.setattr("subprocess.Popen", _boom)
    monkeypatch.setattr("shutil.which", _boom)


class TestProductionWiring:
    """main.py selects the backend; the capability module never does."""

    def test_main_injects_the_selection_into_prime(self):
        import ast
        src = open(os.path.join("PRIMORA", "main.py"), encoding="utf-8").read()
        tree = ast.parse(src)
        prime_calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "Prime"
        ]
        assert len(prime_calls) == 1, "one production Prime construction"
        assert any(kw.arg == "research_provider"
                   for kw in prime_calls[0].keywords), (
            "main() must pass the selected research provider to Prime")
        assert "research_provider_from_env" in src

    def test_ddg_selection_wires_the_real_provider_without_network(
            self, monkeypatch):
        import PRIMORA.main as main_mod

        def _no_network(*args, **kwargs):
            raise AssertionError("provider selection reached the network")

        monkeypatch.setattr("urllib.request.urlopen", _no_network)
        selected = main_mod.research_provider_from_env(
            {"PRIMORA_RESEARCH_PROVIDER": "ddg"})
        assert isinstance(selected, DuckDuckGoResearchProvider)
        # Case and padding are not part of the contract's failure modes.
        assert isinstance(
            main_mod.research_provider_from_env(
                {"PRIMORA_RESEARCH_PROVIDER": " DDG "}),
            DuckDuckGoResearchProvider)
        # The seam injects it: Prime's research specialist is now live.
        prime = _prime(research_provider=selected)
        wired = prime.registry.get_specialist("research").provider
        assert isinstance(wired, DuckDuckGoResearchProvider)

    @pytest.mark.parametrize("value", ["google", "ddg2", "bing"])
    def test_unsupported_selection_preserves_fail_closed(self, value,
                                                         monkeypatch):
        import PRIMORA.main as main_mod
        monkeypatch.setenv("PRIMORA_RESEARCH_PROVIDER", value)
        assert main_mod.research_provider_from_env() is None
        out = _prime().handle(PrimeRequest("research", {"query": "q"}))
        assert out.status is RequestStatus.FAILURE
        assert "research provider" in out.error

    @pytest.mark.parametrize("value", ["", "   "])
    def test_an_unset_selection_selects_the_default_backend(self, value,
                                                            monkeypatch):
        """Empty means the default, not "no backend".

        The whole reason this changed: an unset provider left research
        failing closed on every request, so a current-information question
        fell through to the base model's own weights and came back with a
        fabricated release date and a made-up source number.
        """
        import PRIMORA.main as main_mod
        monkeypatch.setenv("PRIMORA_RESEARCH_PROVIDER", value)
        assert isinstance(
            main_mod.research_provider_from_env(),
            DuckDuckGoResearchProvider)

    def test_searxng_selection_wires_the_provider_without_io(
            self, monkeypatch):
        import PRIMORA.main as main_mod
        _forbid_io(monkeypatch)
        selected = main_mod.research_provider_from_env({
            "PRIMORA_RESEARCH_PROVIDER": "searxng",
            "PRIMORA_RESEARCH_ENDPOINT": "http://127.0.0.1:8888",
        })
        assert isinstance(selected, SearXNGResearchProvider)
        assert selected.endpoint == "http://127.0.0.1:8888"
        # Case and padding are not part of the contract's failure modes.
        selected = main_mod.research_provider_from_env({
            "PRIMORA_RESEARCH_PROVIDER": " SEARXNG ",
            "PRIMORA_RESEARCH_ENDPOINT": "http://127.0.0.1:8888",
        })
        assert isinstance(selected, SearXNGResearchProvider)
        # And from the real process environment, still without I/O.
        monkeypatch.setenv("PRIMORA_RESEARCH_PROVIDER", "searxng")
        monkeypatch.setenv(
            "PRIMORA_RESEARCH_ENDPOINT", "http://127.0.0.1:8888")
        assert isinstance(
            main_mod.research_provider_from_env(), SearXNGResearchProvider)

    @pytest.mark.parametrize("endpoint", [
        None, "", "   ", "http://localhost:8888",
        "https://127.0.0.1:8888", "http://127.0.0.1:9999",
        "http://example.com:8888", "http://127.0.0.1:8888/",
        "http://127.0.0.1:8888/search",
    ])
    def test_searxng_without_the_local_endpoint_disables_research(
            self, endpoint, monkeypatch, capsys):
        """Missing or invalid endpoint: warn, disable, zero I/O, no fallback."""
        import PRIMORA.main as main_mod
        _forbid_io(monkeypatch)
        env = {"PRIMORA_RESEARCH_PROVIDER": "searxng"}
        if endpoint is not None:
            env["PRIMORA_RESEARCH_ENDPOINT"] = endpoint
        selected = main_mod.research_provider_from_env(env)
        assert selected is None, "an invalid endpoint must not select a backend"
        assert "WARNING" in capsys.readouterr().out
        out = _prime().handle(PrimeRequest("research", {"query": "q"}))
        assert out.status is RequestStatus.FAILURE
        assert "research provider" in out.error

    def test_absent_selection_selects_the_default_backend(self, monkeypatch):
        import PRIMORA.main as main_mod
        monkeypatch.delenv("PRIMORA_RESEARCH_PROVIDER", raising=False)
        assert isinstance(
            main_mod.research_provider_from_env(),
            DuckDuckGoResearchProvider)

    def test_an_explicit_duckduckgo_is_accepted_alongside_the_short_form(
            self, monkeypatch):
        import PRIMORA.main as main_mod
        monkeypatch.setenv("PRIMORA_RESEARCH_PROVIDER", "DuckDuckGo")
        assert isinstance(
            main_mod.research_provider_from_env(),
            DuckDuckGoResearchProvider)

    def test_the_default_is_still_wired_into_prime(self):
        import PRIMORA.main as main_mod

        selected = main_mod.research_provider_from_env(
            {"PRIMORA_RESEARCH_PROVIDER": ""})
        prime = _prime(research_provider=selected)
        assert prime.registry.get_specialist("research").is_configured is True
        # An explicit unsupported value still leaves Prime fail-closed, so
        # "no backend chosen" and "backend errored" remain different states.
        assert main_mod.research_provider_from_env(
            {"PRIMORA_RESEARCH_PROVIDER": "google"}) is None
