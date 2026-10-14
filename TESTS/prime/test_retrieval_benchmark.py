"""Operational tests for the retrieval benchmark.

These tests assert that the benchmark itself is trustworthy and safe. They
deliberately do NOT assert baseline metric values: the numbers are a recorded
measurement, not a gate, and M6 is expected to improve them. Pinning them here
would make a correct improvement look like a regression.

The only quality invariant that is gated is the FTS query-construction fix:
zero terms may be silently dropped.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from TESTS.prime import retrieval_benchmark as bench


@pytest.fixture(scope="module")
def report():
    return bench.run()


@pytest.fixture(scope="module")
def fixture():
    return bench._load_fixture()


# ----------------------------------------------------------------------
# Production safety
# ----------------------------------------------------------------------
def test_benchmark_never_touches_the_production_database(report):
    assert report.production_db_untouched is True
    assert report.production_db_sha256 is not None


def test_benchmark_writes_only_to_its_own_temp_directory(tmp_path):
    before = set(os.listdir(os.path.dirname(bench.PRODUCTION_DB)))
    bench.run(tmpdir=str(tmp_path))
    after = set(os.listdir(os.path.dirname(bench.PRODUCTION_DB)))
    assert before == after


# ----------------------------------------------------------------------
# Drift guards
# ----------------------------------------------------------------------
def test_no_drift_guards_pass():
    bench.assert_no_drift()


def test_drift_guard_detects_changed_weights(monkeypatch):
    with pytest.raises(AssertionError, match="drift"):
        monkeypatch.setattr(
            bench.scoring, "score",
            lambda **kwargs: {"score": 0.0, "relevance": 0.0, "recency": 0.0,
                              "importance": 0.0, "redundancy": 0.0},
        )
        bench.assert_no_drift()


# ----------------------------------------------------------------------
# Fixture integrity
# ----------------------------------------------------------------------
def test_fixture_ground_truth_is_explicit_and_resolvable(fixture):
    keys = {e["key"] for entries in fixture["sessions"].values()
            for e in entries}
    for query in fixture["queries"]:
        assert "gold" in query, query["id"]
        for key in query["gold"]:
            assert key in keys, (query["id"], key)
    assert any(len(q["gold"]) == 1 for q in fixture["queries"])
    assert any(not q["gold"] for q in fixture["queries"])


def test_fixture_covers_every_required_category(fixture):
    required = {
        "exact", "paraphrase", "indirect", "entity", "relationship",
        "temporal", "no_match", "cross_session", "false_positive_control",
        "punctuation",
    }
    assert required <= {q["category"] for q in fixture["queries"]}


def test_fixture_has_distractors(fixture):
    assert len(fixture["distractor_keys"]) >= 10
    controls = [q for q in fixture["queries"]
                if q.get("expect_distractor")]
    assert controls, "the false-positive control needs at least one query"


# ----------------------------------------------------------------------
# Fixture self-consistency (added by M6 after the G05 labelling defect)
# ----------------------------------------------------------------------
def test_empty_gold_iff_declared_negative_category(fixture):
    """Only the `no_match` category may have an empty gold set.

    `false_positive_control` is mixed by design: some of its queries have a
    gold memory (and a distractor that must not outrank it) and one does not.
    This rule guards the coherence of the negative category. The defect that
    made G05 wrong is caught by the adjacency rule below, not by this one.
    """
    for query in fixture["queries"]:
        if query["category"] == "no_match":
            assert query["gold"] == [], (query["id"], query["gold"])
        else:
            assert query["gold"] or query["category"] == (
                "false_positive_control"
            ), (query["id"], query["category"])


def test_every_negative_query_declares_its_topical_adjacency(fixture):
    """A negative query must declare every corpus message adjacent to it.

    Forcing the declaration is the point. It makes "there is nothing
    relevant here" a reviewed claim rather than an unexamined default, and
    it is what surfaced the G05 defect in the first place.
    """
    for query in fixture["queries"]:
        if query["gold"]:
            continue
        assert "related_keys" in query, (
            f"{query['id']} has no gold and must declare related_keys"
        )
        assert isinstance(query["related_keys"], list)
        for key in query["related_keys"]:
            assert key not in query["gold"]


def test_declared_adjacency_matches_the_corpus(fixture):
    """No undeclared topical adjacency may exist for a negative query.

    A corpus message that shares a distinctive (low document-frequency)
    content term with a negative query must appear in that query's
    `related_keys`. This is the automatic guard: when a message mentioning
    the query subject is added to the corpus, the negative query stops
    validating until a human declares and judges the adjacency.
    """
    from PRIMORA.prime.context import scoring

    keys = [e["key"] for entries in fixture["sessions"].values()
            for e in entries]
    tokens = {
        entry["key"]: set(scoring.tokenize(entry["text"]))
        for entries in fixture["sessions"].values()
        for entry in entries
    }
    document_frequency: dict[str, int] = {}
    for terms in tokens.values():
        for term in terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1

    for query in fixture["queries"]:
        if query["gold"]:
            continue
        query_terms = set(scoring.tokenize(query["text"]))
        expected = {
            key for key in keys
            if any(document_frequency[term] <= 3
                   for term in query_terms & tokens[key])
        }
        declared = set(query["related_keys"])
        assert expected == declared, (
            f"{query['id']}: undeclared adjacency {sorted(expected - declared)}"
            f", over-declared {sorted(declared - expected)}"
        )


def test_related_keys_resolve_and_are_never_gold(fixture):
    keys = {e["key"] for entries in fixture["sessions"].values()
            for e in entries}
    for query in fixture["queries"]:
        for key in query.get("related_keys", []):
            assert key in keys, (query["id"], key)
            assert key not in query["gold"], query["id"]


def test_g05_correction_is_recorded(fixture):
    """The specific defect this guard was added for must stay fixed."""
    by_id = {q["id"]: q for q in fixture["queries"]}
    g05 = by_id["G05"]
    assert g05["category"] != "no_match"
    assert g05["gold"] == ["fill_4"]
    assert "CORRECTED" in g05["note"]


def test_fixture_never_infers_truth_from_ranking(fixture):
    # Ground truth is a static declaration; no category may be derived from
    # what the system returns.
    for query in fixture["queries"]:
        assert "returned" not in query
        assert "expected" not in query


# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------
def test_every_query_ran_in_every_mode(report, fixture):
    expected = len(fixture["queries"])
    for mode in bench.RUN_MODES:
        assert report.modes[mode]["query_count"] == expected
        assert report.modes[mode]["failed_count"] == 0
    assert len(report.observations) == expected * len(bench.RUN_MODES)


def test_all_required_metrics_are_reported(report):
    channel = report.modes["channel"]
    for key in (
        "recall_at_1", "recall_at_3", "recall_at_5", "precision_at_3",
        "false_positive_rate_at_3", "false_positive_rate_all",
        "no_match_accuracy", "punctuation_success", "query_count",
        "relevant_result_count", "failed_count", "latency_ms",
    ):
        assert key in channel, key
    latency = channel["latency_ms"]
    for key in ("mean", "median", "p95", "min", "max", "count"):
        assert key in latency, key
    for category in ("exact", "paraphrase", "indirect", "entity",
                     "relationship", "temporal", "cross_session",
                     "punctuation"):
        assert f"recall_at_3__{category}" in channel, category


def test_observations_carry_a_structured_trace(report, fixture):
    by_id = {q["id"]: q for q in fixture["queries"]}
    for observation in report.observations:
        assert observation.query_id in by_id
        assert observation.category
        assert observation.match
        assert observation.session_id
        assert observation.scope
        assert observation.expected_keys == by_id[observation.query_id]["gold"]
        assert isinstance(observation.ranks, dict)
        assert isinstance(observation.scores, dict)
        assert observation.error is None
    # Scores are recorded where they are observable, and the end-to-end mode
    # is honest about not exposing them.
    channel = [o for o in report.observations if o.mode == "channel"]
    assert any(o.scores for o in channel)
    context = [o for o in report.observations if o.mode == "context"]
    assert all(not o.scores for o in context)


def test_trace_never_enters_the_prompt(report):
    # The benchmark only reads the assembled view; it injects nothing.
    for observation in report.observations:
        if observation.mode != "context":
            continue
        assert not any(
            "retrieval_benchmark" in line
            for line in observation.synthesized_lines
        )


# ----------------------------------------------------------------------
# The one gated quality invariant: the FTS fix
# ----------------------------------------------------------------------
def test_no_query_term_is_silently_dropped(report):
    # Before the FTS fix this was non-zero: apostrophes, hyphens and question
    # marks raised FTS5 OperationalError and were dropped per term.
    for mode in bench.RUN_MODES:
        assert report.modes[mode]["skipped_term_total"] == 0, mode
        assert report.modes[mode]["search_failure_total"] == 0, mode


def test_punctuation_queries_now_retrieve(report):
    channel = report.modes["channel"]
    assert channel["punctuation_success"] == 1.0
    assert channel["recall_at_3__punctuation"] == 1.0


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------
def test_channel_ranking_is_deterministic():
    fixture = bench._load_fixture()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "determinism.db")
        corpus = bench.build_corpus(path, fixture)
        from PRIMORA.prime.context.store import ContextStore
        store = ContextStore(path)
        query = next(q for q in fixture["queries"]
                     if q["category"] == "paraphrase")
        first, first_foreign, _terms, _diag = bench._rank_candidates(
            store, query["session"], query["text"]
        )
        second, second_foreign, _terms2, _diag2 = bench._rank_candidates(
            store, query["session"], query["text"]
        )
        assert [row[0] for row in first] == [row[0] for row in second]
        assert [row[2].message_id for row in first] == [
            row[2].message_id for row in second
        ]
        assert [row[3].message_id for row in first_foreign] == [
            row[3].message_id for row in second_foreign
        ]
        assert corpus.key_to_message_id


# ----------------------------------------------------------------------
# Baseline is recorded, not gated
# ----------------------------------------------------------------------
def test_baseline_summary_is_reported(report):
    """A human-readable snapshot of what the current system does.

    This documents the measured baseline. It asserts only that the run
    produced numbers, not that they equal a fixed value.
    """
    channel = report.modes["channel"]
    context = report.modes["context"]
    assert 0.0 <= channel["recall_at_1"] <= 1.0
    assert 0.0 <= channel["recall_at_3"] <= 1.0
    assert channel["recall_at_1"] <= channel["recall_at_3"]
    assert channel["recall_at_3"] <= channel["recall_at_5"]
    assert channel["recall_at_5"] <= channel["recall_at_all"]
    assert 0.0 <= context["coverage"] <= 1.0
    print("\nBASELINE (channel, ranked):")
    for key in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_all",
                "precision_at_3", "false_positive_rate_at_3",
                "no_match_accuracy"):
        print(f"  {key:28} {channel[key]}")
    print("BASELINE (per category, recall@3):")
    for key in sorted(k for k in channel if k.startswith("recall_at_3__")):
        print(f"  {key.split('__', 1)[1]:28} {channel[key]}")
    print(f"BASELINE (context, coverage): {context['coverage']}")
