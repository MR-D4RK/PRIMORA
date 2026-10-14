"""Acceptance tests for the M6 benchmark comparison.

Runs the 47-query benchmark through the production retrieval path in four
configurations and checks the M6 shadow-acceptance criteria. These assert
*relationships* between configurations (hybrid must not be worse than the
baseline on precision or no-match behaviour) rather than pinning absolute
metric values, so a legitimate future improvement cannot fail the suite.
"""

import os
import sys

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from TESTS.prime import retrieval_benchmark as bench
from TESTS.prime import retrieval_benchmark_compare as compare
from PRIMORA.prime.context import semantic as semantic_mod


@pytest.fixture(scope="module")
def result():
    bench.assert_no_drift()
    if semantic_mod.load_model() is None:
        pytest.skip(f"{semantic_mod.MODEL_ID} unavailable offline")
    return compare.run()


# ----------------------------------------------------------------------
# Safety and integrity
# ----------------------------------------------------------------------
def test_comparison_never_touches_the_production_database(result):
    """The harness must not write to production memory.

    `production_db_untouched` is the real assertion: the harness hashes the
    production database before and after the run and compares, so it proves
    the benchmark and comparison code opened it read-only. That is the
    property worth protecting.

    A hardcoded whole-file hash is deliberately *not* asserted. Production
    memory is a live database that changes whenever the user talks to
    Aurora, so pinning its contents makes this test fail on correct
    behaviour and pass only while the database is frozen.
    """
    assert result["production_db_untouched"] is True
    assert len(result["production_db_sha256"]) == 64
    int(result["production_db_sha256"], 16)  # a real sha256, not a placeholder


def test_benchmark_mirror_has_not_drifted_from_production(result):
    """Membership must match exactly, or the mirror is lying to the suite."""
    assert result["mirror_membership_agreements"] == result["mirror_checked"]


def test_semantic_index_was_actually_populated(result):
    assert result["index_synced"]["states"] > 0
    assert result["index_synced"]["entities"] > 0


# ----------------------------------------------------------------------
# Acceptance criteria
# ----------------------------------------------------------------------
def test_semantic_retrieval_works(result):
    semantic = result["metrics"]["B"]
    assert semantic["recall_at_3"] > 0.0
    assert semantic["no_match_accuracy"] > 0.0


def test_hybrid_improves_recall_over_the_lexical_baseline(result):
    lexical = result["metrics"]["A"]
    hybrid = result["metrics"]["C"]
    assert hybrid["recall_at_3"] > lexical["recall_at_3"]
    assert hybrid["recall_at_1"] > lexical["recall_at_1"]


def test_hybrid_does_not_explode_false_positives(result):
    lexical = result["metrics"]["A"]
    hybrid = result["metrics"]["C"]
    structural = result["metrics"]["D"]
    assert hybrid["false_positive_rate_at_3"] <= lexical[
        "false_positive_rate_at_3"
    ]
    assert structural["false_positive_rate_at_3"] <= hybrid[
        "false_positive_rate_at_3"
    ]


def test_hybrid_does_not_regress_no_match_behaviour(result):
    lexical = result["metrics"]["A"]
    hybrid = result["metrics"]["C"]
    structural = result["metrics"]["D"]
    assert hybrid["no_match_accuracy"] >= lexical["no_match_accuracy"]
    assert structural["no_match_accuracy"] >= lexical["no_match_accuracy"]


def test_paraphrase_recall_improves(result):
    lexical = result["metrics"]["A"]["recall_at_3__paraphrase"]
    hybrid = result["metrics"]["C"]["recall_at_3__paraphrase"]
    assert hybrid > lexical, (lexical, hybrid)


def test_entity_and_relationship_queries_improve(result):
    lexical = result["metrics"]["A"]
    structural = result["metrics"]["D"]
    assert structural["recall_at_3__entity"] > lexical["recall_at_3__entity"]
    assert structural["recall_at_3__relationship"] > (
        lexical["recall_at_3__relationship"]
    )


def test_cross_session_and_punctuation_never_regress(result):
    lexical = result["metrics"]["A"]
    for name in ("C", "D"):
        config = result["metrics"][name]
        assert config["recall_at_3__cross_session"] >= (
            lexical["recall_at_3__cross_session"]
        )
        assert config["recall_at_3__punctuation"] >= (
            lexical["recall_at_3__punctuation"]
        )
        assert config["recall_at_3__temporal"] >= (
            lexical["recall_at_3__temporal"]
        )


def test_every_query_was_scored_in_every_configuration(result):
    for name in ("A", "B", "C", "D", "E", "F"):
        assert result["metrics"][name]["query_count"] == 47
        assert len(result["per_query"][name]) == 47


# ----------------------------------------------------------------------
# The live switch, measured against the live rollback control
# ----------------------------------------------------------------------
def test_live_hybrid_beats_live_rollback_on_recall(result):
    """E is the live path; F is the same pipeline with hybrid disabled.

    This is the meaningful live comparison. E must beat F on recall, and F
    exists so the live numbers are not confounded with the channel-mode
    numbers, which measure retrieval only.
    """
    rollback = result["metrics"]["F"]
    live = result["metrics"]["E"]
    for metric in ("recall_at_1", "recall_at_3", "recall_at_5",
                   f"precision_at_3"):
        assert live[metric] > rollback[metric], (metric, live, rollback)
    assert live["false_positive_rate_at_3"] < rollback[
        "false_positive_rate_at_3"
    ]


def test_live_hybrid_improves_the_targeted_classes(result):
    rollback = result["metrics"]["F"]
    live = result["metrics"]["E"]
    assert live["recall_at_3__entity"] > rollback["recall_at_3__entity"]
    assert live["recall_at_3__relationship"] > (
        rollback["recall_at_3__relationship"]
    )
    assert live["recall_at_3__exact"] > rollback["recall_at_3__exact"]
    assert live["recall_at_3__paraphrase"] >= (
        rollback["recall_at_3__paraphrase"]
    )
    # No regression anywhere the lexical channel already worked.
    for category in ("temporal", "cross_session", "punctuation", "indirect"):
        assert live[f"recall_at_3__{category}"] >= (
            rollback[f"recall_at_3__{category}"]
        ), category


def test_live_overhead_over_rollback_is_bounded(result):
    rollback = result["metrics"]["F"]["latency_ms"]["p95"]
    live = result["metrics"]["E"]["latency_ms"]["p95"]
    assert live > 0 and rollback > 0
    assert live < rollback * 3, (rollback, live)



def test_latency_is_reported_and_bounded(result):
    for name in ("A", "B", "C", "D", "E", "F"):
        latency = result["metrics"][name]["latency_ms"]
        assert latency["mean"] > 0
        assert latency["p95"] >= latency["median"] - 1e-6
        assert latency["p95"] < 5000, (name, latency)
    # The full hybrid path stays within a small multiple of lexical alone.
    assert result["metrics"]["D"]["latency_ms"]["p95"] < (
        result["metrics"]["A"]["latency_ms"]["p95"] * 5
    )


def test_results_are_reproducible(result):
    """Retrieval quality must be bit-identical across runs.

    Latency is excluded on purpose: it is wall-clock and is asserted as a
    bound elsewhere, not as a fixed value.
    """
    again = compare.run()
    for name in ("A", "B", "C", "D", "E", "F"):
        first = dict(result["metrics"][name])
        second = dict(again["metrics"][name])
        first.pop("latency_ms")
        second.pop("latency_ms")
        assert first == second, name
    for name in ("A", "B", "C", "D", "E", "F"):
        for query_id, row in result["per_query"][name].items():
            replay = again["per_query"][name][query_id]
            assert row["ranked"] == replay["ranked"], (name, query_id)
            assert row["gold"] == replay["gold"]


def test_report_renders(result):
    report = compare.format_report(result)
    assert "M6 BENCHMARK COMPARISON" in report
    assert "lexical only (rollback baseline)" in report
    assert "M5 structural" in report
    assert "LIVE build() rollback" in report
    print("\n" + report)
