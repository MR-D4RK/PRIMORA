"""Operational tests for the M6 retrieval calibration experiments.

EXPERIMENTAL. These tests check that the calibration is reproducible, safe,
and honest. They do not assert that a specific metric value will never change:
the numbers are measurements, and M6 is expected to move them.

They also do not import anything from `PRIMORA.prime.context` for writing. The
calibration reads the benchmark corpus and the production modules read-only.
"""

import os
import sys

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from TESTS.prime import retrieval_calibration as cal


@pytest.fixture(scope="module")
def result():
    if cal.load_encoder() is None:
        pytest.skip(f"{cal.MODEL_ID} unavailable offline")
    return cal.run()


# ----------------------------------------------------------------------
# Production safety
# ----------------------------------------------------------------------
def test_calibration_never_touches_the_production_database(result):
    assert result["production_db_untouched"] is True
    assert result["production_db_sha256_before"] == (
        result["production_db_sha256_after"]
    )


def test_calibration_uses_the_approved_local_model(result):
    assert result["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert result["device"] == "cpu"
    assert result["dimension"] == 384


def test_calibration_module_does_not_write_to_production():
    source_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "TESTS", "prime", "retrieval_calibration.py",
    )
    with open(source_path, "r", encoding="utf-8") as handle:
        source = handle.read()
    for banned in (
        "INSERT INTO relationships", "UPDATE relationships",
        "DELETE FROM", "PRIMORA/data", "aurora_context.db",
        "CREATE VIRTUAL TABLE", "sqlite_vec", "faiss",
    ):
        assert banned not in source, banned
    # The production database is only ever hashed through the benchmark
    # helper, never opened for writing.
    assert "PRODUCTION_DB" in source
    assert "bench._sha256" in source
    assert "bench.PRODUCTION_DB" in source


def test_gated_keys_are_excluded_from_the_semantic_channel(result):
    """A gated message must not be reachable by similarity, at any tau."""
    import tempfile
    from TESTS.prime import retrieval_benchmark as bench

    fixture = bench._load_fixture()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        workdir = os.path.join(tmp, "gated")
        os.makedirs(workdir, exist_ok=True)
        built = cal.build_calibration(workdir, fixture)
        # A05 is the exact-query case the audit flagged: the gold message
        # sources both a superseded and a current state.
        blocked = cal.gated_keys(built, "A05")
        assert blocked, "expected the fixture to exercise the gate"
        ranked = cal.semantic_ranking(built, "A05", 0.0)
        assert not (set(ranked) & blocked), (
            "a similarity score re-admitted a supersession-gated message"
        )


# ----------------------------------------------------------------------
# The benchmark remains the source of truth
# ----------------------------------------------------------------------
def test_calibration_reuses_the_benchmark_fixture(result):
    from TESTS.prime import retrieval_benchmark as bench
    fixture = bench._load_fixture()
    assert result["rrf"]["lexical_only"]["query_count"] == len(
        fixture["queries"]
    )


def test_m6_boundary_in_production():
    """M6 is implemented; the M6 *exclusions* must still hold.

    This replaced a pre-M6 guard that asserted no hybrid code existed. What
    matters now is that the approved boundary was respected: the semantic
    channel exists, and nothing on the exclusion list came with it.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    package = os.path.join(root, "PRIMORA", "prime", "context")
    files = {
        name: open(os.path.join(package, name), "r", encoding="utf-8").read()
        for name in sorted(os.listdir(package)) if name.endswith(".py")
    }
    # The semantic channel is present and CPU-first.
    assert "all-MiniLM-L6-v2" in files["semantic.py"]
    assert 'DEVICE = "cpu"' in files["semantic.py"]
    # Exclusions: no graph database, no vector-search dependency, no
    # cross-encoder, no second generative model, no unrestricted traversal.
    for name, source in files.items():
        for banned in (
            "neo4j", "Neo4j", "faiss", "hnswlib", "usearch", "sqlite_vec",
            "sqlite-vec", "CrossEncoder", "cross_encoder", "AutoModelForCausalLM",
            "AutoModelForSequenceClassification", "onnxruntime",
        ):
            assert banned not in source, f"{name} contains {banned!r}"
    # No persistent message embeddings: only states and entities are indexed.
    assert "EMBEDDABLE_KINDS = (\"state\", \"entity\")" in files["semantic.py"]
    assert '"message"' not in files["semantic.py"].split(
        "EMBEDDABLE_KINDS = "
    )[1].split("\n", 1)[0]
    # Confidence is never read by the hybrid layer.
    assert ".confidence" not in files["hybrid.py"]


def test_no_new_production_dependency():
    """The semantic layer must degrade, not import, when unavailable."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    package = os.path.join(root, "PRIMORA", "prime", "context")
    source = open(
        os.path.join(package, "semantic.py"), "r", encoding="utf-8"
    ).read()
    # sentence_transformers and numpy must be imported lazily, inside
    # functions, so importing the context package never requires them.
    # Only column-0 lines are module-level imports.
    module_level = [
        line for line in source.splitlines()
        if line[:1] not in (" ", "\t", "") and line.strip()
    ]
    for line in module_level:
        if line.startswith(("import ", "from ")):
            assert "sentence_transformers" not in line, line
            assert line.strip() != "import numpy", line
    # ... and the lazy imports really are present inside functions.
    assert "    import numpy" in source
    assert "        from sentence_transformers import SentenceTransformer" in source
    assert "HF_HUB_OFFLINE" in source
    assert "TRANSFORMERS_OFFLINE" in source


# ----------------------------------------------------------------------
# Hard gates are deterministic and precede any score
# ----------------------------------------------------------------------
def test_similarity_cannot_override_a_hard_gate(result):
    gates = result["gates"]
    assert gates["gold_pairs"] > 0
    # The experiment must actually exercise the dangerous case, otherwise the
    # invariant is untested.
    assert gates["gated_pairs_that_would_pass_tau"] > 0
    assert "cannot re-admit" in gates["invariant"]


def test_tau_recommendation_follows_its_stated_rule(result):
    rec = result["recommendation"]
    sweep = result["semantic"]["tau_sweep"]
    qualifying = [
        row for row in sweep
        if row["false_positive_rate_at_3"] <= 0.15
        and row["gold_recall"] >= 0.70
    ]
    assert qualifying, "no threshold satisfies the tau rule"
    assert rec["tau"] == qualifying[0]["tau"]
    assert rec["tau"] in cal.TAU_GRID


def test_ratio_recommendation_is_inside_its_plateau(result):
    rec = result["recommendation"]
    plateau = rec["ratio_plateau"]
    assert plateau
    assert min(plateau) <= rec["ratio_recommendation"] <= max(plateau)
    evidence = rec["ratio_evidence"]
    baseline = result["post_fusion_floor"]["ratios"]["0.0"]
    assert evidence["recall_at_1"] == baseline["recall_at_1"]
    assert evidence["recall_at_3"] == baseline["recall_at_3"]
    assert (evidence["false_positive_rate_at_3"]
            < baseline["false_positive_rate_at_3"])


def test_k_non_identifiability_is_reported_not_hidden(result):
    rec = result["recommendation"]
    # On this fixture every k produces the same decisions. The calibration
    # must say so rather than presenting 60 as calibrated.
    assert rec["k_identifiable"] is False
    assert rec["k_distinct_outcomes"] == 1
    assert rec["k_recommendation"] == 60.0
    assert "not identifiable" in rec["k_reason"]


# ----------------------------------------------------------------------
# Experiment coverage
# ----------------------------------------------------------------------
def test_all_four_experiments_ran(result):
    assert result["semantic"]["tau_sweep"]
    assert result["rrf"]["lexical_only"]
    assert result["rrf"]["semantic_only"]
    assert result["same_session"]
    assert result["structural"]
    assert result["gates"]
    assert result["post_fusion_floor"]["ratios"]


def test_same_session_policies_were_all_measured(result):
    for label in ("current", "dominance", "gated"):
        assert f"lexical_{label}" in result["same_session"]
        assert f"hybrid_{label}" in result["same_session"]


def test_semantic_channel_beats_lexical_on_precision(result):
    lexical = result["rrf"]["lexical_only"]
    semantic = result["rrf"]["semantic_only"]
    assert semantic["false_positive_rate_at_3"] < lexical["false_positive_rate_at_3"]
    assert semantic["no_match_accuracy"] > lexical["no_match_accuracy"]


def test_structural_channel_needs_mention_detection_not_new_types(result):
    structural = result["structural"]
    assert structural["names_resolved_from_name"], "M5 resolution is broken"
    assert not structural["names_unresolved_from_name"]
    assert structural["targeted_resolved_from_free_text"] == 0
    assert structural["mention_detection_gap"]
    with_struct = structural["hybrid_with_structural"]
    without = structural["hybrid_without_structural"]
    assert (with_struct["recall_at_3__relationship"]
            > without["recall_at_3__relationship"])


def test_confidence_and_authority_are_not_used_as_thresholds(result):
    """M1 confidence must never become a semantic threshold or a rank input."""
    source_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "retrieval_calibration.py",
    )
    with open(source_path, "r", encoding="utf-8") as handle:
        source = handle.read().lower()
    # The calibration never reads a confidence or authority value at all:
    # similarity is compared only against tau, and lifecycle decisions are
    # delegated to the deterministic M1-M5 gates.
    assert "confidence" not in source
    assert ".authority" not in source


def test_summary_is_printed(result):
    report = cal.format_report(result)
    assert "RECOMMENDATION" in report
    assert "HARD GATES" in report
    assert "not identifiable" in report
    print("\nRECOMMENDED (derived from the sweeps)")
    rec = result["recommendation"]
    print(f"  tau   = {rec['tau']}   (rule: {rec['tau_rule']})")
    print(f"  k     = {rec['k_recommendation']}   "
          f"(identifiable={rec['k_identifiable']})")
    print(f"  ratio = {rec['ratio_recommendation']}   "
          f"(plateau={rec['ratio_plateau']})")
    print(f"  same-session: foreign>local cases = "
          f"{result['same_session']['hybrid_current']['foreign_outranks_local_gold']}")
