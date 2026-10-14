"""Focused tests for the validator's own rules.

The branching rule exists because C2-D03 carried a target shape of
"... / optionally tool / ...", which admitted a step that might not be there.
A reviewer could not tell whether a passing example had that turn or not.

These tests pin the rule to what it should reject and, more importantly, to what
it must not reject. An over-broad rule here would be worse than none: it would
push authors to reword correct seeds instead of fixing wrong ones.

Run: python -m pytest TESTS/training/test_validate_design_v2.py
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from TESTS.training.validate_design_v2 import (
    CONCRETENESS_EXEMPT,
    branching_shape_report,
    concreteness_report,
)


class _Shape:
    """Minimal stand-in carrying only the fields the rules read."""

    def __init__(self, seed_id: str, target_shape: str,
                 user_or_context: str = "", expected_behavior: str = "") -> None:
        self.seed_id = seed_id
        self.target_shape = target_shape
        self.user_or_context = user_or_context
        self.expected_behavior = expected_behavior


def _flagged(seed_id: str, target_shape: str) -> bool:
    return bool(branching_shape_report(
        [_Shape(seed_id, target_shape)]))


# -- must FAIL: optional / branching target descriptions ------------------

def test_optionally_tool_is_rejected():
    """The exact C2-D03 defect."""
    assert _flagged(
        "probe",
        "user / assistant(recall request) / tool(hit) / assistant(one "
        "decision from the result) / optionally tool / assistant(answer)")


def test_optional_request_is_rejected():
    assert _flagged(
        "probe",
        "user / assistant(optional request) / tool(success) / assistant(answer)")


def test_alternate_capabilities_are_rejected():
    """Two capabilities, one of which the author never chose."""
    assert _flagged(
        "probe", "user / assistant(consult_specialist or research request)")


def test_alternate_request_forms_are_rejected():
    assert _flagged(
        "probe", "user / assistant(calculate request or plain answer)")


# -- must PASS: legitimate sequences --------------------------------------

def test_concrete_conditional_sequence_passes():
    """A first result that determines whether a second step happens is one
    reviewable sequence, not two shapes."""
    assert not _flagged(
        "probe",
        "user / assistant(recall) / tool(hit) / assistant(second request) / "
        "tool(success) / assistant(answer)")


def test_tool_outcome_slot_may_vary():
    """A recovery seed's tool result legitimately varies. That is the point of
    the slot, and the target sequence around it is still fixed."""
    assert not _flagged(
        "probe",
        "user / assistant(recall request) / tool(hit or empty) / "
        "assistant(answer or honest no)")


def test_degree_pair_is_not_a_branch():
    """"brief or minimal" describes latitude in prose, not two shapes."""
    assert not _flagged(
        "probe",
        "user / assistant(learn request) / tool(success) / "
        "assistant(brief or minimal)")


def test_sentence_count_is_not_a_branch():
    assert not _flagged(
        "probe", "user / assistant(one or two natural sentences)")


def test_optional_in_user_context_passes():
    """"optional" in the scenario a user describes is ordinary English, and the
    rule only inspects target_shape."""
    seed = _Shape(
        "probe",
        target_shape="user / assistant(direct answer)",
        user_or_context="Amir: 'The limit field is optional for this query.'")
    assert not branching_shape_report([seed])
    assert not concreteness_report([seed])


# -- the live bank must satisfy its own rule ------------------------------

def test_no_seed_in_the_bank_branches():
    """C1-A06 and C6-X14 used to read "consult_specialist or research request"
    and "calculate request or plain answer" - the capability was never chosen,
    so a reviewer could not tell which shape was intended. Both were corrected
    to a single path, and this asserts the whole bank stays that way."""
    from TESTS.training.design_v2 import SEEDS

    found = branching_shape_report(SEEDS)
    assert found == [], (
        f"branching target shapes: {found}. Every target_shape must name "
        f"exactly one capability or request path.")


def test_concreteness_exemption_is_anchored_only():
    from TESTS.training.design_v2 import SEEDS

    assert CONCRETENESS_EXEMPT == {"C15-R02"}
    anchor = next(s for s in SEEDS if s.seed_id == "C15-R02")
    assert not anchor.training_allowed
    assert anchor.trained_assistant_indexes(4) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))