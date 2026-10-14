"""Aurora must not claim an action happened when no capability ran.

Told "remember for next time: I take my coffee black", Aurora replied "I've
recorded that preference" - with no learn request behind it and nothing stored.
That is a truthfulness failure about an *action*, which the grounding
instruction did not cover: it covered invented facts and invented results, not
announcements of things she never did.

Prompt wording did not fix it. Sharpening the grounding clause moved the
false-claim rate from 5/6 to 4/6, which is noise. So the loop checks, because
the loop is the only thing that knows whether anything executed.

These tests pin the claim detector's precision - a false correction is worse
than a missed one, because it would tell Aurora she is wrong when she is right
- and the loop behaviour that depends on it.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.claims import _claims_research_action, unsupported_claim_error
from PRIMORA.aurora.loop import (
    CapabilityExecutor,
    CognitiveLoop,
    _claims_completed_action,
)
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeResult, PrimeRequest
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.providers import Provider
from PRIMORA.research import ResearchProvider, ResearchResult


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.success({"ok": True})


class _Failing(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.failure("nope")


class _Web(ResearchProvider):
    """A research backend that succeeds, for claims about searching."""

    name = "web"

    def search(self, query, limit=5):
        return [ResearchResult(
            title="Source", url="https://source.test/", snippet="found",
        )]


def _research_block():
    return (
        "<prime_request>\n"
        '{"action": "research", "payload": {"query": "x"}, '
        '"context": null, "metadata": null}\n'
        "</prime_request>"
    )


def _db(name: str) -> str:
    directory = tempfile.mkdtemp(prefix="unbacked_")
    return os.path.join(directory, name + ".db")


# ======================================================================
# 1. The detector
# ======================================================================


class TestCompletionClaimDetector:
    @pytest.mark.parametrize("text", [
        "I've recorded that preference.",
        "I've learned that from you.",
        "I've stored it.",
        "I have noted that preference.",
        "I checked that for you.",
        "I booked the survey.",
        "I sent it.",
    ])
    def test_detects_a_completed_action(self, text):
        assert _claims_completed_action(text) is True

    @pytest.mark.parametrize("text", [
        # In-conversation remembering needs no capability and is true.
        "I'll remember that you said Tuesday.",
        "I've noted that you said Tuesday.",
        "I remember that you wanted me to record that.",
        # Negations.
        "I did not save it.",
        "I couldn't record that preference.",
        "I have not learned that.",
        # Not claims at all.
        "The invoice is queried and payment is on hold.",
        "Paris.",
        "Nine words.",
        "We never actually made a decision on the roof.",
        "",
    ])
    def test_does_not_fire_on_truthful_or_unrelated_text(self, text):
        assert _claims_completed_action(text) is False


# ======================================================================
# 2. The executor records success, not just attempts
# ======================================================================


class TestExecutorRecordsSuccess:
    def test_successful_call_is_recorded(self):
        executor = CapabilityExecutor(Prime(_Provider()))
        executor.begin_session()
        executor.execute(PrimeRequest("calculate", {"expression": "1+1"}))
        assert executor.succeeded == ["calculate"]

    def test_failed_call_is_not_recorded_as_success(self):
        # An unregistered capability fails at validation, so it never reaches
        # the provider. A built-in like calculate would succeed regardless of
        # the provider, which would not test this at all.
        executor = CapabilityExecutor(Prime(_Failing()))
        executor.begin_session()
        executor.execute(PrimeRequest("not_a_capability", {}))
        assert executor.succeeded == []

    def test_attempts_and_successes_are_distinguished(self):
        """A refused attempt is not evidence that anything happened."""
        executor = CapabilityExecutor(Prime(_Provider()))
        executor.begin_session()
        executor.execute(PrimeRequest("not_a_capability", {}))
        assert len(executor.calls) == 1
        assert executor.succeeded == []

    def test_success_resets_with_the_turn(self):
        executor = CapabilityExecutor(Prime(_Provider()))
        executor.begin_session()
        executor.execute(PrimeRequest("calculate", {"expression": "1+1"}))
        executor.begin_session()
        assert executor.succeeded == []


# ======================================================================
# 3. The loop corrects an unbacked claim, once
# ======================================================================


class _ScriptedBrain(Brain):
    def __init__(self, *replies):
        self._replies = list(replies)
        self.calls = []

    def generate(self, messages, max_output_tokens=None):
        self.calls.append([dict(m) for m in messages])
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        return BrainResult.success(self._replies[index], "scripted")

    def get_model_name(self):
        return "scripted"

    def count_tokens(self, messages):
        return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _loop(replies, provider=None, research_provider=None, **kw):
    prime = Prime(provider or _Provider(), research_provider=research_provider)
    manager = ContextManager(db_path=_db("loop"))
    prime._context = manager
    prime._mind = prime._mind.__class__(manager)
    for name in ("recall", "learn"):
        specialist = prime.registry.get_specialist(name)
        if specialist is not None:
            specialist.bind(prime._mind)
    brain = _ScriptedBrain(*replies)
    executor = CapabilityExecutor(prime)
    return brain, CognitiveLoop(prime._mind, brain, executor=executor, **kw)


class TestUnbackedClaimCorrection:
    def test_false_claim_is_corrected_with_one_more_cycle(self):
        brain, loop = _loop(
            ["I've recorded that preference.",
             "Right, it isn't stored. Shall I request it?"])
        loop.run([{"role": "user", "content": "remember this"}], "s")
        assert len(brain.calls) == 2, "she must get a turn to correct herself"
        correction = [m for m in brain.calls[1] if m.get("role") == "tool"]
        assert correction, "the correction must be visible to her"
        assert "unbacked_claim" in correction[0]["content"]

    def test_granted_only_once(self):
        """A claim that repeats must not loop forever."""
        brain, loop = _loop(
            ["I've recorded that.", "I've recorded that.",
             "I've recorded that.", "I've recorded that."])
        loop.run([{"role": "user", "content": "remember this"}], "s")
        assert len(brain.calls) <= 3

    def test_no_correction_when_something_actually_ran(self):
        """A true claim must never be contradicted."""
        request = (
            "<prime_request>\n"
            '{"action": "calculate", "payload": {"expression": "1+1"}, '
            '"context": null, "metadata": null}\n'
            "</prime_request>"
        )
        brain, loop = _loop([request, "I've recorded that preference."])
        loop.run([{"role": "user", "content": "remember this"}], "s")
        corrections = [
            m for call in brain.calls for m in call
            if m.get("role") == "tool" and "unbacked_claim" in m["content"]]
        assert not corrections

    def test_no_correction_for_a_truthful_statement(self):
        brain, loop = _loop(["I'll remember that you said Tuesday."])
        loop.run([{"role": "user", "content": "note this"}], "s")
        assert len(brain.calls) == 1

    def test_correction_grants_a_generation_not_a_call(self):
        """The capability budget must be untouched by the safeguard."""
        brain, loop = _loop(
            ["I've recorded that.", "Understood, it isn't stored yet."])
        loop.run([{"role": "user", "content": "remember this"}], "s")
        assert loop._executor.calls == []

    def test_a_failed_capability_does_not_count_as_success(self):
        """A refused request leaves the claim unbacked, so it is corrected."""
        request = (
            "<prime_request>\n"
            '{"action": "count_words", "payload": {"text": "x"}, '
            '"context": null, "metadata": null}\n'
            "</prime_request>"
        )
        brain, loop = _loop(
            [request, "I've recorded that preference."],
            provider=_Failing())
        loop.run([{"role": "user", "content": "remember this"}], "s")
        corrections = [
            m for call in brain.calls for m in call
            if m.get("role") == "tool" and "unbacked_claim" in m["content"]]
        assert corrections


# ======================================================================
# 4. Research claims: backed only by research
# ======================================================================


class TestResearchClaimDetector:
    @pytest.mark.parametrize("text", [
        "I've searched the web for sources.",
        "I searched it.",
        "I researched that.",
        "I browsed the site.",
        "I fetched the page.",
        "I queried the database.",
        "I have searched the database.",
    ])
    def test_detects_a_research_completion(self, text):
        assert _claims_research_action(text) is True

    @pytest.mark.parametrize("text", [
        # Not first person, not a claim.
        "You searched the web.",
        "The invoice is queried and payment is on hold.",
        "The pages were fetched yesterday.",
        # The object frame keeps acknowledgements out.
        "I'll remember that you searched it.",
        # Negations and non-completions.
        "I never searched for it.",
        "I'll search the docs.",
        "",
    ])
    def test_ignores_truthful_or_unrelated_text(self, text):
        assert _claims_research_action(text) is False

    def test_the_loop_matcher_is_still_capability_free(self):
        """The generic detector must not have absorbed the new verbs."""
        assert _claims_completed_action("I've searched the web.") is False


class TestResearchClaimBacking:
    def test_nothing_ran_yields_the_generic_correction(self):
        error = unsupported_claim_error("I've searched the web.", [])
        assert error is not None
        assert "unbacked_claim" in error
        assert "No capability request has run" in error

    def test_unrelated_success_does_not_back_the_claim(self):
        error = unsupported_claim_error(
            "I've searched the web.", ["calculate"])
        assert error is not None
        assert "unbacked_claim" in error
        assert "No research request has run" in error

    def test_successful_research_backs_the_claim(self):
        assert unsupported_claim_error(
            "I've searched the web.", ["research"]) is None
        assert unsupported_claim_error(
            "I've searched the web.", ["research", "calculate"]) is None

    def test_non_research_claims_stay_with_the_loop_rule(self):
        """This module never decides another capability's claims."""
        assert unsupported_claim_error("I've recorded that.", ["calculate"]) is None
        assert unsupported_claim_error("I've recorded that.", []) is None


class TestResearchClaimCorrection:
    def test_unrelated_success_does_not_suppress_the_check(self):
        """A calculation succeeding must not buy a false 'I searched'."""
        request = (
            "<prime_request>\n"
            '{"action": "calculate", "payload": {"expression": "1+1"}, '
            '"context": null, "metadata": null}\n'
            "</prime_request>"
        )
        brain, loop = _loop([request, "I've searched the web for sources."])
        loop.run([{"role": "user", "content": "go"}], "s")
        assert loop.executor.succeeded == ["calculate"]
        corrections = [
            m for call in brain.calls for m in call
            if m.get("role") == "tool" and "unbacked_claim" in m["content"]]
        assert corrections, "the research claim must still be checked"
        assert "No research request has run" in corrections[0]["content"]

    def test_successful_research_produces_no_correction(self):
        """A true claim must never be contradicted."""
        brain, loop = _loop(
            [_research_block(), "I've searched the web for sources."],
            research_provider=_Web())
        loop.run([{"role": "user", "content": "go"}], "s")
        assert loop.executor.succeeded == ["research"]
        corrections = [
            m for call in brain.calls for m in call
            if m.get("role") == "tool" and "unbacked_claim" in m["content"]]
        assert not corrections

    def test_a_research_claim_with_nothing_run_is_corrected(self):
        brain, loop = _loop(["I've searched the web for sources."])
        loop.run([{"role": "user", "content": "go"}], "s")
        corrections = [
            m for call in brain.calls for m in call
            if m.get("role") == "tool" and "unbacked_claim" in m["content"]]
        assert corrections
        assert len(corrections) == 1, "corrected exactly once"