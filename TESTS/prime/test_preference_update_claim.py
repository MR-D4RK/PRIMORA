"""A preference-update claim must be backed by what was actually stored.

Production evidence: Aurora replied "Your favorite color has been updated to
violet!" while the stored current value stayed yellow and no state change was
ever written. Announcing an action that did not happen is a truthfulness
failure; the loop is the only place that can check it before the reply is
returned, because the loop owns the answer path and the Mind owns the store.

These tests pin the verifier's precision (no correction when the update really
did happen) and the loop behaviour that depends on it.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, r"C:\PRIMORA PROJECT")

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import Brain, BrainResult
from PRIMORA.aurora.configuration import AuroraConfig
from PRIMORA.prime import Prime
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.state import preference_update_claim_error
from PRIMORA.prime.providers import Provider

_TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
_SEQ = [0]


def _db(name="update"):
    _SEQ[0] += 1
    return os.path.join(_TMP.name, "%s-%d.db" % (name, _SEQ[0]))


def _config():
    return AuroraConfig(
        model_path=os.path.join("RESOURCES", "aurora", "v0.1", "model"),
        adapter_path=os.path.join("RESOURCES", "aurora", "v0.1", "aurora_qlora"),
    )


class _Provider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.success({"ok": True})


class ScriptedBrain(Brain):
    def __init__(self, *replies):
        self._replies = list(replies) or ["plain answer"]
        self.calls = []

    def generate(self, messages, max_output_tokens=None, on_delta=None,
                 cancelled=None):
        self.calls.append([dict(m) for m in messages])
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        return BrainResult.success(self._replies[index], "scripted")

    def get_model_name(self):
        return "scripted"

    def count_tokens(self, messages):
        return sum(len(m.get("content", "").split()) + 2 for m in messages)


def _aurora(brain, manager):
    prime = Prime(_Provider())
    return Aurora(prime, brain=brain, config=_config(),
                  context_manager=manager)


def _manager_with_yellow(name):
    mgr = ContextManager(db_path=_db(name))
    mgr.reconcile([
        {"role": "user", "content": "What is my favorite color?"},
        {"role": "user", "content": "It's yellow."},
    ], "default")
    return mgr


# ======================================================================
# 1. The store-level verifier
# ======================================================================
def test_verifier_flags_claim_without_stored_change():
    mgr = _manager_with_yellow("verify")
    error = preference_update_claim_error(
        mgr._get_store(), "Your favorite color has been updated to violet!")
    assert error is not None
    assert "yellow" in error
    assert "violet" in error


def test_verifier_accepts_claim_that_matches_store():
    mgr = _manager_with_yellow("verify_ok")
    mgr.reconcile([
        {"role": "user", "content": "my favorite color is violet"},
    ], "default")
    assert preference_update_claim_error(
        mgr._get_store(), "Your favorite color has been updated to violet!") is None


def test_verifier_flags_claim_with_no_stored_slot():
    mgr = _manager_with_yellow("verify_missing")
    assert preference_update_claim_error(
        mgr._get_store(), "Your favorite song has been updated to X.") is not None


def test_verifier_ignores_future_and_plain_phrasing():
    mgr = _manager_with_yellow("verify_safe")
    store = mgr._get_store()
    assert preference_update_claim_error(
        store, "If you'd like to change it, please let me know your new "
               "favorite color.") is None
    assert preference_update_claim_error(
        store, "I'll update your favorite color whenever you tell me.") is None
    assert preference_update_claim_error(
        store, "Your favorite color is important.") is None


# ======================================================================
# 2. The loop uses the verifier
# ======================================================================
def test_false_update_claim_is_corrected_before_reaching_user():
    mgr = _manager_with_yellow("loop_false")
    brain = ScriptedBrain(
        "Your favorite color has been updated to violet!",
        "Sorry - nothing was changed. Your favorite color is still yellow.",
    )
    result = _aurora(brain, mgr).respond(
        [{"role": "user", "content":
            "i wanna change my fav color to something else"}], "s")
    assert result.success
    assert result.content == (
        "Sorry - nothing was changed. Your favorite color is still yellow.")
    assert len(brain.calls) == 2


def test_true_update_claim_is_not_corrected():
    mgr = _manager_with_yellow("loop_true")
    brain = ScriptedBrain("Your favorite color has been updated to violet!")
    result = _aurora(brain, mgr).respond(
        [{"role": "user", "content": "my favorite color is violet"}], "s")
    assert result.success
    assert result.content == "Your favorite color has been updated to violet!"
    assert len(brain.calls) == 1


def test_repeated_false_update_claim_fails_instead_of_being_printed():
    mgr = _manager_with_yellow("loop_repeat")
    claim = "Your favorite color has been updated to violet!"
    brain = ScriptedBrain(claim, claim)
    result = _aurora(brain, mgr).respond(
        [{"role": "user", "content":
            "i wanna change my fav color to something else"}], "s")
    assert result.success is False
    assert result.content == ""
    assert "violet" not in (result.error or "")


if __name__ == "__main__":
    test_verifier_flags_claim_without_stored_change()
    test_verifier_accepts_claim_that_matches_store()
    test_verifier_flags_claim_with_no_stored_slot()
    test_verifier_ignores_future_and_plain_phrasing()
    test_false_update_claim_is_corrected_before_reaching_user()
    test_true_update_claim_is_not_corrected()
    test_repeated_false_update_claim_fails_instead_of_being_printed()
    print("All preference update claim tests passed!")
