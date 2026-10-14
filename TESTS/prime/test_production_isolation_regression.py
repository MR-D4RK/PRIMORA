"""Regression: no test can write to the production database.

A bare ``Prime()`` / ``ContextManager()`` with no ``db_path`` used to fall
back to Aurora's real database the moment its store opened, so a test that
called ``respond``/``ensure_session`` wrote test rows into production memory.
``TESTS/conftest.py`` now redirects that fallback to a per-test temp file.

These tests prove the isolation at three levels:

1. Store path — the fallback store is never the production file.
2. Content — a write through a bare ``Prime`` leaves ``production_digest``
   unchanged.
3. Structural — no test file constructs a store pointed at production.
"""

import ast
import glob
import os
import sys

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime import Prime
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.isolation import (
    production_db_path,
    production_digest,
    production_fingerprint,
)
from PRIMORA.prime.context.store import ContextStore
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.providers import Provider


class _UnusedProvider(Provider):
    def invoke(self, capability, payload):
        return PrimeResult.failure("unused in isolation regression")


def test_bare_context_manager_fallback_is_not_production():
    """The silent db_path=None fallback must resolve to a temp file."""
    ctx = ContextManager()
    # The observable fallback invariant is preserved...
    assert ctx._db_path is None
    # ...but the store that opens is never production.
    store = ctx._get_store()
    assert os.path.abspath(str(store.db_path)) != str(production_db_path())


def test_bare_prime_write_does_not_touch_production():
    """A write through a bare Prime leaves production content unchanged."""
    fingerprint = production_fingerprint()
    if not fingerprint.get("exists"):
        pytest.skip("no production database")
    before = production_digest()

    prime = Prime(_UnusedProvider())
    # Trigger the lazy store open and a real write path.
    store = prime._context._get_store()
    store.ensure_session("isolation_regression_probe")

    assert os.path.abspath(str(store.db_path)) != str(production_db_path()), (
        "bare Prime() opened the production database"
    )
    assert production_digest() == before, (
        "bare Prime() wrote to production memory"
    )


def test_repeated_fallbacks_share_the_test_temp_store(tmp_path):
    """Many bare managers in one test share the per-test temp file."""
    a = ContextManager()
    b = ContextManager()
    assert os.path.abspath(str(a._get_store().db_path)) == os.path.abspath(
        str(b._get_store().db_path))
    assert a._get_store().db_path != str(production_db_path())


def test_no_test_constructs_a_store_pointed_at_production():
    """Structural scan: no TESTS file builds a store on the production path.

    The isolation module and this file legitimately reference production for
    their guard assertions; every other test file must not.
    """
    production = str(production_db_path())
    offenders = []
    for path in sorted(glob.glob("TESTS/**/*.py", recursive=True)):
        if "conftest" in path or "__pycache__" in path:
            continue
        # This file and the isolation module intentionally reference it.
        if path.endswith(("test_production_isolation_regression.py",
                          "test_harness_isolation.py")):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        if production in src:
            offenders.append(path)
    assert not offenders, (
        "tests must not point a store at production: "
        + ", ".join(offenders)
    )


if __name__ == "__main__":
    test_bare_context_manager_fallback_is_not_production()
    test_bare_prime_write_does_not_touch_production()
    test_repeated_fallbacks_share_the_test_temp_store()
    test_no_test_constructs_a_store_pointed_at_production()
    print("All production-isolation regression tests passed!")
