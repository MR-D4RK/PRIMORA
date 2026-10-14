"""Test-wide production-memory isolation.

Bare ``Prime()`` (and any ``ContextManager()`` with no ``db_path``) resolves
its store to the production database the moment ``_get_store()`` is first
called. Several test files construct a bare ``Prime`` and then call
``respond``/``ensure_session``, which opened the real database and wrote test
rows into Aurora's memory. It happened: a digest row and a session timestamp
were polluted by a test run.

This fixture intercepts the single choke point — ``ContextManager._get_store``
— and redirects the ``db_path=None`` fallback to a per-test temporary file.
It is the same interception philosophy as
``PRIMORA.prime.context.isolation.production_guard``, applied globally so a
test cannot forget to isolate itself. Explicit ``db_path`` values
(``harness_session``, the ``_isolate`` helpers) are never touched: only the
silent ``None`` fallback is redirected.

The observable ``context._db_path is None`` invariant is preserved — the
fallback store is built on a temp path without mutating the manager, so tests
that assert the production fallback exists still pass while the real file is
never opened.
"""

import os
import sys

import pytest

# The project root (parent of TESTS/) must be importable before any test
# module loads, so the interception can reference ContextManager.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PRIMORA.prime.context.manager import ContextManager  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_context_fallback(tmp_path):
    """Redirect the ``db_path=None`` store fallback to a per-test temp file.

    Installed for every test. Uses the same choke point production_guard
    intercepts. Restored on teardown so production behaviour is unchanged
    outside the test run.
    """
    original = ContextManager._get_store

    def safe_get_store(self):
        if self._db_path is None and self._store is None:
            # Build the fallback store on a temp path, without mutating the
            # manager's observable _db_path (which some tests assert is None).
            from PRIMORA.prime.context.store import ContextStore

            self._store = ContextStore(
                str(tmp_path / "fallback_context.db"))
        return original(self)

    ContextManager._get_store = safe_get_store
    try:
        yield
    finally:
        ContextManager._get_store = original
