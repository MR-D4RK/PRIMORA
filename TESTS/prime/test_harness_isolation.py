"""Harnesses, evaluation, and training runs cannot write production memory.

The failure these guard against is silent: a harness that constructs a
``ContextManager`` without a path writes into Aurora's real database, the run
looks perfectly healthy, and the damage is only noticed by a guard in an
unrelated test. That happened once, in the manual T1 data-collection path.

These tests cover the guard, the session helper, the T1 runtime, and the real
validation entry points. The production database is checked by content
fingerprint rather than file hash, because opening a WAL database folds the
WAL back into the main file and changes bytes without changing rows.
"""

import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.isolation import (
    IsolationError,
    IsolatedSession,
    harness_session,
    production_digest,
    production_fingerprint,
    production_guard,
    require_isolated,
)
from PRIMORA.prime.context.store import ContextStore, default_db_path

PRODUCTION = os.path.abspath(str(default_db_path()))


def _rows(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {
            t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("messages", "states", "candidates", "entities")
        }
    finally:
        con.close()


class TestProductionIsReachableOnlyExplicitly:
    def test_production_path_is_where_we_think_it_is(self):
        assert os.path.abspath(str(default_db_path())) == PRODUCTION

    def test_require_isolated_refuses_production(self):
        with pytest.raises(IsolationError):
            require_isolated(PRODUCTION)

    def test_require_isolated_allows_a_temporary_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert str(require_isolated(tmp)).startswith(
                os.path.abspath(tmp))


class TestProductionGuard:
    def test_it_refuses_an_implicit_fallback(self):
        """The dangerous case: no path at all, invisible at the call site."""
        with production_guard():
            with pytest.raises(IsolationError):
                ContextManager()

    def test_it_refuses_an_explicit_production_path(self):
        with production_guard():
            with pytest.raises(IsolationError):
                ContextManager(db_path=PRODUCTION)

    def test_it_refuses_a_store_built_directly(self):
        with production_guard():
            with pytest.raises(IsolationError):
                ContextStore(PRODUCTION)

    def test_it_allows_an_isolated_store(self):
        tmp = tempfile.mkdtemp()
        with production_guard():
            store = ContextStore(os.path.join(tmp, "ok.db"))
            assert store.count("s") == 0

    def test_the_guard_is_released_on_exit(self):
        # The guard must stop refusing paths once the block ends. We prove it
        # with a *temporary* store: constructing one at PRODUCTION would make
        # SQLite create the real database file if it is absent, which is
        # exactly the production write this module exists to prevent.
        # mkdtemp (not TemporaryDirectory): ContextStore keeps the SQLite file
        # open, and Windows refuses to unlink an open file on context exit.
        released_path = os.path.join(tempfile.mkdtemp(), "released.db")
        with production_guard():
            pass
        released = ContextStore(released_path)
        assert released is not None

    def test_the_guard_is_released_even_after_an_error(self):
        # Same as above, but the guard block exits via an exception: the
        # finally clause must still restore normal construction. A temporary
        # path proves it without risking a production-file create.
        released_path = os.path.join(
            tempfile.mkdtemp(), "released_after_error.db")
        with pytest.raises(ValueError):
            with production_guard():
                raise ValueError("boom")
        released = ContextStore(released_path)
        assert released is not None

    def test_it_restores_both_hooks(self):
        before_store = ContextStore.__init__
        before_manager = ContextManager.__init__
        with production_guard():
            assert ContextStore.__init__ is not before_store
            assert ContextManager.__init__ is not before_manager
        assert ContextStore.__init__ is before_store
        assert ContextManager.__init__ is before_manager

    def test_it_refuses_a_manager_pointed_into_the_production_directory(self):
        production_dir = os.path.dirname(PRODUCTION)
        with pytest.raises(IsolationError):
            require_isolated(os.path.join(production_dir, "experiment.db"))


class TestHarnessSession:
    def test_it_creates_a_private_store(self):
        with harness_session("test_iso_") as session:
            context = session.context_manager()
            path = os.path.abspath(context._db_path)
            assert path != PRODUCTION
            assert path.startswith(os.path.abspath(session.root))

    def test_it_refuses_production_inside_the_session(self):
        with harness_session("test_iso_"):
            with pytest.raises(IsolationError):
                ContextManager()

    def test_it_fails_closed_when_isolation_cannot_be_established(self):
        with pytest.raises(IsolationError):
            with harness_session("test_iso_", root=PRODUCTION):
                pass

    def test_it_fails_closed_on_an_unusable_root(self):
        with pytest.raises(IsolationError):
            with harness_session("test_iso_",
                                 root=os.path.join(PRODUCTION, "nope", "x")):
                pass

    def test_separate_runs_get_separate_stores(self):
        paths = []
        for _ in range(3):
            with harness_session("test_iso_") as session:
                paths.append(os.path.abspath(session.context_manager()._db_path))
        assert len(set(paths)) == 3, paths

    def test_one_run_cannot_see_another_runs_memory(self):
        with harness_session("test_iso_a_") as first:
            first_store = ContextStore(first.context_manager()._db_path)
            first_store.append("s", 0, "user", "a secret only run one has")
        with harness_session("test_iso_b_") as second:
            second_store = ContextStore(second.context_manager()._db_path)
            assert second_store.count("s") == 0
            assert second_store.search(None, "secret", limit=5) == []

    def test_it_cleans_up_its_own_directory(self):
        with harness_session("test_iso_") as session:
            root = session.root
            session.context_manager()
            assert os.path.isdir(root)
        assert not os.path.exists(root)

    def test_cleanup_never_touches_production(self):
        before = _rows(PRODUCTION)
        with harness_session("test_iso_"):
            pass
        assert _rows(PRODUCTION) == before

    def test_it_verifies_production_after_the_run(self):
        before = production_digest()
        with harness_session("test_iso_") as session:
            session.context_manager()
        assert production_digest() == before


class TestT1RuntimeIsIsolated:
    def _runtime(self, tmp):
        from TESTS.t1.runtime import T1Runtime

        class _Brain:
            def warmup(self):
                return None

            def get_model_name(self):
                return "fake"

            def count_tokens(self, messages):
                return 8

            def generate(self, messages, max_output_tokens=None):
                from PRIMORA.aurora.brain import BrainResult

                return BrainResult.success("ok", "fake")

        runtime = T1Runtime(workdir=os.path.join(tmp, "rt"), brain=_Brain())
        runtime.warmup()
        return runtime

    def test_t1_uses_a_temporary_database(self, tmp_path):
        runtime = self._runtime(str(tmp_path))
        try:
            assert os.path.abspath(runtime.prime._context._db_path) != PRODUCTION
        finally:
            runtime.close()

    def test_t1_installs_the_production_guard(self, tmp_path):
        runtime = self._runtime(str(tmp_path))
        try:
            with pytest.raises(IsolationError):
                ContextManager()
        finally:
            runtime.close()

    def test_t1_releases_the_guard_on_close(self, tmp_path):
        runtime = self._runtime(str(tmp_path))
        runtime.close()
        ContextStore(PRODUCTION)
        ContextManager()

    def test_each_scenario_gets_its_own_database(self, tmp_path):
        from TESTS.t1.scenarios import SCENARIOS, Scenario

        runtime = self._runtime(str(tmp_path))
        try:
            seen = []
            for scenario in SCENARIOS[:3]:
                built = runtime.build_scenario(scenario)
                seen.append(os.path.abspath(built.db_path))
                built.dispose()
            assert len(set(seen)) == len(seen), seen
        finally:
            runtime.close()

    def test_a_t1_run_writes_nothing_to_production(self, tmp_path):
        from TESTS.t1.scenarios import SCENARIOS

        before = production_digest()
        runtime = self._runtime(str(tmp_path))
        try:
            for scenario in SCENARIOS[:2]:
                built = runtime.build_scenario(scenario)
                built.ask("hello?")
                built.dispose()
        finally:
            runtime.close()
        assert production_digest() == before


class TestValidationEntryPointsAreIsolated:
    def test_the_validation_build_helper_refuses_production(self):
        import TESTS.validation.validate_loop_v03_real_model as harness

        class _Brain:
            def get_model_name(self):
                return "fake"

            def count_tokens(self, messages):
                return 8

            def warmup(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            harness.build(_Brain(), tmp, None)
            with pytest.raises(IsolationError):
                ContextManager()
            harness.release_all()

    def test_production_is_untouched_by_the_validation_helper(self):
        import TESTS.validation.validate_loop_v03_real_model as harness

        class _Brain:
            def get_model_name(self):
                return "fake"

            def count_tokens(self, messages):
                return 8

        before = production_digest()
        with tempfile.TemporaryDirectory() as tmp:
            harness.build(_Brain(), tmp, None)
            harness.release_all()
        assert production_digest() == before


class TestProductionIsNotDirty:
    def test_production_memory_is_unchanged(self):
        """"Not dirty" means content-stable, never row-empty.

        A used install legitimately holds real memories (MEMORY_AUDIT
        C2), so the invariant is digest equality, not zero rows. The
        isolated run below is the work the emptiness check used to
        stand in for: it gets a private empty store, the guard refuses
        production opens, and the digest is re-checked on exit.
        """
        fingerprint = production_fingerprint()
        if not fingerprint.get("exists"):
            pytest.skip("no production database")
        before = production_digest()
        with harness_session("c2_not_dirty_") as session:
            store = session.context_manager()._get_store()
            assert store.fetch_states(limit=5) == []
        assert production_digest() == before, (
            "production memory changed during an isolated run"
        )


class TestGuardDoesNotTouchProductionBehaviour:
    def test_ordinary_production_construction_still_works(self):
        """Outside a harness nothing changes, by design."""
        store = ContextStore(PRODUCTION)
        assert os.path.abspath(str(store.db_path)) == PRODUCTION
        context = ContextManager()
        assert context._db_path is None, "production fallback is preserved"

    def test_the_guard_module_imports_cleanly(self):
        import importlib

        importlib.reload(
            importlib.import_module("PRIMORA.prime.context.isolation"))
