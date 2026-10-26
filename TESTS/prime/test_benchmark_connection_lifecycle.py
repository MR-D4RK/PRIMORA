"""Windows-compatible checks for benchmark SQLite handle ownership."""

import sqlite3
from pathlib import Path

import pytest

from PRIMORA.prime.context import benchmark_v2
from PRIMORA.prime.context.store import ContextStore


def test_benchmark_temporary_databases_are_removed_immediately(monkeypatch):
    real_temporary_directory = benchmark_v2.tempfile.TemporaryDirectory
    created: list[Path] = []

    class TrackingTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            self._inner = real_temporary_directory(*args, **kwargs)
            created.append(Path(self._inner.name))

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            return self._inner.__exit__(exc_type, exc_value, traceback)

    monkeypatch.setattr(
        benchmark_v2.tempfile, "TemporaryDirectory", TrackingTemporaryDirectory
    )
    report = benchmark_v2.run_benchmark(seed=11)

    assert report["production_db_touched"] is False
    assert created
    # On Windows this assertion raises PermissionError if any SQLite handle
    # remains open when TemporaryDirectory.cleanup() runs.
    assert all(not directory.exists() for directory in created)


def test_store_connection_closes_after_exception(tmp_path):
    path = tmp_path / "exception.db"
    store = ContextStore(path)
    with pytest.raises(sqlite3.OperationalError):
        with store._connection() as connection:
            connection.execute("SELECT definitely_not_a_real_table")

    # A Windows delete is a direct ownership check: an open SQLite handle
    # would make this raise PermissionError instead of silently retrying.
    path.unlink()
    assert not path.exists()

