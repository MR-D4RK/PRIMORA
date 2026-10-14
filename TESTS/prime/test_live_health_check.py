"""Read-only memory health check used by the live launcher.

The live driver verifies production memory is reachable and has the
expected schema without writing test messages, facts, sessions, or
digests. This test exercises that exact pattern against an isolated
temporary database (never production).
"""

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.context.manager import ContextManager


def _temp_manager():
    path = os.path.join(tempfile.mkdtemp(prefix="health_check_"), "ctx.db")
    return ContextManager(db_path=path), path


def _schema_tables(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


def test_schema_check_sees_expected_tables():
    manager, path = _temp_manager()
    store = manager._get_store()
    tables = _schema_tables(store.db_path)
    expected = {"messages", "digests", "sessions", "states",
                "candidates", "entities", "relationships"}
    assert expected <= tables, f"missing: {expected - tables}"


def test_recall_with_nonsense_query_writes_nothing():
    manager, path = _temp_manager()
    # Force store creation (and schema init) before snapshotting counts.
    manager._get_store()
    before = _count_rows(path)
    hits = manager.recall("__aurora_live_health_probe_nonsense__", limit=1)
    assert isinstance(hits, list)
    after = _count_rows(path)
    assert before == after, (
        f"read-only recall wrote rows: {before} -> {after}")


def _count_rows(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'")]
        counts = {}
        for t in tables:
            counts[t] = con.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
        return counts
    finally:
        con.close()


if __name__ == "__main__":
    test_schema_check_sees_expected_tables()
    test_recall_with_nonsense_query_writes_nothing()
    print("All health-check tests passed!")
