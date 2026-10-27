"""Offline, provider-free checks for the evaluation harness (not Aurora IQ)."""
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

from runner import ANSWERS, bench_env, judge, render, source_audit


class RunnerTests(unittest.TestCase):
    def test_source_roles_detect_invented_user_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            with closing(sqlite3.connect(db)) as con:
                con.executescript(
                    "CREATE TABLE messages(id INTEGER, role TEXT);"
                    "CREATE TABLE states(id INTEGER, authority TEXT, source_ids TEXT, "
                    "concept TEXT, value TEXT, status TEXT, supersedes_id INTEGER);"
                    "CREATE TABLE experience_events("
                    "id INTEGER, authority TEXT, source_message_id INTEGER);")
                con.execute("INSERT INTO messages VALUES(1, 'user')")
                con.execute("INSERT INTO messages VALUES(2, 'assistant')")
                con.execute(
                    "INSERT INTO states (id, authority, source_ids) VALUES(1, 'user', ?)",
                    ("1",))
                con.execute(
                    "INSERT INTO states (id, authority, source_ids) VALUES(2, 'user', ?)",
                    ("2",))
                con.execute(
                    "INSERT INTO states (id, authority, source_ids) VALUES(3, 'user', ?)",
                    ("1,2",))
                con.execute(
                    "INSERT INTO states (id, authority, source_ids) VALUES(4, 'user', ?)",
                    ("bad",))
                con.execute(
                    "INSERT INTO experience_events VALUES(1,'user',1)")
                con.commit()
            report = source_audit(db)
            self.assertNotIn("error", report, report)
            self.assertEqual(report["sqlite_quick_check"], "ok")
            self.assertEqual(len(report["violations"]), 3)
            self.assertEqual(
                [v["state_id"] for v in report["violations"]], [2, 3, 4])
            self.assertEqual(report["state_count"], 4)
            self.assertEqual(len(report["state_summary"]), 4)

    def test_source_roles_cover_all_user_authority_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            with closing(sqlite3.connect(db)) as con:
                con.executescript(
                    "CREATE TABLE messages(id INTEGER, role TEXT);"
                    "CREATE TABLE candidates(id INTEGER, authority TEXT, source_ids TEXT);"
                    "CREATE TABLE entities(id INTEGER, authority TEXT, source_ids TEXT);"
                    "CREATE TABLE entity_aliases(id INTEGER, authority TEXT, source_ids TEXT);"
                    "CREATE TABLE relationships(id INTEGER, authority TEXT, source_ids TEXT);"
                    "CREATE TABLE observations(id INTEGER, authority TEXT, source_ids TEXT);"
                    "CREATE TABLE states(id INTEGER, authority TEXT, source_ids TEXT, "
                    "concept TEXT, value TEXT, status TEXT, supersedes_id INTEGER);"
                    "CREATE TABLE experience_events(id INTEGER, authority TEXT, source_message_id INTEGER);"
                )
                con.execute("INSERT INTO messages VALUES(1, 'user')")
                con.execute("INSERT INTO messages VALUES(2, 'assistant')")
                for table in ("candidates", "entities", "entity_aliases",
                              "relationships", "observations"):
                    con.execute(f"INSERT INTO {table} VALUES(1, 'user', '2')")
                con.commit()
            report = source_audit(db)
            self.assertEqual(report["authority_table_violations"], 5)
            self.assertEqual(len(report["violations"]), 5)

    def test_isolation_overrides_operator_memory_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic = Path(tmp) / "synthetic.db"
            with patch.dict(os.environ, {"PRIMORA_MEMORY_V2_DB": "production.db"}):
                env = bench_env(synthetic, "validation")
            self.assertEqual(env["PRIMORA_MEMORY_V2_DB"],
                             str(synthetic.resolve()))
            self.assertEqual(env["PRIMORA_RESEARCH_PROVIDER"], "disabled")
            self.assertEqual(env["PRIMORA_SESSION_ID"], "validation")

    def test_output_parsing_and_replaceable_fixture(self):
        log = ("You: Aurora: Hello!\n\nTimings: {\"total_ms\": 12}\n\n"
               "You: Aurora: I remember the design.\n\nTimings: {\"total_ms\": 5}"
               "\n\nGoodbye!")
        self.assertEqual(ANSWERS.findall(log),
                         ["Hello!", "I remember the design."])
        sample = render(
            {"prompt": "I like {{object_a}}.",
             "expect_contains": ["{{object_a}}"]},
            {"object_a": "Spruce Map"})
        self.assertEqual(sample["expect_contains"], ["Spruce Map"])
        self.assertTrue(judge(sample, "Spruce Map"))
        self.assertFalse(judge(sample, "Pine Map"))


if __name__ == "__main__":
    unittest.main()
