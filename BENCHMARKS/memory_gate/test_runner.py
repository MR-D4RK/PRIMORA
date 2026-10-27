"""Offline, provider-free checks for the evaluation harness (not Aurora IQ)."""
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from runner import ANSWERS, bench_env, judge, render, source_audit


class RunnerTests(unittest.TestCase):
    def test_source_roles_detect_invented_user_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            with sqlite3.connect(db) as con:
                con.executescript(
                    "CREATE TABLE messages(id INTEGER, role TEXT);"
                    "CREATE TABLE states(id INTEGER, authority TEXT, source_ids TEXT);"
                    "CREATE TABLE experience_events("
                    "id INTEGER, authority TEXT, source_message_id INTEGER);")
                con.execute("INSERT INTO messages VALUES(1, 'user')")
                con.execute("INSERT INTO messages VALUES(2, 'assistant')")
                con.execute(
                    "INSERT INTO states VALUES(1, 'user', ?)",
                    (json.dumps([1]),))
                con.execute(
                    "INSERT INTO states VALUES(2, 'user', ?)",
                    (json.dumps([2]),))
                con.execute(
                    "INSERT INTO experience_events VALUES(1,'user',1)")
            report = source_audit(db)
            self.assertEqual(report["sqlite_quick_check"], "ok")
            self.assertEqual(len(report["violations"]), 1)
            self.assertEqual(report["violations"][0]["state_id"], 2)

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
