"""Interactive Memory V2 console tests; no network or production DB access."""

import json

from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context.isolation import production_db_path
from PRIMORA.prime.context.memory_v2_console import MemoryV2Console, main


class FakeFormationProvider:
    def __init__(self):
        self.calls = 0

    def invoke(self, capability, payload):
        self.calls += 1
        content = payload["messages"][-1]["content"]
        source_id = payload["source_message_id"]
        if not content.casefold().startswith("i like "):
            return PrimeResult.success({"content": '{"candidates": []}'})
        value = content[7:].rstrip(".")
        return PrimeResult.success({"content": json.dumps({"candidates": [{
            "kind": "preference", "concept": "preferred flarnic texture",
            "value": value, "claim_kind": "assertion", "actor": "user",
            "scope": "global", "source_ids": [source_id],
            "source_quote": content,
        }]})})


def test_console_supports_sessions_evidence_and_diagnostics(tmp_path):
    provider = FakeFormationProvider()
    output = []
    console = MemoryV2Console(
        tmp_path / "console.db", provider, output=output.append,
    )
    lines = iter([
        ":session alpha",
        "I like flarnic textures.",
        ":assistant I will remember that.",
        ":session beta",
        "I like a second flarnic texture.",
        ":evidence What are the last three things I told you I liked?",
        ":diagnostics",
        ":history alpha",
        ":quit",
    ])
    console.run(lambda prompt: next(lines))
    assert provider.calls == 2
    joined = "\n".join(output)
    assert "database=" in joined and "console.db" in joined
    assert "Memory evidence:" in joined
    assert "source_ids=" in joined
    assert "Formation status:" in joined
    assert "History for alpha:" in joined
    assert "I like flarnic textures." in joined
    assert console.manager.memory_v2_enabled is True
    assert console.manager._db_path != production_db_path()


def test_console_assistant_turn_is_not_formation_input(tmp_path):
    provider = FakeFormationProvider()
    console = MemoryV2Console(tmp_path / "assistant.db", provider, output=lambda _: None)
    console.handle(":assistant I like an assistant suggestion.")
    assert provider.calls == 0
    assert not console.manager._get_store().fetch_states(scope="global", status="current")


def test_cli_rejects_production_database_path_without_opening_it(capsys):
    assert main(["--db", str(production_db_path())]) == 2
    assert "production memory path" in capsys.readouterr().err

