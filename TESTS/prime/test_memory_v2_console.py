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
    assert "requested count: 3" in joined
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


def test_console_m6_preview_sources_and_explicit_apply(tmp_path):
    output = []
    console = MemoryV2Console(
        tmp_path / "m6-console.db", FakeFormationProvider(), output=output.append
    )
    store = console.manager._get_store()
    first, *_ = store.append("alpha", 0, "user", "I like flarnic ceramics.", timestamp=10)
    second, *_ = store.append("beta", 0, "user", "I still like flarnic ceramics.", timestamp=20)
    store.create_state("alpha", "global", "preferred flarnic ceramics", "flarnic ceramics", [first], memory_type="PREFERENCE", timestamp=10)
    store.create_state("beta", "global", "preferred flarnic ceramics", "flarnic ceramics", [second], memory_type="PREFERENCE", timestamp=20)

    assert console.handle(":m6 preview")
    assert console.handle(":m6 sources 1")
    assert console.handle(":m6 apply 1")
    joined = "\n".join(output)
    assert "M6 preview" in joined
    assert f"source_id={first}" in joined and f"source_id={second}" in joined
    assert "admitted as observation" in joined and "proof_count=2" in joined
    rows = store.fetch_observations(scope="global", status="candidate")
    assert len(rows) == 1 and rows[0].actor == "derived"


def test_console_m7_uses_source_backed_prime_approval_and_queries_alias(tmp_path):
    output = []
    console = MemoryV2Console(
        tmp_path / "m7-console.db", FakeFormationProvider(), output=output.append
    )
    store = console.manager._get_store()
    source, *_ = store.append(
        "default", 0, "user", "Flarnic is part of the Aurora suite.", timestamp=10
    )
    console.handle(f':m7 entity Flarnic PROJECT {source}')
    console.handle(":m7 approve entity 1")
    console.handle(f':m7 entity "Aurora suite" PROJECT {source}')
    console.handle(":m7 approve entity 2")
    console.handle(f':m7 alias 1 Flar {source}')
    console.handle(":m7 approve alias 1")
    console.handle(f":m7 relationship part_of 1 2 {source}")
    console.handle(":m7 approve relationship 1")
    console.handle(":m7 query Flar")
    joined = "\n".join(output)
    assert "status=candidate" in joined
    assert "approved by Prime" in joined
    assert "M7 relational evidence packets=2" in joined
    assert "sources=[1]" in joined and "user-stated part_of" in joined


def test_console_m8_displays_read_only_review_findings(tmp_path):
    output = []
    console = MemoryV2Console(
        tmp_path / "m8-console.db", FakeFormationProvider(), output=output.append
    )
    store = console.manager._get_store()
    first, *_ = store.append("default", 0, "user", "I like one flarnic color.", timestamp=10)
    second, *_ = store.append("default", 1, "user", "I like another flarnic color.", timestamp=11)
    store.create_state("default", "default", "flarnic color", "red", [first], memory_type="PREFERENCE", timestamp=10)
    store.create_state("default", "default", "flarnic color", "blue", [second], memory_type="PREFERENCE", timestamp=11)
    console.handle(":m8 audit session")
    joined = "\n".join(output)
    assert "M8 read-only audit" in joined
    assert "conflict" in joined
    assert "recommendation=" in joined
    assert store.fetch_message(first) is not None and store.fetch_message(second) is not None

