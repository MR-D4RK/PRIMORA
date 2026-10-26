# Memory V2 M9 Batch 1: live runtime integration

Batch 1 connects the existing typed formation path to the existing Prime/Mind
boundary. `Prime` passes its one configured provider to the one
`ContextManager`; `Mind.build()` calls `ContextManager.reconcile()` before
context assembly, so CLI and Verse GUI turns use the same formation and SQLite
lifecycle. No second database, model, agent, or retrieval path is created.

Formation remains explicitly opt-in through the existing flags:

```text
PRIMORA_MEMORY_V2=1
PRIMORA_MEMORY_V2_FORMATION=1
PRIMORA_MEMORY_V2_RETRIEVAL=1
```

The flags are disabled by default. Setting them back to `0` (or unsetting
them) rolls back typed formation/retrieval without deleting existing rows.
`ContextManager.set_memory_v2_enabled(False)` is the in-process rollback
control. Provider failures are recorded in formation diagnostics and leave
the ordinary Aurora response available.

For acceptance, set `PRIMORA_MEMORY_V2_DB` to a disposable path. The shorter
`PRIMORA_MEMORY_DB_PATH` name is also accepted. When neither is set, the
normal ContextStore default is retained; do not enable the feature against a
production database during validation.

## Windows PowerShell acceptance

Start llama.cpp `llama-server` separately with the local Qwen3.5-9B GGUF and
its OpenAI-compatible endpoint. From the repository root, configure one
isolated database and the same endpoint for Aurora's brain and Prime's
formation provider:

```powershell
$db = Join-Path $env:TEMP "primora-m9-memory-v2.db"
Remove-Item -LiteralPath $db -Force -ErrorAction SilentlyContinue

$env:PRIMORA_AURORA_BACKEND = "openai"
$env:PRIMORA_PROVIDER = "openai"
$env:PRIMORA_OPENAI_URL = "http://127.0.0.1:8080/v1"
$env:PRIMORA_OPENAI_MODEL = "Qwen3.5-9B-Q4_K_M"
$env:PRIMORA_OPENAI_TIMEOUT = "300"
$env:PRIMORA_MODEL_SERVER = "off"
$env:PRIMORA_MEMORY_V2 = "1"
$env:PRIMORA_MEMORY_V2_FORMATION = "1"
$env:PRIMORA_MEMORY_V2_RETRIEVAL = "1"
$env:PRIMORA_MEMORY_V2_DB = $db

python -m PRIMORA.main --cli
```

Use separate processes or `PRIMORA_SESSION_ID` values to verify persistence:

```powershell
$env:PRIMORA_SESSION_ID = "m9-alpha"
python -m PRIMORA.main --cli

$env:PRIMORA_SESSION_ID = "m9-beta"
python -m PRIMORA.main --cli
```

Enter a preference in `m9-alpha`, then ask for it in `m9-beta`. Check the
SQLite-backed evidence with the existing isolated console when needed:

```powershell
python -m PRIMORA.prime.context.memory_v2_console `
  --db $db `
  --url $env:PRIMORA_OPENAI_URL `
  --model $env:PRIMORA_OPENAI_MODEL `
  --session m9-beta
```

The console prints source IDs, formation diagnostics, provider timing, and
current states. Verify that a repeated transcript does not create another
formation call, that source IDs remain the original user-message IDs, and
that a provider outage still produces a normal Aurora response with a
diagnostic. Remove the disposable file after acceptance:

```powershell
Remove-Item -LiteralPath $db -Force
Remove-Item Env:PRIMORA_MEMORY_V2_DB -ErrorAction SilentlyContinue
```

To roll back the runtime switch in the current PowerShell session:

```powershell
$env:PRIMORA_MEMORY_V2 = "0"
$env:PRIMORA_MEMORY_V2_FORMATION = "0"
$env:PRIMORA_MEMORY_V2_RETRIEVAL = "0"
```

## Regression checks

The disposable end-to-end integration tests cover `Aurora.respond()`, shared
Prime/Mind provider wiring, disabled-mode behavior, provider failure
fallback, duplicate protection for persisted messages, restart, and
cross-session retrieval:

```powershell
python -m pytest -q `
  TESTS/prime/test_memory_v2_runtime_integration.py `
  TESTS/prime/test_evidence_backed_memory_v2.py `
  TESTS/prime/test_memory_v2_console.py
```

No test uses the production database. Live Qwen latency is measured by the
formation diagnostics (`provider_ms`, `db_ms`, `admission_ms`, and
`total_ms`); this repository test suite uses fake providers and cannot claim
the local model's wall-clock performance.
