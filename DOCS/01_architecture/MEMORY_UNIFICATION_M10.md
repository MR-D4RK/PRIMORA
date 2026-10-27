# Memory unification milestone (M10)

This milestone makes Prime's admission boundary the shared write seam for
deterministic extraction, typed formation, `learn`, and working-memory
promotion. Aurora may interpret a turn and propose meaning; Prime checks the
source role, scope, lifecycle, and provenance before a durable row is written.

## Runtime shape

Raw turns remain append-only in `messages`. A user turn is admitted by the
deterministic path immediately, then (when Memory V2 and a formation provider
are enabled) gets one idempotent row in `formation_jobs`. The job stores no
model output. A worker claims it transactionally, calls the configured local
provider, and sends proposals through the same gateway and existing typed
validation. A process restart reclaims `processing` jobs; failed jobs retry up
to five times. Provider or JSON failure leaves the raw turn and deterministic
memory intact and is visible in job diagnostics.

`learn(user, ...)` accepts only an exact, already persisted user message in the
same session. Aurora's assistant wording is never treated as user evidence.
Other provenance classes remain reviewable and cannot become a current
user-authority state. Existing databases are additive-migrated only; no
production path is opened by this milestone.

## Rollback controls

Memory V2 can be disabled with `PRIMORA_MEMORY_V2=0` (or
`ContextManager.set_memory_v2_enabled(False)`). Deferred processing can be
drained explicitly with `ContextManager.drain_formation()` or disabled with
`PRIMORA_MEMORY_FORMATION_DEFERRED=0`. The old deterministic and lexical
paths remain available, and the durable queue is additive and safe to leave
behind when the feature is disabled.

## Disposable Windows / PowerShell acceptance

Use a fresh file below `%TEMP%`; never point these commands at the production
database. The local llama.cpp server must already be running with its
OpenAI-compatible endpoint.

```powershell
$ErrorActionPreference = "Stop"
$db = Join-Path $env:TEMP "primora-memory-unification-$([guid]::NewGuid()).db"
$env:PRIMORA_MEMORY_V2 = "1"
$env:PRIMORA_MEMORY_V2_FORMATION = "1"
$env:PRIMORA_MEMORY_V2_RETRIEVAL = "1"
$env:PRIMORA_MEMORY_FORMATION_DEFERRED = "1"
$env:PRIMORA_PROVIDER = "openai"
$env:PRIMORA_MEMORY_V2_DB = $db
$env:PRIMORA_OPENAI_URL = "http://127.0.0.1:8080/v1"
$env:PRIMORA_OPENAI_MODEL = "Qwen3.5-9B-Q4_K_M"

python -m pytest -q `
  TESTS/prime/test_memory_unification_m10.py `
  TESTS/prime/test_memory_v2_runtime_integration.py `
  TESTS/prime/test_evidence_backed_memory_v2.py

# Preview/import is explicit and only targets the disposable destination.
python -m PRIMORA.prime.context.memory_restore preview `
  --source .\path\to\approved-source.db --target $db
python -m PRIMORA.prime.context.memory_restore import `
  --source .\path\to\approved-source.db --target $db

# Start the normal CLI against the disposable store.  Exit and inspect the
# formation_jobs table, then restart the CLI to verify durable recovery.
python -m PRIMORA.main --cli
```

For a live-Qwen queue drain without starting the CLI, run this from the
repository root after setting the variables above:

```powershell
python -c "from PRIMORA.prime.context import ContextManager; from PRIMORA.main import prime_provider_from_env; m=ContextManager(db_path=r'$db', enable_memory_v2=True, formation_provider=prime_provider_from_env()); print(m.drain_formation())"
```


## Acceptance evidence and limits

Automated tests use disposable SQLite files and deterministic provider doubles;
they do not validate Qwen output quality. Live acceptance must verify source
quotes and source IDs in `formation_jobs` diagnostics, a restart while a job
is `processing`, retry after a provider failure, and a repeated transcript
without a second provider call. No external memory framework or second model
is introduced. The model's semantic quality, throughput, and llama.cpp
template behavior remain hardware-dependent and require the local Qwen run.
