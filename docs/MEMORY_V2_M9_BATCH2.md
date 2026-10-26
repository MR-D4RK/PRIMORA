# Memory V2 M9 Batch 2

Batch 2 keeps the original database read-only and adds two explicit restoration
operations. `MemoryRestoreService.preview()` opens the source with SQLite
`mode=ro`, reports transferable rows, idempotent matches, conflicts, and
unsupported derived records. `import_into()` writes only to a destination that
passes the existing production-path isolation guard. Message, state, entity,
relationship, event, observation, and history ids are remapped and all source
references are rewritten through those maps. A second import is a no-op.
Legacy tables that do not contain the identity or foreign-key columns needed
for safe copying are reported as unsupported. Existing destination rows are
never overwritten; same-session sequence and current-state disagreements are
reported as conflicts. Existing files are checked for hard-link aliases
before any destination connection is opened.
`blocking_issues` contains conflicts and unsupported primary-memory records;
the derived-index list is informational, because vectors and index metadata
are rebuilt. The CLI exits with code 2 when blocking issues are found and
performs no destination import in that case.

Current state learning now checks current user states before creating a pending
candidate. An equivalent current state returns `status=already_known`; a
different value still follows the existing supersession path; held and rejected
outcomes remain distinct. The raw lesson source is retained in either case.

Natural memory questions with an explicit noun phrase are no longer stopped by
a model's over-conservative `clarify=true`. Unresolved pronouns and genuinely
bare questions still clarify. Prime supplies only source-backed evidence, so a
missing memory remains an abstention.

## Windows PowerShell validation

The source may be the production database for this read-only preview. The
destination must be disposable and outside the production directory:

```powershell
$source = Join-Path $PWD "PRIMORA\data\context\aurora_context.db"
$target = Join-Path $env:TEMP "primora-m9-batch2-v2.db"
Remove-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue

python -m PRIMORA.prime.context.memory_restore preview `
  --source $source `
  --target $target

# Continue only when the JSON report has no blocking_issues.
python -m PRIMORA.prime.context.memory_restore import `
  --source $source `
  --target $target

# Repeating import is safe and should report changed=false.
python -m PRIMORA.prime.context.memory_restore import `
  --source $source `
  --target $target

Remove-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
```

For live Qwen recall, point `PRIMORA_MEMORY_V2_DB` at the disposable target,
enable the existing V2 flags, and run the normal CLI. The original production
path is never a destination. No migration or recall path requires a second
model or external memory framework.

## Tests

```powershell
python -m pytest -q `
  TESTS/prime/test_m9_batch2.py `
  TESTS/prime/test_memory_v2_runtime_integration.py `
  TESTS/prime/test_unified_memory_intelligence.py `
  TESTS/prime/test_learn_capability.py
```

The repository tests use temporary databases and fake providers. Live Qwen
latency and model-led recall still require the local llama.cpp acceptance run.
The current checkout's production database was inspected through a read-only
connection before validation; its memory tables contain no user rows. The
import service does not rely on that state and never opens production as a
write destination.
