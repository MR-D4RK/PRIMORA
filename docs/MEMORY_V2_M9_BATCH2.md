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

Preference reasons are admitted only from user text. Both direct `because`
clauses and an explicit causal follow-up sentence such as “It helps me focus”
are stored as a rationale state linked to the preference with the same source
ID. A typed provider may omit that field; Prime binds it from the exact source
clause, and never invents a reason. Concise follow-ups are interpreted by
Aurora's normal model-generated recall request; assistant replies remain
non-evidence and no phrase table routes the follow-up.

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

## Windows PowerShell contextual-recall acceptance

Use a disposable database and one session for the dialogue below; then repeat
the final two turns after restarting the CLI with a different
`PRIMORA_SESSION_ID`. The question and answer turns are context pointers only;
Prime's response must cite the original user message as the evidence source.

```powershell
$db = Join-Path $env:TEMP "primora-m9-contextual-recall.db"
Remove-Item -LiteralPath $db -Force -ErrorAction SilentlyContinue
$env:PRIMORA_AURORA_BACKEND = "openai"
$env:PRIMORA_PROVIDER = "openai"
$env:PRIMORA_OPENAI_URL = "http://127.0.0.1:8080/v1"
$env:PRIMORA_OPENAI_MODEL = "Qwen3.5-9B-Q4_K_M"
$env:PRIMORA_OPENAI_TIMEOUT = "300"
$env:PRIMORA_MEMORY_V2 = "1"
$env:PRIMORA_MEMORY_V2_FORMATION = "1"
$env:PRIMORA_MEMORY_V2_RETRIEVAL = "1"
$env:PRIMORA_MEMORY_V2_DB = $db
$env:PRIMORA_SESSION_ID = "m9-context-a"
python -m PRIMORA.main --cli
```

Enter these turns in the first process:

```text
I enjoy techno music when working. It helps me stay focused and calm.
What music do I listen to when working?
Why?
I think I have told you.
```

Aurora should answer the last two turns from the verified original reason. It
must abstain when the original statement has no reason, and it must not cite or
repeat an assistant reply as a user fact. Check the same topic from a fresh
session, then remove the disposable database:

```powershell
$env:PRIMORA_SESSION_ID = "m9-context-b"
python -m PRIMORA.main --cli
# Ask: What music do I listen to when working? / Why? / I already told you
Remove-Item -LiteralPath $db -Force
```

For generalization checks, repeat the run with unrelated user-created subjects
and paraphrases rather than the example wording: an unfamiliar object or
activity, an imperfectly typed memory question, a correction, and an ambiguous
pronoun that should receive clarification. Confirm that Qwen emits a `recall`
request only when its dialogue understanding requires stored memory, that the
ordinary turn uses one generation, and that every returned quote carries the
original user source. The implementation contains no subject or follow-up
phrase table to tune against these prompts.

To display per-turn latency without changing the conversation path, opt in to
the CLI timing line before starting it:

```powershell
$env:PRIMORA_CLI_TIMINGS = "1"
python -m PRIMORA.main --cli
```

Each completed turn prints JSON containing `planning_ms`,
`response_generation_ms`, `formation_ms` (and provider/database sub-timings),
capability timings, Jev timings when `learn` runs, and `total_ms`. Formation
remains Prime-owned and replay-safe; Aurora decides whether a memory read is
needed in the same model-driven conversational path. The source message and
Prime's normal provenance checks remain unchanged. Remove the
variable or set it to `0` to return to the normal CLI output.

Aurora's production path reports `planning_ms` as zero because there is no
pre-turn semantic planner. The model emits a Prime `recall` request directly
when the current conversation needs stored memory; the returned tool result is
then available to the next ordinary answer generation.

## Tests

```powershell
python -m pytest -q `
  TESTS/prime/test_m9_batch2.py `
  TESTS/prime/test_memory_v2_runtime_integration.py `
  TESTS/prime/test_unified_memory_intelligence.py `
  TESTS/prime/test_learn_capability.py `
  TESTS/prime/test_m9_contextual_recall.py
```

The repository tests use temporary databases and fake providers. Live Qwen
latency and model-led recall still require the local llama.cpp acceptance run.
The current checkout's production database was inspected through a read-only
connection before validation; its memory tables contain no user rows. The
import service does not rely on that state and never opens production as a
write destination.
