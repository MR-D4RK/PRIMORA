# Advanced Memory V1: M6, M7 and M8

This document records the integrated Issue #2 implementation on
`milestone/evidence-backed-memory-v2`.  The existing SQLite store remains the
only memory source of truth.  Aurora still requests context and Prime still
owns authority, provenance, lifecycle and approval decisions.

## Trusted data flow

```text
immutable messages/events/states/entities/relationships
             |
             v
   M6 proposal grouping  ---> Prime source/scope/duplicate gates
             |                         |
             |                   candidate observation
             v
   M7 bounded SQLite traversal ---> typed evidence packets ---> Aurora context
             |
   M8 explicit read-only audit ---> reviewable findings/proposals
```

M6 (`prime/context/consolidation.py`) groups only records that already carry
user source messages, use the same canonical concept and value, and fall
within a bounded time compatibility window.  A proposal contains every
supporting source ID, exact source quote, proof count, and occurrence/report
time fields.  A dry run never writes.  Replay accepts explicit source IDs and
is idempotent.  Applied proposals are `actor="derived"`,
`observation_type="derived"`, and `status="candidate"`; Prime review is
required before any current claim can be created.  `add_observation_evidence`
adds support links atomically without changing the observation or raw message.

M7 (`prime/context/relational_retrieval.py`) uses the current entity, alias,
relationship and temporal lifecycle in SQLite.  It resolves an alias only
when it maps to one active user entity, checks scope, authority, status,
source roles and interval validity before returning a packet, and caps
traversal at four hops and 100 packets.  Inverse traversal is marked
`inferred=True` with a reason that identifies the user-stated edge.  No edge
is created during retrieval, and no second graph database is used.

M8 (`prime/context/memory_audit.py`) is a manual, read-only integrity job.  It
reports conflicts, duplicate current facts, superseded records, unsupported
source links, derived records with insufficient proof, and unresolved
candidates.  Each finding includes record IDs, source IDs and a proposed
review action.  The job makes zero model calls and cannot modify identity,
prompts, tools, policy or memory rows.

## Flags, migration and rollback

All three services are disabled by default.  Tests and operators explicitly
construct them with `enabled=True`; environment flags are available for
isolated tooling:

```text
PRIMORA_MEMORY_V2_CONSOLIDATION=1
PRIMORA_MEMORY_V2_RELATIONAL=1
PRIMORA_MEMORY_V2_AUDIT=1
```

The only schema change in this cycle is the additive evidence-link API over
the existing `observation_sources` table.  Existing migrations remain
transactional and idempotent.  Disable the flags to roll back behavior; this
does not delete rows.  Candidate derived observations can be rejected or
superseded through the existing Prime lifecycle.  No production backfill is
performed by these services.

## Benchmark and regression evidence

`prime/context/benchmark_v2.py` runs baseline and candidate paths over 200
synthetic conversations and 60 unfamiliar domains using disposable SQLite
files and the same evidence.  It measures source-aware formation and
retrieval scores plus alias resolution, corrections, two-clock events,
abstention, M6 proof/source coverage and idempotent replay, M7 evidence
coverage/bounded traversal/association precision, and M8 read-only behavior,
review proposals and model-call count.  Values are calculated from stored
rows and returned evidence packets; no success metric is hardcoded.  Qwen,
VRAM and RAM measurements are reported as `NOT RUN` in this environment.

Run the benchmark with:

```bash
/workspace/.venv/bin/python -m PRIMORA.prime.context.benchmark_v2
```

The focused Issue #2 regression tests are:

```bash
/workspace/.venv/bin/pytest -q \
  TESTS/prime/test_memory_m6.py \
  TESTS/prime/test_memory_m7.py \
  TESTS/prime/test_memory_m8.py \
  TESTS/prime/test_evidence_backed_memory_v2.py
```

The test fixtures use `tmp_path` databases and verify source immutability,
cross-session evidence, aliases, collisions, inverse relations, corrections,
two-clock events, and unsupported or unresolved records.

## Local Qwen verification

Codex cannot access the Windows llama.cpp process or the local Qwen weights.
Run the following in PowerShell from a disposable checkout after starting
llama.cpp on its OpenAI-compatible endpoint:

```powershell
$env:PRIMORA_MEMORY_V2="1"
$env:PRIMORA_MEMORY_V2_FORMATION="1"
$env:PRIMORA_MEMORY_V2_RETRIEVAL="1"
$env:PRIMORA_USER_TIMEZONE="America/New_York"
python -m PRIMORA.prime.context.memory_v2_console `
  --db-path "$env:TEMP\primora-memory-v2-qwen-test.db" `
  --base-url "http://127.0.0.1:8080/v1" `
  --model "Qwen3.5-9B-Q4_K_M"
```

Use `:session second`, `:evidence`, `:memory`, `:diagnostics`, and
`:sessions` while entering these prompts across sessions:

1. `I like translucent basalt mosaics.`
2. `I prefer asymmetrical architecture.`
3. `Yesterday I visited an art gallery.`
4. `I visited a botanical conservatory yesterday.`
5. `I finished cataloguing the amber samples.`
6. `I hate hardstyle.`
7. `What are the last three things I told you I liked?`
8. `What did I previously like?`
9. `When did I visit the art gallery?`
10. `What have I finished?`

Check that every accepted item shows its source message ID and quote, that
the gallery answer excludes the conservatory, that the completion query
excludes preferences, and that the two clocks use the original message
timestamp and `PRIMORA_USER_TIMEZONE`.  Inspect the provider diagnostics for
empty content, reasoning fallback, JSON parsing, and timing.  Delete the
temporary database after the run.

## Limitations and next step

M6 does not invent aliases, merge unlike concepts, or infer causes and
emotions.  M7 reads the existing relationship vocabulary and therefore does
not create observation-to-observation edges.  M8 identifies candidates for
review but does not approve them.  Local Qwen latency, structured-output
quality, VRAM use and real Windows llama.cpp compatibility remain untested
here.  M9 should connect these explicit packets and audit proposals to the
Aurora integration seam while preserving Prime's admission boundary.
