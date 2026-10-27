# Memory V2 final reliability gate

This branch adds the final trust-boundary checks before a freeze decision.
User-authority memory now requires provenance that resolves to a user message
whenever the source row exists. Assistant, tool, and system messages can be
retained as non-user evidence, but cannot be promoted by passing
`authority="user"`. User-provenance `learn` requests also require an exact
normalized match to an already persisted user statement in the active session;
the immutable user row is reused as the source instead of creating an
assistant evidence row. Unsupported typed rationales remain rejected unless
the exact reason is present in a user source.

The smoke gate audits all available authority-bearing tables, not only states:
candidate memories, entities, aliases, relationships, observations, and
experience events. It uses read-only SQLite access, disposable per-case
databases, and fails the automated gate on any user-authority source violation.

## PowerShell checks

```powershell
python BENCHMARKS/memory_gate/runner.py --dry-run
python -m unittest discover -s BENCHMARKS/memory_gate -p "test_*.py"
python -m pytest -q TESTS/prime/test_memory_v2_final_gate.py
python -m PRIMORA.prime.context.benchmark_v2
```

For live Qwen, start the existing llama.cpp server and set only a disposable
database path:

```powershell
$db = Join-Path $env:TEMP "primora-memory-v2-final-gate.db"
Remove-Item $db -Force -ErrorAction SilentlyContinue
$env:PRIMORA_OPENAI_URL = "http://127.0.0.1:8080/v1"
$env:PRIMORA_OPENAI_MODEL = "Qwen3.5-9B-Q4_K_M"
python BENCHMARKS/memory_gate/runner.py --seed 20261027
Remove-Item $db -Force -ErrorAction SilentlyContinue
```

The runner itself creates isolated synthetic databases and overrides
`PRIMORA_MEMORY_V2_DB`; it never migrates or opens the production database.

## Freeze status

The offline candidate benchmark meets the deterministic integrity targets:
formation F1 1.0, retrieval Recall@3/Precision@3 1.0, abstention 1.0,
unsupported-claim rate 0.0, source preservation 1.0, correction 1.0,
alias resolution 1.0, and two-clock event handling 1.0. Candidate local
operation latency was p50 8.8 ms and p95 9.7 ms; this excludes live Qwen
generation. The real Qwen smoke gate and blinded answer adjudication remain
pending. Mem0, Graphiti, and Letta were not run, so no cross-system score is
claimed.

Memory V2 is suitable for a **conditional freeze of code expansion** after
the disposable live-Qwen run and human review pass. It is not sufficient to
claim universal semantic reliability or competitor superiority until those
pending checks execute.
