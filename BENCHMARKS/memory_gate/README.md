# Final Memory V2 reliability gate (unmerged evaluation branch)

This is an **evaluation harness**, not an Aurora memory feature. Run it on the
isolated validation branch only. Neither the harness nor its fixtures teach
Aurora any PRIMORA architecture facts or provide hardcoded dialogue responses.

## Setup (Windows PowerShell)

Start your existing llama.cpp Qwen server on `http://127.0.0.1:8080/v1`.
Do **not** point the runner at production databases. The runner overrides
PRIMORA_MEMORY_V2_DB and creates a **fresh synthetic-only SQLite file for each
case**; no legacy migration takes place. It also disables outbound research.

```powershell
git fetch origin
git switch validation/memory-v2-final-gate
git pull --ff-only
$env:PRIMORA_OPENAI_URL = "http://127.0.0.1:8080/v1"
$env:PRIMORA_OPENAI_MODEL = "Qwen3.5-9B-Q4_K_M"
python BENCHMARKS/memory_gate/runner.py --dry-run
python -m unittest discover -s BENCHMARKS/memory_gate -p "test_*.py"
python BENCHMARKS/memory_gate/runner.py --seed 20261027
python BENCHMARKS/memory_gate/runner.py --seed 20261028
```

Each run writes logs, a separate disposable DB for each case, and a
`results.json` under `BENCHMARKS/memory_gate/runs/`. Keep this output out of Git. Each
test's second session is a fresh Aurora process with the **same test DB**, so
cross-session recall and persistence are exercised. The CLI logs timing
breakdowns and Prime capability calls.

**What the runner actually verifies:** answer-text smoke probes; DB quick
check; that source IDs on every present user-authority memory table (states,
candidates, entities, aliases, relationships, observations, and episodes)
point to real user messages; restart and persistence; observed total latency
p50/p95.

**What it does NOT establish:** semantic accuracy from equivalent paraphrases,
absence of every invented claim, perfect correction semantics, model call
counts, retrieval precision/recall@k, or Mem0/Graphiti/Letta benchmark parity.
Those require a blinded human/LLM-adjudicated evaluation set, retrieval traces,
and identical running baselines. Do not turn a passing probe into a PASS for
the entire memory milestone.

## Human adjudication

Review every answer in `results.json` plus the SQLite states and source rows.
For each answered turn, record in a separate review sheet:
- Supported / unsupported / partially supported, with user source IDs
- Did Aurora invent an association between separate facts?
- Did she treat a hypothetical as an asserted fact?
- Did a correction update the current preference, without erasing history?
- Did an assistant-origin statement become a user-authority memory?
- Did Aurora claim a successful save when only raw text was persisted?
- Did she retrieve facts without declaring incorrectly that memory is empty?

Report denominators and category-specific error rates. Failed infrastructure
sessions are **not** scored as memory errors.

## Acceptance / gate

Tentative goals, agreed **before results**:
- >=95% supported answer accuracy on a meaningful blind sample
- Zero assistant-origin memories promoted as user-source facts
- All asserted successful writes correspond to verified durable writes
- Zero production memory writes
- Correct cross-session evidence and preserved corrections
- No new phrase-based intelligence logic introduced in repairs

Any of these unmeasured criteria remains **PENDING**, not PASS. Record actual
hardware, local model hash, prompt/config, DB path for the synthetic test,
provider calls, and performance.

The independent comparison methodology is in `COMPETITORS.md`.
After this gate passes, freeze Memory V2 expansion and resume Phase 2 on
`milestone/cognition-foundation`; that phase must inspect **actual repository
files** rather than receive a hardcoded PRIMORA map.
