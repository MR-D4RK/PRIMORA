# PRIMORA — instructions for coding agents

These instructions apply to the entire repository. Read the actual source and current
issues before making claims about the architecture or changing behavior.

## Mission and boundaries

- **PRIMORA = Prime + Aurora.** Aurora is the single evolving conversational
  identity and uses models and specialist tools as capabilities, not as replacement
  personalities. Prime owns orchestration, execution limits, permissions, and
  evidence/persistence safety; Aurora/model interprets meaning and proposes actions.
- Prioritize genuine, generalizable capabilities rather than prompt-specific or
  domain-specific fixes. Never hardcode particular colors, cars, objects, benchmark
  terms, PRIMORA component names, or expected dialogue answers merely to pass tests.
- Never assume stored summaries or agent responses are user statements.
  User-authority memory requires actual persisted user messages and verified
  evidence, with compatible session/scope; no assistant/tool/system messages or
  fabricated, missing, or empty source references. Distinguish explicitly stated
  causes from mere circumstances; preserve uncertainty, polarity, and strength.
- Avoid separate write paths that bypass the shared admission/trust boundary.
  Preserve provenance, correction/supersession history, stable identities,
  and honest saved/pending/failed results.
- External services, publishing, consequential actions, and merges require user
  approval. Do not expand an agent's tools beyond Prime's permissions.

## Active development plan

- Current working branch: `fix/m10-final-reliability` (from M10 `ddfb61a`).
- Current task: GitHub Issue #7, the final memory reliability corrections.
  Read Issues #6 and #5 for architectural context, and
  `DOCS/01_architecture/ROADMAP_AURORA_AFTER_MEMORY_V2.md`.
- Finish M10 critical safety, durability and source-fidelity gates before official
  Phase 2. Noncritical memory quality improvements may continue during cognition.
- Phase 2 begins with **read-only inspection of actual PRIMORA repository code**,
  then planning, model routing, and approved development tools. The model must
  obtain architectural knowledge from source, not fixed facts embedded in prompts.

## Implementation rules

1. Work only in an isolated branch. Never change or run tests against production
   memory files, especially `PRIMORA/data/context/aurora_context.db`. Use a
   disposable SQLite DB and an isolated repo copy when migrations are involved.
2. Prefer the smallest cohesive change that fixes the underlying invariant.
   Preserve existing public contracts or explicitly migrate and test changes.
   Do not hide failing tests, delete coverage to manufacture a green result,
   or label a known behavior change as an unexplained legacy failure.
3. For durable deferred formation: persist messages first; bound retries,
   concurrency, resource usage and job recovery; prevent duplication and
   unbounded concurrent Qwen calls on local hardware.
4. Write cross-domain regression tests including negative/contradictory cases:
   non-user or missing sources, unrelated reasons, hypotheticals, corrected facts,
   paraphrases, restarts, failures and idempotent replay.
5. Report what actually ran, with PASS/FAIL/SKIP counts and reasons. Distinguish
   unit tests, smoke/lexical checks, human semantic review, and live-model tests.
   A smoke PASS is not proof of semantic fidelity.
6. Avoid introducing unnecessary frameworks, new truth stores, or unrelated
   architecture changes. Reuse the existing SQLite foundation and Prime APIs.

## Development and delivery

- Typical local workflow (PowerShell, from repository root):
  ```powershell
  git status --short
  git branch --show-current
  python -m pytest -q TESTS/prime/test_memory_unification_m10.py
  python BENCHMARKS/memory_gate/runner.py --dry-run
  ```
- Run relevant focused tests plus the broadest available suite, and identify
  environment-only collection failures separately.
- Final delivery for a milestone: one coherent integration, a single `.patch`
  relative to the specified base, in-chat PowerShell commands, audit of changed
  files, tested results and known limitations. No zip archive or unnecessary
  command-only README attachments.
- Never merge, rewrite history, delete user memory, or claim a successful live
  Qwen evaluation without direct evidence and authorization.
