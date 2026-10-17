# PRIMORA — Full Architecture, Health, and Cleanup Audit

**Date:** 2026-10-13 (evening) · **Mode:** read-only · **Only file written:** this report.

> Moved from the project root to `DOCS/02_audits/` in Stage 2E (2026-10-14).

> **Addendum (2026-10-17, post-recovery):** `python -m PRIMORA.main` is now
> the single official entry point (GUI + Verse; `--cli` = terminal only) and
> owns the llama-server lifecycle via `PRIMORA/model_server.py`. The
> launcher pair this audit calls the "canonical live path"
> (`launch-aurora-llamacpp.ps1` + `launch_aurora_llamacpp.py`) has been
> retired; live verification uses `TESTS/live_smoke.py`. The audit text
> below is kept unchanged as a point-in-time record.

All claims below were verified in this session against the live workspace (source
reads, read-only SQLite inspection via `mode=ro`, one safe pytest run). Where a
prior document (`CLEANUP_INVENTORY.md`, `MEMORY_AUDIT.md`, agent reports)
disagrees with direct observation, the disagreement is called out explicitly.

---

## 1. Executive summary (plain language)

PRIMORA is a personal AI assistant ("Aurora") built as three layers: a **UI and
agent loop** (`PRIMORA/aurora`), a **capability router with tools** (`PRIMORA/prime`
+ `PRIMORA/specialists`), and a **SQLite-backed long-term memory**
(`PRIMORA/prime/context`). It runs today against a local `llama-server` serving
Heretic Qwen3-8B (Q4_K_M) through an OpenAI-compatible HTTP API. The cognitive
loop asks the model to emit a structured `<prime_request>` when it wants a tool
(recall/research/learn/calculate…), executes it, feeds the result back, and
repeats up to 8 cycles.

**Overall health: the core is sound and unusually well-guarded for a project this
size, but the live path is only partially wired, and one test actively recreates
the production database file.**

The good news, verified this session:
- Memory architecture (append-only messages, supersession chains, authority
  gates, hybrid lexical+semantic+entity+relationship retrieval with RRF fusion,
  hard fail-closed philosophy) is genuinely sophisticated and mostly correct.
- 1,744 of 1,748 runnable tests pass (99.8%). The memory, retrieval, learn,
  recall, research, and isolation suites all pass.
- Production-memory isolation is designed thoughtfully (two-layer guard,
  content digest, temp-dir harnesses, conftest fallback redirect).

The bad news, verified this session:
1. **A test recreates the production DB file.** You cleaned your memory (deleted
   `aurora_context.db`); running the test suite silently recreated it as an
   empty schema-only file, because two tests in
   `TESTS/prime/test_harness_isolation.py` (lines 90, 97) deliberately construct
   `ContextStore(PRODUCTION)` *outside* the guard to prove the guard was
   released. SQLite's "create if absent" semantics turn that proof into a
   production write. **The file now exists with 0 messages — your memories are
   still gone (good), but the file is back (surprising).**
2. **The ordinary live path (`python -m PRIMORA.main`) writes production memory
   with no snapshot or backup** — only the llamacpp launcher isolates.
3. **The live path advertises tools that always fail**: `consult_specialist` and
   `use_tool` are told to the model, but their registries are empty by default.
   `learn` is invisible because no Jev is configured on the live path.
4. **No automatic production backup exists anywhere in current code.** The
   `context/backups/` directory has one file that no current code path writes.
5. **51 throwaway memory databases sit in `%TEMP%`** (Sep 23 – Oct 13), some
   containing full copies of your real conversation history — a privacy and disk
   hygiene issue.
6. **Large uncommitted work**: 21 modified + ~100 untracked files (the entire
   capability/memory/tooling upgrade since checkpoint `b052eae`) sit uncommitted
   on `master`. This is the single largest loss risk in the project.

Recommended target (detailed in §8): commit the pending work, make
`launch-aurora-llamacpp.ps1` the one live launcher pointing at the real DB with
an automatic pre-launch backup, stop advertising empty registries, fix the two
guard-release tests, purge temp DBs, then archive transformers/Ollama legacy
paths — without deleting any model weights.

---

## 2. Architecture and dependency map

### 2.1 Component inventory

| Component | Path | Purpose | Size/notes |
|---|---|---|---|
| Composition root | `PRIMORA/main.py` (232 L) | Selects providers, builds Prime+Aurora, starts Web UI + CLI/GUI | `python -m PRIMORA.main` |
| Aurora agent | `PRIMORA/aurora/` | Identity, config, brain backends, cognitive loop | `loop.py` 786 L is the heart |
| Brains | `aurora/brain.py` (739 L) | `AuroraBrain` (transformers+QLoRA) / `OpenAIBrain` (llama-server) | Selected by `PRIMORA_AURORA_BACKEND` |
| Cognitive loop | `aurora/loop.py` (786 L) | Up to 8 cycles: build context → generate → extract `<prime_request>` → execute tool → correct unbacked claims/dangling citations | v0.3 |
| Identity/instruction | `aurora/identity.py`, `configuration.py` | System prompt, capability catalogues, Qwen guidance | Assembled fresh each turn |
| Citations/claims | `aurora/citations.py`, `claims.py` | Detect dangling `[N]` refs and unbacked "I searched" claims; one corrective regeneration each | New, untracked |
| Prime | `PRIMORA/prime/prime.py` (138 L) | Request validation, registry, orchestration | 7 capabilities |
| Providers | `prime/providers.py` | `OllamaProvider` / `OpenAICompatibleProvider` | Prime's *consult* backend (not Aurora's brain) |
| Capability registry | `prime/capability.py` (263 L) | calculate, recall, research, research_read, consult_specialist, learn, use_tool | |
| Memory core | `prime/context/` (15 files, ~400 KB) | SQLite store, context builder, hybrid retrieval, state/entity/relationship management, isolation | `manager.py` 105 KB, `store.py` 90 KB |
| Learning | `PRIMORA/learning/` | `jev.py` (hold/learn decisions), `provenance.py` (attestation) | Jev = the admission model |
| Specialists | `PRIMORA/specialists/tools/` | calculate, recall, research, research_read, consult, learn, use_tool | All return `PrimeResult` |
| Research | `PRIMORA/research/` | DuckDuckGo provider, SearXNG provider+service, page fetcher | Default: DDG |
| Toolbox | `PRIMORA/toolbox/` | Allowlisted tools (convert_units, describe_text, current_time, count_words) | **Never wired into production** |
| Consult catalogue | `PRIMORA/consult/` | Env-driven specialist catalogue (`from_env`) | **Only called from tests** |
| Verse GUI | `PRIMORA/verse/` | PySide6/QML immersive UI + Web UI (`aurora/interface/server.py`, port 8420) | |
| Training | `TRAINING/aurora/` | QLoRA training pipeline | Separate concern |
| Tests | `TESTS/` | ~1,750 tests, benchmarks, T1 harness, validation scripts | |
| Resources | `RESOURCES/` | v0.1 base model + QLoRA adapter (present on disk), v0.2–v0.5 configs/metrics, SearXNG compose | Weights gitignored |
| Docs | `DOCS/` (9 files, ~250 KB) | Architecture + memory design docs | Two generations (see §2.4) |

### 2.2 Communication and dependencies (live path)

```
launch-aurora-llamacpp.ps1
  └─ starts llama-server (Heretic Qwen3-8B Q4_K_M, port 8080)
  └─ env: PRIMORA_AURORA_BACKEND=openai, PRIMORA_PROVIDER=openai
  └─ launch_aurora_llamacpp.py  (live-test driver; snapshots prod → temp DB)
        └─ Prime(provider=OpenAICompatibleProvider, research=DuckDuckGo)
              ├─ ContextManager  → SQLite (temp copy on launcher path)
              ├─ Mind            → retrieval + learn admission
              └─ CapabilityRegistry (7 specialists)
        └─ Aurora(prime)
              ├─ OpenAIBrain     → POST http://127.0.0.1:8080/v1/chat/completions
              ├─ Mind (shared with Prime)
              └─ CognitiveLoop(max_cycles=8, capability_budget=4)
                    └─ loop: mind.build() → brain.generate() →
                       extract_prime_request → executor → Prime.handle →
                       specialist.handle → PrimeResult → (repeat)
```

Dependency direction is clean and acyclic:
`verse/interface → aurora → prime → specialists → context → store → sqlite`.
`research`, `toolbox`, `consult`, `learning` are leaves depended upon by
`specialists`/`prime`. No circular imports were found.

**Prime's provider vs Aurora's brain — the distinction that confuses most
readers:** Aurora's *generation* goes through the brain (`OpenAIBrain` on the
live path). Prime's *provider* (`OpenAICompatibleProvider`) is only used by the
`consult_specialist` capability (which always fails today, §3) and would be the
fallback for unregistered capabilities (dead code, §3). On the live path both
point at the same llama-server, so the duplication is currently harmless but
architecturally redundant.

### 2.3 Two entry points, one canonical

| Entry | Backend default | Memory target | Backups | Verdict |
|---|---|---|---|---|
| `python -m PRIMORA.main` (GUI/CLI/Web) | transformers + ollama | **Production, no snapshot** | none | **Legacy path today** |
| `launch-aurora-llamacpp.ps1` → driver | openai + openai (llama-server) | temp copy of production | one manual .bak exists, written by no current code | **Canonical live path** |

Recommendation: keep the ps1+driver pair as the live launcher (§8); change
`main.py` defaults later so the GUI also uses openai — not during this audit.

### 2.4 Documentation state

Two generations coexist:
- `DOCS/architecture/AUDIT_01/02_*.md` — early audits.
- `DOCS/01_architecture/*.md` (7 files, all untracked) — current design docs
  (grounding, hybrid retrieval, entity-relationship, calibration).
- `MEMORY_AUDIT.md` (`DOCS/02_audits/`, 45 KB; moved from root in Stage 2E) — the Oct 13 repair-session audit.
- `CLEANUP_INVENTORY.md` (`DOCS/02_audits/`; moved from root in Stage 2E) — the earlier cleanup inventory.

**Contradiction found:** `CLEANUP_INVENTORY.md` claims the launcher "snapshots
production into `context/backups/`" and that the backup file was created by
`snapshot_production_db`. Verified false: `snapshot_production_db`
(`launch_aurora_llamacpp.py:57-78`) writes only to the temp dir passed by the
caller; no code anywhere writes `PRIMORA/data/context/backups/`. The single
`.bak` file there predates the current snapshot function (or was copied
manually). Treat CLEANUP_INVENTORY.md §4/§7 backup claims as stale.

### 2.5 Uncommitted work, grouped by purpose

21 modified + ~100 untracked files, all uncommitted on `master`
(last commit: `b052eae "chekcpoint: Verse V0.5 working"`):

| Group | Files | Purpose |
|---|---|---|
| Capability/tools system (new) | `specialists/tools/*` (6), `capability.py`, `consult/`, `toolbox/`, `learning/`, `research/` | The entire tool layer |
| Memory system upgrade (new) | `prime/context/{entities,entity_formation,hybrid,isolation,mind,queries,relationships,semantic,working}.py` | Hybrid retrieval + isolation + entity memory |
| Cognitive loop v0.3 (new) | `aurora/{loop,citations,claims}.py` | Multi-cycle tool loop + truthfulness corrections |
| Launcher pair (new) | `launch-aurora-llamacpp.ps1`, `launch_aurora_llamacpp.py` | Live path |
| Brain fixes (modified) | `aurora/{brain,identity,configuration,aurora,__init__}.py` | OpenAIBrain, thinking-flag fix, echo filter era |
| Main wiring (modified) | `main.py`, `providers.py`, `prime.py`, `context/{manager,store,models,scoring,state,__init__}.py` | |
| Tests (~70 new files) | `TESTS/prime/test_*`, `TESTS/conftest.py`, `TESTS/aurora/test_{identity,openai_brain}.py`, benchmarks | |
| Docs (new) | `DOCS/01_architecture/*` (7), `MEMORY_AUDIT.md` | |
| GUI tweaks (modified) | `verse/qml/*.qml` (4), `AuroraVeil.frag(.qsb)` | |
| Training (new) | `TRAINING/aurora/*`, `TESTS/training/*` | |
| Scratch (root) | `test_alloc.py`, `verse_crash.log`, `verse_startup.log` | Not referenced |

This work **must be committed** before any cleanup branch work (§8, step 1).

---

## 3. Confirmed problems, suspected problems, healthy components

### 3.1 Confirmed problems (code + runtime evidence)

| # | Problem | Evidence | Impact |
|---|---|---|---|
| C1 | **Guard-release tests recreate production DB** — `test_the_guard_is_released_on_exit` and `test_the_guard_is_released_even_after_an_error` construct `ContextStore(PRODUCTION)` after the guard exits (`TESTS/prime/test_harness_isolation.py:90,97`). `ContextStore.__init__` → `_init_db()` → `executescript(_SCHEMA)` creates the file if absent (`store.py:230-254`). | Reproduced: prod DB absent → run those 2 tests → file present again (159,744 B, 0 rows). | Any test run resurrects a phantom production DB; digest checks flip from `"<absent>"` to a real hash; undermines the "fresh memory" state. |
| C2 | **`main.py` live path writes production with no snapshot** — `Prime()` → `ContextManager()` → `default_db_path()` → production; Web UI starts unconditionally (`main.py:146-169`). | Source trace; confirmed by conftest docstring's own admission that this "happened". | GUI/CLI sessions mutate real memory with zero backup. |
| C3 | **Temp snapshot DBs never cleaned** — `mkdtemp(prefix="aurora_llamacpp_")` per launch, no `rmtree` anywhere in the driver. | 51 dirs in `%TEMP%\aurora_*` (Sep 23–Oct 13), several containing full production copies (316–335 messages). | Disk creep + copies of private conversation data outside the project. |
| C4 | **No backup writer or retention** — `context/backups/` has one orphan `.bak`; no rotation code exists. | Grep + file inspection. | The "automatic pre-launch backup" the launcher comments promise does not exist on the production path. |
| C5 | **Empty registries advertised to the model** — `consult_specialist` and `use_tool` have no `is_configured` flag, so `build_capability_instruction` always lists them (`configuration.py:116-133,153`), yet `Prime` builds empty `SpecialistRegistry()`/`ToolRegistry()` (`prime.py:49-51,65`) and no caller passes tools/specialists. Every attempt fails with "none configured/registered". | Source + live checklist history. | Wasted cycles, model confusion, dead-end turns. |
| C6 | **`learn` invisible on live path** — `Prime` constructed without `jev=`; `UnconfiguredJev` → `is_configured=False` → capability omitted from prompt; any request fails closed. | `prime.py:58-62`, `learn.py:50,71-79`, `jev.py:172`. | Aurora cannot learn on the canonical live path at all. (Intentional fail-closed, but undocumented for operators.) |
| C7 | **Dual context-budget defaults behind the same env names** — `AuroraConfig` defaults 1500/1600/256/1856 (`configuration.py:255-264`) vs `BudgetConfig` 1000/1100/128/1228 (`models.py:319-323`); same `PRIMORA_AURORA_CONTEXT_*` vars feed both. | Source. | Silent misconfiguration for any direct `ContextManager` user. |
| C8 | **`PRIMORA_RESEARCH_ENDPOINT` dual meaning** — SearXNG must equal `http://127.0.0.1:8888` (`main.py:63-72`); for DDG it is the API override (`research/provider.py:126-128`). | Source. | Setting it for one backend breaks the other silently. |
| C9 | **`harness_session(root=...)` deletes caller-supplied dirs** — explicit root sets `owned=True` → `dispose()` rmtree (`isolation.py:248-262`), contradicting its docstring. | Source. | Data-loss footgun for harness authors. |
| C10 | **Reads mutate the DB** — live retrieval runs `maybe_sync_semantic_index()` which writes `retrieval_vectors` (`manager.py:1920`, `semantic.py:316-327`). | Source. | Any read-only-looking tool changes content digests. |
| C11 | **Three failing tests (pre-existing)** — `TESTS/verse/test_main_startup.py` × 3: `main.py:165` accesses `provider._model` but the test's recorder exposes `.model`. | pytest run this session. | GUI-startup regression suite is red. |
| C12 | **`test_calculate.py` fails at import** — `from PRIMORA.prime.router import CAPABILITY_MAP`; the symbol doesn't exist (`router.py` is 11 lines). | pytest collection error. | Specialist test suite unreachable. |
| C13 | **Stale comment** — `loop.py:124-127` says "default max_cycles = 2"; actual default is 8 (`loop.py:106`). | Source. | Misleads maintainers about which budget binds. |
| C14 | **Triple `PRIMORA_OPENAI_MODEL` default** — `"llama-server"` (main/brain) vs `"qwen3-8b-heretic"` (driver raw timing, ps1 alias). | Source. | Measured tok/s uses a different model string than production would. |
| C15 | **`OpenAIBrain` does no context trimming / no VRAM ceiling** — unlike `AuroraBrain`; overflow delegated to server. | Source. | Long turns can fail opaquely. |
| C16 | **Research defaults to live network** — absent `PRIMORA_RESEARCH_PROVIDER` → DuckDuckGo, contradicting the "no network without configuration" claim in `provider.py:90-94`. | Source. | Startup surprise; DDG flakiness observed Oct 13. |
| C17 | **`PRIMORA_SESSION_ID` set by launcher but inert** — only `run_cli` reads it; driver passes session explicitly. | Source. | Dead configuration. |

### 3.2 Suspected problems (need runtime confirmation)

| # | Problem | Why suspected |
|---|---|---|
| S1 | Recall echo filter (`recall.py:29-69`, token-set equality) can drop a genuine short fact identical in tokens to the query. | Rare; silent. |
| S2 | Test-fixture residue in production states — `DOCS/01_architecture/AURORA_T1_GROUNDING.md:471,526-531` says injected state lines in production were test data (that DB is now gone, but the same risk returns once real states accumulate). | Doc + design gap; eligibility has no test-residue concept. |
| S3 | `build()` re-runs `reconcile()`+retrieval on every loop cycle (≤8×/turn) — scaling concern on long sessions, not a correctness bug. | Source reading; idempotent by design. |
| S4 | Consult/specialist calls run at server-default sampling (provider sends no `temperature`/`top_p` unless payload `options` carries them), diverging from Aurora's controlled sampling. | Source. |

### 3.3 Healthy components (verified, no action needed)

- **Fail-closed philosophy** applied consistently (UnconfiguredJev, unknown
  research backend, unknown action, unknown tool, unknown specialist).
- **Memory write path**: append-only messages, `reconcile()` idempotence,
  conflict diagnostics, state supersession chains (`supersedes_id`),
  single-current invariant with loud diagnostics, user-authority gate,
  unresolved conflicts never rendered as truth, superseded values only on
  historical queries.
- **Hybrid retrieval**: 4 channels → hard gates → RRF fusion → evidence floor;
  graceful degradation to lexical on any failure; builder and recall share one
  implementation.
- **Truthfulness loop**: unbacked-claim and dangling-citation corrections each
  get exactly one regeneration; repeat-request guard; capability budget.
- **Isolation design**: conftest fallback redirect, `production_guard` double
  hook, `harness_session` digest verification, structural AST scan test.
  (Undermined only by C1's deliberate post-guard construction.)
- **Tool safety**: AST-based calculator (no eval), allowlisted toolbox,
  reserved-key blocking on `learn`, page-fetch limited to model-cited refs,
  char caps everywhere.
- **1,744 passing tests** covering memory formation/collisions/temporal/
  relationships/entities, retrieval calibration/benchmarks, all 7
  capabilities, loop protocol, identity, and OpenAIBrain.

---

## 4. Memory and runtime findings

### 4.1 Runtime path (condensed)

`ps1` → llama-server up → env → driver:
`production_digest()` → `snapshot_production_db(temp)` → patch
`ContextManager` → `Prime` + `Aurora` → checklist (no-HF-load, answer, tok/s,
isolated recall, research, digest-unchanged) → optional `--chat`.

Per turn: `_with_instruction` (system + identity + catalogues + Qwen guidance)
→ loop cycle ≤8: `Mind.build` (persist → working memory → history →
`[state current]` lines → hybrid retrieval → digests → reassembly within
budget 1856 tokens for AuroraConfig) → `OpenAIBrain.generate` (with
`enable_thinking` off by default, thinking fences stripped, empty content →
annotated failure) → `<prime_request>` extraction → `CapabilityExecutor`
(budget 4 calls/turn, repeat guard 2) → `Prime.handle` → specialist → result
appended as tool turn → corrections if the model claimed success it didn't
have. Identity block reports model/adapter paths and forbids inventing
capabilities or memories.

### 4.2 Memory end-to-end

1. **Persist**: `reconcile()` stores every snapshot row idempotently by
   `(session, seq)`; new user rows feed state extraction (M4) and entity
   formation (M5).
2. **Admission (learn/Jev)**: assistant sentence persisted as assistant (never
   auto-truth) → `Jev.evaluate` → policy override can force `hold` → pending
   candidate → `confirm_pending` (same gate as fresh extraction) → `current`.
   Provenance attested against turn evidence and user texts; self-reported
   "user" provenance downgraded when research supported the claim better.
3. **Correction**: user claims supersede; assistant contradictions recorded
   `unresolved`; superseded rows retained with lineage, excluded from normal
   retrieval.
4. **Retrieve**: hybrid (lexical+semantic MiniLM-L6-v2+entity+relationship),
   RRF fusion, gates never re-admitted by fusion; states/entities resolve to
   their verbatim source messages.
5. **Inject**: `[state current src:…]` lines (user-authority, time-valid,
   newest-per-concept), budgeted by relevance/overlap rules; older passages
   capped; cross-session messages tagged.
6. **Answer**: model told state lines are its own stored memory.

### 4.3 Production DB reconciliation (the 312 vs 314 question)

**Direct evidence gathered this session (read-only):**

| Artifact | Messages | Notes |
|---|---|---|
| `PRIMORA/data/context/aurora_context.db` (before your cleanup, earlier today) | **314** | 4 states, 1 entity, 5 vectors, 21 digests — the file the earlier audit measured |
| `backups/aurora_context.db.20261013T102431Z.bak` (current, inspected now) | **314** | Byte-identical content to the above: 3 sessions (`3455cf09…` 255 rows, `default` 57, `probe` 2), sha256 `31b8cb39…`, size 454,656 B |
| `%TEMP%\aurora_verify_ez8vfw30\live_context.db` | **312** | Snapshot taken 17:55 — two turns earlier |
| `%TEMP%\aurora_verify_j8hhltsa\live_context.db`, `oj36zbme` | **316** | Snapshots at 17:57/18:10 — two turns later |
| `%TEMP%\aurora_llamacpp_x99_vbnd\live_context.db` | 329 | 18:45 snapshot (audit-era turns) |
| `%TEMP%\aurora_llamacpp_o2_4mrww\live_context.db` | 335 | 18:51 snapshot (largest) |

**Resolution:** both numbers were correct for their files at their times.
312 = the 17:55 verify-run snapshot; 314 = production state at the 18:24
backup (and the number the earlier audit reported for the live file). Growth
of 2 messages per live turn explains 312 → 314 → 316 across successive
snapshots. There was never a discrepancy inside one file.

**Current state (verified):** the production DB **does not exist** as of this
audit's inspection at ~20:35 — only the backup remains. **After the test run
(20:39), the file reappeared**: 159,744 B, empty schema (0 messages/states/
entities), `-wal` 0 B, `-shm` 32 KB. Cause: C1 (guard-release tests). No
memories were restored; the file is a schema shell. If you want the pre-test
state back, delete `aurora_context.db`, `-wal`, `-shm` again — but fix C1
first or the next test run recreates them.

**All DB copies identified:** 1 backup (above), 11 `aurora_llamacpp_*` temp
DBs, 3 `aurora_verify_*`, ~37 older empty/temp `aurora_*` dirs (Sep 23–Oct 12,
most with no `.db` left). None were deleted or altered by this audit.

**Launcher digest check caveat:** with the DB absent (this evening),
`prod_before == prod_after` was `"<absent>" == "<absent>"` — the "production
unchanged" check passes vacuously (`isolation.py:103-109`).

### 4.4 Backup/restore/retention status

| Mechanism | Status |
|---|---|
| Pre-launch snapshot to temp (launcher) | Exists (`snapshot_production_db`), read-only source via backup API, failure → empty isolated store, never production |
| Backup to `context/backups/` | **No writer in code**; one orphan file |
| Retention/rotation | **None** |
| Restore path | **None** |
| Temp-store cleanup | **None** (C3) |
| Ordinary live path snapshot | **None** (C2) |

---

## 5. Test-health summary

**Command run (safe subset):**
`python -m pytest TESTS -q --ignore TESTS/test_aurora_model.py --ignore TESTS/training --ignore TESTS/validation --ignore TESTS/t1 --ignore TESTS/specialists/test_calculate.py`

**Result: 3 failed, 1,744 passed, 11 skipped (185 s).**

| Bucket | Detail |
|---|---|
| Failed (3) | `TESTS/verse/test_main_startup.py` — all three GUI-startup tests hit `AttributeError: 'ModelRecorder' has no attribute '_model'` at `main.py:165` (test recorder predates the print line). Pre-existing; not caused by this audit's activity. |
| Skipped (11) | Tests guarded by `if not production_db_path().exists(): pytest.skip(...)` — correctly skipped while prod DB was absent (learn/research/tool-call/safety/cognitive-loop live probes). They would run now that C1 recreated the empty file; none of them write to it (they use temp stores; the skip guard is a "only if a real DB exists" opt-in for optional deeper checks). |
| Not run (by design) | `TESTS/test_aurora_model.py` (needs transformers weights + VRAM held by llama-server), `TESTS/training/*` (training pipeline), `TESTS/validation/*` (one-off diagnostic scripts importing AutoTokenizer), `TESTS/t1/` (T1 harness, uses its own isolation but slower), `TESTS/specialists/test_calculate.py` (import error C12). |
| Collection error (1) | `test_calculate.py`: `CAPABILITY_MAP` no longer exists in `router.py`. |

**Coupling assessment:**
- Transformers-coupled: `test_aurora_model.py`, `test_brain_integration.py`
  (mocked but exercises load path), most `TESTS/validation/*`, `TESTS/training/*`.
  All correctly excluded from the default run when GPU is busy.
- Ollama-coupled: `test_ollama_provider.py` (self-contained HTTP mock, passes),
  `test_verse_main_startup.py` (fails on the recorder issue, not Ollama itself),
  `test_consult_specialist.py` (asserts OllamaProvider is *not* imported — passes).
- Old-launch-path-coupled: `TESTS/base_only_launch.py` (manual harness, not in
  pytest run), `standalone_chat_v03.py` (manual).
- Temp-copy isolation: universal and correct — every memory test builds a
  private `ContextStore`/`harness_session`/`tmp_path`; conftest redirects the
  `db_path=None` fallback for all 1,744 tests.

**Isolation coverage gaps:**
1. C1 — the two guard-release tests are *intentional* production constructions
   and are the gap. Structural scan test exempts `test_harness_isolation.py`
   itself (`test_production_isolation_regression.py:93-95`), so this cannot
   self-detect.
2. `production_guard` is opt-in; ordinary tests rely on conftest only. A test
   that constructs `ContextStore("PRIMORA/data/context/anything.db")` by
   literal relative path would evade both (the structural scan only checks for
   the absolute production path string). Low risk today.
3. No test asserts "production file does not exist stays non-existent" —
   which is exactly the property C1 violates.

**Missing integration tests (worth adding later):**
- End-to-end live turn against llama-server that asserts the *real* production
  DB digest is unchanged **and the file's existence state is unchanged**.
- Learn admission with a configured `ModelJev` on the launcher path (today the
  live path never wires Jev, so learn is untested end-to-end in production
  wiring).
- Backup creation + restore round-trip (no code exists to test yet).

---

## 6. Complete KEEP / ARCHIVE / DELETE / INVESTIGATE recommendations

Nothing below was executed. "Dependencies" = verified imports/launch-path/test
references checked this session unless noted.

### KEEP (necessary or currently useful)

| Path | Reason | Dependencies | Data-loss risk if removed |
|---|---|---|---|
| `PRIMORA/` (all production packages) | The product | launchers, tests | n/a |
| `PRIMORA/prime/context/` (all 15 files) | Memory core; hybrid retrieval is the differentiator | manager/mind/specialists/tests (~40 memory tests) | Destroys memory system |
| `PRIMORA/specialists/tools/` (7 tools) | Live capabilities | capability registry, loop | Breaks loop |
| `PRIMORA/learning/{jev,provenance}.py` | Admission policy (currently unconfigured live, but core design) | learn specialist | |
| `PRIMORA/research/` | DDG/SearXNG/fetcher | research + research_read, main.py | |
| `PRIMORA/aurora/{loop,citations,claims,identity,configuration,brain,aurora}.py` | Agent core | main, tests | |
| `PRIMORA/verse/` + `aurora/interface/server.py` | GUI/Web UI | main.py | |
| `launch-aurora-llamacpp.ps1` + `launch_aurora_llamacpp.py` | Canonical live launcher pair | each other; driver imports PRIMORA | |
| `PRIMORA/data/context/backups/aurora_context.db.20261013T102431Z.bak` | **Only surviving copy of 314 real memories** (3 sessions incl. your main conversation, 4 states: name=amir, fav color, car preferences) | none (data) | **Irreplaceable conversation history** — copy it somewhere safe before any disk cleanup |
| `RESOURCES/aurora/v0.1/model/` + `v0.1/aurora_qlora/` | Base model + trained adapter + optimizer checkpoint | transformers backend, identity block, TRAINING | Weights cannot be recreated without retraining |
| `TESTS/` (active suites + `conftest.py`) | 1,744-test safety net | CI/manual | |
| `TESTS/conftest.py` | Global production-redirect for tests | autouse fixture | Weakening it re-exposes production to tests |
| `PRIMORA/specialists/__init__.py`, `contract.py`, `orchestrator.py`, `router.py`, `execution.py` | Prime plumbing (router/execution are thin but wired) | prime.py | |
| `DOCS/01_architecture/*.md` (7 files) | Current design truth (grounding, hybrid retrieval, calibration) | referenced by audit docs | |
| `MEMORY_AUDIT.md` | Oct 13 repair evidence | none | Historical evidence |
| `TRAINING/` | QLoRA pipeline + its tests | separate concern | |
| `RESOURCES/searxng/` | Optional research backend config | searxng provider/service | |
| `RESOURCES/aurora/v0.2..v0.5/` configs+metrics | Training run history (small JSON) | TRAINING history | Historical evidence |
| `TESTS/t1/` + `TESTS/t1/reports/` | T1 evaluation harness + baseline evidence | validation docs | Historical evidence |
| `logs/launch-aurora-llamacpp.log` (rotate, don't delete) | Session evidence incl. Oct 13 verification | none | Low; rotate instead |
| `.gitignore` | Already ignores `PRIMORA/data/`, models, `*.log`, `.env` | — | Fix gaps (§8) |

### ARCHIVE (legacy/experimental — preserve, remove from runtime path)

| Path | Reason | Dependencies | Migration needed first |
|---|---|---|---|
| `PRIMORA/prime/providers.py` `OllamaProvider` + `main.py` ollama branch | Legacy backend; live path is llama-server | `test_ollama_provider.py`, `test_verse_main_startup.py`, `base_only_launch.py` | Update those tests + `main.py` defaults; keep one release as escape hatch |
| `AuroraBrain` (transformers path in `brain.py`) + `PRIMORA_AURORA_BACKEND=transformers` default | Slow, VRAM-conflicting, not the live path | `test_brain.py` (mocked), `test_brain_integration.py`, identity block, TRAINING warmup assumptions | Keep weights on disk; change default only after GUI wired to openai; never delete `RESOURCES/aurora/v0.1` |
| `PRIMORA/research/searxng.py` + `service.py` + `RESOURCES/searxng/` | Optional backend, heavier than DDG | `test_searxng_provider.py`, `test_research_service.py` | Only if you decide DDG-only permanently |
| `PRIMORA/toolbox/` + `use_tool` capability | Built-ins never wired into production | `test_tool_calling.py` (passes against registry-injected tools) | Either wire `ToolRegistry.with_builtins()` into `Prime(...)` (preferred) or archive both together |
| `PRIMORA/consult/` catalogue `from_env` | Only called from tests | `test_consult_specialist.py` | Wire from_env into main or archive with consult |
| `TESTS/validation/*.py` (16 scripts) | One-off diagnostic scripts, many import transformers | none (manual) | Move to `ARCHIVE/validation/`; they are evidence of past investigations |
| `TESTS/training/generate*.py`, `seed_bank_v*.py`, `responses_v4.py`, `new_seeds_v*.py` (~1 MB of generation scripts) | Dataset-build experiments superseded by v5 pipeline | `TESTS/training/test_dataset_v*` reference some | Confirm v5 is canonical, then archive v1–v4 generators |
| `TESTS/base_only_launch.py`, `TESTS/standalone_chat_v03.py` | Manual pre-launcher harnesses | none (manual) | Superseded by launch_aurora_llamacpp.py |
| `DOCS/architecture/AUDIT_01/02_*.md` | Superseded by DOCS/01_architecture | none | Optional; small |
| `RESOURCES/aurora/v0.1_smoke_test/`, `v0.2..v0.5` large artifacts if any appear later | Experiment configs (currently small JSON — KEEP as-is; listed for completeness) | — | — |

### DELETE (demonstrably disposable)

| Path | Reason | Verified references | Migration/verification first |
|---|---|---|---|
| `PRIMORA/aurora/cognition.py`, `aurora/learning.py`, `aurora/memory.py`, `prime/memory.py`, `prime/control.py`, `prime/infrastructure.py` (all 0 bytes) | Empty stubs; real code lives in `loop.py`, `learning/`, `prime/context/` | Zero imports found (grep verified) | None |
| `PRIMORA/specialists/tools/__init__.py` (0 bytes) | Empty package marker — **KEEP actually** (needed for package import). Do not delete. | imports of tools work | n/a — reclassify KEEP |
| `test_alloc.py` (root, 1.3 KB) | Scratch allocator probe | None | None |
| `verse_crash.log`, `verse_startup.log` (root, ~16 KB) | Stale GUI crash logs | None (`logs/` dir is the current log location) | None |
| `%TEMP%\aurora_*` (51 dirs, ~11 MB) | Throwaway harness DBs; some contain full copies of real conversation data | None (outside project) | **Copy nothing out; delete all.** Highest privacy-value cleanup |
| `TESTS/prime/test_ollama_provider.py` | Only when OllamaProvider is archived (pair with ARCHIVE row) | — | After archiving Ollama |
| `PRIMORA/specialists/test_calculate.py` broken import file | **Not delete — FIX**: `CAPABILITY_MAP` import must be rewritten to the registry API | registry API in `capability.py` | Small fix, not deletion |

### INVESTIGATE (uncertain / needs your decision)

| Path | Question | Risk if mishandled |
|---|---|---|
| `PRIMORA/data/context/backups/*.bak` (314-msg backup) | Restore into production DB before next live session? (Your memories live only here.) | Only copy of real history; restore = your choice to resume memory vs stay fresh |
| `RESOURCES/aurora/v0.1/aurora_qlora/checkpoint-600/` (contains `optimizer.pt`, `rng_state.pth`, ~30+ MB) vs final adapter | Keep training checkpoint for resume, or is the final adapter enough? | Resume capability vs disk |
| `TESTS/t1/reports/*.json` (~460 KB of baseline/holdout evidence) | Needed as historical baselines or archivable? | Evaluation history |
| `TESTS/prime/RETRIEVAL_BENCHMARK.md` + `retrieval_benchmark*.py`, `retrieval_calibration.py` (100 KB) | Keep as living calibration harness (recommended) or archive after calibration is settled? | Re-tuning capability |
| `NOTES/` (gitignored: model notes, first-convo transcript, a 1.1 MB PNG) | Personal notes; `first convo with aurora.txt` may be sentimental evidence | Personal data — your call |
| `PRIMORA/verse/qml/AuroraVeil.frag.qsb` (compiled shader) | Regenerable from `.frag` via qsb tool? If yes, could be build output | GUI rendering |
| `PRIMORA/specialists/__init__.py` incomplete exports (5 of 7 specialists) | Complete the exports or leave (registry imports directly) | Cosmetic confusion |
| Uncommitted ~120 files | **Commit strategy**: one big commit vs logical series (capability system / memory system / launcher / GUI / training / docs / tests) | Loss risk until committed — investigate only *how*, not *whether* |

---

## 7. External-project ideas worth adopting

Verified against actual docs/source (Letta docs, Mem0 docs + OSS source,
Graphiti README + `graphiti_core` source + arXiv 2501.13956, Hermes Agent
docs + repo) this session. `[V]` = verified from fetched source; `[R]` =
recommendation.

| # | Idea | Source | PRIMORA already has | Delta worth doing |
|---|---|---|---|---|
| 1 | **Dedup context before extraction** — feed top-k existing same-concept memories into the extraction prompt so the model proposes hold/merge instead of duplicates | Mem0 v3 pipeline `[V]` | `idx_candidates_scope_concept`, pending checks | Pass confirmed states of the same concept into `Jev.evaluate`'s prompt; small change, cuts `candidate_pending` noise |
| 2 | **Hash-normalized exact dedup** | Mem0 MD5 dedup `[V]` | Verbatim storage | Casefold+whitespace hash column on the learn path |
| 3 | **Human approval queue for held items** — Hermes `write_approval` + pending list + target pinning `[V]` | `candidates.status='pending'`, `confirm_pending` already exist | Only the surface is missing: a `/pending`-style command (CLI or Verse panel) listing pendings with approve/reject, refusing if the row changed |
| 4 | **Post-turn reviewer producing pendings only** — Hermes background self-improvement review `[V]` | Storage + promotion exist | End-of-turn/idle hook that proposes pendings (never truths), token-capped, optionally on a cheap model — preserves `AUTO_ADMIT={user}` |
| 5 | **Render supersession lineage in injected state** — Graphiti invalidates-with-history, every fact traces to episodes `[V]` | `supersedes_id`, temporal queries exist | Presentation only: show `superseded ⟨old⟩` + source ids so the model sees the timeline |
| 6 | **Capacity-visible bounded memory** — Hermes `MEMORY [67% — 1,474/2,200 chars]` header `[V]` | `working.py` bounded block | Add usage header (display first) for observability |
| 7 | **LLM compaction behind existing digest contract** — Letta self-compaction keeps identical prefix for cache hits `[V]` | `digest.py` is extractive, contract-ready for a model summarizer | When long sessions matter: same `(content, source_ids)` contract, reuse agent prefix, cheap model, extractive fallback |
| 8 | **"Boost, don't expand" for weak signals** — Mem0: only semantic results are candidates; BM25/entity only boost `[V]` | RRF where lexical is its own channel | Run ONE calibration experiment (semantic-gated candidates; lexical/entity as rank-only) against `MEMORY_RETRIEVAL_CALIBRATION.md`; adopt only if FP@3 improves without recall loss |
| 9 | **Mention-frequency importance** | Graphiti `episode_mentions` reranker `[V]` | `importance_of` exists for digests only | Secondary rank signal for states/entities — only after #8, outside hard gates |

**Explicitly not worth it for a single-user local setup:** graph DB (Neo4j)
for entities `[V]` — PRIMORA's `relationships` table + relationship channel
already covers the structural signal in SQLite; LLM/cross-encoder rerankers —
PRIMORA's deterministic RRF is a feature (Graphiti itself advertises no
LLM-in-the-loop reranking `[V]`); multi-tenant scoping; a second temporal-
extraction LLM pass (PRIMORA writes `valid_from/valid_until` deterministically
at admission); fully-automatic extraction as the *primary* write path — it
would collide with the `AUTO_ADMIT={user}` authority gate that PRIMORA's
whole trust model rests on (keep auto-extraction as pendings via #4).

**What PRIMORA already does better than these projects (verified):** hard
authority gates on read (only user-authority states inject as truth);
unresolved conflicts rendered as `(conflicting, not established)` rather than
silently picked; provenance attestation that *downgrades* self-reported claims;
fail-closed capability advertising. These are rare and worth keeping.

---

## 8. Recommended cleanup order, with risks and prerequisites

**Phase 0 — Safety (no code changes)**
1. Copy `backups/aurora_context.db.20261013T102431Z.bak` to a safe location
   outside the project (e.g. Documents). *Risk: none. Prerequisite: none.*
2. Delete the phantom empty `PRIMORA/data/context/aurora_context.db` (+`-wal`,
   `-shm`) **only after Phase 1 item 3**, or it comes back on the next test
   run. *Risk: none (0 rows).*
3. Delete the 51 `%TEMP%\aurora_*` directories. *Risk: loses throwaway test
   DBs only; some contain conversation copies you may not want lingering.
   Prerequisite: none.*

**Phase 1 — Stop the bleeding (small, testable code fixes; commit after)**
4. Commit the current ~120 uncommitted files first (one commit or a short
   logical series). *Risk: none vs losing them. Prerequisite: decide message.*
5. Fix C1: change the two guard-release tests to construct
   `ContextStore(os.path.join(tmp_path, "released.db"))` instead of
   `ContextStore(PRODUCTION)` — the guard-release property is still proven by
   the construction *succeeding* outside the guard. Add a regression test:
   "production file existence state is unchanged across the whole suite"
   (absent stays absent). *Risk: very low. Prerequisite: none.*
6. Fix C11 (`provider._model` in `main.py:165` → tolerate `model` attr or
   give the recorder `_model`) and C12 (`test_calculate.py` registry API).
   *Risk: low. Prerequisite: none.*
7. Fix C5: give `ConsultSpecialist`/`UseToolSpecialist` an `is_configured`
   property (False when registries are empty) so the prompt stops advertising
   dead ends. *Risk: low; model behavior improves. Prerequisite: none.*
8. Update `.gitignore`: add `logs/`, `*.db-wal`, `*.db-shm`, and a rule
   keeping only `PRIMORA/data/context/backups/*.bak`. *Risk: none.*

**Phase 2 — One launcher, one backend, one DB (focused fixes, no rewrites)**
9. Make `main.py` honor the live configuration: default
   `PRIMORA_AURORA_BACKEND=openai` and `PRIMORA_PROVIDER=openai` (env still
   overridable; transformers/ollama remain selectable). Wire
   `ToolRegistry.with_builtins()` and (if you want consult) a minimal
   `SpecialistRegistry` into `Prime(...)`, or accept and document that only
   the 5 core capabilities are live. *Risk: medium — GUI users on transformers
   need an explicit opt-in path. Prerequisite: Phase 1 green.*
10. Add automatic pre-launch backup on the **production** path: call
    `snapshot_production_db` into `context/backups/aurora_context.db.<UTC>.bak`
    before the first write when `main.py` runs against production; keep last
    N=5, delete older. *Risk: low. Prerequisite: decide N.*
11. Clean temp dirs at launcher exit (`shutil.rmtree` in a `finally`). Add
    C3 fix. *Risk: low.*
12. Decide restore: either copy the 314-message backup into
    `PRIMORA/data/context/aurora_context.db` (resume memory) or leave absent
    (stay fresh) — and make the launcher create-and-use the real DB
    deliberately rather than a temp copy, per your stated desired end state.
    Keep tests on temp stores (conftest already ensures this). *Risk: medium —
    this is the memory-continuity decision. Prerequisite: your call.*
13. Rename `PRIMORA_RESEARCH_ENDPOINT` split (C8): e.g.
    `PRIMORA_SEARXNG_ENDPOINT`; fix dual budget defaults (C7): single source
    of truth for `PRIMORA_AURORA_CONTEXT_*`. *Risk: low-medium (env var
    rename). Prerequisite: grep all docs/tests for old names.*

**Phase 3 — Archive (after Phase 2 has soaked)**
14. Archive transformers/Ollama/SearXNG/toolbox-as-dead-code per §6 ARCHIVE
    rows: move to `ARCHIVE/` or gate behind explicit env opt-ins; update the
    coupled tests. **Never delete `RESOURCES/aurora/v0.1` weights.**
15. Delete the verified-empty stubs and root scratch files (§6 DELETE).
16. Move `TESTS/validation/*` and superseded training generators to
    `ARCHIVE/`.

**Phase 4 — Memory quality (adopt external ideas)**
17. Items #1–#3 from §7 (dedup context, hash dedup, pending-approval
    command) — small, high leverage. Then #4–#6. Run the #8 calibration
    experiment only when you next tune retrieval.

Each phase ends with a full safe-suite run (`1744+ passed`, 0 production-digest
changes, 0 resurrected production file) before the next begins.

---

## 9. The first three highest-priority actions

1. **Commit the uncommitted work (~120 files).** Everything else in this
   report is worthless if the capability/memory/launcher upgrade is lost to a
   disk mishap. One commit or a short logical series; verify with
   `git status` clean afterwards.
2. **Fix the guard-release tests (C1) and add an existence-state regression
   test.** Until this is fixed, every test run resurrects a phantom
   `aurora_context.db`, digest checks behave differently on consecutive runs,
   and your "fresh memory" state silently flips. Then delete the current empty
   `aurora_context.db`/`-wal`/`-shm` again (they contain nothing).
3. **Restore or formally abandon the 314-message backup, and give the live
   path a real pre-launch backup.** The only copy of your real memories is
   `context/backups/aurora_context.db.20261013T102431Z.bak`. Decide: copy it
   back to `aurora_context.db` to resume memory, or keep it archived and stay
   fresh. Either way, implement Phase 2 item 10 so the production DB is
   automatically snapshotted before every normal live launch — currently no
   code creates backups at all.

---

### Audit verification statement

- Files **written**: this report only (`PRIMORA_FULL_AUDIT.md`) plus one temp
  helper script at
  `C:\Users\M\AppData\Local\Temp\opencode\audit_db.py` (outside the project).
- Files **modified/moved/deleted/renamed** in the project: **none by direct
  action**.
- **Side effect observed and disclosed (not a direct action):** the pytest run
  required by §4 recreated `PRIMORA/data/context/aurora_context.db` (empty
  schema, 0 rows) via the two guard-release tests (C1). Production *content*
  was not modified — no memories exist in the recreated file, and the 314-row
  backup is untouched (sha256 `31b8cb39ab3e45c0ffa4df2302a1ddd67c3a033f6a6024
  1477a071f750c7346e`, verified read-only after the run).
- Model weights, adapters, datasets, training artifacts: **not touched**.
- `%TEMP%` directories: **not deleted** (listed for your cleanup).

*Audit complete. Stopping here — awaiting your review. No cleanup or fixes will
be implemented until you say so.*
