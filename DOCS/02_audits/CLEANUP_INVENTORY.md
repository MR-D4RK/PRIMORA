# PRIMORA Cleanup Inventory (Read-Only Audit — 2026-10-13)

> Moved from the project root to `DOCS/02_audits/` in Stage 2E (2026-10-14).

Status: **inventory only**. Nothing was modified, deleted, or moved to produce this
report. The production DB (`PRIMORA/data/context/aurora_context.db`), its backup,
and all model/data files are untouched (production digest verified unchanged by the
earlier repair work; see MEMORY_AUDIT.md).

---

## 1. Git state (where the repo stands)

`git status --porcelain` shows **20 modified** tracked files and a large **untracked**
set. This is unrelated to the four Oct 13 live-failure fixes but means a commit is
pending before any cleanup branch work.

Modified (tracked) files:
`PRIMORA/aurora/{brain,identity,loop,citations,claims}.py`, `PRIMORA/main.py`,
`PRIMORA/prime/*`, `PRIMORA/specialists/__init__.py`, Verse QML, and
`TESTS/verse/*`. Untracked additions include new capability tools
(`specialists/tools/{research,learn,recall,consult,use_tool}.py`), new prime
context modules (`{hybrid,entities,entity_formation,semantic,...}.py`), new
`{consult,learning,research,toolbox}/` packages, `RESOURCES/`, `MEMORY_AUDIT.md`,
`DOCS/01_architecture/*.md`, and most of `TESTS/`.

`.gitignore` covers: `PRIMORA/data/` (the runtime DB), `*.log`, `.env`, model
dirs (`RESOURCES/aurora/v0.1/model/`, `.../aurora_qlora/`), one large GGUF,
`NOTES/`, `.venv`, `__pycache__`. **It does not cover** `*.db`/`*.db-wal`/`*.db-shm`
outside `PRIMORA/data/`, `logs/` (only `*.log` files), root `*.log`, backup
`.bak` files, or `%TEMP%` session DBs. A stray `*.db` dropped in the project root
would be committed.

---

## 2. Entry points / launchers

| Path | What it is |
|---|---|
| `PRIMORA/main.py` | Canonical module entry (`python -m PRIMORA.main`). Builds provider + research backend + Aurora + Prime, starts the Verse GUI or `--cli` loop. |
| `launch_aurora_llamacpp.py` | Python live-test driver. Snapshots the production DB, builds an **isolated** temp-store ContextManager, runs the capability checklist. Supports `--chat`. |
| `launch-aurora-llamacpp.ps1` | PowerShell wrapper: starts/connects llama-server, sets env, invokes the driver. `-Chat`, `-KeepServer`, `-Port`, `-NoPause`. |
| `TESTS/base_only_launch.py` | Test helper that forces `PRIMORA_OLLAMA_MODEL`. Not a product entry point. |
| root `test_alloc.py` | Scratch script. Not referenced by the product. |

Only **one real launcher pair** exists (`launch_aurora_llamacpp.py` +
`launch-aurora-llamacpp.ps1`); no `.bat` files. There is **no** second product
launcher — the duplication problem is *backend selection inside one entry point*,
not multiple entry-point scripts.

---

## 3. Backend / provider variants (how many ways to run)

**Aurora generation** (`PRIMORA/aurora/brain.py:52` `aurora_backend_from_env`):
`PRIMORA_AURORA_BACKEND` ∈ {`transformers` (default), `openai`/`llamacpp`}.
- `AuroraBrain` — transformers + bitsandbytes + QLoRA adapter. Slow start, needs HF model/adapter/tokenizer on disk.
- `OpenAIBrain` — llama.cpp llama-server `/v1/chat/completions`. Live path in use.

**Prime fallback provider** (`PRIMORA/main.py:81` `prime_provider_from_env`):
`PRIMORA_PROVIDER` ∈ {`ollama` (default), `openai`}.
- `OllamaProvider` (`PRIMORA/prime/providers.py:16`) — `PRIMORA_OLLAMA_URL` default `127.0.0.1:11434`.
- `OpenAICompatibleProvider` (`providers.py:81`) — `PRIMORA_OPENAI_URL` default `localhost:8080/v1`.

**Research provider** (`PRIMORA/main.py:39` `research_provider_from_env`):
`PRIMORA_RESEARCH_PROVIDER` ∈ {`ddg` (default), `searxng` (needs fixed
`PRIMORA_RESEARCH_ENDPOINT`)}. SearXNG has an auto-starting service
(`SearXNGService`).

Net: **2 generation backends × 2 Prime providers × 2 research backends**, chosen
per-process via env. The live target is `openai`/`openai`/`ddg`. Every other
combination is reachable but only `ollama` and `transformers` are the legacy
defaults that nothing enforces against.

---

## 4. Memory stores (how many places memory lives)

| Store | Location | Size | Contents |
|---|---|---|---|
| **Production context DB** | `PRIMORA/data/context/aurora_context.db` (+ `-shm` 32KB, `-wal` 0B) | 454,656 B | messages=314, states=4, entities=1, vectors=5. **Keep.** |
| Production backup | `PRIMORA/data/context/backups/aurora_context.db.20261013T102431Z.bak` | 454,656 B | Pre-launch snapshot (created by the new `snapshot_production_db`). Keep as the one known-good backup. |
| Temp session DBs | `%TEMP%\aurora_llamacpp_*\live_context.db` — **11 dirs** (Oct 13) + ~40 older `aurora_*` dirs (Sep 23–Oct 12: `eval`, `validation`, `repeat_root`, `recall`, `holdout`, `base_only`, `offload`, `verify`, …) | 163KB–491KB each | Isolated harness DBs. Largest live-test copies hold 316–335 messages (a full snapshot of production + new turns). |

Memory is written by exactly one component: the Context Manager stack
(`PRIMORA/prime/context/{store,manager,mind,hybrid,isolation}.py`). `isolation.py`
exposes `production_db_path`/`production_digest`/`harness_session` so the harness
never opens production for write. There is **one** real memory implementation; the
rest are throwaway copies.

---

## 5. Leftovers, stubs, dead weight

- **0-byte stubs**: `PRIMORA/prime/memory.py`, `PRIMORA/aurora/memory.py` — empty files. The real memory lives in `prime/context/`. These are dead.
- **Root scratch/logs**: `test_alloc.py` (1.2KB), `verse_crash.log` (10.8KB), `verse_startup.log` (5.2KB). Stale diagnostics.
- **`logs/launch-aurora-llamacpp.log`** — session transcript from the launcher; `*.log` is gitignored but the `logs/` dir itself is not.
- **`%TEMP%` junk** — ~51 `aurora_*` directories (Sep 23–Oct 13), a few MB total, never cleaned.
- **`.pytest_cache/`, `__pycache__/`** — ignored, harmless.
- **`MEMORY_AUDIT.md`** — 45KB audit doc; keep (current). Lives in `DOCS/02_audits/` since Stage 2E.

---

## 6. Tests that depend on transformers / temp-copy / Ollama

- **Transformers (can't run while llama-server holds VRAM)**: `TESTS/test_aurora_model.py`, most of `TESTS/training/*`, `TESTS/validation/*` (diagnose/investigate/verify scripts import `AutoTokenizer`).
- **OllamaProvider**: `TESTS/prime/test_ollama_provider.py`, `TESTS/verse/test_main_startup.py`, `TESTS/prime/test_consult_specialist.py` (asserts `OllamaProvider` is *not* imported on the openai path).
- **Temp-copy isolation (correct, keep)**: nearly every `TESTS/prime/test_*` and `TESTS/aurora/*` uses `tempfile.TemporaryDirectory(prefix=...)` to build a private ContextManager. This is the intended pattern, not a defect — it is what keeps tests off production.
- **Pre-existing failures (not cleanup targets)**: `TESTS/specialists/test_calculate.py` (import error), `TESTS/verse/test_main_startup.py` (3 failures on `provider._model` when `OllamaProvider` is patched out).

---

## 7. Recommendation table (target: 1 launcher, 1 backend = llama.cpp via OpenAIBrain, 1 real DB + automatic pre-launch backup)

| Item | Path | Rec | Rationale |
|---|---|---|---|
| Production context DB | `PRIMORA/data/context/aurora_context.db` | **KEEP** | The single real memory store. |
| Pre-launch backup dir | `PRIMORA/data/context/backups/` | **KEEP** | Automatic snapshot before launch; add a retention rule (keep last N). |
| Live launcher pair | `launch_aurora_llamacpp.py` + `launch-aurora-llamacpp.ps1` | **KEEP** | Already does isolated-store + backup; make this *the* entry. |
| Module entry | `PRIMORA/main.py` | **KEEP** | Needed for Verse GUI / `python -m`. Should just default to `openai` provider + `openai` Aurora backend instead of ollama/transformers. |
| `OpenAIBrain` | `PRIMORA/aurora/brain.py` | **KEEP** | The one backend. |
| `OpenAICompatibleProvider` | `PRIMORA/prime/providers.py` | **KEEP** | Prime fallback speaking to same server. |
| `DuckDuckGoResearchProvider` | `PRIMORA/research/` | **KEEP** | Default research backend. |
| `AuroraBrain` (transformers) | `PRIMORA/aurora/brain.py` | **ARCHIVE** | Keep model files & code on disk until you approve removal; stop selecting it by default. Do not delete weights. |
| `OllamaProvider` | `PRIMORA/prime/providers.py` | **ARCHIVE** | No longer the default; retain for one release as an escape hatch, then drop. |
| `SearXNG*` research | `PRIMORA/research/` | **ARCHIVE** | Optional alternate; harmless but not on the live path. |
| 0-byte stubs | `PRIMORA/prime/memory.py`, `PRIMORA/aurora/memory.py` | **DELETE** | Dead; real code is in `prime/context/`. |
| Root scratch | `test_alloc.py`, `verse_crash.log`, `verse_startup.log` | **DELETE** | One-off diagnostics, not referenced. |
| `%TEMP%\aurora_*` dirs | `C:\Users\M\AppData\Local\Temp\aurora_*` | **DELETE** | ~51 throwaway harness DBs, safe to purge (outside project). |
| `logs/launch-aurora-llamacpp.log` | `logs/` | **ARCHIVE** | Rotate; add `logs/` to `.gitignore` (currently only `*.log` is ignored, not the dir). |
| `TESTS/prime/test_ollama_provider.py` | TESTS | **ARCHIVE** | Keep while `OllamaProvider` is archived; delete with it. |
| `TESTS/verse/test_main_startup.py` | TESTS | **FIX** | Update to exercise the `openai` provider path (fixes 3 pre-existing failures). |
| transformers-based training/validation tests | `TESTS/training`, `TESTS/validation`, `TESTS/test_aurora_model.py` | **ARCHIVE** | Still valid while model files remain; group so they are excluded from the default `pytest` run. |
| `.gitignore` | root | **FIX** | Add `logs/`, `*.db-wal`, `*.db-shm`, `*.bak` (outside data/backups), so a stray DB/backup is never committed. |

**One-sentence target state:** one launcher (`launch-aurora-llamacpp.ps1` → driver) +
`PRIMORA/main.py` defaulting to `PRIMORA_AURORA_BACKEND=openai` /
`PRIMORA_PROVIDER=openai`, transformers+Ollama archived (not deleted) until you
approve, one production DB with an automatic pre-launch snapshot, and the temp
store / stubs / root scratch purged.

---

*Inventory complete. No files were changed. Awaiting your review before any KEEP/ARCHIVE/DELETE action is executed.*
