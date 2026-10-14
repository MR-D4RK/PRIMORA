# Stage 2 — Cleanup Closure Record

**Closed:** 2026-10-14 · **Result:** complete, verification green · **Next:** Stage 3 (Memory)

## What Stage 2 did

| Sub-stage | Scope | Outcome |
|---|---|---|
| 2A | Read-only audit of source redundancy, tests, docs, dead files | Findings baseline; no changes |
| 2B | Cleanup batch 1: startup attr access, consult/use_tool `is_configured`, specialist exports, 4 zero-byte stubs deleted | Startup + exports tests added |
| 2C | Live runtime: launcher rewritten to run production-direct (no temp copies, no session override), web UI gated to GUI mode, read-only memory health check | Deterministic live driver |
| 2D | Filesystem hygiene: 27 `__pycache__` dirs / 360 pyc removed, `.gitignore` reviewed (no change) | Zero caches after runs |
| 2E | Final cleanup and closure (this record) | See below |

## Stage 2E changes

**Source**
- `PRIMORA/prime/router.py` **deleted** — `Router.route` was a pass-through returning `request.action`.
- `PRIMORA/prime/orchestrator.py` — dispatches straight to `Execution`; duplicate `validate_action` removed.
  **One authoritative validation path remains: `Prime.handle`** (rich, actionable error naming the
  registered set and the `use_tool` / `consult_specialist` wrappers).
- `PRIMORA/prime/execution.py` — provider fallback **retained and documented**: a capability
  registered without a specialist is provider-backed; this is a tested contract, not dead code.
- `PRIMORA/main.py` — stale comment updated (validation is Prime.handle's, not the router's).

**Tests**
- `TESTS/test_aurora_model.py` — rewritten as a real pytest test: heavy imports deferred into the
  body; skips on missing CUDA, VRAM < 6 GB free, or CPU/disk dispatch refusal. Previously it
  crashed collection (module-level model load). Passes standalone (~44 s); skips in-suite when a
  prior test holds VRAM. **Collection is now clean: 1896 collected, 0 errors.**
- `test_ollama_provider_malformed_response` flake **root-caused and fixed**: mock handlers never
  read the POST body, so the server closed with unread data → Windows sends RST → the already-
  written response is discarded → intermittent `WinError 10053` instead of the malformed JSON.
  Fix: consume the request body (and send `Content-Length`) in every mock handler.
  Same bug fixed in `test_openai_provider_malformed_response`.
  Verified deterministic: 90/90 and 50/50 consecutive clean runs.
- `TESTS/prime/test_router.py` and the router test in `test_calculate.py` deleted (obsolete);
  dead `_make_test_prime` helper (stale signature, called nowhere) removed from
  `test_instruction.py`.

**Retained extension interfaces** (intentional, not obsolete — kept and documented):
- `SpecialistRegistry.from_env` (`consult/catalogue.py`) — env-driven specialist catalogue config.
- `ToolRegistry.with_builtins` + `toolbox/builtin` — built-in tool set, used by validation scripts
  and capability-protocol tests.

**Docs**
- `DOCS/architecture/AUDIT_01/02_*.md` — banners added marking them historical/superseded.
- Root reports moved to `DOCS/02_audits/`: `CLEANUP_INVENTORY.md`, `PRIMORA_FULL_AUDIT.md`,
  `MEMORY_AUDIT.md`; in-document path claims updated.

## Verification (final, 2026-10-14)

- Collect: **1896 tests, 0 errors** (was 1894 + 1 collection error).
- Full suite `pytest TESTS -q`: **1886 passed, 10 skipped, 0 failed** (~3m45s).
  Skips: 9 × no local SearXNG (pre-existing), 1 × `test_aurora_model` VRAM held by earlier test.
- Targeted suites (prime + specialists): 1616 passed / 9 skipped.
- `git diff --check`: no whitespace errors. Export imports OK; all 7 capabilities registered.
- Caches after runs: 0 pyc / 0 `__pycache__` / 0 `.pytest_cache`.
- Production DB byte-identical throughout: sha256 `add411b4…768`, 159744 B, memory tables 0 rows;
  backup untouched: sha256 `31b8cb39…346e`, 454656 B.
- Git status delta in 2E = exactly the changes listed above (142 lines total, all pre-existing
  uncommitted work plus these edits; nothing committed, as instructed).

## Known limitation

- `logs/launch-aurora-llamacpp.log` remains locked by PowerShell PID 26000 — a bare interactive
  shell started from Explorer (6:51 PM, 2026-10-13), **not safely identifiable** as a launcher
  process, so it was preserved rather than killed. Delete the log after closing that window.
  This is the sole Stage 2 leftover.
