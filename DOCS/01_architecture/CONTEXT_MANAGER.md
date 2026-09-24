# PRIMORA Intelligent Context Manager

## Purpose

Conversations can run for months or years, but a model can only think about a
bounded window of tokens at once. The Context Manager answers one question:

> **What should Aurora think about right now?**

It keeps the **complete raw conversation forever** (source of truth) and builds
a small, budgeted, relevant **working view** (active context) for each model
call. Removing something from the working view never deletes it from storage.

## Architecture

```
Amir → Aurora.respond(messages, session_id)
         ├─ manager.reconcile(snapshot, session)   # idempotent persist
         ├─ manager.build(...) → ContextView       # budgeted selection
         ├─ Brain.generate(view.messages, max_output_tokens=view.allowed_output)
         ├─ [PrimeRequest path] Prime → PrimeResult
         ├─ manager.reconcile(assistant + tool result)
         └─ manager.build(...) → Brain.generate(...) → final response
```

- **Owner:** Prime. `Prime.__init__` constructs the `ContextManager` (same
  pattern as `CapabilityRegistry`) and exposes it read-only. Aurora receives
  it and calls it; Aurora never implements storage, scoring, or budgeting.
- **Brain independence:** the manager never imports Brain. It receives an
  injected `token_counter` callable (supplied by Aurora from Brain).
- **Memory separation:** Memory answers *"What should Aurora remember?"*;
  the Context Manager answers *"What should Aurora think about right now?"*
  A minimal optional `MemoryProvider` seam exists (`retrieve()`); V1 ships it
  as a no-op. `PRIMORA/prime/memory.py` is untouched.

## Data model (`models.py` — records only, no logic)

- `StoredMessage`: one verbatim persisted row — `message_id` (rowid),
  `session_id`, `seq`, `role`, `content`, `timestamp`, `fingerprint`
  (`sha256(session|seq|role|content)`), `importance` (0.0–1.0, set at ingest
  from user-authority + keyword signals).
- `ContextItem`: a candidate for the working view + its score breakdown.
- `ContextView`: final ordered `messages`, `input_tokens`, `allowed_output`,
  `session_id`, `diagnostics` (list of machine-readable event strings).
- `DigestRecord`: `digest_id`, `session_id`, `source_ids`, `seq_start`,
  `seq_end`, `created_at`, `digest_type`, `content` (verbatim source spans).
- `BudgetConfig`: `input_target=1000`, `input_ceiling=1100`,
  `output_reserve=128`, `total_budget=1228` (all env-overridable).
- `MemoryProvider`: `Protocol` with `retrieve(session_id, query, limit)` —
  optional, defaults to no-op.

## Store (`store.py`)

SQLite at `PRIMORA/data/context/aurora_context.db` (directory auto-created),
WAL mode, per-operation connections (thread-safe for `ThreadingHTTPServer`),
parameterized SQL, index on `(session_id, seq)`.

- **Identity:** `UNIQUE(session_id, seq, fingerprint)`. Re-ingesting the same
  snapshot is a no-op (`INSERT OR IGNORE`). The same text at different
  positions yields distinct rows. Same `(session, seq)` with a *different*
  fingerprint is stored as a separate row plus a `seq_conflict` diagnostic —
  raw history is append-only and never overwritten or deleted.
- **Search:** FTS5 virtual table when the runtime build supports it (probed
  with try/except at startup; never claimed otherwise), transparent LIKE
  fallback otherwise.
- **Digests** live in their own table referencing source message IDs/ranges.

## Scoring (`scoring.py` — pure, deterministic, independently testable)

`score(text, role, seq, now_seq, query_terms, selected_texts)`:

- **relevance** — word-overlap between candidate and query (current + most
  recent user request); user-role weight ×2 via authority multiplier.
- **recency** — `1 / (1 + age_in_messages)`.
- **importance** — stored per-message value from keyword signals
  (`remember`, `important`, `always`, `never`, `we decided`, `agreed`,
  `preference`, `constraint`) with **user-authored ×1.0 / assistant ×0.4**
  authority split. Assistant confidence alone scores nothing durable.
- **redundancy penalty** — max Jaccard similarity against already-selected
  items, subtracted so near-duplicates lose.

Ties break by sequence number: fully deterministic, no randomness, no model.

## Digest system (`digest.py` — extractive only, no LLM)

Splits older messages into sentences (regex), scores them against the query,
picks top spans that fit a token allowance, groups them by source message
with IDs. Output is verbatim source text only — a digest **cannot** contain a
fact absent from its sources. Record carries full provenance. A future
model-assisted summarizer can replace `build_extractive_digest()` without
touching store consumers.

## Retrieval

Queries = current user request + most recent user request. Candidates come
from FTS5/LIKE over older rows (excluding already-selected seqs), ranked by
the scorer, admitted only while budget remains, re-emitted in chronological
order. Historical ranges are never dumped wholesale.

## Budget system

Assembly order, each step metered by the injected counter: system instruction
(mandatory, counted) → current user turn (mandatory; alone-over-ceiling →
controlled failure, message preserved, no inference) → newest turns →
active/recent → ranked historical hits → digests → memory hook (no-op).
Fingerprint dedupe across sections. Compaction/retrieval engage **before**
the hard ceiling. `allowed_output = min(max_new_tokens, total − input)`;
total input + output never exceeds the configured total. Env overrides:
`PRIMORA_AURORA_CONTEXT_TARGET/CEILING/OUTPUT_RESERVE/TOTAL`.

## Failure behavior

| Failure | Response |
|---|---|
| Retrieval error | Drop retrieval, keep recent + current |
| Digest error | Drop digests, keep recent + retrieved |
| DB unavailable | In-memory pass-through of current request + `store_unavailable` diagnostic |
| Oversized single message | Controlled failure, raw preserved, no inference |
| Token counter raises | Conservative char-based estimate + diagnostic |

Raw history is never at risk in any path.

## Session model

**Session: "Which conversation is this?"** A session is a lightweight identity
row (`sessions` table: `session_id`, `created_at`, `last_activity_at`,
optional `title`, `status` active/archived). One continuous browser
conversation keeps one `session_id` (persisted in `localStorage`, sent with
each request, echoed back in every response); the New-chat button mints a
fresh UUID. Legacy callers without an id keep working on `"default"`.

**Session vs history:** session identity is independent from long-term
history. Messages are permanently bound to their session, sessions never
delete or rewrite rows, and retrieval may span sessions — but session A's
rows are never treated as session B's recent history. Cross-session hits are
capped, ranked below same-session material, and tagged
`[from another conversation]` so provenance survives into the prompt.

**History: "What has ever happened?"** The append-only message tables across
all sessions. Nothing here is ever mutated by session or state operations.

## Session scope and resolution

State rows carry an explicit `scope`: a session id, or `"global"`. A
session-scoped state belongs only to that session: two sessions may hold
different current values for the same concept (Session A `test_phrase =
OLDSKOOL`, Session B `test_phrase = BYLAW`), corrections supersede only
within the same scope, and no session's correction propagates elsewhere. No
session is ever treated as global automatically.

Question routing, in order:
- **Session-specific** ("for Session A", or a session title): that session's
  states first (matched by id or title substring, normalized).
- **Current-session** ("what is the test phrase?"): current session (+global)
  only. Other sessions' values are never substituted.
- **Explicit cross-session** ("in our other sessions", "each session", …):
  per-session blocks — `[states session <id> — current] concept = value
  (src:…)` — never flattened into one generic list.
- **Global** ("what is Aurora's current name?"): `global`-scoped rows.
- **Historical** ("what was the old…"): `superseded` rows, tagged as such.

## Session metadata authority

The `sessions` row is authoritative for current session metadata (`id`,
name/title, `created_at`, `last_activity_at`, `status`). "What is this
session called?" is answered from the record, never from old chat mentions —
a rename (`rename_session`, id stable, old title appended to
`previous_titles`) immediately changes the answer, while the old name stays
discoverable only as history (`[session history] … was previously called …`,
or verbatim raw mentions). The model never writes session names implicitly;
renames happen only through the explicit API. Assembly order enforces it:
session metadata → current scoped state → active context → historical state
→ digests → raw retrieval.

## State model

**State: "What do we currently believe is true?"** A `states` row carries
`state_id`, `session_id`/`scope`, `concept`, `value`, `source_ids`,
`created_at`, `status` (`current`/`superseded`/`unresolved`),
`supersedes_id`, and `authority` (`user`/`assistant`).

**Authority:** only user-authored text can create or change current state.
Assistant claims never become current on their own; an assistant claim that
contradicts tracked current state is recorded as `unresolved`, never as
truth. When the user explicitly establishes a new value, the prior current
row is marked `superseded` (linked via `supersedes_id`) and kept as history.

**Supersession:** explicit user corrections ("the new X is Y", "change X to
Y", "from now on…", "instead use…", "X is no longer correct") transition
old→superseded and create the new current row. Bare invalidations retire the
current row without replacement. Raw messages and existing digests are never
touched; old digests are left in place (current state disambiguates them at
assembly time by being listed first).

**Conflict handling:** two values with no explicit correction coexist as
`unresolved` — the system never silently picks one. An explicit user
correction resolves by supersession.

**Scopes:** `session` (default; stored under the session id) and `global`
(user text containing "for all conversations", "globally", etc.). Smallest
abstraction that fits: no ontology beyond that.

**Concept aliasing:** candidate concepts resolve onto already-tracked ones by
exact match first, then longest whole-word containment ("test phrase" and
"phrase" unify; "passphrase" does not). Deterministic, word-boundaried,
documented here so future strangeness is traceable, not mysterious.

## Current vs historical information

Context assembly prefers current state: query-relevant `current` rows are
listed as `[state current src:…] concept = value` ahead of digests.
`superseded` rows appear only on explicit historical queries (markers like
*previous/prior/before/old/was/history*), tagged `[state previous …]
concept was: value (superseded)`. `unresolved` rows appear when relevant,
tagged as conflicting, never as established fact.

## Cross-session retrieval

Same-session retrieval first (existing ranking), then a capped (3),
provenance-tagged cross-session pass with per-session recency. Session
isolation holds: other sessions inform, never impersonate, the current one.

## Aurora integration

`Aurora.respond(messages, session_id="default")`: reconcile → build →
generate → (optional Prime block → `request()` → record tool result →
rebuild with fresh budget) → generate → return. Both Brain calls go through
the manager. `Brain.generate` accepts per-request `max_output_tokens`
(default preserves legacy behavior); `Brain.count_tokens` is the shared
counter (manager never loads a second tokenizer). Brain's internal trim stays
as a defensive backstop.

## Future extension points

Model-assisted summarizer (swap digest builder), real `MemoryProvider`
wiring, per-session budget profiles, digest pruning policies, FTS ranking
upgrades — all behind the existing seams; no contract changes required.

---

## File roles (ELI5 → technical)

- **`models.py`** — *ELI5:* the shapes of the boxes everything comes in.
  *Technical:* frozen dataclasses + `MemoryProvider` Protocol; zero I/O.
- **`store.py`** — *ELI5:* the permanent notebook that never erases a page.
  *Technical:* SQLite/WAL persistence, idempotent ingest, FTS5-or-LIKE search.
- **`scoring.py`** — *ELI5:* the librarian deciding which old notes matter now.
  *Technical:* deterministic relevance/recency/importance/redundancy scorer.
- **`digest.py`** — *ELI5:* photocopying the important sentences onto one card.
  *Technical:* extractive sentence selection with full source provenance.
- **`manager.py`** — *ELI5:* the desk organizer laying out only today's work.
  *Technical:* reconcile → select → budget → view orchestration + fallbacks.
- **`CONTEXT_MANAGER.md`** (this file) — the map of the whole system.
