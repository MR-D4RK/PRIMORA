# Aurora Memory Architecture — Current State

**Status:** CURRENT — describes the memory system as implemented in the codebase.
**Scope:** Read-only inspection. This document records what exists today. It
does **not** propose or begin M2 (Advanced Personal Memory).
**Last verified against code:** `PRIMORA/prime/context/` and
`PRIMORA/aurora/` (see file reference guide at the end).

> Accuracy note. This document is grounded in the actual source, not in
> earlier design notes. Where an older document disagrees with the code, the
> code wins and the discrepancy is called out explicitly (§9).
>
> In particular, `DOCS/01_architecture/MEMORY_HYBRID_RETRIEVAL.md` labels M6
> hybrid retrieval "PROPOSED / NOT IMPLEMENTED". That is **out of date**:
> hybrid retrieval is implemented and live by default (§4, §9).

---

## 1. Concise overview

Aurora's memory is a **single append-only SQLite store** (`aurora_context.db`)
fronted by a layered pipeline:

1. **Raw messages** are persisted idempotently, keyed by a content fingerprint,
   and never deleted (except by whole-store reset, which the application does
   not do).
2. **Derived structured memory** is formed deterministically from *user-authored
   text only*: typed states (M4), entities/aliases/relationships (M5), and
   extractive digests. No LLM participates in extraction.
3. **Retrieval** assembles a bounded context window from (a) recent raw
   messages, (b) authoritative current state, and (c) older material selected
   by lexical, semantic, and structural channels fused with reciprocal rank
   fusion (M6).
4. **Working memory** is a runtime-only, non-persistent scratch set; durable
   writes happen only through an explicit promotion call.
5. **Authority is asymmetric**: user text can establish facts; assistant text
   can never establish or overwrite a fact. An assistant claim that conflicts
   with an established user fact is recorded as `unresolved`, never as truth.

The system is deliberately conservative. Capability that could infer, merge,
summarize with a model, or promote scope automatically is either explicitly
absent or left dormant behind a hook.

---

## 2. Component map

```
Verse (QML/UI)
  └─ VerseApp.clear_conversation / _restore_conversation
        │  writes CONVERSATION_CLEARED marker / reloads transcript
        ▼
Aurora.respond(messages, session_id, on_delta)
  └─ CognitiveLoop.run(...)
        │  per segment:
        │    Mind.build(...)      ── context assembly
        │    Brain.generate(...)  ── model call
        │    Prime.handle(...)    ── capability execution
        │  at end:
        │    Mind.reconcile(...)  ── persist + form memory
        ▼
Mind  (façade / cognitive boundary)
  ├─ ContextManager
  │    ├─ Store            (append-only SQLite, FTS5/LIKE, fingerprint)
  │    ├─ build()          (assemble working context)
  │    ├─ reconcile()      (persist → M4 states → M5 entities)
  │    ├─ _retrieve()      (M6 hybrid, default ON)
  │    ├─ promote_to_candidate()  (explicit durable write from WorkingMemory)
  │    └─ load_transcript()/next_seq()/clear marker handling
  ├─ M4 state formation      (state.py, slots.py, slots.json)
  ├─ M5 entity formation     (entity_formation.py → entities.py, relationships.py)
  ├─ M6 retrieval            (hybrid.py → queries.py, semantic.py)
  ├─ Digests                 (digest.py)
  └─ WorkingMemory           (working.py, runtime-only)
        ▼
SQLite: messages, sessions, digests, states, candidates,
        entities, entity_aliases, relationships,
        retrieval_vectors, retrieval_index_meta, messages_fts
```

Authority/lifecycle invariants shared across the pipeline:

- Only `role == "user"` messages form state (M4) or entities (M5).
- `authority == "user"` is required for a state to be retrieval-eligible.
- `confidence` is evidential support, **never** an ordering or admission input.
- Current-state validity is half-open (`[valid_from, valid_to)`); supersession
  links rows, it never deletes.
- Persistence is idempotent and best-effort; a store failure must not break a
  response.

---

## 3. Full data flow

### Stage 1 — Input

- UI (`Verse`) sends the visible conversation to `Aurora.respond(...)`.
- `Aurora` prepends its instruction/system framing via `_with_instruction`
  and delegates to `CognitiveLoop.run`.
- The loop may run several segments (model → capability → model …); each
  segment rebuilds context from the Mind.

### Stage 2 — Context assembly (`Mind.build` → `ContextManager.build`)

`build()` reconciles the incoming messages (see Stage 4), then assembles a
bounded `ContextView` from ordered layers:

1. System/instruction framing.
2. Recent raw conversation turns (most recent messages, within a token budget).
3. Structured authority: session info and current scoped state, rendered as
   attributed lines (e.g. `[remembered] …`, `[possible conflict] …`).
4. Retrieved older material (M6 hybrid over older messages, digests, states,
   entities/relationships).
5. Digests of compacted history.
6. Selected persistent memories (candidates/current) promoted into context.

Each layer is capped; items are scored deterministically
(`scoring.py`: relevance, recency, importance, redundancy penalty). Order is
stable so repeated builds of the same state are byte-identical.

### Stage 3 — Model + execution

- `Brain.generate(context.messages)` is called (streaming optional; the loop
  passes `on_delta` through when present).
- If the model emits a `prime_request`, `Prime.handle` dispatches it to a
  registered capability; the result is appended and the loop runs another
  segment.
- Working memory may be updated during the turn, but that state is
  process-local.

### Stage 4 — Persist and form memory (`Mind.reconcile` → `ContextManager.reconcile`)

`reconcile` is the durable write path:

1. **Persist messages.** Each message is hashed as
   `sha256(session_id \x00 seq \x00 role \x00 content)` and inserted with a
   `UNIQUE` fingerprint. Re-persisting the same turn is a no-op. `seq` is
   monotonic per session. The store is append-only.
2. **M4 state formation** (`state.py`), user messages only:
   - Candidate states are extracted deterministically (regex/slot driven, no
     LLM).
   - Slots (`slots.json`, `slots.py`) route known personal facts (e.g.
     favorite color/music) into global-scope slots so they are visible from
     every session.
   - Confirmed current state is created; contradicting user text supersedes
     the prior row with a half-open validity interval.
   - Assistant text can create a *candidate* observation at most; a conflict
     with an established user fact becomes `unresolved`, never `current`.
3. **M5 entity formation** (`entity_formation.py`), user messages only:
   - Only explicit phrasing forms entities (`my <type> is <NAME>` or
     `<NAME> is a <type>`).
   - Exact normalized-name matches reuse an entity within a scope. No
     fuzzy merging, no automatic promotion from session to global.
   - `part-of` is confirmed (explicit assertion); `concerns` links between a
     state and an entity remain candidates (inference, not assertion).

Reconcile is best-effort: failures are isolated so a response is never
blocked by a memory write.

### Stage 5 — Retrieval (`ContextManager._retrieve`, M6)

When older material must be consulted, retrieval runs through
`hybrid.py::hybrid_retrieve` (enabled by default; see §5). Order is
load-bearing and must not be reordered:

```
hard gates → channels → RRF → relative floor → bounded result
```

- **Hard gates** are absolute and deterministic: authority must be `user`;
  `unresolved`/`rejected` states are excluded; temporal filters apply; the
  gate uses no `confidence` value.
- **Channels:**
  - `lexical` — FTS5 when available, LIKE fallback (`queries.py`).
  - `semantic` — local embeddings (`semantic.py`) for states and entities
    only; degrades silently to "no semantic channel" if the model/index is
    unavailable.
  - `entity` / `relationship` — structural traversal over M5 rows.
- **Fusion** — reciprocal rank fusion (`RRF_K = 60`).
- **Relative floor** — post-fusion floor (`EVIDENCE_FLOOR_RATIO = 0.725`).
- **Bound** — capped candidate count and output.

### Stage 6 — Digests / compaction (`digest.py`)

Old messages may be compacted into **extractive** digests: verbatim source
sentences with `[src:id]` provenance, grouped by message. No model, no
invented facts. The builder contract is stable so a future model-assisted
summarizer could replace it without changing consumers (not implemented).

### Stage 7 — Working memory (`working.py`)

`WorkingMemory` holds task, objective, temporary decisions, adopted active
memory references, raw evidence, and files. It is:
- never written to SQLite,
- never shared across instances,
- bounded by item count and token estimate,
- evicted deterministically (evidence → announced memories → decisions →
  files; task/objective pinned).

Promotion to durable memory happens only via
`ContextManager.promote_to_candidate`; there is no automatic promotion.

### Stage 8 — Clear conversation

`VerseApp.clear_conversation` does **not** delete messages or states. It
appends a `CONVERSATION_CLEARED` system marker row through `reconcile`.
Readers such as `Store.fetch_conversation` and transcript reload only
consider rows *after* the newest marker, so the visible thread is empty
across a restart while raw history and memory states remain intact. An
in-flight response is invalidated by an epoch guard.

### Stage 9 — Restart / continuity

`VerseApp._restore_conversation` calls `ContextManager.load_transcript`
(which also honours the clear marker) and restores the visible thread.
Because messages and structured memory are durable, continuity survives a
process restart without any special case for memory formation.

---

## 4. Component inventory

Classification is evidence-based, per the request: *implemented and active*,
*implemented but partial/limited*, *stored but unused*, or *planned/absent*.

| Component | Classification | Where (code) | Notes / evidence |
|---|---|---|---|
| Raw conversation messages | Implemented + active | `store.py` (`messages`, `append`), `manager.py` (`reconcile`) | Append-only; `UNIQUE(fingerprint)`; never deleted in normal operation. |
| Session continuity | Implemented + active | `store.py` (`sessions`, `ensure_session`, `fetch_conversation`), `manager.py` (`load_transcript`) | Fetches only `user`/`assistant` rows after the newest clear marker. |
| Working / context memory | Implemented + active | `manager.py` (`build`), `scoring.py` | Bounded, deterministic scoring; recomputed each build. |
| Active / persistent states (M4) | Implemented + active | `state.py`, `manager.py` (`_extract_states`), `slots.py`, `slots.json` | User-authored only; current/superseded/unresolved lifecycle; half-open validity. |
| Memory candidates & confirmation | Implemented + active | `store.py` (`candidates`), `manager.py` (`confirm_pending`, `promote_to_candidate`) | Create-then-confirm; assistant cannot confirm a user fact. |
| Digests / summaries | Implemented + active | `digest.py`, `manager.py` (`_add_digests`, `maybe_compact`) | Extractive only, provenance-tagged; no LLM. |
| Entities & aliases (M5) | Implemented + active (limited) | `entities.py`, `entity_formation.py`, `manager.py` (`_extract_entities`) | Explicit wording only; exact identity; session scope by default. |
| Relationships (M5) | Implemented + active (limited) | `relationships.py`, `entity_formation.py` | `part-of` confirmed; `concerns` left as candidate. |
| FTS5 search | Implemented + active | `store.py` (`messages_fts`), `queries.py` | LIKE fallback when FTS5 is unavailable. |
| Hybrid retrieval (M6) | Implemented + active | `hybrid.py`, `semantic.py`, `queries.py`, `manager.py` (`_retrieve`, `_admit_hybrid`) | **Live by default**; rollback switch `PRIMORA_HYBRID_RETRIEVAL=0`. |
| Semantic index | Implemented + active (degradable) | `semantic.py` (`retrieval_vectors`, `retrieval_index_meta`) | `all-MiniLM-L6-v2`, CPU, offline; states+entities only; degrades to lexical when unavailable. |
| WorkingMemory | Implemented + active (runtime-only) | `working.py` | Non-persistent; promotion is explicit. |
| MemoryProvider hook | Stored but unused | `models.py` (`MemoryProvider`, `noop_memory`) | Default no-op returns `[]`; not wired to a real backend. |
| Model-assisted summariser | Planned / absent | `digest.py` docstring | Contract reserved; not implemented. |
| Semantic entity inference / auto-merge | Planned / absent | `entity_formation.py` docstring | Explicitly "NOT IMPLEMENTED". |
| Session→global scope promotion | Planned / absent | `entity_formation.py` docstring | Explicitly "NOT IMPLEMENTED". |

---

## 5. Hybrid retrieval status (and the doc discrepancy)

- **Code:** `hybrid.py` implements hard gates, four channels, RRF, relative
  floor, and bounded output. `semantic.py` implements the local embedding
  index. `manager.py::_retrieve` calls `hybrid_mod.hybrid_retrieve` when
  `self._hybrid_retrieval` is true.
- **Default:** `ContextManager.__init__` sets `hybrid_retrieval = True` unless
  `PRIMORA_HYBRID_RETRIEVAL` is `0`/`false`/`no`/`off`. This is the documented
  rollback switch.
- **Discrepancy:** `MEMORY_HYBRID_RETRIEVAL.md` describes M6 as proposed and
  not implemented. That document predates the implementation and is
  superseded by `hybrid.py` / `semantic.py`. Treat the code as authoritative.

---

## 6. What the system demonstrably remembers

The following behaviours follow directly from the implemented logic. They are
statements about mechanism, not claims about observed model output.

### 6.1 A favourite song, stated explicitly

- User: "My favourite song is *X*." → M4/slot extraction creates a
  **global-scope** state (favourite-song slot), authority `user`, status
  `current`. It is visible from other sessions.
- User later: "Actually my favourite song is *Y*." → the prior state is
  **superseded** (`valid_to` set); *Y* becomes current. Neither row is
  deleted.
- Assistant says "Your favourite song is *Z*" → at most a candidate
  observation; it cannot become `current` and cannot supersede the user
  state.

### 6.2 Explicit fact vs. recommendation

- Explicit user assertion ("my project is PRIMORA", "I prefer dark mode") is
  eligible to establish state/entity.
- A recommendation or model-inferred preference ("you might like jazz") is
  not user-authored authority and does not establish current state. It can at
  most be recorded as a candidate observation.

### 6.3 Retrieval over older material

- Recent turns are always in-window. Older material is admitted only through
  retrieval (M6 by default) or an extractive digest.
- If the embedding model/index is unavailable, the semantic channel is
  skipped; lexical and structural channels continue. Recall may drop; the
  system does not error.

### 6.4 Conflict handling

- When assistant text contradicts an established user fact, the system records
  the conflict as `unresolved` and renders it as a possible conflict; it does
  not overwrite the user fact and does not assert the assistant claim as
  truth.

### 6.5 Raw history vs. persistent retrieval

- **Raw history** is the complete append-only transcript in `messages`.
- **Persistent memory** is the derived layer (states, entities,
  relationships, candidates, digests). Retrieval/context uses the derived
  layer and selected raw evidence; it does not silently treat the entire raw
  transcript as established fact.
- Clearing the conversation hides raw history from the UI going forward
  without deleting it and without touching memory states.

---

## 7. Limitations and unknowns (current system)

These are observed properties of the implementation, not defects to fix in
this phase.

- **Explicit-only extraction.** M4/M5 rely on patterns and slots. Facts
  phrased indirectly are not formed. This is intentional for auditability.
- **No inference or merging.** Entity aliases are exact; similar names are
  never merged; no semantic entity inference. This caps recall and avoids
  false merges.
- **Semantic channel is optional/degradable.** It covers only states and
  entities (not raw messages or digests) and silently disables itself when
  the local model/numpy is unavailable. Effective retrieval quality therefore
  depends on environment.
- **Scope promotion is manual.** Entities default to session scope; there is
  no automatic session→global promotion.
- **Digests are extractive.** They cannot paraphrase or generalize; they only
  select source sentences.
- **WorkingMemory is ephemeral.** Anything not explicitly promoted is lost on
  process exit.
- **`MemoryProvider` is a dormant seam.** No real provider is wired; the
  default returns no memories.
- **Doc drift.** Some architecture docs (notably the M6 doc) lag the code.
  This document should be preferred for current behaviour.

**Open questions for M2 (not decided here):**

- Should extraction move beyond explicit phrasing, and if so under what
  authority/audit model?
- Should scope promotion become automatic, and what guards prevent cross-
  session leakage?
- Should raw messages be embedded, or should the semantic layer stay limited
  to structured memory?
- How should model-assisted summarisation be introduced without breaking the
  verbatim-provenance contract?

---

## 8. Current system vs. future M2

| Concern | Current (this document) | Future M2 (not started) |
|---|---|---|
| Extraction | Deterministic, explicit-user-only | TBD; must preserve authority invariants |
| Entity identity | Exact match, session scope | TBD; promotion/merge policy undecided |
| Summaries | Extractive, verbatim provenance | Possibly model-assisted behind same contract |
| Retrieval | Hybrid lexical+semantic+structural, default ON | Calibration/tuning TBD |
| Provider seam | Dormant `MemoryProvider` no-op | Possibly wired to a real backend |
| Working memory | Ephemeral, manual promotion | Unchanged unless M2 says otherwise |

M2 work must not be assumed to already exist. Nothing in the CURRENT columns
above is a proposal; it is verified code.

---

## 9. File reference guide

Simple descriptions first, then technical anchors.

- **`PRIMORA/prime/context/store.py`** — *The filing cabinet.* Append-only
  SQLite store: schema, idempotent append via fingerprint, FTS5 with LIKE
  fallback, session/transcript fetches, clear marker handling. Key constant:
  `CONVERSATION_CLEARED`.
- **`PRIMORA/prime/context/manager.py`** — *The assembler.* `ContextManager`:
  `build` (layered context), `reconcile` (persist → M4 → M5), `_retrieve`
  (M6), `_admit_hybrid`/`_admit_lexical`, `_add_*` context layers,
  `promote_to_candidate`, `load_transcript`, `next_seq`, hybrid rollback
  switch.
- **`PRIMORA/prime/context/mind.py`** — *The front desk.* `Mind` façade over
  `ContextManager`; `build`/`reconcile`; exposes the `CognitiveContext`.
- **`PRIMORA/prime/context/models.py`** — *The forms.* Immutable records:
  `StoredMessage`, `ContextItem`/`ContextView`, `DigestRecord`, state/entity/
  relationship models, and the dormant `MemoryProvider`/`noop_memory`.
- **`PRIMORA/prime/context/scoring.py`** — *The ranking rules.* Deterministic
  relevance, recency, importance, redundancy; `tokenize`, `singular`.
- **`PRIMORA/prime/context/state.py`** — *The fact checker.* Deterministic M4
  extraction, confirmation, supersession, and conflict handling. No LLM.
- **`PRIMORA/prime/context/slots.py` / `slots.json`** — *The personal-fact
  forms.* Config-driven global-scope slots (favourite colour/music, etc.).
- **`PRIMORA/prime/context/entities.py`** — *The address book.* Entity and
  alias CRUD, exact-identity rules, status lifecycle.
- **`PRIMORA/prime/context/entity_formation.py`** — *The clerk.* Explicit-
  pattern M5 formation from user messages; writes only via entities/
  relationships APIs.
- **`PRIMORA/prime/context/relationships.py`** — *The link registry.*
  Typed edges (`part-of`, `concerns`), candidate/current lifecycle.
- **`PRIMORA/prime/context/queries.py`** — *The search desk.* Lexical/FTS
  queries and candidate fetching used by retrieval.
- **`PRIMORA/prime/context/hybrid.py`** — *The fusion engine.* M6 gates,
  channels, RRF, relative floor, bounded results.
- **`PRIMORA/prime/context/semantic.py`** — *The embedding index.* Local
  `all-MiniLM-L6-v2`; states+entities only; derived, lossless-to-delete
  cache; degrades gracefully.
- **`PRIMORA/prime/context/digest.py`** — *The clipper.* Extractive digests
  with verbatim provenance; compaction.
- **`PRIMORA/prime/context/working.py`** — *The whiteboard.* Runtime-only,
  bounded, non-persistent working set.
- **`PRIMORA/prime/context/isolation.py`** — *The seal.* Production
  fingerprint/digest, read-only guard, test session isolation.
- **`PRIMORA/prime/prime.py`** — *The wiring.* Builds `ContextManager` and
  `Mind`, binds the recall capability, owns context/mind accessors.
- **`PRIMORA/aurora/aurora.py`** — *The persona.* `respond`, instruction
  framing, warmup; hands off to the cognitive loop.
- **`PRIMORA/aurora/loop.py`** — *The turn loop.* Segment build → generate →
  execute → persist; final reconcile.
- **`PRIMORA/aurora/brain.py`** — *The model client.* `generate`,
  thinking-strip, optional streaming.
- **`PRIMORA/verse/app.py`** — *The window.* `clear_conversation` (marker
  row + epoch guard), `_restore_conversation` (transcript reload).
- **`TESTS/prime/`** — Test evidence for memory, retrieval, state, entity,
  and mind behaviour (see the suite names in that directory).

---

## 10. Verification notes

- This document was produced read-only. No production DB was modified; no
  Python/QML behaviour was changed; nothing was committed.
- Production DB integrity was checked read-only (`mode=ro`). During this
  session the content-based fingerprint was
  `8b84160c3d8b7d0f2a04ffab021cc4bba214d5285a884ad14b78c0eb6c788dc1`
  (messages=268, states=14, candidates=6) and was **identical across
  repeated checks**, confirming this documentation task performed no writes.
  The DB's last write was earlier the same day and predates this session; the
  counts advanced beyond an earlier baseline due to intervening normal usage,
  not this task.
- Where code and older docs disagree, §5 and §9 record the discrepancy.
