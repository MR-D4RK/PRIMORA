# MEMORY_AUDIT — Aurora / PRIMORA memory system

> Moved from the project root to `DOCS/02_audits/` in Stage 2E (2026-10-14).

Read-only audit. No project file was created, modified, or deleted by this
audit except this document (`MEMORY_AUDIT.md`). The application was **not**
run; no training was run. Test commands that were executed are listed in §7
and they use isolated throwaway databases (the production database SHA-256
was identical before and after every run — see §7).

Everything below is quoted from the code with `file:line` references.
Function/class names are given so each quote can be located.

---

## 0. Where the memory system lives

```
PRIMORA/prime/context/
    store.py             SQLite schema + append-only storage (ContextStore)
    state.py             M4: deterministic state extraction & transitions
    entity_formation.py  M5: entity/alias/relationship formation from user text
    entities.py, relationships.py   M5 lifecycle (create → confirm → …)
    scoring.py           pure scoring functions (relevance/recency/importance/redundancy)
    hybrid.py            M6: hybrid retrieval (channels, RRF fusion, evidence floor)
    semantic.py          embedding index (sentence-transformers, CPU)
    digest.py            extractive digests (no LLM)
    manager.py           ContextManager: reconcile / build / recall / learn
    mind.py              Mind facade + CognitiveContext
    models.py            dataclasses incl. BudgetConfig
    queries.py           structural (entity/relationship) channel queries
PRIMORA/learning/jev.py  Jev: natural language → structured memory candidates
PRIMORA/specialists/tools/{learn,recall}.py   capability surface
PRIMORA/aurora/{aurora,loop,brain,configuration,identity}.py   prompt + generation
```

`PRIMORA/prime/memory.py` and `PRIMORA/aurora/memory.py` both exist and are
**0 bytes** (empty stubs):

```
FullName                                  Length
C:\PRIMORA PROJECT\PRIMORA\prime\memory.py      0
C:\PRIMORA PROJECT\PRIMORA\aurora\memory.py     0
```

Two facts that shape everything else:

1. **State formation is deterministic and has no LLM.** `state.py:1`:
   > `"""Deterministic user-state extraction and transitions. No LLM.`
   > `Only user-authored messages can create or change current state.`

   `entity_formation.py:26-27`:
   > `NOT IMPLEMENTED: semantic entity inference, automatic merging, session to`
   > `global promotion, and any model or LLM involvement of any kind.`

   `digest.py:1`: `"""Extractive digest builder. No LLM. No invented facts.`

2. **The only LLM in the write path is Jev**, and only for the `learn`
   capability (§3.3). The prompt it uses is quoted there.

---

## 1. Schema

All tables are created in one script, `_SCHEMA`, at `store.py:49`. The
database file is `PRIMORA/data/context/aurora_context.db`
(`ContextStore.__init__`, `store.py:230-234`).

### 1.1 Raw conversation (append-only)

`store.py:50-62`:

```sql
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp REAL NOT NULL,
    fingerprint TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.0,
    UNIQUE(session_id, seq, fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_messages_session_seq
    ON messages(session_id, seq);
```

`digests` (`store.py:63-74`) stores compacted history:
`source_ids, seq_start, seq_end, digest_type, content`.
`sessions` (`store.py:75-82`) holds `title`, `status`,
`previous_titles`.

### 1.2 Structured memory (M4 states)

`store.py:83-104`:

```sql
CREATE TABLE IF NOT EXISTS states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    concept TEXT NOT NULL,
    value TEXT NOT NULL,
    source_ids TEXT NOT NULL,
    created_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'current',
    supersedes_id INTEGER,
    authority TEXT NOT NULL DEFAULT 'user',
    memory_type TEXT NOT NULL DEFAULT '',
    confidence REAL,
    observed_at REAL,
    valid_from REAL,
    valid_until REAL,
    superseded_at REAL
);
CREATE INDEX IF NOT EXISTS idx_states_session_concept ON states(session_id, concept);
CREATE INDEX IF NOT EXISTS idx_states_scope_concept    ON states(scope, concept);
```

Lifecycle statuses observed in code: `current`, `superseded`,
`unresolved` (`manager.py:2465`, `state.py:773`), and `candidates` rows use
`pending` / `confirmed` / `rejected` (`state.py:667`, `store.py:116`).

Memory types (`state.py:17-22`):
`FACT, PREFERENCE, DECISION, INSTRUCTION, RATIONALE, UNCERTAIN`.

Schema migrations are applied lazily on open (`store.py:252-278`), adding
`previous_titles`, `memory_type`, `confidence`, `observed_at`, `valid_from`,
`valid_until`, `superseded_at` if missing.

### 1.3 Candidates (pending / held memory)

`store.py:105-123`:

```sql
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    source_ids TEXT NOT NULL,
    proposed_type TEXT NOT NULL DEFAULT '',
    proposed_concept TEXT NOT NULL DEFAULT '',
    proposed_value TEXT NOT NULL DEFAULT '',
    proposed_rationale TEXT NOT NULL DEFAULT '',
    authority TEXT NOT NULL DEFAULT 'user',
    confidence REAL NOT NULL DEFAULT 0.3,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL,
    expires_at REAL
);
```

### 1.4 Graph (M5)

- `entities` `store.py:124-145` — `entity_type`, `display_name`,
  `normalized_name`, `status` (`candidate` default), `merged_into_entity_id`,
  `authority`, `confidence`.
- `entity_aliases` `store.py:146-166` — plus `valid_from/valid_until/superseded_at`.
- `relationships` `store.py:167-195` — with a partial unique index keeping one
  current edge per key:

```sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_relationships_current_key
    ON relationships(scope, relationship_type, subject_id, object_id)
    WHERE status = 'current';
```

### 1.5 Retrieval vectors

`store.py:196-215`:

```sql
CREATE TABLE IF NOT EXISTS retrieval_vectors (
    vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    ref_id INTEGER NOT NULL,
    scope TEXT NOT NULL,
    session_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_retrieval_vectors_identity
    ON retrieval_vectors(kind, ref_id, model_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_vectors_kind_scope
    ON retrieval_vectors(kind, scope, model_id);
CREATE TABLE IF NOT EXISTS retrieval_index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
```

### 1.6 Full-text search

`store.py:218-224` — FTS5 over **messages only**:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
    USING fts5(content, content='messages', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
```

`ContextStore` probes FTS availability at runtime (`store.py:241-244`) and
the lexical channel degrades gracefully when it is absent.

### 1.7 Live database contents (as audited)

```
messages 314 | states 4 | candidates 3 | entities 1 | entity_aliases 0
relationships 1 | digests 20 | sessions 3 | retrieval_vectors 5
sessions by message count: 3455cf09-371f-4dea-bf2e-a46914a370d2 (255),
                           default (57), probe (2)
states by status: current 4
```

---

## 2. Write path

### 2.1 Entry point: `ContextManager.reconcile`

`manager.py:1321-1376` (verbatim, abridged only by eliding the timestamp
plumbing):

```python
def reconcile(self, messages, session_id=DEFAULT_SESSION_ID) -> list[str]:
    """Persist a conversation snapshot idempotently.

    Sequence identity = position in the snapshot. Re-sending the same
    snapshot is a no-op. Same position with different content is stored
    as a separate row plus a diagnostic; nothing is overwritten.
    Returns diagnostics (empty on the happy path).
    """
    ...
    for seq, msg in enumerate(messages):
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        ...
        importance = scoring_mod.importance_of(content, role)
        message_id, is_new, conflict = store.append(
            session_id, seq, role, content, importance, timestamp=timestamp,
        )
        if conflict:
            diagnostics.append(conflict)
        if is_new:
            # M4 structured-memory formation first. M5 entity,
            # alias, and relationship formation runs after it, from
            # the same persisted message id, and is independently
            # guarded so it can never suppress or alter M4.
            diagnostics.extend(
                self._extract_states(store, session_id, message_id, role, content)
            )
            if role == "user":
                diagnostics.extend(
                    self._extract_entities(store, session_id, message_id, content, timestamp)
                )
```

`reconcile` is called from `build` as step 1 — *"Persist the snapshot first;
raw truth before selection"* (`manager.py:1451-1452`).

### 2.2 M4 — state extraction (`state.extract_candidates` → `apply_user_candidates`)

Dispatch, `manager.py:1400-1422`:

```python
def _extract_states(store, session_id, message_id, role, content):
    """Deterministic state extraction. Never invents on failure."""
    try:
        if role == "user":
            candidates = state_mod.extract_candidates(content, role)
            if not candidates:
                return []
            return state_mod.apply_user_candidates(store, session_id, message_id, candidates)
        if role == "assistant":
            return state_mod.check_assistant_conflict(store, session_id, message_id, content)
    except Exception as e:
        return [f"state_failed: extraction failed: {e}"]
    return []
```

`extract_candidates` (`state.py:330-408`) is a two-stage matcher over the
message text:

- **Legacy patterns** `_extract_legacy` (`state.py:208-251`):
  `_INVALIDATE_RE` (`X is no longer correct`), `_CORRECT_NEW_RE`,
  `_CORRECT_WORD_RE`, `_CHANGE_RE`, `_FROM_NOW_ON_RE`, `_INSTEAD_RE`, and
  the generic establishment form `_ESTABLISH_RE = r"^(?:the|my)\s+(.+?)\s+is\s+(.+?)\.?$"` (`state.py:71-73`).
- **Extended phrasings** `_extract_extended` (`state.py:254-327`):
  `I prefer/like/love …`, `we decided …`, `always/never/don't …`,
  `from now on …`, etc., each mapped to a memory type.
- **Corrections unwrap first**: `_CORRECTION_PREFIX_RE` (`actually|well|no|correction`)
  and `_THATS_WRONG_RE` (`state.py:353-365`).
- **Hedges never auto-confirm**: `_HEDGE_RE` / `_HEDGE_PREFIX_RE` force
  `auto_confirm=False, confidence=0.3` (`state.py:403-407`), i.e. a pending
  candidate rather than truth.
- **Scope**: `_scope_of` looks for `GLOBAL_MARKERS`
  (`"for all conversations"`, `"globally"`, … `state.py:42-48`).

Admission — `apply_user_candidates` (`state.py:698-736`) →
`_apply_confirmed` (`state.py:555-647`), which:

* resolves the concept against tracked ones (`resolve_concept`, `state.py:465`),
* **supersedes** older currents (`store.create_state_successor`) — "Newest
  user value wins; old rows kept" (`state.py:704`),
* attaches a `rationale for <concept>` row plus an `explains` relationship
  when the user gave a "because …" reason (`state.py:628-644`),
* rejects stale pending candidates for that concept (`_reject_pending`).

Assistant turns can only **record a conflict, never resolve one**
(`state.py:739-782`):

```python
new_id = store.create_state(
    session_id, session_id, concept, value, [message_id],
    status="unresolved", supersedes_id=None, authority="assistant",
    confidence=0.3,
)
```

### 2.3 M5 — entity formation (`entity_formation.form_from_user_message`)

Called only for `role == "user"` (`manager.py:1366-1372`).
`entity_formation.py:10-27` states the constraints verbatim:

> `* User messages only. Assistant, system, and tool turns never establish an`
> `  M5 fact.`
> `* Explicit wording only. Every entity pattern requires the user to name the`
> `  type, either as ``my <type> is <NAME>`` or ``<NAME> is a <type>``. Nothing`
> `  is inferred from capitalisation, from a bare noun, or from context.`
> `* Exact identity only. … Similar names are never merged, …`
> `* Session scope by default. No automatic promotion to global.`

Type vocabulary is closed (`entity_formation.py:43-50`):

```python
_EXPLICIT_TYPE_WORDS: dict[str, str] = {
    "name": "PERSON", "person": "PERSON", "agent": "AGENT",
    "project": "PROJECT", "component": "COMPONENT", "file": "FILE",
}
```

Plus two special forms: aliasing (`_ALSO_CALLED`, `_I_CALL_HEAD`,
`entity_formation.py:68-73`) and filesystem paths
(`_PATH_TOKEN`, `entity_formation.py:83-86`), the latter routed through
`_path_entities` (`entity_formation.py:138-149`) → FILE entities.

Every write goes through `entities_mod.create_candidate` → `entities_mod.confirm`
(`_ensure_entity`, `entity_formation.py:194-209`).

### 2.4 M6-independent compaction — digests

`_add_digests` (`manager.py:2529-2600`) opportunistically compacts when
`store.count(session_id) >= COMPACT_AFTER_MESSAGES` (= 50,
`manager.py:142`) using `digest.maybe_compact` with
`build_extractive_digest` — verbatim source sentences with `[src:id]` tags,
no model involved (`digest.py:1-32`).

### 2.5 `learn` — the LLM-assisted write path (Jev)

Capability surface: `LearnSpecialist.handle` (`learn.py:70-134`).

Input guard — Aurora may **not** supply structured memory
(`learn.py:32-35`, `learn.py:74-80`):

```python
_RESERVED_KEYS = frozenset({
    "concept", "value", "scope", "memory_type", "authority", "confidence",
    "status", "candidate_id", "state_id", "kind", "auto_confirm",
})
...
offending = sorted(set(payload) & _RESERVED_KEYS)
if offending:
    return PrimeResult.failure(
        "Aurora must not supply the internal memory structure "
        f"(refused: {', '.join(offending)}). Describe the lesson in "
        "'text' and name its origin in 'provenance'.")
```

Admission — `ContextManager.learn` (`manager.py:245-378`), docstring:

> `1. ``store.append`` persists Aurora's sentence as an *assistant*`
> `   message. Assistant-authored statements are not auto-admitted by`
> `   formation, so the sentence cannot become memory merely by being`
> `   written down, and its id is the provenance anchor.`
> `2. ``store.create_candidate`` records Jev's structured reading as a`
> `   pending candidate carrying the provenance as its ``authority``.`
> `3. ``state.confirm_pending`` promotes it, and only if the lifecycle`
> `   agrees.`

Provenance policy is enforced at admission, not trusted from Jev
(`manager.py:320-328`):

```python
# The provenance policy is enforced HERE, not trusted from Jev.
# Jev is a pluggable, model-driven component and a buggy or
# third-party one could return "confirm" for an inference; the
# admission layer is the trust boundary, so it decides for itself.
from PRIMORA.learning.jev import HOLDS
disposition = ("hold" if request_provenance in HOLDS else outcome.disposition)
```

`jev.py:57-69`:

```python
# Only what Amir himself said may become a current fact. Everything else is
# held as a pending candidate.
AUTO_ADMIT = frozenset({USER})
# Every provenance that is never auto-admitted.
HOLDS = frozenset(p for p in PROVENANCE if p not in AUTO_ADMIT)
```

**The only LLM prompt in the write path** — `ModelJev.SYSTEM_PROMPT`
(`jev.py:204-215`, verbatim):

```python
SYSTEM_PROMPT = (
    "You convert one statement into structured memory proposals. "
    "Reply with JSON only, no prose and no code fence, shaped as:\n"
    '{"candidates": [{"concept": "...", "value": "...",\n'
    '  "memory_type": "FACT|PREFERENCE|DECISION|INSTRUCTION|RATIONALE",\n'
    '  "scope": "session|global"}]}\n'
    "concept is a short lowercase noun phrase naming what is known. "
    "value is the concrete content. scope is 'global' only when the "
    "statement is true for every conversation. "
    'Return {"candidates": []} when the statement is not something worth '
    "remembering, such as a greeting or a question."
)
```

Bounds: `MAX_TEXT_CHARS = 2000`, `MAX_CANDIDATES = 4`
(`jev.py:85-86`). Fail-closed parsing: `ModelJev._parse`
(`jev.py:257-310`) rejects malformed replies outright; `UnconfiguredJev`
raises (`jev.py:172-185`).

---

## 3. Retrieval

### 3.1 Hybrid retrieval — channels, fusion, gates

Documented pipeline (constants at `hybrid.py:32-41`, used by
`hybrid_retrieve`):

```python
SEMANTIC_TAU = 0.45
RRF_K = 60.0
EVIDENCE_FLOOR_RATIO = 0.725
MAX_CANDIDATES = 24
MAX_SEMANTIC_CANDIDATES = 12
MAX_STRUCTURAL_CANDIDATES = 8
MAX_SCOPE_FANOUT = 4
CHANNELS = ("lexical", "semantic", "entity", "relationship")
```

`hybrid_retrieve` (`hybrid.py:428-540`) runs, in order:

1. **lexical** — `lexical_channel(collected, cross_cap=3)` from
   `collect_lexical_candidates` (`manager.py:1740`), i.e. FTS + scoring.
2. **semantic** — `semantic_channel(index, store, query_text, scope, …)`
   (`hybrid.py:463-478`), fail-soft: any exception becomes an empty channel
   plus a diagnostic.
3. **entity / relationship** — structural channel, *only when the query
   names an entity* (`hybrid.py:480-498`).

then fusion and floor (`hybrid.py:508-515`):

```python
scores, order = rrf_fuse(channels, k=k)
order, floor_applied = apply_evidence_floor(scores, order, floor_ratio)
order = order[:limit]
```

`hybrid_retrieve`'s docstring: *"Read-only. Never raises: each channel
failure degrades to that channel being empty with a diagnostic, so a missing
model or a broken structural query can never break lexical retrieval."*
(`hybrid.py:439-442`).

Hard gates precede candidates: superseded states are excluded before the
channels run (`manager.py:1720-1731`, `_superseded_source_ids`), and only
`authority == "user"` currents are eligible for injection
(`manager.py:2400-2404`).

### 3.2 Semantic index

`semantic.py:527-597` (`SemanticIndex.search`):

```python
def search(self, query_text, scopes, kinds=EMBEDDABLE_KINDS,
           tau: float = 0.45, limit: int = 20, exclude=None):
    """Bounded, gated semantic candidate search.

    Deterministic: exact float32 dot product against L2-normalized
    vectors, ordered by similarity then by `ref_id` so equal scores keep
    a stable order. …
    """
    ...
    rows = self._store.fetch_vector_rows(model_id=self._model_id, scope=None)
    ...
    scores = matrix @ query_vector
    order = sorted(range(len(scoped)), key=lambda i: (-float(scores[i]), scoped[i][1]))
    ...
    if similarity < tau:
        break
```

Model: `sentence-transformers/all-MiniLM-L6-v2` on CPU
(`semantic.py:25-35`); only `state` and `entity` kinds are embedded
(`_STATE_INDEX_STATUSES = ("current","superseded")`, `semantic.py:25-35`).
Index refresh is opportunistic in the live path
(`manager.maybe_sync_semantic_index`, `manager.py:512`).

### 3.3 Scoring (lexical channel and state ranking)

`scoring.py:123-151`, weights are module defaults:

```python
def score(text, role, seq, now_seq, query_terms, selected_texts=None,
          stored_importance=0.0, w_relevance=0.45, w_recency=0.25,
          w_importance=0.30, w_redundancy=0.50) -> dict[str, float]:
    """Score one candidate. Returns the full breakdown for provenance."""
    ...
    total = (w_relevance * rel + w_recency * rec + w_importance * imp) \
        - w_redundancy * red
```

Header comment (`scoring.py:5-9`):

```
Weights:
  relevance  — word overlap with the retrieval query
  recency    — 1 / (1 + age in messages)
  importance — stored per-message value (user authority > assistant)
  redundancy — max Jaccard similarity to already-selected texts (penalty)
```

`importance_of` (`scoring.py:72-85`): `IMPORTANCE_SIGNALS` =
`("remember","important","always","never","we decided","agreed","preference","constraint")`;
user base `0.15 + 0.35*hits`, system `0.5`, assistant `0.4*base`.
Plural folding happens only at comparison time via `singular`
(`scoring.py:42-63`) — *"Folding happens here, at the point of comparison,
and not in tokenize"* (`scoring.py:91-94`).

### 3.4 `recall` capability (model-initiated retrieval)

Registered at `capability.py:61-80`:

```python
name="recall",
description="Search your own stored conversation memory for "
            "passages relevant to a question; returns verbatim messages.",
```

`RecallSpecialist.handle` (`recall.py:54-90`) → `Mind.recall`
(`mind.py:371-381`) → `ContextManager.recall` (`manager.py:401-490`), which
runs the *same* `hybrid_retrieve` the builder uses and then:

```python
if message.role in INTERNAL_ROLES:      # {"tool", "system"}, manager.py:110
    continue
...
if len(content) > RECALL_MAX_CONTENT_CHARS:      # 1200, manager.py:124
    content = content[:RECALL_MAX_CONTENT_CHARS] + RECALL_TRUNCATION_MARKER
```

Bounds: `DEFAULT_LIMIT = 5`, `MAX_LIMIT = 12` (`recall.py:20-21`);
cross-session cap `CROSS_SESSION_CAP = 3` (`manager.py:89`).

---

## 4. Prompt building and persona

### 4.1 Persona

Base instruction — `DEFAULT_SYSTEM_INSTRUCTION`, `configuration.py:80-113`
(full text):

```python
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are Aurora. Prime is your technical execution system: you decide "
    "WHAT is needed and request it; Prime handles HOW.\n"
    "When execution is required, emit exactly ONE request block:\n"
    "<prime_request>\n"
    '{"action":"...", "payload": {...}, "context": null, "metadata": null}\n'
    "</prime_request>\n"
    "That block is your reply; text outside it is discarded. Never emit "
    "more than one <prime_request>, and never a block beside an answer. If "
    "no technical execution is required, answer normally and emit no block "
    "- most questions need nothing at all.\n"
    "\n"
    "ACT, DO NOT ANNOUNCE. Saying what you will do is not doing it:\n"
    "  X  \"I'll research the latest version.\"   -> emit the request\n"
    "  X  \"I'll ask the reasoning specialist.\"  -> emit the request\n"
    "  X  \"I'll retrieve that from memory.\"     -> emit the request\n"
    "\n"
    "The action is a capability name, never a tool or specialist name. "
    "research_read: a numbered source's page. E.g.\n"
    '{"action":"research","payload":{"query":"latest news"},'
    '"context":null,"metadata":null}\n'
    "\n"
    "ONE STEP AT A TIME: request only the FIRST step, then decide the next "
    "from its result. USE EACH RESULT: answer with it, never repeat a "
    "call. A result carrying a failure or an error did not happen: say "
    "so.\n"
    "GROUNDING. If a result gives you nothing usable, say so. Never invent "
    "what someone said or agreed to, and never say you stored, learned, "
    "searched or remembered something unless that request succeeded. A "
    "[state ...] line is your own stored memory: use it as given; if none "
    "answers, say nothing is stored.\n"
    "\n"
    "Never do Prime's work yourself and never claim that you did."
)
```

Note the memory-specific clause: **"A `[state ...]` line is your own stored
memory: use it as given; if none answers, say nothing is stored."**

Identity block — `build_identity_instruction` (`identity.py:134-149`),
header `IDENTITY_HEADER`, first body sentence (`identity.py:143-145`):

> `You are Aurora, the conversational identity in PRIMORA. PRIMORA is`
> `the project this runtime belongs to; Prime is the orchestration and`
> `tool-execution layer you use.`

Capability / specialist / tool catalogues — `build_capability_instruction`
(`configuration.py:136-161`, emitted as `"Available Prime Capabilities:"`
with one `- name: description` line and `Required: …` fields, closed by
`"Send exactly that name as the action."`), plus
`build_specialist_instruction` (`configuration.py:164-195`) and
`build_tool_instruction` (`configuration.py:198-228`).

Family guidance, appended last — `QWEN_GUIDANCE` (`configuration.py:32-37`)
under header `GUIDANCE_HEADER = "REQUEST DISCIPLINE"` (`configuration.py:13`),
applied only when `detect_base_family(...)` reports a `qwen*` architecture
(`configuration.py:62-73`).

Assembly — `Aurora._with_instruction` (`aurora.py:273-309`), order is:
system instruction → identity → capability/specialist/tool catalogues →
guidance. Comment at `aurora.py:286-290`:

> `# Placed last, after the catalogues, because it is a behavioural`
> `# nudge rather than a description: the nearest instruction to`
> `# generation is the one that has the most effect on whether a`
> `# request block is emitted at all.`

`Aurora.respond` (`aurora.py:328-338`) → `CognitiveLoop.run(instructed, session_id)`.

### 4.2 Per-turn assembly order (`ContextManager.build`)

`build` (`manager.py:1427`) and its numbered steps:

| step | code | what it inserts |
|---|---|---|
| 0 | `ensure_session` (`manager.py:1448`) | session row |
| 1 | `reconcile` (`manager.py:1451-1452`) | persist + form M4/M5 |
| 2 | split system msg (`manager.py:1454-1459`) | |
| 3 | mandatory current turn (`manager.py:1461-1491`) | never truncated; raises `ContextTooLargeError` if it alone exceeds the ceiling |
| 3b | `working.to_messages()` (`manager.py:1505-1529`) | task-local Working Memory |
| 4 | newest history (`manager.py:1531-1557`) | fill up to `input_target - state_reserve` |
| 5 | `_add_session_info` (`manager.py:1564`) + `_add_current_state` (`manager.py:1572`) | structured authority first |
| 6 | `_retrieve` (`manager.py:1583`) | hybrid retrieval over older material |
| 6b | `_add_cross_session_states` (`manager.py:1597`) | explicit cross-session questions |
| 7 | `_add_digests` (`manager.py:1607`) | history digests |
| 8 | `_add_memories` (`manager.py:1617-1625`) | *"Optional memory hook (no-op in V1)"* |
| 9 | reassembly (`manager.py:1627-1647`) | see below |

Final ordering (`manager.py:1627-1647`):

```python
# 9. Reassembly: system instruction, then synthesized context
# (digests/memory, oldest first), then conversation in original
# chronological order with the current turn in its true position.
...
ordered = (
    ([system_msg] if system_msg is not None else [])
    + synth
    + [m for m in convo if not (system_msg is not None and m is system_msg)]
)
```

### 4.3 Exact rendered formats of memory lines

All are `role="system"` messages inserted into the working prompt.

Current state — `manager.py:2446-2449`:

```python
content = (
    f"[state current src:{','.join(str(i) for i in st.source_ids)}] "
    f"{st.concept} = {st.value}"
)
```

Admission rule (`manager.py:2435-2443`): overlapping facts may use the full
`input_ceiling`; non-overlapping ones are admitted only inside
`reserve_limit = min(cfg.input_ceiling, cfg.input_target)` and only from the
current session.

Unresolved conflict — `manager.py:2476-2479`:

```python
content = (
    f"[state unresolved src:{','.join(str(i) for i in st.source_ids)}] "
    f"{st.concept}: {st.value} (conflicting, not established)"
)
```

Superseded (only when the question carries a history marker,
`HISTORY_MARKERS`, `manager.py:48-63`) — `manager.py:2512-2515`:

```python
content = (
    f"[state previous src:{','.join(str(i) for i in st.source_ids)}] "
    f"{st.concept} was: {st.value} (superseded)"
)
```

Cross-session state — `manager.py:2354-2358`:

```python
content = (
    f"[states session {st.session_id} — {tag}"
    f" src:{','.join(str(i) for i in st.source_ids)}] "
    f"{st.concept} = {st.value}"
)
```

Session metadata — `manager.py:2256-2264`:

```python
f"[session current] id={rec.session_id} name={label!r} status={rec.status}"
f"[session history] {rec.session_id} was previously called {old!r}"
```

Digest — `manager.py:2585-2588`:

```python
{"role": "system",
 "content": f"[history digest {d['seq_start']}-{d['seq_end']}]"
            f"\n{d['content']}"}
```

Retrieved foreign history — `CROSS_SESSION_PREFIX = "[from another conversation]\n"`
(`manager.py:66`) prepended in `_admit_cross_session` (`manager.py:2132`).

Pluggable memory hook (inactive by default) — `manager.py:2633`:

```python
msg = {"role": role, "content": f"[memory]\n{content}"}
```

### 4.4 Budget

`BudgetConfig` defaults (`models.py:319-323`):

```python
input_target: int = 1000
input_ceiling: int = 1100
output_reserve: int = 128
total_budget: int = 1228
state_reserve: int = 100
```

`AuroraConfig` overrides these on the live path
(`configuration.py:255-264`):

```python
context_target: int = 1500
context_ceiling: int = 1600
context_output_reserve: int = 256
context_total: int = 1856
context_state_reserve: int = 100
```

`Aurora._budget` (`aurora.py:311-318`) builds the `BudgetConfig` handed to
the loop, so the **live prompt budget is 1500/1600/256/1856/100**.
`state_reserve` semantics: `models.py:311-316` — *"The fill stops at
`input_target - state_reserve` … so states and retrieval get at least that
much space on a turn where history would otherwise consume everything."*

### 4.5 Where a prompt could be dumped (nothing is logged today)

There is no prompt transcript anywhere in the repository (searched logs,
`DOCS/`, `NOTES/`, `TESTS/t1/reports`, holdout evaluations, git history,
`%TEMP%`). The emit sites, should logging ever be added, are:

* `Aurora._with_instruction` / `Aurora.respond` / `Aurora.context_for` — `aurora.py:273`, `328`, `340`
* `CognitiveLoop.run` → `self._mind.build(...)` then `self._brain.generate(context.messages, …)` — `loop.py:541`, `loop.py:566`
* `ContextManager.build` final `ordered` list — `manager.py:1643`

No such logging was added by this audit.

---

## 5. Model configuration

**Two different model stacks.**

### 5.1 Aurora's brain — local HF transformers + bitsandbytes 4-bit + QLoRA

`brain.py:158-192` (verbatim):

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
...
base_model = AutoModelForCausalLM.from_pretrained(
    self._config.model_path,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    dtype=torch.float16,
    attn_implementation="sdpa",
)
...
self._model = PeftModel.from_pretrained(
    base_model, self._config.adapter_path, device_map={"": 0},
)
```

Defaults (`configuration.py:274-275`): model
`RESOURCES/aurora/v0.1/model`, adapter
`RESOURCES/aurora/v0.1/aurora_qlora`.
Generation defaults (`configuration.py:235-238`): `max_new_tokens=256`,
`temperature=0.75`, `top_p=0.9`, `repetition_penalty=1.08`.
Context safety ceiling (`brain.py:67-72`): `PRIMORA_AURORA_MAX_CONTEXT_TOKENS`
else `config.context_total` (1856).
`Aurora.warmup` (`aurora.py:248-268`) loads the brain, then the local
embedding model (a missing embedding model is not an error — retrieval
degrades to lexical).

### 5.2 Prime's provider — Ollama (separate from the brain)

`main.py:92-112`:

```python
provider = OllamaProvider(
    model=ollama_model,          # PRIMORA_OLLAMA_MODEL, else
                                 # DEFAULT_OLLAMA_MODEL = "qwen3-8b-16k:latest"
    base_url=ollama_url,         # PRIMORA_OLLAMA_URL default 127.0.0.1:11434
    timeout=ollama_timeout,
)
...
prime = Prime(provider, research_provider=research_provider)
aurora = Aurora(prime)
```

`main.py:23-28` states the provider is currently inert on the live path:

> `# The provider is Prime's fallback for non-specialist capabilities and is`
> `# currently unreachable in the live path (the router rejects unregistered`
> `# actions before execution), so this default is inert unless a future`
> `# capability actually invokes the provider.`

`consult_specialist` is the capability wired to that provider
(`prime.py:52-55`).

**`Prime(...)` is constructed without `jev=` in the shipped entry point**
(`main.py:110-112`), while Jev is only wired when passed
(`prime.py:58-62`):

```python
learn = self._registry.get_specialist("learn")
if learn is not None:
    learn.bind(self._mind)
    if jev is not None:
        learn.set_jev(jev)
```

---

## 6. Logs / samples

Root-level logs are UTF-16 LE and contain **no prompts, no memory lines, and
no retrieval traces** — only QML styling warnings, Python/PowerShell
redirect noise, and startup banners.

`verse_crash.log` (10 788 bytes), first lines:

```
file:///C:/PRIMORA PROJECT/PRIMORA/verse/qml/Conversation.qml:119:29: QML QQuickRectangle: The current
style does not support customization of this control (property: "background" item:
QQuickRectangle(0x16681e77e20, parent=0x0, geometry=0,0 0x0)). …
+ python -X faulthandler -m PRIMORA.main *> verse_crash.log
Starting PRIMORA...
Type 'exit' or 'quit' to quit.
```

`verse_startup.log` (5 164 bytes):

```
…Conversation.qml:153:29: QML QQuickRectangle: …
W0923 15:37:51.554000 7204 site-packages\torch\utils\flop_counter.py:29] triton not found; flop counting will
not work for triton kernels
Starting PRIMORA...
```

The closest thing to a prompt sample in the repository is the *rendered
format strings* quoted in §4.3 — i.e. the code, not a captured run.

---

## 7. Tests

No `pytest.ini`, `conftest.py`, or `pyproject.toml` exists; tests hard-code
`sys.path.insert(0, "C:\\PRIMORA PROJECT")` and must be run from the project
root.

Documented command (`TESTS/prime/RETRIEVAL_BENCHMARK.md:35`):

```
python -m pytest TESTS -q --ignore=TESTS\specialists\test_calculate.py --ignore=TESTS\aurora\test_brain_integration.py
```

with the stated acceptance rule *"exit code 0 only when the production DB
hash is unchanged"*.

### 7.1 Memory / retrieval test inventory (executed)

| file | result |
|---|---|
| `test_memory_formation.py` | 17 passed |
| `test_memory_formation_m5.py` | 52 passed |
| `test_memory_lifecycle.py` | 18 passed |
| `test_memory_correctness.py` | 6 passed |
| `test_memory_entities.py` | 22 passed |
| `test_memory_relationships.py` | 26 passed |
| `test_memory_temporal.py` | 18 passed |
| `test_memory_queries.py` | 29 passed |
| `test_memory_working.py` | 19 passed |
| `test_memory_write_path_uniqueness.py` | **1 failed**, 20 passed |
| `test_retrieval_hybrid.py` | 64 passed |
| `test_retrieval_fts_terms.py` | 13 passed |
| `test_retrieval_comparison.py` | 17 passed |
| `test_retrieval_calibration.py` | 17 passed |
| `test_retrieval_benchmark.py` | 21 passed |
| `test_retrieval_live.py` | 19 passed |
| `test_recall_capability.py` | 39 passed |
| `test_recall_nesting.py` | **HANGS** (see §9-C1) |
| `test_learn_capability.py` | **1 failed**, 96 passed |
| `test_context.py` | 24 passed |
| `test_context_budget_fit.py` | 24 passed |
| `test_context_integration.py` | 2 passed |
| `test_mind.py` | 40 passed |
| `test_session_state.py`, `test_session_scope.py`, `test_state_injection_fixes.py` | passed (in the combined run before it hung) |
| `test_safety_foundations.py` | **1 failed**, 37 passed |
| `test_tool_calling.py` | **1 failed**, 70 passed |
| `test_consult_specialist.py` | **1 failed**, 65 passed |

Production DB hash before and after every run — unchanged:

```
64FF37BDCC36E368CE7133F0F7A8D29D8428DAD503913E7166B73DE56BD8F4B1
```

(`PRIMORA/data/context/aurora_context.db`, 491 520 bytes.)

Environment: pytest 9.1.1, Python 3.12.7, `pytest-timeout` **not**
installed (so a hanging test cannot be bounded by the runner).

---

## 8. Suspected problems

### CONFIRMED

**C1 — Quadratic regex scan in M5 entity formation stalls the write path on
long messages.**

`entity_formation._path_entities` runs an unbounded `finditer` over the raw
message (`entity_formation.py:141`), using `_PATH_TOKEN`
(`entity_formation.py:83-86`):

```python
_PATH_TOKEN = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s\"'<>|]+"
    r"|(?:[\w.\-]+[\\/])+[\w.\-]+\.[A-Za-z0-9]{1,8})"
)
```

On a long separator-free run the second alternative re-scans from every start
position, so cost grows with the square of the length. Measured on this
machine by calling `_path_entities` directly:

| input length | time |
|---|---|
| 25 000 chars | 3.30 s |
| 50 000 chars | 13.07 s |
| 100 000 chars | 52.81 s |
| 200 000 chars | 211.05 s |

(4× time per 2× length — textbook quadratic.) A faulthandler dump of a
blocked `reconcile` shows the same frame:

```
File "C:\PRIMORA PROJECT\PRIMORA\prime\context\entity_formation.py", line 141 in _path_entities
File "C:\PRIMORA PROJECT\PRIMORA\prime\context\entity_formation.py", line 256 in _extract_entity_assertions
File "C:\PRIMORA PROJECT\PRIMORA\prime\context\entity_formation.py", line 407 in form_from_user_message
File "C:\PRIMORA PROJECT\PRIMORA\prime\context\manager.py", line 1393 in _extract_entities
File "C:\PRIMORA PROJECT\PRIMORA\prime\context\manager.py", line 1368 in reconcile
```

Consequences:
* Any user message of a few hundred thousand characters (a pasted log, a
  source file, a long document) blocks `reconcile`, which runs as step 1 of
  every `build` (`manager.py:1451-1452`) — i.e. the turn never completes.
  There is no length cap on stored messages and no timeout around formation
  (`reconcile` only catches `Exception`, it does not bound time).
* It already breaks the suite: `test_recall_nesting.py` hangs at
  `TestNoExponentialGrowth::test_recall_output_is_bounded_by_construction`
  (`TESTS/prime/test_recall_nesting.py:261-268`, which inserts two
  400 000-char messages), so the memory test suite can never finish.

**C2 — The "production database untouched" guard fails on any used install
(5 tests).**

`TESTS/prime/test_memory_write_path_uniqueness.py:278-285`:

```python
def test_production_database_untouched(self):
    db = os.path.join("PRIMORA", "data", "context", "aurora_context.db")
    if not os.path.exists(db):
        pytest.skip("no production database")
    ...
    for table in ("messages", "states", "candidates", "entities"):
        assert con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
```

The production DB currently holds `messages=314, states=4, candidates=3,
entities=1`, so this asserts an impossible condition once the app has been
used:

```
FAILED TESTS/prime/test_memory_write_path_uniqueness.py::…::test_production_database_untouched
  AssertionError: messages
  assert 314 == 0
```

Same test, same failure, in `test_learn_capability.py:777`,
`test_safety_foundations.py:485`, `test_tool_calling.py:806`,
`test_consult_specialist.py:657`. The documented "exit code 0 only when the
production DB hash is unchanged" rule (`RETRIEVAL_BENCHMARK.md:35`) is not
what the code checks; the code checks emptiness.

**C3 — The `learn` capability is advertised but can never succeed in the
shipped entry point.**

`main.py:110-112` builds `Prime(provider, research_provider=…)` with no
`jev=`; `prime.py:61-62` only calls `learn.set_jev(jev)` when `jev is not
None`; `LearnSpecialist.__init__` therefore keeps `UnconfiguredJev()`
(`learn.py:50`), whose `evaluate` raises (`jev.py:181-185`):

```python
raise RuntimeError(
    "No Jev is configured. Pass jev=... to Prime, for example "
    "ModelJev(provider)."
)
```

`ContextManager.learn` catches it (`manager.py:305-310`) and returns
`{"status": "rejected", "reason": "jev_failed: …"}`. Meanwhile
`_specialist_is_configured` (`configuration.py:116-133`) treats a specialist
with no `is_configured` attribute as available, so `learn` **is** listed in
"Available Prime Capabilities" (`configuration.py:145-155`). `ModelJev` is
referenced only from tests (`TESTS/prime/test_learn_capability.py:32` etc.);
nothing in `PRIMORA/` or `main.py` constructs it.

**C4 — Two empty memory modules ship in the tree.**

`PRIMORA/prime/memory.py` = 0 bytes, `PRIMORA/aurora/memory.py` = 0 bytes.

**C5 — The pluggable `[memory]` injection hook is dead by default.**

`ContextManager.__init__(…, memory_provider=None, …)` (`manager.py:179`) and
`_add_memories` (`manager.py:2615-2617`):

```python
provider = self._memory_provider
if provider is None:
    return total
```

`build` labels the step *"Optional memory hook (no-op in V1)"*
(`manager.py:1617`). `noop_memory` exists (`models.py:452`) but is not wired
by `Prime`.

**C6 — The same environment variable name has two different defaults.**

`BudgetConfig.from_env` (`models.py:336`):
`input_target=_int("PRIMORA_AURORA_CONTEXT_TARGET", 1000)`.
`AuroraConfig.from_env` (`configuration.py:291`):
`context_target=_int("PRIMORA_AURORA_CONTEXT_TARGET", 1500)`.
Same for `…_CEILING` (1100 vs 1600) and `…_OUTPUT_RESERVE` (128 vs 256) and
`…_TOTAL` (1228 vs 1856). The Aurora path always passes its own budget
(`aurora.py:311-318`), so the live numbers are the 1500/1600 set — but any
caller that builds a context through `Prime().context.build(...)` without a
`budget=` argument silently gets the smaller, different budget.

**C7 — Semantic retrieval scans every stored vector in Python per query.**

`semantic.search` (`semantic.py:553-568`) fetches **all** rows for the model
(`fetch_vector_rows(model_id=…, scope=None)`) and then filters by scope/kind
in Python before a dense `matrix @ query_vector` product. There is no ANN
structure and no per-scope prefilter. This is correct and cheap at today's
scale (5 vectors in the live DB, `MAX_SEMANTIC_CANDIDATES = 12`), but it is
linear in total stored memory per query.

**C8 — Provenance attestation failure silently accepts the label.**

`LearnSpecialist.handle` (`learn.py:112-117`):

```python
try:
    effective, diagnostic = self._attest(text, provenance)
except Exception:
    # Attestation must never block a legitimate lesson. If it cannot
    # be checked, the claim stands as labelled.
    effective, diagnostic = provenance, ""
```

By design a failed check falls back to trusting Aurora's self-report; the
`HOLDS` gate in `ContextManager.learn` (`manager.py:324-328`) still keeps
non-`user` provenances out of `current`, so the blast radius is limited to a
`user`-labelled claim that could not be verified.

### SUSPECTED (not confirmed — inference from the code/data above)

**S1 — State formation recall is very low on real conversation.**
314 stored messages produced 4 `current` states, 1 entity, 1 relationship,
3 candidates. Regex formation is deliberately narrow (`state.py` patterns are
anchored whole-message forms), so this may be exactly the intended
conservative behaviour — but the practical effect is that the `[state …]`
section of the prompt is usually empty, and §4.1 tells Aurora "if none
answers, say nothing is stored". I did not sample the 314 messages to
measure how many actually matched the patterns, so the *cause* of the low
yield is a guess.

**S2 — `learn` writes an assistant row at `seq = count + 1`, which can
disagree with later snapshot positions.**
`manager.py:290-293` appends Aurora's sentence at `store.count(session_id) + 1`,
whereas `reconcile` assigns `seq` by *position in the snapshot*
(`manager.py:1340`). If the two sequences disagree the store reports a
`conflict` diagnostic rather than overwriting (`store.py:50` `UNIQUE(...)`,
`manager.py:1356-1357`) — so the failure mode is a duplicate/conflicting row,
not data loss. I did not construct a scenario that produces it.

**S3 — Assistant-authored claims can populate the prompt with unresolved
conflicts.**
`check_assistant_conflict` writes `status='unresolved', authority='assistant'`
rows (`state.py:771-775`) for any assistant sentence matching
`_ESTABLISH_RE`; `_add_current_state` then surfaces relevant ones as
`[state unresolved …] (conflicting, not established)` (`manager.py:2476`).
The rows are never truth and never displace `current`, but a talkative
assistant can add prompt noise that Aurora herself generated. Whether this
actually happens in practice was not measured.

**S4 — Digests and retrieval both compete for the same ceiling with no
cross-section priority.**
Steps 6, 6b, 7, 8 all check `total >= cfg.input_ceiling` independently
(`manager.py:1584-1625`), so whichever runs first consumes the budget;
retrieval runs before digests. On a full turn, stored history digests are the
first thing to be dropped. This follows from the code order but I did not
measure how often it bites.

**S5 — No prompt/memory telemetry exists, so none of the above is observable
in production.**
Confirmed absence (§4.5, §6): the only logs are QML warnings and startup
banners, and diagnostics from `build` are returned on the `BuildResult` /
`CognitiveContext` and then discarded by `CognitiveLoop.run`. The suspicion
is that this makes memory regressions invisible until someone files a bug.

---

## 9. What a follow-up would look at first

1. Bound `_path_entities` (length cap before regex, or a non-backtracking
   scan) — C1 is a live hang with a clean reproduction.
2. Make the production-DB guard compare against a recorded hash instead of
   `COUNT(*) == 0` — C2 keeps the suite permanently red on a used install.
3. Either wire a `Jev` in `main.py` or stop advertising `learn` — C3.
4. Decide whether the `[memory]` hook and the two empty `memory.py` stubs
   are intended seams or leftovers — C4/C5.
