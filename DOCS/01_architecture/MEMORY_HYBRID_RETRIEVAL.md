# PRIMORA M6 — Advanced Retrieval

- **Milestone**: M6 — Advanced Retrieval (memory milestone stream)
- **Status**: **PROPOSED** architecture specification — **NOT IMPLEMENTED**
- **Project Root**: `C:\PRIMORA PROJECT`
- **Owner**: Prime (`PRIMORA/prime/context/`)
- **Extends**: `DOCS/01_architecture/CONTEXT_MANAGER.md`, `DOCS/01_architecture/MEMORY_ENTITY_RELATIONSHIP.md`, and the implemented M1–M5 memory architecture
- **Date**: 2026-09-25

---

## Label convention

Every factual claim in this document carries one of three labels:

| Label | Meaning |
|---|---|
| **CURRENT** | Verified by reading the code in this repository, or by an isolated read-only probe. File and line references given. |
| **PROPOSED** | A design decision in this document. Not implemented. Requires approval before any code is written. |
| **NOT IMPLEMENTED** | Explicitly deferred to a later milestone or excluded entirely. |

Where the current code disagrees with an older document, **the code wins** and the
disagreement is recorded in §3.4.

---

## 1. Purpose

M1–M5 answer *what PRIMORA believes, remembers, and how things are connected*.
They deliberately stopped at identity and structure. M6 answers the next
question:

> **Which memories and evidence are relevant to what Aurora is being asked right now?**

CURRENT: Structured Memory and M5 identity/relationships exist and are queried,
but they are **not reachable through retrieval**. The only retrieval channel is
lexical, over raw `messages`.

PROPOSED: M6 adds a **hybrid retrieval layer** — lexical + semantic + temporal
+ entity + relationship channels — feeding the existing bounded context
assembly. It does not replace FTS, does not add a second model, and does not
change how memory is written.

M6 explicitly does **not** build a RAG system. See §15.

---

## 2. Current retrieval architecture

All statements in this section are **CURRENT** and were verified against the
code. Line references are to the files as of 2026-09-25.

### 2.1 Entry point and call chain

```
Aurora.respond(messages, session_id)            PRIMORA/aurora/aurora.py
  ├─ context.build(...)      # first_view        aurora.py:115
  ├─ Brain.generate(...)
  ├─ [Prime tool path] Prime.handle() → tool result
  └─ context.build(...)      # second_view       aurora.py:147
```

`build()` is the only entry point into retrieval. It is called **twice per
turn**, so any per-request retrieval work is paid twice.

### 2.2 Assembly order inside `build()`

CURRENT (`manager.py:build`, step comments in source order):

| # | Step | Function |
|---|---|---|
| 0 | Ensure session row | `ensure_session` |
| 1 | Persist snapshot (raw truth first) | `reconcile` |
| 2 | Split system instruction | — |
| 3 | Current user turn is mandatory, never truncated | — |
| 3b | Working Memory layer (optional, runtime-only) | `working.to_messages()` |
| 4 | Newest conversation turns up to `input_target` | — |
| 5 | Session metadata, then current scoped state | `_add_session_info`, `_add_current_state` |
| 6 | **Ranked historical retrieval** | `_retrieve` |
| 6b | Explicit cross-session **state** blocks | `_add_cross_session_states` |
| 7 | Digests (extractive) | `_add_digests` |
| 8 | `MemoryProvider` hook (no-op) | `_add_memories` |
| 9 | Reassemble chronologically | — |

Retrieval runs **after** the recency window. It receives only the *residual*
budget. CURRENT: there is no reserved retrieval quota; if steps 0–5 consume
`input_ceiling`, `_retrieve` returns nothing.

### 2.3 Query construction

CURRENT (`manager.py:_retrieve`):

```python
query_terms = scoring_mod.build_query(
    current.get("content", "") if current else "",
    self._recent_user_text([m for m in history if m is not current]),
)
```

- The query is the **current user turn** plus the **most recent earlier user
  turn**. Nothing else. No entity names, no concept names, no session
  metadata, no history of the conversation beyond one turn.
- `build_query` concatenates `tokenize()` output. There is no phrase, no
  field weighting, and no distinction between "the term I just typed" and
  "a term I typed last turn".

### 2.4 Tokenization and normalization

CURRENT (`scoring.py:15,28-39`):

```python
_WORD_RE = re.compile(r"[a-z0-9']+")
def tokenize(text): return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]
```

Verified by probe:

| Input | Tokens |
|---|---|
| `What did we decide about the car design?` | `['decide', 'about', 'car', 'design']` |
| `what colour do I like` | `['colour', 'like']` |
| `don't stop believing` | `["don't", 'stop', 'believing']` |
| `running runs runner ran` | `['running', 'runs', 'runner', 'ran']` |
| `The QUICK brown-fox... jumps!` | `['quick', 'brown', 'fox', 'jumps']` |
| `Project Apollo's mobile app` | `['project', "apollo's", 'mobile', 'app']` |
| `... !!! ???` | `[]` |

Consequences, all **CURRENT**:

- **No stemming, no lemmatization.** `running`/`runs`/`runner`/`ran` are four
  unrelated tokens.
- **No synonym or paraphrase handling.** `colour` ≠ `color`, `car` ≠ `automobile`.
- **Stopword list is 60 words** and does not include `about`. Verified:
  `about` survives tokenization and is used as a *matching term*, so it pulls
  in unrelated messages.
- **Possessives are kept inside the token.** `"apollo's"` and `"don't"` reach
  SQLite FTS5 as a `MATCH` expression and **raise a syntax error** (§2.6).
- Empty token list → `_retrieve` returns early (`manager.py:1303`), so a
  punctuation-only query degrades to recency only. This is correct and tested
  (`test_context.py:295`).

### 2.5 Stopword behavior

CURRENT: purely subtractive. Stopwords are removed from **both** the query and
the candidate at scoring time, but the FTS `MATCH` is issued on the *already
tokenized* query terms, so stopwords never reach FTS. There is no stopword
handling inside the FTS index itself.

### 2.6 FTS usage

CURRENT (`store.py:_init_db`, `_probe_fts`, `search`):

```sql
CREATE VIRTUAL TABLE messages_fts USING fts5(content, content='messages', content_rowid='id')
```

- FTS5 is an **external-content** table kept in sync by triggers.
- Availability is **probed at startup** with a real query
  (`store.py:_probe_fts`), never assumed. Fallback is `content LIKE '%term%'`.
- Verified on this machine: `fts_available == True`.
- The tokenizer is the FTS5 default `unicode61`. **No stemmer, no stoplist,
  no `remove_diacritics`** is configured.
- **Ranking is done by SQLite FTS5: not at all.** `search()` ends with
  `ORDER BY m.session_id ASC, m.seq ASC`. `bm25()` is never called anywhere in
  the repository. FTS5 is used purely as a *filter*; all ranking is redone in
  Python.
- Queries are issued **one term at a time**, up to `query_terms[:12]`, each
  with `limit=10`. Maximum raw candidates per request: 120.

Probe results for `MATCH` on realistic tokens:

| Term | Result |
|---|---|
| `car` | 0 rows, no error |
| `don't` | `OperationalError: fts5: syntax error near "'"` |
| `apollo's` | `OperationalError: fts5: syntax error near "'"` |
| `BLUE-FALCON-7` | `OperationalError: no such column: FALCON` |
| `design?` | `OperationalError: fts5: syntax error near "?"` |

**This is a real, currently-live defect class.** Because `_WORD_RE` allows
apostrophes inside tokens, ordinary possessives and contractions are handed to
FTS5 as query syntax, fail, and are silently dropped by the per-term
`try/except` with a `retrieval_skipped_term` diagnostic. The retrieval pass
degrades quietly rather than loudly.

### 2.7 Fallback behavior

CURRENT (`manager.py:_retrieve`):

- One bad term → `retrieval_skipped_term` diagnostic, remaining terms still
  retrieved. Tested: `test_context.py:268`.
- Whole retrieval error → `retrieval_failed`, recent + current context kept.
- FTS5 unavailable → `LIKE` fallback inside `store.search`. **There is no test
  anywhere in the repository for this branch.** Verified by search: zero
  references to `fts_available` or `LIKE` in `TESTS/`.
- DB unavailable → in-memory pass-through + `store_unavailable`.

### 2.8 Scope behavior

CURRENT:

- `_retrieve` searches `session_id` only for the primary pass, then
  `_retrieve_cross_session` searches **all** sessions
  (`store.search(None, term, limit=20)`).
- **The cross-session lexical pass is NOT gated on an explicit request.** It
  runs on every retrieval. Verified by probe: a query in `sessB` retrieved
  `BLUE-FALCON-7` from `sessA`, tagged `[from another conversation]`.
  Only *state* passing (`_add_cross_session_states`) is gated on
  `CROSS_SESSION_MARKERS`.
- Cross-session results are capped at **3** (`cap: int = 3`,
  `manager.py:1403`) and sorted by `(-score, session_id, seq)`.
  **Because `session_id` is only a tiebreak, a high-scoring foreign item can
  outrank a same-session item.** No test asserts this cap.
- `global` scope applies to `states`, not to `messages`. Raw message
  retrieval has no `global` concept; it is session-id keyed.

### 2.9 Superseded filtering

CURRENT:

- `_superseded_source_ids(store, [session_id, "global"])` collects the
  **source message IDs of every superseded state**; those messages are
  dropped from the candidate list unless the query contains a
  `HISTORY_MARKERS` word (14 markers, `manager.py:41-56`).
- Emits `retrieval_suppressed_superseded: N message(s)`.
- Verified: three color corrections → only `green` is retrievable,
  `retrieval_suppressed_superseded: 2 message(s)`.
- **Granularity caveat:** suppression is per *message*, keyed by source ID. A
  message that is the source of any superseded state is dropped **entirely**,
  including any other still-relevant content it contained. This is coarse.

### 2.10 Scoring

CURRENT (`scoring.py:score`), pure and deterministic:

```
score = 0.45*relevance + 0.25*recency + 0.30*importance - 0.50*redundancy
```

- `relevance` = `|unique candidate terms ∩ query terms| / |query terms|`.
  Note the denominator is the **whole** query set, including terms from the
  previous user turn and including near-stopwords like `about`.
- `recency` = `1 / (1 + age_in_messages)`.
- `importance` = `max(stored_importance, importance_of(text, role))`, where
  `importance_of` = `min(1, 0.35*signal_hits)` scaled by role:
  user `0.15 + base`, system `0.5`, assistant `0.4 * base`.
- `redundancy` = max Jaccard similarity against already-selected texts.
- Ties break by `seq`. No randomness, no model call.

**Verified false-positive mechanism.** With a 9-term query:

| Candidate | score | relevance | recency | importance | redundancy |
|---|---|---|---|---|---|
| `The car design is a matte charcoal three-door.` | **0.1514** | 0.222 | 0.333 | 0.150 | 0.154 |
| `unrelated small talk about lunch` | **0.1815** | 0.111 | 0.500 | 0.150 | 0.077 |
| `we decided the emblem is a stylised fox` | 0.4083 | 0.111 | 1.000 | 0.500 | 0.083 |

The lunch message outranks the car design message, purely because it matched
the near-stopword `about` and was newer. `recency + importance` carry 0.55 of
weight, and **there is no relevance floor or threshold** anywhere in the
pipeline.

### 2.11 Redundancy handling

CURRENT: Jaccard token overlap against `selected_texts`, recomputed per
candidate at scoring time, penalized `-0.50 * red`. Separately, a
`role\x00content` fingerprint set prevents exact duplicates from being emitted
twice.

### 2.12 Limits and budgets

CURRENT (`BudgetConfig`, env-overridable):

| Field | Default | Env var |
|---|---|---|
| `input_target` | 1000 | `PRIMORA_AURORA_CONTEXT_TARGET` |
| `input_ceiling` | 1100 | `PRIMORA_AURORA_CONTEXT_CEILING` |
| `output_reserve` | 128 | `PRIMORA_AURORA_CONTEXT_OUTPUT_RESERVE` |
| `total_budget` | 1228 | `PRIMORA_AURORA_CONTEXT_TOTAL` |

Hard invariants: the current user turn is never truncated; a current turn alone
over `input_ceiling` raises `ContextTooLargeError` after the message is
persisted; `allowed_output = min(max_new_tokens, total_budget - input_tokens)`.
Digests start at `COMPACT_AFTER_MESSAGES = 50`.

Retrieval-specific limits: `query_terms[:12]`, `limit=10` per term
(120 raw candidates), `limit=20` per term cross-session, `cap=3` cross-session
admitted, `fetch_latest(session_id, 200)` for the recency clock.

### 2.13 No-match behavior

CURRENT: if the query yields no terms, or no term matches anything, the
candidate list is empty and retrieval contributes nothing. **There is no
"no results" signal to the model** — the absence of context is indistinguishable
from "nothing relevant exists", which is correct, and there is no explicit
no-match test for a populated store.

### 2.14 How results reach ContextView / Aurora

CURRENT:

- A retrieved raw message is emitted as `{"role": hit.role, "content": hit.content}`
  — it **keeps its original role** (`user`/`assistant`/`tool`).
- Cross-session hits keep the original role and get the content prefix
  `[from another conversation]\n`.
- Synthesized context (states, digests, session metadata) is emitted with
  `role="system"`.
- Admission is a greedy pass in descending score order, breaking when
  `total >= input_ceiling` or when the next item would not fit.
- Reassembly (§9 of `build`) re-sorts by original conversation position, so
  retrieved old messages land in **chronological** position, not at the end.
- `MemoryProvider` is `None` by default; `_add_memories` returns immediately.
  Verified: no production wiring.

### 2.15 What retrieval cannot currently reach

CURRENT — the following exist in the store and are **never** consulted by
retrieval:

| Structure | Reachable by retrieval? |
|---|---|
| `messages` | Yes (lexical only) |
| `states` | Only via `_add_current_state`, which is a *separate pass*, not retrieval |
| `digests` | Only via `_add_digests`, a separate pass |
| `entities` | No |
| `entity_aliases` | No |
| `relationships` | No |

M5's entity/relationship layer and the M1–M4 state layer are therefore
**invisible to the retriever**. A query that matches a stored DECISION's text
but not any raw message finds nothing, because `concerns`/`explains` are never
traversed.

---

## 3. Verified weaknesses

### 3.1 Probe results (isolated, read-only, temp databases)

Seed corpus (one session, three messages):

```
The car design is a matte charcoal three-door.
we decided the emblem is a stylised fox
the launch date for the model is in march
```

| Query class | Query | Older material retrieved |
|---|---|---|
| Paraphrase | `what colour did we pick for the car` | 1 line — matched only on the token `car` |
| Exact | `what did we decide about the car design?` | 1 line — correct |
| Paraphrase | `what did we choose for the badge?` | **NONE** |
| Indirect | `that thing we agreed on for the logo` | **NONE** |
| Generic | `what is completely unrelated to this project` | NONE (correct) |

Cross-session probe: `what is the vault code?` asked in `sessB` retrieved
`BLUE-FALCON-7` from `sessA`, correctly tagged.

### 3.2 Weaknesses, each tied to evidence

| # | Weakness | Evidence | Severity |
|---|---|---|---|
| W1 | **Paraphrase recall fails.** Synonym substitution defeats retrieval entirely. | Probe: `choose`/`badge` → NONE | Critical |
| W2 | **Indirect reference recall fails.** Pronouns and descriptive references ("that thing", "the logo") retrieve nothing. | Probe: indirect → NONE | Critical |
| W3 | **No relevance floor.** A candidate matching one near-stopword outranks a genuinely relevant older candidate. | §2.10 measured scores | High |
| W4 | **Query dilution.** `relevance` divides by the entire multi-turn query set, so adding terms lowers every candidate's score. | `scoring.py:58-64` | High |
| W5 | **FTS syntax errors on ordinary tokens.** Apostrophes, hyphens and `?` in tokens raise `OperationalError` and are silently dropped. | §2.6 probe table | High |
| W6 | **FTS5 ranking is unused.** `bm25()` never called; FTS is a filter only. | `store.py:search` | Medium |
| W7 | **No stemming.** Four inflections of one word are four terms. | §2.4 probe | Medium |
| W8 | **Structured memory, entities and relationships are unreachable by retrieval.** | §2.15 | High |
| W9 | **Superseded suppression is message-granular** and can drop still-relevant content. | §2.9 | Medium |
| W10 | **Cross-session items can outrank same-session items.** `session_id` is only a sort tiebreak. | `manager.py:1463` sort key | Medium |
| W11 | **Cross-session retrieval is ungated**, so scope leakage is bounded only by tagging. | §2.8 probe | Medium |
| W12 | **Retrieval gets only residual budget**; no reserved quota. | §2.2 | Medium |
| W13 | **Near-stopwords act as matching terms** (`about`, `project`). | §2.4 probe | Medium |
| W14 | **No retrieval benchmark exists.** No recall, precision, latency or false-positive measurement anywhere in the repository. | Repo-wide search: zero hits | Critical (process) |
| W15 | **The `LIKE` fallback branch is untested.** | §2.7 | Medium |

### 3.3 Historical benchmark claims — NOT VERIFIABLE

The following claims were presented as known historical findings:

- lexical retrieval had strong exact-word performance
- paraphrase recall was weak
- indirect-reference recall was weaker than exact matches
- cross-session behavior was previously strong
- false-positive rate was high

**Audit result: no benchmark, golden set, metric, or retrieval-quality
measurement exists anywhere in this repository.** A search across all
`.py`/`.md`/`.txt`/`.json` files for `recall`, `precision`, `ndcg`, `mrr`,
`hit_rate`, `false_positive`, `latency`, `benchmark`, `p95` returned **no
retrieval-quality artifact**. There is no `conftest.py`, `pytest.ini`,
`pyproject.toml`, or eval harness of any kind.

Therefore: the *direction* of those claims is independently confirmed by the
code and the §3.1 probe (exact works, paraphrase and indirect fail,
cross-session works, false positives are mechanically possible), but **the
claims are not measurements and no numbers exist to reproduce or refute.**
Treat them as qualitative priors, not as data. M6 must produce real numbers
before any retrieval behavior is changed.

### 3.4 Documentation drift found during the audit

CURRENT — `CONTEXT_MANAGER.md` is accurate in outline but wrong in three
places, and predates M1–M5:

| Claim in `CONTEXT_MANAGER.md` | Actual code |
|---|---|
| "relevance — word-overlap … user-role weight ×2 via authority multiplier" (line 73) | `relevance()` has **no** role weight. The role split exists only in `importance_of`. |
| "importance … user-authored ×1.0 / assistant ×0.4" (line 77) | User is `0.15 + base`, not `×1.0`. |
| "Candidates come from FTS5/LIKE over older rows (excluding already-selected seqs)" (line 97) | Exclusion is by `role\x00content` **fingerprint**, not by `seq`. |
| Retrieval described as one pass | Cross-session state passing and the M1–M5 state/entity/relationship layers are absent from the document entirely. |

`DOCS/architecture/AUDIT_01_CURRENT_STATE.md` is a **stale snapshot** (dated
2026-09-16): it describes a tree with no `prime/context/` directory and
reports "41 tests pass" where the repository now has **359 passing**. It
contains no retrieval analysis and must not be used as a current-state
reference.

`DOCS/01_architecture/MEMORY_ENTITY_RELATIONSHIP.md` is current and explicitly
reserves semantic retrieval, hybrid retrieval, and reranking for a later
milestone (§ Explicit exclusions). M6 is that milestone.

---

## 4. Target hybrid architecture

**PROPOSED — NOT IMPLEMENTED**

```
        Query (current turn + resolved entities + explicit temporal intent)
          │
          ├── A. Lexical      FTS5/LIKE over messages          [fuzzy]
          ├── B. Semantic     cosine over stored vectors        [fuzzy]
          ├── C. Temporal     M4 validity / as_of / history     [deterministic gate]
          ├── D. Entity       M5 resolve_entity                 [high precision]
          ├── E. Relationship M5 bounded one-hop                [high precision]
          └── F. Provenance   source_ids carried on every item  [audit]
          │
          ▼
   Deterministic gates (scope → authority → status → temporal → supersession)
          │
          ▼
   Candidate pool (deduped by item key, per-channel ranks recorded)
          │
          ▼
   Fusion (rank fusion, not score mixing)
          │
          ▼
   Relevance floor + bounded tie-break layer
          │
          ▼
   Rerank (deterministic features only — no second model)
          │
          ▼
   Redundancy suppression → budgeted admission (existing `input_ceiling`)
          │
          ▼
   ContextView (unchanged format and order)
          │
          ▼
   ONE Aurora brain
```

Design rules:

1. **Aurora remains the only model.** No second model, no cross-encoder, no
   second agent. (See §15 and §22-D1.)
2. **Gates run before scoring.** M1–M5 lifecycle, authority, scope, and
   temporal rules are absolute filters, never ranking inputs.
3. **FTS is preserved.** It remains a first-class channel and the lexical
   floor; hybrid augments it, it does not replace it.
4. **No unbounded traversal.** The relationship channel is M5's bounded,
   one-hop, mandatory-limit query surface. M6 adds no new graph capability.
5. **Read-only with respect to memory.** M6 never writes states, entities, or
   relationships. The only permitted write is the derived vector index
   (§16.2), which is a cache of already-committed content.

---

## 5. Channel A — Lexical retrieval

**CURRENT (preserved unchanged) — plus two PROPOSED fixes.**

PROPOSED M6.1 changes, both minimal and behavior-preserving on the exact-match
class:

- **FTS term quoting.** Before `MATCH`, wrap each term as a quoted FTS5 string
  literal (or strip characters outside `[a-z0-9]`) so `don't`, `apollo's`,
  `BLUE-FALCON-7`, and `design?` stop raising `OperationalError` and are
  silently discarded. Fixes W5 without changing the tokenizer.
- **Query-term weighting by origin.** Terms from the current turn and terms
  from the previous turn are kept separate so `relevance` is not diluted by
  the previous turn's vocabulary. Fixes W4.

PROPOSED M6.4 changes:

- Optionally add `bm25()` as a *within-channel* ordering signal. It is
  **not** used for cross-item admission ordering, which stays deterministic
  and explicit in Python. Fixes W6 partially.
- Near-stopword suppression for *matching* (not for tokenization) so
  `about`/`project` do not act as evidence. Fixes W13.

**NOT IMPLEMENTED:** replacing FTS, semantic tokenizers inside FTS5, learned
sparse retrieval, BM25-driven final admission.

---

## 6. Channel B — Semantic retrieval

**PROPOSED — NOT IMPLEMENTED**

### 6.1 What is embedded

Evaluated and decided in §16. Summary: embed `states`, `entities`, and
`messages`. Do **not** embed `relationships` (they hold no text — only stable
IDs) and do **not** embed `digests` (they are verbatim spans of messages whose
vectors already exist).

### 6.2 Model recommendation

**`sentence-transformers/all-MiniLM-L6-v2`, CPU, float32, 384 dimensions,
mean pooling, L2-normalized, `max_seq_length=256`.**

Recommendation basis — this is not a popularity choice. Measured on this
machine, and verified as already present:

| Fact | Value | Source |
|---|---|---|
| Model already on disk | `models--sentence-transformers--all-MiniLM-L6-v2`, **87.3 MB** | HF cache inspection |
| Library already installed | `sentence-transformers 6.0.0` | import probe |
| Parameters / dimension | 22.7 M / 384 | probe |
| CPU latency, 6 queries | **11.7 ms** (1.96 ms/query) | probe, 6 torch threads |
| GPU latency, 6 queries | **6.3 ms** (1.05 ms/query) | probe |
| GPU VRAM, weights / peak | **9.1 MB / 105.6 MB** | probe |
| 48 short docs, CPU / GPU | 80.5 ms / 11.5 ms | probe |
| Exact cosine, 48×384 | **0.109 ms** | probe |

**Why CPU-first, not GPU:**

- Aurora already holds the 4-bit NF4 Qwen3-8B model on the same 8 GB card
  with `device_map={"": 0}` (`brain.py:157-174`). Adding a second CUDA model
  in the same process risks fragmentation and OOM for a 1 ms saving.
- 1.96 ms is negligible next to a 4-bit 8B generation on the same machine.
  Optimizing it buys nothing observable.
- float32 CPU inference is bit-reproducible; fp16 GPU inference is not. M1–M5
  are deterministic, and M6 must not silently introduce nondeterminism.
- The 6 available CPU threads are shared with Aurora's generation path;
  a short 2 ms embed is a smaller disturbance than a CUDA context switch.

PROPOSED: CPU by default, with an opt-in device switch recorded as a
configuration value. GPU is an optimization to revisit only if the benchmark
shows a real latency problem.

### 6.3 Alternatives evaluated and rejected

| Option | Verdict | Reason |
|---|---|---|
| Larger embedder (bge-m3, e5-large, gte-large, nomic) | Deferred, not rejected outright | Needs a download, 100–500 MB VRAM or much more CPU time. Revisit only if the benchmark shows paraphrase recall below target. |
| Aurora itself as embedder | **Rejected** | Violates the one-brain rule and puts generation latency on every retrieval. |
| Cross-encoder reranker | **Rejected for M6** | Second model. See §22-D1. |
| `onnxruntime` CPU inference | Rejected | Not installed; adds a dependency to save ~1 ms. |
| Hashing / TF-IDF + SVD "embeddings" | Rejected | Cannot capture paraphrase; defeats the purpose of the channel. |
| `sqlite-vec`, `faiss`, `hnswlib`, `usearch` | Rejected for now | None installed. Brute force is already 0.109 ms at this scale (§16.1). |

### 6.4 Semantic query handling

PROPOSED:

- The query text embedded is the **current user turn only**, not the
  previous-turn concatenation, so the vector reflects what the user is asking
  now.
- Long messages are chunked at sentence boundaries with overlap; each chunk
  is a separate vector row that references its parent `message_id`, so results
  remain attributable and re-joinable.
- A **similarity floor τ** is required for admission. τ must be calibrated on
  the §18 benchmark, not guessed. It is a named constant in one place.

---

## 7. Channel C — Temporal retrieval

**PROPOSED — reuse M4 exactly, expose as a channel.**

M4 already implements half-open `[valid_from, valid_until)` intervals,
`as_of`, `history`, `previous`, `changes`, `when_changed`, and
`what_replaced`. M6 adds no new temporal semantics.

- `temporal="current"` (default): `valid_from <= t < valid_until`, status
  `current`, authority `user`.
- `temporal="as_of"(t)`: includes `superseded` rows effective at `t`.
- `temporal="historical"`: includes accepted superseded assertions.
- The existing `HISTORY_MARKERS` list is the **only** current trigger for
  historical behavior. It is 14 substrings including `"was"` and `"old"`, which
  over-triggers. PROPOSED: keep the markers as the explicit-request signal but
  log when they fire, and treat a fired marker as *permission to include
  history*, never as a command.

**NOT IMPLEMENTED:** natural-language temporal parsing ("after July",
"last week"). M4's test suite explicitly pins this as deferred
(`test_memory_temporal.py::test_natural_language_temporal_parsing_remains_deferred`).
"what changed after July" therefore remains a lexical/semantic question over
already-stored text, not a temporal query, until a separate approved milestone.

---

## 8. Channel D — Entity retrieval

**PROPOSED — read-only use of the existing M5 Stage C query surface.**

Inputs: the current user turn is resolved with M5's deterministic
`resolve_entity(scope, name)` — exact canonical name or confirmed alias only.
The M5 doc and Stage C tests already pin that `resolve_concept()` is **not**
entity resolution and that no semantic alias inference occurs.

Two uses, both bounded:

1. **Entity-scoped memory lookup.** `get_entity_memories(scope, entity_id)`
   returns states linked by a stored `concerns` edge. This is how "which
   memories concern X" is answered structurally.
2. **Entity-seeded lexical/semantic expansion.** Resolved entity names and
   their accepted aliases become additional *query terms* for channels A and
   B, not additional evidence.

PROPOSED constraint: an entity hit is **high-precision and low-recall**. It may
promote a candidate, never create one, and it must never bypass the M5 gates.
Ambiguous resolution stays ambiguous and contributes nothing.

**NOT IMPLEMENTED:** semantic alias inference, entity-type guessing, automatic
entity creation from a query.

---

## 9. Channel E — Relationship retrieval

**PROPOSED — read-only use of M5 Stage C, no new traversal.**

- `get_entity_relationships(scope, entity_id, direction=...)` for one-hop
  outbound/inbound.
- `get_contained_entities(...)` for the **derived** `contains` inverse of
  inbound `part_of`. The inverse is derived at read time and is never stored.
- `get_relationship_explanation(...)` for the fixed
  `rationale -explains-> decision` path, including its fixed second segment
  `decision -concerns-> entity`.

Bounds that carry over unchanged and are non-negotiable: `max_hops = 1`,
mandatory result limit, `authority == user`, `status == current`,
temporally valid, active endpoints, scope-explicit, and no cross-session
traversal without an explicit request.

**NOT IMPLEMENTED:** multi-hop traversal, connected components, shortest
path, generic `related_to`, persisted inverse rows, a graph database.

---

## 10. Channel F — Provenance

**PROPOSED**

Provenance is a **requirement on every candidate**, not a ranking channel:

- Every retrieved item carries `source_ids` → `messages.id`, resolvable to
  verbatim text.
- Every state/entity/relationship candidate carries its M1–M5 lifecycle and
  temporal metadata unchanged.
- No channel may synthesize, paraphrase, or merge text into a candidate.
  Semantic chunks always reference a real parent `message_id`.
- Retrieval never mutates provenance. It reads it.

This is what makes §17 explainability possible and what lets a wrong result be
traced back to the exact evidence that caused it.

---

## 11. Candidate fusion

**PROPOSED — NOT IMPLEMENTED**

### 11.1 Why not score mixing

Cosine similarity, Jaccard overlap, RRF rank, and entity-match are
incommensurable. Adding them with weights would require calibrating four
independent scales against each other, and the calibration would silently rot
as the corpus changes.

### 11.2 Proposed mechanism: rank fusion with a lexical floor

1. Each channel produces a bounded, independently ranked candidate list with
   its own rank positions. Channels do not see each other's scores.
2. Lists are unioned by item key (`message_id` | `state_id` | `entity_id`).
3. **Positive-evidence floor** — a candidate may only be admitted if at least
   one channel produced it as a *positive* hit:
   - lexical: matched ≥1 non-near-stopword query term, **or**
   - semantic: cosine ≥ τ, **or**
   - entity/relationship: an M5 hit survived the M5 gates.

   This is the direct fix for W3. A candidate can no longer be admitted on
   recency and importance alone.
4. **Reciprocal Rank Fusion** across channels:
   `fused(c) = Σ_channels weight_channel / (k + rank_channel(c))`, with a
   single constant `k`. RRF is used because it is scale-free, monotone,
   deterministic, and needs one tunable constant instead of four.
5. The **channel-agreement set** — which channels produced this candidate — is
   retained and is the strongest explainability signal available.

### 11.3 Deterministic tie-break layer

Applied **after** fusion, as a bounded multiplier with named, individually
justified weights. Never folded into one "relevance" number:

| Signal | Role | Rationale |
|---|---|---|
| M5 entity/relationship hit | bounded boost | Explicit structural connection is stronger evidence than similarity. |
| Stored `importance` (M1) | small boost | Author-declared salience; already authority-weighted. |
| Recency | small decay | Ties and near-ties only; must not beat a strong evidence match. |
| Stored M1 `confidence` | **never a ranking input** | M5: "Confidence never makes an assertion current." It gates, it does not rank. |

### 11.4 Concept separation (mandatory)

These are five different things and the implementation must keep them
separate:

| Concept | Meaning | M6 role |
|---|---|---|
| **Evidence relevance** | Did the query touch this item? | The only thing ranking measures. |
| **Truth / authority** | Is this user-authored? | **Absolute gate.** Never scored. |
| **Confidence** | How well evidenced is the assertion? | **Gate for currentness.** Never scored, never boosted. |
| **Currentness** | Is it valid now / at `t`? | **Absolute gate** via M4 validity. |
| **Importance** | Author-declared salience signal | Ranking tie-break only. |
| **Retrieval score** | Internal, per-request ordering artifact | **Never persisted, never shown to Aurora, never used to change memory.** |

A retrieval score must never change memory truth. Retrieval is read-only with
respect to states, entities, and relationships.

---

## 12. Query type behavior

**PROPOSED — NOT IMPLEMENTED.** Examples are illustrative, not hardcoded. No
query-type classifier is required: the existing channel set already produces
the right emphasis, and the floor plus gates do the discriminating.

| # | Type | Example | Dominant channel | Notes |
|---|---|---|---|---|
| 1 | Exact | `what did we decide about the car design?` | A lexical | Must not regress. Pinned by the §18 lexical lock. |
| 2 | Paraphrase | `why did we reject that earlier approach?` | B semantic | Today's failure (W1). Needs cosine ≥ τ over states and messages. |
| 3 | Indirect reference | `what did I mean when I said that last week?` | B + E | Pronouns need semantic context; a temporal hint routes to the right window. Genuinely hard; expected to be the weakest class. |
| 4 | Entity-based | `what do I currently know about Ferrari?` | D + E | Resolve the entity, then `get_entity_memories` / `get_relationship_explanation`. Highest precision. |
| 5 | Relationship-based | `what projects involve this component?` | D + E | One hop, derived `contains` for the inbound direction. |
| 6 | Temporal | `what changed after July?` | A + B over a window | No NL temporal parsing (§7). Relies on stored text containing the date. Explicitly a known limitation. |
| 7 | No-match | `tell me a joke about cats` | none | Floor plus gates must yield zero candidates. Must be measured, not assumed — there is no such test today. |
| 8 | Mixed | `what did we decide about the car design, and does it affect the mobile app?` | A + B + D + E | All channels contribute; fusion arbitrates. |

Note on the probe in §6: MiniLM's top-1 for query 5 was
*"The car design is a matte charcoal three-door"* rather than the
*"settings module belongs to the mobile app, which is part of Apollo"* line.
That is a real, observed semantic false positive and is the canonical example
for §19.

---

## 13. Scope handling

**PROPOSED — extends CURRENT M1–M5 rules without altering them.**

| Rule | Behavior |
|---|---|
| Default | Current session + `global`, exactly as today. |
| Session A / Session B | Separate identity, separate results. Never merged. |
| Foreign sessions | Only when explicitly requested (named session, or a `CROSS_SESSION_MARKERS` phrase) **or** under the existing always-on-but-tagged policy — see §22-D2. |
| Global rows | Visible per existing rules; `session_id` on a global row is provenance, not ownership. |
| Entity/relationship scope | M5 APIs are scope-explicit. A query never traverses another session's graph. |
| State scope | Unchanged: a session's value is never substituted for another's. |
| Temporal scope | Unchanged: M4 validity gates. |

Two PROPOSED safeguards, both fixing verified findings:

- **Same-session dominance.** Foreign-session candidates are ranked strictly
  below same-session candidates of equal fused score, and are additionally
  capped (existing cap: 3). Fixes W10, where `session_id` is currently only a
  sort tiebreak.
- **Tagged provenance is mandatory and unchanged.** Every foreign item keeps
  the `[from another conversation]` prefix. No automatic context leakage: a
  foreign item may inform, never impersonate.

**No automatic context leakage is permitted in any M6 configuration.**

---

## 14. Active Memory and Working Memory interaction

### 14.1 Active Memory

**PROPOSED — no redesign.**

Active Memory stays a **computed, on-demand projection**. M6 does not add a
second stored copy, does not change its membership rule (current states, user
authority, temporally valid, newest per scope/concept), and does not expand it
globally.

Intended dataflow:

```
retrieval candidates
  → relevance floor + gates
  → bounded proposal set
  → Active Memory selection (existing `select_active_memory` seam)
  → item/token budget
```

Constraints:

- A retrieved item may be **proposed** to Active Memory. It may not bypass
  the existing membership rule.
- Active Memory is never auto-expanded with every entity or relationship.
- The `input_target` / `input_ceiling` budget is never weakened by retrieval.
- CURRENT, unresolved from M5: Active Memory items have **no hard count cap**
  today. `MEMORY_ENTITY_RELATIONSHIP.md` makes a bounded projection a
  prerequisite for any M5 expansion and defers the cap to M5.4. M6 must not
  expand Active Memory before that cap exists. See §22-D6.

### 14.2 Working Memory

**PROPOSED — no redesign, no persistence.**

- Runtime-only, task-scoped, bounded (`max_items`, `max_tokens`).
- Relevant retrieval results may be adopted into Working Memory through an
  **explicit** call, alongside the existing `add_evidence`.
- No persistent copy of the vector index or of any retrieval result.
- No automatic live refresh. Long-running tasks re-resolve explicitly, because
  adopted results describe the state at adoption time.
- Promotion back to durable memory still goes only through
  `promote_to_candidate` → Candidates → confirmation. Retrieval never writes
  current truth.

---

## 15. RAG boundary

**M6 = "Which memories and evidence are relevant?"**
**M7 = "How do we optimally assemble that retrieved knowledge for Aurora's brain?"**

| Concern | Owner |
|---|---|
| Candidate generation across channels | **M6** |
| Fusion, ranking, relevance floor | **M6** |
| Temporal / entity / relationship gating | **M6** |
| Bounded selection under the token budget | **M6** |
| Retrieval explainability trace | **M6** |
| ContextView format, section order, prompt wording | M7 |
| Citation formatting and inline attribution | M7 |
| Query rewriting, decomposition, expansion | M7 |
| Iterative / multi-hop / agentic retrieval | M7 |
| Chunking strategy for long documents (beyond M6's minimal message chunking) | M7 |
| Context compression / summarization of retrieved material | M7 |
| Answer synthesis strategy | M7 |
| Prompt layout and instruction design for retrieved context | M7 |

M6 explicitly does **not** change: `ContextView`'s structure or the assembly
order in `build()`, the token budget arithmetic, the message role conventions
(`role="system"` for synthesized context, original role for raw evidence), or
any prompt text. M6 changes *which* items are admitted, never *how* they are
rendered or ordered in the final view.

---

## 16. Embedding and index strategy

### 16.1 What gets embedded — the smallest useful set

**PROPOSED — NOT IMPLEMENTED**

| Structure | Embed? | Rationale |
|---|---|---|
| `states` (`concept = value`) | **Yes** | The authoritative memory, small in cardinality, highest value per vector. Directly fixes W1/W2 for remembered facts. |
| `entities` (display name + type + accepted aliases) | **Yes** | Tiny. Gives the entity channel a vector form and makes name variants reachable. |
| `messages` | **Yes, second** | Largest and most valuable for indirect reference, but the growth driver. See §22-D3. |
| `relationships` | **No** | They contain **no text** — only stable IDs, by M5 design. There is nothing to embed. Use the structured query surface instead. |
| `digests` | **No** | Verbatim spans of messages whose vectors already exist. Embedding them duplicates the message index and creates a second, drifting copy. |

Deliberately minimal: two tables first, one table second, two tables never.

### 16.2 Storage — SQLite, brute force, no new dependency

**PROPOSED — NOT IMPLEMENTED**

```
retrieval_vectors(
  vector_id     INTEGER PRIMARY KEY,
  kind          TEXT,        -- 'state' | 'entity' | 'message_chunk'
  ref_id        INTEGER,     -- state_id | entity_id | message_id
  chunk_index   INTEGER,     -- 0 for unchunked rows
  scope         TEXT,        -- session id or 'global'
  session_id    TEXT,        -- provenance
  model_id      TEXT,        -- e.g. 'all-MiniLM-L6-v2@fp32'
  dim           INTEGER,     -- 384
  norm          REAL,        -- 1.0; reject non-normalized rows
  content_hash  TEXT,        -- sha256 of embedded text
  vector        BLOB,        -- float32[dim], little-endian
  created_at    REAL
)
retrieval_index_meta(key TEXT PRIMARY KEY, value TEXT)
```

Rules:

- **Same SQLite file, same transaction discipline.** No new service, no
  separate vector database, no daemon. Local-first is preserved.
- **Brute-force cosine in numpy.** Verified 0.109 ms for 48×384. Even at
  100 k vectors this is a ~15 GFLOP matmul, well inside a CPU budget.
  **No ANN index is justified at PRIMORA's scale.** Revisit only above
  ~250 k vectors or a measured p95 above 50 ms.
- **`model_id` + `content_hash` make the index rebuildable and verifiable.**
  A model change or a content edit changes the hash and forces re-embedding.
  Stale rows can never be silently served.
- The index is a **derived cache**. Deleting it must be lossless: it can
  always be rebuilt from `states`, `entities`, and `messages`.
- **NOT PROPOSED:** `sqlite-vec`, `faiss`, `hnswlib`, `usearch`, or any native
  extension. All are absent from this environment and none is needed.

### 16.3 Indexing policy

**PROPOSED — NOT IMPLEMENTED**

- Incremental: embed on write (same transaction as the memory write) for
  `states` and `entities`, which are low-volume and authoritative.
- For `messages`, embed in bounded background batches at reconcile time, with
  an explicit cap per request so indexing never delays a user turn.
- A `rebuild` entry point rebuilds the whole index deterministically from
  source rows. Diagnostic first, mutation second — matching the M5
  reconciliation rule.
- Indexing failures degrade diagnostically: retrieval falls back to lexical
  only, emits a diagnostic, and never fails the request.

---

## 17. Retrieval explainability

**PROPOSED — NOT IMPLEMENTED**

An internal structured trace, one entry per considered candidate:

```
RetrievalTraceItem
  item_key            # message_id / state_id / entity_id
  kind                # message | state | entity
  channels            # which channels produced it
  channel_ranks       # per channel rank + raw score
  lexical_terms       # matched query terms
  cosine              # semantic score, if any
  entity_match        # resolved entity id, if any
  relationship_match  # relationship id + type, if any
  fused               # RRF value
  gate_results        # scope / authority / status / temporal, pass|fail|reason
  floor_passed        # bool
  tie_break           # applied tie-break signals
  final_rank          # int or None
  admitted            # bool
  admit_reason        # admitted | below_floor | failed_gate | out_of_budget
RetrievalTrace
  query_text
  resolved_entities   # entity ids
  temporal_mode
  scope
  at_time
  items               # list[RetrievalTraceItem]
  stage_timings_ms    # per-channel and total
  degraded            # bool
  diagnostics         # list[str]
```

Rules:

- **Never injected into the prompt.** It lives on `BuildResult`, not on
  `ContextView`, and is off by default in production and on for benchmarks.
- Deterministic and comparable across runs, which is what makes the §18
  benchmark possible.
- Every admitted item can answer "why was this retrieved?" with a channel
  name and a number — never with prose alone.
- A retrieval score is **never persisted** and never changes memory.

---

## 18. Benchmark design

**PROPOSED — NOT IMPLEMENTED.** There is no retrieval benchmark today (W14),
so this must be built and run **before** any retrieval behavior changes.

### 18.1 Harness

`TESTS/prime/test_retrieval_benchmark.py`, plus a non-test runner for
reporting. Properties:

- Builds its corpus in a **temp database** with the real `ContextStore`,
  `ContextManager`, M5 entities, and M5 relationships.
- **Never touches `PRIMORA/data/context/aurora_context.db`.** The benchmark
  asserts this by hashing the production file before and after.
- No benchmark run may alter production memory, ranking weights, or scope.
- Corpus is a **fixed JSON fixture** (seeded sessions, states, entities,
  relationships, gold labels) committed alongside the tests, so results are
  reproducible and reviewable.
- Purely lexical and semantic retrieval; no Aurora generation.

### 18.2 Query classes and minimum counts

| Class | Minimum | Example shape |
|---|---|---|
| exact | 40 | shares content tokens with the target |
| paraphrase | 40 | synonym substitution, no shared content token |
| indirect-reference | 20 | pronoun / descriptive reference |
| entity | 20 | resolvable canonical entity name or confirmed alias |
| relationship | 20 | answerable only through a stored edge or derived inverse |
| temporal | 15 | requires history, `as_of`, or a superseded assertion |
| cross-session | 15 | target lives in another session; some explicit, some not |
| no-match | 30 | correct answer is *nothing* |

### 18.3 Metrics

| Metric | Definition | Target |
|---|---|---|
| Recall@1 / @3 / @5 | gold item present in top-k | ≥ baseline + measured delta |
| Precision@k | relevant items in top-k / k | report, do not gate initially |
| MRR | mean reciprocal rank of the first gold | report |
| False-positive rate | non-gold items admitted / total admitted | **must not regress** vs lexical baseline |
| No-match accuracy | fraction of no-match queries admitting nothing | **hard target: 1.00** |
| Paraphrase recall | Recall@3 on the paraphrase class | the primary M6 success metric |
| Indirect-reference recall | Recall@3 on the indirect class | report honestly; expected weakest |
| Cross-session recall | Recall@3 on the cross-session class | must not regress |
| Superseded safety | stale items admitted on current queries | **hard target: 0** |
| Scope leakage | foreign-session items admitted without an explicit request | **hard target: 0** |
| Determinism | identical ordering across 5 runs | **hard target: identical** |
| Latency | p50/p95 per stage: embed, index read, fuse, admit | p95 total budget documented |

### 18.4 Baselines and the lexical lock

- The benchmark must first record the **current lexical system** as the
  baseline on the same fixture. Hybrid claims are only meaningful as deltas.
- **Lexical lock:** every query in the exact class whose answer the current
  system already produces must keep producing it. This is the regression gate
  that makes "hybrid augments FTS" testable rather than aspirational.
- A **shadow mode** stage (M6.3) runs the semantic channel, records the trace,
  and changes no output. Deltas are measured from shadow traces before the
  semantic channel is allowed to affect admission.

---

## 19. Failure modes and deterministic safeguards

| Risk | Deterministic safeguard |
|---|---|
| **Semantic false positives** — a vaguely similar item outranks the right one | Cosine floor τ; positive-evidence floor; no-match accuracy target 1.00; false-positive rate must not regress vs baseline. |
| **Embedding similarity overriding exact evidence** | Lexical exact hits get a bounded boost inside the tie-break layer only. A candidate whose evidence is a weak cosine match can never outrank a strong lexical match. |
| **Retrieval of stale historical memory** | M4 gate: `status == current` and `valid_from <= t < valid_until` for current queries. Existing superseded suppression retained. Benchmark target: 0 stale admissions. |
| **Scope leakage** | Explicit scope everywhere; entity/relationship queries are M5 scope-gated; same-session dominance; cross-session cap retained; benchmark target: 0 unauthorized foreign admissions. |
| **Duplicate evidence** | Existing `role\x00content` fingerprint dedupe, plus cross-channel dedupe by item key at fusion, plus the existing Jaccard redundancy penalty. |
| **Relationship overexpansion** | M5 bounds are inherited verbatim: one hop, mandatory limit, active endpoints. No new traversal is added in M6. |
| **Irrelevant semantic matches** | τ floor; near-stopword suppression; channel-agreement requirement is not mandatory but is recorded; no-match class measured. |
| **High embedding latency** | CPU-first, 1.96 ms/query measured; indexing is incremental and off the user-turn critical path; a per-stage latency budget with a documented p95; retrieval degrades to lexical-only on timeout. |
| **Excessive context size** | The existing `input_ceiling` is a hard stop and is never relaxed. Candidate pools are bounded per channel. Admission is unchanged. |
| **Index/model drift** | `model_id` + `content_hash` per row; a mismatch forces re-embedding; a mismatch at query time disables the semantic channel and degrades to lexical. |
| **Index corruption or loss** | The index is a derived cache; `rebuild` is deterministic and lossless; retrieval degrades to lexical-only. |
| **Benchmark leaking into production** | Benchmarks run against temp databases only and assert the production DB hash is unchanged. |
| **Ranking score corrupting memory truth** | Retrieval is read-only. No score is persisted, and no score ever gates lifecycle, authority, or confidence. |

---

## 20. Hardware and latency considerations

**CURRENT — measured on this machine**

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti, **8.00 GB** |
| GPU free at probe time | 6.91 GB (Aurora not resident in the probe process) |
| CPU | Intel Core i5-12400F, 6 cores / 12 threads, 2.5 GHz |
| RAM | 15.8 GB |
| torch | 2.11.0+cu128 (6 torch threads configured) |
| Aurora model | Qwen3-8B architecture (hidden 4096, 36 layers), 4-bit NF4, `device_map={"": 0}` |
| Aurora weights on disk | 15.6 GB bf16 base + 14.7 MB LoRA adapter |
| `sentence-transformers` | **6.0.0, already installed** |
| SQLite | 3.45.3, FTS5 available, JSON1 available |
| Absent | `faiss`, `hnswlib`, `usearch`, `sqlite-vec`, `onnxruntime` |

**CURRENT — embedding cost, measured**

| Operation | CPU | GPU |
|---|---|---|
| Encode 6 queries | 11.7 ms (1.96 ms/query) | 6.3 ms (1.05 ms/query) |
| Encode 48 short docs | 80.5 ms | 11.5 ms |
| VRAM weights / peak | — | 9.1 MB / 105.6 MB |
| Exact cosine 48×384 @ 6 queries | 0.109 ms | — |
| Storage per vector | 1536 B (384×float32) | — |
| 10 k vectors / 100 k vectors | 14.6 MB / 146.5 MB | — |

**Assessment**

- The embedding channel costs **~2 ms per query on CPU** and **~0.1 ms** of
  cosine search. Both are negligible against a 4-bit 8B generation on the same
  machine. Latency is a solved problem at this scale, not a design constraint.
- VRAM is the real constraint, and it argues **against** the GPU: Aurora
  already pins the 4-bit 8B model to the full card. A 105.6 MB peak is
  affordable in isolation but is not worth the fragmentation and OOM risk
  during generation.
- The binding constraint is **RAM and disk, not VRAM**: 100 k message vectors
  is 146.5 MB on disk, and a 12-thread CPU matmul over that is still
  milliseconds. A larger embedding model would multiply both figures for a
  quality gain that must first be demonstrated by the §18 benchmark.
- Conclusion: **small local model, CPU, brute-force cosine, SQLite BLOB
  storage, zero new dependencies.** Revisit only with benchmark evidence.

---

## 21. Acceptance invariants

An implementation satisfies M6 only when all of the following hold:

1. Raw `messages` remain append-only and unchanged.
2. `states` remain the authoritative memory record.
3. Retrieval never writes states, candidates, entities, or relationships.
4. No retrieval score is ever persisted, and no score changes memory truth,
   authority, confidence, or lifecycle.
5. M1–M4 authority, status, confidence, and temporal gates are applied
   **before** ranking and are absolute.
6. M4 half-open interval semantics are preserved exactly.
7. Unknown historical validity is never fabricated.
8. M5 scope rules are preserved; no automatic context leakage in any mode.
9. M5 entity/relationship query bounds are inherited verbatim: one hop,
   mandatory limit, active endpoints, no inverse rows persisted.
10. The lexical channel is preserved, and the exact-query class does not
    regress against the recorded baseline.
11. Every retrieval operation is bounded in items and in tokens, and the
    existing `input_ceiling` is never relaxed.
12. Retrieval is deterministic: identical inputs produce identical ordering.
13. Every admitted item is explainable by a structured, non-prompt-visible
    trace naming the channel and score that admitted it.
14. Active Memory remains computed, current, scope-valid, and bounded, with no
    second stored copy.
15. Working Memory remains runtime-only and uses the candidate flow for
    promotion.
16. The vector index is a rebuildable derived cache; deleting it is lossless.
17. The vector index is scoped, and a scope or model mismatch degrades to
    lexical-only retrieval.
18. The embedding model is a small local model; no second generative model is
    introduced; no cross-encoder is introduced.
19. No graph database, no vector-search service, no network dependency, and no
    mandatory download is introduced.
20. `ContextView` structure, assembly order, message role conventions, and
    prompt wording are unchanged.
21. No benchmark run may alter production memory or the production database.
22. No natural-language temporal parsing is introduced.
23. Retrieval failures degrade diagnostically and never fail a user turn.

---

## 22. Future implementation stages

**PROPOSED — each stage requires separate approval. None is implemented.**

| Stage | Deliverable | Boundary |
|---|---|---|
| M6.0 | This specification and audit | Documentation only |
| M6.1 | Benchmark harness + golden fixture + lexical baseline recorded | **No retrieval behavior changes at all** |
| M6.2 | Vector store, indexer, rebuild, and the two lexical fixes (§5) | Index is off the query path; lexical fixes are bug fixes |
| M6.3 | Semantic channel in **shadow mode** (trace only, no output effect) | Measures deltas before anything is trusted |
| M6.4 | Gated hybrid in the query path: floor + fusion + rerank | Lexical lock enforced; shadow deltas must justify it |
| M6.5 | Entity and relationship channels wired into retrieval | M5 bounds inherited; no new traversal |
| M6.6 | Optional device switch and larger-model evaluation | Only with benchmark evidence of a quality gap |

M6.1 must land before M6.2. There is no measured baseline today, so any change
made before the benchmark exists is unmeasurable.

### Unresolved decisions requiring Amir's approval

| ID | Decision | Recommendation |
|---|---|---|
| **D1** | Is a small **cross-encoder reranker** allowed later? | **No for M6.** It is a second model. Revisit only as an explicitly approved milestone with its own VRAM budget. |
| **D2** | Cross-session retrieval: keep the current always-on-but-tagged policy, or gate it on an explicit request? | **Keep always-on but tagged** (no recall regression), and add the same-session dominance rule from §13. Gating is a separate, later decision because it changes current behavior. |
| **D3** | Embed `messages` at M6.2, or defer to keep the first stage minimal? | **Embed `states` + `entities` first.** They are small, authoritative, and fix paraphrase recall for remembered facts. Add `messages` once the index-growth and latency profile is measured. |
| **D4** | Embedding model: `all-MiniLM-L6-v2` (already cached, 87 MB) or a larger model? | **MiniLM-L6-v2.** No download, no VRAM risk, 2 ms/query, already installed. Re-evaluate only against benchmark numbers. |
| **D5** | Values for the cosine floor τ and the RRF constant `k` | **Do not guess.** Calibrate on the M6.1 benchmark and record the chosen values with the results. |
| **D6** | Should M6 add a hard item cap to Active Memory, or leave it to M5.4? | **Leave it to M5.4.** M6 must not expand Active Memory before the cap exists. |
| **D7** | Is the vector index committed to git, or built locally on first run? | **Build locally on first run** and gitignore it. It is a derived cache, and 146 MB of binary does not belong in version control. |
| **D8** | Should the two lexical fixes (§5) ship in M6.1, before the benchmark? | **Yes.** FTS term quoting is a defect fix with no ranking semantics, and it is independently testable. Query-term weighting is a ranking change and should wait for the baseline. |

---

## 23. Explicit exclusions

M6 does **not** add:

- Advanced RAG (see §15 for the full M6/M7 split);
- a second generative model, a cross-encoder, or any agentic controller;
- query rewriting, decomposition, expansion, or multi-hop retrieval;
- a graph database, Neo4j, or a generic knowledge graph;
- a vector-search service, `sqlite-vec`, `faiss`, `hnswlib`, or `usearch`;
- an approximate-nearest-neighbour index (unjustified at this scale);
- a mandatory model download or any network dependency at runtime;
- natural-language temporal parsing;
- consolidation, confidence reinforcement, forgetting, or decay;
- new relationship types, persisted `contains`, or a `related_to` predicate;
- changes to M1–M5 write paths, lifecycle, authority, confidence, or scope;
- changes to `ContextView`, its assembly order, its role conventions, or its
  prompt wording;
- changes to Active Memory or Working Memory structure;
- changes to the Aurora brain, Prime routing, specialists, or Verse.

---

## Appendix A — Verified current-state summary

**CURRENT.** One line per audited fact, for quick reference.

- Retrieval starts at `ContextManager._retrieve`, called from `build()` step 6,
  twice per turn.
- Query = current user turn + most recent earlier user turn.
- Tokenizer `[a-z0-9']+` on lowercased text, minus a 60-word stopword list.
- No stemming, no lemmatization, no synonym handling.
- FTS5 external-content table, default `unicode61`, no stemmer or stoplist;
  availability probed at startup; `LIKE` fallback exists and is untested.
- `bm25()` is never called; FTS5 orders by `(session_id, seq)` and is a filter
  only.
- One `MATCH` per query term, `query_terms[:12]`, `limit=10`, ≤120 raw
  candidates.
- Tokens containing `'`, `-`, or `?` raise `OperationalError` and are dropped
  with `retrieval_skipped_term`.
- Score = `0.45·relevance + 0.25·recency + 0.30·importance − 0.50·redundancy`.
- `relevance` divides by the whole multi-turn query set.
- No relevance floor; recency and importance can outrank weak evidence.
- Budget: 1000 / 1100 / 128 / 1228, env-overridable; retrieval gets residual
  budget only.
- Superseded suppression is by source message id, coarse-grained.
- Cross-session retrieval always runs, is capped at 3, is tagged, and can
  outrank same-session items on score.
- Structured memory, digests, entities, and relationships are **not** reachable
  through retrieval.
- `MemoryProvider` is `None` in production.
- No retrieval benchmark exists anywhere in the repository.
- 359 tests pass; retrieval-sensitive assertions pin prompt prefixes
  (`[state current`, `[state previous`, `[states session`, `[from another
  conversation]`), diagnostic strings
  (`retrieval_suppressed_superseded`, `retrieval_skipped_term`), the
  `scoring` weights, and a `<= 10` filler bound in the long-conversation test.

## Appendix B — Audit method

**Read-only.** Nothing in `C:\PRIMORA PROJECT` was modified by this audit.

1. Read `scoring.py`, `store.py`, `manager.py`, `models.py`, `entities.py`,
   `relationships.py`, `queries.py`, `state.py`, `working.py`, `digest.py`,
   `brain.py`, `aurora.py`, and all four documents under `DOCS/`.
2. Inventoried every retrieval-relevant assertion in the test suite.
3. Environment probe: library versions, SQLite feature set, GPU model and
   VRAM, CPU, RAM, local model cache contents. Imported modules only; wrote
   nothing.
4. Retrieval probe: temporary SQLite stores under the OS temp directory,
   driving the real `ContextManager`. Tokenization, FTS `MATCH` behavior,
   paraphrase/indirect/exact recall, cross-session recall, score
   decomposition, and superseded suppression.
5. Embedding probe: `sentence-transformers/all-MiniLM-L6-v2` with
   `HF_HUB_OFFLINE=1` (already cached; nothing downloaded), measuring CPU and
   GPU latency, VRAM, cosine cost, and storage per vector.

Probe scripts were written outside the repository, to
`%TEMP%\opencode\`, and read PRIMORA modules by import only.
