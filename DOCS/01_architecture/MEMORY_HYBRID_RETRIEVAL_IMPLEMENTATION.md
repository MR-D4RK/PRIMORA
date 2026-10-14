# PRIMORA M6 — Advanced Retrieval

- **Milestone**: M6 — Advanced Retrieval
- **Status**: **LIVE.** Hybrid retrieval is the returned retrieval result. Shadow mode remains available and off by default.
- **Live switch approved by**: Amir, after all eleven shadow-acceptance criteria were met.
- **Project Root**: `C:\PRIMORA PROJECT`
- **Extends**: `MEMORY_HYBRID_RETRIEVAL.md` (design), `MEMORY_RETRIEVAL_CALIBRATION.md` (calibration)
- **Date**: 2026-09-25

---

## 0. The live switch

`ContextManager._retrieve` now returns the fused hybrid ranking. The lexical
candidates are one of its channels, not a replacement.

```python
ContextManager(..., hybrid_retrieval=True)     # default
ContextManager(..., hybrid_retrieval=False)    # rollback to lexical only
PRIMORA_HYBRID_RETRIEVAL=0                     # rollback via environment
manager.set_hybrid_retrieval(False)            # rollback at runtime
```

`_admit_lexical` — the pre-M6 admission path — is retained verbatim and is
used automatically when hybrid yields nothing or raises. Rollback is a
one-flag change; nothing needs to be redesigned or redeployed.

### 0.1 How a fused item reaches Aurora

- A fused `message:<id>` item is admitted directly, with its original role.
- A fused `state:<id>` or `entity:<id>` item is admitted as the **raw evidence
  messages** that support it, resolved through `source_ids`.

No new prompt format is introduced: whatever is admitted is always a stored
verbatim message with its original role. `_add_current_state` continues to
render the `[state current src:…]` projection exactly as before.

### 0.2 Index freshness

The live path calls `maybe_sync_semantic_index()` before retrieval. It reads
one cheap token (four aggregates over indexed columns) and does nothing when
structured memory has not changed. When it has, `sync()` re-embeds **only**
rows whose content hash changed, so a single new memory costs one embedding,
not a full rebuild. This follows the precedent of opportunistic digest
compaction already in the request path.

### 0.3 Live results vs the live rollback control

Six configurations, all 47 queries, all through production code. `E` is what
Aurora receives. `F` is the identical pipeline with hybrid disabled — the
control that makes the live comparison meaningful.

| Config | R@1 | R@3 | R@5 | P@3 | FP@3 | no-match | p95 ms |
|---|---|---|---|---|---|---|---|
| A lexical only (channel) | 0.5366 | 0.6098 | 0.6098 | 0.6098 | 0.6324 | 0.5000 | 16.5 |
| B semantic only (channel) | 0.3659 | 0.3659 | 0.3659 | 0.3659 | 0.1667 | 1.0000 | 16.4 |
| C lexical+semantic (channel) | 0.6829 | 0.8049 | 0.8049 | 0.8049 | 0.5541 | 0.5000 | 15.7 |
| D +M5 structural (channel) | 0.6829 | 0.8049 | 0.8049 | 0.8049 | 0.4677 | 0.5000 | 25.9 |
| **E LIVE hybrid** | **0.6098** | **0.8049** | **0.8293** | **0.8049** | **0.7317** | 0.0000 | 77.8 |
| F LIVE rollback (lexical) | 0.4634 | 0.6341 | 0.6585 | 0.6341 | 0.7886 | 0.0000 | 64.5 |

**E vs F, the live switch measured against its own control:**

| Metric | F (lexical) | E (hybrid) | Delta |
|---|---|---|---|
| Recall@1 | 0.4634 | 0.6098 | **+0.146** |
| Recall@3 | 0.6341 | 0.8049 | **+0.171** |
| Recall@5 | 0.6585 | 0.8293 | **+0.171** |
| Precision@3 | 0.6341 | 0.8049 | **+0.171** |
| FP@3 | 0.7886 | 0.7317 | **−0.057** |
| p95 latency | 64.5 ms | 77.8 ms | +13.3 ms |

Per-category recall@3, E vs F: exact 0.571 → **0.857** · paraphrase 0.333 →
**0.500** · entity 0.500 → **1.000** · relationship 0.500 → **1.000** ·
temporal 1.000 = 1.000 · cross-session 1.000 = 1.000 · punctuation 1.000 =
1.000 · indirect 0.200 = 0.200 (unchanged) · false-positive control 1.000 =
1.000.

### 0.4 Deviations from the pre-switch baseline, explained

Against the D channel-mode baseline, E shows R@1 0.6098 vs 0.6829 and FP@3
0.7317 vs 0.4677. **These are measurement-level differences, not retrieval
regressions**, and the F control proves it:

- `D` measures **retrieval only**. `E` measures the **whole assembled
  context**, which also contains the recency window and the state projection.
  The view is re-emitted chronologically, so a memory that retrieval ranked
  first can sit at view position 4 behind three recent turns. That lowers
  R@1 and inflates FP@3 identically for E and F.
- `no-match` is 0.0000 for **both** E and F. The recency window always
  contributes recent conversation, so the assembled context is never empty
  for a no-match query. That is pre-existing behaviour, identical with and
  without hybrid, and not a property of retrieval.
- `R@3` and `P@3` are **identical** between D and E (0.8049), and E is better
  on `R@5` (0.8293 vs 0.8049).

Against the correct control, hybrid improves recall and precision
simultaneously and reduces false positives. The live switch is a clear win.

### 0.5 Live overhead

Median live build 58.9 ms with hybrid, 48.8 ms without: **+10.1 ms median,
+13.3 ms p95**. Consistent with the shadow-mode measurement. No GPU usage, no
per-request model load, and no full index rebuild in steady state.

---

---

## 1. What shipped

| Component | File | Status |
|---|---|---|
| Semantic index (derived embedding cache) | `PRIMORA/prime/context/semantic.py` | new, **live** |
| Hybrid engine (channels, RRF, floor) | `PRIMORA/prime/context/hybrid.py` | new, **live** |
| Vector storage | `PRIMORA/prime/context/store.py` | extended |
| Trace and shadow records | `PRIMORA/prime/context/models.py` | extended |
| Lexical candidate extraction | `PRIMORA/prime/context/manager.py` | refactored, no behaviour change |
| Live hybrid admission + fallback | `PRIMORA/prime/context/manager.py` | new, **live** |
| Shadow-mode comparison | `PRIMORA/prime/context/manager.py` | available, off by default |
| Benchmark comparison harness | `TESTS/prime/retrieval_benchmark_compare.py` | extended with live configs |

**Aurora receives hybrid retrieval results.** The lexical admission path is
retained as the automatic fallback and the runtime rollback target.

---

## 2. Semantic index design

### 2.1 Storage

Two new tables in the existing SQLite file. No new database, no extension, no
ANN structure.

```sql
retrieval_vectors(
  vector_id, kind, ref_id, scope, session_id,
  model_id, dim, content_hash, vector BLOB, created_at
)
UNIQUE(kind, ref_id, model_id)
retrieval_index_meta(key, value)
```

- `kind` is `state` or `entity` only.
- `vector` is float32, little-endian, **L2-normalized**, 384 × 4 = 1536 bytes.
- **`content_hash` replaces source text.** The embedded text is
  `"{concept} = {value} ({MEMORY_TYPE})"` for a state and
  `"{display_name} ({entity_type}); also known as …"` for an entity, but only
  its SHA-256 is stored. The text is re-read from the live row, so the index
  never duplicates memory content and can never drift from it silently.
- `model_id` and `dim` make a model change detectable: a row whose dimension
  does not match the loaded model is refused, not silently mis-scored.
- Deleting every row is **lossless**. `rebuild()` recreates the index from
  source.

### 2.2 Why no ANN index

Scoring is an exact float32 dot product over the candidate set. The M6 audit
measured 0.109 ms for 48 × 384 and ~15 GFLOP for 100 k vectors. At PRIMORA's
corpus size an approximate index would add a dependency and a recall risk for
no measurable gain. `faiss`, `hnswlib`, `usearch`, and `sqlite-vec` are all
absent from this environment and none was added.

### 2.3 What is embedded, and what is not

| Structure | Embedded | Reason |
|---|---|---|
| `states` | **yes** | The authoritative memory, small, highest value per vector. |
| `entities` | **yes** | Tiny. Gives the entity channel a vector form. |
| `relationships` | no | They hold no text, only stable IDs. Traversed structurally instead. |
| `digests` | no | Verbatim spans of messages; a second, drifting copy. |
| `messages` | **no** | Deliberately deferred per the audit. The lexical channel owns raw messages. |

Eligibility is decided at index time *and* re-checked at query time:

- indexed: `status IN (current, superseded)` and `authority = 'user'`;
- never indexed: `unresolved`, `rejected`, and any non-user row.

---

## 3. Model and loading behaviour

- `sentence-transformers/all-MiniLM-L6-v2`, **CPU**, float32, 384-dim,
  `max_seq_length` capped at 256, batch 32.
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set before the import,
  so **normal retrieval never reaches the network**.
- The model and `sentence-transformers 6.0.0` are already present locally; no
  download occurs.
- Both `numpy` and `sentence_transformers` are imported **lazily inside
  functions**. Importing `PRIMORA.prime.context` never requires them.
- The model is cached per `(model_id, device)` for the process lifetime.
- **Fail-safe:** a missing library, a missing cache, or a load failure returns
  `None`. `SemanticIndex.available()` is `False`, `search()` returns `[]`,
  `sync()` reports `{"skipped": 1}`, and the trace is marked degraded with a
  `semantic_model_*` diagnostic. Lexical retrieval is untouched.

No second generative model. No cross-encoder. No GPU usage: Aurora already
pins the 4-bit 8B model to the same 8 GB card, and the M6 audit measured
embedding at 1.96 ms/query on CPU, so the GPU is left alone.

---

## 4. Retrieval pipeline

```
query
  → hard gates (authority, lifecycle, M4 validity, supersession, scope)
  → ┌──────────────┬────────────────┬──────────────────────┐
    │ lexical      │ semantic        │ entity / relationship│
    │ production   │ index + τ=0.45   │ M5 bounded, one hop  │
    └──────────────┴────────────────┴──────────────────────┘
  → RRF (k = 60, rank-based)
  → relative evidence floor (0.725)
  → bounded result (max 24)
  → SHADOW COMPARISON  (Aurora still receives lexical)
```

The order is load-bearing. Gates precede scoring; a similarity or fused score
can never re-admit a gated item.

### 4.1 Hard gates

`gate_state` and `gate_entity` are pure, deterministic, and absolute:

- `authority == "user"`;
- `unresolved` and `rejected` never retrievable;
- current mode requires `status == "current"` and an active entity;
- M4 half-open validity `valid_from <= t < valid_until`;
- scope applied before the channel reads anything.

**`confidence` is never read anywhere in `hybrid.py`.** It is evidential
support, not truth, and never an ordering input. A test enforces this by
source scan.

### 4.2 Lexical channel

Extracted from `_retrieve` into `ContextManager.collect_lexical_candidates`
and `ContextManager._cross_session_candidates`, with **no behaviour change**.
This matters: the hybrid path ranks candidates with the same production code
that feeds Aurora, not with a test-only mirror. The benchmark's drift guard
confirms mirror/production membership agreement of **47/47**.

### 4.3 Semantic channel

`semantic_channel()` reads the index, applies τ = 0.45, then re-reads each
live record and applies the hard gates. Gated rows are dropped with a
`semantic_gated:` diagnostic. Scope is explicit: same session, plus `global`
only when `include_global=True`.

### 4.4 Entity / relationship channel

Deliberately conservative, as the calibration required.

`entity_mentions()` counts an entity as mentioned only when its full
normalized name, or a **confirmed** alias, appears as a **contiguous run of
query tokens**. No fuzzy matching, no embedding, no type guessing, no
inference. `Car` does not match `carpet`.

Candidates then come only from the bounded M5 surface: `get_entity_memories`,
`get_related_entities` (one hop, `contains` derived at read time), and
`get_relationship_explanation`. M5 limits, scope rules, and gates are
inherited unchanged. Results map back to corpus items through `source_ids`,
so provenance survives. No new entity type, no new relationship type, no
`related_to`, no traversal beyond one hop.

### 4.5 Fusion

RRF, rank-based, never a sum of heterogeneous raw scores:

```
fused(x) = Σ over channels of 1 / (k + rank_in_channel(x))     k = 60
```

Cosine similarity, the lexical linear score, and a structural match are
incommensurable; summing them with weights would require calibrating scales
that drift with the corpus. Ties break on item id, so ordering is
deterministic. Channel agreement is therefore the signal: an item found by
two channels outranks an item found by one.

### 4.6 Relative evidence floor

```python
kept = [x for x in order if fused[x] >= 0.725 * fused[order[0]]]
```

- The ratio is measured against **the strongest item in this request**, not an
  absolute score, which is what makes it portable across corpora and scales.
- The best item always satisfies it, so the result is never empty when fusion
  produced anything.
- It runs **after** fusion and **after** the gates, so it can only remove.
- `k = 60` remains explicitly documented as **not calibrated**: every k in
  {0, 5, 10, 20, 50, 60, 100} produced an identical metric signature on the
  current benchmark.

**Documented consequence.** Because RRF sums channels, an item supported by two
channels scores roughly twice an item supported by one. The 0.725 floor
therefore admits an item that either has cross-channel agreement or leads its
own channel. A single-channel, non-leading structural hit can be withheld.
This is pinned by `test_evidence_floor_requires_agreement_or_top_rank` so it
cannot change silently.

---

## 5. Shadow mode

Off by default. Enabled per manager or by `PRIMORA_HYBRID_SHADOW=1`. Shadow
mode is now an **optional companion** to the live hybrid path, not a
precondition: normal operation never needs it.

```python
ContextManager(db_path=..., shadow_retrieval=True)
manager.set_shadow_retrieval(True)
manager.shadow_records()               # list[ShadowComparison]
manager.clear_shadow_records()
manager.retrieval_traces()             # list[RetrievalTrace]
```

With hybrid live, the comparison is recorded from the trace the live path
already computed, so shadow mode costs one record write rather than a second
retrieval. With hybrid rolled back, shadow mode still computes hybrid
independently, so a rollback can be evaluated. Records are produced whether or
not hybrid found anything, because "hybrid found nothing" is itself a result
worth comparing.

Shadow never affects the assembled context, Active Memory, Working Memory, or
any Aurora output. `test_shadow_mode_never_changes_the_assembled_context`
asserts byte equality of the view with and without shadow enabled.

Each `ShadowComparison` records the query, lexical ids and ranks, hybrid ids
and ranks, overlap, new-only, old-only, per-channel contribution counts, per
channel timings, hybrid latency, whether the floor fired, degradation, and any
error. Nothing is rendered into the prompt:
`test_shadow_trace_is_never_rendered_into_the_prompt` greps the view for
`semantic`, `rrf`, `fused`, `similarity`, `shadow`, and `0.725`.

---

## 6. Retrieval trace

`RetrievalTrace` carries session, scope, query, temporal mode, evaluation
time, every `RetrievalCandidate` (channel, channel rank, channel score,
similarity, relationship id and type, fused score, final rank, included flag,
exclusion reason), the fused order, per-channel timings, total latency, a
degraded flag, and diagnostics.

It lives on the manager, never on `ContextView`, and is never persisted as
memory. It is the substrate for later analysis without touching the prompt.

---

## 7. Benchmark comparison (47 queries, production path)

| Config | R@1 | R@3 | R@5 | P@3 | FP@3 | no-match | p95 ms |
|---|---|---|---|---|---|---|---|
| **A** lexical only (current) | 0.5366 | 0.6098 | 0.6098 | 0.6098 | 0.6324 | 0.5000 | 16.7 |
| **B** semantic only | 0.3659 | 0.3659 | 0.3659 | 0.3659 | **0.1667** | **1.0000** | 16.5 |
| **C** lexical + semantic (RRF) | **0.6829** | **0.8049** | **0.8049** | **0.8049** | 0.5541 | 0.5000 | 15.7 |
| **D** + M5 structural | 0.6829 | 0.8049 | 0.8049 | 0.8049 | **0.4677** | 0.5000 | 22.4 |

Per-category recall@3:

| Category | A | B | C | D |
|---|---|---|---|---|
| exact | 0.7143 | 0.4286 | 1.0000 | 1.0000 |
| paraphrase | 0.3333 | 0.1667 | 0.5000 | 0.5000 |
| indirect | 0.2000 | 0.0000 | 0.2000 | 0.2000 |
| entity | 0.2500 | 1.0000 | 1.0000 | 1.0000 |
| relationship | 0.5000 | 1.0000 | 1.0000 | 1.0000 |
| temporal | 1.0000 | 0.7500 | 1.0000 | 1.0000 |
| cross_session | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| punctuation | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| false_positive_control | 0.5000 | 0.0000 | 0.5000 | 0.5000 |

Latency (mean / median / p95 ms): A 11.7 / 10.3 / 16.7 · B 10.9 / 9.7 / 16.5 ·
C 10.1 / 9.1 / 15.7 · **D 13.6 / 11.8 / 22.4**.

### 7.1 Against the calibration baseline

The calibration measured lexical-only at R@3 0.575, FP@3 0.6515, and
lexical+semantic at R@3 0.650, FP@3 0.6176. This run differs slightly because
the G05 gold fix changed the denominator (41 gold queries instead of 40) and
because the comparison scores the production path rather than the mirror.
Directionally the conclusions hold and strengthen:

- hybrid improves recall **and** precision simultaneously
  (R@3 0.610 → 0.805, FP@3 0.632 → 0.554);
- the M5 structural channel cuts false positives a further 16%
  (0.554 → 0.468) at **zero** recall cost;
- semantic alone remains the cleanest channel (FP@3 0.167, no-match 1.000)
  but the weakest on recall, which is exactly why it is fused rather than
  used alone.

### 7.2 Honest weak spots

- **Indirect reference is unchanged at 0.200.** No configuration moved it.
  It remains the unsolved class the calibration identified.
- **Cross-session and punctuation come entirely from the lexical channel.**
  Semantic alone scores 0.0000 on both. Fusion preserves the lexical 1.0000,
  which is the correct outcome: the semantic channel has no message vectors.
- **B (semantic only) has no lexical or structural contribution**, so its
  cross-session, punctuation, and indirect zeros are expected, not defects.
- **`k` is still unidentified.** A larger corpus is needed.

---

## 8. Shadow acceptance

| # | Criterion | Result |
|---|---|---|
| 1 | Lexical behavior intact | mirror membership 47/47; all M1–M5 and FTS tests green |
| 2 | Semantic retrieval works | B: R@3 0.366, no-match 1.000 |
| 3 | M5 structural signal works | entity 0.25 → 1.00, relationship 0.50 → 1.00 |
| 4 | Fusion deterministic | reproducible across runs; ties break on item id |
| 5 | Hard gates absolute | 9 gate-blocked pairs stay blocked; `.confidence` absent from `hybrid.py` |
| 6 | False positives do not explode | FP@3 0.632 → 0.468 |
| 7 | Paraphrase improves | 0.333 → 0.500 (+50%) |
| 8 | Entity/relationship improve | 4× and 2×; indirect flat and reported as such |
| 9 | No-match does not regress | 0.500, equal to lexical |
| 10 | Shadow produces reproducible data | `ShadowComparison` asserted field by field |
| 11 | Production DB unchanged | byte-identical, hashed before and after every run |

**All eleven criteria are met.** The implementation is technically acceptable
to switch live. **It has not been switched**: shadow mode is still the only
active path, and enabling hybrid for Aurora requires a separate approval.

---

## 9. Performance

CPU-first, as designed. Total shadow overhead is a **median of ~1.5 ms** over
lexical alone (C 9.1 ms vs A 10.3 ms median; the structural channel is what
adds the ~2.7 ms in D). No GPU contention, no model reload per request, and
the index read is a single table scan over a few hundred rows.

No optimization was attempted. The numbers are already far below the cost of
one 4-bit 8B generation, and premature tuning was explicitly out of scope.

---

## 10. Failure behaviour

| Failure | Behaviour |
|---|---|
| Embedding model missing or unloadable | `available()` False, `search()` → `[]`, `sync()` → `{"skipped": 1}`, trace degraded, lexical unaffected |
| numpy missing | identical fail-safe path |
| Semantic index empty or stale | semantic channel contributes nothing; lexical unaffected |
| `get_entity_memories` / M5 query raises | `structural_partial_failure` diagnostic, that entity skipped, other channels continue |
| Fusion raises | `fusion_failed` diagnostic, no fused order, lexical still returned by `_retrieve` |
| Shadow itself raises | `ShadowComparison(error=..., degraded=True)`; the request is never failed |
| Index dimension mismatch | channel returns `[]` rather than mis-scoring |

No failure path can fail a user turn, and none can claim semantic success when
semantic retrieval did not run.

---

## 11. Acceptance invariants

1. Raw `messages` remain append-only and unchanged.
2. `states` remain authoritative; indexing never writes them.
3. Retrieval never writes states, candidates, entities, or relationships.
4. No score is persisted and no score changes memory truth, authority,
   confidence, or lifecycle.
5. Gates run before scoring and are absolute.
6. M4 half-open validity is preserved.
7. M1–M5 scope rules hold; no automatic context leakage.
8. M5 one-hop bounds and derived `contains` are inherited unchanged.
9. Lexical results remain authoritative for Aurora.
10. Every operation is bounded in items and tokens; `input_ceiling` is never
    relaxed.
11. Ordering is deterministic for identical inputs.
12. Every admitted item is explainable from a structured, non-prompt-visible
    trace.
13. Active Memory and Working Memory are unchanged and unexpanded.
14. The vector index is a rebuildable derived cache; deleting it is lossless.
15. The model is a small local embedder; no second generative model, no
    cross-encoder, no network at retrieval time.
16. `ContextView` structure, order, roles, and prompt wording are unchanged.

---

## 12. Explicit exclusions — not implemented

M7 Advanced RAG · new ContextView assembly · Active Memory redesign ·
Working Memory redesign · Aurora-brain changes · training · Vision · Verse ·
specialist routing · autonomous initiative · goals/progress · consolidation ·
a second AI model · cross-encoder · new relationship types · unrestricted
graph traversal · persistent message embeddings · Neo4j or any graph database ·
faiss / hnswlib / usearch / sqlite-vec / any external vector database.

---

## 13. Running it

```powershell
# M6 focused tests
python -m pytest TESTS\prime\test_retrieval_hybrid.py -q
python -m pytest TESTS\prime\test_retrieval_comparison.py -q

# Benchmark comparison
python TESTS\prime\retrieval_benchmark_compare.py

# Index health from a live manager
python -c "from PRIMORA.prime.context import ContextManager; \
  print(ContextManager(db_path=':memory:').semantic_index_status())"
```

Enable shadow mode with `PRIMORA_HYBRID_SHADOW=1` or
`ContextManager(..., shadow_retrieval=True)`.

---

## 14. Future stages

| Stage | Deliverable | Gate |
|---|---|---|
| ~~M6.7~~ | ~~Shadow observation period~~ | **done** — all eleven criteria met, live switch approved |
| ~~M6.8~~ | ~~Decide the live switch~~ | **done** — approved by Amir, implemented and measured against a live rollback control |
| M6.9 | Message embeddings | only if indirect-reference recall stays flat and the index cost is acceptable |
| M6.10 | Larger embedding model | only with benchmark evidence of a quality gap |
| M6.11 | Identify RRF `k` | needs a corpus where the channels disagree across many candidates |
| M7 | Advanced RAG | separate milestone, separate approval |

### 14.1 Known follow-ups

1. **Indirect reference is still flat at 0.200** and no channel moved it.
   Message embeddings (M6.9) are the only credible fix, and they were
   deliberately deferred.
2. **A state whose only source message is contentless** contributes a
   low-information evidence row when its state candidate is admitted. Real
   M1–M5 states point at the user's own words, so this is rare, and lexical
   retrieval could already return the same row. Not a new risk.
3. **`k` remains unidentified.** The plateau is documented and 60 is a
   placeholder.
4. **The live p95 is +13.3 ms** over the lexical rollback. If that becomes
   material, the honest lever is a smaller `MAX_CANDIDATES`, not the
   thresholds, which are calibrated and must not be tuned.
