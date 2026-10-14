# Retrieval Benchmark (M6 preparation, stage M6.1)

Measures the **current lexical retrieval system** so that later M6 stages can
be calibrated against real numbers instead of assumptions.

This directory implements **only** two things:

1. the FTS5 query-construction defect fix (in `PRIMORA/prime/context/store.py`);
2. this benchmark.

It deliberately contains **no** semantic retrieval, no embeddings, no vector
index, no RRF fusion, no shadow retrieval, and no RAG.

## Files

| File | Role |
|---|---|
| `TESTS/prime/retrieval_benchmark.py` | The benchmark. Builds its corpus, runs every query, computes metrics, prints a report. |
| `TESTS/prime/fixtures/retrieval_benchmark_corpus.json` | The controlled corpus and the gold set. Ground truth is declared here. |
| `TESTS/prime/test_retrieval_benchmark.py` | Operational tests: production safety, fixture integrity, determinism, trace shape. |
| `TESTS/prime/test_retrieval_fts_terms.py` | Focused regression tests for the FTS5 query-construction fix. |
| `PRIMORA/prime/context/store.py` | `ContextStore._fts_phrase()` and the FTS branch of `search()`. The only production change. |

## Running it

```powershell
# Human-readable report plus a per-query trace
python TESTS\prime\retrieval_benchmark.py

# Both preparation test suites
python -m pytest TESTS\prime\test_retrieval_fts_terms.py -q
python -m pytest TESTS\prime\test_retrieval_benchmark.py -q

# Everything
python -m pytest TESTS -q --ignore=TESTS\specialists\test_calculate.py `
                           --ignore=TESTS\aurora\test_brain_integration.py
```

Exit code is `0` only when the production database hash is unchanged after the
run.

## Two run modes

The two modes answer two different questions, so both are measured for every
query.

### `channel` — lexical channel mirror (ranked metrics)

A read-only mirror of `ContextManager._retrieve`: the same
`ContextStore.search` calls, the same pure `scoring.score`, the same query-term
cap, per-term limits, superseded-source suppression, and cross-session cap.
**This is the mode that exposes true retrieval rank**, so Recall@k, Precision@k
and the false-positive rates come from here.

`assert_no_drift()` runs before every benchmark and fails loudly if the
ranking weights in `scoring.score` or the query limits in `manager.py` change,
so the mirror cannot silently diverge from production. The guard reads
`manager.py` as text because the limits are inline literals; no production
module was modified to make it testable.

### `context` — end-to-end `build()` (membership metrics)

The real path: the whole seeded conversation is passed to
`ContextManager.build()` and the metric is whether each gold memory survives
into the assembled view.

**Retrieval rank is not observable in this mode**, and the report says so.
`ContextView` deliberately re-emits retrieved material in chronological order,
so position in the view is not retrieval rank. For the same reason
`no_match_accuracy` and `Precision@k` are not reported here: the recency
window always contributes messages, which would swamp both.

## Ground truth

Gold is declared per query in the fixture as a list of corpus keys, and the
loader asserts every key resolves to exactly one stored message. One, many, and
none are all supported, and there is at least one of each. Nothing is inferred
from the system's own ranking: the fixture has no field for what the system
returned, and a test enforces that.

Categories: `exact`, `paraphrase`, `indirect`, `entity`, `relationship`,
`temporal`, `no_match`, `cross_session`, `false_positive_control`,
`punctuation`. A distractor set is declared so unrelated material competes for
the budget.

## Metrics

Ranked (`channel`): `recall_at_1`, `recall_at_3`, `recall_at_5`,
`recall_at_all`, `precision_at_3`, `false_positive_rate_at_3`,
`false_positive_rate_all`, `no_match_accuracy`, per-category `recall_at_3`,
`punctuation_success`, `punctuation_success_at_3`, same-session dominance
counters, `query_count`, `relevant_result_count`, `failed_count`, and
`latency_ms` with mean / median / p95 / min / max.

Membership (`context`): `coverage`, `coverage_at_1`, per-category `coverage`,
`mean_input_tokens`, and the same counts and latency statistics.

## Production safety

- The corpus is built into a temporary directory and copied per run. Nothing is
  written to `PRIMORA/data`.
- The production database is hashed before and after every run; the report and
  the tests both fail if it changed.
- A test asserts the production data directory is byte-for-byte unchanged after
  a benchmark run.
- The benchmark injects nothing into any prompt. It only reads the assembled
  view, and the trace is kept out of the view entirely.

## Baseline results

Recorded, not gated. `test_baseline_summary_is_reported` prints them; it does
not assert fixed values, because M6 is expected to improve them. The single
gated quality invariant is `skipped_term_total == 0`, which is the FTS fix.

See `DOCS/01_architecture/MEMORY_HYBRID_RETRIEVAL.md` §18 for the full metric
definitions and the target values.
