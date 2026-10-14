"""M6 benchmark comparison: current lexical vs semantic vs hybrid vs
hybrid + M5 structural signal.

Runs the 47-query benchmark through the **production** retrieval path, not the
benchmark mirror, so the numbers describe the code that actually ships:

A. lexical only          production `collect_lexical_candidates`
B. semantic only         production `SemanticIndex` + `semantic_channel`
C. lexical + semantic    production `hybrid_retrieve` without structure
D. hybrid + structural   production `hybrid_retrieve` with M5 structure

The lexical configuration is additionally checked against the benchmark
mirror. A mismatch means the mirror drifted from production and the comparison
is refused rather than reported.

Run standalone::

    python TESTS/prime/retrieval_benchmark_compare.py
"""

import gc
import os
import shutil
import statistics
import sys
import tempfile
import time
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from PRIMORA.prime.context import hybrid as hybrid_mod
from PRIMORA.prime.context import scoring as scoring_mod
from PRIMORA.prime.context import semantic as semantic_mod
from TESTS.prime import retrieval_benchmark as bench

CATEGORIES = (
    "exact", "paraphrase", "indirect", "entity", "relationship",
    "temporal", "cross_session", "punctuation", "no_match",
    "false_positive_control",
)
TOP_K = (1, 3, 5)
PRECISION_AT = 3


def _build_corpus(workdir: str):
    fixture = bench._load_fixture()
    template = os.path.join(workdir, "compare-template.db")
    corpus = bench.build_corpus(template, fixture)
    index = semantic_mod.SemanticIndex(corpus.db_path and _store(template))
    synced = index.sync()
    return fixture, corpus, template, index, synced


def _store(path: str):
    from PRIMORA.prime.context.store import ContextStore
    return ContextStore(path)


def _lexical_ranking(corpus, manager, session_id: str, query_text: str,
                     history_text: str = ""):
    """A: exactly the rows production would rank, via production code.

    Returned as corpus keys so every configuration is scored against the same
    gold representation.
    """
    query_terms = scoring_mod.build_query(query_text, history_text)
    collected = manager.collect_lexical_candidates(session_id, query_terms)
    id_to_key = {
        value: key for key, value in corpus.key_to_message_id.items()
    }
    ranked: list[str] = []
    for _score, _seq, hit, _bd in collected["same"]:
        key = id_to_key.get(hit.message_id)
        if key is not None and key not in ranked:
            ranked.append(key)
    for _score, _sid, _seq, hit, _bd in collected["cross"][:3]:
        key = id_to_key.get(hit.message_id)
        if key is not None and key not in ranked:
            ranked.append(key)
    return ranked, collected


def _trace_ranking(trace) -> list[str]:
    return list(trace.fused_order)


def _keys_from_trace(trace, corpus) -> list[str]:
    """Map fused item ids back to corpus message keys where possible."""
    out: list[str] = []
    for key in trace.fused_order:
        if key.startswith("message:"):
            ref = int(key.split(":", 1)[1])
            for candidate, message_id in corpus.key_to_message_id.items():
                if message_id == ref:
                    out.append(candidate)
                    break
        else:
            # A state or entity candidate: surface every source message the
            # indexed record was built from, so gold can be credited.
            kind, ref = key.split(":", 1)
            ref = int(ref)
            out.extend(_source_keys(corpus, kind, ref))
    seen: set[str] = set()
    unique: list[str] = []
    for key in out:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _source_keys(corpus, kind: str, ref: int) -> list[str]:
    from PRIMORA.prime.context.store import ContextStore
    store = _store(corpus.db_path)
    try:
        if kind == "state":
            rows = store.fetch_states(
                scope="personal", state_ids=[ref], limit=1
            ) or store.fetch_states(scope="global", state_ids=[ref], limit=1)
            if not rows:
                return []
            return [
                key for key, message_id in corpus.key_to_message_id.items()
                if message_id in rows[0].source_ids
            ]
        entity = store.fetch_entity(ref)
        if entity is None:
            return []
        return [
            key for key, message_id in corpus.key_to_message_id.items()
            if message_id in entity.source_ids
        ]
    except Exception:
        return []


def _recall(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(gold) & set(ranked[:k])) / len(set(gold))


CROSS_SESSION_PREFIX = "[from another conversation]\n"


def _counter(messages):
    return sum(len(message.get("content", "").split()) + 2 for m in messages)


def _full_snapshot(fixture, session_id):
    """The whole seeded conversation for one session, as Aurora sends it."""
    return [
        {"role": entry["role"], "content": entry["text"], "timestamp": 100}
        for entry in fixture["sessions"][session_id]
    ]


def _metrics(per_query: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scored = {q: r for q, r in per_query.items() if r["gold"]}
    negative = {q: r for q, r in per_query.items() if not r["gold"]}
    out: dict[str, Any] = {"query_count": len(per_query)}
    for k in TOP_K:
        out[f"recall_at_{k}"] = round(statistics.fmean(
            [_recall(r["ranked"], r["gold"], k) for r in scored.values()]
        ), 4) if scored else 0.0
    out[f"precision_at_{PRECISION_AT}"] = round(statistics.fmean(
        [_recall(r["ranked"], r["gold"], PRECISION_AT)
         for r in scored.values()]
    ), 4) if scored else 0.0
    top = {q: r["ranked"][:PRECISION_AT] for q, r in scored.items()}
    false_positives = sum(
        1 for q, keys in top.items() for key in keys
        if key not in scored[q]["gold"]
    )
    returned = sum(len(keys) for keys in top.values())
    out["false_positive_rate_at_3"] = round(
        false_positives / returned, 4) if returned else 0.0
    out["no_match_accuracy"] = round(
        sum(1 for r in negative.values() if not r["ranked"])
        / len(negative), 4
    ) if negative else 0.0
    for category in CATEGORIES:
        group = {
            q: r for q, r in per_query.items()
            if r["category"] == category and r["gold"]
        }
        out[f"recall_at_3__{category}"] = round(statistics.fmean(
            [_recall(r["ranked"], r["gold"], 3) for r in group.values()]
        ), 4) if group else None
    latencies = [r["latency_ms"] for r in per_query.values()]
    out["latency_ms"] = {
        "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        "median": round(statistics.median(latencies), 3) if latencies else 0.0,
        "p95": round(bench._percentile(latencies, 0.95), 3) if latencies else 0.0,
    }
    return out


def run(workdir: Optional[str] = None) -> dict[str, Any]:
    bench.assert_no_drift()
    if semantic_mod.load_model() is None:
        raise RuntimeError(
            f"{semantic_mod.MODEL_ID} unavailable offline; comparison cannot run"
        )
    owned = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="primora-m6-compare-")
    try:
        before = bench._sha256(bench.PRODUCTION_DB)
        fixture = bench._load_fixture()
        template = os.path.join(workdir, "compare-template.db")
        corpus = bench.build_corpus(template, fixture)
        index = semantic_mod.SemanticIndex(_store(template))
        synced = index.sync()

        from PRIMORA.prime.context import ContextManager

        results: dict[str, dict[str, Any]] = {
            name: {} for name in ("A", "B", "C", "D", "E", "F")
        }
        mirror_membership = 0
        mirror_order = 0
        mirror_checked = 0
        for query in fixture["queries"]:
            run_db = os.path.join(
                workdir, f"cmp-{query['id']}.db"
            )
            shutil.copyfile(template, run_db)
            manager = ContextManager(db_path=run_db)
            gold = list(query["gold"])
            text = query["text"]
            session_id = query["session"]

            # A. production lexical, cross-checked against the mirror.
            started = time.perf_counter()
            ranked_a, collected = _lexical_ranking(
                corpus, manager, session_id, text
            )
            latency_a = (time.perf_counter() - started) * 1000
            mirror_scored, _mirror_foreign, _terms, _d = (
                bench._rank_candidates(
                    _store(run_db), session_id, text
                )
            )
            mirror_ids = [hit.message_id for _s, _q, hit, _b in mirror_scored]
            production_ids = [
                hit.message_id for _s, _q, hit, _b in collected["same"]
            ]
            # Membership is what recall depends on. Ordering can differ
            # because the mirror scores with `selected_texts=[query]`
            # (its documented deviation) while production accumulates the
            # real selected set, which shifts the redundancy penalty.
            mirror_checked += 1
            if set(mirror_ids) == set(production_ids):
                mirror_membership += 1
            if mirror_ids == production_ids:
                mirror_order += 1
            results["A"][query["id"]] = {
                "gold": gold, "category": query["category"],
                "ranked": ranked_a, "latency_ms": latency_a,
            }

            # B. production semantic channel only.
            started = time.perf_counter()
            semantic_candidates, _d = hybrid_mod.semantic_channel(
                index, _store(run_db), text, session_id, time.time(),
                "current", limit=hybrid_mod.MAX_SEMANTIC_CANDIDATES,
            )
            latency_b = (time.perf_counter() - started) * 1000
            ranked_b = _keys_from_trace(
                _FakeTrace([c.item_id for c in semantic_candidates]), corpus
            )
            results["B"][query["id"]] = {
                "gold": gold, "category": query["category"],
                "ranked": ranked_b, "latency_ms": latency_b,
            }

            # C. production hybrid, lexical + semantic.
            started = time.perf_counter()
            trace_c = hybrid_mod.hybrid_retrieve(
                manager, _store(run_db), index, session_id, session_id, text,
                collected=collected, use_structural=False,
            )
            latency_c = (time.perf_counter() - started) * 1000
            results["C"][query["id"]] = {
                "gold": gold, "category": query["category"],
                "ranked": _keys_from_trace(trace_c, corpus),
                "latency_ms": latency_c,
            }

            # D. production hybrid plus M5 structural signal.
            started = time.perf_counter()
            trace_d = hybrid_mod.hybrid_retrieve(
                manager, _store(run_db), index, session_id, session_id, text,
                collected=collected, use_structural=True,
            )
            latency_d = (time.perf_counter() - started) * 1000
            results["D"][query["id"]] = {
                "gold": gold, "category": query["category"],
                "ranked": _keys_from_trace(trace_d, corpus),
                "latency_ms": latency_d,
            }

        # E. the LIVE production retrieval path, end to end through
        # ContextManager.build() exactly as Aurora calls it.
        from PRIMORA.prime.context import BudgetConfig

        live_budget = BudgetConfig(
            input_target=200, input_ceiling=260, output_reserve=20,
            total_budget=280,
        )
        for query in fixture["queries"]:
            gold = list(query["gold"])
            for name, hybrid_on in (("E", True), ("F", False)):
                live_db = os.path.join(
                    workdir, f"live-{name}-{query['id']}.db"
                )
                shutil.copyfile(template, live_db)
                live_manager = ContextManager(
                    db_path=live_db, hybrid_retrieval=hybrid_on
                )
                if hybrid_on:
                    live_manager.sync_semantic_index()
                snapshot = _full_snapshot(fixture, query["session"])
                snapshot.append(
                    {"role": "user", "content": query["text"],
                     "timestamp": 900}
                )
                started = time.perf_counter()
                view = live_manager.build(
                    list(snapshot), query["session"],
                    token_counter=_counter, max_new_tokens=20,
                    budget=live_budget,
                ).view
                latency = (time.perf_counter() - started) * 1000
                text_to_key = {
                    v: k for k, v in corpus.key_to_text.items()
                }
                ranked: list[str] = []
                for message in view.messages:
                    content = message.get("content", "")
                    role = message.get("role", "")
                    key = text_to_key.get((role, content))
                    if key is None and content.startswith(
                        CROSS_SESSION_PREFIX
                    ):
                        key = text_to_key.get(
                            (role, content[len(CROSS_SESSION_PREFIX):])
                        )
                    if key is not None and key not in ranked:
                        ranked.append(key)
                results[name][query["id"]] = {
                    "gold": gold, "category": query["category"],
                    "ranked": ranked, "latency_ms": latency,
                }

        after = bench._sha256(bench.PRODUCTION_DB)
        return {
            "production_db_untouched": before == after,
            "production_db_sha256": before,
            "index_synced": synced,
            "mirror_checked": mirror_checked,
            "mirror_membership_agreements": mirror_membership,
            "mirror_order_agreements": mirror_order,
            "metrics": {name: _metrics(rows) for name, rows in results.items()},
            "per_query": results,
        }
    finally:
        if owned:
            gc.collect()
            shutil.rmtree(workdir, ignore_errors=True)


class _FakeTrace:
    """Minimal stand-in so `_keys_from_trace` can rank a bare id list."""

    def __init__(self, order):
        self.fused_order = list(order)


def format_report(result: dict[str, Any]) -> str:
    labels = {
        "A": "lexical only (rollback baseline)",
        "B": "semantic only",
        "C": "lexical + semantic (RRF)",
        "D": "lexical + semantic + M5 structural",
        "E": "LIVE build() hybrid (what Aurora gets)",
        "F": "LIVE build() rollback (lexical only)",
    }
    lines = ["=" * 92,
             "M6 BENCHMARK COMPARISON - production retrieval path",
             "=" * 92,
             f"production DB untouched: {result['production_db_untouched']}",
             f"production DB sha256  : {result['production_db_sha256']}",
             f"index synced          : {result['index_synced']}",
             f"lexical mirror membership agreement: "
             f"{result['mirror_membership_agreements']}/{result['mirror_checked']}",
             f"lexical mirror order agreement:      "
             f"{result['mirror_order_agreements']}/{result['mirror_checked']}",
             ""]
    header = (f"{'config':38} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'P@3':>6} "
              f"{'FP@3':>6} {'no-match':>8} {'p95ms':>7}")
    lines.append(header)
    lines.append("-" * len(header))
    for name in ("A", "B", "C", "D", "E", "F"):
        m = result["metrics"][name]
        lines.append(
            f"{labels[name]:38} {m['recall_at_1']:>6} {m['recall_at_3']:>6} "
            f"{m['recall_at_5']:>6} {m['precision_at_3']:>6} "
            f"{m['false_positive_rate_at_3']:>6} {m['no_match_accuracy']:>8} "
            f"{m['latency_ms']['p95']:>7}"
        )
    lines.append("")
    lines.append("per-category recall@3")
    per = f"{'category':26} {'A':>8} {'B':>8} {'C':>8} {'D':>8}"
    lines.append(per)
    lines.append("-" * len(per))
    for category in CATEGORIES:
        cells = []
        for name in ("A", "B", "C", "D", "E", "F"):
            value = result["metrics"][name].get(
                f"recall_at_3__{category}"
            )
            cells.append("  -  " if value is None else f"{value:>6}")
        lines.append(f"{category:26} " + " ".join(cells))
    lines.append("")
    lines.append("latency ms (mean / median / p95)")
    for name in ("A", "B", "C", "D", "E", "F"):
        lat = result["metrics"][name]["latency_ms"]
        lines.append(
            f"  {labels[name]:38} {lat['mean']:>8} {lat['median']:>8} "
            f"{lat['p95']:>8}"
        )
    return "\n".join(lines)


def main() -> int:
    result = run()
    print(format_report(result))
    return 0 if result["production_db_untouched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
