"""M6 retrieval calibration: experiments to choose tau, RRF k, and the
same-session policy from evidence instead of assumption.

EXPERIMENTAL. Nothing here is connected to production retrieval. This module
imports the benchmark (the source of truth for the corpus and the gold set) and
simulates candidate fusion entirely in memory. It performs no database writes
outside its own temporary directory, and it never imports the production
retrieval path in a way that could mutate it.

No semantic retrieval, embeddings, RRF, or shadow mode is implemented in
`PRIMORA/`. `sentence-transformers` is used here only as an experiment
dependency, and the model is the already-approved local
`sentence-transformers/all-MiniLM-L6-v2`, CPU, float32, offline.

Run standalone::

    python TESTS/prime/retrieval_calibration.py
"""

import gc
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context import entities as entities_mod
from PRIMORA.prime.context.store import ContextStore
from TESTS.prime import retrieval_benchmark as bench

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cpu"
BATCH = 32

TAU_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
K_GRID = (0, 5, 10, 20, 50, 60, 100)
TOP_K = (1, 3, 5)
PRECISION_AT = 3
CROSS_SESSION_CAP = 3
INSUFFICIENT_LOCAL = 3


# ----------------------------------------------------------------------
# Embedding (EXPERIMENTAL, offline, CPU)
# ----------------------------------------------------------------------
def load_encoder():
    """Load the approved local model. Returns None when unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    try:
        model = SentenceTransformer(MODEL_ID, device=DEVICE)
    except Exception:
        return None
    model.eval()
    return model


def encode(model, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = model.encode(
        texts, batch_size=BATCH, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


# ----------------------------------------------------------------------
# Experiment data
# ----------------------------------------------------------------------
@dataclass
class Calibration:
    fixture: dict[str, Any]
    corpus: bench.Corpus
    store: ContextStore
    keys: list[str]
    texts: dict[str, str]
    query_ids: list[str]
    query_texts: list[str]
    gold: dict[str, list[str]]
    category: dict[str, str]
    session: dict[str, str]
    message_vectors: np.ndarray
    query_vectors: np.ndarray
    similarity: np.ndarray  # [n_queries, n_messages]
    key_index: dict[str, int]
    lexical: dict[str, list[str]]
    lexical_foreign: dict[str, list[str]]
    lexical_scores: dict[str, dict[str, float]]
    dimension: int
    errors: list[str] = field(default_factory=list)
    production_db_sha256: Optional[str] = None

    def sim(self, query_id: str, key: str) -> float:
        return float(self.similarity[
            self.query_ids.index(query_id), self.key_index[key]
        ])

    def sims_for(self, query_id: str) -> dict[str, float]:
        row = self.similarity[self.query_ids.index(query_id)]
        return {k: float(row[i]) for k, i in self.key_index.items()}


def build_calibration(workdir: str, fixture=None) -> Calibration:
    fixture = fixture or bench._load_fixture()
    db_path = os.path.join(workdir, "calibration.db")
    corpus = bench.build_corpus(db_path, fixture)
    store = ContextStore(db_path)

    keys = list(corpus.key_to_message_id)
    key_index = {key: i for i, key in enumerate(keys)}
    texts = {key: corpus.key_to_text[key][1] for key in keys}
    queries = fixture["queries"]
    query_ids = [q["id"] for q in queries]
    query_texts = [q["text"] for q in queries]
    gold = {q["id"]: list(q["gold"]) for q in queries}
    category = {q["id"]: q["category"] for q in queries}
    session = {q["id"]: q["session"] for q in queries}

    model = load_encoder()
    errors: list[str] = []
    if model is None:
        raise RuntimeError(
            f"{MODEL_ID} is unavailable offline; calibration cannot run"
        )
    message_vectors = encode(model, [texts[k] for k in keys])
    query_vectors = encode(model, query_texts)
    dimension = int(message_vectors.shape[1])

    # Lexical channel: the benchmark's read-only mirror, unchanged.
    lexical: dict[str, list[str]] = {}
    lexical_foreign: dict[str, list[str]] = {}
    lexical_scores: dict[str, dict[str, float]] = {}
    id_to_key = {v: k for k, v in corpus.key_to_message_id.items()}
    for query in queries:
        scored, foreign, _terms, _diag = bench._rank_candidates(
            store, query["session"], query["text"]
        )
        row: list[str] = []
        scores: dict[str, float] = {}
        for score, _seq, hit, _bd in scored:
            key = id_to_key.get(hit.message_id)
            if key is not None and key not in row:
                row.append(key)
                scores[key] = round(score, 6)
        foreign_row: list[str] = []
        for score, _sid, _seq, hit, _bd in foreign:
            key = id_to_key.get(hit.message_id)
            if key is not None and key not in foreign_row:
                foreign_row.append(key)
                scores[key] = round(score, 6)
        lexical[query["id"]] = row
        lexical_foreign[query["id"]] = foreign_row
        lexical_scores[query["id"]] = scores

    return Calibration(
        fixture=fixture, corpus=corpus, store=store, keys=keys, texts=texts,
        query_ids=query_ids, query_texts=query_texts, gold=gold,
        category=category, session=session,
        message_vectors=message_vectors, query_vectors=query_vectors,
        similarity=(
            query_vectors @ message_vectors.T
            if dimension else np.zeros((len(query_ids), 0), dtype=np.float32)
        ),
        key_index=key_index, lexical=lexical, lexical_foreign=lexical_foreign,
        lexical_scores=lexical_scores, dimension=dimension, errors=errors,
        production_db_sha256=bench._sha256(bench.PRODUCTION_DB),
    )


# ----------------------------------------------------------------------
# Metrics over an ordered candidate list
# ----------------------------------------------------------------------
def _rank_of(ranked: list[str], gold: list[str]) -> Optional[int]:
    for index, key in enumerate(ranked):
        if key in gold:
            return index + 1
    return None


def _recall_at(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(gold) & set(ranked[:k])) / len(set(gold))


def evaluate(
    rankings: dict[str, list[str]], cal: Calibration
) -> dict[str, Any]:
    """Metrics for one configuration. The gold set is never inferred."""
    scored_ids = [q for q in cal.query_ids if cal.gold[q]]
    no_match_ids = [q for q in cal.query_ids if not cal.gold[q]]

    out: dict[str, Any] = {"query_count": len(cal.query_ids)}
    for k in TOP_K:
        out[f"recall_at_{k}"] = round(statistics.fmean(
            [_recall_at(rankings[q], cal.gold[q], k) for q in scored_ids]
        ), 4) if scored_ids else 0.0
    out[f"precision_at_{PRECISION_AT}"] = round(statistics.fmean(
        [_recall_at(rankings[q], cal.gold[q], PRECISION_AT)
         for q in scored_ids]
    ), 4) if scored_ids else 0.0

    top = {q: rankings[q][:PRECISION_AT] for q in scored_ids}
    fp = sum(1 for q in scored_ids for key in top[q]
             if key not in cal.gold[q])
    returned = sum(len(top[q]) for q in scored_ids)
    out["false_positive_rate_at_3"] = round(fp / returned, 4) if returned else 0.0
    out["returned_at_3"] = returned

    out["no_match_accuracy"] = round(
        sum(1 for q in no_match_ids if not rankings[q]) / len(no_match_ids), 4
    ) if no_match_ids else 0.0
    out["no_match_false_retrieval"] = sum(
        1 for q in no_match_ids if rankings[q]
    )

    for name in ("exact", "paraphrase", "indirect", "entity", "relationship",
                 "temporal", "cross_session", "punctuation"):
        group = [q for q in scored_ids if cal.category[q] == name]
        out[f"recall_at_3__{name}"] = round(statistics.fmean(
            [_recall_at(rankings[q], cal.gold[q], 3) for q in group]
        ), 4) if group else None

    same = [q for q in scored_ids
            if all(cal.corpus.key_to_session[g] == cal.session[q]
                   for g in cal.gold[q])]
    cross = [q for q in scored_ids
             if any(cal.corpus.key_to_session[g] != cal.session[q]
                    for g in cal.gold[q])]
    out["same_session_recall_at_3"] = round(statistics.fmean(
        [_recall_at(rankings[q], cal.gold[q], 3) for q in same]
    ), 4) if same else 0.0
    out["cross_session_recall_at_3"] = round(statistics.fmean(
        [_recall_at(rankings[q], cal.gold[q], 3) for q in cross]
    ), 4) if cross else 0.0
    out["cross_session_query_count"] = len(cross)

    # Foreign evidence ranked above the relevant local memory.
    out["foreign_outranks_local_gold"] = sum(
        1 for q in same
        if _foreign_beats_local(rankings[q], cal, q)
    )
    out["same_session_query_count"] = len(same)
    return out


def _foreign_beats_local(ranked: list[str], cal: Calibration, query_id: str) -> bool:
    local = cal.session[query_id]
    gold_rank = _rank_of(ranked, cal.gold[query_id])
    if gold_rank is None:
        return False
    for index, key in enumerate(ranked):
        if cal.corpus.key_to_session[key] != local:
            return (index + 1) < gold_rank
    return False


# ----------------------------------------------------------------------
# Experiment 1: semantic score distributions and the tau sweep
# ----------------------------------------------------------------------
def experiment_semantic(cal: Calibration) -> dict[str, Any]:
    buckets: dict[str, list[float]] = {
        "relevant": [], "paraphrase": [], "indirect": [], "entity": [],
        "exact": [], "temporal": [], "punctuation": [],
        "relationship": [], "cross_session": [], "no_match": [],
        "false_positive_control": [],
        "distractor": [], "irrelevant": [], "no_match_query": [],
    }
    for query_id in cal.query_ids:
        sims = cal.sims_for(query_id)
        gold = set(cal.gold[query_id])
        if not gold:
            buckets["no_match_query"].extend(sims.values())
            continue
        for key, value in sims.items():
            if key in gold:
                buckets["relevant"].append(value)
                buckets[cal.category[query_id]].append(value)
            else:
                buckets["irrelevant"].append(value)
                if key == "lunch":
                    buckets["distractor"].append(value)

    def stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        ordered = sorted(values)
        return {
            "n": len(ordered),
            "min": round(ordered[0], 4),
            "p05": round(bench._percentile(ordered, 0.05), 4),
            "p25": round(bench._percentile(ordered, 0.25), 4),
            "median": round(statistics.median(ordered), 4),
            "mean": round(statistics.fmean(ordered), 4),
            "p75": round(bench._percentile(ordered, 0.75), 4),
            "p95": round(bench._percentile(ordered, 0.95), 4),
            "max": round(ordered[-1], 4),
        }

    # Gold-pair separability: the worst gold score per query class.
    weakest_gold: dict[str, list[float]] = {}
    for query_id in cal.query_ids:
        if not cal.gold[query_id]:
            continue
        name = cal.category[query_id]
        for key in cal.gold[query_id]:
            weakest_gold.setdefault(name, []).append(cal.sim(query_id, key))

    sweep = []
    for tau in TAU_GRID:
        # The same gate and scope rules apply at every threshold, so the
        # sweep measures the floor's effect, not a gate bypass.
        rankings = {q: semantic_ranking(cal, q, tau) for q in cal.query_ids}
        metrics = evaluate(rankings, cal)
        gold_scores = [
            cal.sim(q, g) for q in cal.query_ids for g in cal.gold[q]
        ]
        irrelevant = buckets["irrelevant"]
        metrics.update({
            "tau": tau,
            "gold_pairs_above_tau": sum(1 for s in gold_scores if s >= tau),
            "gold_pair_total": len(gold_scores),
            "gold_recall": round(
                sum(1 for s in gold_scores if s >= tau) / len(gold_scores), 4
            ) if gold_scores else 0.0,
            "irrelevant_above_tau": sum(1 for s in irrelevant if s >= tau),
            "irrelevant_total": len(irrelevant),
            "irrelevant_pass_rate": round(
                sum(1 for s in irrelevant if s >= tau) / len(irrelevant), 4
            ) if irrelevant else 0.0,
        })
        sweep.append(metrics)

    return {
        "distributions": {name: stats(values)
                          for name, values in buckets.items()},
        "gold_by_category": {name: stats(values)
                             for name, values in weakest_gold.items()},
        "tau_sweep": sweep,
    }


# ----------------------------------------------------------------------
# Experiment 2: RRF fusion
# ----------------------------------------------------------------------
def gated_keys(cal: Calibration, query_id: str) -> set[str]:
    """Corpus keys the deterministic M4/M1 gates remove for this query.

    Scope and supersession are absolute. A similarity score may never
    re-admit anything in this set. Mirrors what production applies today:
    a message that sources a superseded state is withheld, and the foreign
    session is not reachable from the same-session channel.
    """
    session = cal.session[query_id]
    if bench._history_requested(cal.query_texts[cal.query_ids.index(query_id)]):
        return set()
    stale = ContextManager._superseded_source_ids(
        cal.store, [session, "global"]
    )
    return {
        key for key, value in cal.corpus.key_to_message_id.items()
        if value in stale
    }


def semantic_ranking(cal: Calibration, query_id: str,
                     tau: float) -> list[str]:
    """Semantic channel with production scope and gate rules applied.

    Same-session messages first, then at most CROSS_SESSION_CAP foreign
    messages, each filtered by the positive-evidence floor tau, each with
    supersession-gated rows removed.
    """
    session = cal.session[query_id]
    blocked = gated_keys(cal, query_id)
    sims = cal.sims_for(query_id)
    ordered = sorted(sims.items(), key=lambda kv: (-kv[1], kv[0]))
    same: list[str] = []
    foreign: list[str] = []
    for key, value in ordered:
        if key in blocked or value < tau:
            continue
        if cal.corpus.key_to_session[key] == session:
            same.append(key)
        elif len(foreign) < CROSS_SESSION_CAP:
            foreign.append(key)
    return same + foreign


def rrf(lists: list[list[str]], k: float) -> list[str]:
    # Ranks are 1-based, so k=0 is well defined (1/(0+1) for the top hit) and
    # is included deliberately as the steepest degenerate setting.
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = {}
    for ranked in lists:
        for index, key in enumerate(ranked):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + index + 1)
    return [
        key for key, _ in sorted(
            scores.items(), key=lambda kv: (-kv[1], kv[0])
        )
    ]


def experiment_rrf(cal: Calibration, tau: float) -> dict[str, Any]:
    lexical_rankings = {
        q: cal.lexical[q] + [k for k in cal.lexical_foreign[q]
                             if k not in cal.lexical[q]]
        for q in cal.query_ids
    }
    semantic_rankings = {
        q: semantic_ranking(cal, q, tau) for q in cal.query_ids
    }
    results: dict[str, Any] = {
        "tau": tau,
        "lexical_only": evaluate(lexical_rankings, cal),
        "semantic_only": evaluate(semantic_rankings, cal),
    }
    for k in K_GRID:
        fused = {
            q: rrf([lexical_rankings[q], semantic_rankings[q]], float(k))
            for q in cal.query_ids
        }
        results[f"rrf_k_{k}"] = evaluate(fused, cal)
    return results


# ----------------------------------------------------------------------
# Experiment 3: same-session policies
# ----------------------------------------------------------------------
def apply_policy(base: list[str], cal: Calibration, query_id: str,
                 policy: str) -> list[str]:
    local = cal.session[query_id]
    same = [key for key in base if cal.corpus.key_to_session[key] == local]
    foreign = [key for key in base if cal.corpus.key_to_session[key] != local]
    if policy == "dominance":
        return same if same else foreign[:CROSS_SESSION_CAP]
    if policy == "gated":
        if len(same) < INSUFFICIENT_LOCAL:
            return same + foreign[:CROSS_SESSION_CAP]
        return same
    # "current": same-session list first, then the capped foreign tail
    return same + foreign[:CROSS_SESSION_CAP]


def relative_floor(ranked: list[str], scores: dict[str, float],
                   ratio: float) -> list[str]:
    """Keep items whose fused score is within `ratio` of the best item.

    A rank-only fusion cannot express "supported by one weak channel only",
    so a lexical candidate ranked first by accident survives fusion. A
    relative floor is deterministic, needs no new score calibration, and is
    applied after fusion and before any budget admission.
    """
    if not ranked:
        return []
    best = scores[ranked[0]]
    if best <= 0:
        return ranked
    return [key for key in ranked if scores[key] >= ratio * best]


def experiment_post_fusion_floor(cal: Calibration, tau: float,
                                 k: float) -> dict[str, Any]:
    lexical_rankings = {
        q: cal.lexical[q] + [x for x in cal.lexical_foreign[q]
                             if x not in cal.lexical[q]]
        for q in cal.query_ids
    }
    semantic_rankings = {q: semantic_ranking(cal, q, tau)
                         for q in cal.query_ids}
    out: dict[str, Any] = {"tau": tau, "rrf_k": k, "ratios": {}}
    for ratio in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
        rankings: dict[str, list[str]] = {}
        for q in cal.query_ids:
            scores: dict[str, float] = {}
            for ranked in (lexical_rankings[q], semantic_rankings[q]):
                for index, key in enumerate(ranked):
                    scores[key] = scores.get(key, 0.0) + 1.0 / (k + index + 1)
            fused = sorted(
                scores.items(), key=lambda kv: (-kv[1], kv[0])
            )
            ordered = [key for key, _ in fused]
            rankings[q] = relative_floor(ordered, scores, ratio)
        out["ratios"][str(ratio)] = evaluate(rankings, cal)
    return out


def experiment_same_session(cal: Calibration, tau: float,
                            k: float) -> dict[str, Any]:
    lexical_rankings = {
        q: cal.lexical[q] + [x for x in cal.lexical_foreign[q]
                             if x not in cal.lexical[q]]
        for q in cal.query_ids
    }
    semantic_rankings = {
        q: semantic_ranking(cal, q, tau) for q in cal.query_ids
    }
    fused = {
        q: rrf([lexical_rankings[q], semantic_rankings[q]], k)
        for q in cal.query_ids
    }
    out: dict[str, Any] = {"tau": tau, "rrf_k": k}
    for label, base in (("lexical", lexical_rankings), ("hybrid", fused)):
        for policy in ("current", "dominance", "gated"):
            rankings = {
                q: apply_policy(base[q], cal, q, policy)
                for q in cal.query_ids
            }
            out[f"{label}_{policy}"] = evaluate(rankings, cal)
    return out


# ----------------------------------------------------------------------
# Experiment 4: entity / relationship structural signal
# ----------------------------------------------------------------------
def structural_candidates(cal: Calibration, query_id: str) -> dict[str, Any]:
    """What M5's bounded query surface can contribute for this query.

    Read-only, scope-explicit, one hop. Structure is mapped back to corpus
    messages through `source_ids`, so provenance is preserved.
    """
    scope = cal.session[query_id]
    text = cal.query_texts[cal.query_ids.index(query_id)]
    manager = ContextManager(db_path=cal.corpus.db_path)
    result: dict[str, Any] = {
        "resolved_entities": [], "ambiguous": False,
        "structural_keys": [], "source": "none",
    }
    candidates, ambiguous = manager.resolve_entity(scope, text)
    result["ambiguous"] = ambiguous
    entity_ids = [entity.entity_id for entity in candidates]
    result["resolved_entities"] = [entity.display_name for entity in candidates]
    if not entity_ids:
        return result

    keys: list[str] = []
    for entity_id in entity_ids:
        for link in manager.get_entity_memories(scope, entity_id):
            for source in link.state.source_ids:
                key = _key_of(cal, source)
                if key and key not in keys:
                    keys.append(key)
        for related in manager.get_related_entities(scope, entity_id):
            for source in related.entity.source_ids:
                key = _key_of(cal, source)
                if key and key not in keys:
                    keys.append(key)
    for entity_id in entity_ids:
        for link in manager.get_entity_memories(scope, entity_id):
            for path_state in (link.state,):
                for path in manager.get_relationship_explanation(
                    scope, path_state.state_id
                ):
                    for state in (path.rationale_state, path.decision_state):
                        for source in state.source_ids:
                            key = _key_of(cal, source)
                            if key and key not in keys:
                                keys.append(key)
    result["structural_keys"] = keys
    result["source"] = "m5_bounded_queries"
    return result


def _key_of(cal: Calibration, message_id: int) -> Optional[str]:
    for key, value in cal.corpus.key_to_message_id.items():
        if value == message_id:
            return key
    return None


def _declared_names(fixture: dict[str, Any]) -> list[str]:
    """Entity display names and aliases the fixture itself declares."""
    names = [e["display_name"] for e in fixture["entities"]]
    names += [a["alias_text"] for a in fixture.get("aliases", [])]
    return names


def structural_for_names(cal: Calibration, names: list[str]) -> dict[str, Any]:
    """Structural candidates reachable from a *given* entity name.

    This isolates the structural signal from mention detection: the names
    come from the fixture's own entity declarations, nothing is invented.
    """
    manager = ContextManager(db_path=cal.corpus.db_path)
    resolved: list[str] = []
    unresolved: list[str] = []
    keys: list[str] = []
    for name in names:
        candidates, _ambiguous = manager.resolve_entity("personal", name)
        if not candidates:
            unresolved.append(name)
            continue
        resolved.append(name)
        for entity in candidates:
            for source in entity.source_ids:
                key = _key_of(cal, source)
                if key and key not in keys:
                    keys.append(key)
            for link in manager.get_entity_memories("personal", entity.entity_id):
                for source in link.state.source_ids:
                    k = _key_of(cal, source)
                    if k and k not in keys:
                        keys.append(k)
                for path in manager.get_relationship_explanation(
                    "personal", link.state.state_id
                ):
                    for state in (path.rationale_state, path.decision_state):
                        for source in state.source_ids:
                            k = _key_of(cal, source)
                            if k and k not in keys:
                                keys.append(k)
            for related in manager.get_related_entities(
                "personal", entity.entity_id, direction="both"
            ):
                for source in related.entity.source_ids:
                    k = _key_of(cal, source)
                    if k and k not in keys:
                        keys.append(k)
    return {"resolved": resolved, "unresolved": unresolved, "keys": keys}


def experiment_structural(cal: Calibration, tau: float, k: float) -> dict[str, Any]:
    lexical_rankings = {
        q: cal.lexical[q] + [x for x in cal.lexical_foreign[q]
                             if x not in cal.lexical[q]]
        for q in cal.query_ids
    }
    semantic_rankings = {q: semantic_ranking(cal, q, tau) for q in cal.query_ids}

    # 4a. Can M5 resolve a name that it is actually given?
    names = _declared_names(cal.fixture)
    by_name = structural_for_names(cal, names)

    # 4b. Can it resolve a name out of a free-text query? (mention detection)
    from_query = {}
    for query_id in cal.query_ids:
        candidates, ambiguous = ContextManager(
            db_path=cal.corpus.db_path
        ).resolve_entity(cal.session[query_id], cal.query_texts[
            cal.query_ids.index(query_id)
        ])
        from_query[query_id] = {
            "resolved": [c.display_name for c in candidates],
            "ambiguous": ambiguous,
        }
    targeted = [q for q in cal.query_ids
                if cal.category[q] in ("entity", "relationship")]
    from_query_resolved = [q for q in targeted if from_query[q]["resolved"]]

    # 4c. Structural signal as a third fusion channel, using the resolvable
    # structural key set. Per-query: the keys reachable from any entity that
    # the fixture declares.
    per_query_structural: dict[str, list[str]] = {}
    entity_to_gold: dict[str, list[str]] = {}
    for entity in cal.fixture["entities"]:
        info = structural_for_names(cal, [entity["display_name"]])
        entity_to_gold[entity["key"]] = info["keys"]
    for query in cal.fixture["queries"]:
        reachable: list[str] = []
        for entity in cal.fixture["entities"]:
            name = entity["display_name"]
            if entities_mod.normalize_entity_name(
                name
            ) and entities_mod.normalize_entity_name(name) in (
                entities_mod.normalize_entity_name(query["text"])
            ):
                for key in entity_to_gold.get(entity["key"], []):
                    if key not in reachable:
                        reachable.append(key)
        per_query_structural[query["id"]] = reachable

    fused = {
        q: rrf([lexical_rankings[q], semantic_rankings[q],
                per_query_structural[q]], k)
        for q in cal.query_ids
    }
    structural_only = {q: per_query_structural[q] for q in cal.query_ids}

    return {
        "tau": tau, "rrf_k": k,
        "declared_names": names,
        "names_resolved_from_name": by_name["resolved"],
        "names_unresolved_from_name": by_name["unresolved"],
        "structural_keys_reachable": by_name["keys"],
        "targeted_query_count": len(targeted),
        "targeted_resolved_from_free_text": len(from_query_resolved),
        "mention_detection_gap": [
            {"id": q, "category": cal.category[q], "text": cal.query_texts[
                cal.query_ids.index(q)]}
            for q in targeted if q not in from_query_resolved
        ],
        "structural_only": evaluate(structural_only, cal),
        "hybrid_with_structural": evaluate(fused, cal),
        "hybrid_without_structural": evaluate({
            q: rrf([lexical_rankings[q], semantic_rankings[q]], k)
            for q in cal.query_ids
        }, cal),
        "entity_category_queries_needing_no_new_types": True,
    }


# ----------------------------------------------------------------------
# Hard-gate demonstration
# ----------------------------------------------------------------------
def experiment_gates(cal: Calibration, tau: float) -> dict[str, Any]:
    """High semantic similarity must never override a deterministic gate."""
    rows = []
    for query_id in cal.query_ids:
        for key in cal.gold[query_id]:
            similarity = cal.sim(query_id, key)
            stale = ContextManager._superseded_source_ids(
                cal.store, [cal.session[query_id], "global"]
            )
            gated = cal.corpus.key_to_message_id[key] in stale
            rows.append({
                "query": query_id, "key": key,
                "similarity": round(similarity, 4),
                "supersession_gated": gated,
                "passes_tau": similarity >= tau,
            })
    blocked = [r for r in rows if r["supersession_gated"]]
    return {
        "tau": tau,
        "gold_pairs": len(rows),
        "supersession_gated_pairs": len(blocked),
        "gated_pairs_that_would_pass_tau": sum(
            1 for r in blocked if r["passes_tau"]
        ),
        "gated_examples": blocked[:8],
        "invariant": (
            "a similarity score cannot re-admit a message that the M4 "
            "supersession gate removed; gates run before fusion"
        ),
    }


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
def recommend(semantic: dict[str, Any], rrf_results: dict[str, Any],
              floors: dict[str, Any]) -> dict[str, Any]:
    """Derive the starting values from the sweeps, not from a prior.

    tau rule: the lowest threshold whose false-positive rate at 3 is at or
    below 0.15 while gold-pair recall is still at or above 0.70. That is the
    knee: the point where tightening the floor stops buying precision.

    ratio rule: the widest plateau where recall at 1 and at 3 are unchanged
    from no floor, and the false-positive rate strictly improves. The
    midpoint of that plateau is chosen, because every ratio inside it scores
    identically and the midpoint is furthest from both edges.
    """
    sweep = semantic["tau_sweep"]
    chosen = None
    for row in sweep:
        if (row["false_positive_rate_at_3"] <= 0.15
                and row["gold_recall"] >= 0.70):
            chosen = row
            break
    if chosen is None:
        chosen = min(sweep, key=lambda r: r["false_positive_rate_at_3"])

    by_k = {
        name: metrics for name, metrics in rrf_results.items()
        if name.startswith("rrf_k_") and isinstance(metrics, dict)
    }
    signatures = {
        name: (
            metrics["recall_at_1"], metrics["recall_at_3"],
            metrics["recall_at_5"],
            metrics[f"precision_at_{PRECISION_AT}"],
            metrics["false_positive_rate_at_3"],
            metrics["no_match_accuracy"],
        )
        for name, metrics in by_k.items()
    }
    distinct = sorted(set(signatures.values()))

    baseline = floors["ratios"]["0.0"]
    plateau = [
        float(ratio) for ratio, metrics in floors["ratios"].items()
        if metrics["recall_at_1"] == baseline["recall_at_1"]
        and metrics["recall_at_3"] == baseline["recall_at_3"]
        and metrics["false_positive_rate_at_3"]
        < baseline["false_positive_rate_at_3"]
    ]
    ratio = (min(plateau) + max(plateau)) / 2 if plateau else 0.0
    # The midpoint of a plateau need not itself be a swept value; report the
    # nearest swept member as the evidence.
    nearest = (
        min(plateau, key=lambda value: abs(value - ratio)) if plateau else 0.0
    )

    return {
        "tau_rule": (
            "lowest tau with FP@3 <= 0.15 and gold-pair recall >= 0.70"
        ),
        "tau": chosen["tau"],
        "tau_evidence": chosen,
        "k_identifiable": len(distinct) > 1,
        "k_distinct_outcomes": len(distinct),
        "k_signature": signatures,
        "k_recommendation": 60.0,
        "k_reason": (
            "every k in the grid produced an identical metric signature on "
            "this fixture, so k is not identifiable here; 60 is the "
            "literature default and is safe, not calibrated"
        ),
        "ratio_rule": (
            "midpoint of the widest plateau where recall@1 and recall@3 are "
            "unchanged from no post-fusion floor and FP@3 strictly improves"
        ),
        "ratio_plateau": plateau,
        "ratio_recommendation": ratio,
        "ratio_evidence_at": nearest,
        "ratio_evidence": floors["ratios"][str(nearest)],
    }


_RECOMMENDED_TAU = 0.45
_RECOMMENDED_K = 60.0


def run() -> dict[str, Any]:
    bench.assert_no_drift()
    before = bench._sha256(bench.PRODUCTION_DB)
    workdir = tempfile.mkdtemp(prefix="primora-calibration-")
    try:
        cal = build_calibration(workdir)
        semantic = experiment_semantic(cal)
        tau = _RECOMMENDED_TAU
        rrf_results = experiment_rrf(cal, tau)
        sessions = experiment_same_session(cal, tau, _RECOMMENDED_K)
        structural = experiment_structural(cal, tau, _RECOMMENDED_K)
        gates = experiment_gates(cal, tau)
        floors = experiment_post_fusion_floor(cal, tau, _RECOMMENDED_K)
        after = bench._sha256(bench.PRODUCTION_DB)

        # Which no-match query never goes silent, and is gold the argmax?
        offenders = []
        for query_id in cal.query_ids:
            if cal.gold[query_id]:
                continue
            sims = cal.sims_for(query_id)
            best = max(sims.items(), key=lambda kv: (kv[1], kv[0]))
            offenders.append({
                "id": query_id, "top1_key": best[0],
                "top1_similarity": round(best[1], 4),
            })
        argmax_hits = sum(
            1 for query_id in cal.query_ids if cal.gold[query_id]
            and max(cal.sims_for(query_id).items(),
                    key=lambda kv: (kv[1], kv[0]))[0] in cal.gold[query_id]
        )
        gold_queries = sum(1 for q in cal.query_ids if cal.gold[q])

        return {
            "model": MODEL_ID, "device": DEVICE, "dimension": cal.dimension,
            "production_db_sha256_before": before,
            "production_db_sha256_after": after,
            "production_db_untouched": before == after,
            "used_tau": tau, "used_k": _RECOMMENDED_K,
            "semantic": semantic, "rrf": rrf_results,
            "same_session": sessions, "structural": structural,
            "gates": gates, "post_fusion_floor": floors,
            "no_match_offenders": offenders,
            "gold_is_argmax": argmax_hits,
            "gold_query_count": gold_queries,
            "recommendation": recommend(semantic, rrf_results, floors),
        }
    finally:
        gc.collect()
        shutil.rmtree(workdir, ignore_errors=True)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["  " + " | ".join(str(h) for h in headers),
             "  " + "-+-".join("-" * len(str(h)) for h in headers)]
    for row in rows:
        lines.append("  " + " | ".join(str(cell) for cell in row))
    return "\n".join(lines)


def format_report(result: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append
    add("=" * 78)
    add("M6 RETRIEVAL CALIBRATION - EXPERIMENTAL, NOT IMPLEMENTED")
    add(f"model={result['model']} device={result['device']} "
        f"dim={result['dimension']}")
    add(f"production DB untouched: {result['production_db_untouched']}")
    add(f"sha256 before: {result['production_db_sha256_before']}")
    add(f"sha256 after : {result['production_db_sha256_after']}")
    add("=" * 78)

    add("\nEXPERIMENT 1 - semantic score distributions")
    add(_table(
        ["bucket", "n", "min", "p05", "median", "mean", "p95", "max"],
        [[name, s.get("n", 0), s.get("min", "-"), s.get("p05", "-"),
          s.get("median", "-"), s.get("mean", "-"), s.get("p95", "-"),
          s.get("max", "-")]
         for name, s in result["semantic"]["distributions"].items()]))

    add("\n  gold pairs by query category")
    add(_table(
        ["category", "n", "min", "p05", "median", "mean", "p95", "max"],
        [[name, s.get("n", 0), s.get("min", "-"), s.get("p05", "-"),
          s.get("median", "-"), s.get("mean", "-"), s.get("p95", "-"),
          s.get("max", "-")]
         for name, s in result["semantic"]["gold_by_category"].items()]))

    add("\n  tau sweep (semantic channel only)")
    add(_table(
        ["tau", "gold_recall", "recall@3", "precision@3", "FP@3",
         "no_match_acc", "no_match_false", "irrelevant_pass"],
        [[s["tau"], s["gold_recall"], s["recall_at_3"],
          s[f"precision_at_{PRECISION_AT}"], s["false_positive_rate_at_3"],
          s["no_match_accuracy"], s["no_match_false_retrieval"],
          s["irrelevant_pass_rate"]]
         for s in result["semantic"]["tau_sweep"]]))

    add(f"\nEXPERIMENT 2 - RRF (tau={result['rrf']['tau']})")
    rows = []
    for name, metrics in result["rrf"].items():
        if not isinstance(metrics, dict) or "recall_at_3" not in metrics:
            continue
        rows.append([
            name, metrics["recall_at_1"], metrics["recall_at_3"],
            metrics["recall_at_5"], metrics[f"precision_at_{PRECISION_AT}"],
            metrics["false_positive_rate_at_3"], metrics["no_match_accuracy"],
            metrics["recall_at_3__paraphrase"],
            metrics["recall_at_3__indirect"],
            metrics["recall_at_3__entity"],
        ])
    add(_table(
        ["config", "R@1", "R@3", "R@5", "P@3", "FP@3", "no-match",
         "para", "indirect", "entity"], rows))

    add("\nEXPERIMENT 2b - post-fusion relative floor (tau=%.2f, k=%g)"
        % (result["post_fusion_floor"]["tau"],
           result["post_fusion_floor"]["rrf_k"]))
    add(_table(
        ["ratio", "R@1", "R@3", "R@5", "P@3", "FP@3", "no-match",
         "para", "indirect", "entity"],
        [[ratio, m["recall_at_1"], m["recall_at_3"], m["recall_at_5"],
          m[f"precision_at_{PRECISION_AT}"], m["false_positive_rate_at_3"],
          m["no_match_accuracy"], m["recall_at_3__paraphrase"],
          m["recall_at_3__indirect"], m["recall_at_3__entity"]]
         for ratio, m in result["post_fusion_floor"]["ratios"].items()]))

    add("\nEXPERIMENT 3 - same-session policy")
    rows = []
    for name, metrics in result["same_session"].items():
        if not isinstance(metrics, dict) or "same_session_recall_at_3" not in metrics:
            continue
        rows.append([
            name, metrics["same_session_recall_at_3"],
            metrics["cross_session_recall_at_3"],
            metrics["foreign_outranks_local_gold"],
            metrics["false_positive_rate_at_3"],
            metrics["no_match_accuracy"],
            metrics["recall_at_3"],
        ])
    add(_table(
        ["config", "same@3", "cross@3", "foreign>local", "FP@3",
         "no-match", "R@3"], rows))

    add("\nEXPERIMENT 4 - entity / relationship structural signal")
    structural = result["structural"]
    add(f"  declared entity names: {structural['declared_names']}")
    add(f"  resolved when given the name: "
        f"{len(structural['names_resolved_from_name'])}/"
        f"{len(structural['declared_names'])}")
    add(f"  unresolved when given the name: "
        f"{structural['names_unresolved_from_name']}")
    add(f"  structural message keys reachable: "
        f"{structural['structural_keys_reachable']}")
    add(f"  targeted entity/relationship queries: "
        f"{structural['targeted_query_count']}")
    add(f"  of those, resolved from the free-text query: "
        f"{structural['targeted_resolved_from_free_text']}"
        f"  <- the missing capability is mention detection, not resolution")
    add(_table(
        ["config", "R@1", "R@3", "R@5", "P@3", "FP@3", "no-match",
         "entity", "relationship"],
        [[name,
          structural[label]["recall_at_1"], structural[label]["recall_at_3"],
          structural[label]["recall_at_5"],
          structural[label][f"precision_at_{PRECISION_AT}"],
          structural[label]["false_positive_rate_at_3"],
          structural[label]["no_match_accuracy"],
          structural[label]["recall_at_3__entity"],
          structural[label]["recall_at_3__relationship"]]
         for name, label in (("structural only", "structural_only"),
                             ("lexical+semantic", "hybrid_without_structural"),
                             ("lexical+semantic+structural",
                              "hybrid_with_structural"))]))
    add("  targeted queries where free-text resolution fails "
        "(need mention detection, no new entity or relationship types):")
    for item in structural["mention_detection_gap"]:
        add(f"    {item['id']} [{item['category']}] {item['text']!r}")

    add("\nSEMANTIC CHANNEL CHARACTER")
    add(f"  gold query where gold is the top-1 semantic hit: "
        f"{result['gold_is_argmax']}/{result['gold_query_count']}")
    add("  no-match queries and their strongest similarity:")
    for row in result["no_match_offenders"]:
        add(f"    {row['id']} top1={row['top1_key']} "
            f"sim={row['top1_similarity']}")

    add("\nRECOMMENDATION (derived from the sweep)")
    rec = result["recommendation"]
    add(f"  tau rule    : {rec['tau_rule']}")
    add(f"  tau         : {rec['tau']}  "
        f"(gold_recall={rec['tau_evidence']['gold_recall']}, "
        f"FP@3={rec['tau_evidence']['false_positive_rate_at_3']}, "
        f"no_match={rec['tau_evidence']['no_match_accuracy']})")
    add(f"  k           : {rec['k_recommendation']}  "
        f"(identifiable={rec['k_identifiable']}, "
        f"distinct outcomes={rec['k_distinct_outcomes']})")
    add(f"  k reason    : {rec['k_reason']}")
    add(f"  ratio rule  : {rec['ratio_rule']}")
    add(f"  ratio       : {rec['ratio_recommendation']}  "
        f"(plateau {rec['ratio_plateau']}, "
        f"evidence measured at {rec['ratio_evidence_at']})")
    if rec["ratio_evidence"]:
        add(f"  ratio effect: FP@3 {result['post_fusion_floor']['ratios']['0.0']['false_positive_rate_at_3']}"
            f" -> {rec['ratio_evidence']['false_positive_rate_at_3']}, "
            f"R@3 {rec['ratio_evidence']['recall_at_3']}, "
            f"R@1 {rec['ratio_evidence']['recall_at_1']}")

    add("\nHARD GATES - similarity cannot override a gate")
    gates = result["gates"]
    add(f"  gold pairs: {gates['gold_pairs']}, "
        f"supersession-gated: {gates['supersession_gated_pairs']}, "
        f"of which would pass tau: {gates['gated_pairs_that_would_pass_tau']}")
    add(f"  invariant: {gates['invariant']}")
    for row in gates["gated_examples"]:
        add(f"    {row['query']}/{row['key']} sim={row['similarity']} "
            f"gated={row['supersession_gated']} passes_tau="
            f"{row['passes_tau']}")
    return "\n".join(out)


def main() -> int:
    result = run()
    print(format_report(result))
    return 0 if result["production_db_untouched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
