"""M6 hybrid retrieval: lexical + semantic + M5 structural channels.

Calibrated constants, all from `MEMORY_RETRIEVAL_CALIBRATION.md`:

* ``SEMANTIC_TAU = 0.45`` — the semantic positive-evidence floor, selected as
  the lowest threshold with FP@3 <= 0.15 and gold-pair recall >= 0.70;
* ``RRF_K = 60.0`` — reciprocal rank fusion constant. **Explicitly not
  calibrated**: every k in {0, 5, 10, 20, 50, 60, 100} produced an identical
  metric signature on the current benchmark, so 60 is the literature default,
  a safe placeholder, not a tuned value;
* ``EVIDENCE_FLOOR_RATIO = 0.725`` — post-fusion relative floor, the midpoint
  of the measured plateau [0.60, 0.95] where recall@1/3/5 were unchanged and
  FP@3 fell from 0.6176 to 0.4583.

Order of operations is load-bearing and must not be reordered:

    hard gates -> channels -> RRF -> relative floor -> bounded result

The hard gates are deterministic and absolute. A semantic similarity, a fused
score, or a structural match can never re-admit anything a gate removed, and
M1 ``confidence`` is never read here and never influences ranking.
"""

import time
from typing import Any, Optional

from . import queries as queries_mod
from . import scoring as scoring_mod
from . import semantic as semantic_mod
from .models import RetrievalCandidate, RetrievalTrace

SEMANTIC_TAU = 0.45
RRF_K = 60.0
EVIDENCE_FLOOR_RATIO = 0.725

MAX_CANDIDATES = 24
MAX_SEMANTIC_CANDIDATES = 12
MAX_STRUCTURAL_CANDIDATES = 8
MAX_SCOPE_FANOUT = 4

CHANNELS = ("lexical", "semantic", "entity", "relationship")


def item_id(kind: str, ref_id: int) -> str:
    return f"{kind}:{int(ref_id)}"


# ----------------------------------------------------------------------
# Hard gates. Absolute, deterministic, applied before any scoring.
# ----------------------------------------------------------------------
def gate_state(state, at_time: float, temporal: str) -> tuple[bool, str]:
    """M1-M4 gate for a structured memory row.

    Authority, lifecycle, and M4 half-open validity. `confidence` is
    deliberately not consulted: it is evidential support, never an ordering
    input and never a substitute for authority. No channel in this module
    reads a confidence value at all, which keeps the rule auditable by a
    plain source scan.
    """
    if state.authority != "user":
        return False, "authority"
    if state.status == "unresolved":
        return False, "unresolved"
    if state.status == "rejected":
        return False, "rejected"
    if temporal == "current" and state.status != "current":
        return False, "not_current"
    if temporal in ("as_of", "historical") and state.status not in (
        "current", "superseded"
    ):
        return False, "status"
    if state.valid_from is None:
        return False, "no_valid_from"
    if not (state.valid_from <= at_time
            and (state.valid_until is None or at_time < state.valid_until)):
        return False, "outside_validity"
    return True, "pass"


def gate_entity(entity, temporal: str) -> tuple[bool, str]:
    if entity.authority != "user":
        return False, "authority"
    if temporal == "current":
        if entity.status != "active":
            return False, "not_active"
        return True, "pass"
    if entity.status == "archived":
        return False, "archived"
    return True, "pass"


# ----------------------------------------------------------------------
# Channel A: semantic
# ----------------------------------------------------------------------
def semantic_channel(
    index, store, query_text: str, scope: str, at_time: float,
    temporal: str, tau: float = SEMANTIC_TAU,
    limit: int = MAX_SEMANTIC_CANDIDATES, include_global: bool = False,
) -> tuple[list[RetrievalCandidate], list[str]]:
    """Semantic candidates, hard-gated before scoring.

    Reads the derived index only. Never writes, and never relaxes a gate.
    """
    diagnostics: list[str] = []
    if index is None or not query_text or not query_text.strip():
        return [], diagnostics
    scopes = [scope] + (["global"] if include_global and scope != "global" else [])
    started = time.perf_counter()
    try:
        rows = index.search(query_text, scopes, tau=tau, limit=limit)
    except Exception as exc:
        return [], [f"semantic_channel_failed: {type(exc).__name__}: {exc}"]
    if not rows and not index.available():
        diagnostics.extend(index.diagnostics())
    out: list[RetrievalCandidate] = []
    for rank, row in enumerate(rows, start=1):
        kind, ref_id = row["kind"], int(row["ref_id"])
        if kind == "state":
            live = store.fetch_states(
                scope=row["scope"], state_ids=[ref_id], limit=1
            )
            record = live[0] if live else None
            ok, reason = (
                gate_state(record, at_time, temporal) if record
                else (False, "missing")
            )
        else:
            record = store.fetch_entity(ref_id)
            if record is None or record.scope != row["scope"]:
                record, ok, reason = None, False, "missing"
            else:
                ok, reason = gate_entity(record, temporal)
        if not ok:
            diagnostics.append(
                f"semantic_gated: {kind}:{ref_id} reason={reason}"
            )
            continue
        out.append(RetrievalCandidate(
            item_id=item_id(kind, ref_id), kind=kind, ref_id=ref_id,
            scope=row["scope"], session_id=row["session_id"],
            channel="semantic", channel_rank=rank,
            channel_score=float(row["similarity"]),
            similarity=float(row["similarity"]),
        ))
    diagnostics.append(
        f"semantic_channel_ms: {(time.perf_counter() - started) * 1000:.3f}"
    )
    return out, diagnostics


# ----------------------------------------------------------------------
# Channel D/E: M5 structural
# ----------------------------------------------------------------------
def entity_mentions(store, scope: str, query_text: str) -> list[Any]:
    """Exact, word-boundary entity mentions found in the query text.

    Deliberately conservative: an entity counts as mentioned only when its
    full normalized name, or a confirmed alias, appears as a contiguous run
    of query tokens. No fuzzy matching, no embedding, no type guessing, and
    no inference. An entity that is not literally named contributes nothing.
    """
    query_tokens = scoring_mod.tokenize(query_text or "")
    if not query_tokens:
        return []
    try:
        entities = store.fetch_entities(scope=scope, status="active")
    except Exception:
        return []
    mentioned: list[Any] = []
    for entity in entities:
        variants = [scoring_mod.tokenize(entity.display_name)]
        try:
            for alias in store.fetch_entity_aliases(
                entity_id=entity.entity_id, status="current", limit=8
            ):
                variants.append(scoring_mod.tokenize(alias.alias_text))
        except Exception:
            pass
        for tokens in variants:
            if not tokens:
                continue
            span = len(tokens)
            for start in range(0, len(query_tokens) - span + 1):
                if query_tokens[start:start + span] == tokens:
                    mentioned.append(entity)
                    break
            if mentioned and mentioned[-1] is entity:
                break
    return mentioned


def structural_channel(
    manager, store, scope: str, query_text: str, at_time: float,
    temporal: str, limit: int = MAX_STRUCTURAL_CANDIDATES,
) -> tuple[list[RetrievalCandidate], list[str]]:
    """Entity and relationship candidates through the bounded M5 surface.

    One hop only, M5 limits inherited unchanged, scope-explicit. `contains`
    is the derived inverse computed at read time; nothing is written.
    """
    diagnostics: list[str] = []
    started = time.perf_counter()
    out: list[RetrievalCandidate] = []
    try:
        mentioned = entity_mentions(store, scope, query_text)
    except Exception as exc:
        return [], [
            f"structural_channel_failed: {type(exc).__name__}: {exc}"
        ]
    if not mentioned:
        diagnostics.append("structural_channel_no_mention")
    entity_rank = 0
    relationship_rank = 0
    for entity in mentioned[:MAX_SCOPE_FANOUT]:
        if entity.status != "active":
            diagnostics.append(
                f"structural_gated: entity:{entity.entity_id} "
                f"reason=not_active"
            )
            continue
        entity_rank += 1
        out.append(RetrievalCandidate(
            item_id=item_id("entity", entity.entity_id), kind="entity",
            ref_id=entity.entity_id, scope=entity.scope,
            session_id=entity.session_id, channel="entity",
            channel_rank=entity_rank, channel_score=1.0,
            entity_id=entity.entity_id,
        ))
        try:
            for link in manager.get_entity_memories(
                scope, entity.entity_id, temporal=temporal, at_time=at_time,
                limit=limit, follow_merge=False,
            ):
                relationship_rank += 1
                ok, reason = gate_state(link.state, at_time, temporal)
                if not ok:
                    diagnostics.append(
                        f"structural_gated: state:{link.state.state_id} "
                        f"reason={reason}"
                    )
                    continue
                out.append(RetrievalCandidate(
                    item_id=item_id("state", link.state.state_id),
                    kind="state", ref_id=link.state.state_id,
                    scope=link.state.scope, session_id=link.state.session_id,
                    channel="entity", channel_rank=relationship_rank,
                    channel_score=1.0,  # rank comes from channel_rank only
                    relationship_id=link.relationship.relationship_id,
                    relationship_type=link.relationship.relationship_type,
                    entity_id=entity.entity_id,
                ))
            for related in manager.get_related_entities(
                scope, entity.entity_id, direction="both", temporal=temporal,
                at_time=at_time, limit=limit, follow_merge=False,
            ):
                if related.entity.entity_id == entity.entity_id:
                    continue
                ok, reason = gate_entity(related.entity, temporal)
                if not ok:
                    diagnostics.append(
                        f"structural_gated: entity:"
                        f"{related.entity.entity_id} reason={reason}"
                    )
                    continue
                relationship_rank += 1
                out.append(RetrievalCandidate(
                    item_id=item_id("entity", related.entity.entity_id),
                    kind="entity", ref_id=related.entity.entity_id,
                    scope=related.entity.scope,
                    session_id=related.entity.session_id,
                    channel="relationship", channel_rank=relationship_rank,
                    channel_score=1.0,  # rank comes from channel_rank only
                    relationship_id=related.relationship.relationship_id,
                    relationship_type=related.relationship.relationship_type,
                    entity_id=entity.entity_id,
                ))
            for link in manager.get_entity_memories(
                scope, entity.entity_id, temporal=temporal, at_time=at_time,
                limit=limit, follow_merge=False,
            ):
                for path in manager.get_relationship_explanation(
                    scope, link.state.state_id, direction="outbound",
                    temporal=temporal, at_time=at_time, limit=2,
                ):
                    for state in (path.rationale_state, path.decision_state):
                        ok, reason = gate_state(state, at_time, temporal)
                        if not ok:
                            continue
                        relationship_rank += 1
                        out.append(RetrievalCandidate(
                            item_id=item_id("state", state.state_id),
                            kind="state", ref_id=state.state_id,
                            scope=state.scope, session_id=state.session_id,
                            channel="relationship",
                            channel_rank=relationship_rank,
                            channel_score=1.0,  # rank comes from channel_rank only
                            relationship_id=path.relationship.relationship_id,
                            relationship_type=(
                                path.relationship.relationship_type
                            ),
                        ))
        except Exception as exc:
            diagnostics.append(
                f"structural_partial_failure: {type(exc).__name__}: {exc}"
            )
    diagnostics.append(
        f"structural_channel_ms: {(time.perf_counter() - started) * 1000:.3f}"
    )
    return out, diagnostics


# ----------------------------------------------------------------------
# Channel A (lexical): production candidates, re-identified for fusion
# ----------------------------------------------------------------------
def lexical_channel(
    collected: dict[str, Any], cross_cap: int = 3,
) -> list[RetrievalCandidate]:
    """Re-identify production lexical candidates under fusion item ids.

    The rows and their order come from `ContextManager.collect_lexical_candidates`
    so the hybrid path ranks exactly what Aurora would receive today.
    """
    out: list[RetrievalCandidate] = []
    for rank, (score, _seq, hit, _bd) in enumerate(collected.get("same", []),
                                                  start=1):
        out.append(RetrievalCandidate(
            item_id=item_id("message", hit.message_id), kind="message",
            ref_id=hit.message_id, scope=hit.session_id,
            session_id=hit.session_id, channel="lexical", channel_rank=rank,
            channel_score=float(score),
        ))
    for rank, (score, _sid, _seq, hit, _bd) in enumerate(
        collected.get("cross", [])[:cross_cap], start=1
    ):
        out.append(RetrievalCandidate(
            item_id=item_id("message", hit.message_id), kind="message",
            ref_id=hit.message_id, scope=hit.session_id,
            session_id=hit.session_id, channel="lexical", channel_rank=rank,
            channel_score=float(score),
        ))
    return out


# ----------------------------------------------------------------------
# Fusion
# ----------------------------------------------------------------------
def rrf_fuse(
    channels: dict[str, list[RetrievalCandidate]], k: float = RRF_K,
) -> tuple[dict[str, float], list[str]]:
    """Reciprocal rank fusion over per-channel ranked lists.

    `fused(x) = sum over channels of 1 / (k + rank_in_channel(x))`, ranks
    starting at 1. Deliberately rank-based: cosine similarity, the lexical
    linear score, and a structural match are incommensurable, and summing
    them with weights would require calibrating scales that drift with the
    corpus. Ties break on item id so ordering is deterministic.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[str, float] = {}
    for name in CHANNELS:
        for candidate in channels.get(name, []):
            scores[candidate.item_id] = scores.get(candidate.item_id, 0.0) + (
                1.0 / (k + candidate.channel_rank)
            )
    order = [
        key for key, _ in sorted(
            scores.items(), key=lambda kv: (-kv[1], kv[0])
        )
    ]
    return scores, order


def apply_evidence_floor(
    scores: dict[str, float], order: list[str], ratio: float,
) -> tuple[list[str], bool]:
    """Drop fused items supported only by weak evidence.

    Definition, applied deterministically: keep an item when
    `fused(item) >= ratio * fused(best_item)`. The best item always scores
    1.0 of itself, so the floor is a ratio against the strongest evidence in
    this request rather than an absolute score, which is what makes it
    portable across corpora. This runs *after* fusion and *after* the hard
    gates, so it can never widen what a gate allowed.
    """
    if not order:
        return [], False
    best = scores[order[0]]
    if best <= 0:
        return list(order), False
    kept = [key for key in order if scores[key] >= ratio * best]
    return kept, len(kept) < len(order)


def assign_ranks(
    candidates: list[RetrievalCandidate], scores: dict[str, float],
    order: list[str], floor_applied: bool,
) -> list[RetrievalCandidate]:
    """Attach fused score, final rank, and inclusion to each candidate."""
    rank_of = {key: index + 1 for index, key in enumerate(order)}
    out: list[RetrievalCandidate] = []
    for candidate in candidates:
        key = candidate.item_id
        included = key in rank_of
        reason = "" if included else (
            "below_evidence_floor" if floor_applied else "not_fused"
        )
        out.append(RetrievalCandidate(
            item_id=candidate.item_id, kind=candidate.kind,
            ref_id=candidate.ref_id, scope=candidate.scope,
            session_id=candidate.session_id, channel=candidate.channel,
            channel_rank=candidate.channel_rank,
            channel_score=candidate.channel_score,
            similarity=candidate.similarity,
            relationship_id=candidate.relationship_id,
            relationship_type=candidate.relationship_type,
            entity_id=candidate.entity_id,
            fused_score=round(scores.get(key, 0.0), 9),
            final_rank=rank_of.get(key),
            included=included, exclusion_reason=reason,
        ))
    return out


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------
def hybrid_retrieve(
    manager, store, index, session_id: str, scope: str, query_text: str,
    collected: Optional[dict[str, Any]] = None,
    temporal: str = "current", at_time: Optional[float] = None,
    tau: float = SEMANTIC_TAU, k: float = RRF_K,
    floor_ratio: float = EVIDENCE_FLOOR_RATIO,
    limit: int = MAX_CANDIDATES, include_global: bool = False,
    use_structural: bool = True, cross_cap: int = 3,
) -> RetrievalTrace:
    """Run every channel, fuse, floor, and return a bounded trace.

    Read-only. Never raises: each channel failure degrades to that channel
    being empty with a diagnostic, so a missing model or a broken structural
    query can never break lexical retrieval.
    """
    started = time.perf_counter()
    evaluation = (
        at_time if at_time is not None else time.time()
    )
    diagnostics: list[str] = []
    timings: dict[str, float] = {}
    all_candidates: list[RetrievalCandidate] = []

    # Channel A: lexical, from production collection.
    channel_started = time.perf_counter()
    if collected is None:
        from . import scoring as _scoring
        query_terms = _scoring.build_query(query_text, "")
        collected = manager.collect_lexical_candidates(
            session_id, query_terms
        )
    lexical = lexical_channel(collected, cross_cap=cross_cap)
    timings["lexical_ms"] = (time.perf_counter() - channel_started) * 1000
    all_candidates.extend(lexical)

    # Channel B: semantic.
    channel_started = time.perf_counter()
    try:
        semantic, semantic_diagnostics = semantic_channel(
            index, store, query_text, scope, evaluation, temporal, tau=tau,
            limit=min(MAX_SEMANTIC_CANDIDATES, limit),
            include_global=include_global,
        )
    except Exception as exc:
        semantic = []
        semantic_diagnostics = [
            f"semantic_channel_failed: {type(exc).__name__}: {exc}"
        ]
    timings["semantic_ms"] = (time.perf_counter() - channel_started) * 1000
    diagnostics.extend(semantic_diagnostics)
    all_candidates.extend(semantic)

    # Channels D and E: M5 structural, only when the query names an entity.
    structural: list[RetrievalCandidate] = []
    if use_structural:
        channel_started = time.perf_counter()
        try:
            structural, structural_diagnostics = structural_channel(
                manager, store, scope, query_text, evaluation, temporal,
                limit=min(MAX_STRUCTURAL_CANDIDATES, limit),
            )
        except Exception as exc:
            structural = []
            structural_diagnostics = [
                f"structural_channel_failed: {type(exc).__name__}: {exc}"
            ]
        timings["structural_ms"] = (
            time.perf_counter() - channel_started
        ) * 1000
        diagnostics.extend(structural_diagnostics)
        all_candidates.extend(structural)

    channels: dict[str, list[RetrievalCandidate]] = {
        "lexical": lexical, "semantic": semantic,
        "entity": [c for c in structural if c.channel == "entity"],
        "relationship": [
            c for c in structural if c.channel == "relationship"
        ],
    }

    # Fusion.
    channel_started = time.perf_counter()
    try:
        scores, order = rrf_fuse(channels, k=k)
        order, floor_applied = apply_evidence_floor(
            scores, order, floor_ratio
        )
        order = order[:limit]
    except Exception as exc:
        scores, order, floor_applied = {}, [], False
        diagnostics.append(
            f"fusion_failed: {type(exc).__name__}: {exc}"
        )
    timings["fusion_ms"] = (time.perf_counter() - channel_started) * 1000
    if floor_applied:
        diagnostics.append("fusion_evidence_floor_applied")

    ranked = assign_ranks(all_candidates, scores, order, floor_applied)
    total_ms = (time.perf_counter() - started) * 1000
    degraded = any(
        item.startswith((
            "semantic_model_unavailable", "semantic_model_load_failed",
            "semantic_channel_failed", "structural_channel_failed",
            "fusion_failed",
        ))
        for item in diagnostics
    )
    return RetrievalTrace(
        session_id=session_id, scope=scope, query=query_text,
        temporal=temporal, at_time=evaluation, candidates=ranked,
        fused_order=list(order), channel_timings_ms=timings,
        total_ms=total_ms, degraded=degraded, diagnostics=diagnostics,
    )


def shadow_comparison(
    lexical_ids: list[int], trace: RetrievalTrace, session_id: str,
    query_text: str,
) -> Any:
    """Compare what hybrid would have returned against what Aurora got."""
    from .models import ShadowComparison

    hybrid_ids = [
        int(key.split(":", 1)[1]) for key in trace.fused_order
        if key.startswith("message:")
    ]
    lexical_set = list(lexical_ids)
    hybrid_set = hybrid_ids
    contributions: dict[str, int] = {}
    for candidate in trace.candidates:
        if candidate.included:
            contributions[candidate.channel] = (
                contributions.get(candidate.channel, 0) + 1
            )
    return ShadowComparison(
        session_id=session_id, query=query_text,
        lexical_ids=lexical_set,
        lexical_ranks={rid: i + 1 for i, rid in enumerate(lexical_set)},
        hybrid_ids=hybrid_set,
        hybrid_ranks={rid: i + 1 for i, rid in enumerate(hybrid_set)},
        overlap=[rid for rid in hybrid_set if rid in lexical_set],
        new_only=[rid for rid in hybrid_set if rid not in lexical_set],
        old_only=[rid for rid in lexical_set if rid not in hybrid_set],
        channel_contributions=contributions,
        channel_timings_ms=dict(trace.channel_timings_ms),
        hybrid_latency_ms=trace.total_ms,
        fused_floor_applied=any(
            d == "fusion_evidence_floor_applied" for d in trace.diagnostics
        ),
        degraded=trace.degraded,
        error="; ".join(
            d for d in trace.diagnostics
            if d.startswith((
                "semantic_model_unavailable", "semantic_model_load_failed",
                "semantic_channel_failed", "structural_channel_failed",
                "fusion_failed",
            ))
        ),
    )
