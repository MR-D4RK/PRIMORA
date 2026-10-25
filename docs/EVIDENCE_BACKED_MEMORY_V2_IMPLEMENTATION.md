# PRIMORA — Evidence-Backed Memory V2: Integrated Implementation

Status: **implementation specification, not proof of completed implementation**.
Base: `milestone/unified-evidence-memory` (reference checkpoint `04fb6a80817e634835ca4c7b97b06a40e79de950`).

## Objective

Deliver five interdependent capabilities as **one integrated, reviewable implementation**. Use internal commit/checkpoint boundaries, but do not require manual patch application or intermediate merges. Prime remains the source of admission and authority. SQLite remains the only persistent memory source of truth. Aurora retains identity and cognitive ownership.

## Milestones and required outputs

### M1 — Evidence-backed observation layer

- Add additive, idempotent SQLite migrations for `observations`, `observation_sources`, and `observation_history` with foreign keys and indexes for scope, status, normalized concept, valid-time interval, and source.
- Add frozen `ObservationRecord` and `ObservationHistoryRecord` models.
- Implement `prime/context/observation.py`: validate user-sourced evidence, create/read observation, deduplicate sources, derive proof_count from actual links, supersede with immutable history snapshots, and reject unsupported or cross-scope assertions.
- Preserve immutable `messages` and all original state/entity/relationship/episode rows. Do not automatically backfill production.
- Make all new reads/writes feature-flag controlled, disabled by default until acceptance.

### M2 — General typed memory formation

- Introduce structured, domain-independent candidate types (preference, state, event, relationship, observation proposal). Preserve distinctions among assertion, hypothesis, question, fiction, negation, and third-party/assistant content.
- One *optional*, bounded provider call for a new user message; the provider produces **proposals only**, never SQL or accepted facts.
- Deterministically validate authority, source IDs, actor/subject, allowed predicates, temporal bounds, source-quote support, and lifecycle before Prime writes.
- On malformed JSON/provider failure/timeout, fall back to existing deterministic extraction, never silently admit inferred facts.
- No specialized car/color/music dictionaries or user-specific hacks. Use unfamiliar synthetic domains.

### M3 — Typed preference queries and ranking

- Interpret query intent (predicate/fact kind, scope, count, latest/earliest, current/as-of, filters) separately from generation.
- Construct candidate pools from lexical, semantic (over-fetched), state, observation, entity, relationship and temporal channels. Lexical-only relevant records must be eligible.
- Apply **authority, status, validity and provenance hard gates before ranking**. Keep `confidence` from overriding authority. Deduplicate canonical concept/value + shared evidence.
- Validate question relevance *before* applying ordering and exact top-N count. In last-N queries, use source/occurred times according to user wording; tie-breaking deterministic.
- Optional entity boosts, query-length-aware lexical calibration and reranking only after gates. Bounded low-latency deterministic path by default; ordinary recall adds zero LLM calls.
- Return compact evidence packets with source IDs and abstain when evidence is insufficient.

### M4 — Temporal entities, contradictions and corrections

- Extend existing entity/relationship structures additively: source episode IDs and valid/invalid intervals (distinguish reported time and occurred time).
- Resolve aliases with bounded graph traversal. Avoid introducing graph services.
- Contradictions generate a reviewable new candidate/observation supersession, not deletion or silent fact rewriting. Preserve old versions and immutable sources.
- A new assertion can be current, while earlier assertions remain accessible for historical queries.
- No Aurora self-modification of identity, runtime policy, or approval controls.

### M5 — Integration, evaluations, operational guardrails

- Unit and integration tests use only disposable SQLite DBs; prohibit any test connection to `PRIMORA/data/context/aurora_context.db`.
- Benchmark at least 200 seeded multi-session synthetic conversations over 50+ unfamiliar preference domains, with distractors, corrections, hedges, negation, fiction, assistant suggestions, aliases and two-clock events.
- Capture baseline vs candidate (same evidence, same runner), report formation P/R/F1, P@3, Recall@3, ordering/count, corrections, evidence ID coverage, unsupported-claim rate, cross-session continuity, model calls, p50/p95 latency, RAM/VRAM where measurable.
- Targets: preference P@3 >= .90, Recall@3 >= .85, exact count/order >= .95, correction current-state >= .95, immutable source preservation 100%, unsupported personal claims <= 1% (0% on evidence-only), cross-session continuity >= .90, recall +0 LLM calls, formation <= 1 bounded Qwen call, and latency <= baseline * 1.25 with neural reranking off.
- If Qwen, VRAM, or runtime unavailable, mark measurements **NOT RUN**; never substitute upstream hosted benchmark results.

## Runtime and migration contract

- Additive SQLite migrations only, transactional/idempotent.
- Production backfill requires explicit opt-in CLI command with dry-run and source validation.
- Default mode preserves legacy behavior and tables. Feature flags permit immediate disable/rollback without modifying production memories.
- No new database, model service, Qdrant, Neo4j, PostgreSQL, LangGraph, Trustcall, or cloud inference dependency.
- No automatic merge, deployment, production database access, or destructive operation.

## Integration sequence (one delivery, internal checkpoints)

1. Implement schema/models/observation service and tests.
2. Implement proposed typed formation + validation/fallback and tests.
3. Implement typed retrieval/query semantics and temporal ranking plus tests.
4. Implement temporal relationship lifecycle and contradiction tests.
5. Integrate into Mind/MemoryPlan, run full deterministic suite and benchmark, audit provenance, document performance and known limitations.
6. Submit a **single draft PR** to the existing milestone branch, with test evidence, migration plan, rollback and explicit PASS/FAIL for each checkpoint.

## Definition of done

- All five milestones implemented and demonstrably tested, not merely documented.
- Existing regression tests pass.
- New synthetic benchmark thresholds pass, or failures are reported with reproductions.
- Production DB untouched; source preservation independently verified.
- One reviewable PR, no automatic merge.
