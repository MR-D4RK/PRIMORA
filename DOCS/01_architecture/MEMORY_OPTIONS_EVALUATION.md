# Memory Options Evaluation for Aurora

**Status:** Research and recommendation only. No implementation, installation, migration, or production-data change was performed.
**Date:** 2026-10-19
**Scope:** M2 Phase 2 — evaluate Mem0 OSS, Graphiti, LangMem, Letta, and the existing PRIMORA system against Aurora's actual memory requirements.
**Decision in one sentence:** Retain the PRIMORA memory core, fix its two binding gaps natively, and do not adopt a candidate framework wholesale; the only defensible external experiment is a tightly isolated Mem0 OSS shadow run for candidate extraction, never as a source of truth.

---

## 1. Evidence boundary and method

### 1.1 Source hierarchy

This evaluation used, in descending order of authority:

1. The current PRIMORA source code and tests in this repository.
2. A read-only fingerprint and row-count inspection of the production memory database.
3. Candidate project documentation, package metadata, repository metadata, and license files retrieved on 2026-10-19.
4. Community reports for issues that vendor documentation does not settle.

Where documentation and code disagree, code wins. Where a candidate capability could not be established, it is marked unverified rather than assumed. Vendor benchmarks are not treated as evidence about Aurora.

### 1.2 What was and was not done

- No package was installed.
- No candidate was run.
- No benchmark was executed.
- No production memory was exported, copied, or sent to an external service.
- No application code, configuration, dependency, or database schema was changed.
- No commit was created.

The production database was opened with SQLite `mode=ro` only. The fingerprint is content-based, not file-hash-based, so WAL folding cannot produce a false alarm (`PRIMORA/prime/context/isolation.py:75`).

### 1.3 Local deployment envelope

All fit judgments assume the stated target environment:

- Windows host, local-first operation.
- NVIDIA RTX 4060 Ti with 8 GB VRAM and approximately 16 GB system RAM.
- Qwen-family 8B model served through a local OpenAI-compatible endpoint.
- Existing Python and SQLite/FTS5 stack.
- No mandatory cloud dependency for memory.
- Identity and conversational continuity must survive any memory change.

Scores in §6 are fit-to-Aurora scores, not general product-quality scores.

### 1.4 Confidence labels

- **Known:** verified in source code, tests, the read-only database, or primary package/repository metadata.
- **Inferred:** follows from documented architecture or an observed code path, but was not run locally.
- **Needs testing:** depends on model behavior, resource use, or an integration path that this phase did not execute.

---

## 2. Executive recommendation

### 2.1 Retain PRIMORA as the memory source of truth

No evaluated candidate is a drop-in replacement for Aurora's current authority model. PRIMORA already provides the properties that are hardest to retrofit:

- Append-only raw message history.
- User-only authority for current personal facts.
- Provenance through `source_ids`.
- Half-open validity windows and explicit supersession.
- Assistant claims recorded as unresolved rather than promoted to truth.
- Candidate confirmation for uncertain and hedged statements.
- Deterministic, offline operation with no model call required for memory formation.
- An isolation harness that refuses production paths during experiments.

Those are not side features. They are the reason Aurora can distinguish "the user told me" from "the model inferred it" and "the assistant suggested it." Mem0, Graphiti, LangMem, and Letta each have strengths, but none preserves that distinction as a first-class invariant.

### 2.2 The two gaps worth fixing first

The evaluation found two binding gaps, both fixable without a framework:

1. **Assistant recommendations are not represented structurally.** If the user says "I love Radiohead" and Aurora says "You might like Portishead," Aurora can retrieve the raw suggestion as conversation, but there is no memory object that says "Aurora suggested Portishead; the user has not confirmed it." This is an authority-and-semantics gap, not a storage gap.
2. **Indirect retrieval is too lexical.** A current state is admitted to the prompt when it overlaps the query; a non-overlapping state can still enter only if it is scoped to the current session and fits the reserve (`PRIMORA/prime/context/manager.py:2526`, `2576`). Most personal facts are global, so a question such as "What should I listen to tonight?" does not overlap "The user loves Radiohead" and the preference can be missed. The semantic channel exists, but production has zero vectors, so it currently contributes nothing.

The third gap, extraction breadth, is real but secondary: deterministic patterns cover many explicit statements, while the broad `notes` fallback can over-capture. The right next step is measured extraction expansion, not a wholesale LLM memory layer.

### 2.3 Candidate verdicts

| Candidate | Verdict for M2 | Reason |
|---|---|---|
| **Mem0 OSS** | **Conditional / do not adopt as truth** | Closest deployment fit and strongest local vector-search story, but the current V3 add pipeline is ADD-only, agent-generated facts are first-class, and telemetry/cloud defaults must be disabled. It cannot own current truth without an authority layer. |
| **Graphiti** | **Reject for M2** | Strongest bi-temporal contradiction model, but requires a graph database, structured-output-capable LLM calls, and substantially more operational surface. Aurora already has validity windows and supersession. No measured problem justifies the cost at 8 GB VRAM. |
| **LangMem** | **Reject as a dependency; borrow concepts** | Useful memory-manager operations and consolidation ideas, but it is not a memory store or retriever, pulls the LangChain/LangGraph ecosystem, and its last release is old relative to its repository activity. |
| **Letta** | **Reject for M2** | A persistent-agent runtime, not an embeddable memory layer. Adoption would move Aurora into Letta's agent model and put identity continuity at risk. Its Docker server is documented as no longer actively maintained. |
| **PRIMORA** | **Retain and adapt** | Best fit on authority, provenance, local-first operation, data safety, and integration. Binding weaknesses are recommendation semantics, indirect retrieval, and extraction breadth. |

### 2.4 Recommended sequence

1. Add a first-class recommendation/suggestion record with speaker, provenance, status, and an explicit "not a user preference until accepted" rule.
2. Canonicalize preference concepts across `favorite artist`, `preference`, and keyed `likes` phrasings, and add a compact stable-preference admission path so indirect questions can reach known global preferences without a model call.
3. Add a six-case regression suite before changing extraction.
4. Run the fixed evaluation in §9.2. Only if native changes miss the pre-registered thresholds should a Mem0 shadow extractor be considered.
5. Do not install, dual-write, or migrate anything as part of M2.

---

## 3. Requirements and six cases

### 3.1 Functional requirements

Aurora's memory must answer five different questions without conflating them:

| Question | Required distinction |
|---|---|
| What did the user actually say? | Raw history, immutable, attributable. |
| What is currently true about the user? | User-authority only, with provenance and validity. |
| What did Aurora suggest or infer? | Assistant-origin, provisional, never silently current truth. |
| What changed or was contradicted? | Supersession, unresolved conflict, or explicit retraction. |
| What is relevant right now? | Retrieval and admission that can handle indirect language. |

The six evaluation cases below operationalize those questions.

### 3.2 The six cases

**Case 1 — Explicit preference versus assistant recommendation.**
The user says "I love Radiohead." Aurora says "You might like Portishead." The system must retain Radiohead as a user preference and Portishead as an unconfirmed suggestion, not as a user preference. A later music question may surface both, but must label the suggestion correctly.

**Case 2 — Factual accuracy and provenance.**
The user says their favorite artist is Radiohead. A later assistant statement, web snippet, or hallucinated summary says the user loves Portishead. The user statement remains authoritative. The conflicting claim is recorded as contested or ignored, never silently overwritten, and retrieval must not present it as established.

**Case 3 — Correction and changing preference.**
The user says "Actually, my favorite artist is Portishead now," or "I don't like Radiohead anymore." The current value changes, the old value remains in history with an end time, and the system does not retain two competing "current" values for the same concept.

**Case 4 — Natural, indirect retrieval.**
The user asks "What should I listen to tonight?" or "Recommend a book." The system should use known preferences even when the query does not contain the stored words "Radiohead" or "favorite artist." Retrieval must be relevant, not merely recent.

**Case 5 — Useful cross-conversation learning.**
In one session the user mentions living in Berlin, a timezone, and a favorite artist. In a later session the user asks for a weekend plan. The system should use the home location, timezone, and preferences without requiring the user to repeat them, while preserving provenance.

**Case 6 — Truthful uncertainty.**
The user asks "What is my favorite album?" when no album was stated, or when two sources conflict. The system must say it does not know or must label the conflict, rather than inventing a plausible answer or treating an assistant suggestion as user testimony.

### 3.3 Non-functional requirements

- Local and offline by default.
- No production memory leaves the machine.
- No destructive schema migration.
- Reversible changes with an isolation path.
- Bounded latency and bounded memory use.
- Deterministic behavior where practical.
- A measurable improvement over the current baseline before adoption.
- Documentation and tests updated with any behavior change.

---

## 4. Current PRIMORA baseline

### 4.1 Architecture snapshot

| Layer | Implementation | Evidence |
|---|---|---|
| Raw history | Append-only SQLite messages with idempotent per-message fingerprint; FTS5 index | `store.py`, `messages` and `messages_fts` tables |
| Structured current state | `states` and `candidates`; concept/value/type/scope/authority/confidence/validity; `source_ids` | `models.py:60`, `state.py` |
| Formation | Deterministic patterns and slot registry; user-only current facts; assistant conflicts recorded as unresolved | `state.py:570`, `state.py:1329`, `manager.py:1441` |
| Correction/update | Newest user value wins; old rows retained; explicit correction prefixes and retractions | `state.py:634`, `state.py:1222` |
| Retrieval | Hybrid lexical + semantic + M5 structural channels; relevance/recency ranking; hard admission rules | `hybrid.py`, `manager.py:2520` |
| Digests | Extractive summaries, no LLM invention | `digest.py` |
| Working memory | Runtime-only task state, not persisted | `working.py` |
| Isolation | Temporary stores and production-path refusal for harnesses | `isolation.py` |
| Replaceable seam | `MemoryProvider.retrieve` protocol, currently unused | `models.py:433` |

### 4.2 Production evidence

Read-only fingerprint on 2026-10-19:

```text
8b84160c3d8b7d0f2a04ffab021cc4bba214d5285a884ad14b78c0eb6c788dc1
```

| Table | Rows |
|---|---:|
| messages | 268 |
| sessions | 1 |
| digests | 7 |
| states | 14 |
| candidates | 6 |
| entities | 0 |
| entity_aliases | 0 |
| relationships | 0 |
| retrieval_vectors | 0 |
| retrieval_index_meta | 5 |
| messages_fts | 268 |

State distribution:

- Status: current 7, superseded 2, unresolved 5.
- Scope: default 5, global 9.
- Type: empty 5, FACT 6, PREFERENCE 3.

Last database write observed: 2026-10-19 12:11:20 PM. The earlier baseline digest `8ac6a53...` with 192 messages is stale; the database advanced through normal use, not through this evaluation.

Implications:

- M4 states are active.
- M5 entity/relationship machinery exists but has produced no production rows.
- The semantic index has no production vectors; the semantic channel does not contribute in the observed state.
- The five unresolved states are consistent with the assistant-conflict path being active.
- Claims about graph or semantic retrieval are therefore architectural potential, not observed production behavior.

### 4.3 What works well

- **Authority:** only `authority == "user"` states are admitted as current facts (`manager.py:2530`).
- **Provenance:** `source_ids` links a fact to the message that established it.
- **History:** superseded states remain; validity windows preserve when a fact was true.
- **Conflict handling:** assistant establishment claims that contradict a current fact become unresolved, not current (`state.py:1329`).
- **Uncertainty rendering:** unresolved states are injected only when relevant and are labeled "Possible conflict (not established)" (`manager.py:2606`).
- **Hedged statements:** statements containing hedge cues do not auto-confirm; they become pending candidates (`state.py:682`).
- **Anaphora:** elliptical answers can resolve against the immediately preceding question through the slot registry.
- **Duplicate control:** current-state reads keep the newest row per scope/concept and report duplicates.
- **Isolation:** experimental harnesses cannot silently resolve to the production database.

### 4.4 Gaps

| ID | Gap | Consequence | Evidence |
|---|---|---|---|
| G1 | Assistant recommendations are not a structured memory type | Aurora cannot distinguish "I suggested this" from "the user likes this" in memory; only raw text survives | Assistant path only calls `check_assistant_conflict` (`manager.py:1460`); no recommendation type |
| G2 | Global preference admission requires lexical overlap | Indirect questions can miss known preferences even though they are stored | `manager.py:2576` skips non-overlapping global states |
| G3 | Semantic channel has zero production vectors | No measured semantic recall; hybrid retrieval is effectively lexical/structural | Production `retrieval_vectors = 0` |
| G4 | Preference schemas can diverge | "Favorite artist" and "I prefer..." can land on different concepts, so a correction may not supersede the earlier fact | Slot-first routing (`state.py:618`) plus `preference` concept in `state.py:354` |
| G5 | Broad `notes` fallback can over-capture | Unmatched "I ..." statements become auto-confirmed global notes up to 20 | `state.py:629`, `slots.json` notes slot |
| G6 | Extraction is pattern-bound | Paraphrases, implied preferences, and novel attributes can be missed | Deterministic patterns only |
| G7 | No six-case benchmark exists | Improvements cannot be compared against a frozen baseline | Test inventory reviewed; no recommendation/indirect-retrieval suite |
| G8 | M5 and semantic code are ahead of production use | Documentation and operational assumptions can overstate actual capability | Zero entities/relationships/vectors |
| G9 | Documentation drift | `MEMORY_HYBRID_RETRIEVAL.md` describes an earlier state; hybrid is live by default | `manager.py:197` |

### 4.5 Test evidence

Representative existing coverage includes:

- `TESTS/prime/test_memory_correctness.py` — explicit factual/preference formation.
- `TESTS/prime/test_memory_lifecycle.py` — supersession, retraction, and lifecycle.
- `TESTS/prime/test_memory_fact_collisions.py` — same-topic collisions.
- `TESTS/prime/test_memory_queries.py` — current/superseded query behavior.
- `TESTS/prime/test_state_anaphora.py` — elliptical answers.
- `TESTS/prime/test_session_state.py`, `test_session_scope.py` — scope isolation.
- `TESTS/prime/test_retrieval_hybrid.py` — channel and fusion behavior.
- `TESTS/prime/test_retrieval_live.py` — live retrieval path.
- `TESTS/prime/test_memory_write_path_uniqueness.py` — duplicate-write control.
- `TESTS/prime/test_production_isolation_regression.py` — production-path refusal.
- `TESTS/prime/test_unbacked_claim.py` — truthfulness around unsupported claims.

What is covered is strong for recognized statements and conflict safety. What is not covered is the exact combination that matters most here: an assistant suggestion followed by an indirect request, and a cross-schema preference correction.

---

## 5. Candidate evaluations

### 5.1 Mem0 OSS

**What it is.** A memory layer for agents with an add/search API, a vector store, metadata, and optional graph memory. It can run as a Python library or as a self-hosted server.

**Version and license (checked 2026-10-19).**

- Package: `mem0ai` 2.2.1, released 2026-09-25.
- License: Apache-2.0.
- Python: >=3.10, <4.0.
- Repository commit: `0724291bd864cf807e88ee8981c8226644e8fb42` (2026-10-14).
- Core dependencies include `httpx`, `openai`, `posthog`, `protobuf`, `pydantic`, `pytz`, `qdrant-client`, and `sqlalchemy`.

**Extraction and update semantics.** The current V3 add pipeline is documented as a single-pass, ADD-only extraction: one LLM call, no automatic UPDATE or DELETE. Memories accumulate; nothing is overwritten by the add path. Explicit update, delete, and history APIs exist, so an application can implement corrections itself. Documentation also describes agent-generated facts as first-class. The docs acknowledge that knowledge updates are the hardest category and that stale or contradictory facts can surface.

This is the decisive architectural mismatch. Aurora's core invariant is "newest user statement supersedes the old current value." Mem0's default add path does the opposite: it preserves both, leaving conflict resolution to the caller.

**Local operation.** The library can use a local Qdrant instance and an OpenAI-compatible local endpoint. LM Studio, Ollama, and vLLM are documented paths. The self-hosted server uses Postgres with pgvector by default and adds dashboard/API-key/audit features. Graph memory is documented as a platform feature rather than an OSS feature. Telemetry is on by default through PostHog; `MEM0_TELEMETRY=False` disables it.

**Strengths for Aurora.**

- Strong semantic extraction and vector recall.
- Local deployment is genuinely possible.
- Active project with a large ecosystem.
- Explicit update/delete/history operations exist for an authority layer to call.
- A library integration is smaller than a graph database or a new agent runtime.

**Limits for Aurora.**

- Automatic ADD-only behavior conflicts with supersession.
- Assistant-generated facts are not subordinated to user authority.
- Defaults are cloud-oriented (OpenAI, telemetry), so safe operation depends on configuration discipline.
- Requires an LLM and an embedding model for each add/search path, adding latency and VRAM contention.
- A parallel memory store creates dual-write and synchronization risk.
- Graph memory is not available in the OSS path.
- The API has changed across versions; pinning and migration testing are required.

**Fit verdict.** The best external deployment fit, but not a source-of-truth replacement. Use only as a shadow extractor or behind a retrieval-only seam after the experiment in §9.2.

### 5.2 Graphiti

**What it is.** A temporal context-graph framework. It ingests episodes, extracts entities and relationships, and maintains bi-temporal edges so facts can be valid, invalidated, or historically queried.

**Version and license (checked 2026-10-19).**

- Package: `graphiti-core` 0.31.1, released 2026-10-15.
- License: Apache-2.0.
- Python: >=3.10, <4.0.
- Repository commit: `7b0182f94d27075db488636cd5456981ee10b987` (2026-10-19).
- Core dependencies include `neo4j>=5.26.0`, `numpy`, `openai`, `posthog`, `pydantic`, `python-dotenv`, and `tenacity`.

**Architecture.** Graphiti stores episodes, entities, and temporal edges in a graph database. It supports hybrid search over semantic, keyword, and graph structure, and it tracks when facts became valid or invalid. The project is associated with the Zep temporal-knowledge-graph paper (arXiv 2501.13956).

**Local operation.** Supported graph backends include Neo4j 5.26+, FalkorDB, and Neptune/OpenSearch. Neo4j Community is GPLv3; FalkorDB is SSPLv1. Community reports indicate APOC is needed for Neo4j setups. Local Ollama is possible through `OpenAIGenericClient` with a custom base URL, but structured-output failures are reported with generic OpenAI-compatible clients. Community reports place the practical local-model floor around 27–32B; an 8B Qwen model is therefore a serious reliability risk for extraction and schema-constrained output.

**Strengths for Aurora.**

- Best temporal model of the candidates: explicit validity and invalidation.
- Strong hybrid retrieval over entities and relationships.
- Good fit for multi-entity, multi-hop questions if those become a measured need.
- Episode provenance is preserved.

**Limits for Aurora.**

- Requires a separate graph database and its operational lifecycle.
- Multiple LLM calls per episode are likely; exact count and latency were not measured.
- Structured output is central; a small local model is a weak foundation.
- Hardware cost is high relative to an 8 GB VRAM / 16 GB RAM envelope.
- Source provenance is not the same as user authority; an assistant-originated episode can still become a graph fact.
- Aurora already has temporal validity and supersession at the state level; the incremental value is unproven.
- Integrating it would create a second memory model and a second retrieval semantics.

**Fit verdict.** Reject for M2. Revisit only if a measured multi-entity temporal-reasoning requirement appears and the hardware envelope expands.

### 5.3 LangMem

**What it is.** A set of memory-management primitives for LangChain/LangGraph agents. `create_memory_manager` produces a `Runnable` that extracts and consolidates memories using a chat model; it supports inserts, updates, and deletes, and uses `trustcall` for structured tool calls.

**Version and license (checked 2026-10-19).**

- Package: `langmem` 0.0.30, released 2025-10-27.
- License: MIT.
- Python: >=3.10.
- Repository commit: `48e3c11f5bb527282c7d5339c6a87a0b35abccfc` (2026-10-02).
- Dependencies include `langchain-anthropic`, `langchain-core`, `langchain-openai`, `langchain`, `langgraph-checkpoint`, `langgraph<2,>=0.6.0`, `langsmith`, and `trustcall`.

**Architecture.** LangMem does not ship a database or a retriever. It provides memory operations over a store you supply. Defaults enable inserts and updates and disable deletes. The manager is model-driven and schema-driven; the quality of extracted/updated memories depends on the model's structured tool-calling ability.

**Strengths for Aurora.**

- Clear separation between memory management and storage.
- Explicit insert/update/delete operations are a useful conceptual model.
- A Pydantic schema can distinguish fact types and provenance fields.
- No database lock-in if a custom store is used.

**Limits for Aurora.**

- Not a complete memory system: no retrieval, no authority model, no temporal validity.
- Pulls the LangChain/LangGraph/LangSmith ecosystem even if only the manager is used.
- Last release is old relative to repository activity; API stability is not established.
- Small local models may struggle with the required structured tool calls.
- Deletes are off by default, so conflict/retraction behavior still needs application logic.
- Using it would add a second agent framework around Aurora's existing loop.

**Fit verdict.** Reject as a dependency. Borrow the explicit-operation and consolidation concepts for the native design if needed.

### 5.4 Letta

**What it is.** A stateful agent runtime with memory blocks, archival/recall memory, tool-driven self-editing, and MemFS: a git-backed memory filesystem that projects memory into versioned markdown files.

**Version and license (checked 2026-10-19).**

- Package: `letta` 0.34.12, released 2026-10-12.
- License: Apache-2.0.
- Python: >=3.9.
- Repository commit: `8816d3a8516893bfdc0fa14bfb1235226012e295` (2026-10-18).
- Dependency metadata could not be verified from the repository's current `pyproject.toml` path (404 after repository restructuring).

**Architecture.** Letta is an agent server/runtime, not an embeddable memory library. Memory blocks are part of the agent state; archival and recall memory support search; MemFS stores memory as markdown files with git history. The local backend stores memories in local git repositories; hosted repositories are a cloud feature.

**Local operation.** Letta Code has a local mode. The Docker server is documented as no longer an actively maintained or supported product surface. Self-hosting documentation references Postgres with pgvector. The agent harness is demanding; community reports indicate it performs best with strong models and is unlikely to be reliable with most small open models.

**Strengths for Aurora.**

- Persistent agent memory across sessions.
- Versioned, inspectable MemFS files.
- Memory blocks are a useful human-readable abstraction.
- Strong tooling around agent state.

**Limits for Aurora.**

- Adoption means moving Aurora into Letta's runtime/agent architecture, not adding a component.
- That is a direct risk to identity continuity and the existing Prime/Aurora loop.
- Local model performance is a major uncertainty.
- Docker server support is explicitly de-emphasized.
- Postgres and an agent runtime increase operational and hardware cost.
- Agent self-editing is less deterministic than Aurora's user-authority state model.
- The feature set solves a broader problem than Aurora has.

**Fit verdict.** Reject for M2. Borrow only the general idea of versioned, path-addressed memory files if a future audit requirement justifies it; do not adopt the runtime.

### 5.5 Existing PRIMORA

**Strengths.**

- Correct authority model for Aurora's stated goals.
- Provenance and history are built in.
- Local, deterministic, and cheap to run.
- Existing tests cover corrections, collisions, lifecycle, retrieval, scope, and isolation.
- A dormant `MemoryProvider` seam exists for future replaceable retrieval.
- Production data proves the core path is used, not merely present.

**Weaknesses.**

- No first-class recommendation memory.
- Conservative lexical admission misses indirect retrieval.
- Semantic/M5 infrastructure is not active in production.
- Extraction is pattern-bound and the notes fallback is broad.
- No frozen six-case benchmark.
- Documentation drift around hybrid retrieval.

**Fit verdict.** Best fit. The correct move is targeted evolution, not replacement.

---

## 6. Decision matrix

### 6.1 Criteria

Scores are 1–5, where 5 is best fit for Aurora under the local deployment envelope.

1. Extraction coverage.
2. Provenance and authority.
3. Preference/recommendation semantics.
4. Update and conflict handling.
5. Retrieval quality.
6. Temporal history.
7. Local-model compatibility.
8. Privacy and offline operation.
9. Hardware cost.
10. Latency.
11. Integration cost.
12. Data safety and reversibility.
13. Maintenance burden.
14. Demonstrable value for Aurora.

### 6.2 Matrix

| # | Criterion | PRIMORA | Mem0 | Graphiti | LangMem | Letta |
|---|---|---:|---:|---:|---:|---:|
| 1 | Extraction coverage | 3 | 4 | 5 | 4 | 4 |
| 2 | Provenance / authority | 5 | 2 | 3 | 2 | 2 |
| 3 | Preference / recommendation semantics | 2 | 3 | 3 | 3 | 2 |
| 4 | Update / conflict handling | 4 | 2 | 5 | 4 | 3 |
| 5 | Retrieval quality | 3 | 4 | 5 | 2 | 4 |
| 6 | Temporal history | 5 | 3 | 5 | 2 | 3 |
| 7 | Local-model compatibility | 5 | 3 | 2 | 3 | 1 |
| 8 | Privacy / offline | 5 | 3 | 3 | 3 | 3 |
| 9 | Hardware cost | 5 | 4 | 1 | 4 | 2 |
| 10 | Latency | 5 | 2 | 1 | 2 | 1 |
| 11 | Integration cost | 5 | 3 | 1 | 3 | 1 |
| 12 | Data safety / reversibility | 5 | 3 | 2 | 3 | 2 |
| 13 | Maintenance burden | 4 | 3 | 2 | 3 | 1 |
| 14 | Demonstrable value for Aurora | 3 | 2 | 2 | 2 | 1 |
|  | **Descriptive average** | **4.2** | **2.9** | **2.9** | **2.9** | **2.1** |

The averages are descriptive only. They must not be used as a decision rule, because several criteria are hard constraints. A system that scores well on extraction but cannot fit the hardware or preserve user authority is not a viable candidate.

All candidate scores are architecture/documentation assessments; no candidate was run locally. Scores marked by uncertainty are discussed below.

### 6.3 Decisive scores

**PRIMORA: authority 5, temporal 5, local 5, integration 5, data safety 5.**
These scores reflect the existing invariants and the fact that the system is already the source of truth. The low scores are equally important: preference/recommendation semantics 2, extraction 3, retrieval 3, and demonstrable value 3.

**Mem0: extraction 4, retrieval 4, but update 2 and authority 2.**
The ADD-only default is an architectural mismatch, not a prompt-tuning problem. Its vector retrieval is attractive, but it cannot be the current-truth store without an application-level authority and supersession layer. The extraction score is unverified on Qwen 8B.

**Graphiti: update 5, temporal 5, retrieval 5, but hardware 1, latency 1, integration 1.**
This is the strongest temporal model and the weakest operational fit. The graph database and structured-output requirements are hard costs. The score is not a criticism of Graphiti's design; it is a statement about Aurora's envelope.

**LangMem: update 4, but retrieval 2, provenance 2, temporal 2.**
It is a manager, not a memory system. It could inform a native design, but it cannot carry Aurora's truth, retrieval, or temporal requirements on its own. The update score is model-dependent and unverified.

**Letta: retrieval 4, but local 1, integration 1, maintenance 1.**
The runtime is the product. Adopting it is an architecture migration, not a memory upgrade. The local-model and Docker-support risks are decisive.

### 6.4 Hard-constraint verdicts

| Constraint | Mem0 | Graphiti | LangMem | Letta | PRIMORA |
|---|---|---|---|---|---|
| Runs fully local on the target envelope | Yes, with configuration | Doubtful at 8B; graph DB adds load | Yes, in-process | Doubtful | Yes |
| Preserves user-only authority | No, not first-class | No, not first-class | No | No | Yes |
| Provides deterministic corrections | No automatic | Yes, temporal | Model-dependent | Agent-dependent | Yes for recognized statements |
| Handles indirect retrieval | Yes, vector | Yes, hybrid graph | No | Yes, archival | Partial |
| Avoids a second source of truth | No | No | N/A (no store) | No | Yes |
| Reversible within an experiment | Yes, isolated | Yes, isolated | Yes, isolated | Harder | Yes |
| Operational cost acceptable | Medium | High | Low–medium | High | Low |

### 6.5 Unknowns affecting scores

- Mem0's extraction and update behavior on Qwen 8B.
- Mem0's exact OSS/platform boundary for graph and advanced memory features.
- Graphiti's actual RAM, LLM-call count, and structured-output failure rate on 8B.
- LangMem's structured tool-calling reliability on the local model and its compatibility with the current release gap.
- Letta's current dependency set and local-runtime performance.
- Whether a production semantic-index backfill would materially improve PRIMORA retrieval, and at what latency.
- Whether the local embedding model can coexist with the 8B chat model in 8 GB VRAM.
- PRIMORA's repository license status; no root license file was found.

---

## 7. Six-case analysis

### Case 1 — Explicit favorite versus assistant recommendation

**Scenario.** User: "I love Radiohead." Aurora: "You might like Portishead." Later: "What music should I put on?"

**Required behavior.** Radiohead is a user preference. Portishead is an assistant suggestion. A recommendation may use both, but must not say the user likes Portishead.

**PRIMORA now.** "I love Radiohead" is captured through the `likes` slot as a global preference. The assistant message is not parsed as a recommendation; only establishment-like claims are checked for conflict. The suggestion survives as raw text. Retrieval is the weak point: "music" does not overlap "loves Radiohead," and a global fact does not get the current-session reserve, so the preference may not be surfaced at all. The authority is right; the representation and retrieval are incomplete.

**Mem0.** Both turns can be added. LLM extraction may create a user preference and an assistant-origin fact. Because agent-generated facts are first-class and the add path is ADD-only, the suggestion can persist alongside the preference with no structural "unconfirmed" status. Retrieval is likely better than PRIMORA's lexical gate, but attribution is weaker.

**Graphiti.** Episodes can carry provenance, and temporal edges can represent suggestions, but there is no first-class user-versus-assistant authority invariant. It would require schema and prompt work plus a graph database.

**LangMem.** A manager could extract both statements, and a custom schema could add an origin field, but there is no store or retrieval. It is a partial building block, not a solution.

**Letta.** An agent could remember the suggestion in a block or MemFS file, but the behavior is agent-driven and less deterministic.

**Verdict.** PRIMORA is strongest on authority and weakest on recommendation representation. Add a recommendation record; do not use an external store to compensate.

### Case 2 — Wrong artist and provenance

**Scenario.** User: "My favorite artist is Radiohead." Later an assistant claim says "The favorite artist is Portishead" (a parseable establishment) or "You love Portishead" (not a parseable establishment).

**Required behavior.** The user statement remains current. The conflicting claim is contested or ignored, never silently promoted. Retrieval labels the conflict.

**PRIMORA now.** Assistant text that parses as an establishment and contradicts a current fact is written as an unresolved state with `authority="assistant"` and confidence 0.3 (`state.py:1329`). It is never current. The build path injects unresolved conflicts only when relevant and labels them "Possible conflict (not established)" (`manager.py:2606`). Claims that do not match the establishment parser remain raw text and are not truth. This is the strongest behavior among the candidates.

**Mem0.** The claim can be added as a memory. With ADD-only behavior, the old and new statements coexist; nothing marks the assistant claim as subordinate. This is a direct risk of stale or contradictory facts surfacing.

**Graphiti.** Temporal invalidation can represent the contradiction and preserve history, but the graph does not encode user-only authority by default. It is better than Mem0 on temporal reasoning, but still requires an application-level authority policy.

**LangMem.** The manager may update or delete a memory based on the assistant claim, depending on the prompt. There is no first-class authority model.

**Letta.** The agent may self-edit its memory. That is flexible but not a guarantee of user authority.

**Verdict.** PRIMORA wins. Preserve this behavior in any change and add explicit tests for non-parsing assistant claims, because raw claims can still influence a model if retrieved without labels.

### Case 3 — Correction and changing preference

**Scenario A.** "Actually, my favorite artist is Portishead now."
**Scenario B.** "I don't like Radiohead anymore."
**Scenario C.** "I prefer Portishead."

**Required behavior.** The current value changes; the old value remains historical; no duplicate current truth.

**PRIMORA now.** Recognized slot corrections work: correction prefixes are unwrapped, the newest user value wins, and old rows are superseded with validity windows. Explicit rule retirements and invalidations are supported. Hedged corrections remain pending. The gap is schema divergence: a statement phrased as `favorite artist` may land in a learned-favorite slot, while `I prefer ...` lands in the generic `preference` concept. A correction across those phrasings may not target the original fact, producing parallel memories rather than one superseded chain. This is a canonicalization problem, not a reason to adopt a framework.

**Mem0.** The ADD-only path will keep both the old and new preference unless the application calls an explicit update/delete. History exists, but automatic current truth does not.

**Graphiti.** Bi-temporal invalidation is the strongest generic answer. It can close the old edge's validity and keep the history. The cost is the graph stack and structured-output risk.

**LangMem.** Insert/update/delete operations can model the change, but correctness depends on the model and prompt, and there is no temporal store by default.

**Letta.** Agent edits can change memory and MemFS retains versions, but the behavior is less deterministic.

**Verdict.** PRIMORA is good for recognized corrections and should be extended with cross-schema canonicalization. Graphiti's model is conceptually strong but operationally disproportionate. This case does not justify a graph database.

### Case 4 — Natural, indirect retrieval

**Scenario.** "What should I listen to tonight?" or "Recommend a book for the weekend."

**Required behavior.** Use stored preferences even without lexical overlap, rank them above irrelevant recent chatter, and preserve provenance.

**PRIMORA now.** Hybrid retrieval exists, but production has zero semantic vectors. More importantly, the current-state admission rule gives a state a place without overlap only when it is scoped to the current session and fits the reserve (`manager.py:2576`). Most personal preferences are global, so "music" does not overlap "loves Radiohead" and the preference is skipped. The result is a real indirect-retrieval gap, independent of whether the semantic index is enabled.

**Mem0.** Vector search over memory text is the natural fit for this case. With a local embedder, it could retrieve Radiohead for a music query. The risk is that it may also retrieve assistant suggestions as if they were user facts.

**Graphiti.** Hybrid semantic and graph search is strong here, especially for entity-linked queries, but the operational cost remains.

**LangMem.** No retrieval. It would need an external vector store and query layer, at which point the retrieval is not LangMem's contribution.

**Letta.** Archival/recall search can handle this, but it comes with the runtime migration.

**Verdict.** This is the case where external systems have a genuine advantage. The smallest fix is not a framework: add a compact stable-preference admission path and deterministic alias/query expansion, or activate a validated local semantic index. Measure before choosing.

### Case 5 — Useful cross-conversation learning

**Scenario.** Session 1: "I live in Berlin, my timezone is CET, and I love Radiohead." Session 2: "Plan a weekend."

**Required behavior.** Use home location, timezone, and preferences across sessions without repetition; keep provenance and scope correct.

**PRIMORA now.** Personal facts are global by design and survive sessions. The current-state read includes global states. The weakness is the same lexical admission rule: "weekend" does not overlap Berlin, CET, or Radiohead, so the facts may not be admitted. Session metadata may provide some context, but preferences and home location can be missed. The storage model is correct; the activation rule is too conservative for broad requests.

**Mem0.** Global user-scoped vector search is likely to recall these facts. Authority remains the caveat.

**Graphiti.** Entity/relationship retrieval could connect Berlin, timezone, and music, but again at high operational cost.

**LangMem.** No storage or retrieval, so it does not solve the case.

**Letta.** Persistent memory and archival recall can solve it, but only by adopting the runtime.

**Verdict.** PRIMORA has the right storage and the wrong activation threshold. A small, bounded "stable preferences always available" block or semantic retrieval would close the gap without a framework.

### Case 6 — Truthful uncertainty

**Scenario.** "What is my favorite album?" when no album was stated, or when Radiohead and Portishead claims conflict.

**Required behavior.** Say "you haven't told me" or label the conflict; do not invent a fact or treat a suggestion as testimony.

**PRIMORA now.** The facts heading explicitly instructs the model to use only listed facts and to say it does not know when a fact is absent. Current states are user-authority only. Unresolved conflicts are injected only when relevant and labeled as not established. This is the strongest uncertainty behavior among the candidates.

**Mem0.** There is no first-class uncertainty state. ADD-only memory can surface stale or contradictory items, and the model may assert one. The docs acknowledge knowledge updates as a hard category.

**Graphiti.** Temporal validity can prevent an invalid edge from being current, but the retrieval layer still needs an uncertainty policy. It is better than Mem0, weaker than PRIMORA's explicit user-authority invariant.

**LangMem.** No truth model. It manages memories; it does not decide what is established.

**Letta.** Agent-driven memory can assert or rewrite. It is not a guarantee of truthful uncertainty.

**Verdict.** PRIMORA wins clearly. Any future change must preserve the "not in facts means say you do not know" behavior.

### 7.7 Case summary

| Case | PRIMORA | Mem0 | Graphiti | LangMem | Letta |
|---|---|---|---|---|---|
| 1. Preference vs suggestion | Partial: right authority, no recommendation object | Partial: good recall, weak attribution | Partial: provenance, no authority | Weak: manager only | Partial: agent-driven |
| 2. Wrong artist / provenance | Strong | Weak | Partial | Weak | Weak |
| 3. Correction / change | Partial–strong | Weak | Strong | Partial | Partial |
| 4. Indirect retrieval | Weak–partial | Strong | Strong | Not applicable | Strong |
| 5. Cross-session learning | Partial | Strong | Strong | Not applicable | Strong |
| 6. Truthful uncertainty | Strong | Weak | Partial | Weak | Weak |

No candidate wins every case. PRIMORA wins the cases that protect trust; external systems win the cases that depend on semantic retrieval. That is exactly why the recommendation is to keep the trust core and improve retrieval natively.

---

## 8. Licenses, privacy, dependencies, hardware, and integration

### 8.1 License and dependency assessment

| System | Core license | Version / commit | Python | Required services / deps | Separately licensed or external components | Local operation |
|---|---|---|---|---|---|---|
| PRIMORA | **No repository license file found** | HEAD `5bedac3` | Existing project | SQLite/FTS5; no memory service | None identified for memory; legal status unverified | Yes |
| Mem0 OSS | Apache-2.0 | `mem0ai` 2.2.1 / `0724291` | >=3.10,<4 | LLM + embedding provider; local Qdrant in library mode; SQLAlchemy, Postgres+pgvector in server mode | Platform features (including graph memory per current docs) separate; PostHog telemetry | Yes, with local providers |
| Graphiti | Apache-2.0 | `graphiti-core` 0.31.1 / `7b0182f` | >=3.10,<4 | Neo4j 5.26+, FalkorDB, or Neptune/OpenSearch; LLM; embeddings | Neo4j Community GPLv3; FalkorDB SSPLv1; Neptune proprietary; APOC reported by community | Possible, but constrained |
| LangMem | MIT | `langmem` 0.0.30 / `48e3c11` | >=3.10 | LangChain chat model; custom store; trustcall | LangChain/LangGraph/LangSmith ecosystem; LangSmith is a separate service if enabled | Yes, with custom store |
| Letta | Apache-2.0 | `letta` 0.34.12 / `8816d3a` | >=3.9 | Agent runtime; Postgres+pgvector for Docker; local git for MemFS local backend | Docker server no longer actively maintained; cloud repositories/hosting separate | Partial; Letta Code local mode |

Notes:

- Mem0's OSS core is Apache-2.0, but the platform and graph features are separate. Do not describe the platform as OSS.
- Graphiti's core license does not change the licenses of Neo4j Community (GPLv3) or FalkorDB (SSPLv1). Those are infrastructure obligations, not just Python dependencies.
- LangMem's MIT license is permissive, but its dependency graph is large and includes observability tooling that can send data externally if enabled.
- Letta's Apache-2.0 license is permissive, but the supported deployment surface is the issue, not the license.
- PRIMORA has no root `LICENSE`, `COPYING`, or `pyproject.toml` in the inspected tree. Its distribution license is therefore unverified. This should be resolved before any external distribution or contributor onboarding; it does not block internal evaluation.
- Versions and commits were checked on 2026-10-19. Pin exact versions and record hashes before any future experiment.

### 8.2 Privacy assessment

| Risk | Mem0 | Graphiti | LangMem | Letta | PRIMORA |
|---|---|---|---|---|---|
| Default cloud LLM | Yes (OpenAI defaults) | Yes (OpenAI defaults) | Depends on chosen model | Depends on chosen model | No |
| Telemetry on by default | Yes (PostHog; disable with `MEM0_TELEMETRY=False`) | README documents telemetry | LangSmith tracing if enabled | Depends on deployment | No memory telemetry observed in the context path |
| Memory data leaves the machine | Only if cloud provider/telemetry enabled | Only if cloud provider/telemetry enabled | Only if cloud model/tracing enabled | Possible with cloud features | No |
| Local-only mode supported | Yes, configuration-dependent | Yes, with local graph DB and local LLM | Yes, with local chat model and store | Partial | Yes |
| Production data should be used | No | No | No | No | Read-only only |

For any experiment, the safe rule is stricter than "disable telemetry": use synthetic data, no production memories, no external network, and an isolated store. Configuration mistakes are more likely than malicious behavior, and a default that phones home is a risk even when it can be disabled.

### 8.3 Hardware and latency assessment

The target envelope is 8 GB VRAM and approximately 16 GB RAM, already shared with an 8B chat model.

| System | Incremental memory risk | Incremental latency risk | Confidence |
|---|---|---|---|
| PRIMORA | Low: SQLite and deterministic Python | Low: no memory LLM call in the current path | Known |
| Mem0 | Medium: vector store + embedding model; LLM call on add | Medium–high: one LLM + embedding per add; embedding per search | Inferred; local run not performed |
| Graphiti | High: graph DB + LLM + embeddings | High: multiple LLM calls per episode; graph queries | Inferred; resource use unverified |
| LangMem | Low–medium: in-process manager | Medium: LLM call per memory operation | Inferred; model-dependent |
| Letta | High: agent runtime + Postgres/pgvector or local services | High: agent loop and memory tools | Inferred; local performance unverified |

Exact RAM, VRAM, and p95 latency were not measured. Any claim that a candidate "fits" is provisional until the preflight in §9.2 is run.

### 8.4 Integration surface

| System | Integration shape | Main coupling risk | Reversibility |
|---|---|---|---|
| PRIMORA | Native | None beyond current code | High |
| Mem0 | Library calls plus a parallel memory store | Dual-write divergence; authority inversion; schema/API drift | High if isolated |
| Graphiti | New graph service plus ingestion pipeline | Operational dependency; second retrieval semantics | Medium |
| LangMem | Manager calls plus custom store and LangChain model plumbing | Ecosystem coupling; no retrieval/authority | High if isolated |
| Letta | Runtime/agent migration | Identity, control flow, deployment, data migration | Low–medium |

---

## 9. Smallest useful adoption and experimental plan

### 9.1 Smallest useful adoption strategy

**Decision: adopt no candidate package as part of M2.** If an external comparison is required, the smallest useful slice is a bounded, isolated experiment with Mem0 OSS as a shadow extractor, not a memory store. Everything else is native work or concept borrowing.

Why this is the smallest defensible slice:

- It addresses a real deficiency: deterministic extraction may miss paraphrased preferences and corrections.
- It uses a local library rather than a graph database or agent runtime.
- It can be run entirely offline with synthetic data and telemetry disabled.
- It does not require a production schema change, dual-write, or migration.
- It can be discarded without touching Aurora's truth model.

Why Mem0 rather than LangMem for the experiment:

- Mem0 is a complete local memory library with extraction and retrieval; LangMem is only a manager and would still need a store/retriever.
- Mem0's ADD-only behavior is a known hazard, which makes it a useful worst-case comparison for authority and duplication.
- Mem0 has an active release line and documented local-provider support.
- The experiment is about extraction/recall, not about adopting the framework.

Reuse versus native implementation:

- For authority, provenance, uncertainty, and recommendation semantics, reuse cannot beat a native fix: no candidate models user-only current truth as a first-class invariant. Native work is required regardless.
- For indirect retrieval, a native alias/stable-preference path is cheaper and safer than adopting a second store. A validated semantic index is the only external-style upgrade worth considering, and it is already scaffolded in PRIMORA.
- For extraction recall, an external LLM extractor could be better than deterministic patterns. That is the one place where reuse might have a real advantage, so it is the only place the external slice is aimed.
- If a candidate's value is only its prompt or schema, adopt that pattern natively rather than adding the dependency. The experiment therefore compares patterns as well as packages.

Scope of the slice:

1. Build a synthetic dataset covering the six cases and negative controls.
2. Run the current deterministic PRIMORA extractor as the baseline.
3. Run Mem0 locally with the same 8B model and a local embedder, `MEM0_TELEMETRY=False`, and an isolated Qdrant directory.
4. Compare extracted facts/preferences and retrieval results. Do not copy Mem0 output into production.
5. If Mem0 materially improves recall/precision without violating hard gates, adopt the **prompt/schema pattern** natively first. Adopt the package only if there is a separate, approved integration decision.

What must not happen:

- No Mem0 store becomes the current-truth store.
- No assistant-generated Mem0 fact becomes a user preference without an explicit user statement.
- No production memories are added to Mem0.
- No background dual-write.
- No dependency is added to the main environment during the experiment.

Rollback and isolation:

- Use a temporary database outside the production directory.
- Use `harness_session`/`production_guard` patterns already present in `isolation.py`.
- Keep the experiment in a separate virtual environment or disposable container.
- Remove the test store and configuration after the run.
- The production database fingerprint must be identical before and after.

Evidence required before any package adoption:

- All hard gates in §9.2 pass.
- Mem0 beats the baseline on the target extraction metrics by the pre-registered margin.
- The improvement cannot be achieved more cheaply by native patterns or a prompt/schema change.
- The integration can be retrieval-only and cannot become a second source of truth.
- A rollback plan is documented and tested.

**Native alternatives that should be tried first.**

| Gap | Native option | Why it may be enough | Risk |
|---|---|---|---|
| G1 recommendations | Add a recommendation/suggestion record linked to the originating assistant message and the user preference it relies on | Directly models the missing distinction; no new dependency | Schema and rendering changes; must not become current truth |
| G2 indirect retrieval | Add a compact stable-preference admission block and deterministic query aliases | Fixes the lexical gate without a model; cheap and testable | Prompt budget; relevance ordering |
| G3 semantic recall | Backfill/validate the existing semantic index, or keep it disabled and use aliases | Existing code already supports the channel; no new framework | Embedding model may not coexist with 8B in 8 GB VRAM |
| G4 schema divergence | Canonicalize `favorite <noun>`, `preference`, and `likes` into a shared concept family | Prevents parallel truth and makes corrections reliable | Requires careful migration of existing rows |
| G5 notes over-capture | Tighten fallback rules and require confirmation for unmatched statements | Reduces unsupported memories without an LLM | Could reduce recall; measure |

### 9.2 Experimental plan

**Status:** proposed, not executed.

**Objective.** Determine whether any candidate, or a bounded native change, improves Aurora's six-case behavior without violating authority, privacy, hardware, or reversibility constraints.

**Hypotheses.**

- H1: PRIMORA's authority and uncertainty behavior is already superior; the measurable gaps are recommendation representation and indirect retrieval.
- H2: A compact native preference-admission path closes most of the indirect-retrieval gap without an LLM.
- H3: Mem0's LLM extraction may improve recall on paraphrases, but its ADD-only default will increase stale/duplicate memory unless an update layer is added.
- H4: Graphiti's temporal model is strong but cannot fit the target envelope within an acceptable setup/latency budget.
- H5: LangMem and Letta do not solve Aurora's retrieval-plus-authority problem at acceptable integration cost.

**Dataset.**

- Synthetic only. No production memories.
- Minimum 72 scenarios: 12 per case category; at least 12 of the total are negative controls where the correct behavior is "do not store" or "say you do not know."
- Each scenario contains 2–5 turns across one or two sessions.
- Each scenario is stored as JSONL with:
  - `id`, `category`, `turns`
  - `expected_operations` (ADD, UPDATE, SUPERSEDE, RETRACT, UNRESOLVED, NOOP)
  - `expected_current_facts` with concept/value/authority
  - `expected_retrieval` with target memory IDs or fact IDs
  - `expected_answer_constraints` (must mention, must not assert, must label uncertainty)
  - `provenance_expectation`
  - `notes`
- Include paraphrases, negations, hedges, assistant suggestions, corrections, duplicates, cross-session references, and conflicting claims.
- Freeze the dataset and expected labels before running any candidate.

**Conditions.**

| Condition | Configuration |
|---|---|
| B0 | Current PRIMORA, semantic channel off |
| B1 | Current PRIMORA, semantic channel on if a local index can be built |
| N1 | Native recommendation record + native preference admission |
| M1 | Mem0 OSS local, telemetry off, isolated Qdrant, same model/embedder |
| L1 | LangMem manager with a custom in-memory store, same model |
| G1 | Graphiti local only if preflight passes; otherwise report "not tested" |
| T1 | Letta conceptual review only; run only if preflight passes and identity risk is accepted |

**Controlled variables.**

- Same Qwen 8B model, quantization, context size, temperature (0 where possible), and system prompt.
- Same hardware and power profile.
- Same scenario order and seeds; record model hash and server flags.
- No external network. Localhost only.
- No LangSmith tracing, no PostHog, no telemetry.
- Same time budget and maximum turns.

**Metrics.**

| Dimension | Metrics |
|---|---|
| Extraction | Precision, recall, F1 per category and macro |
| Attribution | User-vs-assistant accuracy; recommendation-as-preference false-positive rate |
| Update | Correction/supersession accuracy; stale-current rate; duplicate rate |
| Retrieval | Recall@k, MRR, answer groundedness, irrelevant-memory rate |
| Uncertainty | No-evidence assertion rate; conflict-labeling accuracy |
| Safety | Unsupported-memory rate; unsupported-claim rate; production-path violation count |
| Cost | Model calls per turn; tokens; p50/p95 latency; peak RAM/VRAM; setup time; disk |
| Privacy | External connection attempts; telemetry events; cloud-provider calls |

**Pre-registered acceptance thresholds.**

Hard gates (any failure rejects the condition):

- No production database access; fingerprint unchanged after the run.
- No external network or telemetry; verified by logs or a local monitor.
- Rollback in under 10 minutes.
- No destructive migration.
- The run fits the 8 GB VRAM / 16 GB RAM envelope, or the failure is recorded as a disqualifier.

Quality gates for a candidate or native change to be considered useful:

- Extraction precision >= 0.90 and recall >= 0.80 (macro).
- User-vs-assistant attribution >= 0.95.
- Recommendation-as-preference false positives <= 0.02.
- Correction/supersession accuracy >= 0.90.
- Stale-current rate <= 0.05.
- Duplicate-memory rate <= 0.05.
- Retrieval recall@5 >= 0.85 and at least +0.10 over B0.
- MRR >= 0.75.
- No-evidence assertion rate <= 0.02.
- Conflict-labeling accuracy >= 0.95.
- Unsupported-memory rate <= 0.02.

Efficiency gates:

- p95 end-to-end memory latency <= 2x B0.
- Model calls attributable to memory <= 2 per turn.
- Setup time <= 1 working day for an experimental condition.
- No production dependency added without a separate approval.

Decision rules:

- **Native fix:** if B0 fails a case and N1 passes the quality gates with acceptable cost, implement the native fix.
- **Concept adaptation:** if M1 or L1 improves a target metric but fails authority, data-safety, or integration gates, borrow the prompt/schema concept and keep it native.
- **Package adoption:** only if the package passes all hard gates and quality gates, beats the best native condition, and has a retrieval-only integration path with a tested rollback.
- **Reject:** if a candidate fails a hard gate, requires an external service, or cannot be isolated.
- **No-go for Graphiti:** setup > 1 day, structured-output failure > 10%, or peak memory outside the envelope.
- **No-go for Letta:** any requirement to move Aurora's identity/loop into the runtime for the experiment.

**Procedure.**

1. Freeze the dataset and thresholds.
2. Run the production-isolation preflight and verify the fingerprint.
3. Run B0 and B1.
4. Run N1.
5. Run M1 in a separate environment.
6. Run L1 only if it can be done without touching the main environment.
7. Run Graphiti/Letta only if preflight passes; otherwise record "not tested" and rely on the architectural assessment.
8. Score outputs blind to condition where practical.
9. Write a decision memo with raw outputs, metrics, resource logs, and failures.
10. Re-check the production fingerprint and confirm no production change.

**Deliverables.**

- Frozen JSONL dataset and label guide.
- Run manifests with model/version/hash and configuration.
- Raw outputs per condition.
- Metrics table and confidence notes.
- Resource and privacy logs.
- Decision memo with adopt/adapt/reject recommendations.
- Updated six-case regression tests for whatever is adopted.

---

## 10. Risks, uncertainties, sources, and final decision

### 10.1 Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Assistant suggestion becomes a user preference | Medium | High | First-class recommendation type with explicit status; user-only truth; tests |
| ADD-only memory accumulates stale/duplicate facts | High for Mem0 | High | Do not use as truth; explicit update/delete layer; stale/duplicate metrics |
| Default telemetry or cloud LLM leaks data | Medium | High | Synthetic data; telemetry off; network monitoring; local-only providers |
| Graph database license/ops burden | Medium | High | Reject for M2; if revisited, legal and operational review first |
| 8B model cannot produce reliable structured output | High for Graphiti/LangMem | High | Preflight; structured-output failure threshold; keep deterministic core |
| Semantic index consumes VRAM or adds latency | Medium | Medium | Measure; keep channel optional; fall back to aliases |
| Cross-schema preference divergence | High | Medium | Canonical concept family; migration tests; six-case suite |
| Dual-write divergence | High if integrated | High | No dual-write; single source of truth |
| Evaluation overfits a small synthetic set | Medium | Medium | Freeze labels; negative controls; blind scoring; repeat runs |
| Documentation drift misleads operators | High | Medium | Update docs with code changes; reference current file/line evidence |
| PRIMORA license is unclear | Medium | Medium | Resolve before distribution; not a blocker for internal work |
| Identity continuity is damaged by runtime replacement | Low if rejected | Very high | Do not adopt Letta as the runtime in M2 |

### 10.2 Known, inferred, and needs testing

**Known.**

- Current PRIMORA behavior for user-only current state, assistant conflict, supersession, validity, and isolation.
- Production counts and the current fingerprint.
- Candidate versions, commits, licenses, and core dependency lists as of 2026-10-19.
- Mem0 telemetry defaults and ADD-only V3 add behavior from documentation/source.
- Graphiti's graph-backend requirements and bi-temporal model.
- LangMem's manager-only design and dependency set.
- Letta's runtime/MemFS design and Docker support status.
- No repository-level PRIMORA license file was found.

**Inferred.**

- Mem0/LangMem extraction quality on Qwen 8B.
- Graphiti's actual LLM-call count and memory footprint.
- Letta's local-runtime performance.
- The effect of semantic-index activation on PRIMORA retrieval.
- The relative value of a compact preference block versus vector search.

**Needs testing.**

- All six cases on a frozen dataset.
- Whether native canonicalization closes case 3 without collateral damage.
- Whether the semantic channel can operate within 8 GB VRAM alongside the 8B chat model.
- Whether Mem0's extraction gain (if any) survives authority and duplication constraints.
- Whether any candidate can meet the p95 and model-call budgets.
- Whether Graphiti can run reliably with an 8B local model at all.

### 10.3 Source register

All sources accessed 2026-10-19 unless noted.

| Source | Version / evidence | License | URL |
|---|---|---|---|
| Mem0 repository | commit `0724291bd864cf807e88ee8981c8226644e8fb42` | Apache-2.0 | https://github.com/mem0ai/mem0 |
| Mem0 package | `mem0ai` 2.2.1, 2026-09-25 | Apache-2.0 | https://pypi.org/project/mem0ai/ |
| Mem0 OSS overview | current docs | — | https://docs.mem0.ai/open-source/overview |
| Mem0 configuration | current docs | — | https://docs.mem0.ai/open-source/configuration |
| Mem0 add-memory API | V3 ADD-only pipeline | — | https://docs.mem0.ai/api-reference/memory/add-memories |
| Mem0 local model guide | LM Studio/OpenAI-compatible | — | https://docs.mem0.ai/components/llms/models/lmstudio |
| Mem0 telemetry source | `MEM0_TELEMETRY` default True | Apache-2.0 | https://raw.githubusercontent.com/mem0ai/mem0/main/mem0/memory/telemetry.py |
| Graphiti repository | commit `7b0182f94d27075db488636cd5456981ee10b987` | Apache-2.0 | https://github.com/getzep/graphiti |
| Graphiti package | `graphiti-core` 0.31.1, 2026-10-15 | Apache-2.0 | https://pypi.org/project/graphiti-core/ |
| Graphiti README | Neo4j 5.26+, temporal graph, telemetry | — | https://raw.githubusercontent.com/getzep/graphiti/main/README.md |
| Graphiti local structured-output issue | community report #337 | — | https://github.com/getzep/graphiti/issues/337 |
| Graphiti paper | arXiv 2501.13956 | — | https://arxiv.org/abs/2501.13956 |
| Neo4j Community license | GPLv3 | GPLv3 | https://github.com/neo4j/neo4j/blob/2026.07/LICENSE.txt |
| FalkorDB license | SSPLv1 | SSPLv1 | https://docs.falkordb.com/references/license.md |
| LangMem repository | commit `48e3c11f5bb527282c7d5339c6a87a0b35abccfc` | MIT | https://github.com/langchain-ai/langmem |
| LangMem package | `langmem` 0.0.30, 2025-10-27 | MIT | https://pypi.org/project/langmem/ |
| LangMem extraction source | manager primitives/operations | MIT | https://raw.githubusercontent.com/langchain-ai/langmem/main/src/langmem/knowledge/extraction.py |
| Letta repository | commit `8816d3a8516893bfdc0fa14bfb1235226012e295` | Apache-2.0 | https://github.com/letta-ai/letta |
| Letta package | `letta` 0.34.12, 2026-10-12 | Apache-2.0 | https://pypi.org/project/letta/ |
| Letta memory docs | blocks/archival/recall | — | https://docs.letta.com/letta-agent/memory |
| Letta agent memory SDK | memory APIs | — | https://docs.letta.com/agent-sdk/memory |
| Letta MemFS docs | git-backed memory filesystem | — | https://docs.letta.com/letta-code/memfs |
| Letta self-hosting | Postgres/pgvector; Docker support status | — | https://docs.letta.com/guides/selfhosting/postgres |
| PRIMORA source | current repository | no root license found | local |
| PRIMORA production DB | read-only fingerprint `8b84160c...` | — | local |

### 10.4 Final decision

**Retain PRIMORA. Adapt it natively. Do not adopt a candidate framework in M2.**

The system already solves the trust-critical half of the problem better than any candidate: user authority, provenance, temporal history, conflict visibility, and offline operation. Its weaknesses are real but narrow: assistant recommendations lack a memory object, global preferences are admitted too conservatively, and extraction/canonicalization is incomplete.

The recommended order is:

1. Add the recommendation/suggestion record and its tests.
2. Add canonical preference concepts and a compact stable-preference admission path.
3. Freeze and run the six-case benchmark against the current baseline.
4. If and only if native changes miss the pre-registered thresholds, run Mem0 OSS as an isolated shadow extractor.
5. Reject Graphiti, LangMem, and Letta as dependencies for this milestone. Revisit Graphiti only under a measured multi-entity temporal requirement and a larger hardware envelope. Revisit Letta only if Aurora's identity/runtime architecture is intentionally being reconsidered.

### 10.5 What would change this recommendation

- If a measured benchmark shows PRIMORA's extraction recall is materially below 0.80 on the target cases, an LLM candidate generator becomes more compelling, but still only behind the candidate gate.
- If Mem0 adds first-class authority, automatic supersession, and a fully local telemetry-off default, its fit improves.
- If Graphiti can run reliably on an 8B local model within 16 GB RAM and a one-day setup, its temporal model becomes worth a larger spike.
- If Letta's local runtime becomes lightweight and stable with 8B models, it could be evaluated as a host, but the identity-continuity risk remains.
- If Aurora acquires a measured need for multi-hop entity reasoning across many sources, graph structure becomes more valuable than it is today.
- If semantic retrieval on the existing index closes the indirect-retrieval gap at acceptable cost, no external retrieval system is needed.

---

## Appendix A — Proposed evaluation record schema

```json
{
  "id": "case1-001",
  "category": "preference_vs_recommendation",
  "turns": [
    {"role": "user", "content": "I love Radiohead."},
    {"role": "assistant", "content": "You might like Portishead."},
    {"role": "user", "content": "What should I listen to tonight?"}
  ],
  "expected_operations": [
    {"op": "ADD", "concept": "likes", "value": "Radiohead", "authority": "user"},
    {"op": "SUGGEST", "concept": "recommendation:music", "value": "Portishead", "authority": "assistant"}
  ],
  "expected_current_facts": [
    {"concept": "likes", "value": "Radiohead", "authority": "user"}
  ],
  "expected_retrieval": ["Radiohead preference", "Portishead suggestion"],
  "expected_answer_constraints": [
    "may mention Portishead as a suggestion",
    "must not state the user likes Portishead as fact"
  ],
  "provenance_expectation": "both items link to their source messages",
  "notes": "negative control for authority inversion"
}
```

## Appendix B — Minimum test matrix

| Test | Purpose |
|---|---|
| User preference + assistant suggestion | Authority distinction |
| Conflicting assistant claim | Unresolved, never current |
| Same-slot correction | Supersession and history |
| Cross-schema correction | Canonicalization gap |
| Indirect music/book query | Retrieval recall without lexical overlap |
| Cross-session weekend plan | Global preference activation |
| No-evidence question | Truthful "I do not know" |
| Duplicate statement | No duplicate current truth |
| Hedged statement | Pending candidate, not current |
| Assistant recommendation accepted by user | Status transition from suggestion to preference |
| Production-path attempt | Isolation refusal |
| Post-run fingerprint | No production change |

---

*End of document. No implementation was performed. The production fingerprint remained unchanged during this evaluation.*
