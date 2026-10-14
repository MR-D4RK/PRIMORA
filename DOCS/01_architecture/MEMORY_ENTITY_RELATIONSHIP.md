# PRIMORA M5 Entity & Relationship Memory

- **Milestone**: M5 — Entity & Relationship Memory (memory milestone stream)
- **Status**: Approved specification; implementation not started
- **Project Root**: `C:\PRIMORA PROJECT`
- **Owner**: Prime (`PRIMORA/prime/context/`)
- **Extends**: `DOCS/01_architecture/CONTEXT_MANAGER.md` and the implemented M1–M4 memory architecture

---

## Purpose

Structured Memory answers one question:

> **What does PRIMORA currently believe?**

M5 adds the identity and connection layer required to answer a second question:

> **Which stable things are being talked about, and how are they connected?**

M5 introduces a small canonical entity registry, scoped aliases, and explicit typed relationships. It does not replace Raw Experience, Structured Memory, Candidates, lifecycle, or M4 temporal validity. Structured Memory remains authoritative; relationships reference stable IDs and never copy state text.

**Milestone namespace:** this document uses the memory milestone stream (M1 formation, M2 lifecycle, M3 Active/Working Memory, M4 temporal validity, M5 entities/relationships). It is distinct from the Verse UI milestone numbering.

## Scope boundary

**M5 owns:** canonical entity identity, scoped aliases, entity lifecycle, a closed set of typed relationships, entity-to-memory associations, rationale-to-decision explanation links, and bounded relationship queries.

**M5 does not own:** raw conversation persistence, state truth, retrieval ranking, semantic retrieval, RAG, consolidation, forgetting/decay, goals/progress, event/episode memory, Aurora Brain behavior, Verse rendering, or a graph database.

**Owner:** Prime. Prime owns deterministic identity, persistence, lifecycle, scope, provenance, and traversal. Aurora may later propose semantic alias or relationship candidates; Aurora never becomes the authority that makes them current.

**No second model:** M5 uses the existing Aurora Brain for optional proposals. It does not introduce another model, controller, or autonomous agent.

## M1–M4 foundation

| Milestone | Implemented responsibility | Authoritative code | Authoritative tests |
|---|---|---|---|
| M1 | Deterministic user-only Memory Formation; typed candidates and states; rationale extraction | `PRIMORA/prime/context/state.py` | `TESTS/prime/test_memory_formation.py` |
| M2 | Candidate and state lifecycle; confidence; authority guard; supersession | `state.py`, `store.py` | `TESTS/prime/test_memory_lifecycle.py`, `test_memory_correctness.py` |
| M3 | Computed Active Memory; runtime-only bounded Working Memory; promotion to candidates | `manager.py`, `working.py` | `TESTS/prime/test_memory_working.py` |
| M4 | Creation/observation/validity/supersession separation; half-open intervals; atomic successor replacement; temporal queries | `store.py`, `manager.py`, `models.py` | `TESTS/prime/test_memory_temporal.py` |

Where `CONTEXT_MANAGER.md` predates or omits implemented M1–M4 behavior, the current code and M1–M4 tests are authoritative. This document does not amend `CONTEXT_MANAGER.md`; it specifies the M5 layer that extends the implemented system.

## Architecture

```text
Raw Experience
    messages (append-only, timestamped, source of evidence)
        |
        v
M1 Memory Formation
    user text -> deterministic candidates / states
        |
        v
M2 + M4 Structured Memory
    states: concept/value + authority + confidence + lifecycle + validity
    candidates: proposals only, never truth
        |
        +-----------------------+
        |                       |
        v                       v
M5 Entity Identity         M5 Typed Relationships
    entities                 part_of:   entity -> entity
    entity_aliases           concerns:  memory state -> entity
    merged_into_entity_id    explains:  memory state -> memory state
        |                              |
        +---------------+--------------+
                        v
                 Bounded M5 Queries
        entity lookup / scoped resolution / alias lookup
        one-hop traversal / entity -> memory / explanation path
                        |
          +-------------+-------------+
          |                           |
          v                           v
    Active Memory                Working Memory
    computed projection          runtime-only selection
    bounded, current             bounded, task-local
```

**Hybrid model:** Raw Experience remains evidence. Structured Memory remains authoritative. M5 persists only the identity and typed connections that cannot be recovered reliably from flat state text. Inverse or weak relationships are derived, not stored.

## Terminology

**Entity: "What stable thing is being referred to?"** A repeatable referent such as `PRIMORA`, `Verse`, `Aurora`, or `manager.py`.

**Memory: "What is asserted about something?"** A sourced, temporal `StateRecord` such as `aurora visual = frozen`.

**Relationship: "What typed connection is asserted between two referents?"** A temporal assertion such as `Verse part_of PRIMORA`.

**Alias: "Which names may resolve to this canonical entity?"** A scoped, provisional or accepted name link such as `the Verse interface` → `Verse`.

**Concept: "What category or property is being described?"** A normalized state concept such as `interface`, `project`, or `personality`. A concept is not automatically an entity.

**Decision:** an M1–M4 `DECISION` memory record identified by `state_id`; not an entity.

**Rationale:** an M1–M4 `RATIONALE` memory record identified by `state_id`; not an entity.

## Entity model

M5 adds one small canonical entity registry.

| Field | Type | Semantics |
|---|---|---|
| `entity_id` | integer | Stable canonical identity inside the database |
| `session_id` | text | Session that established the row; provenance on global rows |
| `scope` | text | Session ID or `global`, matching `states.scope` |
| `entity_type` | text | `PERSON`, `AGENT`, `PROJECT`, `COMPONENT`, `FILE` |
| `display_name` | text | Human-readable canonical name |
| `normalized_name` | text | Deterministic exact-normalization key for candidate lookup |
| `status` | text | `candidate`, `active`, `merged`, `archived` |
| `merged_into_entity_id` | integer or null | Canonical target after a confirmed merge |
| `source_ids` | text | Raw message IDs supporting the identity assertion |
| `authority` | text | `user`, `assistant`, `system`, or `tool` |
| `confidence` | real | Evidential support; never truth probability |
| `created_at` | real | Entity-row creation time |
| `observed_at` | real | Source evidence observation time |
| `merged_at` | real or null | Time of a confirmed merge transition |

**Entity identity:** `entity_id` is the only authoritative identity. `display_name` and `normalized_name` are lookup attributes and may change; they are not identity.

**Normalized name:** case folding, whitespace normalization, and safe punctuation normalization may be deterministic. A normalized-name collision is a resolution candidate, not permission to merge.

**No hard delete:** entities are retained. Identity changes use `merged_into_entity_id` or `archived`, preserving provenance and history.

**Entity types are closed for M5:** `PERSON`, `AGENT`, `PROJECT`, `COMPONENT`, `FILE`. The list may be extended by a later approved milestone, not opportunistically at runtime.

## Alias model

Aliases are a separate structure, not relationship edges.

| Field | Type | Semantics |
|---|---|---|
| `alias_id` | integer | Stable alias-assertion identity |
| `entity_id` | integer | Referenced canonical entity |
| `session_id` | text | Session that established the alias |
| `scope` | text | Session ID or `global` |
| `alias_text` | text | Surface form written or observed |
| `normalized_alias` | text | Deterministic exact-normalization key |
| `status` | text | `candidate`, `current`, `superseded`, `rejected` |
| `source_ids` | text | Raw message IDs |
| `authority` | text | Who proposed or established the alias |
| `confidence` | real | `1.0` explicit user assertion; `0.3` hedged/inferred proposal |
| `created_at` | real | Alias-row creation time |
| `observed_at` | real | Source evidence observation time |
| `valid_from` | real or null | Inclusive effective start |
| `valid_until` | real or null | Exclusive effective end |
| `superseded_at` | real or null | Alias lifecycle transition time |

**Deterministic alias handling:** exact normalized matches may resolve to an active entity within the same scope.

**Semantic alias handling:** interpretations such as `Aurora's world` → `Verse` or `the Verse world` → `Verse` may be proposed with Aurora Brain assistance, but remain `candidate` until explicit confirmation.

**No alias implies identity:** an alias record expresses a proposal or accepted name mapping; it does not merge two canonical entities.

## Relationship model

M5 persists a closed set of typed relationship assertions.

| Type | Allowed subject | Allowed object | Meaning |
|---|---|---|---|
| `part_of` | entity | entity | Structural containment: Verse `part_of` PRIMORA |
| `concerns` | memory state | entity | A state is about an entity |
| `explains` | memory state | memory state | A rationale state explains a decision state |

`RelationshipRecord` fields:

| Field | Type | Semantics |
|---|---|---|
| `relationship_id` | integer | Stable relationship-assertion identity |
| `session_id` | text | Session that established the row; provenance on global rows |
| `scope` | text | Session ID or `global` |
| `relationship_type` | text | `part_of`, `concerns`, or `explains` |
| `subject_kind` | text | `entity` or `memory_state` |
| `subject_id` | integer | `entity_id` or `state_id` |
| `object_kind` | text | `entity` or `memory_state` |
| `object_id` | integer | `entity_id` or `state_id` |
| `source_ids` | text | Raw message IDs supporting the assertion |
| `authority` | text | `user`, `assistant`, `system`, or `tool` |
| `confidence` | real | Evidential support, never truth |
| `status` | text | `candidate`, `current`, `superseded`, `unresolved`, `rejected` |
| `supersedes_id` | integer or null | Previous relationship-assertion version |
| `created_at` | real | Relationship-row creation time |
| `observed_at` | real | Source evidence observation time |
| `valid_from` | real or null | Inclusive effective start |
| `valid_until` | real or null | Exclusive effective end |
| `superseded_at` | real or null | Relationship lifecycle transition time |

**Closed endpoint matrix:** only the three allowed type combinations may be persisted. M5 does not define arbitrary predicates or arbitrary node types.

**No state duplication:** a relationship stores stable IDs only. It never copies a state's `concept`, `value`, rationale text, or raw message content.

**Derived relationship:** `contains` is the inverse traversal of inbound `part_of` edges. `contains` is never persisted, preventing inverse-duplication drift.

**No generic `related_to`:** weak co-occurrence may be derived for presentation or future consolidation, but is not a persisted M5 relationship type.

**Current uniqueness:** at most one `current` relationship per `(scope, relationship_type, subject_kind, subject_id, object_kind, object_id)` under normal M5 write paths. Legacy or anomalous duplicates remain detectable and never silently merged.

## Entity lifecycle

```text
candidate -> active -> merged
                     -> archived
```

| Status | Meaning | Query eligibility |
|---|---|---|
| `candidate` | Mentioned or proposed; identity not confirmed | Candidate APIs only |
| `active` | Canonical identity usable by current memory | Current entity and relationship queries |
| `merged` | Redirected to another canonical entity | Resolution follows `merged_into_entity_id`; original remains history |
| `archived` | Retained but no longer active | Historical lookup only |

**Activation:** only a user-authored, adequately sourced assertion may create an `active` entity. Assistant, Aurora, tool, and system output may create candidates.

**Merging:** never automatic. A merge requires explicit user confirmation or a separately approved trusted promotion rule.

**Archiving:** no hard delete. Archiving preserves the entity row, aliases, provenance, and historical relationships.

## Relationship lifecycle

Relationship lifecycle follows the M4 state lifecycle model.

| Status | Meaning |
|---|---|
| `candidate` | Proposed relationship awaiting confirmation |
| `current` | Accepted relationship, still subject to scope and temporal validity |
| `superseded` | Replaced by a newer assertion or invalidated without replacement |
| `unresolved` | Contradictory relationship claim that has not been resolved |
| `rejected` | Candidate refused or overtaken |

**Versioning:** relationship corrections create a new relationship row and link it through `supersedes_id`. Prior rows remain history.

**Invalidation:** a bare invalidation marks the current relationship `superseded` and ends its validity; it does not invent a replacement.

**Authority barrier:** only a user-authored assertion may be `current`. Assistant, Aurora, tool, and system relationships are `candidate` or `unresolved`.

**Conflict rule:** two unresolved relationship claims never silently select a winner. An explicit user correction resolves by supersession.

## Scope rules

M5 reuses the existing M1–M4 scope model exactly.

| Rule | Behavior |
|---|---|
| Default scope | Session-scoped under the current `session_id` |
| Global scope | Only explicit user declaration or a separately approved trusted promotion |
| Session A / Session B | Same display name may represent separate entities |
| Cross-session merge | Never automatic |
| Session → global promotion | Never automatic |
| Global rows | May originate in different sessions; `session_id` is provenance, not ownership |
| Foreign-session access | Explicit request only, preserving per-session identity |
| Aliases and merges | Scope-aware; a session alias never silently becomes global |

Existing global markers remain the M1–M4 user declarations such as `for all conversations`, `globally`, and `across sessions`.

**Scope is isolation, not authorization.** M5 inherits the existing caller-supplied `session_id` model and does not add tenancy or authentication.

## Provenance, authority, and confidence

**Provenance:** every active entity, alias, and relationship must answer:

```text
Who established it?
Which raw message supports it?
When was it observed?
What is its lifecycle state?
```

- `source_ids` references raw `messages.id` values.
- Raw message text is never copied into entity or relationship records.
- A dangling source ID must be detectable; it must not silently become fabricated evidence.
- Confirmation requires non-empty provenance, matching the existing M2 rule.

**Authority:**

| Source | May create candidate? | May create current/active truth? |
|---|---:|---:|
| Explicit user assertion | Yes | Yes |
| Hedged user assertion | Yes | No, until confirmed |
| Assistant statement | Yes | No |
| Aurora Brain inference | Yes | No |
| Tool result | Yes | No |
| System output | Yes | No |

**Deterministically derived relationships** inherit the authority of their supporting accepted user assertion. A derived `contains` relationship has the authority of the underlying current user-authored `part_of` assertion.

**Confidence:**

| Assertion | Confidence |
|---|---:|
| Explicit user assertion | `1.0` |
| Hedged, inferred, assistant, tool, or system proposal | `0.3` |

Confidence is independent of:

- `authority`;
- lifecycle `status`;
- temporal validity;
- retrieval `importance`;
- composite retrieval `score`.

**Confidence never makes an assertion current.** Truth eligibility is determined only by authority, status, scope, and validity.

**No confidence arithmetic in M5:** averaging, reinforcement, and multi-evidence consolidation are deferred.

## Temporal rules

M5 relationship and alias validity use M4 semantics exactly.

| Field | Meaning |
|---|---|
| `created_at` | Relationship or alias row creation time |
| `observed_at` | Source evidence observation time |
| `valid_from` | Inclusive time the assertion became effective |
| `valid_until` | Exclusive time it stopped being effective |
| `superseded_at` | Actual lifecycle transition time |

**Half-open interval:**

```text
valid_from <= t < valid_until
```

**Open interval:** `valid_until IS NULL` means the assertion remains effective until explicitly superseded or invalidated.

**Invalid interval:** `valid_until < valid_from` is rejected. Trusted normalized boundaries may be supplied; natural-language dates remain deferred.

**Ordinary transition:**

```text
old.valid_until = new.valid_from = effective boundary
old.superseded_at = actual lifecycle transition time
```

**Retroactive transition:**

```text
old.valid_until = historical effective boundary
old.superseded_at = actual processing time
observed_at remains the real observation time
```

**Entity identity time:** canonical entity identity uses `created_at`, `observed_at`, and lifecycle transition fields. Entity existence itself is not modeled as a truth interval in M5; changing structural relationships is modeled temporally.

**Migration principle:** M5 must not invent historical entity or relationship validity. Unknown boundaries remain unknown.

## Merge rules

Entity merge is a lifecycle transition, never an automatic identity optimization.

**Required behavior:**

1. The user explicitly confirms the merge, or a separately approved trusted rule supplies a normalized boundary.
2. The losing entity is preserved with `status="merged"`.
3. `merged_into_entity_id` records the canonical target.
4. `merged_at` records the transition time.
5. Provenance is preserved on both entities and their aliases.
6. Future resolution follows the canonical target.
7. Historical queries may still return the merged identity.
8. No row is hard-deleted.

**Relationship handling after merge:** new writes resolve through the canonical entity. Historical relationship rows keep their original endpoint IDs. M5 does not rewrite historical provenance in place.

**Merge must not cascade silently:** alias migration, relationship redirection, and archival remain explicit, auditable operations.

## Transaction invariants

**Invariant:** a supporting Structured Memory change and any persisted M5 relationship change occur atomically in the same SQLite transaction.

For a normal state change:

```text
BEGIN IMMEDIATE
    verify expected current state set
    insert successor StateRecord
    stamp predecessor StateRecord
    insert/update associated Entity or Relationship rows
    update candidate status when applicable
COMMIT
```

If any associated entity, alias, or relationship write fails, the supporting state change rolls back.

**Rules:**

- M5 never commits a state transition and leaves its required relationship partially written.
- Candidate relationship creation may commit independently because a candidate is not truth.
- Confirmation of a candidate relationship and its transition to `current` are one atomic operation.
- Entity activation and its supporting state assertion are one atomic operation when both are required.
- Derived relationships are computed from a consistent read snapshot and are not independently written.
- Unrestricted repair jobs must not rewrite history; any reconciliation is diagnostic first, mutation second.

This mirrors the M4 `create_state_successor()` transaction and compare-and-swap discipline.

## Query APIs and semantics

M5 query APIs are **specified but not implemented**. The following names are architectural proposals, not existing methods.

| Proposed operation | Semantics | Required bound |
|---|---|---|
| `get_entity(scope, entity_id, at_time=None)` | Return one canonical entity by stable ID, following `merged_into_entity_id` when requested | One entity |
| `resolve_entity(scope, name, at_time=None)` | Resolve an exact normalized name or accepted alias within one explicit scope; return ambiguity rather than merging | Bounded candidate set |
| `find_alias(scope, alias_text)` | Return current and candidate aliases with provenance and status | Bounded candidate set |
| `relationships_from(scope, entity_id, relationship_type=None, at_time=None, limit=N)` | Outbound one-hop traversal | Maximum one hop; mandatory limit |
| `relationships_to(scope, entity_id, relationship_type=None, at_time=None, limit=N)` | Inbound one-hop traversal; basis for derived `contains` | Maximum one hop; mandatory limit |
| `entity_memories(scope, entity_id, at_time=None, limit=N)` | Current or historical states linked by `concerns` | Mandatory limit |
| `explanations(scope, state_id, at_time=None, limit=N)` | Return rationale states linked by `explains` | Fixed two-record path; mandatory limit |
| `relationship_context(scope, entity_ids, at_time=None, limit=N)` | Return a small selected set of entities and current relationships for prompt assembly | Mandatory entity and relationship caps |
| `contains(scope, entity_id, at_time=None, limit=N)` | Derived inverse of `part_of`; never persisted | Maximum one hop; mandatory limit |

**Shared query gates:** every accepted relationship or explanation result must satisfy:

```text
authority = user
status = current
valid_from <= evaluation_time
(valid_until IS NULL OR evaluation_time < valid_until)
```

**No unrestricted traversal:** M5 provides no global walk, shortest-path search, connected-component search, or arbitrary-depth query. The explanation path is a fixed shape, not a general graph traversal.

**Ambiguity:** if scoped resolution has multiple plausible entities, return candidates. Never select a winner through string containment alone.

**No relationship ranking change:** M5 queries do not change the existing lexical retrieval scorer, FTS behavior, digest ranking, or Active Memory selection ranking.

## Active Memory interaction

Active Memory remains a computed, on-demand projection. M5 adds no second stored Active Memory copy.

**Allowed exposure:** selected current entities and selected current one-hop relationships that are:

- in the explicit current scope;
- user-authorized;
- lifecycle `current`;
- temporally valid at one captured evaluation time;
- relevant to already-selected structured memory or the current task;
- inside strict item and token bounds.

**Never exposed by default:**

- the entire entity registry;
- the entire relationship graph;
- candidate aliases or relationships;
- unresolved relationships;
- foreign-session relationships without an explicit request;
- historical relationships on a current query.

**Projection rules:**

- Active Memory may add future item kinds for entities and relationships, but only as projections.
- Relationship lines must retain entity IDs, scope, authority, status, and source provenance.
- Inverse `contains` is derived at read time and marked as derived.
- Active Memory never mutates entity or relationship records.
- Entity/relationship expansion must not weaken the existing `input_target` or `input_ceiling` budget.
- The selected relationship set must have a hard cap independent of raw graph size.

**Prerequisite:** current state projection must be explicitly bounded before M5 expansion. An unbounded graph-derived projection is forbidden.

## Working Memory interaction

Working Memory remains runtime-only and non-persistent.

**May hold:**

- selected canonical `entity_id` values;
- selected `relationship_id` values;
- a bounded selected path such as `Verse part_of PRIMORA`;
- entity and relationship provenance metadata;
- aliases explicitly adopted for the current task.

**Must not:**

- persist a graph or relationship snapshot to SQLite;
- refresh adopted entities automatically;
- silently redirect a working relationship after a merge;
- auto-promote an adopted item to current truth;
- hold an entire registry or graph;
- bypass the candidate flow for durable promotion.

**Promotion:** Working Memory → Candidates → confirmed state/entity/relationship only through an explicit flow. Promotion never writes current truth directly.

**Snapshot semantics:** adopted entities and relationships describe the state at adoption time. Long-running tasks must re-resolve explicitly when canonical identity changes.

## Deterministic versus Aurora-brain responsibilities

| Operation | Responsible layer | Rule |
|---|---|---|
| Exact name normalization | DETERMINISTIC | Case, whitespace, safe punctuation only |
| Extract explicit mention from anchored user grammar | DETERMINISTIC | Existing Memory Formation rules |
| Extract explicit `part_of` template | DETERMINISTIC | New narrow rule, when implemented |
| Assign `FILE` from an explicit path | DETERMINISTIC | Strong lexical evidence |
| Assign `PROJECT` / `AGENT` from an explicit approved cue | DETERMINISTIC | Closed vocabulary |
| Create a candidate entity or relationship | DETERMINISTIC | Never current by extraction alone |
| Apply scope, status, provenance, and validity gates | DETERMINISTIC | M1–M4 rules |
| Bounded one-hop traversal and explanation path | DETERMINISTIC | Fixed query shapes and limits |
| Interpret `Aurora's world` as an alias | RULE + AURORA BRAIN | Proposal only |
| Extract entities or relations from free prose | RULE + AURORA BRAIN | Deterministic mentions plus optional proposals |
| Resolve ambiguous entity type | RULE + AURORA BRAIN | Proposal only |
| Detect a contradiction in free prose | RULE + AURORA BRAIN | Never auto-resolved without explicit user correction |
| Propose an entity merge | AURORA BRAIN | Never applied automatically |
| Apply an entity merge | UNKNOWN / user confirmation required | Identity-changing operation |
| Infer goals, ownership, or life-tracking relations | AURORA BRAIN | Deferred; proposal only |

**Invariant:** Aurora Brain may accelerate interpretation, but deterministic Prime code owns identity writes, authority, scope, lifecycle, temporal stamping, and traversal.

## Acceptance invariants

An implementation satisfies M5 only when all of the following hold:

1. Raw `messages` remain append-only and unchanged.
2. Structured `states` remain the authoritative memory record.
3. Every persisted relationship references `entity_id` and/or `state_id`; no state text is copied.
4. Only user-authored assertions may create `active` entities or `current` relationships.
5. Assistant, Aurora, tool, and system assertions remain `candidate` or `unresolved`.
6. Confidence never changes authority, status, or temporal eligibility.
7. All M4 temporal fields and half-open interval semantics are preserved.
8. Unknown historical validity is never fabricated during migration.
9. `contains` is derived from `part_of` and never persisted.
10. `related_to` is never a persisted M5 predicate.
11. No entity merge is automatic.
12. No entity or relationship is hard-deleted.
13. `merged_into_entity_id` redirects future resolution while preserving history.
14. No cross-session identity merge occurs automatically.
15. No session-scoped entity or relationship is promoted to `global` automatically.
16. Supporting state changes and required persisted relationship changes are atomic.
17. All queries are scope-explicit and bounded.
18. General multi-hop traversal is unavailable.
19. Active Memory remains computed, current, scope-valid, and bounded.
20. Working Memory remains runtime-only and uses the candidate flow for promotion.
21. No graph database, second AI model, semantic retrieval, advanced RAG, goals/progress, or event/episode system is introduced.
22. Relationship or entity failures degrade diagnostically without losing the current request or raw evidence.

## Future implementation stages

No stage below is implemented by this specification.

| Stage | Deliverable | Boundary |
|---|---|---|
| M5.0 | Approved specification and test scenarios | Documentation only |
| M5.1 | Canonical entity registry and scoped alias structure | No relationship traversal |
| M5.2 | `part_of`, `concerns`, and `explains` persistence with M4 temporal/provenance semantics | No generic predicates |
| M5.3 | Bounded entity, alias, one-hop, entity-to-memory, and explanation queries | No retrieval-ranking changes |
| M5.4 | Selected entity/relationship exposure in Active Memory | Hard caps; no graph injection |
| M5.5 | Runtime-only selected entity/relationship paths in Working Memory | No persistence or auto-refresh |
| M5.6 | Optional Aurora-brain candidate proposals | Proposals only; explicit confirmation required |

Each stage requires separate implementation approval.

## Future compatibility

**Goals and progress:** not implemented in M5. Later goals/progress can remain M1–M4 state records linked to a `PERSON` or `PROJECT` entity through `concerns`. No new relationship vocabulary is required.

**Visualization:** future read-only UI may query bounded `relationships_from`, `relationships_to`, and `contains` results to draw a local subgraph. M5 itself adds no Verse UI and no visualization runtime.

**Consolidation:** later consolidation may count repeated supporting states, strengthen confidence, or propose entity merges. Until then, relationships remain individual sourced assertions; M5 performs no reinforcement.

**Controlled self-improvement:** Aurora may later propose alias, relationship, merge, or consolidation changes. Every proposal must enter the candidate flow, preserve provenance, and require the applicable explicit confirmation. Self-improvement never directly edits current entity or relationship truth.

## Explicit exclusions

M5 does not add:

- Neo4j or any graph database;
- a generic knowledge graph;
- a generic `related_to` predicate;
- automatic entity merges;
- automatic cross-session identity merging;
- automatic session-to-global promotion;
- a second AI model or controller;
- semantic or vector retrieval;
- hybrid retrieval or reranking;
- advanced RAG;
- consolidation or confidence reinforcement;
- forgetting, decay, or archival deletion;
- goals/progress implementation;
- event/episode implementation;
- natural-language temporal parsing;
- Verse changes.

## Architecture conflicts and documentation notes

- `CONTEXT_MANAGER.md` predates several implemented M1–M4 behaviors and omits Candidates, confidence, M4 temporal fields, Active Memory, and Working Memory. This M5 document references the implemented code and tests rather than copying those stale omissions.
- The current `resolve_concept()` longest-whole-word rule is a Structured Memory concept-continuity heuristic. It must not be reused as entity identity resolution or alias confirmation.
- Current Active Memory state items are not explicitly count-capped. Bounded M5 expansion is forbidden until the projection has a hard selection cap.
- Global state rows already cross sessions while retaining the originating `session_id`. M5 therefore treats `session_id` on global entity/relationship rows as provenance, not ownership.
- Two milestone-number streams exist in the repository: the memory stream and the Verse UI stream. This document belongs to the memory stream.

---

## File roles (ELI5 → technical)

- **`messages`** — *ELI5:* the permanent evidence notebook. *Technical:* append-only raw experience with timestamps and provenance IDs.
- **`states`** — *ELI5:* what PRIMORA currently believes. *Technical:* authoritative M1–M4 structured memory with lifecycle, authority, confidence, and temporal validity.
- **`candidates`** — *ELI5:* suggestions waiting for approval. *Technical:* proposal records that are never truth by themselves.
- **`entities`** — *ELI5:* the name tags for stable things. *Technical:* proposed canonical identity registry.
- **`entity_aliases`** — *ELI5:* other names that may point to one thing. *Technical:* proposed scoped, provisional, temporal alias assertions.
- **`relationships`** — *ELI5:* typed connections between things or memories. *Technical:* proposed closed-vocabulary temporal assertions referencing stable IDs.
- **Active Memory** — *ELI5:* today's relevant facts, entities, and connections. *Technical:* bounded computed projection; never a second store.
- **Working Memory** — *ELI5:* the small desk for the current task. *Technical:* runtime-only, bounded, non-persistent selection.
- **`MEMORY_ENTITY_RELATIONSHIP.md`** (this file) — the M5 architecture specification; implementation not started.
