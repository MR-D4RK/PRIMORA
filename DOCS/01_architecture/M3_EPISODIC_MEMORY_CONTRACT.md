# PRIMORA — Episodic Memory Contract (Milestone 3)

## Permanent architecture rule

**Do not hardcode subjects, people, products, songs, projects, or scenario-specific
answers to pass a test.** Extend general mechanisms and test them on multiple
unrelated subjects, including previously unseen ones. A domain adapter is
acceptable only when clearly separated from the core memory architecture.

## Batch 1: Experience capture

- User-reported actions and decisions are stored separately from factual and
  preference states in `experience_events`.
- Each event points to exactly one original `messages` row; its `statement` is
  verbatim. The originating message must be authored by the user, from the same
  session. Assistant responses cannot establish the user's experiences.
- `reported_at` is when the statement was stored, **not** an inferred time of
  the underlying event. The statement is a user report, not independent proof
  that an outside event occurred.
- Replaying the same stored turn does not duplicate the event. Existing message
  rows and historical preference states are not rewritten.
- Extraction is deliberately conservative: explicit first-person, completed
  or past reported actions; no future plans, questions, negations, ambiguous
  references, or assistant claims. English action patterns are an initial
  extraction interface, not an exhaustive description of human experiences.
- `ContextManager.experience_events()` is a read-only inspection interface;
  Batch 1 does **not** claim conversational episodic retrieval is complete.

## Batch 2: Episode organization

- `ContextManager.experience_episodes()` is an **inspection view** that groups
  original `experience_events` across sessions. Grouping is read-only and
  computed from original immutable user-report records; it does not create a
  duplicate episode database or modify the conversation history.
- Subject keys are generic normalized activity objects, matched conservatively
  and exactly, with a configurable reporting-time gap (45 days by default).
  Different object descriptions remain separate until a future evidence-backed
  relationship layer can establish their identity; *matching does not prove
  the two statements refer to the same real-world thing*.
- Every episode references its individual, unchanged event statements, session
  IDs, source message IDs, and report timestamps. A derived episode ID may
  change when older evidence is imported; it is not a durable external key.
- Source role/content/session/timestamp and event kind/actor must match; any
  inconsistent record is excluded rather than being presented as user evidence.
- Ambiguous subjects, negation, future plans, and assistant statements must not
  be promoted into confirmed experiences. No subject vocabulary is special-cased.
- Episode organization does not claim natural-language episodic recall yet.

## Subsequent batches

1. Retrieve events and episodes using natural language with provenance and
   uncertainty; do not promote model interpretations to user facts.
2. Integrate and verify across unrelated subjects, multiple sessions, and
   realistic speech; benchmark missed experiences and false positives.

Roll back any batch that introduces a significant live regression. Every change
must have an isolated test result and a user-visible demonstration.

## Batch 3: Source-verified retrieval

- Explicit experience-history questions use the episode inspection view, not
  external research or a new dedicated model.
- Dates are the original user-message report dates; they do not establish the
  underlying event's real-world occurrence time. No outcome is inferred.
- Questions without verifiable evidence receive an honest uncertainty response.

## Batch 4: Cross-session integration and bounded conversational references

- The same episode rules are used for music, Verse, projects and unknown topics;
  no recognized-activity vocabulary is embedded in routing or storage.
- A vague follow-up such as "When did we finish it?" may refer ONLY to the
  immediately preceding, explicit *user* episode question in the same chat
  context. Unbound pronouns lead to clarification, never a random episode.
- For explicit completion-date questions, only source-backed completion reports
  are presented. Reports of starting or testing are not completion evidence.
- Select relevant episodes from the available evidence before applying the
  small response budget. Current storage access remains bounded to the 1,000
  most recent source-verified events; this milestone does not guarantee search
  over arbitrarily large histories. A future retrieval index can extend this.
- Retrieval never modifies episode events, source messages or user preferences.
  Corrupt or assistant-authored evidence cannot be promoted into a verified
  user report. New session IDs do not reset the underlying episode store.
- Natural-language support is intentionally bounded to tested English forms.
  Broader paraphrase understanding and inferred episode relationships still
  require later design and evaluation, not ad hoc topic hardcoding.
