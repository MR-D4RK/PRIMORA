# Unified Memory Intelligence (Aurora / Prime)

## Goal
Aurora interprets user-language memory queries with the **same local brain** used
for conversation; Prime validates a narrow structured plan and retrieves actual
user-source evidence; Aurora expresses the answer from that evidence. No new
keyword/typo dictionary, backend, migration, or model is introduced.

## Flow

1. `CognitiveLoop.run` asks the existing Aurora Brain for JSON intent.
2. `memory_coordinator.validate_plan` accepts only supported read-only source
   types, bounded time windows, short named subjects, and boolean modifiers.
3. `Mind.memory_query_evidence` delegates to `ContextManager`, using current
   user states, historical user states when first mention is requested, and
   source-linked experience events (including legacy reports).
4. Source IDs, messages, and UTC **statement** timestamps are included. User
   reactions are labeled *possible* follow-ups; they are not proven causality.
5. The same Brain answers naturally with evidence-only instructions. Invalid
   citations or ISO dates cause fallback to literal evidence.
6. Ordinary/nonmemory questions proceed through the existing cognitive loop.

The pre-existing phrase-based lookup functions remain only for compatibility
with non-production scripted/legacy Brain implementations. Actual `OpenAIBrain`
and `AuroraBrain` instances use model-led interpretation and do not enter those
handlers. No stored memories are edited by this query path.

## Limits / not yet implemented

- The 8B local model can misinterpret an intent; model understanding must be
  verified in **live fresh-session** trials and improved at the prompting or
  capability level, not by adding phrase exceptions.
- A generated answer's semantic entailment cannot be proven by the date/ID
  guard; preserve user quotes and evaluate natural responses critically.
- Evidence retrieval is capped (up to 90 messages); broad inventories are
  explicitly marked incomplete if truncated. No semantic relationship graph
  is inferred in this milestone (planned for Relationship Memory).
- Previously stored mistakes, and source formation logic, are not migrated or
  corrected here. Memory ingestion's own legacy pattern rules remain a
  separate, visible modernization target.
- Model interpretation adds one local-model generation per turn and a second
  for answered memory questions; benchmark latency on the user's hardware.

## Verification

`python -m pytest TESTS/prime/test_unified_memory_intelligence.py -q`

Additional cross-domain / regression suites must pass. In a new Windows CLI
session, try multiple *different* ways of asking for personal preferences,
episodes, statement dates and reasons; include typo/paraphrase and a pronoun
follow-up, and ordinary nonmemory questions. Verify no invented reason or
inferred preference onset, and avoid claiming PASS until actual Qwen3 behaves
correctly.
