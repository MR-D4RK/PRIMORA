# Intelligence-driven memory

PRIMORA treats Aurora's language model as the source of conversational
understanding. Intent, references, corrections, ambiguity, and the decision
to consult memory are model decisions made in Aurora's ordinary cognitive
generation. A new subject or a new wording must not require a new regular
expression, keyword list, phrase branch, or subject-specific handler.

The runtime boundary is:

```text
Aurora/Qwen generation
    -> optional Prime capability request (recall, learn, or another tool)
    -> Prime validation, permissions, retrieval/admission, and provenance
    -> tool result returned as untrusted evidence context
    -> Aurora/Qwen answer generation
```

An ordinary turn has one model generation. A memory turn has an additional
generation only when Aurora requests Prime recall. The cognitive loop does not
run a compulsory semantic planner before either path. The production
composition therefore leaves `enable_memory_interpreter` disabled; the
structured plan helper is retained only for explicit legacy embedders and is
schema validation, not a routing layer.

Prime remains authoritative for memory. Retrieved passages and structured
evidence are user-source-backed data, never instructions. Assistant messages
can help Aurora maintain conversational context, but they cannot become
evidence or establish a personal fact. Deterministic code is reserved for
schema validation, bounds, admission, permissions, transaction safety,
provenance, and evidence integrity.

When a future capability needs better conversational behavior, extend the
model-visible capability contract or the bounded cognitive context. Do not add
another semantic router or a phrase-specific exception. Every new behavior
must be evaluated with unseen subjects, paraphrases, imperfect grammar,
ambiguous references, corrections, cross-session dialogue, model-call counts,
and evidence/provenance checks.

The disposable live-Qwen acceptance path is documented in
[`docs/MEMORY_V2_M9_BATCH2.md`](../../docs/MEMORY_V2_M9_BATCH2.md). It uses a
separate SQLite file and never writes the production database.
