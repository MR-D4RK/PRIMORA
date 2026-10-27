# Memory V2 vs Mem0, Graphiti/Zep, Letta: comparative protocol

**This document is an architecture comparison, not a published scorecard.**
No competitor was installed or benchmarked in this environment. Published
results cannot be compared directly to local Qwen until the model, dataset,
retrieval budget, context budget, prompts and scoring all match.

| Dimension | PRIMORA Memory V2 (code audited at 2caf190) | Reference to compare |
| --- | --- | --- |
| Formation | User-source evidence, typed proposals, append-only raw messages, provenance | Mem0: extraction, dedup, linking, temporal enrichment |
| Retrieval | Hybrid semantic/lexical with current and historical states; model-led Prime recall | Mem0: vector/graph/temporal retrieval; Graphiti: hybrid graph/keyword/vector |
| Temporal | Recorded timestamps, event vs report time, superseded states | Graphiti: temporally aware graph edges, historical queries |
| Persistent agent context | Stored conversational data and states across sessions | Letta: memory blocks and persistent agent state; recent MemFS documentation |
| Trust boundary | User-source validation; assistant turns cannot be sole evidence | Compare provenance and injection defenses under identical adversarial data |
| Autonomous memory maintenance | Consolidation and audit capability exist; production reliability pending | Letta's memory lifecycle, Graphiti episodic processing |

## Primary reference docs (reviewed 2026-10-27)
- Mem0 memory-evaluation: https://github.com/mem0ai/mem0/blob/main/docs/core-concepts/memory-evaluation.mdx
- Mem0 README: https://github.com/mem0ai/mem0
- Graphiti / Zep documentation: https://help.getzep.com/graphiti/getting-started/welcome
- Graphiti overview: https://help.getzep.com/v2/graphiti/getting-started/overview
- Letta docs: https://docs.letta.com
- Letta memory description: https://github.com/letta-ai/letta-docs-md/blob/main/configuration/memory/index.md
- Evaluation datasets: LoCoMo, LongMemEval, MemoryAgentBench

## Fair evaluation protocol
1. Freeze the dataset before tuning the implementation. Keep an unseen test
   split with domains, entities and phrasings not used in development.
2. Feed each system the same conversation turns in chronological order. Mark
   user vs assistant source roles, conflicting updates, temporal qualifiers,
   and missing answers identically.
3. Use the same answering model (or report model differences prominently),
   comparable extraction/generation temperatures, ingestion schedule,
   top-k/context/token budget, and cache policy. Do not compare vendor-hosted
   Mem0 numbers directly to a local open-source SDK.
4. Score **formation**, **retrieval evidence**, and **final answers**
   separately: recall@k, precision@k, provenance and supersession, grounded
   answer accuracy, contradiction/false-association rate, abstention,
   p50/p95 latency, model calls and token usage.
5. Include BM25/FTS and vector-only baselines, and preferably the same
   LoCoMo/LongMemEval subset for all runnable systems. Report unavailable
   backends as 'not run'; document hardware and dependency versions.
6. Adjudicate open-ended answers with blinded human review or independently
   validated judges; lexical matching alone is only a smoke test.

## Important cautions
Mem0's published evaluation describes a managed platform and reports headline
LoCoMo/LongMemEval scores. Those are vendor-reported values under a different
stack, not evidence PRIMORA is better or worse. Graphiti specializes in
temporal knowledge graphs; Letta emphasizes agent-managed context. They are
not identical alternatives and must not be reduced to a single feature count.

**Decision standard:** fix only blocker-class evidence corruption, unsupported
claims and memory persistence failures before moving to cognition. Do not
expand memory indefinitely or add hardcoded language/topic rules.
