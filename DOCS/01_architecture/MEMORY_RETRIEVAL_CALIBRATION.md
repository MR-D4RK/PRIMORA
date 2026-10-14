# PRIMORA M6 — Retrieval Calibration

- **Milestone**: M6 — Advanced Retrieval (calibration stage, between M6.1 and M6.4)
- **Status**: **EXPERIMENTAL** measurements and **RECOMMENDED** starting values — **NOT IMPLEMENTED**
- **Project Root**: `C:\PRIMORA PROJECT`
- **Extends**: `DOCS/01_architecture/MEMORY_HYBRID_RETRIEVAL.md`
- **Experiment harness**: `TESTS/prime/retrieval_calibration.py`
- **Date**: 2026-09-25

---

## Label convention

| Label | Meaning |
|---|---|
| **CURRENT** | Verified in this repository's code, by the existing benchmark, or by a measurement in this document. |
| **EXPERIMENTAL** | Produced by an experiment in this document. Not a production behaviour. |
| **RECOMMENDED** | A proposed starting value, derived from the measurements here. Requires approval before implementation. |
| **NOT IMPLEMENTED** | Nothing in `PRIMORA/` changed. No semantic retrieval, no embeddings, no vector index, no RRF, no shadow mode, no RAG. |

Everything in §2–§7 is EXPERIMENTAL. The starting values in §8 are
RECOMMENDED. §10 is NOT IMPLEMENTED.

---

## 1. Experiment design

### 1.1 Purpose

`MEMORY_HYBRID_RETRIEVAL.md` §22 deferred three values as uncalibrated:
the semantic floor τ, the rank-fusion constant k, and the same-session
dominance policy. This document produces evidence for them.

### 1.2 What was run

| # | Experiment | Question |
|---|---|---|
| 1 | Semantic score distributions + τ sweep | Where do relevant and irrelevant similarities separate? |
| 2 | RRF fusion, k ∈ {0, 5, 10, 20, 50, 60, 100} | Does fusion help, and is k identifiable? |
| 2b | Post-fusion relative floor sweep | Can rank-only fusion be made precise? |
| 3 | Same-session policies: current / dominance / gated | Should local evidence outrank foreign evidence? |
| 4 | Entity + relationship structural signal | Does M5 structure add anything, and what is missing? |
| 5 | Hard-gate demonstration | Can a similarity score override a deterministic gate? |

### 1.3 Source of truth

**CURRENT.** The corpus and the gold labels are the existing 47-query
benchmark (`TESTS/prime/retrieval_benchmark.py` and its fixture). The
calibration imports it and never re-declares ground truth. The lexical channel
is the benchmark's read-only mirror of `ContextManager._retrieve`, and the
benchmark's own drift guards run first, so a silent change to production
ranking weights or query limits fails the calibration instead of quietly
skewing it.

### 1.4 Model

**CURRENT**, per the approved M6 decision:
`sentence-transformers/all-MiniLM-L6-v2`, **CPU**, float32, 384 dimensions,
mean pooling, L2-normalized, batch 32, loaded with `HF_HUB_OFFLINE=1`.
Measured on this machine: 1.96 ms per query on CPU, 22.7 M parameters,
87 MB already on disk, `sentence-transformers 6.0.0` already installed.
**EXPERIMENTAL** use only — this is not a production dependency.

### 1.5 Integrity rules applied to every experiment

Two rules were enforced *before* any number was recorded, because both
initially broke the measurement:

1. **Hard gates run first.** `gated_keys()` removes any message that sources a
   superseded state, mirroring production's `_superseded_source_ids`. An
   earlier version of this experiment ranked the raw similarity matrix and
   would have counted 9 gate-blocked messages as recoverable.
2. **Scope is enforced.** The semantic channel may only see same-session
   messages, plus at most 3 foreign messages through the cross-session path.
   An earlier version ranked all 35 corpus messages for every query, which let
   same-session queries reach the `work` session directly — a scope leak that
   production does not have.

**EXPERIMENTAL.** Both rules reduced the measured semantic recall, and the
numbers below are the corrected ones.

---

## 2. Semantic score analysis

**EXPERIMENTAL.** Cosine distributions over the full 47 × 35 query/corpus
matrix (1645 pairs).

| Bucket | n | min | p05 | median | mean | p95 | max |
|---|---|---|---|---|---|---|---|
| relevant (gold pairs) | 40 | 0.1422 | 0.2771 | **0.6216** | 0.5844 | 0.8451 | 0.9455 |
| irrelevant | 1360 | −0.1489 | −0.0464 | **0.0716** | 0.0880 | 0.2826 | 0.6763 |
| no-match query pairs | 245 | −0.1322 | −0.0622 | 0.0443 | 0.0458 | 0.1657 | **0.6464** |
| declared distractors | 40 | −0.0669 | −0.0574 | 0.0615 | 0.0537 | 0.1338 | 0.1478 |

Gold pairs by query class:

| Class | n | min | p05 | median | mean | max |
|---|---|---|---|---|---|---|
| exact | 6 | 0.6638 | 0.6638 | 0.7307 | 0.7304 | 0.7998 |
| punctuation | 5 | 0.5427 | 0.5427 | 0.8443 | 0.8181 | 0.9455 |
| cross-session | 4 | 0.6420 | 0.6420 | 0.6985 | 0.7210 | 0.8451 |
| entity | 4 | 0.4719 | 0.4719 | 0.6313 | 0.6397 | 0.8245 |
| temporal | 4 | 0.5374 | 0.5374 | 0.5707 | 0.6055 | 0.7433 |
| paraphrase | 6 | **0.3344** | 0.3344 | 0.5702 | 0.5399 | 0.6997 |
| false-positive control | 2 | 0.3363 | 0.3363 | 0.4340 | 0.4340 | 0.5317 |
| relationship | 4 | 0.3080 | 0.3080 | 0.3712 | 0.4086 | 0.5841 |
| indirect | 5 | **0.1422** | 0.1422 | 0.2771 | 0.2589 | 0.3730 |

### 2.1 What the distributions say

1. **There is a clean gap in the middle.** Gold p05 = 0.2771 against
   irrelevant p95 = 0.2826. The two distributions barely overlap, and the
   declared distractors sit far below both (max 0.1478).
2. **The overlap is concentrated in one class.** The gold minimum, 0.1422, is
   an *indirect*-reference pair. Indirect reference is the class the lexical
   channel cannot serve at all (measured recall@3 = 0.200) and the class the
   semantic channel also struggles with (0.0–0.2). A single absolute
   threshold cannot rescue it without admitting noise.
3. **No absolute threshold separates gold from no-match.** The strongest
   no-match pair reaches **0.6464**, which is above the *median* gold score.
   Any τ that keeps indirect-reference recall must also let that pair
   through. This is a property of dense retrieval over a 35-item corpus, not
   of this model, and it is the strongest argument for a **relative** floor in
   addition to an absolute one (§4).
4. **When the channel fires, it is right.** In **35 of 40** gold queries the
   gold memory is the *top-1* semantic hit. At τ = 0.45, recall@1 = recall@3 =
   recall@5 = 0.575. The semantic channel is not a ranking problem on this
   fixture; it is purely a *firing-threshold* problem.

---

## 3. Candidate τ values

**EXPERIMENTAL.** Full sweep, semantic channel only, with hard gates and scope
applied. `gold_recall` is the fraction of all 40 gold pairs clearing τ;
`irrelevant_pass` is the fraction of the 1360 irrelevant pairs clearing τ.

| τ | gold_recall | recall@3 | precision@3 | FP@3 | no-match acc | no-match false retrievals | irrelevant_pass |
|---|---|---|---|---|---|---|---|
| 0.20 | 0.975 | 0.725 | 0.725 | 0.7212 | 0.1429 | 6 / 7 | 0.1221 |
| 0.25 | 0.950 | 0.725 | 0.725 | 0.6848 | 0.7143 | 2 / 7 | 0.0743 |
| 0.30 | 0.925 | 0.725 | 0.725 | 0.6234 | 0.7143 | 2 / 7 | 0.0404 |
| 0.35 | 0.825 | 0.675 | 0.675 | 0.4808 | 0.8571 | 1 / 7 | 0.0191 |
| 0.40 | 0.750 | 0.600 | 0.600 | 0.3514 | 0.8571 | 1 / 7 | 0.0096 |
| **0.45** | **0.725** | **0.575** | **0.575** | **0.1481** | **0.8571** | **1 / 7** | **0.0029** |
| 0.50 | 0.675 | 0.525 | 0.525 | 0.1250 | 0.8571 | 1 / 7 | 0.0022 |
| 0.55 | 0.575 | 0.425 | 0.425 | 0.1053 | 0.8571 | 1 / 7 | 0.0015 |
| 0.60 | 0.525 | 0.425 | 0.425 | 0.0556 | 0.8571 | 1 / 7 | 0.0007 |

### 3.1 Reading the tradeoff

- τ ≤ 0.30 buys recall (0.925–0.975) at a catastrophic precision cost: **FP@3
  0.62–0.72**, meaning two of every three admitted items are irrelevant, and
  6 of 7 no-match queries return something.
- τ ≥ 0.50 keeps tightening precision but starts losing whole classes
  (relationship gold lives at 0.308–0.584, so it disappears first).
- **0.35 → 0.45 is the knee.** FP@3 falls from 0.4808 to 0.1481 — a 3.2×
  reduction — for a gold-recall loss of 0.100 (0.825 → 0.725). Below 0.35 the
  floor is doing almost nothing; above 0.45 the curve is nearly flat on
  precision and only losing recall.

### 3.2 The one persistent no-match failure

**EXPERIMENTAL.** One of seven no-match queries never silences, at any τ up to
0.60: **G05 "What is my dentist called?"**, whose top match is `fill_4`
("I still need to call the dentist eventually") at **0.6464**.

This is not a threshold failure. That distractor genuinely is about a dentist,
so the embedding is behaving correctly and the **fixture label is wrong**: G05
is declared no-match while the corpus contains a dentist mention. This is a
defect in the benchmark's gold set, not in the retrieval design.

**RECOMMENDED:** correct G05's gold label (it should point at `fill_4`, or
`fill_4` should be removed from the corpus) and re-run. Until then, no-match
accuracy is capped at 6/7 = 0.8571 for *any* system, and no τ choice can be
blamed for it. The gold set has not been edited here, because the benchmark is
the source of truth and changing it requires its own approval.

---

## 4. RRF experiments

**EXPERIMENTAL.** τ = 0.45, k swept over {0, 5, 10, 20, 50, 60, 100}.

| Config | R@1 | R@3 | R@5 | P@3 | FP@3 | no-match | paraphrase | indirect | entity |
|---|---|---|---|---|---|---|---|---|---|
| A. lexical only | 0.400 | 0.575 | 0.600 | 0.575 | 0.6515 | 0.4286 | 0.3333 | 0.2000 | 0.0000 |
| B. semantic only (τ=0.45) | 0.550 | 0.575 | 0.575 | 0.575 | **0.1481** | **0.8571** | 0.5000 | 0.0000 | 0.5000 |
| C. lexical + semantic, k=0…100 | 0.600 | 0.650 | 0.675 | 0.650 | 0.6176 | 0.4286 | 0.5000 | 0.2000 | 0.5000 |

### 4.1 Findings

1. **The semantic channel alone matches lexical recall with 4.4× lower
   false positives.** B ties A on recall@3 (0.575) and beats it on FP@3
   (0.1481 vs 0.6515) and no-match accuracy (0.8571 vs 0.4286). It loses
   indirect reference entirely (0.0), which lexical partially serves (0.200).
2. **Fusion improves recall but re-imports the flood.** C lifts recall@3 from
   0.575 to 0.650 — a real gain, and paraphrase recall doubles from 0.333 to
   0.500. But FP@3 lands at 0.6176, essentially lexical-only, and no-match
   accuracy falls back to 0.4286. **RRF is a union, and the lexical channel's
   false positives survive it**, because RRF only sees ranks: an item ranked
   first by accident in the lexical channel scores 1/(k+1) and enters the fused
   top-3 regardless of how weak its evidence was. This is the M6 audit's
   weakness W3, reproduced under fusion.
3. **k is not identifiable on this fixture.** Every value in the grid produced
   an **identical** metric signature — one distinct outcome across seven k
   values. With 35 candidates and two channels that rarely disagree at the
   top, rank differences are small relative to any k ≥ 0, so the ordering is
   invariant. **Reporting a "calibrated k" from this experiment would be
   false.** §4.2 explains what would be needed.

### 4.2 Post-fusion relative floor

**EXPERIMENTAL.** A deterministic post-fusion filter: keep items whose fused
score is at least `ratio ×` the best item's fused score. No new score
calibration, no model, applied after fusion and before budget admission.

| ratio | R@1 | R@3 | R@5 | P@3 | FP@3 | no-match | paraphrase | indirect | entity |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 (no floor) | 0.600 | 0.650 | 0.675 | 0.650 | 0.6176 | 0.4286 | 0.5000 | 0.2000 | 0.5000 |
| 0.50 | 0.600 | 0.650 | 0.675 | 0.650 | 0.5439 | 0.4286 | 0.5000 | 0.2000 | 0.5000 |
| **0.60 – 0.95** | **0.600** | **0.650** | **0.675** | **0.650** | **0.4583** | 0.4286 | 0.5000 | 0.2000 | 0.5000 |
| 1.00 (top item only) | 0.600 | 0.600 | 0.600 | 0.600 | 0.3143 | 0.4286 | 0.5000 | 0.0000 | 0.5000 |

**This is the most actionable result in the document.** Across the whole
plateau from 0.60 to 0.95, the false-positive rate drops from 0.6176 to
**0.4583 — a 26% reduction with literally zero recall loss at k=1, 3 and 5.**
Only at ratio 1.00 does precision improve further, and it costs recall@3.

---

## 5. Same-session policy

**EXPERIMENTAL.** Three policies applied to the fused ranking:

- **current** — same-session items first, then at most 3 foreign items
  (exactly what production does today).
- **dominance** — foreign items admitted only when the same-session list is
  empty.
- **gated** — foreign items admitted only when the same-session list has
  fewer than 3 items ("insufficient local evidence").

| Config | same-session R@3 | cross-session R@3 | foreign outranks local gold | FP@3 | no-match | overall R@3 |
|---|---|---|---|---|---|---|
| lexical + current | 0.5278 | 1.0000 | 0 | 0.6515 | 0.4286 | 0.575 |
| lexical + dominance | 0.5278 | 0.7500 | 0 | 0.6508 | 0.4286 | 0.550 |
| lexical + gated | 0.5278 | 1.0000 | 0 | 0.6515 | 0.4286 | 0.575 |
| hybrid + current | 0.6111 | 1.0000 | 0 | 0.6176 | 0.4286 | 0.650 |
| hybrid + dominance | 0.6111 | 0.7500 | 0 | 0.6154 | 0.4286 | 0.625 |
| hybrid + gated | 0.6111 | 1.0000 | 0 | 0.6176 | 0.4286 | 0.650 |

### 5.1 Findings

1. **The M6 audit's scope-leakage concern (W10) does not materialize.** Zero
   cases in any configuration where a foreign memory outranks the relevant
   local memory. The current sort already places all same-session hits before
   the foreign tail, so the feared failure mode is not observable here.
2. **`gated` is byte-identical to `current` on every metric.** The trigger
   condition "fewer than 3 same-session candidates" essentially never fires on
   this fixture, so the gate is **untested by this benchmark**. It is not
   evidence that gating is useless; it is evidence that this benchmark cannot
   distinguish it.
3. **`dominance` is a bad trade.** It costs 0.25 of cross-session recall
   (1.0000 → 0.7500) and 0.025 of overall recall@3, and buys 0.0007–0.0022 of
   false-positive reduction. Adopting it would be paying real recall for noise.

### 5.2 What the fixture cannot answer

**EXPERIMENTAL limitation.** The 4 cross-session queries have almost no
same-session competitors, so the foreign tail is never under pressure. To
identify a dominance policy the benchmark needs adversarial cases: queries
whose correct answer is local *and* which have strong foreign competitors.
The M6 spec's decision D2 cannot be closed on this evidence.

---

## 6. Entity and relationship structural signal

**EXPERIMENTAL.** M5's bounded query surface, used read-only and mapped back
to corpus messages through `source_ids` so provenance is preserved.

| Measurement | Result |
|---|---|
| Declared entity names + alias in the fixture | 6 |
| Resolved when the name is given directly | **6 / 6** |
| Unresolved when the name is given directly | 0 |
| Corpus messages reachable through structure | `brakes`, `fox`, `berlin`, `reno` |
| Targeted entity/relationship queries | 8 |
| Of those, resolved from the **free-text query** | **0 / 8** |

| Config | R@1 | R@3 | R@5 | P@3 | FP@3 | entity R@3 | relationship R@3 |
|---|---|---|---|---|---|---|---|
| structural only | 0.225 | 0.225 | 0.225 | 0.225 | 0.2500 | 0.7500 | **1.0000** |
| lexical + semantic | 0.600 | 0.650 | 0.675 | 0.650 | 0.6176 | 0.5000 | 0.5000 |
| lexical + semantic + structural | **0.625** | **0.775** | 0.800 | **0.775** | 0.5694 | **1.0000** | **1.0000** |

### 6.1 Findings

1. **M5 resolution is not the bottleneck.** Given a name, all 6 declared names
   resolve, and structure reaches exactly the 4 gold memories the targeted
   queries need.
2. **The bottleneck is mention detection.** Zero of the 8 targeted queries
   resolve an entity from the free-text query, because M5's `resolve_entity`
   is exact-name resolution by design (Stage C pins this, and
   `resolve_concept` must not be reused). A whole sentence is not a name.
3. **The structural channel is a real win where it applies.** As a third
   fusion channel it lifts overall recall@3 from 0.650 to **0.775**,
   relationship recall from 0.500 to **1.000**, entity recall from 0.500 to
   **1.000**, and *reduces* FP@3 from 0.6176 to 0.5694. It is simultaneously
   the most precise channel (structural-only FP@3 = 0.2500) and, on the two
   classes it targets, the most accurate.
4. **No benchmark case requires a new entity or relationship type.** All 8
   targeted queries reference entities the fixture already declares. The gap
   is a retrieval-side mention-detection step, not ontology. No new
   relationship type, no new entity type, and no `related_to` is implied by
   any measurement here.

### 6.2 Cases needing expanded coverage

| Query | Class | Why it fails today | What is actually missing |
|---|---|---|---|
| D01 | entity | "Berlin Office Move" is embedded in a sentence | mention detection |
| D02 | entity | "Apartment Renovation" likewise | mention detection |
| D03 | entity | "Emblem Redesign" likewise | mention detection |
| D04 | entity | alias "Office Move" is not matched in free text | mention detection incl. aliases |
| E01 | relationship | "Brake Assembly" not extracted from the sentence | mention detection |
| E02 | relationship | same, plus needs the derived `contains` inverse | mention detection; inverse already exists |
| E03 | relationship | "Porsche Maintenance" not extracted | mention detection |
| E04 | relationship | "Emblem Redesign" not extracted | mention detection |

One corpus-level note: the closed M5 entity vocabulary (`PERSON`, `AGENT`,
`PROJECT`, `COMPONENT`, `FILE`) cannot represent a car or a house as an
entity. The benchmark deliberately routes those through `PROJECT` and
`COMPONENT` to stay inside the approved vocabulary. A future decision about
whether vehicles and places warrant entity types is **out of scope here** and
is not required by any current measurement.

---

## 7. Hard gates

**EXPERIMENTAL.** The invariant under test: *a similarity score may never
re-admit anything a deterministic M1–M4 gate removed.*

| Measurement | Value |
|---|---|
| Gold pairs | 40 |
| Gold pairs removed by the M4 supersession gate | 11 |
| Of those, pairs whose similarity **exceeds τ = 0.45** | **9** |

Examples that would have been wrongly admitted if gates ran after scoring:

| Query | Gold message | Similarity | Gated | Passes τ |
|---|---|---|---|---|
| A02 | `reno` | 0.7998 | yes | yes |
| A05 | `fox` | 0.7096 | yes | yes |
| D02 | `reno` | 0.6685 | yes | yes |
| B02 | `reno` | 0.6437 | yes | yes |
| D03 | `fox` | 0.5940 | yes | yes |
| E04 | `fox` | 0.5841 | yes | yes |

**Nine of eleven** gated gold pairs clear the recommended floor. Ordering
gates after fusion would therefore manufacture nine confident wrong answers
from the strongest evidence in the corpus. Gates must be applied **before**
candidate scoring, exactly as §1.5 enforced for these measurements.

### 7.1 Concept separation

**CURRENT**, and unchanged by anything in this document:

| Concept | Role in M6 retrieval |
|---|---|
| Authority (`user`) | Absolute gate. Never scored, never boosted. |
| Confidence (M1, 1.0 / 0.3) | Gate for currentness. **Never a similarity threshold and never a rank input.** The calibration reads no confidence value at all; a test enforces this. |
| Temporal validity (M4) | Absolute gate. Half-open intervals. |
| Lifecycle status | Absolute gate. `candidate`, `rejected`, `unresolved` are never retrieved. |
| Importance (M1) | Ranking tie-break only. |
| **Relevance** | The only thing the ranker measures. |
| **Retrieval score** | Internal ordering artifact. Never persisted, never in a prompt, never changes memory. |

---

## 8. Recommended starting values

### 8.1 τ — semantic positive-evidence floor

**RECOMMENDED: τ = 0.45**

Selection rule, stated so it is reproducible and re-runnable:

> the lowest τ whose FP@3 ≤ 0.15 while gold-pair recall ≥ 0.70

Measured evidence at τ = 0.45: gold-pair recall **0.725**, recall@3
**0.575**, FP@3 **0.1481**, no-match accuracy **0.8571**, irrelevant pass rate
**0.0029**.

Why this value:

1. It is the FP/recall knee (§3.1). 0.35 → 0.45 cuts FP@3 by 3.2× for 0.100
   of gold recall; 0.45 → 0.60 buys only 0.093 more precision reduction for
   0.200 of recall.
2. It is a *conservative floor*, as the M6 spec requires. Above 0.45 the
   semantic channel contributes almost nothing new (irrelevant pass 0.29%),
   so the floor can be raised later without redesign.
3. It keeps the two classes lexical retrieval is worst at: paraphrase gold
   reaches 0.6997 and relationship gold 0.5841, both comfortably above 0.45.
4. Indirect-reference gold (median 0.2771) is *not* recoverable at this τ, and
   §2.1 explains why no absolute τ recovers it. That class needs a different
   mechanism, not a looser floor.
5. It is a *floor*, not a selector. Combined with a relative floor (§8.3) the
   admitted set is bounded on both sides.

### 8.2 k — rank-fusion constant

**RECOMMENDED: k = 60, explicitly labelled uncalibrated.**

**EXPERIMENTAL:** all seven values {0, 5, 10, 20, 50, 60, 100} produced one
identical metric signature. k is **not identifiable** on this fixture, so no
experiment here can justify a specific value.

60 is recommended because it is the established default for rank fusion, it is
insensitive on small candidate sets (which is why every swept value agreed
here), and it is a conservative choice: larger k flattens the contribution
differences between channels, which is the safer failure mode when the channel
quality gap is not yet trusted.

**What would make k identifiable:** a corpus large enough that the two
channels disagree across dozens of candidates, with near-duplicate and
long-tail items that appear at different depths in each list. The current
35-message fixture has 1645 total pairs and near-total channel agreement at
the top. Until the benchmark grows, treat k as a placeholder to be revisited,
not a tuned parameter.

### 8.3 Post-fusion relative floor

**RECOMMENDED: ratio = 0.725** (midpoint of the measured plateau 0.60–0.95).

Selection rule: the midpoint of the widest plateau where recall@1 and recall@3
are unchanged from no floor and FP@3 strictly improves.

Measured effect: FP@3 **0.6176 → 0.4583** with recall@1, recall@3 and recall@5
all unchanged. The midpoint is chosen because every ratio inside the plateau
scores identically, so the midpoint is the value furthest from either edge and
the most robust to a corpus that shifts slightly.

This is the highest-value recommendation in the document because it is free:
it removes more than half of the fusion false positives at no measured recall
cost, using one deterministic comparison and no new score calibration.

### 8.4 Same-session policy

**RECOMMENDED: keep `current` behaviour. Do not adopt strict dominance.**

Evidence (§5.1): zero measured cases of foreign evidence outranking relevant
local evidence; `gated` is indistinguishable from `current` on this fixture;
`dominance` costs 0.25 of cross-session recall and 0.025 of overall recall@3
for at most 0.0022 of false-positive reduction.

This is a **provisional** recommendation. The benchmark lacks the adversarial
cross-session cases needed to close M6 decision D2, so the correct action is to
extend the benchmark and re-measure before changing behaviour. The measured
evidence gives no reason to make a change now.

### 8.5 Summary

| Value | Recommendation | Basis |
|---|---|---|
| τ (semantic floor) | **0.45** | Calibrated. FP/recall knee. |
| k (rank fusion) | **60** | **Not calibrated.** Literature default; not identifiable here. |
| Post-fusion ratio | **0.725** | Calibrated. Free 26% FP reduction. |
| Same-session policy | **keep current** | Provisional. No evidence for a change. |
| Structural channel | **yes, as a third channel** | Calibrated. R@3 0.650 → 0.775. |
| Mention detection | **required before the structural channel pays off** | 0 / 8 targeted queries resolve from free text. |

---

## 9. Expected limitations

1. **Fixture size.** 47 queries and 35 messages. Enough to find a knee, not
   enough to identify k or to close the same-session decision.
2. **One embedding model.** All measurements are for
   `all-MiniLM-L6-v2` on English prose. A different model, a different
   language, or a different corpus domain will move every number. τ is a
   property of *(model, corpus)*, not of M6.
3. **The gold set has a known defect.** G05 is mislabelled (§3.2), capping
   no-match accuracy at 0.8571 for any system.
4. **Indirect reference is unsolved.** Median gold similarity 0.2771, below
   every recommended threshold, and recall 0.0–0.2 in every configuration.
   This is the largest remaining gap and no value of τ closes it.
5. **The mirror, not production.** The lexical channel is the benchmark's
   mirror of `_retrieve`, guarded against drift but still a re-implementation.
   Its one documented deviation is that `selected_texts` holds only the current
   turn, so redundancy is slightly under-penalized relative to production.
6. **Fusion was evaluated on two channels.** The structural channel was
   measured separately; a three-channel k sweep was not run, and k's
   insensitivity suggests it would also be flat.
7. **No end-to-end validation.** Nothing here measures what a fused ranking
   does to `ContextView` under a real budget. The M6.3 shadow stage exists for
   exactly that, and must not be skipped.
8. **Single-run latency.** Latency figures come from the benchmark's channel
   mode only and exclude embedding the query, which the M6 audit measured
   separately at 1.96 ms per query on CPU.

---

## 10. Implementation guidance for a future M6 hybrid

**NOT IMPLEMENTED.** Guidance only, for the stage that is separately approved.

### 10.1 Order of operations

The experiments above fix the order, because gates must precede scores (§7):

```
1. resolve explicit scope (session, plus global only if requested)
2. apply deterministic gates: authority, lifecycle status, M4 validity,
   supersession, entity endpoint state          <- absolute, never scored
3. run channels independently:
     A lexical  (existing FTS/LIKE, unchanged)
     B semantic (cosine, filtered at tau = 0.45)
     D entity   (M5 bounded queries, mention detection required)
     E relationship (M5 bounded one-hop, derived contains at read time)
4. union by item key, recording per-channel ranks
5. fuse with RRF, k = 60
6. apply the relative floor at 0.725
7. bounded tie-break layer: structural signal, stored importance, recency
   (never confidence, never authority, never currentness)
8. redundancy suppression
9. admit only while budget remains
```

### 10.2 Non-negotiable rules

- **Gates before scores.** Nine of eleven gated gold pairs clear τ. A
  post-scoring gate is not a safe ordering, it is a correctness bug.
- **No confidence in ranking.** A confidence value must never raise a score,
  select a threshold, or substitute for authority.
- **One hop.** The relationship channel inherits M5's bounds unchanged.
- **Derive, never persist.** `contains` is computed at read time.
- **Degrade, never fail.** A missing index, a failed embedding, or a model
  mismatch must fall back to lexical-only with a diagnostic, never fail a turn.
- **Scope before content.** Foreign-session material is admitted only through
  the existing capped, provenance-tagged path.
- **Determinism.** Identical inputs, identical ordering, always. The drift
  guards in the benchmark must keep passing.

### 10.3 Staging

Per the M6 stage table, and unchanged by this document:

| Stage | Deliverable | Gate to pass |
|---|---|---|
| M6.2 | Vector store, indexer, lexical fixes | index is off the query path |
| M6.3 | Semantic channel in shadow mode | trace-only, zero output change; deltas measured against §3–§4 |
| M6.4 | Gated hybrid in the query path | τ = 0.45, ratio = 0.725; lexical lock holds; FP@3 must not exceed lexical-only 0.6515 and no-match accuracy must not fall below 0.4286 |
| M6.5 | Entity/relationship channel | requires mention detection first (§6.1) |
| M6.6 | Device switch, larger-model evaluation | only with benchmark evidence of a quality gap |

### 10.4 What must be re-measured before M6.4

1. Fix the G05 gold label, then re-baseline no-match accuracy.
2. Add adversarial cross-session cases (local gold with strong foreign
   competitors) to close decision D2.
3. Add enough paraphrase and indirect-reference queries to populate the τ
   confidence interval rather than a point estimate.
4. Re-run the k sweep on a corpus large enough for the channels to disagree.
5. Re-check the τ knee after any corpus growth; τ is not a constant.

---

## 11. Reproducing

```powershell
python TESTS\prime\retrieval_calibration.py     # full report
python -m pytest TESTS\prime\test_retrieval_calibration.py -q
```

The harness is offline (`HF_HUB_OFFLINE=1`), CPU-only, writes exclusively to a
temporary directory, and hashes the production database before and after every
run. The tests skip cleanly if the model is not present in the local cache.

**CURRENT.** Production database SHA-256, identical before and after every
experiment in this document:
`a753287e754a12492ed84eb19e547c4556b509e9cd973d7610009bfaa87ca353`
