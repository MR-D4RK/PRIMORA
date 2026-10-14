"""Generate the Aurora Brain v0.1 dataset from the approved 278-seed bank.

Not a trainer. This produces the material and refuses to produce anything it
cannot verify.

Relationship to ``generate.py``
------------------------------
``generate.py`` is the legacy pipeline for the *old* 3,400-record corpus. It
imports ``design`` (superseded), has no reference to ``design_v2``, and carries
per-category builders for the old nine categories. Its output is the historical
corpus and must not be regenerated, merged, or reused as training data.

What this file reuses from it, rather than reinventing:

*   ``request_block`` / ``tool_turn`` - the canonical ``<prime_request>`` wire
    format and tool-turn spelling, so the emitted strings are byte-identical to
    what the runtime parses.
*   ``content_fingerprint`` - normalised-content hashing for duplicate detection.
*   the isolation discipline: ``harness_session``, a blocked provider, and a
    production-digest comparison before and after.
*   the refusal discipline: validation problems block writing entirely.
*   split-by-seed, so paraphrases of one seed cannot straddle two splits.

What this file adds, because the old pipeline has no equivalent:

*   reads ``design_v2`` and the 278 approved seeds;
*   records the real composed system turn as a reference key plus a recorded
    fingerprint, per ``RUNTIME_FORMAT["system_turn_policy"]``;
*   carries ``train_on`` loss-masking through to the emitted record;
*   three splits, including a protected holdout that is written once and never
    read back into generation decisions.

Run: python TESTS/training/generate_v2.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

sys.path.insert(0, r"C:\PRIMORA PROJECT")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from TESTS.training.design_v2 import (  # noqa: E402
    CATEGORY_BY_KEY,
    EXPANSION_PER_SEED,
    HOLDOUT_SIZE,
    RUNTIME_FORMAT,
    SEEDS,
    SEED_BUDGET,
    TRAIN_MAX,
    VALIDATION_SIZE,
)
from PRIMORA.prime.context.isolation import harness_session  # noqa: E402

GENERATOR_VERSION = "aurora-brain-v0.1-seedgen-1"
SCHEMA_VERSION = "aurora.train.v2"
DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "v2")

# --------------------------------------------------------------------------
# Canonical runtime strings. Copied in behaviour, not in text: these must match
# generate.py exactly or the emitted records stop being parseable by
# extract_prime_request.
# --------------------------------------------------------------------------


def request_block(action: str, payload: dict) -> str:
    return (
        "<prime_request>\n"
        + json.dumps(
            {
                "action": action,
                "payload": payload,
                "context": None,
                "metadata": None,
            },
            ensure_ascii=False,
        )
        + "\n</prime_request>"
    )


def tool_turn(status: str, output: Any = None, error: Any = None) -> str:
    """Serialise a tool turn exactly as CapabilityExecutor.execute does.

    The runtime builds the turn with::

        json.dumps({"status": ..., "output": ..., "error": ...}, default=str)

    - that key order, no indent, and ``default=str`` for non-serialisable
    output. Reproducing it here is deliberate: a second serialisation format
    would make every trained tool turn a shape Aurora never sees at inference.

    No ChatML markup belongs in this string. The Qwen3 chat template supplies
    ``<tool_response>`` itself; the earlier ``<tool_result>`` wrapper nested a
    second set of tags inside it that the runtime has never produced.
    """
    if status not in ("success", "failure"):
        raise ValueError(f"tool status must be success or failure, got {status!r}")
    return json.dumps(
        {"status": status, "output": output, "error": error},
        default=str,
    )


def content_fingerprint(text: str) -> str:
    normalised = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    normalised = re.sub(r"\s+", " ", normalised).strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Situation variation banks.
#
# The old corpus's lesson was that 3,400 examples from 1,506 distinct prompts is
# a template farm. These banks vary the *situation*, not the words: each entry
# is a different concrete circumstance of the same behavioural lesson, written
# out rather than assembled by swapping nouns.
#
# Framing banks change how the user makes the request. A direct ask, a tired
# one, a hedged one and a context-first one are genuinely different situations
# that call the same behaviour for the same reason.
# --------------------------------------------------------------------------

FRAMES = {
    "direct": [
        "{ask}",
        "{ask}",
        "Straight question: {ask}",
        "{ask} - and I'd like a straight answer.",
    ],
    "tired": [
        "Long day. {ask}",
        "Quick one before I log off - {ask}",
        "Sorry, brain's gone. {ask}",
        "Bit exhausted. {ask}",
    ],
    "hedged": [
        "If it's easy - {ask}",
        "Only if you already know: {ask}",
        "No need to dig if not: {ask}",
        "Off the top of your head, {ask}",
    ],
    "polite": [
        "When you get a moment - {ask}",
        "Could you help me with something? {ask}",
        "Sorry to ask again - {ask}",
        "If you've got time, {ask}",
    ],
    "context_first": [
        "Been thinking about this all morning. {ask}",
        "Right, situation: we finally got the survey back. {ask}",
        "Context first, because it matters here. {ask}",
        "Something came up while I was in the shed. {ask}",
    ],
    "direct_sharp": [
        "{ask} Straight answer.",
        "{ask} No preamble needed.",
    ],
    "tired_apologetic": [
        "Sorry, had a long one. {ask}",
        "Knocked off early today. {ask}",
    ],
    "hedged_low": [
        "Not sure it matters, but {ask}",
        "Only if it's easy - {ask}",
    ],
    "polite_formal": [
        "Would you mind checking something? {ask}",
        "If I may ask - {ask}",
    ],
    "context_first_short": [
        "One thing first: {ask}",
        "Before I get back to it - {ask}",
    ],
    "direct_plain": [
        "Quick: {ask}",
        "Single question: {ask}",
    ],
    "hedged_sure": [
        "I'm sure you know this better than me, but {ask}",
        "Correct me if I'm wrong - {ask}",
    ],
    "polite_casual": [
        "Can I pick your brain? {ask}",
        "Second opinion, if you've got a moment? {ask}",
    ],
    "context_first_explain": [
        "Reason it out with me. {ask}",
        "Talk me through it. {ask}",
    ],
}

# Concrete particulars, used where the seed's lesson tolerates a different
# object. Deliberately mundane and factual so no example can teach a falsehood.
ITEMS = [
    "the van", "the survey", "the invoice", "the shed roof", "the vineyard shed",
    "the berth booking", "the storage split", "the notice period",
    "the coffee order", "the deck figure", "the bank details",
    "the panel quote", "the release notes", "the contractor's list",
    "the vitrinedisplay", "the boundary fence", "the water bill",
    "the delivery slot", "the insurance renewal", "the tooling hire",
    "the site survey", "the budget line", "the second batch",
    "the packing list", "the service record", "the tenancy deposit",
    "the equipment order", "the maintenance log", "the stock check",
    "the meter reading", "the rental invoice", "the tender document",
    "the boat lift", "the workshop lease", "the feed delivery",
]

VALUES = ["412", "241", "5,400", "4,200", "three", "six", "fourteen", "nine",
          "22 mm", "28 mm", "3.13", "3.12.4", "foghorn-42", "foghorn-17"]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def frame_ask(rng: random.Random, ask: str, style: str) -> str:
    bank = FRAMES.get(style, FRAMES["direct"])
    return rng.choice(bank).format(ask=ask)


def strip_amir_prefix(text: str) -> str:
    """Seeds quote Amir as 'Amir: ...'. Records store the user turn verbatim,
    so the prefix is kept only where it is part of the speech itself."""
    return text.strip()


# --------------------------------------------------------------------------
# Record
# --------------------------------------------------------------------------


@dataclass
class Example:
    id: str
    seed_id: str
    category: str
    split: str
    messages: list[dict[str, str]] = field(default_factory=list)
    # Indexes into the assistant turns of this record that carry loss.
    train_on: list[int] = field(default_factory=list)
    expect_block: bool = False
    expect_action: Optional[str] = None
    note: str = ""
    variant: int = 0
    system_ref: str = ""
    system_fingerprint: str = ""
    seed_priority: str = "normal"

    @property
    def assistant_turn_count(self) -> int:
        return sum(1 for m in self.messages if m.get("role") == "assistant")

    def trained_assistant_indexes(self) -> list[int]:
        return [i for i in self.train_on if i < self.assistant_turn_count]

    def fingerprint(self) -> str:
        blob = json.dumps(self.messages, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Split assignment: by seed, so no paraphrase straddles two splits.
# --------------------------------------------------------------------------

TRAIN_FRACTION = 0.74


TRAIN_TARGET = 1400


def assign_split(seed_id: str) -> str:
    """Split decided by rank on a stable hash, so counts are exact.

    A threshold on a hash fraction cannot hit an exact split size - it only gets
    close, and the shortfall silently moves examples between splits. Ranking
    every seed on the same hash and taking a prefix does give exact counts and
    stays deterministic.
    """
    return SPLIT_OF.get(seed_id, "train")


SPLIT_OF: dict[str, str] = {}


def _ranked_seeds() -> list[str]:
    return sorted(
        (s.seed_id for s in SEEDS if s.training_allowed),
        key=lambda sid: hashlib.sha256(sid.encode()).hexdigest(),
    )


def plan_expansion() -> dict[str, int]:
    """Per-seed example counts hitting the three split targets exactly.

    Validation and holdout are fixed-size and use whole seeds at a fixed depth.
    Train absorbs the remainder and is grown to TRAIN_TARGET without exceeding
    TRAIN_MAX. The review-only anchor is excluded: it is never emitted.
    """
    ranked = _ranked_seeds()
    base, top = EXPANSION_PER_SEED

    per: dict[str, int] = {}
    SPLIT_OF.clear()

    val_depth = base
    n_val = int(round(VALIDATION_SIZE / val_depth * 1.6))   # reserve headroom
    hold_depth = base
    n_hold = int(round(HOLDOUT_SIZE / hold_depth * 1.6))  # reserve headroom

    for sid in ranked[:n_val]:
        per[sid] = val_depth
        SPLIT_OF[sid] = "validation"
    for sid in ranked[n_val:n_val + n_hold]:
        per[sid] = hold_depth
        SPLIT_OF[sid] = "holdout"

    train_seeds = ranked[n_val + n_hold:]
    for sid in train_seeds:
        per[sid] = base
        SPLIT_OF[sid] = "train"

    # Grow train toward the target, deterministically by rank order.
    shortfall = TRAIN_TARGET - sum(per[s] for s in train_seeds)
    i = 0
    while shortfall > 0 and train_seeds:
        sid = train_seeds[i % len(train_seeds)]
        if per[sid] < top:
            per[sid] += 1
            shortfall -= 1
        i += 1
        if i > len(train_seeds) * (top - base) + 5:
            break

    train_total = sum(per[s] for s in train_seeds)
    if train_total > TRAIN_MAX:
        raise SystemExit(
            f"train plan {train_total} exceeds TRAIN_MAX {TRAIN_MAX}")

    return per


# --------------------------------------------------------------------------
# System context: built from the real runtime, not copied.
#
# RUNTIME_FORMAT["system_turn_policy"] requires the system turn to be stored as
# a reference key plus a recorded fingerprint, with the loader expanding it and
# verifying the fingerprint before use. The composed instruction here comes from
# AuroraConfig.system_instruction plus the three real builders, assembled the
# same way Aurora._with_instruction does it, so the fingerprint is meaningful.
# --------------------------------------------------------------------------


def build_system_context() -> tuple[str, str, list[str]]:
    """Return (reference key, composed instruction, registered capabilities)."""
    import tempfile

    from PRIMORA.aurora.configuration import (
        AuroraConfig,
        build_capability_instruction,
        build_specialist_instruction,
        build_tool_instruction,
    )
    from PRIMORA.consult import SpecialistRegistry, SpecialistSpec
    from PRIMORA.prime import Prime
    from PRIMORA.prime.context.isolation import build_prime_with_isolation_guard
    from PRIMORA.prime.contract import PrimeResult
    from PRIMORA.prime.providers import Provider
    from PRIMORA.research import ResearchProvider, ResearchResult
    from PRIMORA.toolbox import ToolRegistry

    class _Blocked(Provider):
        name = "generator"

        def invoke(self, capability, payload):
            return PrimeResult.failure("generation: execution disabled")

    class _Web(ResearchProvider):
        name = "generator"

        def search(self, query, limit=5):
            return [ResearchResult("t", "https://x.test/", "s", "x.test")]

    root = os.path.join(tempfile.gettempdir(), "aurora_gen_v2")
    os.makedirs(root, exist_ok=True)
    with harness_session("gen_v2_", root=root) as session:
        context = session.context_manager()
        prime = build_prime_with_isolation_guard(
            _Blocked(),
            research_provider=_Web(),
            specialists=SpecialistRegistry([
                SpecialistSpec(name="reasoning",
                               description="Hard analysis.", model="m")]),
            tools=ToolRegistry.with_builtins(),
        )
        prime._context = context
        prime._mind = prime._mind.__class__(context)
        for name in ("recall", "learn"):
            cap = prime.registry.get_specialist(name)
            if cap is not None:
                cap.bind(prime._mind)
        config = AuroraConfig(model_path="generator", adapter_path="generator")
        composed = (
            config.system_instruction
            + build_capability_instruction(prime)
            + build_specialist_instruction(prime)
            + build_tool_instruction(prime)
        )
        capabilities = sorted(prime.registry.get_names())

    fingerprint = hashlib.sha256(composed.encode("utf-8")).hexdigest()
    key = f"aurora.system.v1@{fingerprint[:16]}"
    return key, composed, capabilities


# --------------------------------------------------------------------------
# Generation driver
# --------------------------------------------------------------------------

FRAME_STYLES = [
    "direct", "tired", "hedged", "polite", "context_first",
    "direct_sharp", "tired_apologetic", "hedged_low", "polite_formal",
    "context_first_short", "direct_plain", "hedged_sure", "polite_casual",
    "context_first_explain",
]


def _rng_for(seed_id: str) -> random.Random:
    """Deterministic per seed, so a rerun reproduces byte-identical output."""
    return random.Random(int(hashlib.sha256(seed_id.encode()).hexdigest()[:16], 16))


def generate_all(system_key: str, system_fp: str, per_seed: dict[str, int]) -> list[Example]:
    import builders_v2 as B

    examples: list[Example] = []
    rejected: list[tuple[str, str]] = []

    for seed in SEEDS:
        # The review-only anchor is never emitted as training data. This is the
        # whole point of C15-R02.
        if not seed.training_allowed:
            continue

        depth = per_seed.get(seed.seed_id, EXPANSION_PER_SEED[0])
        if seed.seed_id in B.EMIT_CAPABILITY:
            kind = "emit"
        elif seed.seed_id in B.TOOL_SPEC:
            kind = "tool"
        elif seed.seed_id in B.CYCLE_SPEC:
            kind = "cycle"
        elif seed.seed_id in B.PROSE_TARGETS:
            kind = "prose"
        else:
            rejected.append((seed.seed_id, "no builder"))
            continue

        for variant in range(depth):
            rng = _rng_for(f"{seed.seed_id}:{variant}")
            rngs = {"style": lambda v: FRAME_STYLES[v % len(FRAME_STYLES)]}
            try:
                if kind == "emit":
                    ex = B.emit_example(seed, variant, rng, rngs)
                elif kind == "tool":
                    ex = B.tool_example(seed, variant, rng, rngs)
                elif kind == "cycle":
                    ex = B.cycle_example(seed, variant, rng, rngs)
                else:
                    ex = B.prose_example(seed, variant, rng, rngs)
            except Exception as exc:  # a broken builder must not ship data
                rejected.append((f"{seed.seed_id}-{variant}", f"{type(exc).__name__}: {exc}"))
                continue

            ex.system_ref = system_key
            ex.system_fingerprint = system_fp
            ex.split = assign_split(seed.seed_id)
            examples.append(ex)

    generate_all.rejected = rejected  # type: ignore[attr-defined]
    return examples


# --------------------------------------------------------------------------
# Validation before writing. The old generator refuses to write on any problem;
# the same discipline applies here.
# --------------------------------------------------------------------------


def validate_example(example: Example) -> list[str]:
    problems: list[str] = []
    msgs = example.messages

    if not msgs:
        return ["no messages"]
    if msgs[0]["role"] != "user":
        problems.append("record does not begin on a user turn")

    # RUNTIME_FORMAT["last_turn_rule"]: every example ends on the trained turn.
    if msgs[-1]["role"] != "assistant":
        problems.append("record does not end on an assistant turn")

    assistant_count = sum(1 for m in msgs if m["role"] == "assistant")
    if assistant_count == 0:
        problems.append("no assistant turn")
    trained = example.trained_assistant_indexes()
    if not trained:
        problems.append("no assistant turn carries loss")
    if trained and max(trained) != assistant_count - 1:
        problems.append(
            f"loss stops at assistant turn {max(trained)} of {assistant_count}")

    # A record must emit at most one request block per reply, or the runtime
    # rejects the whole message. C2 exists to teach exactly this.
    for m in msgs:
        if m["role"] == "assistant" and m["content"].count("<prime_request>") > 1:
            problems.append("more than one request block in one reply")

    # Loss must never land on a turn the seed marks as a deliberate failure.
    if example.seed_id.startswith("C15"):
        seed = next((s for s in SEEDS if s.seed_id == example.seed_id), None)
        if seed is not None and seed.train_on is not None and 0 in trained:
            problems.append("loss lands on the deliberately failed request")

    if not example.system_fingerprint:
        problems.append("missing system fingerprint")

    return problems


def validate_set(examples: list[Example]) -> list[str]:
    problems: list[str] = []
    seen_ids: set[str] = set()
    for ex in examples:
        for problem in validate_example(ex):
            problems.append(f"{ex.id}: {problem}")
        if ex.id in seen_ids:
            problems.append(f"{ex.id}: duplicate example id")
        seen_ids.add(ex.id)

    # No paraphrase may straddle two splits.
    by_seed: dict[str, set[str]] = {}
    for ex in examples:
        by_seed.setdefault(ex.seed_id, set()).add(ex.split)
    for sid, splits in by_seed.items():
        if len(splits) > 1:
            problems.append(f"{sid}: examples split across {sorted(splits)}")

    counts = {s: sum(1 for e in examples if e.split == s)
              for s in ("train", "validation", "holdout")}
    if counts["validation"] != VALIDATION_SIZE:
        problems.append(f"validation is {counts['validation']}, expected {VALIDATION_SIZE}")
    if counts["holdout"] != HOLDOUT_SIZE:
        problems.append(f"holdout is {counts['holdout']}, expected {HOLDOUT_SIZE}")
    if counts["train"] > TRAIN_MAX:
        problems.append(f"train {counts['train']} exceeds cap {TRAIN_MAX}")
    return problems


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def record(example: Example) -> dict:
    return {
        "id": example.id,
        "seed_id": example.seed_id,
        "category": example.category,
        "split": example.split,
        "messages": example.messages,
        "train_on": example.train_on,
        "expect_block": example.expect_block,
        "expect_action": example.expect_action,
        "note": example.note,
        "variant": example.variant,
        "seed_priority": example.seed_priority,
        "system_ref": example.system_ref,
        "system_fingerprint": example.system_fingerprint,
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
    }


def write_datasets(examples: list[Example], outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    written: dict[str, int] = {}
    for split in ("train", "validation", "holdout"):
        rows = [e for e in examples if e.split == split]
        path = os.path.join(outdir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for example in rows:
                handle.write(json.dumps(record(example), ensure_ascii=False) + "\n")
        written[split] = len(rows)

    # The holdout is protected: it is written once, marked, and must not be
    # read back into generation decisions.
    with open(os.path.join(outdir, "HOLDOUT_PROTECTED.json"), "w",
              encoding="utf-8") as handle:
        json.dump({
            "split": "holdout",
            "count": written["holdout"],
            "status": "protected - evaluation only",
            "rule": ("Not used during generation, tuning or wording decisions. "
                     "Read only when the holdout is actually being scored."),
            "generator_version": GENERATOR_VERSION,
            "schema_version": SCHEMA_VERSION,
        }, handle, indent=2)
    return written


def dedupe(examples: list[Example], threshold: float = 0.90) -> tuple[list[Example], list[dict]]:
    """Drop examples whose prompt duplicates or near-duplicates a kept one.

    The old corpus shipped 3,400 examples from 1,506 distinct prompts. Framing
    variety alone does not fix that - fourteen wrappers around one ask still
    produce fourteen near-identical prompts. This is the filter that keeps the
    higher-quality dataset, as the design's quality-over-quantity rule requires.

    Keep order is deterministic: higher seed priority first, then lowest
    variant, then id. So a duplicated example from a critical seed is kept and
    the same prompt from a normal seed is dropped.
    """
    import difflib

    def norm(text: str) -> str:
        return " ".join(
            "".join(c.lower() if c.isalnum() or c == " " else " " for c in text)
            .split())

    def first_user(ex: Example) -> str:
        for m in ex.messages:
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    rank = {"critical": 0, "high": 1, "normal": 2}
    ordered = sorted(examples,
                     key=lambda e: (rank.get(e.seed_priority, 3), e.variant, e.id))

    kept: list[Example] = []
    kept_prompts: list[str] = []
    dropped: list[dict] = []
    for ex in ordered:
        prompt = norm(first_user(ex))
        reason = None
        if not prompt:
            reason = "empty prompt"
        elif prompt in kept_prompts:
            reason = "exact duplicate prompt"
        else:
            for other in kept_prompts:
                if abs(len(other) - len(prompt)) > 60:
                    continue
                if difflib.SequenceMatcher(None, other, prompt).ratio() >= threshold:
                    reason = f"near-duplicate prompt (>= {threshold})"
                    break
        if reason:
            dropped.append({"id": ex.id, "seed_id": ex.seed_id, "reason": reason})
        else:
            kept.append(ex)
            kept_prompts.append(prompt)

    kept.sort(key=lambda e: e.id)
    return kept, dropped


WARNINGS: list[str] = []


def trim_split(examples: list[Example], split: str, target: int) -> list[Example]:
    """Trim a split to its exact target, never emptying a seed.

    Validation and holdout use whole seeds at a fixed depth, so a seed that can
    only yield two distinct prompts shortens its split. Reserving more seeds
    than the target needs and trimming afterwards gives exact sizes where enough
    distinct situations exist, and a reported shortfall where they do not.
    """
    rows = [e for e in examples if e.split == split]
    if len(rows) <= target:
        WARNINGS.append(
            f"{split} holds {len(rows)}, short of the {target} target after "
            f"quality filtering")
        return rows

    by_seed: dict[str, list[Example]] = collections.defaultdict(list)
    for ex in rows:
        by_seed[ex.seed_id].append(ex)

    # Drop from whichever seed currently holds the most, so coverage survives.
    while len(rows) > target:
        donor = max(by_seed, key=lambda s: (len(by_seed[s]), s))
        if len(by_seed[donor]) <= 1:
            WARNINGS.append(
                f"{split} cannot reach {target}: every remaining seed holds "
                f"exactly one example")
            break
        victim = max(by_seed[donor], key=lambda e: (e.variant, e.id))
        by_seed[donor].remove(victim)
        rows.remove(victim)
    return rows


def main() -> int:
    from PRIMORA.prime.context.isolation import production_digest

    before = production_digest()

    system_key, composed, capabilities = build_system_context()
    fingerprint = hashlib.sha256(composed.encode("utf-8")).hexdigest()

    per_seed = plan_expansion()
    raw = generate_all(system_key, fingerprint, per_seed)
    rejected = getattr(generate_all, "rejected", [])

    print("=" * 72)
    print("AURORA BRAIN v0.1 - DATASET GENERATION (seed bank v2)")
    print("=" * 72)
    print(f"  seeds approved       : {SEED_BUDGET}")
    print(f"  system reference key : {system_key}")
    print(f"  capabilities         : {', '.join(capabilities)}")
    print(f"  examples built       : {len(raw)}")
    print(f"  builder rejections   : {len(rejected)}")
    for rid, why in rejected[:10]:
        print(f"      rejected {rid}: {why}")

    # The holdout is filtered with the others but never compared against train,
    # so dedup here cannot leak holdout prompts into generation decisions.
    examples, dropped = dedupe(raw)
    final: list[Example] = [e for e in examples if e.split == "train"]
    for split_name, target in (("validation", VALIDATION_SIZE),
                               ("holdout", HOLDOUT_SIZE)):
        final.extend(trim_split(examples, split_name, target))
    examples = final
    print(f"  dropped as duplicate : {len(dropped)}")
    reasons = collections.Counter(d["reason"].split("(")[0].strip() for d in dropped)
    for reason, n in reasons.most_common():
        print(f"      {n:>4}  {reason}")

    problems = validate_set(examples)

    counts = {s: sum(1 for e in examples if e.split == s)
              for s in ("train", "validation", "holdout")}
    print(f"  train / val / holdout: {counts['train']} / "
          f"{counts['validation']} / {counts['holdout']}")
    print(f"  unique seeds covered : {len({e.seed_id for e in examples})}")

    if problems:
        print(f"  REFUSING TO WRITE: {len(problems)} problems")
        for p in problems[:15]:
            print(f"      {p}")
    else:
        outdir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTDIR
        written = write_datasets(examples, outdir)
        with open(os.path.join(outdir, "dropped.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(dropped, handle, indent=2)
        print(f"  written              : {written}")
        print(f"  output directory     : {outdir}")

    print(f"  production memory unchanged: {production_digest() == before}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
