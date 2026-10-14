"""Validate the Aurora Brain v0.4 behavioral dataset.

The audit is pure: it reads the three splits and returns
``{"problems": [...], "report": {...}}``. ``build_dataset_v4`` does not call
it - generation and validation stay independent so a bug in one cannot hide
a bug in the other - but both run over the same files.

Every record is checked (100%, not a sample) in two labelled layers:

CONTRACT checks - a dataset that fails one of these is unusable:

*   malformed JSONL, or anything ``aurora.dataset_v2.validate_record``
    rejects against the live system context;
*   exact split counts (1080/180/180) and exactly six records per seed;
*   seed provenance: every seed known, whole-seed atomic, correct category,
    correct priority, correct generator/schema stamps;
*   per-split coverage: all four difficulties, all seven categories, both
    idk_ok values in every split;
*   the authored behavioural contract of ``responses_v4`` re-checked against
    every stored target: must / must_not, unsourced numerals, unsupported
    certainty, unnecessary IDK, poetic markers (unless style_ok), memory
    claims without conversation grounding, identity errors, Amir scan,
    per-seed brevity;
*   split leakage: distinctive-value pools and final-ask pools disjoint
    across splits;
*   target repetition: no exact final target supervised more than twice;
*   masking through the real ``aurora.masking`` module (when tokenising).

PATHOLOGY checks - heuristics for template-y or degenerate text; the
thresholds below fail, the milder ones only report:

*   one exact final target shared by two seeds -> fail;
*   word-set Jaccard >= 0.9 across seeds -> fail, 0.6-0.9 -> report;
*   one opening (first five words) used by >= 4 seeds -> fail;
*   any content word repeated more than twice in a target -> fail;
*   zero content-word overlap with the ask -> report only (topic drift).

Run: python TESTS/training/validate_dataset_v4.py [outdir]
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
from typing import Optional

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audit_v2 as A2  # noqa: E402
import build_dataset_v4 as BD4  # noqa: E402
import build_dataset_v3  # noqa: E402,F401  (keeps generate_v2 import path warm)
import responses_v4 as RV4  # noqa: E402
import seed_bank_v4 as SB4  # noqa: E402
from aurora import dataset_v2  # noqa: E402

DATA_DIR = BD4.DEFAULT_OUTDIR
SPLIT_ORDER = BD4.SPLIT_ORDER
EXPECTED_RECORDS = {"train": 1080, "validation": 180, "holdout": 180}
TARGET_REPEAT_LIMIT = BD4.TARGET_REPEAT_LIMIT

_TOKEN = re.compile(r"[a-z0-9']+")
_BLOCK = re.compile(r"<prime_request>")

# Function words excluded from the repetition and drift counters: a reply may
# say "the" three times without being a degenerate template.
STOPWORDS = frozenset(
    "a an the and or but if then else of to in on at for from with without "
    "is are was were be been being it its this that these those i you he she "
    "we they me him her us them my your his our their as by so do does did "
    "not no nor yes can could will would shall should may might must have "
    "has had having there here what which who whom when where why how all "
    "any both each few more most other some such only own same than too very "
    "just don t s".split())


def load(split: str, data_dir: str = DATA_DIR) -> list[dict]:
    path = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def content_words(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOPWORDS and len(t) > 2}


def target_of(row: dict) -> str:
    """The one assistant turn this record supervises."""
    assistants = [m for m in row["messages"] if m["role"] == "assistant"]
    return assistants[row["train_on"][-1]]["content"]


def all_targets(rows: list[dict]) -> list[tuple[str, str, str]]:
    """(seed_id, target_text) for every distinct authored reply, deduped so
    the bare/framed twins are compared once."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for row in rows:
        key = (row["seed_id"], A2.normalise(target_of(row)))
        if key in seen:
            continue
        seen.add(key)
        out.append((row["seed_id"], row["id"], target_of(row)))
    return out


# --------------------------------------------------------------------------
# Contract: record-level
# --------------------------------------------------------------------------


def check_schema(records: dict[str, list[dict]],
                 system: dataset_v2.SystemContext,
                 problems: list[str]) -> None:
    seen: set[str] = set()
    for split in SPLIT_ORDER:
        for row in records.get(split, []):
            rid = row.get("id", "<no id>")
            if rid in seen:
                problems.append(f"{rid}: duplicate record id")
            seen.add(rid)
            for problem in dataset_v2.validate_record(
                    row, expected_split=split, system=system):
                problems.append(f"{rid}: {problem}")
            if row.get("expect_block") or row.get("expect_action") is not None:
                problems.append(
                    f"{rid}: expect_block/expect_action must be empty in v0.4")
            for message in row.get("messages", []):
                if message.get("role") == "tool":
                    problems.append(f"{rid}: tool turn in a behavioral record")
                if message.get("role") == "assistant" and \
                        _BLOCK.search(message.get("content", "")):
                    problems.append(
                        f"{rid}: request block in a behavioral record")


def check_counts(records: dict[str, list[dict]],
                 problems: list[str]) -> dict:
    counts = {split: len(records.get(split, [])) for split in SPLIT_ORDER}
    for side, want in EXPECTED_RECORDS.items():
        if counts[side] != want:
            problems.append(
                f"{side} holds {counts[side]} records, expected {want}")

    per_seed: dict[str, list[dict]] = collections.defaultdict(list)
    for split in SPLIT_ORDER:
        for row in records.get(split, []):
            per_seed[row.get("seed_id")].append(row)
    if len(per_seed) != len(SB4.SEEDS):
        problems.append(
            f"{len(per_seed)} seeds emitted, bank holds {len(SB4.SEEDS)}")
    for sid in sorted(per_seed):
        rows = per_seed[sid]
        if len(rows) != BD4.RECORDS_PER_SEED:
            problems.append(
                f"{sid}: {len(rows)} records, expected "
                f"{BD4.RECORDS_PER_SEED}")
        if len({r.get("variant") for r in rows}) != 3:
            problems.append(f"{sid}: variants are not 0/1/2")
        framings = {r["id"].split("-")[-1] for r in rows}
        if framings != {f"{j:02d}" for j in range(6)}:
            problems.append(f"{sid}: record ids do not cover 00..05")
    return {"records": counts, "seeds": len(per_seed)}


def check_provenance(records: dict[str, list[dict]],
                     problems: list[str]) -> dict:
    by_id = {seed["id"]: seed for seed in SB4.SEEDS}
    priorities = {4: "critical", 3: "high", 2: "normal", 1: "normal"}
    seed_split: dict[str, set[str]] = collections.defaultdict(set)
    bad_priority = bad_category = bad_provenance = bad_version = 0
    for split in SPLIT_ORDER:
        for row in records.get(split, []):
            sid = row.get("seed_id")
            seed = by_id.get(sid)
            if seed is None:
                problems.append(f"{row.get('id')}: unknown seed {sid!r}")
                continue
            seed_split[sid].add(split)
            if row.get("category") != seed["category"]:
                bad_category += 1
                problems.append(
                    f"{row['id']}: category {row.get('category')!r} != bank "
                    f"{seed['category']!r}")
            if row.get("seed_priority") != priorities[seed["difficulty"]]:
                bad_priority += 1
                problems.append(f"{row['id']}: wrong seed_priority")
            if row.get("seed_provenance") != "seed_bank_v4":
                bad_provenance += 1
                problems.append(f"{row['id']}: wrong seed_provenance")
            if row.get("generator_version") != BD4.GENERATOR_VERSION:
                bad_version += 1
                problems.append(f"{row['id']}: wrong generator_version")
    for sid, sides in sorted(seed_split.items()):
        if len(sides) > 1:
            problems.append(f"{sid}: whole-seed atomicity broken: {sorted(sides)}")
    return {
        "seeds_emitted": len(seed_split),
        "bad_category": bad_category,
        "bad_priority": bad_priority,
        "bad_provenance": bad_provenance,
        "bad_version": bad_version,
    }


def check_coverage(records: dict[str, list[dict]],
                   problems: list[str]) -> dict:
    by_id = {seed["id"]: seed for seed in SB4.SEEDS}
    coverage: dict[str, dict] = {}
    for split in SPLIT_ORDER:
        rows = records.get(split, [])
        sids = {row.get("seed_id") for row in rows}
        difficulties = {by_id[s]["difficulty"] for s in sids if s in by_id}
        categories = {by_id[s]["category"] for s in sids if s in by_id}
        idk = {RV4.SPECS[s].idk_ok for s in sids if s in RV4.SPECS}
        if difficulties != {1, 2, 3, 4}:
            problems.append(f"{split}: difficulties {sorted(difficulties)}")
        if categories != set(BD4.CATEGORIES):
            problems.append(f"{split}: categories {sorted(categories)}")
        if idk != {True, False}:
            problems.append(f"{split}: idk_ok coverage {sorted(idk)}")
        coverage[split] = {
            "seeds": len(sids),
            "difficulties": sorted(difficulties),
            "categories": "".join(sorted(categories)),
            "idk_ok_true": sum(1 for s in sids
                               if s in RV4.SPECS and RV4.SPECS[s].idk_ok),
        }
    return coverage


# --------------------------------------------------------------------------
# Contract: the authored behavioural spec, re-checked on every record
# --------------------------------------------------------------------------

_COMPILED: dict[str, tuple[list[re.Pattern[str]], list[re.Pattern[str]]]] = {}


def _compiled(sid: str) -> tuple[list[re.Pattern[str]], list[re.Pattern[str]]]:
    if sid not in _COMPILED:
        spec = RV4.SPECS[sid]
        _COMPILED[sid] = (
            [re.compile(p, re.IGNORECASE) for p in spec.must],
            [re.compile(p, re.IGNORECASE) for p in spec.must_not])
    return _COMPILED[sid]


def _numeral_pool(spec: RV4.Spec) -> set[str]:
    pool: set[str] = set()
    for text in (spec.ask, spec.context or "", spec.prior_ask or "",
                 spec.prior_say or ""):
        pool |= RV4._numerals(text)
    for pattern in spec.must:
        pool |= RV4._numerals(RV4._pattern_literals(pattern))
    for extra in spec.numbers_ok:
        pool |= RV4._numerals(extra)
    return pool


def score_reply(sid: str, text: str) -> list[str]:
    """The behavioural contract of one reply, as failed-check names.

    Shared by static validation (against authored targets) and the real-model
    baseline (against Aurora's own output), so the two runs cannot drift
    apart. Empty list means the reply passes every gate.
    """
    spec = RV4.SPECS.get(sid)
    if spec is None:
        return ["no spec for seed"]
    if not text or not text.strip():
        return ["empty reply"]
    fails: list[str] = []
    must_p, must_not_p = _compiled(sid)

    for pattern in must_p:
        if not pattern.search(text):
            fails.append(f"must:{pattern.pattern}")
    for pattern in must_not_p:
        if pattern.search(text):
            fails.append(f"must_not:{pattern.pattern}")

    pool = _numeral_pool(spec)
    for numeral in RV4._numerals(text):
        if numeral not in pool:
            fails.append(f"unsourced:{numeral}")

    if spec.idk_ok and RV4._unnegated(RV4.CERTAINTY_PHRASES, text):
        fails.append("certainty")
    if not spec.idk_ok and RV4._unnegated(RV4.IDK_PHRASES, text):
        fails.append("idk")
    if not spec.style_ok and RV4.POETIC_MARKERS.search(text):
        fails.append("poetic")
    if not spec.context and not spec.prior_ask and \
            RV4.MEMORY_CLAIMS.search(text):
        fails.append("memory")
    if RV4.IDENTITY_ERROR.search(text):
        fails.append("identity")
    if "amir" in text.casefold():
        fails.append("amir")
    cap = spec.max_len or RV4.DEFAULT_MAX_WORDS
    if len(text.split()) > cap:
        fails.append("brevity")
    noisy = {w: n for w, n in collections.Counter(tokens(text)).items()
             if w not in STOPWORDS and n > 2}
    if noisy:
        fails.append("repetition")
    if _BLOCK.search(text):
        fails.append("request block")
    return fails


def check_specs(records: dict[str, list[dict]],
                problems: list[str]) -> dict:
    """Run the responses_v4 contract against 100% of stored targets."""
    tallies: collections.Counter = collections.Counter()
    per_seed_fails: dict[str, int] = {}
    for split in SPLIT_ORDER:
        for row in records.get(split, []):
            rid = row.get("id", "<no id>")
            sid = row.get("seed_id")
            if sid not in RV4.SPECS:
                problems.append(f"{rid}: no spec for seed {sid!r}")
                continue
            for fail in score_reply(sid, target_of(row)):
                problems.append(f"{rid}: {fail}")
                tallies[fail.split(":", 1)[0]] += 1
                per_seed_fails[sid] = per_seed_fails.get(sid, 0) + 1

            # The acknowledgement / prior turns are checked for identity
            # errors too: they are trained-adjacent context, and a wrong
            # name there would poison the record even if the target is clean.
            for message in row["messages"]:
                if message["role"] == "assistant" and \
                        message["content"] != target_of(row) and \
                        RV4.IDENTITY_ERROR.search(message["content"]):
                    problems.append(f"{rid}: identity error in a context turn")
                    tallies["identity_context_fail"] += 1
                if "amir" in message["content"].casefold():
                    problems.append(f"{rid}: mentions Amir")
                    tallies["amir_fail"] += 1
    return {"tallies": dict(tallies),
            "seeds_with_failures": len(per_seed_fails)}


# --------------------------------------------------------------------------
# Contract: split leakage
# --------------------------------------------------------------------------


def check_leakage(records: dict[str, list[dict]],
                  problems: list[str]) -> dict:
    pools: dict[str, set[str]] = {side: set() for side in SPLIT_ORDER}
    finals: dict[str, dict[str, str]] = {side: {} for side in SPLIT_ORDER}
    for side in SPLIT_ORDER:
        for row in records.get(side, []):
            for message in row["messages"]:
                pools[side] |= BD4.extract_values(message["content"])
            final_ask = BD4.normalise_ask(
                [m for m in row["messages"] if m["role"] == "user"][-1]
                ["content"])
            finals[side][final_ask] = row["id"]

    value_sharing = ask_sharing = 0
    for i, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[i + 1:]:
            shared = pools[left] & pools[right]
            if shared:
                value_sharing += len(shared)
                problems.append(
                    f"{left}/{right} share distinctive value(s) in stored "
                    f"text: {sorted(shared)[:5]}")
            shared_asks = set(finals[left]) & set(finals[right])
            if shared_asks:
                ask_sharing += len(shared_asks)
                problems.append(
                    f"{left}/{right} share {len(shared_asks)} final ask(s), "
                    f"e.g. {sorted(shared_asks)[0][:50]!r}")

    # Cross-seed final targets must be unique to one seed (the bare/framed
    # twins of a single seed are the same target by design and are collapsed).
    owners: dict[str, set[str]] = collections.defaultdict(set)
    for side in SPLIT_ORDER:
        for seed_id, _rid, text in all_targets(records.get(side, [])):
            owners[A2.normalise(text)].add(seed_id)
    cross = {t: sorted(s) for t, s in owners.items() if len(s) > 1}
    for text, seeds in list(cross.items())[:10]:
        problems.append(f"final target shared by seeds {seeds}: {text[:50]!r}")
    if len(cross) > 10:
        problems.append(f"... and {len(cross) - 10} more shared targets")
    return {"shared_values": value_sharing, "shared_final_asks": ask_sharing,
            "cross_seed_shared_targets": len(cross)}


# --------------------------------------------------------------------------
# Contract: target repetition
# --------------------------------------------------------------------------


def check_target_repetition(records: dict[str, list[dict]],
                            problems: list[str]) -> dict:
    """No exact final target is supervised more than TARGET_REPEAT_LIMIT times.

    Every seed ships two records per authored reply (bare and framed), so the
    honest ceiling is 2. The gate counts the exact supervised text; the
    normalised figures are reported alongside it because two authored replies
    can differ only by punctuation and collapse onto one target once
    normalised, which the exact gate cannot see.
    """
    exact = collections.Counter(
        target_of(row) for side in SPLIT_ORDER
        for row in records.get(side, []))
    over = {t: n for t, n in exact.items() if n > TARGET_REPEAT_LIMIT}
    for text, times in sorted(over.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
        problems.append(
            f"target repeated {times} times (limit {TARGET_REPEAT_LIMIT}): "
            f"{text[:60]!r}")
    if len(over) > 10:
        problems.append(f"... and {len(over) - 10} more repeated targets")

    normalised = collections.Counter(
        A2.normalise(target_of(row)) for side in SPLIT_ORDER
        for row in records.get(side, []))
    norm_over = {t: n for t, n in normalised.items()
                 if n > TARGET_REPEAT_LIMIT}
    return {
        "limit": TARGET_REPEAT_LIMIT,
        "distinct_exact": len(exact),
        "max_exact_repeat": max(exact.values(), default=0),
        "over_limit_exact": len(over),
        "distinct_normalised": len(normalised),
        "max_normalised_repeat": max(normalised.values(), default=0),
        "over_limit_normalised": [
            [text[:60], norm_over[text]]
            for text in sorted(norm_over, key=lambda t: (-norm_over[t], t))
            [:10]],
        "note": ("the gate fails on exact text; normalised collapses are "
                 "reported so punctuation-only duplicates stay visible"),
    }


# --------------------------------------------------------------------------
# Pathology: diversity of the authored wording
# --------------------------------------------------------------------------


def check_diversity(records: dict[str, list[dict]],
                    problems: list[str]) -> dict:
    targets = [t for side in SPLIT_ORDER
               for t in all_targets(records.get(side, []))]

    openings: dict[str, set[str]] = collections.defaultdict(set)
    for seed_id, rid, text in targets:
        key = " ".join(tokens(text)[:5])
        openings[key].add(seed_id)
    template = {k: sorted(v) for k, v in openings.items() if len(v) >= 4}
    for key, seeds in list(sorted(template.items()))[:10]:
        problems.append(
            f"template opening {key!r} used by {len(seeds)} seeds: "
            f"{seeds[:6]}")
    if len(template) > 10:
        problems.append(f"... and {len(template) - 10} more openings")

    near: list[tuple[float, str, str]] = []
    reported: list[tuple[float, str, str]] = []
    for i in range(len(targets)):
        a = set(tokens(targets[i][2]))
        for j in range(i + 1, len(targets)):
            if targets[i][0] == targets[j][0]:
                continue  # same seed: the three replies are meant to differ
            b = set(tokens(targets[j][2]))
            if not a or not b:
                continue
            score = len(a & b) / len(a | b)
            if score >= 0.9:
                near.append((score, targets[i][1], targets[j][1]))
            elif score >= 0.6:
                reported.append((score, targets[i][1], targets[j][1]))
    for score, left, right in near[:10]:
        problems.append(
            f"Jaccard {score:.2f} between targets {left} and {right}")
    if len(near) > 10:
        problems.append(f"... and {len(near) - 10} more near-duplicates")

    drift = 0
    for side in SPLIT_ORDER:
        for row in records.get(side, []):
            ask = [m for m in row["messages"] if m["role"] == "user"][-1]
            ask_words = content_words(BD4.normalise_ask(ask["content"]))
            target_words = content_words(target_of(row))
            if ask_words and not (ask_words & target_words):
                drift += 1

    return {
        "distinct_targets": len(targets),
        "distinct_openings": len(openings),
        "template_openings": len(template),
        "near_duplicates_ge_0.9": len(near),
        "similar_0.6_to_0.9": len(reported),
        "sample_similar": [[round(s, 2), a, b]
                           for s, a, b in sorted(reported, reverse=True)[:10]],
        "drift_report_only": drift,
        "note": "drift is report-only; the others fail at their threshold",
    }


# --------------------------------------------------------------------------
# Contract: tool representation and masking
# --------------------------------------------------------------------------


def check_tools(records: dict[str, list[dict]], problems: list[str]) -> dict:
    representation = A2.check_tool_representation(records)
    problems.extend(f"tool: {f}" for f in representation["findings"])
    sanity = A2.check_payload_sanity(records)
    problems.extend(f"payload: {f}" for f in sanity["findings"])
    return {"tool_turns": representation["tool_turns"],
            "payload_findings": len(sanity["findings"])}


def check_masking(records: dict[str, list[dict]],
                  system: dataset_v2.SystemContext,
                  problems: list[str]) -> dict:
    try:
        from transformers import AutoTokenizer
        from aurora.config import TrainerConfig
        from aurora.masking import MaskingError, build_mask, verify_mask
        config = TrainerConfig()
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model, trust_remote_code=True)
    except Exception as exc:
        problems.append(f"masking unavailable: {exc}")
        return {"available": False}

    trained_turns = 0
    trained_tokens = 0
    truncated = 0
    for split in SPLIT_ORDER:
        for row in records.get(split, []):
            messages = [{"role": "system", "content": system.content}]
            messages.extend(row["messages"])
            try:
                result = build_mask(tokenizer, messages, row["train_on"],
                                    max_length=config.max_seq_length)
                verify_mask(result, messages, tokenizer,
                            row.get("id", "<rec>"))
            except MaskingError as exc:
                problems.append(f"{row.get('id')}: mask check failed: {exc}")
                continue
            if result.truncated:
                truncated += 1
                problems.append(
                    f"{row.get('id')}: renders past max_seq_length "
                    f"{config.max_seq_length}")
            trained_turns += len(result.trained_turns)
            trained_tokens += result.trained_tokens
    return {"available": True, "records": sum(
                len(records.get(s, [])) for s in SPLIT_ORDER),
            "trained_turns": trained_turns,
            "trained_tokens": trained_tokens,
            "truncated_records": truncated}


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def audit(records: dict[str, list[dict]],
          system: Optional[dataset_v2.SystemContext] = None,
          tokenize: bool = True) -> dict:
    problems: list[str] = []
    report: dict = {"label": "Aurora v0.4 static dataset validation",
                    "contract": {}, "pathology": {}}

    if system is None:
        refs = {row.get("system_ref") for split in SPLIT_ORDER
                for row in records.get(split, [])}
        refs.discard(None)
        if len(refs) != 1:
            problems.append(f"expected one system_ref, found {sorted(refs)}")
            return {"problems": problems, "report": report}
        try:
            system = dataset_v2.resolve_system_context(next(iter(refs)))
        except dataset_v2.DatasetError as exc:
            problems.append(f"system context does not resolve: {exc}")
            return {"problems": problems, "report": report}

    report["contract"]["counts"] = check_counts(records, problems)
    check_schema(records, system, problems)
    report["contract"]["provenance"] = check_provenance(records, problems)
    report["contract"]["coverage"] = check_coverage(records, problems)
    report["contract"]["spec_checks"] = check_specs(records, problems)
    report["contract"]["leakage"] = check_leakage(records, problems)
    report["contract"]["repetition"] = check_target_repetition(
        records, problems)
    report["contract"]["tools"] = check_tools(records, problems)
    report["pathology"]["diversity"] = check_diversity(records, problems)
    report["contract"]["masking"] = (
        check_masking(records, system, problems) if tokenize
        else {"skipped": "tokenize=False"})

    report["problem_count"] = len(problems)
    return {"problems": problems, "report": report}


def main() -> int:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    records = {split: load(split, data_dir) for split in SPLIT_ORDER}
    if not any(records.values()):
        print(f"no v0.4 dataset found under {data_dir}")
        return 1
    result = audit(records)
    problems = result["problems"]
    report = result["report"]
    report["problems"] = problems[:200]

    out = os.path.join(data_dir, "validation_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    if problems:
        print(f"\nFAILURES ({len(problems)}):")
        for problem in problems[:40]:
            print(f"  FAIL {problem}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        print(f"\nreport written: {out}")
        return 1
    print(f"\nOK: every v0.4 dataset gate held. report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
