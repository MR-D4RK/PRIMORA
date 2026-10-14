"""Record-by-record comparison of two holdout evaluation runs.

Reads the per-record JSONL that ``evaluate_holdout_v5`` writes and joins the
two runs on record id. Pure reporting: it never re-scores anything and never
touches a dataset.

Usage: python TESTS/training/compare_holdout.py <label_a> <label_b> [--show N]
"""

from __future__ import annotations

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "holdout_eval")


def load(label: str) -> dict[str, dict]:
    path = os.path.join(OUT_DIR, f"{label}_results.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"missing results: {path}")
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def head(label: str) -> dict:
    path = os.path.join(OUT_DIR, f"{label}_report.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    show = 6
    args: list[str] = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--show":
            show = int(argv[i + 1])
            i += 2
            continue
        args.append(argv[i])
        i += 1
    if len(args) != 2:
        print(__doc__)
        return 2
    a_label, b_label = args
    a, b = load(a_label), load(b_label)

    shared = sorted(set(a) & set(b))
    print("=" * 78)
    print(f"RECORD-BY-RECORD: {a_label}  vs  {b_label}")
    print("=" * 78)
    print(f"  records {a_label}: {len(a)}   {b_label}: {len(b)}   "
          f"shared ids: {len(shared)}")
    ra, rb = head(a_label), head(b_label)
    print(f"  overall {a_label}: {ra['overall']['pass_rate_percent']}% "
          f"({ra['overall']['pass']}/{ra['records']})")
    print(f"  overall {b_label}: {rb['overall']['pass_rate_percent']}% "
          f"({rb['overall']['pass']}/{rb['records']})")

    improved, regressed, both_pass, both_fail = [], [], [], []
    for rid in shared:
        pa, pb = a[rid]["passed"], b[rid]["passed"]
        if not pa and pb:
            improved.append(rid)
        elif pa and not pb:
            regressed.append(rid)
        elif pa and pb:
            both_pass.append(rid)
        else:
            both_fail.append(rid)

    print("\n--- paired outcomes ---")
    print(f"  both pass           : {len(both_pass)}")
    print(f"  both fail           : {len(both_fail)}")
    print(f"  IMPROVED (fail->pass): {len(improved)}")
    print(f"  REGRESSED (pass->fail): {len(regressed)}")
    net = len(improved) - len(regressed)
    print(f"  net                 : {net:+d} records "
          f"({100.0 * net / len(shared):+.1f} pts)")

    # per category
    cats = sorted({a[r]["category"] for r in shared})
    print("\n--- changes by category ---")
    print(f"  {'cat':<5} {'n':>4} {'bothP':>6} {'bothF':>6} "
          f"{'impr':>5} {'regr':>5} {'net':>5}")
    for c in cats:
        ids = [r for r in shared if a[r]["category"] == c]
        i = sum(1 for r in ids if r in improved)
        g = sum(1 for r in ids if r in regressed)
        bp = sum(1 for r in ids if r in both_pass)
        bf = sum(1 for r in ids if r in both_fail)
        print(f"  {c:<5} {len(ids):>4} {bp:>6} {bf:>6} {i:>5} {g:>5} "
              f"{i - g:>+5}")

    # by difficulty
    diffs = sorted({str(a[r]["difficulty"]) for r in shared})
    print("\n--- changes by difficulty ---")
    for d in diffs:
        ids = [r for r in shared if str(a[r]["difficulty"]) == d]
        i = sum(1 for r in ids if r in improved)
        g = sum(1 for r in ids if r in regressed)
        bp = sum(1 for r in ids if r in both_pass)
        print(f"  d{d}: n={len(ids):<4} both_pass={bp:<4} "
              f"improved={i:<4} regressed={g:<4} net={i - g:+d}")

    # failed-check transitions
    print("\n--- failed-check counts (A -> B) ---")
    ca = collections.Counter(f.split(":", 1)[0]
                             for r in shared for f in a[r]["failed_checks"])
    cb = collections.Counter(f.split(":", 1)[0]
                             for r in shared for f in b[r]["failed_checks"])
    for name in sorted(set(ca) | set(cb)):
        delta = cb[name] - ca[name]
        flag = "" if delta == 0 else ("  <-- WORSE" if delta > 0
                                      else "  <-- better")
        print(f"  {name:<14} {ca[name]:>4} -> {cb[name]:>4} "
              f"({delta:+d}){flag}")

    # actions
    aa = collections.Counter(x for r in shared for x in a[r].get("actions", []))
    ab = collections.Counter(x for r in shared for x in b[r].get("actions", []))
    print("\n--- capability actions executed (A -> B) ---")
    for name in sorted(set(aa) | set(ab)):
        print(f"  {name:<14} {aa[name]:>4} -> {ab[name]:>4}")
    if not aa and not ab:
        print("  (none in either run)")

    def show_block(title, ids, limit):
        if not ids:
            print(f"\n--- {title}: none ---")
            return
        print(f"\n--- {title}: {len(ids)} total, showing {min(limit, len(ids))} ---")
        for rid in ids[:limit]:
            ra_, rb_ = a[rid], b[rid]
            print(f"\n  [{rid}] {ra_['category']}/d{ra_['difficulty']} "
                  f"seed={ra_['seed_id']}")
            print(f"    A failed: {ra_['failed_checks'] or 'PASS'}")
            print(f"    B failed: {rb_['failed_checks'] or 'PASS'}")
            if ra_.get("actions") or rb_.get("actions"):
                print(f"    actions A={ra_.get('actions')} "
                      f"B={rb_.get('actions')}")
            print(f"    A reply: {(ra_['reply'] or '')[:230]}")
            print(f"    B reply: {(rb_['reply'] or '')[:230]}")

    show_block(f"IMPROVED in {b_label}", improved, show)
    show_block(f"REGRESSED in {b_label}", regressed, show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
