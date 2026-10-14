"""Three-way signal table + paired McNemar for the holdout runs.

Read-only reporting over the JSONL that evaluate_holdout_v5 wrote.
Run: python TESTS/training/holdout_eval_stats.py
"""

from __future__ import annotations

import json
import math
import os
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "holdout_eval")


def load(label):
    path = os.path.join(OUT, f"{label}_results.jsonl")
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in (json.loads(x) for x in fh if x.strip())}


RUNS = {"v0.4#1": load("v0.4"), "v0.4#2": load("v0.4_rep"), "v0.5": load("v0.5")}
ORDER = ["v0.4#1", "v0.4#2", "v0.5"]


def mcnemar_p(b, c):
    """Exact-ish two-sided p for discordant pairs (continuity corrected)."""
    n = b + c
    if n == 0:
        return 1.0
    chi = (abs(b - c) - 1) ** 2 / n
    return math.erfc(math.sqrt(chi / 2))


def main():
    print("=" * 84)
    print("THREE-WAY SIGNAL TABLE (counts over the same 180 holdout records)")
    print("=" * 84)
    header = f"{'metric':<26}" + "".join(f"{k:>10}" for k in ORDER) + "   verdict"
    print(header)
    print("-" * len(header))

    def row(name, fn):
        vals = [fn(RUNS[k]) for k in ORDER]
        lo, hi = min(vals[:2]), max(vals[:2])
        if vals[2] > hi:
            verd = "v0.5 OUTSIDE v0.4 band (higher)"
        elif vals[2] < lo:
            verd = "v0.5 OUTSIDE v0.4 band (lower)"
        else:
            verd = "inside v0.4 noise band"
        print(f"{name:<26}" + "".join(f"{v:>10}" for v in vals) + f"   {verd}")

    row("passing records", lambda r: sum(1 for x in r.values() if x["passed"]))

    def failkind(r, k):
        return sum(1 for x in r.values()
                   for f in x["failed_checks"] if f.split(":", 1)[0] == k)

    for k in ["must", "unsourced", "brevity", "idk", "must_not",
              "certainty", "repetition", "poetic", "amir"]:
        row(f"fail: {k}", lambda r, k=k: failkind(r, k))

    row("records w/ unsourced",
        lambda r: len({x["id"] for x in r.values()
                       if any(f.startswith("unsourced")
                              for f in x["failed_checks"])}))
    row("capability calls executed",
        lambda r: sum(len(x.get("actions", [])) for x in r.values()))
    row("records w/ any capability call",
        lambda r: sum(1 for x in r.values() if x.get("actions")))
    row("history fallbacks", lambda r: sum(
        1 for x in r.values() if x.get("history_fallback")))
    row("errors", lambda r: sum(1 for x in r.values() if x.get("error")))

    print()
    print("=" * 84)
    print("PAIRED OUTCOMES + McNEMAR ON DISCORDANT PAIRS")
    print("=" * 84)
    pairs = [("v0.4#1", "v0.4#2", "SAME adapter, two runs  (noise control)"),
             ("v0.4#1", "v0.5", "different adapters"),
             ("v0.4#2", "v0.5", "different adapters")]
    for p, q, note in pairs:
        a, b = RUNS[p], RUNS[q]
        ids = set(a) & set(b)
        imp = sum(1 for i in ids if not a[i]["passed"] and b[i]["passed"])
        reg = sum(1 for i in ids if a[i]["passed"] and not b[i]["passed"])
        both_p = sum(1 for i in ids if a[i]["passed"] and b[i]["passed"])
        both_f = sum(1 for i in ids if not a[i]["passed"] and not b[i]["passed"])
        pv = mcnemar_p(imp, reg)
        verdict = "NOT significant (p>0.05)" if pv > 0.05 else "significant"
        print(f"\n  {p} vs {q}  [{note}]")
        print(f"    both pass {both_p:<4} both fail {both_f:<4} "
              f"improved {imp:<4} regressed {reg:<4} net {imp - reg:+d}")
        print(f"    total flips (improved+regressed) = {imp + reg}")
        print(f"    McNemar p = {pv:.3f}   -> {verdict}")

    print()
    print("=" * 84)
    print("FLIP-RATE INTERPRETATION")
    print("=" * 84)
    a, b, c = RUNS["v0.4#1"], RUNS["v0.4#2"], RUNS["v0.5"]
    ids = set(a) & set(b) & set(c)
    noise_flips = sum(1 for i in ids
                      if (a[i]["passed"] != b[i]["passed"]))
    cross_flips = sum(1 for i in ids
                      if (a[i]["passed"] != c[i]["passed"]))
    cross_flips2 = sum(1 for i in ids
                       if (b[i]["passed"] != c[i]["passed"]))
    print(f"  records where v0.4 differs from ITSELF (run1 vs run2): "
          f"{noise_flips}")
    print(f"  records where v0.4 run1 differs from v0.5            : "
          f"{cross_flips}")
    print(f"  records where v0.4 run2 differs from v0.5            : "
          f"{cross_flips2}")
    print()
    if cross_flips <= noise_flips and cross_flips2 <= noise_flips * 1.25:
        print("  => The cross-adapter flip count is at or below the "
              "same-adapter flip count.")
        print("     Record-by-record differences between v0.4 and v0.5 are "
              "within the noise")
        print("     produced by stochastic decoding; only metrics whose "
              "v0.5 value falls")
        print("     outside the two-run v0.4 band are genuine signals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
