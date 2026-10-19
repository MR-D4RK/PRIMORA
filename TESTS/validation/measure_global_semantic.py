"""Measure the context impact of enabling semantic retrieval of globals.

Read-only on production: every build runs on a private temp copy. Baseline is
today's live configuration (scoped sync, ``include_global=False``). Enabled
syncs every scope and passes ``include_global=True``. For each probe query the
script diffs the assembled prompt and reports the semantic candidates.

    python TESTS\\validation\\measure_global_semantic.py

Not collected by pytest (no ``test_`` prefix).
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PRIMORA.prime.context import ContextManager  # noqa: E402
from PRIMORA.prime.context import hybrid as hybrid_mod  # noqa: E402
from PRIMORA.prime.context.isolation import production_db_path  # noqa: E402

QUERIES = (
    "what do you know about me",
    "what are my preferences",
    "what is my favorite color",
    "what car do i love the most",
    "what do i love the most in terms of visual and design",
    "which electronic music genre do I like?",
    "what was my favorite soccer team before",
    "what is my favorite soccer team",
    "what music genre do i like",
    "what do i support",
)


def _copy(path: Path, suffix: str) -> Path:
    target = Path(tempfile.gettempdir()) / "opencode" / "memreg" / (
        f"global_semantic_{suffix}_{time.strftime('%Y%m%d-%H%M%S')}.db")
    target.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    dst = sqlite3.connect(str(target))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return target


def _trace_candidates(mgr) -> list[dict]:
    if not mgr._retrieval_traces:
        return []
    trace = mgr._retrieval_traces[-1]
    return [
        {
            "kind": c.kind, "ref": c.ref_id, "scope": c.scope,
            "similarity": round(c.similarity or 0.0, 4),
            "included": c.included, "reason": c.exclusion_reason,
        }
        for c in trace.candidates if c.channel == "semantic"
    ]


def _build(db: Path, query: str, include_global: bool,
           sync_all: bool) -> dict:
    mgr = ContextManager(db_path=str(db), hybrid_retrieval=True)
    index = mgr.semantic_index()
    index.sync(None if sync_all else "default")
    original = hybrid_mod.hybrid_retrieve

    def patched(*args, **kwargs):
        kwargs["include_global"] = include_global
        return original(*args, **kwargs)

    hybrid_mod.hybrid_retrieve = patched
    try:
        res = mgr.build(
            [{"role": "user", "content": query}], "default")
    finally:
        hybrid_mod.hybrid_retrieve = original
    return {
        "messages": [(m["role"], m["content"]) for m in res.view.messages],
        "semantic": _trace_candidates(mgr),
    }


def _diff(left, right):
    remaining = list(left)
    added = []
    for item in right:
        if item in remaining:
            remaining.remove(item)
        else:
            added.append(item)
    removed = list(remaining)
    return added, removed


def _snippet(item) -> str:
    role, content = item
    text = content.replace("\n", " / ")
    return f"{role}: {text[:160]}"


def main() -> int:
    prod = production_db_path()
    baseline_db = _copy(prod, "baseline")
    enabled_db = _copy(prod, "enabled")
    print(f"production: {prod}")
    print(f"baseline db: {baseline_db}")
    print(f"enabled db : {enabled_db}")

    report = []
    changed = 0
    for query in QUERIES:
        base = _build(baseline_db, query, False, False)
        enabled = _build(enabled_db, query, True, True)
        added, removed = _diff(base["messages"], enabled["messages"])
        if added or removed:
            changed += 1
        report.append({
            "query": query,
            "added": added,
            "removed": removed,
            "baseline_semantic": base["semantic"],
            "enabled_semantic": enabled["semantic"],
        })
        print("\n" + "=" * 78)
        print(query)
        print(f"  baseline semantic candidates: {len(base['semantic'])}")
        for c in base["semantic"]:
            print(f"    {c}")
        print(f"  enabled semantic candidates : {len(enabled['semantic'])}")
        for c in enabled["semantic"]:
            print(f"    {c}")
        print(f"  prompt added: {len(added)}  removed: {len(removed)}")
        for item in added:
            print(f"    + {_snippet(item)}")
        for item in removed:
            print(f"    - {_snippet(item)}")

    target = baseline_db.with_name(
        f"global_semantic_report_{time.strftime('%Y%m%d-%H%M%S')}.json")
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print(f"queries changed by enabling global semantic: {changed}"
          f" / {len(QUERIES)}")
    print(f"report: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
