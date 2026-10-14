"""Structured T1 report rendering.

CURRENT: text and JSON output. No single opaque score is produced; the
report is category-level and per-scenario so a regression can be located.

PROPOSED: trend comparison across training versions, once more than one
baseline exists. The JSON already carries everything needed for that.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from TESTS.t1.evaluate import AMBIGUOUS, FAIL, PASS, Evaluation, language_observations
from TESTS.t1.scenarios import ALL_FAILURE_TYPES, Scenario

SCHEMA_VERSION = 2


@dataclass
class ScenarioResult:
    scenario_id: str
    category: str
    title: str
    intent: str
    prompt: str
    memory_label: str
    response: str
    expected_behavior: str
    observed_behavior: str
    verdict: str
    failure_type: Optional[str]
    dimensions_failed: list[str] = field(default_factory=list)
    dimensions_unclear: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    language: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    sample_index: int = 0
    error: Optional[str] = None
    human_verdict: Optional[str] = None
    human_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["human_verdict"] = self.human_verdict
        return data

    @property
    def effective_verdict(self) -> str:
        return self.human_verdict or self.verdict


@dataclass
class InterrogationReport:
    model: str
    adapter: str
    generated_at: str
    runtime: dict[str, Any]
    results: list[ScenarioResult]
    notes: list[str] = field(default_factory=list)

    # -- aggregates -------------------------------------------------------

    def counts(self) -> dict[str, int]:
        out = {PASS: 0, FAIL: 0, AMBIGUOUS: 0}
        for r in self.results:
            out[r.effective_verdict] += 1
        return out

    def by_category(self) -> dict[str, dict[str, Any]]:
        buckets: dict[str, list[ScenarioResult]] = {}
        for r in self.results:
            buckets.setdefault(r.category, []).append(r)
        out: dict[str, dict[str, Any]] = {}
        for category in sorted(buckets):
            rows = buckets[category]
            tally = {PASS: 0, FAIL: 0, AMBIGUOUS: 0}
            for r in rows:
                tally[r.effective_verdict] += 1
            out[category] = {
                "total": len(rows),
                "pass": tally[PASS],
                "fail": tally[FAIL],
                "ambiguous": tally[AMBIGUOUS],
                "scenario_ids": [r.scenario_id for r in rows],
            }
        return out

    def failure_breakdown(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            if r.effective_verdict == FAIL and r.failure_type:
                out[r.failure_type] = out.get(r.failure_type, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def dimension_breakdown(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            if r.effective_verdict == FAIL:
                for d in r.dimensions_failed:
                    out[d] = out.get(d, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def latency(self) -> dict[str, float]:
        values = [r.latency_s for r in self.results if r.latency_s > 0]
        if not values:
            return {}
        return {
            "count": len(values),
            "mean_s": round(statistics.mean(values), 2),
            "median_s": round(statistics.median(values), 2),
            "min_s": round(min(values), 2),
            "max_s": round(max(values), 2),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model": self.model,
            "adapter": self.adapter,
            "generated_at": self.generated_at,
            "runtime": self.runtime,
            "summary": {
                "total": len(self.results),
                **self.counts(),
                "pass_rate": round(
                    self.counts()[PASS] / len(self.results), 3
                ) if self.results else 0.0,
            },
            "by_category": self.by_category(),
            "failures_by_type": self.failure_breakdown(),
            "failures_by_dimension": self.dimension_breakdown(),
            "latency": self.latency(),
            "notes": self.notes,
            "results": [r.to_dict() for r in self.results],
        }

    # -- rendering --------------------------------------------------------

    def to_text(self, verbose: bool = False) -> str:
        lines: list[str] = []
        add = lines.append
        add("=" * 74)
        add("AURORA T1 INTERROGATION REPORT")
        add("=" * 74)
        add(f"  model        : {self.model}")
        add(f"  adapter      : {self.adapter}")
        add(f"  generated_at : {self.generated_at}")
        for key, value in self.runtime.items():
            add(f"  {key:<12} : {value}")

        counts = self.counts()
        add("")
        add("-" * 74)
        add("OVERALL")
        add("-" * 74)
        add(f"  scenarios : {len(self.results)}")
        add(f"  PASS      : {counts[PASS]}")
        add(f"  FAIL      : {counts[FAIL]}")
        add(f"  AMBIGUOUS : {counts[AMBIGUOUS]}")

        add("")
        add("-" * 74)
        add("BY CATEGORY")
        add("-" * 74)
        add(f"  {'category':<26} {'n':>3} {'PASS':>5} {'FAIL':>5} {'AMB':>5}  ids")
        for category, agg in self.by_category().items():
            add(f"  {category:<26} {agg['total']:>3} {agg['pass']:>5} "
                f"{agg['fail']:>5} {agg['ambiguous']:>5}  "
                f"{','.join(agg['scenario_ids'])}")

        add("")
        add("-" * 74)
        add("FAILURES BY TYPE")
        add("-" * 74)
        breakdown = self.failure_breakdown()
        if not breakdown:
            add("  (none)")
        for code, count in breakdown.items():
            add(f"  {code:<28} {count:>3}")

        add("")
        add("-" * 74)
        add("FAILURES BY DIMENSION")
        add("-" * 74)
        dims = self.dimension_breakdown()
        if not dims:
            add("  (none)")
        for name, count in dims.items():
            add(f"  {name:<32} {count:>3}")

        latency = self.latency()
        if latency:
            add("")
            add("-" * 74)
            add("LATENCY")
            add("-" * 74)
            add(f"  n={latency['count']}  mean={latency['mean_s']}s  "
                f"median={latency['median_s']}s  "
                f"min={latency['min_s']}s  max={latency['max_s']}s")

        add("")
        add("-" * 74)
        add("SCENARIOS")
        add("-" * 74)
        for r in self.results:
            marker = {PASS: "PASS", FAIL: "FAIL", AMBIGUOUS: " AMB"}[
                r.effective_verdict]
            add(f"  [{marker}] {r.scenario_id}  {r.category}")
            add(f"         {r.title}")
            add(f"    Q    : {_one_line(r.prompt)}")
            add(f"    mem  : {r.memory_label}")
            add(f"    A    : {_one_line(r.response)}")
            add(f"    want : {r.expected_behavior}")
            add(f"    got  : {r.observed_behavior}")
            if r.failure_type and r.effective_verdict == FAIL:
                add(f"    type : {r.failure_type}")
            if r.human_verdict:
                add(f"    human: {r.human_verdict} {r.human_note}")
            if verbose:
                for c in r.checks:
                    flag = "ok  " if c["passed"] else "MISS"
                    add(f"      {flag} {c['key']} ({c['severity']}) "
                        f"evidence={c['evidence']}")
                if r.language:
                    add(f"      lang: {r.language}")

        if self.notes:
            add("")
            add("-" * 74)
            add("NOTES")
            add("-" * 74)
            for note in self.notes:
                add(f"  - {note}")

        add("")
        add("=" * 74)
        add("The rubric evaluator is a heuristic, not ground truth. AMBIGUOUS")
        add("means a human should adjudicate; use --interactive to override.")
        add("=" * 74)
        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _one_line(text: str, limit: int = 150) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def build_result(
    scenario: Scenario,
    prompt: str,
    response: str,
    evaluation: Evaluation,
    latency_s: float,
    sample_index: int = 0,
    error: Optional[str] = None,
    memory_label: str = "no memory",
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario.id,
        category=scenario.category,
        title=scenario.title,
        intent=scenario.intent,
        prompt=prompt,
        memory_label=memory_label,
        response=response,
        expected_behavior=_expected_summary(scenario),
        observed_behavior=evaluation.observed,
        verdict=evaluation.verdict,
        failure_type=evaluation.failure_type,
        dimensions_failed=list(evaluation.dimensions_failed),
        dimensions_unclear=list(evaluation.dimensions_unclear),
        checks=[c.to_dict() for c in evaluation.checks],
        language=language_observations(response),
        latency_s=round(latency_s, 2),
        sample_index=sample_index,
        error=error,
    )


def _expected_summary(scenario: Scenario) -> str:
    parts = []
    for e in scenario.expectations:
        if e.severity == "required":
            parts.append(e.note or e.key)
    return "; ".join(parts) or scenario.intent


def expected_behavior_text(scenario: Scenario) -> str:
    """Human-facing expectation block for interactive mode."""
    lines = [scenario.intent or scenario.title, ""]
    for e in scenario.expectations:
        tag = "REQUIRED" if e.severity == "required" else "preferred"
        lines.append(f"  [{tag}] {e.key}: {e.note or e.kind}")
    return "\n".join(lines)
