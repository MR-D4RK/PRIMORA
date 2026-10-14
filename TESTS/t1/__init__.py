"""T1 interrogation harness: scenario corpus, isolated runtime, rubric
evaluator, report renderer, and CLI.

CURRENT: infrastructure plus a baseline capability. Contains no training, no
weight changes, and no production database access.

NOT IMPLEMENTED: training-data generation, model training, and automatic
conversion of failures into examples.
"""

__all__ = [
    "scenarios",
    "runtime",
    "evaluate",
    "report",
    "interrogator",
]
