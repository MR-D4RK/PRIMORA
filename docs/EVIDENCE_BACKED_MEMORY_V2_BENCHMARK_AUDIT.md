# Evidence-backed Memory V2 benchmark audit

The original benchmark was not a valid comparison. It returned constants for
the baseline, candidate retrieval, correction, source preservation, and
unsupported-claim metrics. Its provider call counter was a class attribute,
alias and two-clock coverage were strings saying “seeded metadata only,” and
the correction score was overwritten with `1.0` after the correction check.
It also ran only one candidate path, so it could not measure regression or
abstention behavior.

The corrected runner is [benchmark_v2.py](../PRIMORA/prime/context/benchmark_v2.py).
It creates two disposable databases and feeds both paths the same seeded
messages, roles, timestamps, and scenario labels. Ground truth is the set of
source message IDs defined by the fixture: explicit assertions and explicit
negated preferences are memory, hedges and fictional statements are not, and
positive preference queries target only positive assertions. Formation scores
compare the source IDs of persisted current preference states with those
ground-truth IDs. Retrieval scores parse returned evidence packet message IDs
and compare them with the three latest expected source IDs.

The runner now measures, in both baseline and candidate paths:

- formation precision, recall, and F1;
- P@3, Recall@3, exact order/count, and evidence-ID coverage;
- unsupported returned source rate and abstention on an unknown domain;
- cross-session continuity for a known domain;
- model-call count and p50/p95 formation latency;
- source preservation, current correction plus retained history, alias
  resolution, and separate observed/occurred event clocks.

The deterministic provider is a test double. It is useful for admission and
ranking regression, but it is not a Qwen measurement. Qwen, VRAM, and RAM
remain explicitly `NOT RUN` when the local runtime is unavailable.

The latest disposable run (`seed=7`) produced these measured results:

| metric | baseline | candidate |
|---|---:|---:|
| formation precision / recall / F1 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| P@3 / Recall@3 | 0.333 / 0.333 | 1.000 / 1.000 |
| exact order and count | 0.000 | 1.000 |
| evidence-ID coverage | 0.000 | 1.000 |
| unsupported-claim rate | 0.667 | 0.000 |
| unknown-domain abstention | 0.000 | 1.000 |
| cross-session continuity | 0.000 | 1.000 |
| provider calls | 0 | 200 |
| formation latency p50 / p95 (ms) | 4.4 / 6.6 | 6.1 / 7.0 |

Correction preservation, alias resolution, two-clock event representation,
and raw-source preservation each evaluate to 1.0 from the stored rows. The
fixture contains 200 conversations over 60 domains, with measured distractors,
hedges, negations, fictional statements, assistant suggestions, one correction,
one alias chain, and one two-clock event.

## Existing Prime-suite failures

The 17 failures reproduced by `pytest -q TESTS/prime` are outside the V2
benchmark and were present in the branch before this audit. They classify as:

| classification | tests | cause |
|---|---|---|
| protocol/loop baseline | `test_capability_protocol.py::TestMultipleBlocksRejected::test_loop_executes_nothing_when_two_blocks_emitted`; `test_cognitive_loop.py::TestTermination::test_max_cycles_of_one_never_executes`; `test_cognitive_loop.py::TestTermination::test_deadline_mid_loop_returns_the_last_segment`; `test_cognitive_loop.py::TestAuroraIntegration::test_respond_signature_is_unchanged`; `test_cognitive_loop_v03.py::TestLimitsStillHold::test_deadline_returns_the_latest_valid_segment`; `test_phase_a_repair.py::TestUnrepairableBlockIsVisible::test_an_unrepairable_block_stops_with_protocol_error` | Existing capability-response/termination contract mismatch, unrelated to memory V2. |
| unavailable dependency | `test_capability_protocol.py::TestPreserved::test_system_prompt_fits_the_context_budget`; `test_phase_a_repair.py::TestTheBudgetStillClearsTheFloor::test_headroom_is_at_least_four_hundred_tokens` | `transformers` is not installed in the execution environment. |
| existing memory-window behavior | `test_live_cli_memory_windows.py::test_fav_cars_bridge_yields_to_a_direct_favorite_car_state` | Legacy favorite-car bridge formatting/selection behavior; V2 typed retrieval is feature-flagged and does not change this path. |
| unavailable model fixture/guidance | `test_phase_b_guidance.py::TestTheFamilyIsReadFromTheModel::test_the_configured_model_is_declared_qwen`; `test_phase_b_guidance.py::TestTheGuidanceIsKeyedToTheFamily::test_a_qwen_base_gets_the_guidance`; `test_phase_b_guidance.py::TestTheGuidanceIsKeyedToTheFamily::test_the_guidance_carries_its_header`; `test_phase_b_guidance.py::TestTheGuidanceIsKeyedToTheFamily::test_the_guidance_says_the_thing_the_instruction_does_not`; `test_phase_b_guidance.py::TestTheGuidanceReachesThePrompt::test_a_qwen_run_is_guided`; `test_phase_b_guidance.py::TestTheGuidanceReachesThePrompt::test_the_guidance_is_the_last_block`; `test_phase_b_guidance.py::TestTheGuidanceReachesThePrompt::test_a_qwen_run_still_carries_the_identity_and_catalogues`; `test_phase_b_guidance.py::TestTheBudgetStillClearsTheFloor::test_the_real_runtime_prompt_fits_the_ceiling` | The configured `RESOURCES/aurora/v0.1/model` fixture is absent/unrecognized, so Qwen-family guidance is empty. |

The focused V2 plus related memory regression set passes 107 tests. Full Prime
suite status remains 1,875 passed, 117 skipped, and the 17 classified failures.
The repository-wide collection additionally requires unavailable `aurora`,
`PySide6`, and `torch` packages.

