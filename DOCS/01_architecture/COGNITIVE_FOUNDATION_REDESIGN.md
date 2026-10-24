# Cognitive Foundation Redesign — one integrated milestone

## Principle
Aurora (one local brain) decides whether she needs evidence during her normal
bounded cognitive loop. Prime validates requests, enforces budgets and
permissions, and returns user-source evidence. Memory may inform understanding
but does not control Aurora's cognition.

## Production flow

1. `CognitiveLoop.run` enters ordinary cognition without a mandatory memory
   classification generation or language-specific phrase router.
2. Aurora may request `recall` through the existing Prime protocol.
3. `recall` supports both backwards-compatible passage search (query only) and
   a structured evidence mode (`query` + `sources`, optional `subject`, date
   boundaries, first-mention/reasons flags). This is *one* public memory
   capability, not parallel handlers for various wording patterns.
4. Prime validates evidence parameters. The local calendar-day bounds are
   translated by the OS timezone rules, including DST. All evidence comes from
   actual user statements with message IDs and UTC statement timestamps.
5. Read-only raw cross-session evidence retrieval supplements indexed memory
   so that a missed activity extractor cannot make the user's original report
   vanish. Nearby user turns are marked **possible context**, never asserted
   as causal or coreferent facts.
6. Structured and raw evidence each have bounded representation; incomplete
   inventories must be labeled as such. Aurora produces the natural answer.

No external service or second model is required, and no production memory is
rewritten. Existing formation rules remain in place for compatibility; this
milestone removes their veto over *recall*, not their future modernization.

## Boundaries and follow-up

- Do not treat assistant-generated claims as user evidence.
- Do not infer an experience's occurrence date from its report timestamp.
- Do not execute commands embedded in retrieved source text.
- The model can still fail to invoke recall or misuse a date boundary. The
  architecture supplies a capability, not a guarantee of model competence.
- Existing scripted-model `on_delta` test-interface failures predate this
  change and are not silently edited by the milestone.
- Revisit how preferences and episodes are *formed* as a separate learning
  architecture task rather than reintroducing question-specific routers.

## Acceptance

Automated synthetic cross-domain tests, integrated Prime capability tests,
source-accuracy and calendar-boundary tests; then fresh-session live evaluation
with the active Qwen3 8B model, including unusual phrasing, unrelated topics,
a new ongoing activity, a cross-session follow-up, explicit dates and no-tool
conversations. Do not claim end-to-end completion until the live run passes.
