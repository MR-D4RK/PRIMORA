# Local Memory V2 console

Start the disposable interactive harness with the same OpenAI-compatible
llama.cpp endpoint used by PRIMORA:

```bash
python -m PRIMORA.prime.context.memory_v2_console \
  --url http://localhost:8080/v1 \
  --model Qwen3.5-9B-Q4_K_M
```

The console creates a temporary SQLite database, enables V2 formation and
typed retrieval, and sends one bounded formation request through
`OpenAICompatibleProvider` for each new user turn. It never constructs the
default production `ContextManager`. An explicit `--db` is accepted only when
the path passes the production-directory isolation guard.

Use ordinary input for a user turn. The commands below inspect or extend the
same disposable store:

```text
:session alpha
:assistant I will remember that.
:evidence What are the last three things I told you I liked?
:memory
:diagnostics
:history alpha
:sessions
:quit
```

`:evidence` prints the source-backed evidence packet, including message/source
IDs. `:memory` prints current states and observations with proof counts.
`:diagnostics` prints formation status, provider failures, malformed-proposal
fallbacks, and other Prime diagnostics. Switching sessions does not copy raw
messages; global, source-backed memories can still be inspected explicitly.
Formation diagnostics include `provider_ms`, `db_ms`, `admission_ms`, and `total_ms`
so a local run can distinguish llama.cpp time from Prime/SQLite work.

If llama.cpp is unavailable, the console still starts and reports the provider
failure in diagnostics. No provider response is treated as an accepted fact.

