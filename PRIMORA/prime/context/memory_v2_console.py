"""Interactive local Memory V2 test console.

Run with::

    python -m PRIMORA.prime.context.memory_v2_console \
        --model Qwen3.5-9B-Q4_K_M

The default database is a temporary SQLite file.  An explicit ``--db`` is
accepted only after the same production-path isolation check used by the test
harness.  The console never constructs a ContextManager without an explicit
path, so an omitted option cannot fall back to Aurora's production database.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional, TextIO

from PRIMORA.prime.providers import OpenAICompatibleProvider

from .isolation import IsolationError, require_isolated
from .manager import ContextManager


HELP = """Commands:
  :session NAME       switch to or create a conversation session
  :sessions            list stored sessions
  :assistant TEXT      append an assistant turn to the current session
  :evidence QUERY      retrieve source-backed Memory V2 evidence
  :memory              show current states, observations, and source ids
  :history [NAME]      show raw turns from a session
  :diagnostics         show formation status and the latest diagnostics
  :help                show this help
  :quit                 leave the console

Any other input is recorded as a user turn.  V2 formation runs only for new
user turns; assistant turns are retained as context and never become facts.
"""


class MemoryV2Console:
    """Command loop separated from provider construction for safe testing."""

    def __init__(
        self,
        db_path: Path | str,
        provider,
        *,
        session_id: str = "default",
        output: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.db_path = require_isolated(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.manager = ContextManager(
            db_path=self.db_path,
            enable_memory_v2=True,
            formation_provider=provider,
            # The V2 evidence path is deterministic and does not require an
            # embedding model. This keeps the manual harness local and
            # inspectable on a machine that only has llama.cpp/Qwen running.
            hybrid_retrieval=False,
        )
        self.session_id = self._validate_session(session_id)
        self.output = output or print
        self.last_diagnostics: list[str] = []

    @staticmethod
    def _validate_session(value: str) -> str:
        value = str(value or "").strip()
        if not value or len(value) > 120 or any(char in value for char in "\x00\r\n"):
            raise ValueError("session name must be 1-120 characters without newlines")
        return value

    def _write(self, text: str = "") -> None:
        self.output(text)

    def _snapshot_with(self, role: str, content: str) -> list[dict[str, str | int]]:
        transcript = self.manager.load_transcript(self.session_id, limit=500)
        next_seq = max(
            (int(item["seq"]) for item in transcript if isinstance(item.get("seq"), int)),
            default=-1,
        ) + 1
        transcript.append({"role": role, "content": content, "seq": next_seq})
        return transcript

    def _record(self, role: str, content: str) -> list[str]:
        diagnostics = self.manager.reconcile(
            self._snapshot_with(role, content), self.session_id
        )
        self.last_diagnostics = list(diagnostics)
        return diagnostics

    def _show_current_memory(self) -> None:
        store = self.manager._get_store()
        states = store.fetch_states(scope="global", status="current")
        states += store.fetch_states(scope=self.session_id, status="current")
        observations = store.fetch_observations(scope="global", status="current")
        observations += store.fetch_observations(scope=self.session_id, status="current")
        self._write("Current Memory V2 records:")
        if not states and not observations:
            self._write("  (none)")
        for state in states:
            self._write(
                f"  state {state.state_id} [{state.scope}] {state.concept} = "
                f"{state.value} | sources={state.source_ids}"
            )
        for item in observations:
            self._write(
                f"  observation {item.observation_id} [{item.scope}] "
                f"{item.concept} = {item.value} | sources={item.source_ids} "
                f"proof_count={item.proof_count}"
            )

    def _show_diagnostics(self) -> None:
        self._write("Latest reconcile diagnostics:")
        if self.last_diagnostics:
            for item in self.last_diagnostics:
                self._write(f"  {item}")
        else:
            self._write("  (none)")
        rows = self.manager.formation_status(session_id=self.session_id)
        self._write("Formation status:")
        if not rows:
            self._write("  (none)")
        for row in rows[-10:]:
            self._write(
                f"  message={row['message_id']} status={row['status']} "
                f"attempts={row['attempts']} diagnostics={row['diagnostics']}"
            )

    def _show_history(self, session_id: Optional[str] = None) -> None:
        target = self._validate_session(session_id or self.session_id)
        rows = self.manager._get_store().fetch_conversation(target, limit=500)
        self._write(f"History for {target}:")
        if not rows:
            self._write("  (none)")
        for row in rows:
            self._write(f"  [{row.seq}] {row.role}: {row.content}")

    def _show_sessions(self) -> None:
        rows = self.manager._get_store().list_sessions()
        self._write("Sessions:")
        if not rows:
            self._write("  (none)")
        for row in rows:
            self._write(
                f"  {row.session_id} status={row.status} "
                f"last_activity={row.last_activity_at:.3f}"
            )

    def _evidence(self, query: str) -> None:
        query = query.strip()
        if not query:
            self._write("Usage: :evidence QUERY")
            return
        episode = self.manager.verified_episode_history(query)
        if episode is not None:
            self._write("Episodic evidence:")
            self._write(episode)
            return
        evidence = self.manager.memory_query_evidence(
            {
                "sources": ["preferences", "facts", "experiences"],
                "query": query,
                # Leave the count unset so the typed intent parser honors
                # wording such as "last three" instead of forcing 30.
                "result_count": None,
                "order": "recent",
                # Let query intent distinguish current questions from
                # historical wording such as "previously liked".
                "temporal_scope": None,
            },
            session_id=self.session_id,
        )
        self._write("Memory evidence:")
        self._write(evidence or "  (abstained: no verified evidence)")

    def handle(self, line: str) -> bool:
        """Handle one line. Return False when the caller should exit."""
        line = line.strip()
        if not line:
            return True
        if not line.startswith(":"):
            diagnostics = self._record("user", line)
            self._write(f"Recorded user turn in session {self.session_id}.")
            if diagnostics:
                self._write("Diagnostics: " + " | ".join(diagnostics))
            self._show_current_memory()
            return True
        command, _, argument = line[1:].partition(" ")
        command = command.casefold()
        argument = argument.strip()
        if command in {"quit", "exit"}:
            return False
        if command == "help":
            self._write(HELP.rstrip())
        elif command == "session":
            self.session_id = self._validate_session(argument)
            self.manager.ensure_session(self.session_id)
            self._write(f"Using session {self.session_id}.")
        elif command == "sessions":
            self._show_sessions()
        elif command == "assistant":
            if not argument:
                self._write("Usage: :assistant TEXT")
            else:
                diagnostics = self._record("assistant", argument)
                self._write(f"Recorded assistant turn in session {self.session_id}.")
                if diagnostics:
                    self._write("Diagnostics: " + " | ".join(diagnostics))
        elif command in {"evidence", "ask"}:
            self._evidence(argument)
        elif command == "memory":
            self._show_current_memory()
        elif command == "history":
            self._show_history(argument or None)
        elif command in {"diagnostics", "diag"}:
            self._show_diagnostics()
        else:
            self._write(f"Unknown command :{command}. Use :help.")
        return True

    def run(
        self,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        self._write(f"Memory V2 local console; database={self.db_path}")
        self._write(f"Session={self.session_id}; V2 formation/retrieval enabled.")
        self._write("The database is isolated from production. Use :help for commands.")
        while True:
            try:
                line = input_fn(f"[{self.session_id}]> ")
            except (EOFError, KeyboardInterrupt):
                self._write()
                break
            if not self.handle(line):
                break


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive disposable Memory V2 console")
    parser.add_argument(
        "--model",
        default=os.environ.get("PRIMORA_OPENAI_MODEL", "llama-server"),
        help="model name sent to the local llama.cpp server",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("PRIMORA_OPENAI_URL", "http://localhost:8080/v1"),
        help="OpenAI-compatible llama.cpp base URL",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("PRIMORA_OPENAI_TIMEOUT", "300")),
        help="provider request timeout in seconds",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="optional isolated SQLite path; defaults to a disposable temporary file",
    )
    parser.add_argument("--session", default="default", help="initial session name")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.db is not None:
            db_path = require_isolated(args.db)
            temporary = None
        else:
            temporary = tempfile.TemporaryDirectory(prefix="primora-memory-v2-")
            db_path = require_isolated(Path(temporary.name) / "memory.db")
        provider = OpenAICompatibleProvider(
            model=args.model, base_url=args.url, timeout=args.timeout
        )
        console = MemoryV2Console(
            db_path, provider, session_id=args.session,
        )
        try:
            console.run()
        finally:
            if temporary is not None:
                temporary.cleanup()
        return 0
    except (IsolationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

