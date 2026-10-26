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
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional, TextIO

from PRIMORA.prime.providers import OpenAICompatibleProvider

from .isolation import IsolationError, require_isolated
from . import entities, relationships
from .consolidation import ConsolidationService
from .memory_audit import MemoryAuditService
from .manager import ContextManager
from .relational_retrieval import RelationalRetriever


HELP = """Commands:
  :session NAME       switch to or create a conversation session
  :sessions            list stored sessions
  :assistant TEXT      append an assistant turn to the current session
  :evidence QUERY      retrieve source-backed Memory V2 evidence
  :memory              show current states, observations, and source ids
  :history [NAME]      show raw turns from a session
  :diagnostics         show formation status and the latest diagnostics
  :m6 preview [SCOPE]  preview evidence-backed consolidation candidates
  :m6 sources N        inspect sources supporting preview candidate N
  :m6 apply N          explicitly apply preview candidate N as a Prime candidate
  :m6 apply-all        explicitly apply every preview candidate
  :m7 entity NAME TYPE SOURCE_ID
                        create a source-backed entity candidate
  :m7 alias ENTITY_ID ALIAS SOURCE_ID
                        create a source-backed alias candidate
  :m7 relationship TYPE SUBJECT_ID OBJECT_ID SOURCE_ID
                        create a source-backed relationship candidate
  :m7 approve KIND ID  approve entity, alias, or relationship through Prime
  :m7 query NAME [HOPS] query bounded relational evidence by name or alias
  :m8 audit [SCOPE]    run the read-only metamemory audit
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
        self._m6_report = None
        self._m6_applied: set[int] = set()

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

    def _m6(self, argument: str) -> None:
        """Preview, inspect, and explicitly apply M6 proposals."""
        tokens = shlex.split(argument)
        action = tokens[0].casefold() if tokens else "preview"
        service = ConsolidationService(self.manager._get_store(), enabled=True)
        if action == "preview":
            scope = tokens[1] if len(tokens) > 1 else "global"
            if scope == "session":
                scope = self.session_id
            if scope not in {"global", self.session_id}:
                self._write("Usage: :m6 preview [global|session]")
                return
            self._m6_report = service.run(self.session_id, scope=scope, dry_run=True)
            self._m6_applied.clear()
            self._write(
                f"M6 preview run={self._m6_report.run_id} scanned={self._m6_report.scanned} "
                f"groups={self._m6_report.grouped} proposals={len(self._m6_report.proposals)}"
            )
            if not self._m6_report.proposals:
                self._write("  (no repeated source-backed candidates)")
            for index, proposal in enumerate(self._m6_report.proposals, 1):
                self._write(
                    f"  [{index}] {proposal.concept} = {proposal.value} "
                    f"proof_count={proposal.proof_count} sources={proposal.source_ids} "
                    f"status={proposal.status} uncertainty={proposal.uncertainty}"
                )
                if proposal.occurred_from is not None or proposal.occurred_until is not None:
                    self._write(
                        f"      occurred_from={proposal.occurred_from} "
                        f"occurred_until={proposal.occurred_until}"
                    )
            return
        if action == "sources":
            if self._m6_report is None:
                self._write("Run :m6 preview before inspecting sources.")
                return
            if len(tokens) != 2 or not tokens[1].isdigit():
                self._write("Usage: :m6 sources N")
                return
            index = int(tokens[1]) - 1
            if not 0 <= index < len(self._m6_report.proposals):
                self._write("M6 candidate number is out of range.")
                return
            proposal = self._m6_report.proposals[index]
            self._write(f"M6 supporting sources for candidate {index + 1}:")
            for source_id in proposal.source_ids:
                message = self.manager._get_store().fetch_message(source_id)
                if message is None:
                    self._write(f"  source_id={source_id} (missing; admission will fail closed)")
                    continue
                self._write(
                    f"  source_id={source_id} session={message.session_id} "
                    f"role={message.role} reported_at={message.timestamp}: {message.content}"
                )
                self._write(f"    quote={proposal.source_quotes.get(source_id, '')}")
            return
        if action in {"apply", "apply-all"}:
            if self._m6_report is None:
                self._write("Run :m6 preview before applying candidates.")
                return
            if action == "apply-all":
                indices = range(len(self._m6_report.proposals))
            elif len(tokens) == 2 and tokens[1].isdigit():
                indices = [int(tokens[1]) - 1]
            else:
                self._write("Usage: :m6 apply N | :m6 apply-all")
                return
            for index in indices:
                if not 0 <= index < len(self._m6_report.proposals):
                    self._write(f"M6 candidate {index + 1} is out of range.")
                    continue
                if index in self._m6_applied:
                    self._write(f"M6 candidate {index + 1} was already applied in this preview.")
                    continue
                proposal = self._m6_report.proposals[index]
                try:
                    row = service.apply_proposal(proposal, session_id=self.session_id)
                except (ValueError, RuntimeError) as exc:
                    self._write(f"M6 candidate {index + 1} rejected by Prime: {exc}")
                    continue
                self._m6_applied.add(index)
                self._write(
                    f"M6 candidate {index + 1} admitted as observation {row.observation_id} "
                    f"status={row.status} sources={row.source_ids} proof_count={row.proof_count}"
                )
            return
        self._write("Usage: :m6 preview [global|session] | :m6 sources N | :m6 apply N | :m6 apply-all")

    def _source_id(self, value: str) -> Optional[int]:
        try:
            source_id = int(value)
        except (TypeError, ValueError):
            self._write("SOURCE_ID must be an integer user message ID in the current session.")
            return None
        message = self.manager._get_store().fetch_message(source_id)
        if message is None or message.role != "user" or message.session_id != self.session_id:
            self._write("Prime source validation failed: source must be a user message in the current session.")
            return None
        return source_id

    def _m7(self, argument: str) -> None:
        """Create/approve/query M7 records through existing Prime APIs."""
        try:
            tokens = shlex.split(argument)
        except ValueError as exc:
            self._write(f"M7 argument error: {exc}")
            return
        if not tokens:
            self._write("Usage: :m7 entity|alias|relationship|approve|query ...")
            return
        action = tokens[0].casefold()
        store = self.manager._get_store()
        try:
            if action == "entity" and len(tokens) == 4:
                source_id = self._source_id(tokens[3])
                if source_id is None:
                    return
                entity_id, diagnostics = entities.create_candidate(
                    store, self.session_id, tokens[1], tokens[2].upper(), [source_id]
                )
                self._write(f"M7 entity candidate={entity_id} status=candidate sources={[source_id]}")
                for item in diagnostics:
                    self._write(f"  {item}")
                return
            if action == "alias" and len(tokens) == 4:
                source_id = self._source_id(tokens[3])
                if source_id is None:
                    return
                alias_id, diagnostics = entities.add_alias_candidate(
                    store, int(tokens[1]), tokens[2], self.session_id, [source_id]
                )
                self._write(f"M7 alias candidate={alias_id} status=candidate sources={[source_id]}")
                for item in diagnostics:
                    self._write(f"  {item}")
                return
            if action == "relationship" and len(tokens) == 5:
                source_id = self._source_id(tokens[4])
                if source_id is None:
                    return
                relationship_id, diagnostics = relationships.create_candidate(
                    store, self.session_id, tokens[1].casefold(), int(tokens[2]),
                    int(tokens[3]), [source_id]
                )
                self._write(
                    f"M7 relationship candidate={relationship_id} status=candidate "
                    f"sources={[source_id]}"
                )
                for item in diagnostics:
                    self._write(f"  {item}")
                return
            if action == "approve" and len(tokens) == 3:
                kind, record_id = tokens[1].casefold(), int(tokens[2])
                if kind == "entity":
                    row, diagnostics = entities.confirm(store, record_id)
                    status = row.status
                elif kind == "alias":
                    row, diagnostics = entities.confirm_alias(store, record_id)
                    status = row.status
                elif kind == "relationship":
                    row, diagnostics = relationships.confirm(store, record_id)
                    status = row.status
                else:
                    self._write("Usage: :m7 approve entity|alias|relationship ID")
                    return
                self._write(f"M7 {kind} {record_id} approved by Prime; status={status}")
                for item in diagnostics:
                    self._write(f"  {item}")
                return
            if action == "query" and len(tokens) in {2, 3}:
                hops = int(tokens[2]) if len(tokens) == 3 else 2
                result = RelationalRetriever(store, enabled=True).retrieve(
                    self.session_id, tokens[1], hop_limit=hops
                )
                self._write(
                    f"M7 relational evidence packets={len(result.packets)} "
                    f"truncated={result.truncated}"
                )
                for packet in result.packets:
                    self._write(
                        f"  {packet.packet_type} {packet.subject} -> {packet.object} "
                        f"type={packet.relationship_type} hop={packet.hop} "
                        f"inferred={packet.inferred} sources={packet.source_ids}"
                    )
                    self._write(f"    reason={packet.reason}")
                for item in result.diagnostics:
                    self._write(f"  diagnostic={item}")
                return
        except (ValueError, RuntimeError) as exc:
            self._write(f"M7 Prime validation failed: {exc}")
            return
        self._write(
            "Usage: :m7 entity NAME TYPE SOURCE_ID | :m7 alias ENTITY_ID ALIAS SOURCE_ID | "
            ":m7 relationship TYPE SUBJECT_ID OBJECT_ID SOURCE_ID | "
            ":m7 approve KIND ID | :m7 query NAME [HOPS]"
        )

    def _m8(self, argument: str) -> None:
        tokens = shlex.split(argument)
        if tokens and tokens[0].casefold() not in {"audit", "run"}:
            self._write("Usage: :m8 audit [global|SESSION]")
            return
        scope = tokens[1] if len(tokens) > 1 else None
        if scope == "session":
            scope = self.session_id
        if scope is not None and scope not in {"global", self.session_id}:
            self._write("Usage: :m8 audit [global|session]")
            return
        report = MemoryAuditService(self.manager._get_store(), enabled=True).run(scope=scope)
        self._write(
            f"M8 read-only audit run={report.run_id} scanned={report.scanned} "
            f"findings={len(report.findings)} proposals={len(report.proposals)}"
        )
        if not report.findings:
            self._write("  (no integrity findings)")
        for finding in report.findings:
            self._write(
                f"  {finding.severity} {finding.kind} {finding.record_kind}#{finding.record_id} "
                f"{finding.concept} = {finding.value} sources={finding.source_ids}"
            )
            self._write(f"    {finding.message}; recommendation={finding.recommendation}")
        for item in report.diagnostics:
            self._write(f"  diagnostic={item}")

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
        elif command == "m6":
            self._m6(argument)
        elif command == "m7":
            self._m7(argument)
        elif command == "m8":
            self._m8(argument)
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

