"""Runtime-only Working Memory for one task/session.

Working Memory holds task-specific temporary information Aurora needs
right now: the current task, its objective, temporary decisions, selected
Active Memory references, selected raw evidence, and files under
examination. It is deliberately NOT durable:

* never written to the SQLite store,
* never shared between instances (no registry, no globals),
* cleared explicitly when the task ends,
* bounded in items and estimated tokens.

Promotion back to durable memory happens only through an explicit call
into Memory Formation (ContextManager.promote_to_candidate), never
automatically.
"""

import time
from typing import Any, Optional

from .manager import estimate_tokens
from .models import ActiveMemoryItem

TASK = "task"
OBJECTIVE = "objective"
DECISION = "decision"
MEMORY = "memory"
EVIDENCE = "evidence"
FILE = "file"


class WorkingMemory:
    """Scoped, bounded, non-persistent working set for one task."""

    def __init__(
        self,
        session_id: str,
        task: str = "",
        max_items: int = 12,
        max_tokens: int = 400,
    ):
        self._session_id = session_id
        self._task = task
        self._objective = ""
        self._decisions: list[dict[str, Any]] = []
        self._memories: list[ActiveMemoryItem] = []
        self._evidence: list[dict[str, Any]] = []
        self._files: list[dict[str, Any]] = []
        self._created_at = time.time()
        self.max_items = max_items
        self.max_tokens = max_tokens

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def task(self) -> str:
        return self._task

    def set_task(self, task: str) -> None:
        self._task = task or ""

    def set_objective(self, objective: str) -> None:
        self._objective = objective or ""

    def note_decision(self, text: str, authority: str = "user") -> None:
        if text:
            self._decisions.append(
                {"text": text, "authority": authority,
                 "created_at": time.time()})
            self._enforce_bounds()

    def adopt_active(self, item: ActiveMemoryItem) -> None:
        """Reference a selected Active Memory entry (copy of the record)."""
        self._memories.append(item)
        self._enforce_bounds()

    def add_evidence(
        self,
        role: str,
        content: str,
        source: str = "",
        authority: str = "user",
        timestamp: Optional[float] = None,
    ) -> None:
        """Add verbatim raw evidence. Never touches the history store."""
        if content:
            evidence_timestamp = timestamp if timestamp is not None else time.time()
            self._evidence.append(
                {"role": role, "content": content, "source": source,
                 "authority": authority, "created_at": time.time(),
                 "timestamp": evidence_timestamp})
            self._enforce_bounds()

    def add_file(self, path: str, note: str = "") -> None:
        if path:
            self._files.append({"path": path, "note": note,
                                "created_at": time.time()})
            self._enforce_bounds()

    def entry_count(self) -> int:
        return (len(self._decisions) + len(self._memories)
                + len(self._evidence) + len(self._files)
                + (1 if self._task else 0)
                + (1 if self._objective else 0))

    def size_tokens(self) -> int:
        total = 0
        for msg in self.to_messages():
            total += estimate_tokens(msg.get("content", "")) + 4
        return total

    def to_messages(self) -> list[dict[str, str]]:
        """Render the working set as context-shaped messages."""
        out: list[dict[str, str]] = []
        if self._task:
            out.append({"role": "system",
                        "content": "[working task] " + self._task})
        if self._objective:
            out.append({"role": "system",
                        "content": "[working objective] " + self._objective})
        for d in self._decisions:
            out.append({"role": "system",
                        "content": "[working decision] " + d["text"]})
        for item in self._memories:
            out.append({"role": "system", "content": item.content})
        for ev in self._evidence:
            out.append({"role": ev["role"], "content": ev["content"]})
        for f in self._files:
            line = "[working file] " + f["path"]
            if f["note"]:
                line += " (" + f["note"] + ")"
            out.append({"role": "system", "content": line})
        return out

    def clear(self) -> None:
        """End the task: drop everything except the session binding."""
        self._task = ""
        self._objective = ""
        self._decisions = []
        self._memories = []
        self._evidence = []
        self._files = []

    def _enforce_bounds(self) -> None:
        """Drop oldest evictable entries until within both caps.

        Eviction order: evidence, adopted memories, decisions, files.
        Task and objective are pinned: they define the task itself.
        """
        while ((self.entry_count() > self.max_items
                or self.size_tokens() > self.max_tokens)
               and (self._evidence or self._memories
                    or self._decisions or self._files)):
            if self._evidence:
                self._evidence.pop(0)
            elif self._memories:
                self._memories.pop(0)
            elif self._decisions:
                self._decisions.pop(0)
            else:
                self._files.pop(0)
