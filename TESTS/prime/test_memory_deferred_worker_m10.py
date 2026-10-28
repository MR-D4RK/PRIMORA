"""Deferred formation worker: bounded automatic progress and source safety.

Uses temporary SQLite and a deterministic stub provider; no local Qwen needed.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time

from PRIMORA.local_model_gate import local_model_request
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.store import ContextStore


class _Provider:
    def __init__(self, failures=0):
        self.calls = 0
        self.failures = failures
        self.called = threading.Event()

    def invoke(self, capability, payload):
        assert capability == "memory_formation"
        self.calls += 1
        self.called.set()
        if self.calls <= self.failures:
            return PrimeResult.failure("temporary provider interruption")
        return PrimeResult.success({"content": json.dumps({"candidates": [{
            "kind": "preference",
            "concept": "paper lanterns",
            "value": "enjoys paper lanterns",
            "source_ids": [payload["source_message_id"]],
            "source_quote": payload["messages"][-1]["content"],
            "scope": "global",
        }]})})


def test_processing_lease_is_not_stolen_by_second_store(tmp_path):
    path = tmp_path / "lease.db"
    first, second = ContextStore(path), ContextStore(path)
    mid, _, _ = first.append("session", 0, "user", "I enjoy paper lanterns.")
    assert first.enqueue_formation_job(mid, "session")
    claimed = first.claim_formation_jobs(limit=1)
    assert len(claimed) == 1
    assert second.claim_formation_jobs(limit=1) == []
    assert second.formation_jobs(status="processing")[0]["attempts"] == 1
    # Simulate a process terminated before finishing its leased job.
    with sqlite3.connect(path) as con:
        con.execute(
            "UPDATE formation_jobs SET updated_at=? WHERE job_id=?",
            (time.time() - 2000, claimed[0]["job_id"]),
        )
    recovered = second.claim_formation_jobs(limit=1)
    assert len(recovered) == 1
    assert recovered[0]["message_id"] == mid
    assert recovered[0]["attempts"] == 2
    second.finish_formation_job(recovered[0]["job_id"], "complete")
    assert first.claim_formation_jobs(limit=1) == []


def test_failed_background_job_obeys_retry_backoff(tmp_path):
    store = ContextStore(tmp_path / "backoff.db")
    mid, _, _ = store.append("s", 0, "user", "I enjoy paper lanterns.")
    store.enqueue_formation_job(mid, "s")
    job = store.claim_formation_jobs()[0]
    store.finish_formation_job(job["job_id"], "failed", "temporary")
    assert store.claim_formation_jobs(respect_backoff=True) == []
    # Maintenance drains retain an explicit immediate-retry route.
    assert store.claim_formation_jobs(respect_backoff=False)[0]["attempts"] == 2


def test_env_opt_in_worker_drains_queue_without_manual_call(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIMORA_MEMORY_FORMATION_DEFERRED", "1")
    monkeypatch.setenv("PRIMORA_MEMORY_FORMATION_AUTO_WORKER", "1")
    provider = _Provider()
    path = tmp_path / "automatic.db"
    manager = ContextManager(
        db_path=path, enable_memory_v2=True,
        hybrid_retrieval=False, formation_provider=provider,
    )
    try:
        manager.reconcile([{"role": "user", "content": "I enjoy paper lanterns."}], "s")
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            # The queue job and its derived-memory status are separate writes.
            if (manager.formation_jobs("s", "complete")
                    and manager.formation_status("s", "complete")):
                break
            time.sleep(0.05)
        assert provider.called.is_set()
        assert provider.calls == 1
        assert len(manager.formation_jobs("s", "complete")) == 1
        messages = manager._get_store().fetch_range("s")
        assert len(messages) == 1
        # Deterministic formation already stores this as "likes for paper
        # lanterns". Typed formation reuses that sourced preference rather
        # than creating a duplicate named just "paper lanterns".
        matching = [
            state for state in manager._get_store().fetch_states(
                scope="global", status="current"
            )
            if state.value == "enjoys paper lanterns"
            and "paper lanterns" in state.concept
            and messages[0].message_id in state.source_ids
        ]
        assert len(matching) == 1
        assert any(
            diag.startswith("typed_formation:proposal_deduplicated:")
            for record in manager.formation_status("s", "complete")
            for diag in record["diagnostics"]
        )
    finally:
        manager.stop_formation_worker(timeout=2.0)


def test_env_disabled_worker_remains_manual(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIMORA_MEMORY_FORMATION_DEFERRED", "1")
    monkeypatch.setenv("PRIMORA_MEMORY_FORMATION_AUTO_WORKER", "0")
    provider = _Provider()
    manager = ContextManager(
        db_path=tmp_path / "manual.db", enable_memory_v2=True,
        formation_provider=provider,
    )
    manager.reconcile([{"role": "user", "content": "I enjoy paper lanterns."}], "s")
    assert provider.calls == 0
    assert manager.formation_jobs("s", "pending")
    manager.drain_formation()
    assert provider.calls == 1


def test_local_model_gate_serializes_loopback_requests(monkeypatch):
    monkeypatch.setenv("PRIMORA_LOCAL_MODEL_SERIALIZE", "1")
    started = threading.Event()
    acquired = threading.Event()

    def contender():
        started.set()
        with local_model_request("http://localhost:8080/v1/chat/completions"):
            acquired.set()

    with local_model_request("http://127.0.0.1:8080/v1/chat/completions"):
        thread = threading.Thread(target=contender)
        thread.start()
        assert started.wait(timeout=2)
        assert not acquired.wait(timeout=0.05)
    thread.join(timeout=2)
    assert acquired.is_set()
