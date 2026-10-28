"""M6 semantic index: local embeddings for structured memory and entities.

Scope, deliberately narrow:

* the approved local model `sentence-transformers/all-MiniLM-L6-v2`, CPU
  first, float32, L2-normalized, loaded offline from the local cache;
* only `states` and `entities` are embedded. Relationships hold no text and
  are traversed structurally instead; digests are verbatim spans of messages
  and raw messages are deliberately not embedded in this stage;
* the index is a derived cache. It stores a content hash rather than source
  text, and deleting every row is lossless.

Every failure mode here degrades to "no semantic channel" rather than raising.
Lexical retrieval must keep working when the model is missing, the index is
empty, or numpy is unavailable.
"""

import hashlib
import math
import time
from typing import Any, Optional

from . import entities as entities_mod

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cpu"
EMBED_BATCH = 32
EMBED_MAX_SEQUENCE = 256
DEFAULT_MODEL_DIM = 384

# What this stage is allowed to embed. Anything else is ignored by policy.
EMBEDDABLE_KINDS = ("state", "entity")

_STATE_INDEX_STATUSES = ("current", "superseded")
_ENTITY_INDEX_STATUSES = ("active",)

# Freshness metadata key for the unscoped (all-scope) indexing domain. Scoped
# domains use "<SYNC_TOKEN_KEY>:scope:<scope>" so they cannot overwrite it.
SYNC_TOKEN_KEY = "sync_token"

_MODEL_CACHE: dict[str, Any] = {}
_LOAD_ERROR = ""


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def state_text(state) -> str:
    """The text an embedded state represents. Never includes raw messages."""
    memory_type = f" ({state.memory_type})" if state.memory_type else ""
    return f"{state.concept} = {state.value}{memory_type}"


def entity_text(store, entity) -> str:
    """Canonical name, type, and confirmed aliases. No provenance text."""
    parts = [entity.display_name, f"({entity.entity_type})"]
    try:
        aliases = store.fetch_entity_aliases(
            entity_id=entity.entity_id, status="current", limit=8
        )
    except Exception:
        aliases = []
    if aliases:
        parts.append("also known as " + ", ".join(
            sorted({a.alias_text for a in aliases})
        ))
    return " ".join(parts)


def _numpy():
    try:
        import numpy
    except Exception:
        return None
    return numpy


def load_model(model_id: str = MODEL_ID, device: str = DEVICE):
    """Load the local model once per process. Returns None when unavailable.

    Offline by default so a normal retrieval path never reaches the network.
    A missing model, a missing library, or an empty local cache all return
    None instead of raising.
    """
    key = f"{model_id}@{device}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    global _LOAD_ERROR
    try:
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        _LOAD_ERROR = f"semantic_model_unavailable: {type(exc).__name__}: {exc}"
        _MODEL_CACHE[key] = None
        return None
    try:
        model = SentenceTransformer(model_id, device=device)
        model.eval()
        model.max_seq_length = min(
            getattr(model, "max_seq_length", EMBED_MAX_SEQUENCE),
            EMBED_MAX_SEQUENCE,
        )
    except Exception as exc:
        _LOAD_ERROR = f"semantic_model_load_failed: {type(exc).__name__}: {exc}"
        _MODEL_CACHE[key] = None
        return None
    _LOAD_ERROR = ""
    _MODEL_CACHE[key] = model
    return model


def load_error() -> str:
    return _LOAD_ERROR


def reset_model_cache() -> None:
    """Test seam: forget the cached model so loading can be re-attempted."""
    _MODEL_CACHE.clear()


def encode(model, texts: list[str]):
    """Encode to L2-normalized float32 vectors. Returns None on failure."""
    if model is None or not texts:
        return None
    numpy = _numpy()
    if numpy is None:
        return None
    try:
        vectors = model.encode(
            texts, batch_size=EMBED_BATCH, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
    except Exception:
        return None
    return numpy.asarray(vectors, dtype=numpy.float32)


class SemanticIndex:
    """Derived embedding cache over states and entities.

    Indexing is deterministic and idempotent: the same text under the same
    model always produces the same row content hash. No indexing path writes
    to `states`, `candidates`, `entities`, or `relationships`, and no
    similarity value can change what is considered current.
    """

    def __init__(self, store, model_id: str = MODEL_ID, device: str = DEVICE):
        self._store = store
        self._model_id = model_id
        self._device = device
        self._model = None
        self._probed = False

    # -- availability -------------------------------------------------
    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        return self._device

    def model(self):
        if not self._probed:
            self._probed = True
            self._model = load_model(self._model_id, self._device)
        return self._model

    def available(self) -> bool:
        return self.model() is not None and _numpy() is not None

    def dim(self) -> int:
        model = self.model()
        if model is None:
            return 0
        # Newer sentence-transformers renamed this accessor; support both so
        # the context package does not emit a deprecation warning.
        for accessor in ("get_embedding_dimension",
                         "get_sentence_embedding_dimension"):
            getter = getattr(model, accessor, None)
            if callable(getter):
                try:
                    return int(getter())
                except Exception:
                    continue
        return 0

    def diagnostics(self) -> list[str]:
        """Why the semantic channel is unavailable, if it is.

        Derived from this index's own state rather than a module global, so a
        caller can never be told semantic retrieval is working when this index
        has no model.
        """
        if self.model() is not None and _numpy() is not None:
            return []
        reason = load_error() or "semantic_model_unavailable: no local model"
        return [reason]

    # -- eligibility --------------------------------------------------
    def indexable_states(self, scope: Optional[str] = None) -> list[Any]:
        """States a semantic candidate could legitimately come from.

        `unresolved` and `rejected` rows are excluded here, at index time,
        so they can never be surfaced. Temporal validity is re-checked at
        query time because the evaluation time is not known while indexing.
        """
        statuses = ",".join(f"'{s}'" for s in _STATE_INDEX_STATUSES)
        query = (
            "SELECT id, session_id, scope, concept, value, source_ids,"
            " created_at, status, supersedes_id, authority, memory_type,"
            " confidence, observed_at, valid_from, valid_until, superseded_at"
            f" FROM states WHERE status IN ({statuses}) AND authority = 'user'"
        )
        params: list[Any] = []
        if scope is not None:
            query += " AND scope = ?"
            params.append(scope)
        query += " ORDER BY id ASC"
        with self._store._connection() as con:
            rows = con.execute(query, params).fetchall()
        return self._store._state_rows_to_records(
            self._store._visible("states", rows, lambda r: r[0]))

    def indexable_entities(self, scope: Optional[str] = None) -> list[Any]:
        statuses = ",".join(f"'{s}'" for s in _ENTITY_INDEX_STATUSES)
        query = (
            "SELECT id, session_id, scope, entity_type, display_name,"
            " normalized_name, status, merged_into_entity_id, source_ids,"
            " authority, confidence, created_at, observed_at, merged_at,"
            " source_episode_ids, valid_from, valid_until"
            f" FROM entities WHERE status IN ({statuses})"
            " AND authority = 'user'"
        )
        params: list[Any] = []
        if scope is not None:
            query += " AND scope = ?"
            params.append(scope)
        query += " ORDER BY id ASC"
        with self._store._connection() as con:
            rows = con.execute(query, params).fetchall()
        return self._store._entity_rows_to_records(
            self._store._visible("entities", rows, lambda r: r[0]))

    # -- lifecycle ----------------------------------------------------
    def _alias_map(self, scope=None) -> dict:
        """Confirmed alias text per entity, fetched once per sync."""
        try:
            rows = self._store.fetch_entity_aliases(
                scope=scope, status="current", limit=10000
            )
        except Exception:
            return {}
        out: dict = {}
        for row in rows:
            out.setdefault(row.entity_id, []).append(row.alias_text)
        return out

    def _render_entity(self, aliases: dict, entity) -> str:
        parts = [entity.display_name, f"({entity.entity_type})"]
        names = sorted(set(aliases.get(entity.entity_id, [])))
        if names:
            parts.append("also known as " + ", ".join(names))
        return " ".join(parts)

    def _stale_rows(self, kind: str, records, render) -> list:
        """Records whose embedded text is missing or has changed.

        Incremental by content hash, so a sync after one new memory embeds one
        row rather than the whole index.
        """
        existing = {
            row[1]: row[6] for row in self._store.fetch_vector_rows(
                kind=kind, model_id=self._model_id
            )
        }
        out: list = []
        for record in records:
            ref = record.state_id if kind == "state" else record.entity_id
            text = render(record)
            if existing.get(ref) != content_hash(text):
                out.append((record, text))
        return out

    def sync(self, scope: Optional[str] = None) -> dict:
        """Upsert embeddings for eligible rows whose content changed.

        Incremental and idempotent. Returns counts. Never raises: an
        unavailable model returns zeroes with a diagnostic, leaving lexical
        retrieval untouched.
        """
        report = {"states": 0, "entities": 0, "skipped": 0}
        if not self.available():
            report["skipped"] = 1
            return report
        aliases = self._alias_map(scope)
        targets = (
            ("state", self.indexable_states(scope), state_text, "states"),
            ("entity", self.indexable_entities(scope),
             lambda e: self._render_entity(aliases, e), "entities"),
        )
        for kind, records, render, report_key in targets:
            if not records:
                continue
            stale = self._stale_rows(kind, records, render)
            if not stale:
                continue
            vectors = encode(self.model(), [text for _r, text in stale])
            if vectors is None:
                report["skipped"] = 1
                return report
            now = time.time()
            for (record, text), vector in zip(stale, vectors):
                self._store.upsert_vector(
                    kind,
                    record.state_id if kind == "state" else record.entity_id,
                    record.scope, record.session_id, self._model_id,
                    int(vector.shape[0]), content_hash(text),
                    vector.tobytes(), timestamp=now,
                )
            report[report_key] = len(stale)
        self._store.set_index_meta("model_id", self._model_id)
        self._store.set_index_meta("dim", str(self.dim()))
        self._store.set_index_meta("synced_at", str(time.time()))
        self._store.set_index_meta("device", self._device)
        # Freshness is recorded against the domain that was actually indexed.
        # A scope-limited sync never stamps the global key, so it cannot make
        # an unrelated scope look synchronized.
        self._store.set_index_meta(
            self._token_key(scope), self._sync_token(scope)
        )
        return report

    def _token_key(self, scope: Optional[str]) -> str:
        """Metadata key holding the freshness token for one indexing domain.

        An unscoped sync indexes every scope, so it owns the global
        ``sync_token`` key. A scoped sync indexes exactly one scope and owns
        ``sync_token:scope:<scope>``. Keeping the domains in separate keys is
        what stops one scope's sync from marking another scope fresh, and it
        keeps the global key meaning "all scopes indexed" rather than
        "whatever scope happened to be synced last".
        """
        return SYNC_TOKEN_KEY if scope is None else f"{SYNC_TOKEN_KEY}:scope:{scope}"

    def _sync_token(self, scope: Optional[str] = None) -> str:
        """Cheap fingerprint of the eligible source rows in one domain.

        Four aggregates over indexed columns, filtered to the same scope the
        sync will embed, so the token describes the rows the index is
        supposed to hold. Used to skip all sync work when structured memory
        has not changed, so the live retrieval path pays nothing in the
        steady state.
        """
        state_statuses = ",".join(f"'{s}'" for s in _STATE_INDEX_STATUSES)
        entity_statuses = ",".join(f"'{s}'" for s in _ENTITY_INDEX_STATUSES)
        state_params: list[Any] = []
        entity_params: list[Any] = []
        state_scope = ""
        entity_scope = ""
        if scope is not None:
            state_scope = " AND scope = ?"
            entity_scope = " AND scope = ?"
            state_params.append(scope)
            entity_params.append(scope)
        with self._store._connection() as con:
            states = con.execute(
                "SELECT count(*), coalesce(max(id), 0),"
                f" coalesce(max(created_at), 0) FROM states"
                f" WHERE status IN ({state_statuses})"
                " AND authority = 'user'"
                f"{state_scope}",
                state_params,
            ).fetchone()
            entities = con.execute(
                "SELECT count(*), coalesce(max(id), 0),"
                f" coalesce(max(created_at), 0) FROM entities"
                f" WHERE status IN ({entity_statuses})"
                " AND authority = 'user'"
                f"{entity_scope}",
                entity_params,
            ).fetchone()
        return "|".join(str(value) for value in (*states, *entities))

    def maybe_sync(self, scope: Optional[str] = None) -> dict:
        """Opportunistic incremental sync, skipped when the index is fresh.

        One token read in the steady state, no model call. Never raises. The
        live retrieval path calls this so the semantic channel is genuinely
        populated, following the precedent of opportunistic digest compaction
        already in the request path. Freshness is judged against the same
        domain the sync would index, so a fresh scope never suppresses work
        for a different scope.
        """
        if not self.available():
            return {"states": 0, "entities": 0, "skipped": 1}
        token = self._sync_token(scope)
        if self._store.get_index_meta(self._token_key(scope)) == token:
            return {"states": 0, "entities": 0, "skipped": 0}
        return self.sync(scope)

    def needs_sync(self, scope: Optional[str] = None) -> bool:
        """Whether this indexing domain is behind its sources."""
        if not self.available():
            return False
        return (
            self._store.get_index_meta(self._token_key(scope))
            != self._sync_token(scope)
        )

    def _invalidate_token(self, scope: Optional[str] = None) -> None:
        """Clear one domain's freshness token so the next maybe_sync re-syncs.

        A token is a claim that the index matches its sources. Deleting
        vectors while leaving the claim in place produces an index that
        reports healthy over missing rows, and `maybe_sync` would short-
        circuit on it forever. Clearing the claim is what makes an
        interrupted index build self-healing instead of permanent.
        """
        self._store.set_index_meta(self._token_key(scope), "")

    def _embed_records(self, kind: str, records, render) -> int:
        texts = [render(record) for record in records]
        vectors = encode(self.model(), texts)
        if vectors is None:
            return 0
        written = 0
        now = time.time()
        for record, text, vector in zip(records, texts, vectors):
            self._store.upsert_vector(
                kind, record.state_id if kind == "state" else record.entity_id,
                record.scope, record.session_id, self._model_id,
                int(vector.shape[0]), content_hash(text),
                vector.tobytes(), timestamp=now,
            )
            written += 1
        self._store.set_index_meta("model_id", self._model_id)
        self._store.set_index_meta("dim", str(int(vectors.shape[1])))
        self._store.set_index_meta("synced_at", str(now))
        self._store.set_index_meta("device", self._device)
        return written

    def rebuild(self, scope: Optional[str] = None) -> dict[str, int]:
        """Drop this model's rows and re-embed from source. Lossless.

        The freshness token is invalidated *before* the delete, not after.
        If the model is unavailable, or `encode` fails mid-rebuild, the
        vectors are already gone; a surviving token would then claim a
        healthy index over an empty one and `maybe_sync` would no-op
        permanently, silently disabling the semantic channel. Invalidating
        first means the next `maybe_sync` sees a mismatch and retries.
        """
        self._invalidate_token(scope)
        if not self.available():
            return {"states": 0, "entities": 0, "skipped": 1}
        self._store.delete_vectors(model_id=self._model_id)
        return self.sync(scope)

    def stale(self, scope: Optional[str] = None) -> dict[str, Any]:
        """Report index rows that no longer match a live eligible record."""
        rows = self._store.fetch_vector_rows(model_id=self._model_id)
        missing: list[dict[str, Any]] = []
        mismatch: list[dict[str, Any]] = []
        orphaned: list[dict[str, Any]] = []
        live_states = {r.state_id: r for r in self.indexable_states()}
        live_entities = {r.entity_id: r for r in self.indexable_entities()}
        if scope is not None:
            live_states = {
                k: v for k, v in live_states.items() if v.scope == scope
            }
            live_entities = {
                k: v for k, v in live_entities.items() if v.scope == scope
            }
        seen: set[tuple[str, int]] = set()
        for kind, ref_id, row_scope, _sid, _mid, _dim, digest, _blob, _ts in rows:
            seen.add((kind, ref_id))
            live = live_states.get(ref_id) if kind == "state" else (
                live_entities.get(ref_id) if kind == "entity" else None
            )
            if live is None:
                orphaned.append({"kind": kind, "ref_id": ref_id})
                continue
            if live.scope != row_scope:
                mismatch.append(
                    {"kind": kind, "ref_id": ref_id, "reason": "scope"}
                )
                continue
            text = (
                state_text(live) if kind == "state"
                else entity_text(self._store, live)
            )
            if content_hash(text) != digest:
                mismatch.append(
                    {"kind": kind, "ref_id": ref_id, "reason": "content"}
                )
        for ref_id in live_states:
            if ("state", ref_id) not in seen:
                missing.append({"kind": "state", "ref_id": ref_id})
        for ref_id in live_entities:
            if ("entity", ref_id) not in seen:
                missing.append({"kind": "entity", "ref_id": ref_id})
        return {
            "indexed": len(rows), "missing": missing, "mismatch": mismatch,
            "orphaned": orphaned,
            "healthy": not (missing or mismatch or orphaned),
        }

    def status(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "device": self._device,
            "available": self.available(),
            "dim": self.dim(),
            "states": self._store.count_vectors(
                kind="state", model_id=self._model_id
            ),
            "entities": self._store.count_vectors(
                kind="entity", model_id=self._model_id
            ),
            "synced_at": self._store.get_index_meta("synced_at"),
            "diagnostics": self.diagnostics(),
        }

    # -- retrieval ----------------------------------------------------
    def search(
        self,
        query_text: str,
        scopes: list[str],
        kinds: tuple[str, ...] = EMBEDDABLE_KINDS,
        tau: float = 0.45,
        limit: int = 20,
        exclude: Optional[set[int]] = None,
    ) -> list[dict[str, Any]]:
        """Bounded, gated semantic candidate search.

        Deterministic: exact float32 dot product against L2-normalized
        vectors, ordered by similarity then by `ref_id` so equal scores keep
        a stable order. Gates are applied by the caller, which owns scope
        and temporal policy; this method only applies the floor, the kind
        restriction, and the limit.
        """
        numpy = _numpy()
        if numpy is None or not query_text or not query_text.strip():
            return []
        if limit <= 0:
            return []
        model = self.model()
        if model is None:
            return []
        excluded = exclude or set()
        rows = self._store.fetch_vector_rows(
            model_id=self._model_id, scope=None
        )
        if not rows:
            return []
        query_vector = encode(model, [query_text])
        if query_vector is None or query_vector.shape[0] == 0:
            return []
        query_vector = query_vector[0]

        scoped = [
            row for row in rows
            if row[2] in scopes and row[0] in kinds
            and (row[0] != "state" or row[1] not in excluded)
            and (row[0] != "entity" or row[1] not in excluded)
        ]
        if not scoped:
            return []
        matrix = numpy.frombuffer(
            b"".join(bytes(row[7]) for row in scoped), dtype=numpy.float32
        ).reshape(len(scoped), -1)
        if matrix.shape[1] != query_vector.shape[0]:
            # A model or dimension change invalidates the cache.
            return []
        scores = matrix @ query_vector
        order = sorted(
            range(len(scoped)),
            key=lambda i: (-float(scores[i]), scoped[i][1]),
        )
        out: list[dict[str, Any]] = []
        for index in order:
            similarity = float(scores[index])
            if similarity < tau:
                break
            kind, ref_id, scope, session_id, _mid, dim, digest, _blob, ts = (
                scoped[index]
            )
            out.append({
                "kind": kind, "ref_id": ref_id, "scope": scope,
                "session_id": session_id, "similarity": round(similarity, 6),
                "dim": dim, "content_hash": digest, "indexed_at": ts,
            })
            if len(out) >= limit:
                break
        return out
