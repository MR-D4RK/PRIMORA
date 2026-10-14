# PRIMORA Architecture Audit — Aurora ↔ Prime Integration

> **HISTORICAL — 2026-09-16. Superseded.** Kept as a record of the original
> integration trace. Two parts of it are obsolete as of Stage 2E (Oct 2026):
> the `Router.route` pass-through was removed (Prime.handle is the single
> validation path; Orchestrator dispatches straight to Execution), and the
> capability system described here has since grown to seven registered
> capabilities with specialists. For current behavior see
> `DOCS/01_architecture/` and `DOCS/02_audits/PRIMORA_FULL_AUDIT.md`.

**Date**: 2026-09-16  
**Scope**: End-to-end trace `Aurora → Prime → Orchestrator → Router → Execution → Provider → result → Aurora`

---

## 1. Actual Execution Path (Code Trace)

```
Aurora.request(action, payload, context, metadata)
    │
    ├─► Creates PrimeRequest(action, payload, context, metadata)
    │
    └─► Calls self._prime.handle(req)  [Prime.handle()]

Prime.handle(request: PrimeRequest) → PrimeResult
    │
    ├─► Validates isinstance(request, PrimeRequest)
    │
    └─► Returns self._orchestrator.orchestrate(request)

Orchestrator.orchestrate(request: PrimeRequest) → PrimeResult
    │
    ├─► capability = self._router.route(request)  [Router.route()]
    │
    └─► Returns self._execution.execute(capability, request.payload)  [Execution.execute()]

Router.route(request: PrimeRequest) → str
    │
    └─► Returns request.action  (passes action through as capability)

Execution.execute(capability: str, payload: dict) → PrimeResult
    │
    └─► Returns self._provider.invoke(capability, payload)  [Provider.invoke()]

Provider.invoke(capability: str, payload: dict) → PrimeResult
    │
    └─► OllamaProvider: expects payload["messages"], calls Ollama /api/chat
        └─► Returns PrimeResult.success({"content": "...", "model": "..."})
        └─► Or PrimeResult.failure(error_message)

Result flows back up: Provider → Execution → Orchestrator → Prime → Aurora.request() → caller
```

---

## 2. What Currently Works

| Component | Status | Evidence |
|-----------|--------|----------|
| **Aurora → Prime boundary** | ✅ Works | `Aurora.request()` constructs `PrimeRequest`, calls `prime.handle()` |
| **Prime public entry** | ✅ Works | `Prime.handle(PrimeRequest) → PrimeResult` |
| **Orchestrator coordination** | ✅ Works | Delegates to Router then Execution |
| **Router** | ✅ Works | Returns `request.action` as capability string |
| **Execution** | ✅ Works | Delegates to injected `Provider.invoke(capability, payload)` |
| **Provider abstraction** | ✅ Works | `Provider` ABC + `OllamaProvider` implementation |
| **OllamaProvider real integration** | ✅ Works | Tests pass against live `qwen3-8b-16k:latest` on `http://127.0.0.1:11434` |
| **Contract types** | ✅ Works | `PrimeRequest` (action, payload, context, metadata) / `PrimeResult` (status, output, error, metadata) |
| **Dependency injection** | ✅ Works | Provider → Execution → Orchestrator → Prime all require injected deps |
| **Error handling** | ✅ Works | Invalid request type, unknown capability, provider errors all return `PrimeResult.failure` |

---

## 3. Data/Contracts Passing Between Aurora and Prime

### Aurora → Prime (Request)
```python
PrimeRequest(
    action: str,                    # e.g. "chat", "summarize", "code"
    payload: dict = {               # e.g. {"messages": [...], "options": {...}}
        "messages": list[dict],     # Required by OllamaProvider
        "options": dict             # Optional, passed to Ollama
    },
    context: dict | None = None,    # Passed through, NOT used by Prime internals
    metadata: dict | None = None    # Passed through, NOT used by Prime internals
)
```

### Prime → Aurora (Response)
```python
PrimeResult(
    status: RequestStatus,          # SUCCESS | FAILURE
    output: Any | None = {          # On success: {"content": "...", "model": "..."}
        "content": str,             # Model response text
        "model": str                # Model name used
    },
    error: str | None = None,       # On failure: error message
    metadata: dict | None = None    # Empty in current implementation
)
```

### What Prime Does NOT Use
- `PrimeRequest.context` — passed through but **ignored** by Router/Execution/Provider
- `PrimeRequest.metadata` — passed through but **ignored**
- `PrimeResult.metadata` — always `None` in current implementation

---

## 4. Missing for Real End-to-End Aurora Request

| Gap | Description | Impact |
|-----|-------------|--------|
| **No capability registry** | Router returns `action` directly as capability. No mapping from semantic action → provider capability. | Every action must exactly match a provider capability. No abstraction. |
| **No context propagation** | `context` and `metadata` flow through but are never used. | Cannot pass session info, user identity, or tracing to providers. |
| **Single provider only** | Prime holds exactly one `Provider`. No routing to different providers per capability. | All actions go to same Ollama model. No specialist routing. |
| **No result transformation** | Provider output `{"content": "...", "model": "..."}` passes through unchanged. | Aurora receives raw provider output; no normalization. |
| **No streaming** | `stream: False` hardcoded in OllamaProvider. | No token-by-token streaming to Aurora. |
| **No timeout per-request** | Timeout fixed at Provider construction (120s default). | Cannot set per-action timeouts. |

---

## 5. Architectural Mismatches

| Mismatch | Location | Severity |
|----------|----------|----------|
| **Router returns raw action** | `Router.route()` returns `request.action` directly | **MEDIUM** — Violates "Router selects capability" principle. Should map action → capability. |
| **Provider expects `messages` in payload** | `OllamaProvider.invoke()` requires `payload["messages"]` | **MEDIUM** — Couples Prime contract to chat-specific provider. Other capabilities (tools, search) would need different payload schemas. |
| **Context/metadata dead letters** | Passed through entire chain but never read | **LOW** — Wasted bandwidth, but no functional breakage. |
| **No capability validation** | Unknown capability only fails at Provider (`"Unknown capability: X"`) | **LOW** — Late failure, but still caught. |

---

## 6. Smallest Concrete Next Implementation Step

### Add a Capability Registry to Router

**File**: `PRIMORA/prime/router.py` (or new `PRIMORA/prime/capabilities.py`)

**Change**: Map semantic actions to provider capabilities with payload transformers.

```python
# Minimal addition to Router
class Router:
    def __init__(self):
        self._capabilities = {
            "chat": "chat",           # direct pass-through
            "summarize": "chat",      # maps to chat with system prompt
            "code": "chat",           # maps to chat with coding prompt
        }
    
    def route(self, request: PrimeRequest) -> tuple[str, dict]:
        """Returns (capability, transformed_payload)"""
        capability = self._capabilities.get(request.action, request.action)
        payload = self._transform_payload(request.action, request.payload)
        return capability, payload
    
    def _transform_payload(self, action: str, payload: dict) -> dict:
        # Add system prompts, restructure for provider, etc.
        return payload
```

**Why this step**:
1. Unblocks semantic actions (Aurora sends "summarize", Prime routes to "chat" with proper prompt)
2. Keeps Provider interface stable (still receives capability + payload)
3. Enables future specialist routing without changing Prime/Orchestrator/Execution
4. Context/metadata can be injected here for providers that need them
5. Single file change, backward compatible (defaults to pass-through)

---

## 7. Verification Checklist

| Check | Result |
|-------|--------|
| Aurora imports only `Prime`, `PrimeRequest`, `PrimeResult` from Prime | ✅ |
| Prime imports only `Provider` ABC from providers | ✅ |
| No circular imports | ✅ |
| All layers require injected dependencies | ✅ |
| Real Ollama integration test passes | ✅ |
| Contract types are frozen dataclasses | ✅ |
| Error types are structured (`PrimeResult.failure`) | ✅ |
| Brain is completely separate from Prime | ✅ |

---

**Verdict**: **Integration is functional and architecturally compliant**. The chain executes correctly end-to-end with live Ollama. The missing pieces are semantic capability routing and context propagation — not structural defects.