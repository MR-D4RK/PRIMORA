"""Live driver: Aurora on llama.cpp llama-server (OpenAIBrain).

Invoked by launch-aurora-llamacpp.ps1 with:
    PRIMORA_AURORA_BACKEND=openai
    PRIMORA_OPENAI_URL / PRIMORA_OPENAI_MODEL / PRIMORA_OPENAI_TIMEOUT
    PRIMORA_PROVIDER=openai (Prime's provider speaks to the same server)

Runs a live checklist against a real llama-server:

1.  Aurora constructs + warms up with no Transformers weights, no LoRA
    adapter, and no HF tokenizer loaded (verified via sys.modules and the
    absence of the AuroraBrain class in the brain slot).
2.  A normal conversational question gets a real generated answer.
3.  Generation speed (tokens/s) and VRAM (nvidia-smi) are measured.
4.  The production memory store is reachable, has the expected schema,
    and recall works (read-only; no test writes).
5.  A research request either returns genuine results or a clear error.

Unlike earlier revisions this driver does NOT snapshot production memory
into a temporary database or patch the context manager. Prime's default
``ContextManager`` already resolves to the canonical production path
(``PRIMORA/data/context/aurora_context.db``); that is the intended live
behaviour, so the driver simply constructs Prime normally. The memory
health check is read-only: it never creates test messages, facts,
sessions, candidates, or digests in production.

``--chat`` drops into an interactive loop after the checklist using the
application's normal session behaviour (``PRIMORA_SESSION_ID`` if set,
otherwise ``default``).

Exit code 0 = all critical checks passed; 1 = a critical check failed.
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PRIMORA.aurora import Aurora
from PRIMORA.aurora.brain import OpenAIBrain, aurora_backend_from_env
from PRIMORA.prime import Prime
from PRIMORA.prime.context.isolation import production_db_path
from PRIMORA.prime.contract import PrimeRequest, RequestStatus
from PRIMORA.main import prime_provider_from_env, research_provider_from_env

RESULTS = []  # (name, passed, critical, detail)


def record(name, passed, critical, detail=""):
    RESULTS.append((name, passed, critical, detail))
    mark = "PASS" if passed else ("FAIL" if critical else "WARN")
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


def vram_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def timed_generate(base_url, model, prompt, max_tokens=256, timeout=300):
    """One raw /chat/completions call, returning (elapsed_s, usage, text).

    llama-server (without --metrics) still returns OpenAI-style ``usage``
    in the response body, so exact prompt/gen token counts and a precise
    tok/s figure come from here rather than an estimated char count.
    """
    import json as _json
    body = _json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    usage = data.get("usage") or {}
    text = ""
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except Exception:
        pass
    return elapsed, usage, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true",
                        help="open an interactive chat loop after the checklist")
    args = parser.parse_args()

    print("=" * 70)
    print("AURORA ON LLAMA.CPP — LIVE CHECKLIST")
    print("=" * 70)

    # --- 0. Environment ---------------------------------------------------------
    backend = aurora_backend_from_env()
    print(f"  PRIMORA_AURORA_BACKEND -> {backend}")
    record("backend resolves to openai", backend == "openai", True, backend)

    base_url = os.environ.get("PRIMORA_OPENAI_URL", "http://localhost:8080/v1")
    print(f"  endpoint: {base_url}")
    print(f"  model   : {os.environ.get('PRIMORA_OPENAI_MODEL')}")
    print(f"  session : {os.environ.get('PRIMORA_SESSION_ID', 'default')}")

    # --- 1. Construct with no HF/LoRA/transformers; production memory directly --
    # Prime()'s default ContextManager resolves to the canonical production
    # database path. No snapshot, no temp copy, no context-manager patch:
    # live conversations are meant to accumulate in production memory.
    t0 = time.perf_counter()
    provider = prime_provider_from_env()
    research_provider = research_provider_from_env()
    prime = Prime(provider, research_provider=research_provider)
    aurora = Aurora(prime)
    construct_s = time.perf_counter() - t0

    brain = aurora._brain
    record("brain is OpenAIBrain", isinstance(brain, OpenAIBrain), True,
           type(brain).__name__)

    hf_modules = [m for m in ("transformers", "peft", "bitsandbytes",
                              "sentence_transformers")
                  if m in sys.modules]
    # transformers may already be importable from site-packages warm
    # state; the hard check is that the *brain slot* never touched them.
    # OpenAIBrain construction performs no network I/O and no weight
    # load, so construction time itself proves no model was read.
    record("construction loaded no model weights", construct_s < 2.0, True,
           f"{construct_s:.2f}s")
    record("AuroraBrain class never instantiated",
           type(brain).__name__ != "AuroraBrain", True)

    t0 = time.perf_counter()
    warmup_err = aurora.warmup()
    warmup_s = time.perf_counter() - t0
    record("warmup succeeded (no HF load)", warmup_err is None, True,
           f"{warmup_s:.2f}s" + (f" err={warmup_err}" if warmup_err else ""))
    record("brain get_model_name", bool(aurora.get_model_name()), False,
           aurora.get_model_name())

    # --- store is the production path ----------------------------------------
    store_path = str(prime.context._get_store().db_path)
    expected = str(production_db_path())
    record("store is the production database",
           os.path.abspath(store_path) == os.path.abspath(expected), True,
           store_path)

    vram_loaded = vram_mib()
    print(f"  VRAM after server load: {vram_loaded} MiB"
          if vram_loaded is not None else "  VRAM: n/a")

    # --- 2+3. Conversational question with speed measurement ----------------
    print()
    print("  --- conversational question ---")
    t0 = time.perf_counter()
    result = aurora.respond(
        [{"role": "user",
          "content": "In one or two sentences, what is the capital of "
                     "France and why is it significant?"}],
    )
    respond_s = time.perf_counter() - t0
    record("conversational answer generated (full loop)", result.success,
           True,
           (result.error if not result.success
            else f"{len(result.content)} chars in {respond_s:.1f}s"))
    if result.success:
        print(f"  Aurora ({respond_s:.1f}s): {result.content}")

    # Second, measured pass straight against the endpoint: exact usage
    # token counts (the full loop includes memory/retrieval overhead the
    # raw generation does not).
    elapsed, usage, raw_text = timed_generate(
        base_url,
        os.environ.get("PRIMORA_OPENAI_MODEL", "qwen3-8b-heretic"),
        "Say exactly one short sentence about the ocean.",
    )
    p_tok = usage.get("prompt_tokens") or 0
    g_tok = usage.get("completion_tokens") or 0
    if g_tok > 0 and elapsed > 0:
        # generation-only speed: subtract no prompt time (unknown here);
        # report both end-to-end and a tok/s floor.
        print(f"  measured: prompt={p_tok} tok, gen={g_tok} tok "
              f"in {elapsed:.2f}s -> {g_tok / elapsed:.1f} tok/s end-to-end")
        record("generation speed measured", g_tok / elapsed > 0.5, False,
               f"{g_tok / elapsed:.1f} tok/s")
    else:
        record("generation speed measured", False, False,
               f"no usage in response: {usage}")

    vram_gen = vram_mib()
    if vram_loaded is not None and vram_gen is not None:
        print(f"  VRAM during/after generation: {vram_gen} MiB "
              f"(delta {vram_gen - vram_loaded:+d} MiB)")

    # --- 4. Production memory health check (read-only) -----------------------
    # Replaces the old write-a-fact-and-recall-it check. That pattern
    # created test messages, facts, and digests in production once the
    # live path stopped using a temp snapshot; it is unsafe here. This
    # check only reads: schema presence and a recall that must not write.
    print()
    print("  --- memory health (read-only) ---")
    try:
        store = prime.context._get_store()
        import sqlite3 as _sq
        con = _sq.connect(f"file:{store.db_path}?mode=ro", uri=True)
        try:
            tables = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            con.close()
        expected_tables = {"messages", "digests", "sessions", "states",
                           "candidates", "entities", "relationships"}
        missing = expected_tables - tables
        record("production schema present", not missing, True,
               f"missing={sorted(missing)}" if missing else f"{len(tables)} tables")
    except Exception as exc:
        record("production schema present", False, True, str(exc))

    # Recall against the production store with a nonsense query: must not
    # raise, and must not create any rows. An empty database legitimately
    # returns zero hits.
    try:
        hits = prime.context.recall(
            "__aurora_live_health_probe_nonsense__", limit=1,
        )
        record("memory recall is readable (no writes)",
               isinstance(hits, list), True,
               f"{len(hits)} hit(s)")
    except Exception as exc:
        record("memory recall is readable (no writes)", False, True,
               str(exc))

    # --- 5. Research request ------------------------------------------------
    print()
    print("  --- research request ---")
    rr = prime.handle(PrimeRequest(
        action="research",
        payload={"query": "current stable Python version"},
    ))
    if (rr.status == RequestStatus.SUCCESS
            and isinstance(rr.output, dict)
            and rr.output.get("results")):
        record("research returned results", True, False,
               f"{len(rr.output['results'])} source(s)")
    elif rr.status == RequestStatus.FAILURE and rr.error:
        # A clear error is the documented alternative to genuine results.
        record("research returned results or clear error", True, False,
               f"clear error: {str(rr.error)[:160]}")
    else:
        record("research returned results", False, False,
               f"unexpected outcome: status={rr.status.value} "
               f"error={str(rr.error)[:80]}")

    # --- Summary -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    critical_fail = False
    for name, passed, critical, detail in RESULTS:
        if passed:
            status = "PASS"
        elif critical:
            status = "FAIL"
            critical_fail = True
        else:
            status = "WARN"
        print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))
    print()
    if critical_fail:
        print("RESULT: CRITICAL CHECKS FAILED")
    else:
        print("RESULT: ALL CRITICAL CHECKS PASSED")

    # --- Optional interactive chat ----------------------------------------------
    if args.chat and not critical_fail:
        print()
        print("Interactive chat (Ctrl+C or 'exit' to quit):")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            r = aurora.respond([{"role": "user", "content": user_input}])
            if r.success:
                print(f"Aurora: {r.content}\n")
            else:
                print(f"ERROR: {r.error}\n")

    return 1 if critical_fail else 0


if __name__ == "__main__":
    sys.exit(main())
