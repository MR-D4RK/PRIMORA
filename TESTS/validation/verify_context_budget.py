"""Real-model verification of the new context budget.

Four things to prove on the actual hardware, not in a stub:

  1. a short response still works
  2. a context near 1600 tokens runs
  3. a context near 1800 tokens runs without OOM
  4. a context above the ceiling is rejected by the *mechanism*, before the
     request ever reaches CUDA - so it degrades instead of crashing

Point 4 is the one that motivated the change. At the previous ceiling a long
turn died with "CUDA out of memory, 0 bytes free". It must now be trimmed and
reported by the context assembly, and the generation must still succeed.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, r"C:\PRIMORA PROJECT")

import torch  # noqa: E402

from PRIMORA.aurora.brain import AuroraBrain  # noqa: E402
from PRIMORA.aurora.configuration import AuroraConfig  # noqa: E402
from PRIMORA.prime.context import ContextManager  # noqa: E402


def vram():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3, total / 1024**3


def prompt_of(tokenizer, filler_words, question):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are Aurora. " + ("context note. " * filler_words)},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True)


def generate(brain, text, max_new_tokens=32, label=""):
    tokenizer, model = brain._tokenizer, brain._model
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    n = inputs["input_ids"].shape[1]
    torch.cuda.synchronize()
    used, total = vram()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                use_cache=True, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        produced = out.shape[1] - n
        print(f"  {label:34s} prompt={n:5d} produced={produced:3d} "
              f"{elapsed:5.2f}s  {produced / elapsed:5.2f} tok/s  OK")
        return True
    except torch.OutOfMemoryError as exc:
        print(f"  {label:34s} prompt={n:5d}  *** OOM: {str(exc)[:70]}")
        torch.cuda.empty_cache()
        return False
    except Exception as exc:
        print(f"  {label:34s} prompt={n:5d}  *** {type(exc).__name__}: "
              f"{str(exc)[:70]}")
        return False


def main():
    config = AuroraConfig.from_env()
    report = config.context_budget_report()
    print("=" * 72)
    print("CONTEXT BUDGET VERIFICATION - REAL MODEL")
    print("=" * 72)
    print(f"  target={report['context_target']} ceiling={report['context_ceiling']} "
          f"reserve={report['output_reserve']} total={report['context_total']}")
    print(f"  allocator: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

    brain = AuroraBrain(config)
    started = time.perf_counter()
    problem = brain.warmup()
    print(f"  warmup: {time.perf_counter() - started:.1f}s  {problem or 'ok'}")
    if problem:
        return 1
    used, total = vram()
    print(f"  vram after load: {used:.2f} / {total:.2f} GB "
          f"({used / total * 100:.0f}%)")
    print(f"  brain hard ceiling: {brain._max_context_tokens}")

    tokenizer = brain._tokenizer
    q = "Reply in one short sentence."

    print("\n--- 1/2/3: direct generation at increasing context ---")
    results = {}
    results["short"] = generate(brain, prompt_of(tokenizer, 5, q), 32, "short")
    results["1600"] = generate(brain, prompt_of(tokenizer, 520, q), 32, "~1600 tokens")
    results["1800"] = generate(brain, prompt_of(tokenizer, 600, q), 32, "~1800 tokens")

    print("\n--- 4: above the ceiling, through the real mechanism ---")
    context = ContextManager(db_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_probe.db"))
    history = [{"role": "user", "content": "note " * 60} for _ in range(80)]
    history.append({"role": "user", "content": q})
    built = context.build(history, "s", token_counter=brain.count_tokens,
                          max_new_tokens=256,
                          budget=__import__("PRIMORA.prime.context",
                                            fromlist=["BudgetConfig"]
                                            ).BudgetConfig(
                              input_target=config.context_target,
                              input_ceiling=config.context_ceiling,
                              output_reserve=config.context_output_reserve,
                              total_budget=config.context_total))
    tokens = built.view.input_tokens
    print(f"  mechanism trimmed an 80-turn history to {tokens} tokens "
          f"(ceiling {config.context_ceiling})")
    print(f"  within ceiling: {tokens <= config.context_ceiling}")
    print(f"  diagnostics: {built.view.diagnostics[:3]}")
    trimmed_ok = generate(
        brain, tokenizer.apply_chat_template(
            built.view.messages, tokenize=False, add_generation_prompt=True),
        32, "trimmed context, real generate")
    results["trimmed"] = trimmed_ok
    try:
        os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "_probe.db"))
    except OSError:
        pass

    print("\n--- production brain.generate at the ceiling ---")
    messages = [{"role": "system",
                 "content": "You are Aurora. " + ("context note. " * 600)},
                {"role": "user", "content": q}]
    started = time.perf_counter()
    result = brain.generate(messages, max_output_tokens=32)
    elapsed = time.perf_counter() - started
    produced = len(tokenizer(result.content)["input_ids"]) if result.content else 0
    print(f"  brain.generate: success={result.success} elapsed={elapsed:.2f}s "
          f"produced={produced} ({produced / elapsed:.2f} tok/s incl prefill)")

    print("\n" + "=" * 72)
    ok = all(results.values())
    print(f"  all generations succeeded: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
