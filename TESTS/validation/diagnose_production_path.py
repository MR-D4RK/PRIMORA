"""The production path measured alone.

The first diagnostic had to answer two questions at once and could not:
loading a second model while the benchmark's copy was resident ran the GPU
out of memory, so the real Aurora runtime was never measured.

This loads only Aurora, the way production does, and measures it properly:
short and long generations at the real context size, with prefill separated
from decode. The direct-transformers baseline from the other script
(11.6 tok/s base, 17.3 tok/s decode at 924 prompt tokens) is the comparison.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, r"C:\PRIMORA PROJECT")

import torch  # noqa: E402

from PRIMORA.aurora.brain import AuroraBrain  # noqa: E402
from PRIMORA.aurora.configuration import AuroraConfig  # noqa: E402

try:
    import psutil
except ImportError:
    psutil = None


def vram():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3, total / 1024**3


def devices(model):
    counts = {}
    for _, param in model.named_parameters():
        key = str(param.device)
        counts[key] = counts.get(key, 0) + 1
    return counts


def dtypes(model):
    counts = {}
    for _, param in model.named_parameters():
        key = str(param.dtype)
        counts[key] = counts.get(key, 0) + 1
    return counts


def run(model, tokenizer, prompt_text, max_new_tokens, label):
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_tokens = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    produced = out.shape[1] - prompt_tokens
    print(f"  {label}: prompt={prompt_tokens} produced={produced} "
          f"time={elapsed:.2f}s -> {produced / elapsed:.2f} tok/s")
    return {"prompt": prompt_tokens, "produced": produced,
            "elapsed": elapsed, "tps": produced / elapsed}


def main():
    config = AuroraConfig.from_env()
    print("=" * 72)
    print("PRODUCTION AURORA PATH, MEASURED ALONE")
    print("=" * 72)
    used, total = vram()
    print(f"  vram before: {used:.2f} / {total:.2f} GB")

    brain = AuroraBrain(config)
    started = time.perf_counter()
    problem = brain.warmup()
    load_time = time.perf_counter() - started
    print(f"  warmup (load): {load_time:.1f}s  {problem or 'ok'}")
    if problem:
        return 1

    model = brain._model
    used, total = vram()
    print(f"  vram after load: {used:.2f} / {total:.2f} GB "
          f"({used / total * 100:.0f}% full)")
    print(f"  param devices: {devices(model)}")
    print(f"  param dtypes: {dtypes(model)}")
    quant = getattr(model.config, "quantization_config", None)
    print(f"  quantization: {getattr(quant, 'quant_method', None)}")
    print(f"  attn impl: {model.config._attn_implementation}")
    print(f"  brain max_context_tokens: {brain._max_context_tokens}")

    tokenizer = brain._tokenizer
    short = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are Aurora."},
         {"role": "user", "content": "Reply in one short sentence."}],
        tokenize=False, add_generation_prompt=True)
    long_prompt = tokenizer.apply_chat_template(
        [{"role": "system",
          "content": "You are Aurora. " + ("context note. " * 300)},
         {"role": "user", "content": "Reply in one short sentence."}],
        tokenize=False, add_generation_prompt=True)

    print("\n--- production generate() settings ---")
    print(f"  do_sample={config.temperature > 0}  temperature="
          f"{config.temperature}  top_p={config.top_p}")
    print(f"  repetition_penalty={config.repetition_penalty}")

    print("\n--- decode rate, greedy, long context ---")
    pre = run(model, tokenizer, long_prompt, 1, "prefill only (1 token out)")
    a = run(model, tokenizer, long_prompt, 129, "129 tokens out")
    decode_s = a["elapsed"] - pre["elapsed"]
    decode_n = a["produced"] - pre["produced"]
    print(f"  => decode {decode_n} tokens in {decode_s:.2f}s = "
          f"{decode_n / decode_s:.2f} tok/s")

    print("\n--- the real production call, via brain.generate ---")
    for tokens in (16, 64):
        messages = [
            {"role": "system",
             "content": "You are Aurora. " + ("context note. " * 300)},
            {"role": "user", "content": "Reply in one short sentence."},
        ]
        started = time.perf_counter()
        result = brain.generate(messages, max_output_tokens=tokens)
        elapsed = time.perf_counter() - started
        produced = len(tokenizer(result.content)["input_ids"]) if result.content else 0
        print(f"  budget={tokens:3d} elapsed={elapsed:.2f}s "
              f"produced={produced} -> "
              f"{produced / elapsed if elapsed else 0:.2f} tok/s (incl. prefill)")

    used, total = vram()
    print(f"\n  vram at end: {used:.2f} / {total:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
