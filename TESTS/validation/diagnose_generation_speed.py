"""Diagnose Aurora's generation throughput. Measurement only.

Nothing is modified. The point is to find out *where* the time goes, so this
reports evidence rather than a guess:

  * which device every parameter actually lives on, and whether accelerate
    offloaded anything to CPU or disk
  * whether the 4-bit quantisation is genuinely active
  * prefill cost separated from decode cost
  * the cost of the QLoRA adapter itself, measured by generating with and
    without it from the identical weights
  * CPU utilisation and VRAM while a token is being produced
  * whether the full Aurora runtime costs anything over raw transformers

The last comparison matters: if raw transformers and Aurora are the same
speed, the runtime is innocent and the model stack is responsible.
"""

from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, r"C:\PRIMORA PROJECT")

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel  # noqa: E402

from PRIMORA.aurora.configuration import AuroraConfig  # noqa: E402

try:
    import psutil
except ImportError:
    psutil = None

MODEL = r"C:\PRIMORA PROJECT\RESOURCES\aurora\v0.1\model"
ADAPTER = r"C:\PRIMORA PROJECT\RESOURCES\aurora\v0.1\aurora_qlora"


def vram():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3, total / 1024**3


def device_histogram(model):
    counts = {}
    for _, param in model.named_parameters():
        device = str(param.device)
        counts[device] = counts.get(device, 0) + 1
    return counts


def offload_report(model):
    mapping = getattr(model, "hf_device_map", None) or {}
    buckets = {"cuda:0": 0, "cpu": 0, "disk": 0, "other": 0}
    for value in mapping.values():
        if isinstance(value, (list, tuple)):
            for v in value:
                buckets[_bucket(v, buckets)]
        else:
            buckets[_bucket(value, buckets)]
    return buckets, len(mapping)


def _bucket(value, buckets):
    text = str(value)
    if "disk" in text:
        buckets["disk"] += 1
    elif "cpu" in text:
        buckets["cpu"] += 1
    elif "cuda:0" in text or "cuda" in text:
        if "cuda:0" in text:
            buckets["cuda:0"] += 1
        else:
            buckets["other"] += 1
    else:
        buckets["other"] += 1
    return "other"


class CpuSampler(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self.running = True
        self.process = psutil.Process(os.getpid()) if psutil else None

    def run(self):
        while self.running:
            if self.process is not None:
                try:
                    self.samples.append(self.process.cpu_percent(
                        interval=None))
                except Exception:
                    pass
            time.sleep(self.interval)

    def report(self):
        if not self.samples:
            return "unavailable"
        ordered = sorted(self.samples)
        median = ordered[len(ordered) // 2]
        return f"median {median:.0f}% peak {max(ordered):.0f}% n={len(self.samples)}"


def timed_generate(model, tokenizer, prompt_text, max_new_tokens, sample=False):
    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_tokens = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    before = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=0.75 if sample else None,
            top_p=0.9 if sample else None,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - before
    produced = out.shape[1] - prompt_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "produced": produced,
        "elapsed": elapsed,
        "tok_per_s": produced / elapsed if elapsed else 0.0,
        "text": tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True),
    }


def main():
    config = AuroraConfig.from_env()
    print("=" * 72)
    print("AURORA GENERATION THROUGHPUT DIAGNOSIS")
    print("=" * 72)
    print(f"  torch {torch.__version__}  cuda {torch.version.cuda}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu: {torch.cuda.get_device_name(0)}")
    used, total = vram()
    print(f"  vram before load: {used:.2f} / {total:.2f} GB")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("\nloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    print("loading base model in 4-bit (load_in_4bit=True)...")
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    print(f"  base load: {time.perf_counter() - started:.1f}s")
    base.eval()

    used, total = vram()
    print(f"  vram after base: {used:.2f} / {total:.2f} GB")
    print(f"  param devices: {device_histogram(base)}")
    buckets, mapped = offload_report(base)
    print(f"  hf_device_map entries: {mapped}")
    print(f"  offload buckets: {buckets}")
    quant = getattr(base.config, "quantization_config", None)
    print(f"  quantization_config present: {quant is not None}")
    if quant is not None:
        print(f"  quant method: {getattr(quant, 'quant_method', '?')}")
    dtypes = {}
    for _, p in base.named_parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + 1
    print(f"  param dtypes: {dtypes}")
    print(f"  attn impl: {base.config._attn_implementation}")

    messages = [
        {"role": "system", "content": "You are Aurora."},
        {"role": "user", "content": "Reply in one short sentence."},
    ]
    short_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    long_messages = [
        {"role": "system", "content": "You are Aurora. " + ("context note. " * 300)},
        {"role": "user", "content": "Reply in one short sentence."},
    ]
    long_prompt = tokenizer.apply_chat_template(
        long_messages, tokenize=False, add_generation_prompt=True)

    print("\n--- A. base model, no adapter, short prompt ---")
    r1 = timed_generate(base, tokenizer, short_prompt, 32, sample=False)
    print(f"  prompt={r1['prompt_tokens']} produced={r1['produced']} "
          f"time={r1['elapsed']:.2f}s  {r1['tok_per_s']:.2f} tok/s")

    print("\n--- B. prefill vs decode, base model ---")
    pre = timed_generate(base, tokenizer, long_prompt, 1, sample=False)
    dec = timed_generate(base, tokenizer, long_prompt, 33, sample=False)
    decode_only = dec["elapsed"] - pre["elapsed"]
    decode_tokens = dec["produced"] - pre["produced"]
    print(f"  prefill ({pre['prompt_tokens']} prompt tokens, 1 token out): "
          f"{pre['elapsed']:.2f}s")
    print(f"  33 tokens out: {dec['elapsed']:.2f}s")
    print(f"  implied decode: {decode_tokens} tokens in "
          f"{decode_only:.2f}s = "
          f"{decode_tokens / decode_only if decode_only else 0:.2f} tok/s")

    print("\n--- C. load QLoRA adapter ---")
    started = time.perf_counter()
    model = PeftModel.from_pretrained(base, ADAPTER, device_map="auto")
    print(f"  adapter load: {time.perf_counter() - started:.1f}s")
    model.eval()
    print(f"  param devices with adapter: {device_histogram(model)}")
    buckets, _ = offload_report(model)
    print(f"  offload buckets with adapter: {buckets}")
    used, total = vram()
    print(f"  vram after adapter: {used:.2f} / {total:.2f} GB")

    print("\n--- D. with adapter, short prompt ---")
    r2 = timed_generate(model, tokenizer, short_prompt, 32, sample=False)
    print(f"  prompt={r2['prompt_tokens']} produced={r2['produced']} "
          f"time={r2['elapsed']:.2f}s  {r2['tok_per_s']:.2f} tok/s")
    print(f"  adapter cost: {r1['tok_per_s'] - r2['tok_per_s']:+.2f} tok/s")

    print("\n--- E. with adapter, long context, sampled (as Aurora runs) ---")
    sampler = CpuSampler()
    sampler.start()
    r3 = timed_generate(model, tokenizer, long_prompt, 64, sample=True)
    sampler.running = False
    sampler.join(timeout=1.0)
    used, total = vram()
    print(f"  prompt={r3['prompt_tokens']} produced={r3['produced']} "
          f"time={r3['elapsed']:.2f}s  {r3['tok_per_s']:.2f} tok/s")
    print(f"  cpu during generation: {sampler.report()}")
    print(f"  vram during/after: {used:.2f} / {total:.2f} GB")

    print("\n--- F. the real Aurora runtime, same model+adapter ---")
    from PRIMORA.aurora.brain import AuroraBrain

    brain = AuroraBrain(config)
    started = time.perf_counter()
    problem = brain.warmup()
    print(f"  warmup: {time.perf_counter() - started:.1f}s  {problem or 'ok'}")
    if problem:
        return
    msgs = [
        {"role": "system", "content": "You are Aurora. " + ("context note. " * 300)},
        {"role": "user", "content": "Reply in one short sentence."},
    ]
    started = time.perf_counter()
    result = brain.generate(msgs, max_output_tokens=64)
    elapsed = time.perf_counter() - started
    produced = len(brain._tokenizer(result.content)["input_ids"]) if result.content else 0
    print(f"  elapsed {elapsed:.2f}s  produced~{produced} tokens  "
          f"{produced / elapsed if elapsed else 0:.2f} tok/s (incl. prefill)")
    print(f"  text: {result.content[:100]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
