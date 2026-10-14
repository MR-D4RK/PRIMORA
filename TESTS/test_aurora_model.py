"""Standalone Aurora adapter smoke test.

Loads the v0.1 base model + QLoRA adapter and generates one reply.
Heavy imports and weights loading are deferred into the test body so
pytest can import this file safely in any suite order; hardware limits
(CUDA, VRAM) become explicit skips instead of collection errors.
"""
import pytest

MODEL = r"C:\PRIMORA PROJECT\RESOURCES\aurora\v0.1\model"
ADAPTER = r"C:\PRIMORA PROJECT\RESOURCES\aurora\v0.1\aurora_qlora"

# 4-bit weights plus CUDA overhead. Lower than this the loader falls back to
# CPU/disk dispatch and raises instead of producing a valid GPU test.
_MIN_FREE_VRAM_BYTES = 6 * 1024**3


def test_aurora_model_generates():
    torch = pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("transformers", reason="transformers not installed")
    pytest.importorskip("peft", reason="peft not installed")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    free, total = torch.cuda.mem_get_info()
    if free < _MIN_FREE_VRAM_BYTES:
        pytest.skip(
            f"insufficient free VRAM: {free / 1024**3:.2f} GB free / "
            f"{total / 1024**3:.2f} GB total"
        )

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=True,
    )

    try:
        base = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
    except ValueError as e:
        # device_map="auto" refused to place every module on the GPU: the
        # free-VRAM check above was too optimistic (transient pressure).
        # Environmental, not a code failure.
        if "dispatched on the CPU" in str(e) or "dispatched on the disk" in str(e):
            pytest.skip(f"insufficient VRAM for full-GPU load: {e}")
        raise

    model = PeftModel.from_pretrained(
        base,
        ADAPTER,
        device_map="auto",
    )
    model.eval()

    messages = [
        {
            "role": "user",
            "content": "Hello Aurora. Reply in one short sentence so I know you are working.",
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
        )

    reply = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    assert reply, "Aurora generated an empty reply"


if __name__ == "__main__":
    test_aurora_model_generates()
