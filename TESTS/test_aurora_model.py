import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL = r"C:\PRIMORA PROJECT\RESOURCES\aurora\v0.1\model"
ADAPTER = r"C:\PRIMORA PROJECT\RESOURCES\aurora\v0.1\aurora_qlora"

print("=== AURORA STANDALONE TEST ===")
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM before: {free/1024**3:.2f} GB free / {total/1024**3:.2f} GB")

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True,
)

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.float16,
    attn_implementation="sdpa",
)

print("Base model loaded.")

print("Loading Aurora adapter...")
model = PeftModel.from_pretrained(
    base,
    ADAPTER,
    device_map="auto",
)

model.eval()

print("Aurora adapter loaded.")

if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM after load: {(total-free)/1024**3:.2f} GB used / {total/1024**3:.2f} GB")

messages = [
    {
        "role": "user",
        "content": "Hello Aurora. Reply in one short sentence so I know you are working."
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

print("\nGenerating...")
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
    )

reply = tokenizer.decode(
    output[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
).strip()

print("\n=== AURORA RESPONSE ===")
print(reply)
print("\n=== TEST COMPLETE ===")