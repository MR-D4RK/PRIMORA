"""Tests for CPU offload of the frozen embed_tokens and lm_head.

Two claims have to hold, and they are different in kind:

  * exactness of the forward - chunking the head over the vocab axis is an
    output-axis split, so every logit is still one dot product over the full
    K=hidden_size reduction. Asserted with ``torch.equal``, not a tolerance.
  * safety - both tensors stay bf16, unquantised, frozen, receive no gradient,
    and stay out of the optimizer.

The 8B base is never loaded. A tiny genuine Qwen2ForCausalLM stands in, and the
CPU path (both operands already co-resident) is exercised directly.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_ROOT = r"C:\PRIMORA PROJECT"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "TRAINING"))

from aurora.offload import (  # noqa: E402
    CpuEmbedding,
    OffloadError,
    StreamedLinear,
    offload_frozen_head,
)

VOCAB, HIDDEN = 512, 64


def _tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).to(torch.float32).eval()


# -- 1. forward exactness -------------------------------------------------

def test_cpu_embedding_gather_is_exact():
    weight = torch.randn(VOCAB, HIDDEN)
    ref = torch.nn.Embedding(VOCAB, HIDDEN, _weight=weight)
    off = CpuEmbedding(weight.clone())
    ids = torch.randint(0, VOCAB, (3, 17))
    assert torch.equal(ref(ids), off(ids))


def test_cpu_embedding_respects_padding_idx():
    """padding_idx only masks gradients; forward is still a plain gather."""
    weight = torch.randn(VOCAB, HIDDEN)
    off = CpuEmbedding(weight.clone(), padding_idx=0)
    ids = torch.tensor([[0, 1, 2]])
    assert torch.equal(off(ids), weight[ids])
    assert off.padding_idx == 0


def test_cpu_embedding_matches_on_a_second_device():
    """The point of the module: CPU weight, caller-visible device output."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    weight = torch.randn(VOCAB, HIDDEN)
    off = CpuEmbedding(weight.clone())
    ids = torch.randint(0, VOCAB, (2, 9), device="cuda")
    out = off(ids)
    assert out.device.type == "cuda"
    assert torch.equal(out.cpu(), weight[ids.cpu()])
    # Guards the async-copy bug this module originally had: a temporary source
    # under non_blocking=True made every position decode to the same row.
    assert not torch.equal(out[0, 0], out[0, 1])


def test_streamed_linear_is_bit_identical_to_the_resident_linear():
    """Forward exactness. Chunking the vocab axis must not perturb a single bit.

    Chunked on purpose at an awkward size (and a final partial chunk) so the
    loop boundaries are actually exercised.
    """
    weight = torch.randn(VOCAB, HIDDEN)
    ref = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    ref.weight.data = weight.clone()
    x = torch.randn(7, HIDDEN)

    full = ref(x)
    for chunk in (1, 3, 64, VOCAB - 1, VOCAB, 4096):
        off = StreamedLinear(weight.clone(), chunk_rows=chunk)
        got = off(x)
        assert torch.equal(got, full), f"chunk_rows={chunk} changed the result"
        del off


def test_streamed_linear_matches_on_gpu_chunk_by_chunk():
    """Same claim on the real device, which is where streaming actually happens."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    x = torch.randn(36, HIDDEN, dtype=torch.bfloat16)

    # Compared against the plain functional form. nn.Linear's weight is a
    # Parameter that requires grad, so its output carries a grad_fn and
    # torch.equal against a grad-tracking tensor compares object state rather
    # than values.
    full = torch.nn.functional.linear(x.cuda(), weight.cuda()).cpu()
    assert full.grad_fn is None
    for chunk in (7, 128, 4096):
        off = StreamedLinear(weight.clone(), chunk_rows=chunk)
        got = off(x.cuda()).cpu()
        assert torch.equal(got, full), f"chunk_rows={chunk} changed the result"
        del off


def test_streamed_linear_copresident_path_is_the_plain_matmul():
    """With both operands on one device there must be no streaming at all."""
    weight = torch.randn(VOCAB, HIDDEN)
    off = StreamedLinear(weight.clone(), chunk_rows=8)
    x = torch.randn(4, HIDDEN)
    assert torch.equal(off(x), torch.nn.functional.linear(x, weight))
    assert off._transferred_bytes == 0


# -- 2. gradients ---------------------------------------------------------

def test_gradients_match_the_resident_linear_to_float_precision():
    """Backward reorders the vocab reduction, so this is close, not equal.

    The tolerance is float32-realistic; a real bug would miss by orders of
    magnitude, not by the last few bits.
    """
    weight = torch.randn(VOCAB, HIDDEN)
    x0 = torch.randn(11, HIDDEN)
    grad_out = torch.randn(11, VOCAB)

    xr = x0.clone().requires_grad_(True)
    torch.nn.functional.linear(xr, weight).backward(grad_out)
    ref_grad = xr.grad.clone()

    for chunk in (1, 5, 64, 4096):
        xo = x0.clone().requires_grad_(True)
        StreamedLinear(weight.clone(), chunk_rows=chunk)(xo).backward(grad_out)
        assert torch.allclose(xo.grad, ref_grad, atol=1e-5, rtol=1e-5), chunk


def test_backward_does_not_keep_the_gpu_chunk_alive():
    """The whole point: forward's chunk must not be saved for backward.

    If the chunk were kept alive, the custom Function would be pointless and
    peak VRAM would be no better than leaving the head resident.

    Asserted on *peak* allocated across forward+backward, using a head large
    enough that the signal is unambiguous: 32 MiB resident versus at most one
    4 MiB chunk in flight.

    torch.cuda.memory_allocated() is not used for the live-tensor count: the
    caching allocator keeps freed blocks, so it over-reports long after a
    tensor dies. gc.get_objects() is the honest measure of what is retained.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    import gc

    V, H, chunk = 4096, 4096, 512          # head 32 MiB, chunk 4 MiB
    weight = torch.randn(V, H, dtype=torch.bfloat16)
    off = StreamedLinear(weight.clone(), chunk_rows=chunk)
    x = torch.randn(8, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    full_head = weight.numel() * weight.element_size()
    chunk_bytes = chunk * H * 2

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    out = off(x)
    grad_out = torch.randn(8, V, dtype=torch.bfloat16, device="cuda")
    out.backward(grad_out)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() - baseline
    assert peak < full_head, (
        f"peak {peak} bytes reached the resident-head size {full_head}")

    retained = sum(t.numel() * t.element_size() for t in gc.get_objects()
                   if isinstance(t, torch.Tensor) and t.is_cuda)
    assert retained < full_head, (
        f"{retained} bytes of CUDA tensors still alive after backward; a "
        f"resident head is {full_head} and one chunk is {chunk_bytes}")


def test_no_gradient_reaches_the_offloaded_weights():
    weight = torch.randn(VOCAB, HIDDEN)
    off = StreamedLinear(weight.clone(), chunk_rows=64)
    assert off.weight.requires_grad is False
    assert all(not p.requires_grad for p in off.parameters())
    off(torch.randn(3, HIDDEN, requires_grad=True)).sum().backward()
    assert off.weight.grad is None


# -- 3. dtype, quantisation, state_dict -----------------------------------

def test_offloaded_weights_stay_bf16_and_unquantised():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from aurora.config import TrainerConfig
    from aurora.qlora import load_base_model, load_tokenizer

    cfg = TrainerConfig()
    tok = load_tokenizer(cfg.base_model)
    model = load_base_model(cfg, tok)
    emb = model.model.embed_tokens.weight
    head = model.lm_head.weight
    assert emb.dtype == torch.bfloat16 and head.dtype == torch.bfloat16
    # The whole premise: these two are not 4-bit.
    assert type(emb).__name__ == "Parameter"
    assert type(head).__name__ == "Parameter"

    offload_frozen_head(model, chunk_rows=16384)
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.lm_head.weight.dtype == torch.bfloat16
    assert model.model.embed_tokens.weight.device.type == "cpu"
    assert model.lm_head.weight.device.type == "cpu"
    assert not hasattr(model.lm_head.weight, "quant_state")


def test_offload_keeps_the_exact_weight_values():
    """Not a new tensor: the same bf16 numbers, moved."""
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    original = weight.clone()
    off = StreamedLinear(weight)
    assert torch.equal(off.weight, original)


def test_offloaded_buffers_stay_out_of_the_state_dict():
    """So the model still refers to the on-disk weights, not a CPU copy."""
    model = _tiny_qwen()
    before = set(model.state_dict().keys())
    assert "model.embed_tokens.weight" in before
    assert "lm_head.weight" in before

    model.model.embed_tokens = CpuEmbedding(
        model.model.embed_tokens.weight.data.clone())
    model.lm_head = StreamedLinear(model.lm_head.weight.data.clone())

    after = set(model.state_dict().keys())
    assert "model.embed_tokens.weight" not in after
    assert "lm_head.weight" not in after
    # The 4-bit decoder blocks are untouched and still saved.
    assert "model.layers.0.self_attn.q_proj.weight" in after


@pytest.mark.parametrize("module_name", ["embed", "head"])
def test_offloaded_module_refuses_to_be_moved_to_gpu(module_name):
    """accelerator.prepare() calls .to(device); the offload must survive it.

    This is the bug that OOMed the 8-step smoke run. Both weights are buffers,
    so nn.Module._apply would otherwise move them straight back onto the GPU
    and silently undo the whole offload.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    weight = torch.randn(VOCAB, HIDDEN)
    if module_name == "embed":
        mod = CpuEmbedding(weight.clone())
    else:
        mod = StreamedLinear(weight.clone(), chunk_rows=64)

    mod.to("cuda")
    assert mod.weight.device.type == "cpu", \
        "the CPU weight was moved to the GPU by .to()"
    # And it must still work afterwards, from the original device.
    ids = torch.randint(0, VOCAB, (1, 4), device="cuda")
    out = mod(ids) if module_name == "embed" else mod(torch.randn(1, HIDDEN, device="cuda"))
    assert out.device.type == "cuda"


def test_offloaded_module_refuses_dtype_changes_too():
    """_apply leaves the weight untouched entirely, dtype included.

    Deliberate. The weight is frozen bf16 that must stay numerically identical,
    so silently re-typing it would be worse than ignoring the request. The real
    path never asks: load_base_model builds the model in bfloat16 before
    offload_frozen_head ever runs, so there is no dtype change to lose.
    """
    weight = torch.randn(VOCAB, HIDDEN)
    mod = StreamedLinear(weight.clone(), chunk_rows=8)
    before = mod.weight.clone()
    mod.to(torch.bfloat16)
    assert mod.weight.dtype == torch.float32, "weight dtype was changed"
    assert torch.equal(mod.weight, before)
    assert mod.weight.device.type == "cpu"
    # The rest of the module tree is unaffected, and the module still computes.
    assert mod(torch.randn(2, HIDDEN)).shape == (2, VOCAB)


def test_offload_rejects_a_wrapped_model():
    """Refuses rather than silently doing nothing."""
    class _Fake:
        pass
    with pytest.raises(OffloadError, match="embed_tokens"):
        offload_frozen_head(_Fake())


def test_offload_rejects_a_biased_head():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    model = _tiny_qwen().cuda()
    model.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=True).cuda()
    with pytest.raises(OffloadError, match="bias"):
        offload_frozen_head(model)


def test_offload_rejects_double_wrapping():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    model = _tiny_qwen().cuda()
    model.model.embed_tokens = CpuEmbedding(
        model.model.embed_tokens.weight.data.cpu())
    model.lm_head = StreamedLinear(model.lm_head.weight.data.cpu())
    with pytest.raises(OffloadError, match="already on CPU"):
        offload_frozen_head(model)


def test_offload_rejects_a_gpu_weight():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    with pytest.raises(OffloadError, match="already-CPU"):
        CpuEmbedding(torch.zeros(4, 4, device="cuda"))


def test_offload_rejects_a_non_2d_head_weight():
    with pytest.raises(OffloadError, match="2D"):
        StreamedLinear(torch.zeros(4))


# -- 4. the real training path -------------------------------------------

def test_config_offload_defaults_are_sane():
    from aurora.config import TrainerConfig
    cfg = TrainerConfig()
    assert cfg.cpu_offload_frozen_head is True
    assert cfg.offload_chunk_rows == 16384
    assert not cfg.validate()


def test_config_rejects_a_zero_chunk():
    from aurora.config import TrainerConfig
    assert TrainerConfig(offload_chunk_rows=0).validate()


def test_offloaded_model_still_trains_end_to_end_on_cpu():
    """A tiny model with both tensors offloaded must produce a real gradient."""
    model = _tiny_qwen()
    model.model.embed_tokens = CpuEmbedding(model.model.embed_tokens.weight.data.clone())
    model.lm_head = StreamedLinear(model.lm_head.weight.data.clone(), chunk_rows=64)
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad = False
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM"))
    model.train()

    ids = torch.randint(0, VOCAB, (1, 24))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    got = [n for n, p in model.named_parameters()
           if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    assert got, "no LoRA gradient with the head offloaded"
    assert all(p.grad.isfinite().all() for p in model.parameters()
               if p.grad is not None)


def test_supervised_loss_is_unaffected_by_offload():
    """The end-to-end claim: same loss value with and without offload."""
    from aurora.supervised_loss import compute_supervised_loss

    model_a = _tiny_qwen()
    ids = torch.randint(0, VOCAB, (2, 20))
    labels = torch.full((2, 20), -100, dtype=torch.long)
    labels[0, 8:14] = ids[0, 9:15]
    mask = torch.ones_like(ids)
    with torch.no_grad():
        resident = compute_supervised_loss(model_a, ids, mask, labels)

    model_b = _tiny_qwen()
    model_b.load_state_dict(model_a.state_dict())
    model_b.model.embed_tokens = CpuEmbedding(
        model_b.model.embed_tokens.weight.data.clone())
    model_b.lm_head = StreamedLinear(model_b.lm_head.weight.data.clone(), chunk_rows=64)
    with torch.no_grad():
        offloaded = compute_supervised_loss(model_b, ids, mask, labels)

    assert torch.equal(resident, offloaded), \
        f"resident={float(resident)} offloaded={float(offloaded)}"