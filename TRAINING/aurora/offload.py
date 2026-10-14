"""CPU offload for the two frozen bf16 tensors that dominate VRAM.

On an 8 GiB card the 4-bit base weighs 5,835 MiB, of which ``embed_tokens`` and
``lm_head`` are 1,187 MiB each - 41% of the total - and neither is quantised,
trainable, or otherwise touched by training. They are pure memory.

Both are moved to CPU and streamed back one chunk at a time.

``embed_tokens``
    Only the rows named by ``input_ids`` are needed, so the gather happens on
    CPU against the CPU copy and just the gathered rows are sent over
    (~11 MiB for a 1,440-token sequence, versus 1,187 MiB resident). This is a
    pure gather: no arithmetic, so it is exact.

``lm_head``
    Chunked over the *vocab* dimension. That axis is the output dimension, so
    each output logit is still one dot product over the full K=hidden_size
    reduction; chunking does not split the reduction and therefore does not
    change the arithmetic. Only one chunk (~128 MiB) is resident on GPU at a
    time instead of the whole 1,187 MiB matrix.

    The backward pass is a custom Function so the chunk resident during forward
    is not kept alive for backward. ``grad_x = grad_logits @ W`` is accumulated
    chunk by chunk, which does reorder the reduction over vocab, so gradients
    are equal up to floating-point associativity rather than bit-identical.
    That difference is measured and asserted in the tests.

Nothing here changes the model, the data, or the training semantics: both
tensors stay bf16, unquantised, frozen, and absent from the optimizer.
"""

from __future__ import annotations

import torch

__all__ = [
    "CpuEmbedding",
    "OffloadError",
    "StreamedLinear",
    "offload_frozen_head",
]


class OffloadError(RuntimeError):
    """Raised when a tensor cannot be offloaded safely."""


class CpuEmbedding(torch.nn.Module):
    """``nn.Embedding`` whose weight lives on CPU.

    Accepts and returns GPU tensors so it is a drop-in for the module it
    replaces; callers never learn that the weight moved.
    """

    def __init__(self, weight: torch.Tensor, padding_idx=None):
        super().__init__()
        if weight.device.type != "cpu":
            raise OffloadError("CpuEmbedding expects an already-CPU weight")
        # persistent=False keeps it out of state_dict, so saving or loading the
        # model still refers to the original on-disk weights.
        self.register_buffer("weight", weight, persistent=False)
        self.padding_idx = padding_idx
        self.num_embeddings, self.embedding_dim = weight.shape

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def _apply(self, fn, recurse: bool = True):
        """Refuse to be moved to an accelerator.

        ``nn.Module._apply`` is what ``.to()``/``.cuda()`` calls, and
        ``accelerator.prepare()`` moves the whole model to the device. Without
        this override the CPU weight is silently swept back onto the GPU, which
        reinstates the 1,187 MiB this class exists to save - the smoke run OOMed
        for exactly this reason.

        ``fn`` is still applied to every other buffer and parameter, so the
        module keeps behaving normally for dtype changes and for the rest of the
        module tree.
        """
        skip = self.weight
        try:
            self.weight = torch.empty(0, dtype=skip.dtype)
            return super()._apply(fn, recurse=recurse)
        finally:
            self.weight = skip

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Deliberately blocking (no non_blocking=True) on both hops.
        #
        # `rows` is a freshly allocated temporary. With non_blocking=True the
        # device copy can still be in flight when Python drops the last
        # reference, and the allocator is then free to hand that block to
        # something else - which produced silently wrong embeddings (every
        # position decoding to the same row). These transfers are ~11 KB and
        # ~11 MiB, so blocking costs nothing that matters.
        ids = input_ids.to(self.weight.device)
        rows = torch.nn.functional.embedding(ids, self.weight, self.padding_idx)
        return rows.to(input_ids.device)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}, cpu_offloaded=True"


class _StreamedLinearFunction(torch.autograd.Function):
    """``x @ W.T`` where ``W`` stays on CPU and is streamed in vocab chunks."""

    @staticmethod
    def forward(ctx, x, weight_cpu, chunk_rows):
        vocab = weight_cpu.shape[0]
        target = x.device
        # When both operands already share a device there is nothing to stream;
        # this is also what keeps the CPU unit tests fast.
        if weight_cpu.device == target:
            ctx.streamed = False
            ctx.save_for_backward(x, weight_cpu)
            return torch.nn.functional.linear(x, weight_cpu)

        ctx.streamed = True
        ctx.chunk_rows = chunk_rows
        ctx.target = target
        out = x.new_empty((x.shape[0], vocab))
        with torch.no_grad():
            for start in range(0, vocab, chunk_rows):
                stop = min(start + chunk_rows, vocab)
                chunk = weight_cpu[start:stop].to(target, non_blocking=True)
                out[:, start:stop] = x @ chunk.T
                del chunk
        # x is saved so backward can reuse it; weight_cpu is saved by reference
        # and stays on CPU, which is the entire point.
        ctx.save_for_backward(x, weight_cpu)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight_cpu = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None

        grad_output = grad_output.contiguous()
        if not ctx.streamed:
            # grad_x = grad_y @ W, not F.linear(grad_y, W) which would be
            # grad_y @ W.T. W is (out_features, in_features).
            return grad_output @ weight_cpu, None, None

        vocab = weight_cpu.shape[0]
        chunk_rows = ctx.chunk_rows
        target = ctx.target
        grad_x = torch.zeros_like(x)
        with torch.no_grad():
            for start in range(0, vocab, chunk_rows):
                stop = min(start + chunk_rows, vocab)
                chunk = weight_cpu[start:stop].to(target, non_blocking=True)
                grad_x += grad_output[:, start:stop] @ chunk
                del chunk
        return grad_x, None, None


class StreamedLinear(torch.nn.Module):
    """``nn.Linear`` whose frozen weight lives on CPU and is streamed per chunk.

    Only the vocab axis is chunked, so each output element is produced by the
    same full-length reduction the GPU would have used.
    """

    def __init__(self, weight: torch.Tensor, chunk_rows: int = 16384):
        super().__init__()
        if weight.device.type != "cpu":
            raise OffloadError("StreamedLinear expects an already-CPU weight")
        if weight.dim() != 2:
            raise OffloadError(f"expected a 2D weight, got {tuple(weight.shape)}")
        self.register_buffer("weight", weight, persistent=False)
        self.out_features, self.in_features = weight.shape
        self.chunk_rows = chunk_rows
        self._transferred_bytes = 0

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def reset_transfer_counter(self) -> None:
        self._transferred_bytes = 0

    def _apply(self, fn, recurse: bool = True):
        """Refuse to be moved to an accelerator; see CpuEmbedding._apply.

        Without this, ``accelerator.prepare()`` puts the head back on the GPU
        and the offload silently does nothing.
        """
        skip = self.weight
        try:
            self.weight = torch.empty(0, dtype=skip.dtype)
            return super()._apply(fn, recurse=recurse)
        finally:
            self.weight = skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _StreamedLinearFunction.apply(x, self.weight, self.chunk_rows)
        if x.device != self.weight.device:
            # Counted for the overhead report: forward and backward each stream
            # the whole matrix once.
            self._transferred_bytes += 2 * self.weight.numel() * self.weight.element_size()
        return out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias=False, cpu_offloaded=True, chunk_rows={self.chunk_rows}")


def offload_frozen_head(model, chunk_rows: int = 16384) -> list[str]:
    """Move ``embed_tokens`` and ``lm_head`` to CPU, streaming them per call.

    Returns notes for the setup log. Raises :class:`OffloadError` rather than
    degrading silently if a tensor is not in the expected shape or dtype.
    """
    from aurora.qlora import resolve_causal_lm

    causal_lm = resolve_causal_lm(model)
    backbone = getattr(causal_lm, "model", None) if causal_lm is not None else None
    if backbone is None or not hasattr(backbone, "embed_tokens"):
        raise OffloadError(
            "could not locate embed_tokens; expected a bare CausalLM or a "
            "PeftModelForCausalLM")

    embedding = backbone.embed_tokens
    head = getattr(causal_lm, "lm_head", None)
    if head is None or not hasattr(head, "weight"):
        raise OffloadError("could not locate lm_head")

    # Validate everything before mutating, so a rejected model is left intact.
    if not hasattr(embedding, "weight") or not hasattr(head, "weight"):
        raise OffloadError("embed_tokens and lm_head must both expose .weight")
    already = [n for n, t in (("embed_tokens", embedding), ("lm_head", head))
               if not t.weight.is_cuda]
    if already:
        raise OffloadError(
            f"{' and '.join(already)} already on CPU; refusing to double-wrap")
    if getattr(head, "bias", None) is not None:
        raise OffloadError("lm_head has a bias; only bias-free is supported")
    if chunk_rows < 1:
        raise OffloadError(f"chunk_rows must be >= 1, got {chunk_rows}")

    notes = []
    freed = 0

    weight = embedding.weight.data.to("cpu")
    freed += weight.numel() * weight.element_size()
    backbone.embed_tokens = CpuEmbedding(weight, padding_idx=embedding.padding_idx)
    notes.append(
        f"embed_tokens offloaded to CPU ({weight.shape[0]}x{weight.shape[1]} "
        f"{weight.dtype}), gathered row-wise per forward")

    weight = head.weight.data.to("cpu")
    freed += weight.numel() * weight.element_size()
    causal_lm.lm_head = StreamedLinear(weight, chunk_rows=chunk_rows)
    notes.append(
        f"lm_head offloaded to CPU ({weight.shape[0]}x{weight.shape[1]} "
        f"{weight.dtype}), streamed in {chunk_rows}-row chunks")

    notes.append(f"freed {freed / 1024 ** 3:.2f} GiB of VRAM by CPU offload")
    return notes