"""Operators for an already-quantized MLX affine checkpoint.

Unlike the TorchAO-facing ops, these consume the original unsigned packed
words, scales and additive biases. They never infer zero points or requantize.
CPU bodies are small-tensor mathematical references, not an MLX numerical
oracle. Export uses the fake bodies; the delegate consumes Q/S/B directly.
"""
import torch
from torch import Tensor


def validate_affine(q, scales, biases, group_size, bits, rank):
    if (type(bits) is not int or bits not in (4, 8)
            or type(group_size) is not int or group_size not in (32, 64, 128)):
        raise ValueError("Native affine requires bits4/8 and group32/64/128")
    if q.dtype != torch.uint32 or q.ndim != rank or rank not in (2, 3):
        raise ValueError("Native affine Q must be rank2/3 uint32")
    if (scales.dtype not in (torch.float16, torch.float32) or biases.dtype != scales.dtype
            or scales.shape != biases.shape or scales.ndim != rank
            or q.shape[:-1] != scales.shape[:-1]):
        raise ValueError("Native affine S/B shapes or floating dtypes differ")
    width = q.shape[-1] * (32 // bits)
    if (width % group_size or scales.shape[-1] * group_size != width
            or any(type(d) is not int or d <= 0 for d in q.shape)):
        raise ValueError("Native affine packed/group dimensions differ")
    return width


def dequantize_reference(q, scales, biases, group_size, bits):
    """FP32 mathematical Q*S+B, then storage dtype; never used during lowering."""
    width = validate_affine(q, scales, biases, group_size, bits, q.ndim)
    shifts = torch.arange(32 // bits, device=q.device, dtype=torch.int64) * bits
    codes = ((q.to(torch.int64).unsqueeze(-1) >> shifts) & ((1 << bits) - 1))
    codes = codes.reshape(*q.shape[:-1], width).float()
    return (codes * scales.float().repeat_interleave(group_size, -1)
            + biases.float().repeat_interleave(group_size, -1)).to(scales.dtype)


@torch.library.custom_op("mlx::native_affine_qmm", mutates_args=())
def native_affine_qmm(x: Tensor, q: Tensor, scales: Tensor, biases: Tensor,
                      group_size: int, bits: int) -> Tensor:
    validate_affine(q, scales, biases, group_size, bits, 2)
    w = dequantize_reference(q, scales, biases, group_size, bits)
    return torch.matmul(x.float(), w.float().transpose(-1, -2)).to(x.dtype)


@native_affine_qmm.register_fake
def _qmm_fake(x, q, scales, biases, group_size, bits):
    width = validate_affine(q, scales, biases, group_size, bits, 2)
    if x.ndim < 1 or x.shape[-1] != width or x.dtype != scales.dtype:
        raise ValueError("Native affine QMM input width/dtype differs")
    return x.new_empty((*x.shape[:-1], q.shape[-2]))


@torch.library.custom_op("mlx::native_affine_gather_qmm", mutates_args=())
def native_affine_gather_qmm(x: Tensor, q: Tensor, scales: Tensor, biases: Tensor,
                             indices: Tensor, group_size: int, bits: int,
                             sorted_indices: bool = False) -> Tensor:
    validate_affine(q, scales, biases, group_size, bits, 3)
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expert indices must be integers")
    w = dequantize_reference(q, scales, biases, group_size, bits)[indices.long()]
    return torch.matmul(x.float(), w.float().transpose(-1, -2)).to(x.dtype)


@native_affine_gather_qmm.register_fake
def _gather_fake(x, q, scales, biases, indices, group_size, bits, sorted_indices=False):
    width = validate_affine(q, scales, biases, group_size, bits, 3)
    if (x.ndim < 2 or x.shape[-1] != width or x.dtype != scales.dtype
            or indices.dtype not in (torch.int32, torch.int64)
            or type(sorted_indices) is not bool):
        raise ValueError("Native affine expert input differs")
    batch = torch.broadcast_shapes(x.shape[:-2], indices.shape)
    return x.new_empty((*batch, x.shape[-2], q.shape[-2]))


@torch.library.custom_op("mlx::native_affine_embedding", mutates_args=())
def native_affine_embedding(indices: Tensor, q: Tensor, scales: Tensor, biases: Tensor,
                            group_size: int, bits: int) -> Tensor:
    validate_affine(q, scales, biases, group_size, bits, 2)
    # Select packed rows first, so even the reference avoids a dense vocabulary.
    shape = indices.shape
    flat = indices.long().reshape(-1)
    selected = dequantize_reference(q[flat], scales[flat], biases[flat], group_size, bits)
    return selected.reshape(*shape, selected.shape[-1])


@native_affine_embedding.register_fake
def _embedding_fake(indices, q, scales, biases, group_size, bits):
    width = validate_affine(q, scales, biases, group_size, bits, 2)
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Embedding indices must be integers")
    return scales.new_empty((*indices.shape, width))


@torch.library.custom_op("mlx::native_topk_indices", mutates_args=())
def native_topk_indices(x: Tensor, k: int) -> Tensor:
    """Unordered top-k selection; equal-logit tie/order is backend-defined.

    The MLX lowering is precisely the official Granite argpartition/slice.
    Torch's reference verifies set membership only, not MLX partition order.
    """
    if type(k) is not int or not 0 < k <= x.shape[-1]:
        raise ValueError("Invalid top-k")
    return torch.topk(x, k, dim=-1, largest=True, sorted=False).indices.to(torch.int32)


@native_topk_indices.register_fake
def _topk_fake(x, k):
    if x.ndim < 1 or type(k) is not int or not 0 < k <= x.shape[-1]:
        raise ValueError("Invalid top-k")
    return torch.empty((*x.shape[:-1], k), dtype=torch.int32, device=x.device)


@torch.library.custom_op("mlx::native_take_along_last", mutates_args=())
def native_take_along_last(x: Tensor, indices: Tensor) -> Tensor:
    return torch.take_along_dim(x, indices.long(), dim=-1)


@native_take_along_last.register_fake
def _take_fake(x, indices):
    if (x.ndim < 1 or x.shape[:-1] != indices.shape[:-1]
            or indices.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Native take requires matching leading dimensions and integer indices")
    return x.new_empty(indices.shape)


@torch.library.custom_op("mlx::native_olmoe_softmax", mutates_args=())
def native_olmoe_softmax(x: Tensor) -> Tensor:
    """Original OLMoE precise full-expert softmax, output in storage dtype."""
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


@native_olmoe_softmax.register_fake
def _olmoe_softmax_fake(x):
    if x.ndim != 3 or x.dtype != torch.float16:
        raise ValueError("OLMoE router requires B1/T/E FP16 logits")
    return x.new_empty(x.shape)


@torch.library.custom_op("mlx::native_olmoe_topk_indices", mutates_args=())
def native_olmoe_topk_indices(x: Tensor, k: int) -> Tensor:
    """Reference set only; lowering retains MLX's negative-probability partition order."""
    if type(k) is not int or not 0 < k <= x.shape[-1]:
        raise ValueError("Invalid OLMoE top-k")
    return torch.argsort(-x, dim=-1, stable=True)[..., :k].to(torch.int32)


@native_olmoe_topk_indices.register_fake
def _olmoe_topk_fake(x, k):
    if x.ndim != 3 or x.dtype != torch.float16 or type(k) is not int or not 0 < k <= x.shape[-1]:
        raise ValueError("Invalid OLMoE probability/top-k ABI")
    return torch.empty((*x.shape[:-1], k), dtype=torch.int32, device=x.device)
