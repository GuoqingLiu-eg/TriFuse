"""Byte-preserving lowering of native MLX affine Q/S/B custom operators."""
import torch
from executorch.backends.mlx import native_affine_ops as _registered  # noqa: F401
from executorch.backends.mlx.builder.op_registry import REGISTRY
from executorch.backends.mlx.builder.op_helpers import emit_quantized_gather
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    ArgPartitionNode, GatherQmmNode, IntOrVid, NegNode, QuantizedMatmulNode, SliceNode,
    SoftmaxNode, TakeAlongAxisNode,
)


def _constant_operands(P, n, offset, rank, group_size, bits):
    nodes = n.args[offset:offset + 3]
    if len(nodes) != 3 or any(getattr(node, "op", None) != "placeholder" for node in nodes):
        raise ValueError("Native affine Q/S/B must be direct immutable constants")
    tensors = [P.get_placeholder_target_and_tensor(node)[1] for node in nodes]
    _registered.validate_affine(*tensors, group_size, bits, rank)
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Native affine Q/S/B must be contiguous")
    # No lifted copy, unpacking, scale conversion, or zero-point reconstruction.
    return P.slot_map(list(nodes))


@REGISTRY.register(target=[torch.ops.mlx.native_affine_qmm.default])
def _qmm(P, n):
    x, _, _, _, group_size, bits = P.args(n)
    q, scales, biases = _constant_operands(P, n, 1, 2, group_size, bits)
    out = P.make_or_get_slot(n)
    P.emit(QuantizedMatmulNode(x=P.slot_to_tid(x), w=P.slot_to_tid(q),
        scales=P.slot_to_tid(scales), biases=P.slot_to_tid(biases), out=P.slot_to_tid(out),
        transpose=True, group_size=group_size, bits=bits, mode="affine"))
    return out


@REGISTRY.register(target=[torch.ops.mlx.native_affine_gather_qmm.default])
def _gather(P, n):
    args = P.args(n)
    x, _, _, _, indices, group_size, bits = args[:7]
    sorted_indices = args[7] if len(args) == 8 else False
    if type(sorted_indices) is not bool:
        raise ValueError("Native affine sorted_indices must be static bool")
    q, scales, biases = _constant_operands(P, n, 1, 3, group_size, bits)
    out = P.make_or_get_slot(n)
    P.emit(GatherQmmNode(x=P.slot_to_tid(x), w=P.slot_to_tid(q),
        scales=P.slot_to_tid(scales), biases=P.slot_to_tid(biases), out=P.slot_to_tid(out),
        rhs_indices=P.slot_to_tid(indices), transpose=True, sorted_indices=sorted_indices,
        group_size=group_size, bits=bits, mode="affine"))
    return out


@REGISTRY.register(target=[torch.ops.mlx.native_affine_embedding.default])
def _embedding(P, n):
    indices, _, _, _, group_size, bits = P.args(n)
    q, scales, biases = _constant_operands(P, n, 1, 2, group_size, bits)
    out = P.make_or_get_slot(n)
    emit_quantized_gather(P, out, indices, q, scales, biases,
        group_size=group_size, bits=bits, mode="affine", out_dtype=n.meta["val"].dtype)
    return out


@REGISTRY.register(target=[torch.ops.mlx.native_topk_indices.default])
def _topk(P, n):
    x, k = P.args(n)
    shape = n.args[0].meta["val"].shape
    if type(k) is not int or not 0 < k <= shape[-1]:
        raise ValueError("Invalid native top-k")
    _, tmp = P.make_tmp_slot()
    out = P.make_or_get_slot(n)
    P.emit(ArgPartitionNode(x=P.slot_to_tid(x), out=P.slot_to_tid(tmp),
        kth=IntOrVid.from_literal(-k), axis=-1))
    P.emit(SliceNode(x=P.slot_to_tid(tmp), out=P.slot_to_tid(out),
        axis=IntOrVid.from_literal(-1), start=IntOrVid.from_literal(-k),
        stop=IntOrVid.from_literal(shape[-1]), step=1))
    return out


@REGISTRY.register(target=[torch.ops.mlx.native_take_along_last.default])
def _take(P, n):
    x, indices = P.args(n)
    out = P.make_or_get_slot(n)
    P.emit(TakeAlongAxisNode(x=P.slot_to_tid(x), indices=P.slot_to_tid(indices), out=P.slot_to_tid(out), axis=-1))
    return out


@REGISTRY.register(target=[torch.ops.mlx.native_olmoe_softmax.default])
def _olmoe_softmax(P, n):
    x, = P.args(n)
    out = P.make_or_get_slot(n)
    P.emit(SoftmaxNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out), axis=-1, precise=True))
    return out


@REGISTRY.register(target=[torch.ops.mlx.native_olmoe_topk_indices.default])
def _olmoe_topk(P, n):
    x, k = P.args(n)
    shape = n.args[0].meta["val"].shape
    if type(k) is not int or not 0 < k <= shape[-1]:
        raise ValueError("Invalid OLMoE top-k")
    _, negative = P.make_tmp_slot()
    _, partition = P.make_tmp_slot()
    out = P.make_or_get_slot(n)
    P.emit(NegNode(x=P.slot_to_tid(x), out=P.slot_to_tid(negative)))
    P.emit(ArgPartitionNode(x=P.slot_to_tid(negative), out=P.slot_to_tid(partition),
        kth=IntOrVid.from_literal(k - 1), axis=-1))
    P.emit(SliceNode(x=P.slot_to_tid(partition), out=P.slot_to_tid(out),
        axis=IntOrVid.from_literal(-1), start=IntOrVid.from_literal(0),
        stop=IntOrVid.from_literal(k), step=1))
    return out
