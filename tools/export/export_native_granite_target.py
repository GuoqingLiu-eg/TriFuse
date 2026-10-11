"""Export a local official MLX Granite-MoE affine pack without requantization.

The Torch graph is an explicit adapter of Apple's mlx_lm/models/granitemoe.py
and switch_layers.py (Apple Inc., MIT license), not a replacement checkpoint.
It preserves original Q/S/B and dense norm bytes, original scalar semantics,
FP32 router gates and expert reduction, full-history external KV, and the tied
quantized vocabulary head. No HF dense weights are needed by this exporter.
Export success is structural/packaging evidence, never model accuracy evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ENVIRONMENT = HERE.parent
FORMAT = "mlx_affine_uint32_v1"
REFERENCE = "https://github.com/ml-explore/mlx-lm/blob/v0.32.0/mlx_lm/models/granitemoe.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def configure_local_backend():
    """Use this checkout's Python backend with the installed ExecuTorch wheel."""
    import executorch.backends
    source = HERE.parents[1] / "trifuse/executorch/backends"
    existing = sys.modules.get("executorch.backends.mlx")
    require(existing is None or Path(existing.__file__).resolve().is_relative_to(source),
            "An installed older MLX backend was imported before the local adapter")
    if str(source) not in executorch.backends.__path__:
        executorch.backends.__path__.insert(0, str(source))


def configuration(raw, capacity):
    require(raw.get("model_type") == "granitemoe", "Need original Granite-MoE configuration")
    require(raw.get("hidden_act") == "silu" and raw.get("tie_word_embeddings") is True,
            "Only original SwiGLU and tied vocabulary head are supported")
    require(raw.get("attention_bias") is False and raw.get("attention_dropout", 0.0) == 0.0
            and raw.get("mlp_bias", False) is False and raw.get("sliding_window") in (None, 0),
            "Only bias-free, zero-dropout, full-history Granite-MoE is supported")
    require(raw.get("rope_scaling") is None and raw.get("rope_parameters") is None
            and raw.get("rope_traditional", False) is False
            and raw.get("partial_rotary_factor", 1.0) == 1.0,
            "Only original full default noninterleaved RoPE is supported")
    names = {"layers": "num_hidden_layers", "hidden_size": "hidden_size",
             "intermediate_size": "intermediate_size", "query_heads": "num_attention_heads",
             "kv_heads": "num_key_value_heads", "vocabulary": "vocab_size",
             "num_experts": "num_local_experts", "experts_per_token": "num_experts_per_tok"}
    native = {key: raw.get(source) for key, source in names.items()}
    require(all(type(v) is int and 0 < v <= 2**31 - 1 for v in native.values()), "Invalid integer geometry")
    original = raw.get("max_position_embeddings")
    require(type(original) is int and type(capacity) is int and 128 <= capacity <= original <= 2**31 - 1,
            "Runtime capacity must not extend original context")
    require(native["hidden_size"] % native["query_heads"] == 0
            and native["query_heads"] % native["kv_heads"] == 0
            and native["experts_per_token"] <= native["num_experts"], "Invalid GQA or expert geometry")
    native["head_dim"] = native["hidden_size"] // native["query_heads"]
    require(native["head_dim"] % 2 == 0 and raw.get("head_dim", native["head_dim"]) == native["head_dim"],
            "Invalid full rotary head dimension")
    for key in ("rms_norm_eps", "rope_theta", "embedding_multiplier", "residual_multiplier",
                "attention_multiplier", "logits_scaling"):
        value = raw.get(key)
        require(type(value) in (float, int) and math.isfinite(value) and value > 0,
                "Invalid original scalar " + key)
        native[key] = float(value)
    require(native["rope_theta"] > 1, "Invalid original RoPE base")
    return {"architecture": "granite_moe_v1", **native, "capacity": capacity,
            "quantization_layout": "affine_per_projection", "linear_quantization_bits": 0,
            "linear_quantization_group_size": 0, "tie_word_embeddings": True}


def expected_modules(native):
    h, middle, vocab, experts = (native[k] for k in ("hidden_size", "intermediate_size", "vocabulary", "num_experts"))
    q = native["query_heads"] * native["head_dim"]
    kv = native["kv_heads"] * native["head_dim"]
    result = {"model.embed_tokens": (vocab, h)}
    for layer in range(native["layers"]):
        prefix = f"model.layers.{layer}."
        result.update({prefix + "self_attn." + name + "_proj": shape
                       for name, shape in (("q", (q, h)), ("k", (kv, h)), ("v", (kv, h)), ("o", (h, q)))})
        result[prefix + "block_sparse_moe.router.layer"] = (experts, h)
        for name, shape in (("gate", (experts, middle, h)), ("up", (experts, middle, h)),
                            ("down", (experts, h, middle))):
            result[prefix + f"block_sparse_moe.switch_mlp.{name}_proj"] = shape
    return result


def quantization_contract(raw, native):
    config = raw.get("quantization", raw.get("quantization_config"))
    require(isinstance(config, dict), "Need an official MLX quantization map")
    if "quantization" in raw and "quantization_config" in raw:
        require(raw["quantization"] == raw["quantization_config"], "Quantization aliases disagree")
    modules = expected_modules(native)
    require(not (set(config) - set(modules) - {"bits", "group_size", "mode"}),
            "Unknown quantization override path or option")
    result = {}
    for name, shape in modules.items():
        override = config.get(name, {})
        require(isinstance(override, dict), "Every supported module must be affine quantized: " + name)
        require(not (set(override) - {"bits", "group_size", "mode"}), "Unknown per-module quantization option")
        merged = {key: override.get(key, config.get(key, "affine" if key == "mode" else None))
                  for key in ("bits", "group_size", "mode")}
        require(type(merged["bits"]) is int and merged["bits"] in (4, 8)
                and type(merged["group_size"]) is int and merged["group_size"] == 64
                and merged["mode"] == "affine" and shape[-1] % 64 == 0,
                "Need actual affine4/8 group64 per-module configuration: " + name)
        result[name] = merged
    return result


def tensor_record(tensor):
    data = tensor.detach().contiguous().view(__import__("torch").uint8).numpy().tobytes()
    return {"dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, data


def load_pack(root, native, quantization):
    import torch
    from safetensors import safe_open
    from executorch.backends.mlx.native_affine_ops import validate_affine
    files = sorted(root.glob("*.safetensors"))
    require(files, "No local official packed safetensors")
    tensors = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            for name in stream.keys():
                require(name not in tensors, "Duplicate checkpoint tensor " + name)
                tensors[name] = stream.get_tensor(name)
    shapes = expected_modules(native)
    expected = {f"{name}.{part}" for name in shapes for part in ("weight", "scales", "biases")}
    norm_names = {"model.norm.weight"} | {f"model.layers.{i}.{kind}.weight"
        for i in range(native["layers"]) for kind in ("input_layernorm", "post_attention_layernorm")}
    expected |= norm_names
    require(set(tensors) == expected, "Packed checkpoint has missing/extra tensors: "
            + repr(sorted(set(tensors) ^ expected)))
    for name, shape in shapes.items():
        q, s, b = (tensors[name + "." + part] for part in ("weight", "scales", "biases"))
        spec = quantization[name]
        validate_affine(q, s, b, spec["group_size"], spec["bits"], len(shape))
        require(tuple(q.shape[:-1]) == shape[:-1] and q.shape[-1] * (32 // spec["bits"]) == shape[-1],
                "Actual packed geometry differs: " + name)
        require(s.dtype == torch.float16 and b.dtype == torch.float16,
                "This adapter keeps official FP16 scales/biases without casting")
    for name in norm_names:
        require(tensors[name].dtype == torch.float16 and tuple(tensors[name].shape) == (native["hidden_size"],),
                "Actual dense FP16 norm differs: " + name)
    records = {}; identity = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        require(tensor.is_contiguous(), "Noncontiguous checkpoint tensor " + name)
        require(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()), "Nonfinite checkpoint tensor " + name)
        record, data = tensor_record(tensor); records[name] = record
        identity.update(json.dumps({"name": name, "dtype": record["dtype"], "shape": record["shape"]},
                                  sort_keys=True, separators=(",", ":")).encode() + b"\n")
        identity.update(data)
    return tensors, records, {"effective_dtype": FORMAT, "packed_parameters_sha256": identity.hexdigest(),
        "effective_weights_id": identity.hexdigest(), "tensor_count": len(tensors),
        "identity_algorithm": "sorted canonical name/dtype/shape compact JSON newline, then exact little-endian tensor bytes",
        "scope": "Original official MLX packed Q/S/B plus dense norms; no requantization or dense FP32 identity"}


def make_adapter(tensors, native, quantization):
    import torch
    from torch import nn
    import torch.nn.functional as F
    from executorch.backends.mlx import custom_ops  # noqa: F401
    from executorch.extension.llm.cache import update_and_attend  # noqa: F401

    class Affine(nn.Module):
        def __init__(self, name):
            super().__init__()
            for part in ("weight", "scales", "biases"):
                self.register_buffer(part, tensors[name + "." + part])
            self.bits = quantization[name]["bits"]
            self.group_size = quantization[name]["group_size"]

        def forward(self, x):
            return torch.ops.mlx.native_affine_qmm(x, self.weight, self.scales, self.biases, self.group_size, self.bits)

        def embed(self, indices):
            return torch.ops.mlx.native_affine_embedding(indices, self.weight, self.scales, self.biases, self.group_size, self.bits)

        def expert(self, x, indices, sorted_indices):
            return torch.ops.mlx.native_affine_gather_qmm(x, self.weight, self.scales, self.biases,
                                                        indices, self.group_size, self.bits, sorted_indices)

    class Norm(nn.Module):
        def __init__(self, name):
            super().__init__(); self.register_buffer("weight", tensors[name + ".weight"])
        def forward(self, x):
            return F.rms_norm(x, (native["hidden_size"],), self.weight, native["rms_norm_eps"])

    class Attention(nn.Module):
        def __init__(self, prefix, layer):
            super().__init__(); self.layer = layer
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self, name, Affine(prefix + name))
        def forward(self, x, positions, offset):
            B, T, _ = x.shape; D = native["head_dim"]
            q = self.q_proj(x).reshape(B, T, native["query_heads"], D).transpose(1, 2)
            k = self.k_proj(x).reshape(B, T, native["kv_heads"], D).transpose(1, 2)
            v = self.v_proj(x).reshape(B, T, native["kv_heads"], D).transpose(1, 2)
            q = torch.ops.mlx.rope(q, D, offset, False, native["rope_theta"], 1.0)
            k = torch.ops.mlx.rope(k, D, offset, False, native["rope_theta"], 1.0)
            out = torch.ops.kvcache.update_and_attend(q, k, v, positions.reshape(-1, 1),
                layer_id=self.layer, scale=native["attention_multiplier"], out_dtype=q.dtype)
            return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))

    class MoE(nn.Module):
        def __init__(self, prefix):
            super().__init__(); self.router = nn.Module()
            self.router.layer = Affine(prefix + "router.layer")
            self.switch_mlp = nn.Module()
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(self.switch_mlp, name, Affine(prefix + "switch_mlp." + name))
        def forward(self, x):
            logits = self.router.layer(x)
            indices = torch.ops.mlx.native_topk_indices(logits, native["experts_per_token"])
            selected = torch.ops.mlx.native_take_along_last(logits, indices)
            gates = F.softmax(selected.float(), dim=-1)  # keep FP32 through weighted sum
            B, T, H = x.shape; K = native["experts_per_token"]
            do_sort = B * T * K >= 64
            inputs = x.unsqueeze(-2).unsqueeze(-2)
            if do_sort:
                flat = indices.flatten()
                order = torch.argsort(flat, dim=-1).to(torch.int32)
                inverse = torch.argsort(order, dim=-1).to(torch.int32)
                inputs = torch.index_select(inputs.flatten(0, -3), 0, order // K)
                idx = torch.index_select(flat, 0, order)
            else:
                idx = indices
            up = self.switch_mlp.up_proj.expert(inputs, idx, do_sort)
            gate = self.switch_mlp.gate_proj.expert(inputs, idx, do_sort)
            down = self.switch_mlp.down_proj.expert(F.silu(gate) * up, idx, do_sort)
            if do_sort:
                down = torch.index_select(down, 0, inverse).reshape(B, T, K, 1, H)
            y = down.squeeze(-2)
            return (y.float() * gates.unsqueeze(-1)).sum(dim=-2).to(y.dtype)

    class Layer(nn.Module):
        def __init__(self, layer):
            super().__init__(); p = f"model.layers.{layer}."
            self.input_layernorm = Norm(p + "input_layernorm")
            self.post_attention_layernorm = Norm(p + "post_attention_layernorm")
            self.self_attn = Attention(p + "self_attn.", layer)
            self.block_sparse_moe = MoE(p + "block_sparse_moe.")
        def forward(self, x, positions, offset):
            h = x + self.self_attn(self.input_layernorm(x), positions, offset) * native["residual_multiplier"]
            return h + self.block_sparse_moe(self.post_attention_layernorm(h)) * native["residual_multiplier"]

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__(); self.model = nn.Module()
            self.model.embed_tokens = Affine("model.embed_tokens")
            self.model.layers = nn.ModuleList([Layer(i) for i in range(native["layers"])])
            self.model.norm = Norm("model.norm")
        def forward(self, input_ids, cache_position):
            offset = cache_position[0].item()
            torch._check(offset >= 0)
            torch._check(offset + input_ids.shape[1] <= native["capacity"])
            x = self.model.embed_tokens.embed(input_ids) * native["embedding_multiplier"]
            for layer in self.model.layers:
                x = layer(x, cache_position, offset)
            # No discarded vocabulary rows: norm/head are token-independent.
            x = self.model.norm(x[:, -1:, :])
            return (self.model.embed_tokens(x) / native["logits_scaling"]).float()
    return Adapter().eval()


def validate_exported(exported, length, native):
    import torch
    from collections import Counter
    specs = exported.graph_signature
    require(all(s.kind.name == "USER_OUTPUT" for s in specs.output_specs), "Unexpected graph mutation/output")
    inputs = [s for s in specs.input_specs if s.kind.name == "USER_INPUT"]
    outputs = specs.output_specs
    nodes = {n.name: n for n in exported.graph.nodes}
    require([s.arg.name for s in inputs] == ["input_ids", "cache_position"] and len(outputs) == 1,
            "Only token/position inputs and one logits output are permitted")
    for spec, shape, dtype in ((inputs[0], (1, length), torch.int64), (inputs[1], (length,), torch.int64),
                               (outputs[0], (1, 1, native["vocabulary"]), torch.float32)):
        value = nodes[spec.arg.name].meta["val"]
        require(tuple(value.shape) == shape and value.dtype == dtype, "Actual exported ABI differs")
    updates = [n for n in exported.graph.nodes if n.target == torch.ops.kvcache.update_and_attend.default]
    require(len(updates) == native["layers"], "Missing or duplicate off-graph cache layer")
    for i, node in enumerate(updates):
        layer = node.kwargs.get("layer_id", node.args[4] if len(node.args) > 4 else None)
        scale = node.kwargs.get("scale", node.args[5] if len(node.args) > 5 else None)
        require(layer == i and scale == native["attention_multiplier"], "Actual layer/scaling differs")
    counts = Counter(str(n.target) for n in exported.graph.nodes if n.op == "call_function")
    require(counts["mlx.native_affine_qmm.default"] == native["layers"] * 5 + 1
            and counts["mlx.native_affine_gather_qmm.default"] == native["layers"] * 3
            and counts["mlx.native_affine_embedding.default"] == 1
            and counts["mlx.native_topk_indices.default"] == native["layers"], "Actual MoE/affine graph differs")
    return {"checked": True, "op_counts": dict(counts), "cache_layers": list(range(native["layers"])),
            "full_history_external_cache": True, "public_logits_rows": 1}


def canonical_name(name, originals):
    matches = [candidate for candidate in originals
               if name == candidate or name.endswith("/" + candidate) or name.endswith("." + candidate)]
    require(len(matches) == 1, "Named constant has no unique original binding: " + name)
    return matches[0]


def is_graph_literal_name(name):
    short = name.rsplit("/", 1)[-1]
    return short.startswith("_scalar_") or re.fullmatch(r"_lifted_tensor_constant[0-9]+", short) is not None


def graph_literal(name, layout, payload, native, length):
    """Permit only named scalar graph literals, never renamed tensor weights."""
    require(is_graph_literal_name(name) and layout is not None and list(layout.sizes) == [],
            "Unexpected non-checkpoint named constant " + name)
    formats = {3: ("<i", "int32"), 4: ("<q", "int64"), 5: ("<e", "float16"), 6: ("<f", "float32")}
    require(int(layout.scalar_type) in formats, "Unsupported graph scalar dtype")
    fmt, dtype = formats[int(layout.scalar_type)]
    require(len(payload) == struct.calcsize(fmt), "Invalid graph scalar byte count")
    value = struct.unpack(fmt, payload)[0]
    require(math.isfinite(value), "Nonfinite graph scalar")
    allowed = {0, 1, -1, length, native["capacity"], native["experts_per_token"],
               native["embedding_multiplier"], native["residual_multiplier"], native["logits_scaling"],
               native["attention_multiplier"], native["rms_norm_eps"]}
    def identical(candidate):
        if dtype.startswith("int") and int(candidate) != candidate:
            return False
        return struct.pack(fmt, int(candidate) if dtype.startswith("int") else candidate) == payload
    require(any(identical(candidate) for candidate in allowed), "Unexpected graph literal value")
    return {"canonical_name": None, "origin": "graph_scalar_literal", "dtype": dtype, "shape": [],
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "value": value}


def lowered_evidence(delegate):
    """Read actual FlatBuffer vectors/unions, which debug deserialize skips."""
    import collections
    import importlib
    from executorch.backends.mlx.serialization.mlx_graph_serialize import parse_header
    from executorch.backends.mlx.serialization._generated.mlx_delegate.MLXGraph import MLXGraph
    from executorch.backends.mlx.serialization._generated.mlx_delegate.OpNode import OpNode
    names = {value: key for key, value in vars(OpNode).items() if type(value) is int}
    prefix = "executorch.backends.mlx.serialization._generated.mlx_delegate."
    def unpack(value):
        if value is None or isinstance(value, (int, float, str, bool)):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if type(value).__name__ == "Instruction":
            name = names[value.OpType()]
            cls = getattr(importlib.import_module(prefix + name), name)
            node = cls(); union = value.Op(); node.Init(union.Bytes, union.Pos)
            return {"op": name, "node": unpack(node)}
        result = {}
        for name in dir(value):
            if not name[:1].isupper() or name in ("Init",) or name.startswith("GetRootAs") or name.endswith(("Length", "IsNone", "AsNumpy")):
                continue
            method = getattr(value, name)
            if not callable(method):
                continue
            length = getattr(value, name + "Length", None)
            if length:
                result[name] = [unpack(method(i)) for i in range(length())]
            else:
                try:
                    item = method()
                except TypeError:
                    continue
                result[name] = unpack(item)
        return result
    offset, length, _, _ = parse_header(delegate)
    graph = unpack(MLXGraph.GetRootAs(delegate[offset:offset+length], 0))
    main = graph["InstructionChains"][graph["MainChainIdx"]]["Instructions"]
    graph["main_op_counts"] = dict(collections.Counter(i["op"] for i in main))
    return graph


def named_constant_layouts(lowered):
    """Recover serialized delegate metadata; PTE named blobs omit TensorLayout.

    deserialize_pte_binary deliberately sets named-data tensor_layout=None.
    MLX's own FlatBuffer still carries each named constant's exact tensor slot,
    scalar type, dimensions and dimension order. Read these actual descriptors
    instead of inferring a scalar layout from the blob's name or byte count.
    """
    from executorch.exir.tensor_layout import TensorLayout
    from executorch.exir.scalar_type import ScalarType
    count = lowered["NumConstantTensors"]
    require(type(count) is int and 0 <= count <= len(lowered["TensorMeta"]), "Invalid constant metadata range")
    result = {}; slots = set()
    for item in lowered["NamedSlots"]:
        name = item["Name"]; slot = item["Slot"]
        index = slot["Idx"]
        # NamedSlots also describes public inputs/outputs and value slots.
        # Only tensor indices below NumConstantTensors own named-data blobs.
        if slot["SlotType"] != 0:
            continue
        require(type(index) is int and index >= 0, "Invalid named tensor slot index")
        if index >= count:
            continue
        require(isinstance(name, str) and name and name not in result
                and index not in slots, "Named constant requires a unique actual tensor slot")
        meta = lowered["TensorMeta"][index]
        shape = [dim["Value"] for dim in meta["Shape"]]
        order = meta["DimOrder"]
        require(all(type(d) is int and d > 0 for d in shape)
                and all(type(d) is int for d in order) and sorted(order) == list(range(len(shape))),
                "Named constant has non-static or invalid dimensions " + name)
        require(meta["ScalarType"] in (3, 4, 5, 6, 28), "Unsupported actual constant dtype " + name)
        result[name] = TensorLayout(ScalarType(meta["ScalarType"]), shape, list(order))
        slots.add(index)
    require(len(slots) == count, "Missing named constant tensor metadata")
    return result


def export_one(adapter, length, prefill_tokens, native, destination, common, ptd, originals, trace_only,
               graph_validator=None, pte_filename="native_granite_mlx.pte"):
    import torch
    from executorch.exir.scalar_type import ScalarType
    destination.mkdir()
    example = (torch.ones((1, length), dtype=torch.int64), torch.arange(length, dtype=torch.int64))
    with torch.no_grad():
        exported = torch.export.export(adapter, example, strict=True)
    evidence = (graph_validator or validate_exported)(exported, length, native)
    (destination / "exported_graph.txt").write_text(str(exported.graph) + "\n")
    L = native["layers"]
    metadata = {"get_n_caches": L, "get_kv_heads": torch.full((L,), native["kv_heads"], dtype=torch.int32),
        "get_head_dims": torch.full((L,), native["head_dim"], dtype=torch.int32), "get_windows": torch.zeros(L, dtype=torch.int32),
        "get_prefill_chunk_size": prefill_tokens, "get_max_ctx_len": native["capacity"], "get_kv_dtype": int(ScalarType.HALF)}
    result = {"path": None, "method": "forward", "backend_id": "MLXBackend", "graph_validation": evidence,
        "constant_methods": {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in metadata.items()},
        "abi": {"inputs": [{"name": "input_ids", "dtype": "int64", "shape": [1, length]},
                           {"name": "cache_position", "dtype": "int64", "shape": [length]}],
                "outputs": [{"name": "logits", "dtype": "float32", "shape": [1, 1, native["vocabulary"]]}]}}
    if trace_only:
        return result
    import executorch.exir as exir
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.lowered_backend_module import LoweredBackendModule
    from executorch.exir._serialize._program import deserialize_pte_binary, serialize_pte_binary
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    # Functional ops make data-flow review explicit; no optional reinplace rewrite.
    passes = [p for p in get_default_passes() if type(p).__name__ != "MLXReinplacePass"]
    edge = exir.to_edge_transform_and_lower({"forward": exported}, constant_methods=metadata,
        partitioner=[MLXPartitioner()], transform_passes=passes,
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True))
    ep = edge.exported_program("forward")
    modules = [m for m in ep.graph_module.modules() if isinstance(m, LoweredBackendModule)]
    require(len(modules) == 1 and modules[0].backend_id == "MLXBackend"
            and sum("executorch_call_delegate" in str(n.target) for n in ep.graph.nodes) == 1,
            "Every operation must remain in one whole-model MLX delegate")
    (destination / "edge_graph.txt").write_text(str(ep.graph) + "\n")
    delegate = bytes(modules[0].processed_bytes)
    lowered = lowered_evidence(delegate)
    require(lowered["InitChainIdx"] == -1, "Native packed constants must need no init graph")
    layouts = named_constant_layouts(lowered)
    (destination / "lowered_graph.json").write_text(json.dumps(lowered, indent=2) + "\n")
    program = edge.to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=True,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)))
    memory = io.BytesIO(); program.write_to_file(memory)
    shell = deserialize_pte_binary(memory.getvalue()); data = shell.named_data
    require(data is not None and not data.external_data and data.pte_data, "Missing actual inline named constants")
    require(set(data.pte_data) == set(layouts), "PTE named blobs differ from actual MLX constant slots")
    seen = set(); records = []
    for key, value in data.pte_data.items():
        payload = data.buffers[value.buffer_index]
        layout = layouts[key]
        require(value.tensor_layout is None or value.tensor_layout == layout,
                "PTE and MLX tensor layouts disagree: " + key)
        dtype, width = {3: ("int32", 4), 4: ("int64", 8), 5: ("float16", 2),
                        6: ("float32", 4), 28: ("uint32", 4)}[int(layout.scalar_type)]
        require(len(payload) == math.prod(layout.sizes) * width, "MLX tensor metadata/blob size disagree: " + key)
        if is_graph_literal_name(key):
            record = graph_literal(key, layout, payload, native, length)
        else:
            source = canonical_name(key, originals)
            require(source not in seen, "Duplicate named weight binding " + source); seen.add(source)
            original = originals[source]
            require(layout.sizes == original["shape"] and dtype == original["dtype"]
                    and len(payload) == original["bytes"] and hashlib.sha256(payload).hexdigest() == original["sha256"],
                    "Lowering changed official packed/dense bytes: " + source)
            record = {"canonical_name": source, "origin": "original_official_checkpoint", **original}
        common.add_named_data(key=key, data=payload, alignment=math.lcm(128, value.alignment),
                              external_tag=ptd.name, tensor_layout=layout)
        records.append({"name": key, "layout_source": "serialized_mlx_graph_named_slot", **record})
    require(seen == set(originals), "Not every original checkpoint tensor was retained")
    (destination / "named_constants.json").write_text(json.dumps(records, indent=2) + "\n")
    shell.named_data = None; path = destination / pte_filename
    with path.open("xb") as stream:
        serialize_pte_binary(shell, extract_delegate_segments=True).write_to_file(stream)
    actual = deserialize_pte_binary(path.read_bytes())
    require(actual.named_data is None and actual.program == shell.program and actual.mutable_data == shell.mutable_data,
            "Shared-weight packaging changed program shell")
    result.update(path=str(path), bytes=path.stat().st_size, sha256=digest(path), named_constant_count=len(records),
        delegate_sha256=hashlib.sha256(delegate).hexdigest(), selection={"candidate": "full_mlx_no_reinplace",
        "execution": "single", "transform_passes": [type(p).__name__ for p in passes]},
        packaging_validation={"all_original_tensors_byte_exact": True, "serialized_program_roundtrip": True})
    print(json.dumps({"exported": str(path), "tokens": length, "original_tensors_byte_exact": len(seen)}), flush=True)
    return result


def verify_ptd(path, phase_dirs):
    from executorch.extension.flat_tensor.serialize.serialize import FlatTensorSerializer
    from executorch.exir._serialize._cord import Cord
    payload = FlatTensorSerializer().deserialize(Cord(path.read_bytes()))
    seen = set()
    for directory in phase_dirs:
        for record in json.loads((directory / "named_constants.json").read_text()):
            name = record["name"]; seen.add(name)
            actual = payload.buffers[payload.named_data[name].buffer_index]
            require(len(actual) == record["bytes"] and hashlib.sha256(actual).hexdigest() == record["sha256"],
                    "Serialized PTD changed original tensor " + name)
    require(seen == set(payload.named_data), "Unexpected or missing shared PTD key")
    return {"passed": True, "checked_published_named_slots": len(seen),
            "scope": "Actual serialized PTD bytes equal already verified original official pack tensors"}


def run(args):
    started = time.perf_counter(); output = args.output.resolve(); weights = args.weights.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "started", "stage": "configuration", "role": "target",
        "weights": str(weights), "prefill_tokens": args.prefill_tokens, "verify_tokens": 0,
        "weight_dtype": FORMAT, "hidden_dtype": "float16", "kv_dtype": "float16", "public_logits_dtype": "float32",
        "execution_verified": False, "numerical_validated": False, "quality_validated": False,
        "performance_validated": False, "deployment_cache_window": 0, "target_kv_import_supported": False,
        "official_implementation_reference": REFERENCE,
        "precision_scope": "Original official mixed4/8 affine Q/S/B unchanged; FP16 hidden/KV/norms, FP32 router gates and expert reduction. FP32 public logits are a final cast, not FP32 execution.",
        "adapter_scope": "Explicit Granite-MoE graph adapter; no claim of arbitrary graph equivalence. Native fused SwiGLU versus exported SiLU/multiply requires actual comparison; routing ties are MLX argpartition-defined."}
    try:
        require(args.source_weights_id.strip() == args.source_weights_id and bool(args.source_weights_id),
                "A nonempty exact source repo/revision identity is required")
        raw = json.loads((weights / "config.json").read_text()); native = configuration(raw, args.capacity)
        source = json.loads(args.source_config.read_text()) if args.source_config else raw
        require(configuration(source, args.capacity) == native, "Official converted config changed original architecture")
        require(type(args.prefill_tokens) is int and args.prefill_tokens in (128, 512)
                and args.prefill_tokens <= args.capacity, "Only fixedT128/T512 prefill supported")
        quant = quantization_contract(raw, native)
        configure_local_backend()
        import torch
        from executorch.backends.mlx import native_affine_ops  # noqa: F401
        from executorch.exir._serialize._named_data_store import NamedDataStore
        from executorch.exir._serialize.data_serializer import DataPayload
        from executorch.extension.flat_tensor.serialize.serialize import FlatTensorSerializer
        torch.set_num_threads(4)
        geometry = {k: native[k] for k in ("layers", "hidden_size", "query_heads", "kv_heads", "head_dim", "vocabulary", "capacity")}
        report.update(native_mlx_prefill=native, geometry=geometry,
            original_max_position_embeddings=source["max_position_embeddings"],
            source_weights_identity={"source_weights_id": args.source_weights_id,
                "source_dtype": source.get("torch_dtype", source.get("dtype")),
                "source_config_sha256": digest(args.source_config or weights / "config.json"),
                "scope": "Official source checkpoint provenance; no fabricated dense FP32 parameter digest"},
            quantization={"mode": "affine", "layout": "affine_per_projection", "bits": "mixed", "group_size": 64,
                "bits_by_module": {name: spec["bits"] for name, spec in quant.items()},
                "projection_scope": "granite_moe_affine_per_projection", "dense_tied_embedding_head": False,
                "activation_dtype": "float16", "scales_dtype": "float16", "biases_dtype": "float16"},
            state_shape=[native["layers"], 1, native["kv_heads"], args.capacity, native["head_dim"]],
            runtime_cache_config={"backend_id": "MLXBackend", "kind": "seq", "capacity": args.capacity,
                "n_layers": native["layers"], "layers": [{"policy": {"kind": "Flat", "window": 0},
                    "n_kv_heads": native["kv_heads"], "head_dim": native["head_dim"]}],
                "kv_dtype": "float16", "kv_scalar_type": 5, "initial_capacity": 512, "max_write": args.prefill_tokens})
        normalized = {"rope_type": "default", "rope_theta": native["rope_theta"],
                      "max_position_embeddings": source["max_position_embeddings"]}
        report["rope_configuration"] = {"raw_normalized": normalized, "loaded_normalized": normalized,
                                        "semantically_equivalent": True}
        report["stage"] = "load_original_packed_tensors"
        tensors, originals, identity = load_pack(weights, native, quant)
        report.update(effective_weights_identity=identity, actual_effective_weights_id=identity["effective_weights_id"])
        (output / "source_tensor_identity.json").write_text(json.dumps(originals, indent=2) + "\n")
        adapter = make_adapter(tensors, native, quant)
        common = NamedDataStore(); ptd = output / "target.weights.ptd"; phase_dirs = []
        for phase, length in (("prefill", args.prefill_tokens), ("decode", 1)):
            report["stage"] = "export_" + phase; destination = output / f"T{length}"; phase_dirs.append(destination)
            report[phase + "_pte"] = export_one(adapter, length, args.prefill_tokens, native, destination,
                                                common, ptd, originals, args.trace_only)
        if not args.trace_only:
            report["stage"] = "write_shared_ptd"; store = common.get_named_data_store_output()
            with ptd.open("xb") as stream:
                FlatTensorSerializer().serialize(DataPayload(buffers=store.buffers,
                    named_data=store.external_data[ptd.name])).write_to_file(stream)
            report.update(data_path=str(ptd), weight_packaging={"shared_ptd_bytes": ptd.stat().st_size,
                "unique_named_buffers": len(store.buffers), "named_slots": len(store.external_data[ptd.name]),
                "sha256": digest(ptd), "inline_weight_files_created": False, "requantized": False,
                "runtime_loading_verified": False, "GPU_weight_storage_sharing_verified": False},
                serialized_package_validation=verify_ptd(ptd, phase_dirs))
        report.update(status="graph_validated_no_pte" if args.trace_only else "graph_validated_exported", stage="complete")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        report["preparation_wall_seconds"] = time.perf_counter() - started
        (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": report["status"], "stage": report["stage"], "output": str(output)}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Existing official MLX converted directory")
    parser.add_argument("--source-weights-id", required=True, help="Exact original official repo@revision provenance")
    parser.add_argument("--source-config", type=Path, help="Original HF config, before official MLX conversion")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=8192)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument("--trace-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("Use a fresh output directory")
    raise SystemExit(run(arguments)["status"] == "failed")
