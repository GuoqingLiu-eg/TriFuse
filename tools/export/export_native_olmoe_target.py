"""Export an existing, byte-preserving MLX OLMoE affine pack.

This is an explicit adapter of Apple's mlx_lm/models/olmoe.py, not a new
checkpoint or execution policy. Expert tensors must already be stacked in
official sanitize order (expert 0..E-1), without requantization. The source
config's torch_dtype is provenance; actual Q/S/B/norm dtype is checked below.
Export is structural evidence only. Root runs model quality/performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import export_native_granite_target as shared

require = shared.require
digest = shared.digest
FORMAT = shared.FORMAT
REFERENCE = "https://github.com/ml-explore/mlx-lm/blob/v0.32.0/mlx_lm/models/olmoe.py"


def configuration(raw, capacity):
    require(raw.get("model_type") == "olmoe" and raw.get("hidden_act") == "silu",
            "Need original OLMoE SwiGLU configuration")
    require(raw.get("tie_word_embeddings") is False and raw.get("norm_topk_prob") is False,
            "Only original untied OLMoE with unnormalized top-k probabilities is supported")
    require(raw.get("attention_bias") is False and raw.get("mlp_bias", False) is False
            and raw.get("attention_dropout", 0.0) == 0.0 and raw.get("clip_qkv") is None
            and raw.get("sliding_window") in (None, 0), "Need bias-free unclipped full-history OLMoE")
    require(raw.get("rope_scaling") is None and raw.get("rope_parameters") is None
            and raw.get("rope_traditional", False) is False
            and raw.get("partial_rotary_factor", 1.0) == 1.0, "Only original full default RoPE is supported")
    names = {"layers": "num_hidden_layers", "hidden_size": "hidden_size",
             "intermediate_size": "intermediate_size", "query_heads": "num_attention_heads",
             "kv_heads": "num_key_value_heads", "vocabulary": "vocab_size",
             "num_experts": "num_experts", "experts_per_token": "num_experts_per_tok"}
    native = {key: raw.get(source) for key, source in names.items()}
    require(all(type(v) is int and 0 < v <= 2**31 - 1 for v in native.values()), "Invalid integer geometry")
    original = raw.get("max_position_embeddings")
    require(type(original) is int and type(capacity) is int and 128 <= capacity <= original <= 2**31 - 1,
            "Runtime capacity must not extend original context")
    require(native["hidden_size"] % native["query_heads"] == 0
            and native["query_heads"] % native["kv_heads"] == 0
            and native["experts_per_token"] <= native["num_experts"], "Invalid attention/expert geometry")
    native["head_dim"] = native["hidden_size"] // native["query_heads"]
    require(native["head_dim"] % 2 == 0 and raw.get("head_dim", native["head_dim"]) in (None, native["head_dim"]),
            "Invalid full rotary head dimension")
    for key in ("rms_norm_eps", "rope_theta"):
        value = raw.get(key)
        require(type(value) in (float, int) and math.isfinite(value) and value > 0, "Invalid original scalar " + key)
        native[key] = float(value)
    require(native["rope_theta"] > 1, "Invalid original RoPE base")
    return {"architecture": "olmoe_v1", **native, "capacity": capacity,
            "quantization_layout": "affine_per_projection", "linear_quantization_bits": 0,
            "linear_quantization_group_size": 0, "tie_word_embeddings": False,
            "norm_topk_prob": False, "qk_norm": "full_projection_width",
            "embedding_multiplier": 1.0, "residual_multiplier": 1.0, "logits_scaling": 1.0,
            "attention_multiplier": native["head_dim"] ** -0.5}


def expected_modules(native):
    h, middle, vocab, experts = (native[k] for k in ("hidden_size", "intermediate_size", "vocabulary", "num_experts"))
    q, kv = native["query_heads"] * native["head_dim"], native["kv_heads"] * native["head_dim"]
    result = {"model.embed_tokens": (vocab, h), "lm_head": (vocab, h)}
    for layer in range(native["layers"]):
        p = f"model.layers.{layer}."
        result.update({p + "self_attn." + name + "_proj": shape
                       for name, shape in (("q", (q, h)), ("k", (kv, h)), ("v", (kv, h)), ("o", (h, q)))})
        result[p + "mlp.gate"] = (experts, h)
        for name, shape in (("gate", (experts, middle, h)), ("up", (experts, middle, h)),
                            ("down", (experts, h, middle))):
            result[p + f"mlp.switch_mlp.{name}_proj"] = shape
    return result


def expected_norms(native):
    result = {"model.norm.weight": (native["hidden_size"],)}
    for layer in range(native["layers"]):
        p = f"model.layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            result[p + name + ".weight"] = (native["hidden_size"],)
        result[p + "self_attn.q_norm.weight"] = (native["query_heads"] * native["head_dim"],)
        result[p + "self_attn.k_norm.weight"] = (native["kv_heads"] * native["head_dim"],)
    return result


def quantization_contract(raw, native):
    config = raw.get("quantization", raw.get("quantization_config"))
    require(isinstance(config, dict), "Need original MLX quantization map")
    if "quantization" in raw and "quantization_config" in raw:
        require(raw["quantization"] == raw["quantization_config"], "Quantization aliases disagree")
    modules = expected_modules(native)
    require(not (set(config) - set(modules) - {"bits", "group_size", "mode"}), "Unknown quantization module/option")
    result = {}
    for name, shape in modules.items():
        override = config.get(name, {})
        require(isinstance(override, dict) and not (set(override) - {"bits", "group_size", "mode"}),
                "Every supported projection must have affine quantization: " + name)
        spec = {key: override.get(key, config.get(key, "affine" if key == "mode" else None))
                for key in ("bits", "group_size", "mode")}
        require(type(spec["bits"]) is int and spec["bits"] in (4, 8)
                and type(spec["group_size"]) is int and spec["group_size"] == 64
                and spec["mode"] == "affine" and shape[-1] % 64 == 0, "Need actual affine4/8 group64: " + name)
        result[name] = spec
    return result


def load_pack(root, native, quantization):
    import torch
    from safetensors import safe_open
    from executorch.backends.mlx.native_affine_ops import validate_affine
    paths = sorted(root.glob("*.safetensors"))
    require(paths, "No existing packed safetensors")
    tensors = {}
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            for name in stream.keys():
                require(name not in tensors, "Duplicate tensor " + name)
                tensors[name] = stream.get_tensor(name)
    modules, norms = expected_modules(native), expected_norms(native)
    expected = {name + "." + part for name in modules for part in ("weight", "scales", "biases")} | set(norms)
    require(set(tensors) == expected, "Need exact official sanitize layout (expert axis in 0..E-1 order); missing/extra: "
            + repr(sorted(set(tensors) ^ expected)))
    for name, shape in modules.items():
        q, scales, biases = (tensors[name + "." + part] for part in ("weight", "scales", "biases"))
        spec = quantization[name]
        validate_affine(q, scales, biases, spec["group_size"], spec["bits"], len(shape))
        require(tuple(q.shape[:-1]) == shape[:-1] and q.shape[-1] * (32 // spec["bits"]) == shape[-1],
                "Actual packed geometry differs: " + name)
        require(scales.dtype == biases.dtype == torch.float16, "This adapter preserves existing F16 scales/biases; no casts: " + name)
    for name, shape in norms.items():
        require(tensors[name].dtype == torch.float16 and tuple(tensors[name].shape) == shape,
                "Actual full-width F16 RMSNorm differs: " + name)
    records, identity = {}, hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        require(tensor.is_contiguous() and (not tensor.is_floating_point() or bool(torch.isfinite(tensor).all())),
                "Noncontiguous/nonfinite original parameter " + name)
        record, data = shared.tensor_record(tensor); records[name] = record
        identity.update(json.dumps({"name": name, "dtype": record["dtype"], "shape": record["shape"]},
                                  sort_keys=True, separators=(",", ":")).encode() + b"\n")
        identity.update(data)
    return tensors, records, {"effective_dtype": FORMAT, "packed_parameters_sha256": identity.hexdigest(),
        "effective_weights_id": identity.hexdigest(), "tensor_count": len(tensors),
        "identity_algorithm": "sorted canonical name/dtype/shape compact JSON newline, then exact little-endian bytes",
        "scope": "Existing sanitized official MLX Q/S/B and F16 norms/untied head; no cast or requantization"}


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
            self.bits, self.group = quantization[name]["bits"], quantization[name]["group_size"]
        def forward(self, x):
            return torch.ops.mlx.native_affine_qmm(x, self.weight, self.scales, self.biases, self.group, self.bits)
        def embed(self, ids):
            return torch.ops.mlx.native_affine_embedding(ids, self.weight, self.scales, self.biases, self.group, self.bits)
        def expert(self, x, ids, sorted_indices):
            return torch.ops.mlx.native_affine_gather_qmm(x, self.weight, self.scales, self.biases,
                                                        ids, self.group, self.bits, sorted_indices)
    class Norm(nn.Module):
        def __init__(self, name):
            super().__init__(); self.register_buffer("weight", tensors[name + ".weight"])
        def forward(self, x):
            return F.rms_norm(x, (self.weight.shape[0],), self.weight, native["rms_norm_eps"])
    class Attention(nn.Module):
        def __init__(self, prefix, layer):
            super().__init__(); self.layer = layer
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self, name, Affine(prefix + name))
            self.q_norm, self.k_norm = Norm(prefix + "q_norm"), Norm(prefix + "k_norm")
        def forward(self, x, positions, offset):
            B, T, _ = x.shape; D = native["head_dim"]
            q = self.q_norm(self.q_proj(x)).reshape(B, T, native["query_heads"], D).transpose(1, 2)
            k = self.k_norm(self.k_proj(x)).reshape(B, T, native["kv_heads"], D).transpose(1, 2)
            v = self.v_proj(x).reshape(B, T, native["kv_heads"], D).transpose(1, 2)
            q = torch.ops.mlx.rope(q, D, offset, False, native["rope_theta"], 1.0)
            k = torch.ops.mlx.rope(k, D, offset, False, native["rope_theta"], 1.0)
            out = torch.ops.kvcache.update_and_attend(q, k, v, positions.reshape(-1, 1),
                layer_id=self.layer, scale=native["attention_multiplier"], out_dtype=q.dtype)
            return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))
    class MoE(nn.Module):
        def __init__(self, prefix):
            super().__init__(); self.gate = Affine(prefix + "gate"); self.switch_mlp = nn.Module()
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(self.switch_mlp, name, Affine(prefix + "switch_mlp." + name))
        def forward(self, x):
            probabilities = torch.ops.mlx.native_olmoe_softmax(self.gate(x))
            ids = torch.ops.mlx.native_olmoe_topk_indices(probabilities, native["experts_per_token"])
            scores = torch.ops.mlx.native_take_along_last(probabilities, ids)
            B, T, H = x.shape; K = native["experts_per_token"]; sorted_indices = B * T * K >= 64
            inputs = x.unsqueeze(-2).unsqueeze(-2)
            if sorted_indices:
                flat = ids.flatten(); order = torch.argsort(flat, dim=-1).to(torch.int32)
                inverse = torch.argsort(order, dim=-1).to(torch.int32)
                inputs = torch.index_select(inputs.flatten(0, -3), 0, order // K)
                selected = torch.index_select(flat, 0, order)
            else:
                selected = ids
            up = self.switch_mlp.up_proj.expert(inputs, selected, sorted_indices)
            gate = self.switch_mlp.gate_proj.expert(inputs, selected, sorted_indices)
            out = self.switch_mlp.down_proj.expert(F.silu(gate) * up, selected, sorted_indices)
            if sorted_indices:
                out = torch.index_select(out, 0, inverse).reshape(B, T, K, 1, H)
            return (out.squeeze(-2) * scores.unsqueeze(-1)).sum(dim=-2)
    class Layer(nn.Module):
        def __init__(self, layer):
            super().__init__(); p = f"model.layers.{layer}."
            self.input_layernorm = Norm(p + "input_layernorm")
            self.post_attention_layernorm = Norm(p + "post_attention_layernorm")
            self.self_attn = Attention(p + "self_attn.", layer); self.mlp = MoE(p + "mlp.")
        def forward(self, x, positions, offset):
            h = x + self.self_attn(self.input_layernorm(x), positions, offset)
            return h + self.mlp(self.post_attention_layernorm(h))
    class Adapter(nn.Module):
        def __init__(self):
            super().__init__(); self.model = nn.Module()
            self.model.embed_tokens = Affine("model.embed_tokens")
            self.model.layers = nn.ModuleList([Layer(i) for i in range(native["layers"])])
            self.model.norm = Norm("model.norm"); self.lm_head = Affine("lm_head")
        def forward(self, input_ids, cache_position):
            offset = cache_position[0].item(); torch._check(offset >= 0)
            torch._check(offset + input_ids.shape[1] <= native["capacity"])
            x = self.model.embed_tokens.embed(input_ids)
            for layer in self.model.layers:
                x = layer(x, cache_position, offset)
            return self.lm_head(self.model.norm(x[:, -1:, :])).float()
    return Adapter().eval()


def validate_exported(exported, length, native):
    import torch
    from collections import Counter
    specs = exported.graph_signature; nodes = {n.name: n for n in exported.graph.nodes}
    require(all(s.kind.name == "USER_OUTPUT" for s in specs.output_specs), "Unexpected mutation/output")
    inputs = [s for s in specs.input_specs if s.kind.name == "USER_INPUT"]
    require([s.arg.name for s in inputs] == ["input_ids", "cache_position"] and len(specs.output_specs) == 1,
            "Only input IDs/positions and one vocabulary row are allowed")
    for spec, shape, dtype in ((inputs[0], (1, length), torch.int64), (inputs[1], (length,), torch.int64),
            (specs.output_specs[0], (1, 1, native["vocabulary"]), torch.float32)):
        value = nodes[spec.arg.name].meta["val"]
        require(tuple(value.shape) == shape and value.dtype == dtype, "Actual OLMoE ABI differs")
    updates = [n for n in exported.graph.nodes if n.target == torch.ops.kvcache.update_and_attend.default]
    require(len(updates) == native["layers"], "Missing/duplicate cache layer")
    for i, node in enumerate(updates):
        layer = node.kwargs.get("layer_id", node.args[4] if len(node.args) > 4 else None)
        scale = node.kwargs.get("scale", node.args[5] if len(node.args) > 5 else None)
        require(layer == i and scale == native["attention_multiplier"],
                "Layer or attention scaling differs")
    counts = Counter(str(n.target) for n in exported.graph.nodes if n.op == "call_function")
    L = native["layers"]
    require(counts["mlx.native_affine_qmm.default"] == L * 5 + 1
            and counts["mlx.native_affine_gather_qmm.default"] == L * 3
            and counts["mlx.native_affine_embedding.default"] == 1
            and counts["mlx.native_olmoe_softmax.default"] == L
            and counts["mlx.native_olmoe_topk_indices.default"] == L
            and counts["aten.rms_norm.default"] == L * 4 + 1, "OLMoE operator roles/counts differ")
    return {"checked": True, "op_counts": dict(counts), "cache_layers": list(range(L)),
            "full_history_external_cache": True, "public_logits_rows": 1,
            "qk_norm": "full_projection_width_before_head_split", "router": "full_E_precise_Half_softmax_then_topK_no_renormalization",
            "expert_weighting_reduction_dtype": "float16", "head": "independent_quantized_lm_head"}


def run(args):
    started = time.perf_counter(); output, weights = args.output.resolve(), args.weights.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "started", "stage": "configuration", "role": "target",
        "weights": str(weights), "prefill_tokens": args.prefill_tokens, "verify_tokens": 0,
        "weight_dtype": FORMAT, "hidden_dtype": "float16", "kv_dtype": "float16", "public_logits_dtype": "float32",
        "execution_verified": False, "numerical_validated": False, "quality_validated": False,
        "performance_validated": False, "deployment_cache_window": 0, "target_kv_import_supported": False,
        "official_implementation_reference": REFERENCE,
        "precision_scope": "Existing U32/F16 Q/S/B, F16 hidden/KV/norms and untied head unchanged. Full-E precise softmax outputs Half; weighted experts and reduction remain Half. Public logits are a final FP32 cast.",
        "adapter_scope": "Explicit original OLMoE architecture, not arbitrary graph equality. No new execution policy or context extension. Actual numerical comparison remains required.",
        "export_sources": {str(Path(__file__).resolve()): digest(Path(__file__)),
                           str(Path(shared.__file__).resolve()): digest(Path(shared.__file__))}}
    try:
        require(args.source_weights_id.strip() == args.source_weights_id and bool(args.source_weights_id), "Need exact source repo/revision identity")
        raw = json.loads((weights / "config.json").read_text()); native = configuration(raw, args.capacity)
        source = json.loads(args.source_config.read_text()) if args.source_config else raw
        require(configuration(source, args.capacity) == native, "Converted config changed original architecture")
        require(args.prefill_tokens in (128, 512) and args.prefill_tokens <= args.capacity, "Need fixedT128/T512 prefill")
        quant = quantization_contract(raw, native); shared.configure_local_backend()
        import torch
        from executorch.exir._serialize._named_data_store import NamedDataStore
        from executorch.exir._serialize.data_serializer import DataPayload
        from executorch.extension.flat_tensor.serialize.serialize import FlatTensorSerializer
        torch.set_num_threads(4)
        geometry = {k: native[k] for k in ("layers", "hidden_size", "query_heads", "kv_heads", "head_dim", "vocabulary", "capacity")}
        report.update(native_mlx_prefill=native, geometry=geometry, original_max_position_embeddings=source["max_position_embeddings"],
            source_weights_identity={"source_weights_id": args.source_weights_id,
                "source_dtype": source.get("torch_dtype", source.get("dtype")),
                "source_config_sha256": digest(args.source_config or weights / "config.json"),
                "scope": "Declared source config provenance; effective packed dtypes are validated from existing tensor bytes, not inferred from torch_dtype"},
            quantization={"mode": "affine", "layout": "affine_per_projection", "bits": "mixed", "group_size": 64,
                "bits_by_module": {name: spec["bits"] for name, spec in quant.items()},
                "projection_scope": "olmoe_affine_per_projection", "dense_tied_embedding_head": False,
                "activation_dtype": "float16", "scales_dtype": "float16", "biases_dtype": "float16"},
            state_shape=[native["layers"], 1, native["kv_heads"], args.capacity, native["head_dim"]],
            runtime_cache_config={"backend_id": "MLXBackend", "kind": "seq", "capacity": args.capacity,
                "n_layers": native["layers"], "layers": [{"policy": {"kind": "Flat", "window": 0},
                    "n_kv_heads": native["kv_heads"], "head_dim": native["head_dim"]}],
                "kv_dtype": "float16", "kv_scalar_type": 5, "initial_capacity": 512, "max_write": args.prefill_tokens})
        rope = {"rope_type": "default", "rope_theta": native["rope_theta"], "max_position_embeddings": source["max_position_embeddings"]}
        report["rope_configuration"] = {"raw_normalized": rope, "loaded_normalized": rope, "semantically_equivalent": True}
        report["stage"] = "load_existing_packed_tensors"
        tensors, originals, identity = load_pack(weights, native, quant)
        report.update(effective_weights_identity=identity, actual_effective_weights_id=identity["effective_weights_id"])
        (output / "source_tensor_identity.json").write_text(json.dumps(originals, indent=2) + "\n")
        adapter = make_adapter(tensors, native, quant); common = NamedDataStore(); ptd = output / "target.weights.ptd"
        phases = []
        for phase, length in (("prefill", args.prefill_tokens), ("decode", 1)):
            report["stage"] = "export_" + phase; destination = output / f"T{length}"; phases.append(destination)
            report[phase + "_pte"] = shared.export_one(adapter, length, args.prefill_tokens, native, destination,
                common, ptd, originals, args.trace_only, graph_validator=validate_exported, pte_filename="native_olmoe_mlx.pte")
        if not args.trace_only:
            report["stage"] = "write_shared_ptd"; store = common.get_named_data_store_output()
            with ptd.open("xb") as stream:
                FlatTensorSerializer().serialize(DataPayload(buffers=store.buffers,
                    named_data=store.external_data[ptd.name])).write_to_file(stream)
            report.update(data_path=str(ptd), weight_packaging={"shared_ptd_bytes": ptd.stat().st_size,
                "unique_named_buffers": len(store.buffers), "named_slots": len(store.external_data[ptd.name]),
                "sha256": digest(ptd), "inline_weight_files_created": False, "requantized": False,
                "runtime_loading_verified": False, "GPU_weight_storage_sharing_verified": False},
                serialized_package_validation=shared.verify_ptd(ptd, phases))
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
    parser.add_argument("--weights", type=Path, required=True, help="Existing officially sanitized Q/S/B directory")
    parser.add_argument("--source-weights-id", required=True, help="Exact official converted source repo@revision")
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--trace-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("Use a fresh output directory")
    raise SystemExit(run(arguments)["status"] == "failed")
