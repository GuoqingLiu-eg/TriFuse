#!/usr/bin/env python3
"""Export fixed g64 min/max assistance; never load/predict the Core ML model.

The owner runs conversion explicitly. No checkpoint, Hadamard weights, training,
GPU, or ANE inference is involved in this exporter. Output must be a fresh
.mlpackage path; a small adjacent .export.json records source and package hashes.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

SHAPE = (1, 128, 1, 4096)
OUTPUT_SHAPE = (1, 4, 1, 4096)
SEED = 20261009
METADATA = {
    "trifuse.component": "affine_bounds",
    "trifuse.format": "mlx_affine_uint32_g64_b8",
    "trifuse.group_size": "64",
    "trifuse.bits": "8",
    "trifuse.trained": "false",
    "trifuse.bounds_order": "min0,min1,max0,max1",
    "trifuse.ane_execution_verified": "false",
}


def specification(graph="reduction"):
    return {
        "schema": "trifuse.cold-kv-affine-codec.v1", "metadata": {**METADATA, "trifuse.graph": graph},
        "graph": graph,
        "input": {"name": "frozen_state", "shape": list(SHAPE), "dtype": "float16",
                  "layout": "channel=dim; width=head*256+token"},
        "output": {"name": "bounds", "shape": list(OUTPUT_SHAPE), "dtype": "float16",
                   "channels": ["min(group0)", "min(group1)", "max(group0)", "max(group1)"]},
        "group_size": 64, "bits": 8, "transport_scale": 1,
        "trained": False, "learned_weights": 0,
        "quantization": "CPU packing from original FP16 snapshot and graph bounds; pinned MLX Metal affine_quantize semantics",
        "stored_one_k_or_v_block_bytes": 557056,
        "stored_k_and_v_block_bytes": 1114112,
        "coreml_prediction_executed": False, "ane_execution_verified": False,
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(output, graph="reduction"):
    import coremltools as ct
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as functional

    output = output.resolve()
    report_path = output.with_suffix(".export.json")
    if output.suffix != ".mlpackage" or output.exists() or report_path.exists():
        raise ValueError("--output must be a fresh .mlpackage path with no adjacent export report")
    output.parent.mkdir(parents=True, exist_ok=True)

    class PoolBounds(nn.Module):
        def forward(self, frozen_state):
            # Each vector is independent; pool over dimensions, never tokens.
            spatial = frozen_state.reshape(1, 1, 128, 4096)
            maximum = functional.max_pool2d(functional.max_pool2d(spatial, (8, 1), (8, 1)), (8, 1), (8, 1))
            minimum = -functional.max_pool2d(functional.max_pool2d(-spatial, (8, 1), (8, 1)), (8, 1), (8, 1))
            return torch.cat((minimum.reshape(1, 2, 1, 4096), maximum.reshape(1, 2, 1, 4096)), dim=1)

    class ReductionBounds(nn.Module):
        def forward(self, frozen_state):
            # Keep NCHW geometry throughout: two channel slices, four channel
            # reductions, one concat. No reshape, pooling, negation or transpose.
            first = frozen_state[:, :64, :, :]
            second = frozen_state[:, 64:, :, :]
            return torch.cat((torch.amin(first, dim=1, keepdim=True),
                              torch.amin(second, dim=1, keepdim=True),
                              torch.amax(first, dim=1, keepdim=True),
                              torch.amax(second, dim=1, keepdim=True)), dim=1)

    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    x = (rng.standard_normal(SHAPE) * .25).astype(np.float16)
    x[0, :, 0, 0] = 0
    x[0, :64, 0, 1] = -2
    x[0, 64:, 0, 1] = 3
    for token, value in ((63, 4), (64, -5), (255, 6), (256, -7), (4095, 8)):
        x[0, token % 128, 0, token] = value
    groups = x.reshape(1, 2, 64, 4096)
    reference = np.concatenate((groups.min(axis=2), groups.max(axis=2)), axis=1).reshape(OUTPUT_SHAPE)
    with torch.inference_mode():
        example = torch.from_numpy(x.copy())
        if graph not in ("pool", "reduction"):
            raise ValueError("Unknown graph")
        network = PoolBounds() if graph == "pool" else ReductionBounds()
        traced = torch.jit.trace(network.eval().cpu(), example)
        actual = traced(example).numpy()
    if actual.dtype != np.float16 or actual.shape != OUTPUT_SHAPE or not np.array_equal(actual, reference):
        raise RuntimeError("CPU trace bounds mismatch; no package published")
    model = ct.convert(traced, source="pytorch", convert_to="mlprogram",
        inputs=[ct.TensorType(name="frozen_state", shape=SHAPE, dtype=np.float16)],
        outputs=[ct.TensorType(name="bounds", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
    model.short_description = "Untrained per-vector g64 min/max for MLX affine KV packing"
    model.user_defined_metadata.update({**METADATA, "trifuse.graph": graph})
    spec = model.get_spec()
    for feature, name, shape in ((spec.description.input[0], "frozen_state", SHAPE),
                                 (spec.description.output[0], "bounds", OUTPUT_SHAPE)):
        array = feature.type.multiArrayType
        if feature.name != name or tuple(array.shape) != shape or array.dataType != ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16:
            raise RuntimeError("Converted CoreML feature ABI differs")
    model.save(str(output))
    files = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(output.rglob("*")) if p.is_file()}
    report = specification(graph)
    report.update(package=str(output), files=files,
        package_bytes=sum(item["bytes"] for item in files.values()),
        tree_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        source_sha256=digest(__file__), seed=SEED, cpu_trace_reference_passed=True,
        cpu_trace_max_abs_error=float(np.max(np.abs(actual.astype(np.float32)-reference.astype(np.float32)))),
        versions={name: importlib.metadata.version(name) for name in ("torch", "coremltools", "numpy")},
        python=sys.version, platform=platform.platform(), skip_model_load=True,
        conversion_compute_units="CPU_ONLY")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"package": str(output), "report": str(report_path), "graph": graph, "coreml_prediction_executed": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--graph", choices=("pool", "reduction"), default="reduction")
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(specification(args.graph), indent=2)); return
    if args.output is None:
        parser.error("--output is required unless --describe")
    export(args.output, args.graph)


if __name__ == "__main__":
    main()
