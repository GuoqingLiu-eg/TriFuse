#!/usr/bin/env python3
"""CPU byte-only OLMoE expert stacking; no tensor framework, casts or quantization."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import tempfile
import time

DTYPES = {'F16': 2, 'U32': 4}
EXPERT = re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$')
BLOCK = 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            result.update(block)
    return result.hexdigest()


def read_header(path):
    with Path(path).open('rb') as stream:
        length_bytes = stream.read(8)
        require(len(length_bytes) == 8, 'Missing safetensors header length')
        length = struct.unpack('<Q', length_bytes)[0]
        require(0 < length <= 16 * 1024 * 1024, 'Invalid header length')
        raw = stream.read(length)
        require(len(raw) == length, 'Incomplete safetensors header')
        def pairs(items):
            result = {}
            for key, value in items:
                require(key not in result, 'Duplicate safetensors tensor/key')
                result[key] = value
            return result
        header = json.loads(raw, object_pairs_hook=pairs)
    require(isinstance(header, dict), 'Expected safetensors object')
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    require(tensors, 'No tensors')
    ranges = []
    for name, tensor in tensors.items():
        shape, dtype, offsets = tensor.get('shape'), tensor.get('dtype'), tensor.get('data_offsets')
        require(dtype in DTYPES, 'Unexpected source dtype: ' + name)
        require(isinstance(shape, list) and all(type(x) is int and x > 0 for x in shape), 'Invalid tensor shape')
        require(isinstance(offsets, list) and len(offsets) == 2
                and all(type(x) is int and x >= 0 for x in offsets), 'Invalid tensor offsets')
        require(offsets[1] - offsets[0] == math.prod(shape) * DTYPES[dtype], 'Tensor byte count mismatch')
        ranges.append((offsets[0], offsets[1]))
    cursor = 0
    for start, end in sorted(ranges):
        require(start == cursor and end > start, 'Tensor payload gap/overlap')
        cursor = end
    require(Path(path).stat().st_size == 8 + length + cursor, 'Incomplete or oversized source file')
    return header, 8 + length


def plan(header, config):
    require(config.get('model_type') == 'olmoe', 'Expected OLMoE configuration')
    layers, experts = config.get('num_hidden_layers'), config.get('num_experts')
    require(type(layers) is int and layers > 0 and type(experts) is int and experts > 0, 'Invalid layer/expert count')
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    direct, groups = {}, {}
    for name, tensor in tensors.items():
        match = EXPERT.fullmatch(name)
        if match:
            layer, expert, projection, suffix = match.groups()
            layer, expert = int(layer), int(expert)
            require(layer < layers and expert < experts, 'Out-of-range expert tensor')
            groups.setdefault((layer, projection, suffix), {})[expert] = (name, tensor)
        else:
            require('.mlp.experts.' not in name and '.mlp.switch_mlp.' not in name,
                    'Unrecognized/already stacked expert layout')
            direct[name] = [name]
    require(len(groups) == layers * 9, 'Expected all gate/up/down packed weight/scales/biases groups')
    for (layer, projection, suffix), members in groups.items():
        require(set(members) == set(range(experts)), 'Missing expert within a projection group')
        first = members[0][1]
        require(len(first['shape']) == 2 and first['dtype'] == ('U32' if suffix == 'weight' else 'F16'),
                'Invalid packed expert dtype/rank')
        require(all(t['dtype'] == first['dtype'] and t['shape'] == first['shape']
                    for _, t in members.values()), 'Mismatched expert shape/dtype')
        direct[f'model.layers.{layer}.mlp.switch_mlp.{projection}.{suffix}'] = [members[e][0] for e in range(experts)]
    output, fragments, cursor = {}, {}, 0
    if '__metadata__' in header:
        output['__metadata__'] = header['__metadata__']
    for name in sorted(direct):
        names = direct[name]
        tensor = tensors[names[0]]
        shape = ([experts] + tensor['shape']) if '.mlp.switch_mlp.' in name else tensor['shape']
        size = math.prod(shape) * DTYPES[tensor['dtype']]
        output[name] = dict(dtype=tensor['dtype'], shape=shape, data_offsets=[cursor, cursor + size])
        fragments[name] = [(n, tensors[n]['data_offsets']) for n in names]
        cursor += size
    require(sum(t['data_offsets'][1] - t['data_offsets'][0] for t in tensors.values()) == cursor,
            'Payload byte count changed')
    return output, fragments


def transfer(source, destination, count, digest):
    while count:
        block = source.read(min(BLOCK, count))
        require(block, 'Unexpected EOF during byte copy')
        if destination is not None:
            destination.write(block)
        digest.update(block)
        count -= len(block)


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(not output.exists(), 'Output must be fresh; failed evidence is never overwritten')
    config = json.loads((source / 'config.json').read_text())
    model = source / 'model.safetensors'
    original_stat = model.stat()
    header, payload_start = read_header(model)
    arranged, fragments = plan(header, config)
    source_sha = sha(model)
    encoded = json.dumps(arranged, separators=(',', ':'), ensure_ascii=False).encode('utf8')
    encoded += b' ' * ((-len(encoded)) % 8)
    output.mkdir(parents=True)
    result = dict(schema='olmoe_exact_expert_stack_v1', status='started', model_executed=False,
                  quantization_performed=False, dtype_cast_performed=False, source=str(model),
                  source_sha256=source_sha, source_bytes=original_stat.st_size,
                  expert_order=list(range(config['num_experts'])), stack_axis=0,
                  source_config_sha256=sha(source / 'config.json'),
                  scope='Byte-only equivalent of upstream OlmoE.sanitize expert0..N stack; all nonexpert tensors unchanged. Both engines use the same resulting checkpoint.')
    receipt = output.parent / (output.name + '-preparation.json')
    require(not receipt.exists(), 'Preparation receipt must be fresh')
    def save():
        receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    save()
    began = time.monotonic()
    try:
        partial = output / 'model.safetensors.partial'
        expected = {}
        with model.open('rb') as inp, partial.open('xb') as out:
            out.write(struct.pack('<Q', len(encoded))); out.write(encoded)
            for name, pieces in fragments.items():
                digest = hashlib.sha256()
                for _, (start, end) in pieces:
                    inp.seek(payload_start + start)
                    transfer(inp, out, end - start, digest)
                expected[name] = digest.hexdigest()
        actual, output_start = read_header(partial)
        require(actual == arranged, 'Saved output header differs')
        with partial.open('rb') as stream:
            for name, digest in expected.items():
                start, end = arranged[name]['data_offsets']
                stream.seek(output_start + start)
                check = hashlib.sha256()
                transfer(stream, None, end - start, check)
                require(check.hexdigest() == digest, 'Copied tensor byte mismatch: ' + name)
        now = model.stat()
        require((now.st_size, now.st_mtime_ns, now.st_ino) ==
                (original_stat.st_size, original_stat.st_mtime_ns, original_stat.st_ino), 'Source changed during preparation')
        copied = []
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                     'generation_config.json', 'added_tokens.json', 'vocab.json', 'merges.txt', 'README.md'):
            path = source / name
            if path.is_file():
                shutil.copyfile(path, output / name)
                require(sha(path) == sha(output / name), 'Metadata copy mismatch')
                copied.append(dict(name=name, sha256=sha(path), bytes=path.stat().st_size))
        require((output / 'tokenizer.json').is_file(), 'Missing local tokenizer')
        final = output / 'model.safetensors'
        partial.rename(final)
        result.update(status='complete', elapsed_cpu_io_seconds=time.monotonic() - began,
                      output=str(final), output_sha256=sha(final), output_bytes=final.stat().st_size,
                      source_tensor_count=len(header) - ('__metadata__' in header), output_tensor_count=len(fragments),
                      stacked_groups=sum(len(x) > 1 for x in fragments.values()),
                      output_payload_bytes=sum(t['data_offsets'][1] - t['data_offsets'][0]
                                               for k, t in arranged.items() if k != '__metadata__'),
                      all_output_tensor_bytes_verified=True, output_tensor_sha256=expected, metadata_files=copied)
        save()
        return result
    except BaseException as error:
        result.update(status='failed', error=repr(error)); save(); raise


def self_test():
    # Independent tiny raw payloads, distinct expert values, and direct tensors.
    with tempfile.TemporaryDirectory(prefix='olmoe-byte-stack-') as temp:
        root = Path(temp); source = root / 'source'; source.mkdir()
        config = dict(model_type='olmoe', num_hidden_layers=2, num_experts=3)
        (source / 'config.json').write_text(json.dumps(config)); (source / 'tokenizer.json').write_text('{}')
        header, payload, expected = {'__metadata__': {'format': 'mlx'}}, bytearray(), {}
        for layer in range(2):
            for projection in ('gate_proj', 'up_proj', 'down_proj'):
                for suffix in ('weight', 'scales', 'biases'):
                    stack = b''
                    for expert in range(3):
                        dtype = 'U32' if suffix == 'weight' else 'F16'
                        name = f'model.layers.{layer}.mlp.experts.{expert}.{projection}.{suffix}'
                        raw = struct.pack('<4I' if dtype == 'U32' else '<4e', *[layer * 10 + expert + 1] * 4)
                        header[name] = dict(dtype=dtype, shape=[2, 2], data_offsets=[len(payload), len(payload) + len(raw)])
                        payload.extend(raw); stack += raw
                    expected[f'model.layers.{layer}.mlp.switch_mlp.{projection}.{suffix}'] = stack
        raw = struct.pack('<2e', 0.25, -2.0)
        header['model.norm.weight'] = dict(dtype='F16', shape=[2], data_offsets=[len(payload), len(payload) + len(raw)])
        payload.extend(raw); expected['model.norm.weight'] = raw
        encoded = json.dumps(header).encode(); encoded += b' ' * ((-len(encoded)) % 8)
        (source / 'model.safetensors').write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
        report = prepare(source, root / 'output')
        saved, begin = read_header(root / 'output/model.safetensors')
        data = (root / 'output/model.safetensors').read_bytes()
        for name, raw in expected.items():
            a, b = saved[name]['data_offsets']; require(data[begin + a:begin + b] == raw, 'Independent byte oracle differs')
        require(report['stacked_groups'] == 18 and report['output_tensor_count'] == 19, 'Incorrect stack count')
        bad = dict(header); del bad['model.layers.0.mlp.experts.1.gate_proj.weight']
        try:
            plan(bad, config)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing expert was accepted')
    return dict(status='passed', independent_tensor_byte_checks=19, missing_expert_rejected=True,
                model_executed=False, tensor_framework_imported=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    else:
        if args.source is None or args.output is None:
            parser.error('--source and --output are required')
        report = prepare(args.source, args.output)
        print(json.dumps({k: report[k] for k in ('status', 'output', 'output_sha256', 'output_bytes', 'stacked_groups', 'model_executed')}))
