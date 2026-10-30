"""Check the local 32B checkpoint without loading tensors or changing weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    config = json.loads((args.model / 'config.json').read_text())
    index = json.loads((args.model / 'model.safetensors.index.json').read_text())
    tensors, shards = {}, []
    for filename in sorted(set(index['weight_map'].values())):
        path = args.model / filename
        with path.open('rb') as stream:
            size, = struct.unpack('<Q', stream.read(8))
            header = json.loads(stream.read(size))
        entries = {k: v for k, v in header.items() if k != '__metadata__'}
        assert set(entries) == {k for k, v in index['weight_map'].items() if v == filename}
        end = 0
        for name, entry in sorted(entries.items(), key=lambda item: item[1]['data_offsets'][0]):
            assert name not in tensors
            assert entry['dtype'] == 'BF16', (name, entry)
            start, stop = entry['data_offsets']
            assert start == end and stop - start == math.prod(entry['shape']) * 2
            end = stop
            tensors[name] = entry['shape']
        assert 8 + size + end == path.stat().st_size
        digest = sha256(path)
        shards.append({'file': filename, 'bytes': path.stat().st_size, 'sha256': digest,
                       'tensor_count': len(entries)})
        print(filename, 'verified', digest, flush=True)
    layers = config['num_hidden_layers']
    shapes = {'self_attn.q_proj': [config['num_attention_heads'] * config['head_dim'], config['hidden_size']],
              'self_attn.k_proj': [config['num_key_value_heads'] * config['head_dim'], config['hidden_size']],
              'self_attn.v_proj': [config['num_key_value_heads'] * config['head_dim'], config['hidden_size']],
              'self_attn.o_proj': [config['hidden_size'], config['num_attention_heads'] * config['head_dim']],
              'mlp.gate_proj': [config['intermediate_size'], config['hidden_size']],
              'mlp.up_proj': [config['intermediate_size'], config['hidden_size']],
              'mlp.down_proj': [config['hidden_size'], config['intermediate_size']]}
    for layer in range(layers):
        for module, shape in shapes.items():
            assert tensors[f'model.layers.{layer}.{module}.weight'] == shape
    tensor_bytes = sum(math.prod(shape) * 2 for shape in tensors.values())
    assert tensor_bytes == index['metadata']['total_size']
    metadata = {}
    for path in sorted(args.model.iterdir()):
        if path.is_file() and path.suffix in ('.json', '.txt'):
            metadata[path.name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    payload = {'complete': True, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'model_path': str(args.model.resolve()), 'config': config, 'shards': shards,
               'metadata': metadata, 'tensor_count': len(tensors),
               'parameter_count': tensor_bytes // 2, 'weight_bytes': tensor_bytes,
               'layer_count': layers, 'projection_tensors_checked': layers * len(shapes),
               'limitation': 'Local SHA identities and layout verified; no upstream content checksum comparison.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({k: payload[k] for k in ('complete', 'tensor_count', 'parameter_count', 'weight_bytes')}, indent=2))


if __name__ == '__main__':
    main()
