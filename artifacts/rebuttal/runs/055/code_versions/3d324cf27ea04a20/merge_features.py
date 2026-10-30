"""Require a complete new-model feature grid before fitting any quality model."""
import csv
import json
import math

from common import RUN, write_json


def main():
    output = RUN / 'data/local_errors.csv'
    assert not output.exists()
    rows = []
    for name in ['local_errors_five.csv', 'local_errors_8bit.csv']:
        with (RUN / 'data' / name).open() as stream:
            rows.extend(csv.DictReader(stream))
    manifest = json.loads((RUN / 'configs/calibration_manifest.json').read_text())
    policy = json.loads((RUN / 'configs/policies/uniform_dense_bf16.json').read_text())
    expected = {(module, phase, method) for module in policy['method_map']
                for phase in ['prefill', 'decode'] for method in manifest['methods'][1:]}
    actual = {(row['module'], row['phase'], row['method']) for row in rows}
    assert actual == expected and len(rows) == len(expected)
    assert all(math.isfinite(float(row['output_rel_mse'])) and float(row['output_rel_mse']) >= 0 for row in rows)
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(RUN / 'results/features_complete.json', {'complete': True, 'rows': len(rows),
        'model_layers': 64, 'fused_modules': 256, 'methods': manifest['methods'],
        'local_windows': 16, 'prefill_tokens': 2048, 'continuation_tokens': 64,
        'scope': 'Identical dense BF16 inputs; continuation features are batched teacher-forced positions, '
                 'not an incremental compressed-KV decode replay. Full-policy labels use real KV.'})


if __name__ == '__main__':
    main()
