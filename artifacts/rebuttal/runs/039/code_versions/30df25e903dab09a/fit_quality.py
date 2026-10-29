"""Refit the unchanged positive calibrator with a genuine eighth method."""
import csv
import hashlib
import json
import time

import torch

from common import RUN, LAYERS, SPEC, write_json
import daso.modeling.quality as quality


def read(path):
    return json.loads(path.read_text())


def main():
    design = read(RUN / 'configs/frozen_design.json')
    assert hashlib.sha256(open(quality.__file__, 'rb').read()).digest() == hashlib.sha256(
        (RUN / 'reference/quality_calibrator_source.py').read_bytes()).digest()
    quality.METHODS = tuple(design['methods'])
    methods = design['methods']
    for scene in ['prefill_only', 'prefill_decode']:
        target = RUN / f'results/quality_model_{scene}.json'
        if target.exists() and (RUN / f'data/actions_{scene}.json').exists():
            continue
        assert not target.exists()
        manifest = read(RUN / f'configs/calibration_{scene}.json')
        errors = {(r['module'], r['phase'], r['method']): float(r['output_rel_mse'])
                  for r in csv.DictReader((RUN / f'data/local_errors_{scene}.csv').open())}
        phases = ['prefill'] if scene == 'prefill_only' else ['prefill', 'decode']
        baseline = read(RUN / f'results/training/{scene}/{manifest[0]["policy_id"]}.json')['avg_nll']
        features, labels = [], []
        for entry in manifest:
            pid = entry['policy_id']
            assert hashlib.sha256((RUN / f'configs/policies/{pid}.json').read_bytes()).hexdigest() == entry['policy_sha256']
            result_path = RUN / f'results/training/{scene}/{pid}.json'
            if 'source_sha256' in entry:
                assert hashlib.sha256(result_path.read_bytes()).hexdigest() == entry['source_sha256']
            result = read(result_path)
            mapping = read(RUN / f'configs/policies/{pid}.json')['method_map']
            x = torch.zeros((2, len(methods), 4, 4), dtype=torch.float64)
            for name, pair in mapping.items():
                bucket = int(name.split('.')[2])*4//LAYERS
                kind = quality.FUSED_TYPES.index(name.rsplit('.', 1)[-1])
                for pi, phase in enumerate(phases):
                    method = pair[phase+'_method']
                    if method != 'dense_bf16':
                        x[pi, methods.index(method), bucket, kind] += errors[name, phase, method]
            features.append(x)
            labels.append(result['avg_nll']-baseline)
        x, y = torch.stack(features), torch.tensor(labels, dtype=torch.float64)
        train = torch.tensor([e['split'] == 'train' for e in manifest])
        started = time.perf_counter()
        fit = quality.QualityCalibrator(**{k: v for k, v in design['quality_fit'].items() if k != 'architecture'}).fit(x, y, train)
        fit.save(target)
        predicted = fit.predict(x)
        metrics = {}
        for name, mask in [('train', train), ('holdout', ~train),
            ('new_methods_holdout', (~train) & torch.tensor([e['origin'] == SPEC['run'] for e in manifest]))]:
            metrics[name] = {'count': int(mask.sum()), 'mae': float((predicted[mask]-y[mask]).abs().mean()),
                             'rmse': float((predicted[mask]-y[mask]).square().mean().sqrt())}
        write_json(RUN / f'results/quality_fit_{scene}.json', {'metrics': metrics,
            'seconds': time.perf_counter()-started, 'methods': methods, 'training_reference_nll': baseline,
            'local_errors_sha256': hashlib.sha256((RUN / f'data/local_errors_{scene}.csv').read_bytes()).hexdigest(),
            'scope': 'Original architecture,4000 steps,lr.03,regularization.05; train split only, no tuning on holdout/confirmation.'})
        write_json(RUN / f'results/quality_predictions_{scene}.json', [dict(e, actual_delta_nll=float(a), predicted_delta_nll=float(b))
            for e, a, b in zip(manifest, y, predicted, strict=True)])
        q = [[0. if method == 'dense_bf16' else errors[name, phase, method] * float(
            fit.coefficients[pi, methods.index(method), int(name.split('.')[2])*4//LAYERS,
                             quality.FUSED_TYPES.index(name.rsplit('.', 1)[-1])]) / fit.feature_scale
            for method in methods] for pi, phase in enumerate(phases) for name in design['names']]
        write_json(RUN / f'data/actions_{scene}.json', {'quality': q, 'bias': fit.bias, 'methods': methods,
            'names': design['names'], 'phases': phases})
    print('Two eight-action quality models fitted.', flush=True)


if __name__ == '__main__':
    main()
