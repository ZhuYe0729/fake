from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from baseline_worker import TEMP, invoke
from common import MODEL, RUN, VLLM, WRAPPER, write_json
from daso.modeling.quality import FUSED_TYPES, QualityCalibrator
from daso.modeling.latency import KernelLatencyBackend
from daso.optimization import Action, ModuleChoices, solve_budget
from daso.policy import METHODS


def prepare():
    names = list(json.loads((RUN / 'configs/policies/uniform_dense_bf16.json').read_text())['method_map'])
    for scenario, seed in [('prefill_only', 731), ('prefill_decode', 732)]:
        rng = np.random.default_rng(seed)
        rates = np.geomspace(.01, 1., 67)
        rng.shuffle(rates)
        manifest = []
        for index in range(72):
            policy_id = f'cal_{scenario}_p{index:02d}'
            choices = {}
            probabilities = rng.dirichlet(np.ones(4))
            for name in names:
                if index < 5:
                    pre = dec = METHODS[index]
                else:
                    pre = str(rng.choice(METHODS[1:], p=probabilities)) if rng.random() < rates[index - 5] else 'dense_bf16'
                    dec = str(rng.choice(METHODS[1:], p=probabilities)) if rng.random() < rates[index - 5] else 'dense_bf16'
                    if scenario == 'prefill_only':
                        dec = pre
                choices[name] = {'prefill_method': pre, 'decode_method': dec}
            policy = {'policy_id': policy_id, 'scenario': scenario,
                'default_prefill_method': 'dense_bf16', 'default_decode_method': 'dense_bf16',
                'modules_to_not_convert': ['lm_head'], 'method_map': choices}
            write_json(RUN / f'configs/policies/{policy_id}.json', policy)
            manifest.append({'policy_id': policy_id, 'split': 'train' if index < 54 else 'holdout',
                             'policy_kind': 'uniform' if index < 5 else 'heterogeneous'})
        write_json(RUN / f'configs/calibration_{scenario}.json', manifest)
    tokens = torch.load(RUN / 'data/calibration_128_2112.pt', weights_only=True)
    torch.save(tokens[:100, :2048].contiguous(), RUN / 'data/calibration_train_100_2048.pt')
    print('Frozen 54 training + 18 held-out policies per scenario', flush=True)


def calibrate(scenario, begin, end):
    manifest = json.loads((RUN / f'configs/calibration_{scenario}.json').read_text())
    for index in range(begin, min(end, len(manifest))):
        policy_id = manifest[index]['policy_id']
        target = RUN / f'results/calibration/{scenario}/{policy_id}.json'
        if target.exists():
            continue
        started = time.perf_counter()
        policy = RUN / f'configs/policies/{policy_id}.json'
        if index < 5:
            checkpoint = TEMP / 'checkpoints' / ('uniform_' + METHODS[index])
            temporary = False
        else:
            checkpoint = TEMP / 'calibration_checkpoints' / policy_id
            temporary = True
            command = [sys.executable, str(RUN / 'code/export_model.py'), '--model-path', str(MODEL),
                '--policy-json', str(policy), '--output-dir', str(checkpoint),
                '--cutlass-wrapper-path', str(WRAPPER)]
            for method in ['sparse_bf16', 'sparse_nvfp4']:
                command += ['--canonical-' + method.replace('_', '-') + '-state',
                            str(TEMP / 'canonical' / method / 'model.pt')]
            if not (checkpoint / 'phase_hetero_export_provenance.json').exists():
                invoke(command, RUN / f'logs/{policy_id}_export.log')
        command = [sys.executable, str(RUN / 'code/run_workflow.py')]
        if scenario == 'prefill_only':
            command += ['prefill_nll', '--checkpoint', str(checkpoint),
                '--samples', str(RUN / 'data/calibration_train_100_2048.pt'), '--policy-json', str(policy)]
        else:
            command += ['decode_nll', '--model', str(checkpoint),
                '--samples', str(RUN / 'data/calibration_128_2112.pt'),
                '--input-tokens','2048','--output-tokens','64','--batch-size','8',
                '--gpu-memory-utilization','.70']
        command += ['--tokenizer', str(MODEL), '--output', str(target), '--label', policy_id,
            '--blocks', '100', '--phase-hetero', '--vllm-root', str(VLLM),
            '--cutlass-wrapper-path', str(WRAPPER)]
        invoke(command, RUN / f'logs/{policy_id}_nll.log')
        if temporary:
            evidence = RUN / f'results/calibration/{scenario}/exports/{policy_id}'
            evidence.mkdir(parents=True, exist_ok=True)
            for source in checkpoint.glob('*.json'):
                shutil.copy2(source, evidence / source.name)
            binary = checkpoint / 'model.safetensors'
            size = binary.stat().st_size
            binary.unlink()
            write_json(evidence / 'temporary_binary_cleanup.json', {
                'path':str(binary), 'bytes':size,
                'reason':'experiment-owned reproducible temporary export after successful NLL',
                'preserved':'all metadata, code, logs and result files',
                'rebuild':'same archived policy with code/export_model.py and canonical states'})
        print(policy_id, 'completed', time.perf_counter() - started, flush=True)


def fit(scenario):
    manifest = json.loads((RUN / f'configs/calibration_{scenario}.json').read_text())
    with (RUN / 'data/local_errors.csv').open() as handle:
        errors = {(row['module'], row['phase'], row['method']): float(row['output_rel_mse'])
                  for row in csv.DictReader(handle)}
    phases = ['prefill'] if scenario == 'prefill_only' else ['prefill', 'decode']
    features, labels = [], []
    reference = json.loads((RUN / f'results/calibration/{scenario}/{manifest[0]["policy_id"]}.json').read_text())['avg_nll']
    for entry in manifest:
        policy = json.loads((RUN / f'configs/policies/{entry["policy_id"]}.json').read_text())
        feature = torch.zeros((2, len(METHODS), 4, len(FUSED_TYPES)), dtype=torch.float64)
        for name, pair in policy['method_map'].items():
            bucket = int(name.split('.')[2]) * 4 // 36
            kind = FUSED_TYPES.index(name.rsplit('.', 1)[-1])
            for phase_index, phase in enumerate(phases):
                method = pair[phase + '_method']
                if method != 'dense_bf16':
                    feature[phase_index, METHODS.index(method), bucket, kind] += errors[name, phase, method]
        features.append(feature)
        result = json.loads((RUN / f'results/calibration/{scenario}/{entry["policy_id"]}.json').read_text())
        labels.append(result['avg_nll'] - reference)
    x, y = torch.stack(features), torch.tensor(labels, dtype=torch.float64)
    train = torch.tensor([entry['split'] == 'train' for entry in manifest])
    fitted = QualityCalibrator().fit(x, y, train)
    output = RUN / f'results/daso/{scenario}'
    output.mkdir(parents=True, exist_ok=True)
    fitted.save(output / 'quality_model.json')
    predicted = fitted.predict(x)
    write_json(output / 'quality_predictions.json', [{**entry, 'actual_delta_nll': actual,
        'predicted_delta_nll': float(prediction)} for entry, actual, prediction in zip(manifest, labels, predicted, strict=True)])
    write_json(output / 'quality_metrics.json', {split:{'count':int(mask.sum()),
        'mae':float((predicted[mask]-y[mask]).abs().mean()),
        'rmse':float((predicted[mask]-y[mask]).square().mean().sqrt())}
        for split, mask in [('train', train), ('holdout', ~train)]})
    backend = KernelLatencyBackend(WRAPPER, RUN/'results/latency_models')
    dimensions = {'qkv_proj':(6144,4096),'o_proj':(4096,4096),
                  'gate_up_proj':(24576,4096),'down_proj':(4096,12288)}
    alias = {'w4a16_ours':'marlin_nvfp4'}
    latency = {}
    for kind, (n,k) in dimensions.items():
        for phase, m in [('prefill',16384),('decode',8)]:
            actions=backend.predict(m,n,k)
            if m%32:
                actions['sparse_nvfp4']=backend.predict(32,n,k)['sparse_nvfp4']
            for method, action in actions.items():
                latency[kind,phase,method] = action
    names = list(json.loads((RUN / 'configs/policies/uniform_dense_bf16.json').read_text())['method_map'])
    modules = []
    for name in names:
        bucket, kind = int(name.split('.')[2])*4//36, name.rsplit('.',1)[-1]
        actions = []
        for pre in METHODS:
            for dec in METHODS:
                if scenario == 'prefill_only' and dec != pre:
                    continue
                quality, milliseconds, supported = 0.,0.,True
                for phase_index, (phase,method) in enumerate([('prefill',pre),('decode',dec)]):
                    if phase not in phases:
                        continue
                    local = 0. if method == 'dense_bf16' else errors[name,phase,method]
                    quality += local*float(fitted.coefficients[phase_index,METHODS.index(method),bucket,FUSED_TYPES.index(kind)])/fitted.feature_scale
                    action = latency[kind,phase,alias.get(method,method)]
                    supported &= action.supported and action.latency_ms is not None
                    if action.latency_ms is not None:
                        milliseconds += action.latency_ms*(63 if phase=='decode' else 1)
                if supported:
                    actions.append(Action(pre,dec,quality,milliseconds))
        modules.append(ModuleChoices(name,tuple(actions)))
    maximum = sum(max(action.quality_cost for action in module.actions) for module in modules)
    points, signatures = [],set()
    for index,budget in enumerate([0,.01,.02,.04,.07,.1,.15,.22,.32,.45,.65,.9,1.2,1.6,2.]):
        if budget < fitted.bias:
            continue
        solution = solve_budget(modules,budget-fitted.bias,max_quality=maximum)
        signature = tuple((action.prefill_method,action.decode_method) for action in solution.actions)
        if signature in signatures:
            continue
        signatures.add(signature)
        policy_id=f'daso_{scenario}_{index:02d}'
        policy={'policy_id':policy_id,'scenario':scenario,'default_prefill_method':'dense_bf16',
            'default_decode_method':'dense_bf16','modules_to_not_convert':['lm_head'],
            'method_map':{module.name:{'prefill_method':action.prefill_method,'decode_method':action.decode_method}
                          for module,action in zip(modules,solution.actions,strict=True)}}
        write_json(RUN/f'configs/policies/{policy_id}.json',policy)
        points.append({'policy_id':policy_id,'quality_budget':budget,
            'predicted_delta_nll':fitted.bias+solution.quality_cost,'predicted_linear_ms':solution.latency_ms})
    write_json(output/'predicted_points.json',points)
    print(points,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','calibrate','fit'])
    parser.add_argument('--scenario',choices=['prefill_only','prefill_decode'])
    parser.add_argument('--begin',type=int,default=0)
    parser.add_argument('--end',type=int,default=72)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='calibrate':calibrate(args.scenario,args.begin,args.end)
    else:fit(args.scenario)


if __name__=='__main__':main()
