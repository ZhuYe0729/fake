"""Experiment-owned checkpoint lifecycle shared by closure and formal tasks."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import shutil
import sys

from calibration_queue import digest, invoke, snapshot_code
from common import MODEL, RUN, WRAPPER, write_json


@contextmanager
def checkpoint(policy, label):
    policy_path = RUN / f'configs/policies/{policy}.json'
    mapping = json.loads(policy_path.read_text())['method_map']
    temporary = any(method != 'dense_bf16' for pair in mapping.values() for method in pair.values())
    code_path, code_signature = snapshot_code()
    if temporary:
        root = Path('/dev/shm/daso_rebuttal_055') / label
        with (RUN / 'export.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert shutil.disk_usage('/dev/shm').free > 140 * 1024**3
            command = [sys.executable, RUN / 'code/export_model.py', '--model-path', MODEL,
                '--policy-json', policy_path, '--output-dir', root, '--cutlass-wrapper-path', WRAPPER]
            for method in ['sparse_bf16', 'sparse_nvfp4']:
                command += ['--canonical-' + method.replace('_', '-') + '-state', RUN / f'data/canonical/{method}/model.pt']
            invoke(command, label + '_export')
    else:
        root = RUN / 'exports/uniform_dense_bf16'
    evidence = RUN / f'results/export_identity/{label}'
    evidence.mkdir(parents=True)
    for source in root.glob('*.json'):
        shutil.copy2(source, evidence / source.name)
    binary = root / 'model.safetensors'
    identity = {'policy': policy, 'policy_sha256': digest(policy_path), 'binary_sha256': digest(binary),
        'binary_bytes': binary.stat().st_size, 'checkpoint': str(root), 'temporary': temporary,
        'code_snapshot': code_path, 'code_signature': code_signature,
        'canonical_identity_sha256': digest(RUN / 'configs/canonical_identity.json'),
        'source_model_identity_sha256': digest(RUN / 'results/model_identity.json'),
        'precompiled_vllm_identity_sha256': digest(RUN / 'configs/vllm_precompiled_binaries.json')}
    write_json(evidence / 'identity.json', identity)
    yield root
    if temporary:
        binary.unlink()
        write_json(evidence / 'temporary_weight_cleanup.json', {'path': str(binary),
            'sha256': identity['binary_sha256'], 'bytes': identity['binary_bytes'],
            'reason': 'Only this successful run-owned reproducible weight binary; all evidence retained.'})


def quality(policy, split, label, scenario=None):
    design = json.loads((RUN / 'configs/design.json').read_text())
    assert digest(RUN / f'configs/policies/{policy}.json') == design['policies'][policy]['map_sha256']
    allowed = design['policies'][policy]['scenario']
    cases = [case for case in design['workloads'] if (allowed is None or case['scenario'] == allowed)
             and (scenario is None or case['scenario'] == scenario)]
    paths = [RUN / f"results/quality/{split}/{policy}/{case['label']}.json" for case in cases]
    closures = [RUN / f"results/quality_closure/{split}/{policy}/{case['label']}.json" for case in cases]
    if all(path.exists() for path in paths) and all(path.exists() for path in closures):
        for path, closure in zip(paths, closures, strict=True):
            record = json.loads(path.read_text())
            assert record['complete'] and len(record['blocks']) == 100
            assert record['sample_sha256'] == design['quality_sha256'][split]
            assert json.loads(closure.read_text())['result_sha256'] == digest(path)
        return paths
    with checkpoint(policy, label) as root:
        command = [sys.executable, RUN / 'code/experiment.py', '--checkpoint', root, '--policy', policy,
            '--mode', 'quality', '--quality-split', split, '--tier', '1', '--label', label]
        if scenario is not None:
            command += ['--scenario', scenario]
        invoke(command, label)
    measurement = RUN / f'results/measurements/{label}.json'
    audit = json.loads(measurement.read_text())
    assert audit['complete'] and audit['policy'] == policy
    assert all(value['exact_equal'] for value in audit['weight_audit'].values())
    for path, closure in zip(paths, closures, strict=True):
        result = json.loads(path.read_text())
        assert result['complete'] and len(result['blocks']) == 100
        assert result['sample_sha256'] == design['quality_sha256'][split]
        write_json(closure, {'complete': True, 'result_sha256': digest(path),
            'measurement': str(measurement), 'measurement_sha256': digest(measurement)})
    return paths
