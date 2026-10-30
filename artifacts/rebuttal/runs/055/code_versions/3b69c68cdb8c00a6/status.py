"""Summarize persisted evidence; never infer completion from a live process."""
import json
from pathlib import Path
import time

RUN = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((RUN / 'configs/calibration_manifest.json').read_text())
    complete = []
    for entry in manifest['entries']:
        path = RUN / f"results/calibration_complete/{entry['policy_id']}.json"
        if path.exists() and json.loads(path.read_text()).get('complete'):
            complete.append(entry['policy_id'])
    trace = RUN / 'results/hessian_trace_p16_b4_s256.json'
    hessian = json.loads(trace.read_text()) if trace.exists() else {}
    value = {'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'formal_methods': ['BF16', 'DASO', 'HAWQ-V3 adaptation'],
        'bf16_capacity_complete': (RUN / 'results/bf16_vllm_feasibility.json').exists(),
        'feature_grid_complete': (RUN / 'results/features_complete.json').exists(),
        'hardware_profile_complete': (RUN / 'results/profile_complete.json').exists(),
        'calibration_complete': len(complete), 'calibration_total': len(manifest['entries']),
        'completed_calibration_policies': complete,
        'primary_hessian_complete': hessian.get('complete', False),
        'primary_hessian_projections': len(hessian.get('records', [])),
        'primary_hessian_target_projections': 448,
        'task_datasets_frozen': (RUN / 'configs/likelihood_datasets.json').exists(),
        'formal_points_frozen': (RUN / 'configs/formal_points.json').exists(),
        'formal_speed_complete': (RUN / 'results/formal_speed_complete.json').exists(),
        'all_downstream_tasks_complete': (RUN / 'results/all_downstream_complete.json').exists()}
    (RUN / 'results/status.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
