"""Validate two proxy-budget backoffs using the unchanged learned models."""
import json
import shutil
import time

from common import RUN, write_json
from policy_runtime import digest
from quality_pool import run


def main():
    assert json.loads((RUN / 'results/control/superseded_scout_child_finished.json').read_text())['complete']
    assert not (RUN / 'configs/formal_points.json').exists()
    extension = RUN / 'configs/low_budget_refinement.json'
    plan = json.loads(extension.read_text())
    for point in plan['points']:
        assert digest(RUN / f"configs/policies/{point['policy_id']}.json") == point['policy_sha256']
        assert digest(RUN / f"data/actions_{point['scenario']}.json") == point['actions_sha256']
    write_json(RUN / 'results/continuation_status.json', {'stage': 'validate_daso_low_budget',
        'unix': time.time(), 'policies': len(plan['points']), 'scope': 'Same frozen quality models, additional validation only.'})
    started = time.time()
    paths = run([{'policy_id': point['policy_id'], 'scenario': point['scenario']} for point in plan['points']],
                'validation', 'low_budget_validation')
    stamp = str(time.time_ns())
    for name in ['validation_pool', 'selection_rules', 'latency_scout_shortlist']:
        path = RUN / f'configs/{name}.json'
        if path.exists():
            shutil.copy2(path, RUN / f'results/control/{name}_before_low_budget_{stamp}.json')
    closed_path = RUN / 'results/daso_validation_complete.json'
    shutil.copy2(closed_path, RUN / f'results/control/daso_validation_complete_before_low_budget_{stamp}.json')
    rules_path = RUN / 'configs/selection_rules.json'
    rules = json.loads(rules_path.read_text())
    rules['daso_candidate_pool'] += '; additional proxy budgets0.003/0.005 using the same frozen model after validation feedback'
    rules['scope'] = ('Original search and selection criteria frozen before initial DASO/HAWQ validation. '
        'Two lower proxy budgets added after strict-budget validation failures; original models and quality guard unchanged. '
        'All final maps freeze before formal test/downstream and independent three-round timing. All attempts retained.')
    write_json(rules_path, rules)
    pool_path = RUN / 'configs/validation_pool.json'
    pool = json.loads(pool_path.read_text())
    existing = {(point['policy_id'], point['scenario']) for point in pool['points']}
    pool['points'] += [{'policy_id': point['policy_id'], 'scenario': point['scenario']} for point in plan['points']
                       if (point['policy_id'], point['scenario']) not in existing]
    pool['selection_rules_sha256'] = digest(rules_path)
    pool['validation_feedback_extension'] = {'path': str(extension), 'sha256': digest(extension)}
    write_json(pool_path, pool)
    closed = json.loads(closed_path.read_text())
    closed['policies'] = sorted(set(closed['policies']) | {point['policy_id'] for point in plan['points']})
    closed['additional_validation_manifest'] = str(RUN / 'configs/low_budget_validation_quality_jobs.json')
    write_json(closed_path, closed)
    write_json(RUN / 'results/low_budget_validation_complete.json', {'complete': True,
        'extension_sha256': digest(extension), 'results': paths, 'elapsed_seconds': time.time() - started,
        'scope': 'Proxy budget backoff motivated by validation. No calibration-model refit or formal test/downstream feedback.'})
    print('Additional four policy validations complete', flush=True)


if __name__ == '__main__':
    main()
