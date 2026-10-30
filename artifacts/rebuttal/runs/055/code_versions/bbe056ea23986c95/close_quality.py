"""Run frozen HAWQ validation or formal test NLL, without selecting by test."""
import argparse
import json
import time

from common import RUN, write_json
from policy_runtime import quality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['hawq-validation', 'test'])
    args = parser.parse_args()
    if args.stage == 'hawq-validation':
        points = json.loads((RUN / 'results/hawq_allocation.json').read_text())['points']
        split = 'validation'
    else:
        points = json.loads((RUN / 'configs/formal_points.json').read_text())['unique_cases']
        split = 'test'
    records = []
    for point in points:
        policy, scenario = point['policy_id'], point['scenario']
        paths = quality(policy, split, f'{split}_{policy}_{time.time_ns()}', scenario)
        records += [str(path) for path in paths]
    write_json(RUN / f'results/{args.stage}_complete.json', {'complete': True, 'results': records,
        'scope': 'All frozen requested cases completed; test values are never read by the selector.'})


if __name__ == '__main__':
    main()
