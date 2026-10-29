"""Adapt completed uniform measurements to the stable main-table schema."""
import argparse
import sys
from pathlib import Path

from common import RUN, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--record-id', required=True)
    parser.add_argument('--outputs', type=int, nargs='+', required=True)
    args = parser.parse_args()
    target = RUN / f'results/speed/node0/round{args.round}/{args.record_id}.json'
    assert not target.exists()
    label = f'formal_round{args.round}_{args.record_id}'
    source = RUN / f'results/measurements/{label}.json'
    if not source.exists():
        import benchmark
        sys.argv = [str(RUN / 'code/benchmark.py'), '--checkpoint', str(args.checkpoint),
                    '--policy', args.policy, '--mode', 'arena_graph', '--label', label,
                    '--outputs', *map(str, args.outputs), '--iterations', '8']
        benchmark.main()
    import json
    data = json.loads(source.read_text())
    data.update(round=args.round, seed=144983, node=0, gpu='0',
                first_token_observer='engine_core_before_FINAL_ONLY_filter',
                original_measurement=str(source))
    for case in data['cases']:
        case['scenario'] = 'prefill_only' if case['output'] == 1 else 'prefill_decode'
        case['median_ms'] = case['medians']['e2e_ms']
        for row in case['samples']:
            row['ttft_observed_ms'] = row['ttft_ms']
            row['decode_remainder_ms'] = row['e2e_ms'] - row['ttft_ms']
            row['generated_tokens'] = 8 * case['output']
    write_json(target, data)


if __name__ == '__main__':
    main()
