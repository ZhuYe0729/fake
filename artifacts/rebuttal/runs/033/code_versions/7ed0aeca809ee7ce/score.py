"""Fixed previously evaluated benchmark, with no new-blind-data claim."""
import argparse
import sys

from common import RUN, write_json
import score_nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--role', choices=['evaluation'], required=True)
    args = parser.parse_args()
    sys.argv = [__file__, '--checkpoint', args.checkpoint, '--policy', args.policy,
                '--split', 'confirmation', '--scenarios', 'prefill_decode']
    score_nll.main()
    path = RUN / f'results/confirmation/{args.policy}/prefill_decode.json'
    data = __import__('json').loads(path.read_text())
    data.update(data_role='fixed_benchmark_test', scope='Frozen-policy sensitivity on previously evaluated test benchmark; no adaptive selection or new blind confirmation claim.')
    write_json(path, data)


if __name__ == '__main__':
    main()
