"""Use the unchanged Graph scorer; label calibration explicitly as training."""
import argparse
import sys

from common import RUN, write_json
import score_nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--role', choices=['calibration', 'validation', 'confirmation'], required=True)
    args = parser.parse_args()
    mode = 'validation' if args.role == 'calibration' else args.role
    score_nll.RUN = RUN / 'calibration' if args.role == 'calibration' else RUN
    sys.argv = [__file__, '--checkpoint', args.checkpoint, '--policy', args.policy,
                '--split', mode, '--scenarios', 'prefill_decode']
    score_nll.main()
    path = score_nll.RUN / f'results/{mode}/{args.policy}/prefill_decode.json'
    data = __import__('json').loads(path.read_text())
    data['data_role'] = args.role
    data['scope'] = ('Surrogate calibration on original training windows; policy holdouts separate; not final validation.'
        if args.role == 'calibration' else 'Prospectively frozen unused internal windows; no confirmation-based model/policy selection.')
    write_json(path, data)


if __name__ == '__main__':
    main()
