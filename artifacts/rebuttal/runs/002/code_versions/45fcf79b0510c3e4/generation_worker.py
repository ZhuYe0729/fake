import argparse
import sys

from baseline_worker import TEMP, invoke
from common import RUN


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',required=True)
    args=parser.parse_args()
    for dataset in ['cnn_dm_1000','dsum','IWSLT']:
        target=RUN/f'results/generation/{args.policy}/{dataset}/metrics_0000_end.json'
        if target.exists():continue
        invoke([sys.executable,str(RUN/'code/generate_tasks.py'),'--policy',args.policy,
            '--checkpoint',str(TEMP/'checkpoints'/args.policy),'--dataset',dataset],
            RUN/f'logs/{args.policy}_generation_{dataset}.log')


if __name__=='__main__':main()
