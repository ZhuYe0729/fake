"""Finish baselines in parallel once calibration releases each assigned GPU."""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

from baseline_worker import invoke
from common import RUN, write_json


def wait(path):
    while not path.exists():time.sleep(5)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--slot',required=True);args=parser.parse_args()
    if args.slot=='primary':
        for dataset in ['cnn_dm_1000','dsum','IWSLT']:
            wait(RUN/f'results/generation/uniform_sq_int8/{dataset}/metrics_0000_end.json')
        while Path('/proc/1400479').exists():time.sleep(2)
        invoke([sys.executable,str(RUN/'code/experiment_queue.py'),'--role','primary'],
               RUN/'logs/queue_primary_rescheduled.log')
        return
    slot=int(args.slot);wait(RUN/f'results/calibration_slot{slot}_complete.json')
    state=json.loads((RUN/f'results/queue_cal{slot}.json').read_text());parent=state['pid']
    command=Path(f'/proc/{parent}/cmdline').read_bytes().decode().replace('\0',' ')
    assert f'experiment_queue.py --role cal{slot}' in command
    assert not (RUN/'configs/evaluation_points.json').exists(), 'baseline scheduling started after quality queues'
    assert not Path(f'/proc/{parent}/task/{parent}/children').read_text().strip()
    os.kill(parent,signal.SIGSTOP)
    policy=['uniform_sq_int8','uniform_fp8','uniform_sq_fp8'][slot]
    try:
        invoke([sys.executable,str(RUN/'code/baseline_worker.py'),'--policies',policy,
            '--actions','export,nll,decode_nll,tasks,generation'],RUN/f'logs/parallel_{policy}_quality.log')
    finally:os.kill(parent,signal.SIGCONT)
    write_json(RUN/f'results/parallel_baseline{slot}_complete.json',{'complete':True,'policy':policy})


if __name__=='__main__':main()
