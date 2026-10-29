"""Use idle speed GPU after anchors, then reload the finite primary queue."""
import json
import os
import signal
import sys
import time
from pathlib import Path

from baseline_worker import invoke
from common import RUN, write_json


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
    primary=json.loads((RUN/'results/queue_primary.json').read_text())['pid']
    if Path(f'/proc/{primary}/cmdline').exists():
        command=Path(f'/proc/{primary}/cmdline').read_bytes().decode().replace('\0',' ')
        assert 'experiment_queue.py --role primary' in command
        while not (RUN/'results/speed_complete/uniform_sq_fp8.json').exists():time.sleep(5)
        while Path(f'/proc/{primary}/task/{primary}/children').read_text().strip():time.sleep(2)
        os.kill(primary,signal.SIGTERM)
    else:
        assert (RUN/'results/primary_reload.json').exists()
    # The primary is now waiting for the 96 calibration labels. Reloading here
    # also picks up the predeclared SmoothQuant+Marlin phase controls.
    write_json(RUN/'results/primary_reload.json',{'old_pid':primary,
        'reason':'idle after six speed anchors; evaluate full 8bit baselines on GPU3 during calibration, then reload updated fixed phase controls'})
    for policy in ['uniform_sq_int8','uniform_fp8','uniform_sq_fp8']:
        invoke([sys.executable,str(RUN/'code/baseline_worker.py'),'--policies',policy,
            '--actions','export,nll,decode_nll,tasks,generation'],RUN/f'logs/early_{policy}_quality.log')
    # Anchor NLL is measured again in the reloaded queue, while task references
    # for the three old baselines stay linked to the frozen original results.
    invoke([sys.executable,str(RUN/'code/experiment_queue.py'),'--role','primary'],
           RUN/'logs/queue_primary_reloaded.log')


if __name__=='__main__':main()
