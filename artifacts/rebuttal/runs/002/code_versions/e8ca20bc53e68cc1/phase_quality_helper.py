"""Use the speed GPU for the two fixed phase controls after timing is finished."""
import json
import os
import time

from common import RUN, write_json
from experiment_queue import worker, release


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
    state=RUN/'results/queue_phase_helper.json'
    write_json(state,{'status':'waiting','pid':os.getpid(),'gpu':'3'})
    while json.loads((RUN/'results/queue_primary.json').read_text())['status']!='complete':time.sleep(5)
    write_json(state,{'status':'running','pid':os.getpid(),'gpu':'3'})
    for policy in ['phase_sq_int8_w4a16_ours','phase_fp8_w4a16_ours']:
        worker(policy,'export,decode_nll,generation')
        release(policy)
    write_json(state,{'status':'complete','pid':os.getpid(),'gpu':'3'})


if __name__=='__main__':main()
