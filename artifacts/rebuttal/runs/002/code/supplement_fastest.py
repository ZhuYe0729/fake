"""Add downstream coverage for speed winners outside the fixed budget anchors."""
import csv
import json
import os
import signal
import time
from pathlib import Path

from common import RUN, write_json
from experiment_queue import worker, release


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
    points=json.loads((RUN/'configs/evaluation_points.json').read_text())['points']
    chosen=[];additional=[]
    for family in ['daso8','daso5_measured']:
        for scenario in ['prefill_only','prefill_decode']:
            key='ttft_median_ms' if scenario=='prefill_only' else 'e2e_median_ms'
            values=[]
            for p in points:
                if p['family']!=family or p['scenario']!=scenario:continue
                row=next(csv.DictReader((RUN/f'results/speed_common_gpu3/{p["policy_id"]}/summary.csv').open()))
                values.append((float(row[key]),p))
            milliseconds,point=min(values,key=lambda pair:pair[0])
            chosen.append({**point,'measured_ms':milliseconds,'selection_metric':key})
            if not point['full']:additional.append(point)
    write_json(RUN/'configs/supplemental_fastest.json',{
        'rule':'after frozen-policy speed sweep: fastest TTFT / E2E per family; uses no NLL or downstream score',
        'original_budget_representatives_preserved':True,'winners':chosen,'additional':additional})
    state=RUN/'results/queue_supplemental.json'
    write_json(state,{'status':'waiting','pid':os.getpid(),'gpu':'3'})
    if not additional:
        write_json(state,{'status':'complete','pid':os.getpid(),'gpu':'3'});return
    parent=json.loads((RUN/'results/queue_phase_helper.json').read_text())['pid']
    command=Path(f'/proc/{parent}/cmdline').read_bytes().decode().replace('\0',' ')
    assert 'quality_queue.py --name phase_helper' in command
    os.kill(parent,signal.SIGSTOP)
    try:
        children=Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
        assert len(children)<=1
        if children:
            child=int(children[0])
            while Path(f'/proc/{child}/stat').exists():
                if Path(f'/proc/{child}/stat').read_text().split()[2]=='Z':break
                time.sleep(2)
        write_json(state,{'status':'running','pid':os.getpid(),'gpu':'3','parked_parent':parent})
        for point in additional:
            policy=point['policy_id']
            worker(policy,'export,tasks' if point['scenario']=='prefill_only' else 'export,generation')
            release(policy)
    finally:os.kill(parent,signal.SIGCONT)
    write_json(state,{'status':'complete','pid':os.getpid(),'gpu':'3'})


if __name__=='__main__':main()
