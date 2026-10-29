"""Migrate at completed subprocess boundaries, preserving every active result."""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

from common import RUN, write_json
from experiment_queue import release


def command(pid):return Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')


def children(pid):return Path(f'/proc/{pid}/task/{pid}/children').read_text().split()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--slot',type=int,required=True);args=parser.parse_args()
    slot=args.slot;name=f'cal{slot}' if slot<3 else 'phase_helper'
    state=json.loads((RUN/f'results/queue_{name}.json').read_text());parent=state['pid']
    expected=f'experiment_queue.py --role cal{slot}' if slot<3 else 'phase_quality_helper.py'
    assert expected in command(parent)
    coordinator=parent
    if slot in [1,2]:
        matches=[]
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():continue
            try:value=command(int(entry.name))
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            if value.split() and 'python' in value.split()[0] and f'residual_queue.py --slot {slot}' in value:matches.append(int(entry.name))
        assert len(matches)==1,matches
        coordinator=matches[0]
    os.kill(coordinator,signal.SIGSTOP)
    if parent!=coordinator:os.kill(parent,signal.SIGSTOP)
    active=children(coordinator)
    assert len(active)<=1,active
    policies=[]
    if active:
        child=int(active[0]);value=command(child)
        assert 'baseline_worker.py' in value,value
        policies=value.split('--policies ',1)[1].split()[0].split(',')
        write_json(RUN/f'results/quality_migration_slot{slot}.json',{
            'stage':'waiting_for_existing_policy','parent':parent,'coordinator':coordinator,
            'child':child,'policies':policies,'rule':'finish existing child; never interrupt its export, NLL, task or generation'})
        while Path(f'/proc/{child}/stat').exists():
            if Path(f'/proc/{child}/stat').read_text().split()[2]=='Z':break
            time.sleep(2)
        for policy in policies:release(policy)
    for pid in set([coordinator,parent]):os.kill(pid,signal.SIGKILL)
    if slot<3:
        policy=['uniform_sq_int8','uniform_fp8','uniform_sq_fp8'][slot]
        assert all((RUN/f'results/tasks/{policy}/{t}/full/result.json').exists()
                   for t in ['wikitext','winogrande','arc_easy','arc_challenge','mmlu'])
        assert all((RUN/f'results/generation/{policy}/{t}/metrics_0000_end.json').exists()
                   for t in ['cnn_dm_1000','dsum','IWSLT'])
        write_json(RUN/f'results/parallel_baseline{slot}_complete.json',{'complete':True,'policy':policy})
    write_json(RUN/f'results/quality_migration_slot{slot}.json',{
        'stage':'completed','old_parent':parent,'old_coordinator':coordinator,
        'finished_policies':policies,'replacement':'quality_queue.py','gpu':os.environ['CUDA_VISIBLE_DEVICES']})
    extra=['--wait-primary'] if slot==3 else []
    os.execv(sys.executable,[sys.executable,str(RUN/'code/quality_queue.py'),'--name',name,*extra])


if __name__=='__main__':main()
