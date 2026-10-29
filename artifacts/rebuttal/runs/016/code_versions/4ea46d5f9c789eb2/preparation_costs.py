"""Keep training/validation preparation separate from steady-state inference."""
import csv
from collections import defaultdict

from common import RUN, write_json
from launch import read


def main():
    rows=[]
    for path in sorted((RUN/'logs').glob('*.command.json')):
        command=read(path)
        done=path.with_name(path.name.replace('.command.json','.completion.json'))
        if not done.exists():continue
        script=next((x.rsplit('/',1)[-1] for x in command['command'] if x.endswith('.py')),'')
        if script not in ['collect_int8_errors.py','score_nll.py','export_model.py','fit_quality.py','allocate_suite.py','benchmark.py']:
            continue
        args=command['command']
        stage=(args[args.index('--split')+1]+'_nll' if script=='score_nll.py' else
            'calibration_speed' if script=='benchmark.py' and any(x.startswith('calibration_') for x in args) else script)
        result=read(done)
        rows.append({'stage':stage,'seconds':result['seconds'],'returncode':result['returncode'],
            'gpu':command['gpu'],'command':str(path.relative_to(RUN))})
    grouped=defaultdict(list)
    for row in rows:grouped[row['stage']].append(row)
    summary=[{'stage':stage,'successful_processes':sum(r['returncode']==0 for r in items),
        'sum_success_process_seconds':sum(r['seconds'] for r in items if r['returncode']==0),
        'failed_processes':sum(r['returncode']!=0 for r in items)} for stage,items in grouped.items()]
    for name,items in [('preparation_processes.csv',rows),('preparation_summary.csv',summary)]:
        with (RUN/'results'/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    write_json(RUN/'results/preparation_scope.json',{
        'scope':'Process sums are not elapsed wall time; CPU/GPU jobs can overlap. Repeated exports are experiment overhead, not a per-request cost.',
        'inherited_labels_per_scenario':120,'new_labels_per_scenario':24,
        'old_preparation_costs':'004/results/preparation_costs.json',
        'cached_hessian_seconds':1765.6385020969901,
        'hessian_scope':'Cached original train Hessian; actual fused candidate weight perturbations are recomputed in the shared preparation.'})


if __name__ == '__main__':main()
