"""Read all prespecified results; no ranking affects measured policies."""
import csv
import json
from collections import Counter

import numpy as np

from common import RUN, write_json


def main():
    design=json.loads((RUN/'configs/design.json').read_text())
    frozen=json.loads((RUN/'configs/evaluation_points.json').read_text())
    workloads={w['id']:w for w in design['workloads']}
    rows=[];missing=[];usage={};rng=np.random.default_rng(143732)
    bootstrap=rng.integers(0,100,(2000,100));intervals={}
    def read(path):return json.loads(path.read_text()) if path.exists() else None
    for point in frozen['points']:
        w=workloads[point['workload']];pid=point['policy_id']
        speed=read(RUN/f'results/speed/{pid}/summary_{w["id"]}.json')
        anchor=read(RUN/f'results/speed/uniform_dense_bf16/summary_{w["id"]}.json')
        key=f'prefill_s{w["input"]}' if w['scenario']=='prefill_only' else w['id']
        quality=read(RUN/f'results/quality/{pid}/{key}.json')
        reference=read(RUN/f'results/quality/uniform_dense_bf16/{key}.json')
        for part,value in [('speed',speed),('BF16_speed',anchor),('quality',quality),('BF16_quality',reference)]:
            if value is None:missing.append({'label':point['label'],'workload':w['id'],'missing':part})
        row={**point,'scenario':w['scenario'],'batch':w['batch'],'input_tokens':w['input'],'output_tokens':w['output']}
        if speed:
            row.update({k:speed[k] for k in ['ttft_ms','e2e_ms','tpot_ms','output_tokens_per_second','peak_allocated_bytes','peak_reserved_bytes']})
            if anchor:
                row['ttft_speedup']=anchor['ttft_ms']/speed['ttft_ms'];row['e2e_speedup']=anchor['e2e_ms']/speed['e2e_ms']
        if quality:
            row.update(measured_nll=quality['avg_nll'],nll_tokens=quality['token_count'])
            if reference:
                delta=quality['avg_nll']-reference['avg_nll'];row['measured_delta_nll']=delta
                cache=(pid,key)
                if cache not in intervals:
                    differences=np.array([a['avg_nll']-b['avg_nll'] for a,b in zip(quality['blocks'],reference['blocks'],strict=True)])
                    intervals[cache]=np.quantile(differences[bootstrap].mean(1),[.025,.975]).tolist()
                row['delta_nll_ci_low'],row['delta_nll_ci_high']=intervals[cache]
                if 'quality_budget' in point:row['budget_violation']=max(0.,delta-point['quality_budget'])
        rows.append(row)
        if pid not in usage:
            policy=json.loads((RUN/f'configs/policies/{pid}.json').read_text())
            usage[pid]={phase:dict(Counter(p[phase+'_method'] for p in policy['method_map'].values())) for phase in ['prefill','decode']}
    fields=list(dict.fromkeys(k for r in rows for k in r))
    for filename in ['summary_current.csv']+(['summary_final.csv'] if not missing else []):
        with (RUN/'results'/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write_json(RUN/'results/candidate_usage.json',usage)
    state={'complete':not missing,'expected_points':len(rows),'unique_policies':len(frozen['policies']),
        'speed_policies_complete':len(list((RUN/'results/speed_complete').glob('*.json'))),
        'quality_policies_complete':len(list((RUN/'results/quality_complete').glob('*.json'))),
        'speed_measurements':len(list((RUN/'results/speed').glob('*/summary_*.json'))),
        'quality_measurements':len(list((RUN/'results/quality').glob('*/*.json'))),
        'missing':missing,'workers':{p.stem:json.loads(p.read_text()) for p in (RUN/'results').glob('worker_*.json')}}
    write_json(RUN/'results/progress.json',state)
    print(json.dumps({k:v for k,v in state.items() if k not in ['missing','workers']},ensure_ascii=False))
    print(json.dumps(state['workers'],ensure_ascii=False))


if __name__=='__main__':main()
