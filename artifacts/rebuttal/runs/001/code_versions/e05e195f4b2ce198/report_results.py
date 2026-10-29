"""Source-derived progress and final comparison tables; missing evidence stays explicit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def points():
    protocol=read(RUN/'configs/final_protocol.json')
    records=[{'policy_id':policy,'family':'uniform','scenario':scenario,'full':True}
             for policy in protocol['main_uniform'] for scenario in ['prefill_only','prefill_decode']]
    for filename,family in [('hawq_marlin_int4_allocation.json','HAWQ-v3 allocator INT4'),
                            ('hawq_fp4_allocation.json','HAWQ-v3 allocator FP4 control')]:
        value=read(RUN/'results'/filename)
        if not value:continue
        selection=read(RUN/('configs/selected_hawq_policies.json' if 'marlin' in filename
                           else 'configs/selected_hawq_fp4_controls.json'))
        for point in value['policies']:
            selected=selection['policies'][point['scenario']] if selection else []
            selected=[selected] if isinstance(selected,str) else selected
            records.append({**point,'family':family,'full':point['policy_id'] in selected})
    for scenario in ['prefill_only','prefill_decode']:
        value=read(RUN/f'results/daso/{scenario}/predicted_points.json')
        selection=read(RUN/f'configs/selected_daso_{scenario}.json')
        for point in value or []:
            records.append({**point,'family':'DASO','scenario':scenario,
                'full':point['policy_id'] in (selection['policies'] if selection else [])})
    records.append({'policy_id':'phase_uniform_nvfp4_marlin','family':'phase-uniform hybrid',
                    'scenario':'prefill_decode','full':True})
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate-final',action='store_true')
    args=parser.parse_args()
    rows=[];missing=[]
    for item in points():
        policy,scenario=item['policy_id'],item['scenario']
        row={'policy_id':policy,'family':item['family'],'scenario':scenario,'full_downstream':item['full'],
             'quality_budget':item.get('quality_budget',''),'predicted_delta_nll':item.get('predicted_delta_nll',''),
             'latency_limit_linear_ms':item.get('latency_limit_ms','')}
        speed=RUN/f'results/speed_fixed_kv3gb/{policy}/summary.csv'
        if speed.exists():
            with speed.open() as handle:measured=next(csv.DictReader(handle))
            row.update(ttft_ms=float(measured['ttft_median_ms']),tpot_ms=float(measured['tpot_ms']),
                       e2e_ms=float(measured['e2e_median_ms']))
        else:missing.append(f'{policy}/speed')
        label='prefill' if scenario=='prefill_only' else 'decode'
        nll=read(RUN/f'results/nll/{policy}_test_{label}.json')
        baseline=read(RUN/f'results/nll/uniform_dense_bf16_test_{label}.json')
        if nll:
            row['measured_nll']=nll['avg_nll'];row['nll_tokens']=nll['token_count']
            if baseline:
                row['measured_delta_nll']=nll['avg_nll']-baseline['avg_nll']
                if 'quality_budget' in item:
                    row['budget_violation']=max(0.,row['measured_delta_nll']-item['quality_budget'])
        else:missing.append(f'{policy}/{label}_NLL')
        if item['full'] and scenario=='prefill_only':
            for task,key,column,multiplier in [('wikitext','word_perplexity,none','wikitext_word_ppl',1),
                ('winogrande','acc,none','winogrande_acc',100),('arc_easy','acc,none','arc_easy_acc',100),
                ('arc_challenge','acc_norm,none','arc_challenge_acc_norm',100),('mmlu','acc,none','mmlu_acc',100)]:
                result=read(RUN/f'results/tasks/{policy}/{task}/full/result.json')
                if result:
                    row[column]=result['metrics'][key]*multiplier
                    if result['limit'] is not None:raise ValueError(f'{policy}/{task}: diagnostic subset used as full result')
                else:missing.append(f'{policy}/{task}')
        elif item['full']:
            for task,column,metric,count in [('cnn_dm_1000','cnn_dm_rougeL','rougeL_percent',1000),
                ('dsum','dialogsum_rougeL','rougeL_percent',1500),('IWSLT','iwslt_sacrebleu','sacre_bleu',333)]:
                result=read(RUN/f'results/generation/{policy}/{task}/metrics_0000_end.json')
                if result:
                    if result['samples']!=count:raise ValueError(f'{policy}/{task}: incomplete generation set')
                    row[column]=result[metric]
                else:missing.append(f'{policy}/{task}')
        rows.append(row)
    for scenario in ['prefill_only','prefill_decode']:
        baseline=next((row for row in rows if row['policy_id']=='uniform_dense_bf16' and row['scenario']==scenario),{})
        for row in rows:
            if row['scenario']!=scenario:continue
            for column in ['ttft','tpot','e2e']:
                key=column+'_ms'
                if row.get(key,0)>0 and baseline.get(key,0)>0:row[column+'_speedup']=baseline[key]/row[key]
    required=['configs/selected_hawq_policies.json','configs/selected_hawq_fp4_controls.json',
              'configs/selected_daso_prefill_only.json','configs/selected_daso_prefill_decode.json']
    for relative in required:
        if not (RUN/relative).exists():missing.append(relative)
    queues={name:read(RUN/f'results/queue_{name}.json') for name in ['gpu0','gpu3','gpu4','gpu7']}
    for name,value in queues.items():
        if not value or value['status']!='complete':missing.append(f'queue/{name}')
    calibration={}
    for scenario in ['prefill_only','prefill_decode']:
        count=len(list((RUN/f'results/calibration/{scenario}').glob('cal_*.json')))
        # Phase traces are separate files and are not policy labels.
        count=len([p for p in (RUN/f'results/calibration/{scenario}').glob('cal_*.json') if '.phase_trace.' not in p.name])
        calibration[scenario]=count
        if count!=72:missing.append(f'calibration/{scenario}/{count}-of-72')
        if not (RUN/f'results/daso/{scenario}/quality_metrics.json').exists():missing.append(f'quality_model/{scenario}')
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (RUN/'results/summary_current.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    status={'complete':not missing,'calibration_labels':calibration,'rows':len(rows),
        'full_task_files':len(list((RUN/'results/tasks').glob('*/*/full/result.json'))),
        'generation_metric_files':len(list((RUN/'results/generation').glob('*/*/metrics_0000_end.json'))),
        'queues':queues,'missing':missing}
    (RUN/'results/progress.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps({key:status[key] for key in ['complete','calibration_labels','rows','full_task_files','generation_metric_files']},indent=2))
    if args.validate_final:
        if missing:raise SystemExit('Incomplete evidence: '+', '.join(missing[:20]))
        (RUN/'results/summary_final.csv').write_text((RUN/'results/summary_current.csv').read_text())
        inventory=[]
        excluded={'manifest.json','progress.json','summary_current.csv'}
        for path in sorted(RUN.rglob('*')):
            if not path.is_file() or path.name in excluded or '__pycache__' in path.parts:continue
            digest=hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
            inventory.append({'path':str(path.relative_to(RUN)),'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
        (RUN/'results/manifest.json').write_text(json.dumps({'files':inventory},indent=2)+'\n')


if __name__=='__main__':main()
