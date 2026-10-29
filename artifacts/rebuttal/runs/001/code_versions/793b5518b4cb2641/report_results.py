"""Source-derived progress and final comparison tables; missing evidence stays explicit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from checkpoint_evidence import inspect_checkpoint

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
    for path in Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints').glob('*/model.safetensors'):
        destination=RUN/'results/export_metadata'/path.parent.name
        if not (destination/'tensor_footprint.json').exists():inspect_checkpoint(path.parent,destination)
    memory={};quality_memory={}
    for path in sorted((RUN/'results').glob('last_memory_*.json'),key=lambda p:p.stat().st_mtime):
        value=read(path);arguments=value.get('arguments',[])
        if value.get('mode') in ['prefill_nll','decode_nll'] and '--output' in arguments:
            quality_memory[arguments[arguments.index('--output')+1]]=value
        if value.get('mode')!='speed' or '--output-dir' not in arguments:continue
        output=Path(arguments[arguments.index('--output-dir')+1])
        if output.parent.name not in ['speed_fixed_kv3gb','speed_common_gpu3']:continue
        memory[output.parent.name,output.name]=value
    rows=[];missing=[]
    for item in points():
        policy,scenario=item['policy_id'],item['scenario']
        row={'policy_id':policy,'family':item['family'],'scenario':scenario,'full_downstream':item['full'],
             'quality_budget':item.get('quality_budget',''),'predicted_delta_nll':item.get('predicted_delta_nll',''),
             'predicted_linear_ms':item.get('predicted_linear_ms',''),
             'latency_limit_linear_ms':item.get('latency_limit_ms','')}
        speed=RUN/f'results/speed_common_gpu3/{policy}/summary.csv'
        primary=speed.exists()
        if not primary:
            missing.append(f'{policy}/common_gpu3_speed')
            speed=RUN/f'results/speed_fixed_kv3gb/{policy}/summary.csv'
        if speed.exists():
            with speed.open() as handle:measured=next(csv.DictReader(handle))
            with (speed.parent/'iterations.csv').open() as handle:iterations=list(csv.DictReader(handle))
            for phase in ['ttft','main']:
                if sum(value['phase']==phase and value['warmup']=='False' for value in iterations)!=5:
                    raise ValueError(f'{policy}: incomplete speed measurements')
            row.update(ttft_ms=float(measured['ttft_median_ms']),tpot_ms=float(measured['tpot_ms']),
                       e2e_ms=float(measured['e2e_median_ms']))
            row['speed_source']='common_gpu3' if primary else 'provisional_cross_gpu'
            memory_key=speed.parent.parent.name,policy
            if memory_key in memory:
                value=memory[memory_key]
                row.update(peak_allocated_gib=value['peak_allocated_bytes']/2**30,
                           peak_reserved_gib=value['peak_reserved_bytes']/2**30,
                           fixed_kv_gib=value['kv_cache_memory_bytes']/2**30)
                if 'host_weight_cache_bytes' in value:
                    cache=value['host_weight_cache_bytes']
                    row.update(host_prefill_weight_cache_gib=cache['prefill_bytes']/2**30,
                        host_decode_weight_cache_gib=cache['decode_bytes']/2**30,
                        pinned_host_weight_cache_gib=cache['pinned_bytes']/2**30,
                        peak_host_rss_gib=value['peak_host_rss_bytes']/2**30)
            else:missing.append(f'{policy}/speed_memory_evidence')
            footprint=read(RUN/f'results/export_metadata/{policy}/tensor_footprint.json')
            if not footprint:footprint=read(RUN/f'results/common_speed_exports/{policy}/tensor_footprint.json')
            if footprint:row['export_gib']=footprint['file_bytes']/2**30
        else:missing.append(f'{policy}/speed')
        label='prefill' if scenario=='prefill_only' else 'decode'
        nll=read(RUN/f'results/nll/{policy}_test_{label}.json')
        baseline=read(RUN/f'results/nll/uniform_dense_bf16_test_{label}.json')
        if nll:
            expected=204700 if label=='prefill' else 6400
            if nll['token_count']!=expected or not math.isfinite(nll['avg_nll']):
                raise ValueError(f'{policy}/{label}: incomplete or nonfinite NLL')
            measurement=quality_memory.get(str(RUN/f'results/nll/{policy}_test_{label}.json'))
            if not measurement or measurement.get('kv_cache_memory_bytes')!=3*1024**3:
                missing.append(f'{policy}/{label}_fixed_KV_evidence')
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
                    if task=='mmlu' and sum(v.get('effective',0) for v in result['raw_lm_eval']['n-samples'].values())!=14042:
                        raise ValueError(f'{policy}: incomplete MMLU')
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
              'configs/selected_daso_prefill_only.json','configs/selected_daso_prefill_decode.json',
              'configs/common_card_speed.json']
    for relative in required:
        if not (RUN/relative).exists():missing.append(relative)
    audit=read(RUN/'results/selected_policy_quality_audit.json')
    if not audit or len(audit['rows'])!=4:missing.append('selected_policy_calibration_audit/4-policies')
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
