"""Validate frozen policy/measurement coverage and source lineage."""
import csv
import hashlib
import json
import math
from pathlib import Path

from common import RUN, write_json


def read(path):return json.loads(path.read_text())


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def main():
    assert read(RUN/'results/kernel_audit.json')['passed']
    assert read(RUN/'results/action_profile_final.json')['rows']==64
    assert read(RUN/'results/frozen_task_initialization.json')['passed']
    assert read(RUN/'results/dp_scale_control_audit.json')['passed']
    assert read(RUN/'results/bf16_anchor_recheck.json')['passed']
    assert read(RUN/'results/uniform_grid_audit.json')['passed']
    for row in read(RUN/'data/likelihood_auxiliary_datasets/index.json')['records']:
        assert hashlib.sha256((RUN/row['saved']).read_bytes()).hexdigest()==row['sha256']
    design=read(RUN/'configs/extended_design.json')
    methods=set(design['methods']);assert len(methods)==7
    storage={}
    for policy,dtype in [('uniform_sq_int8','I8'),('uniform_fp8','F8_E4M3'),('uniform_sq_fp8','F8_E4M3')]:
        footprint=read(RUN/f'results/export_metadata/{policy}/tensor_footprint.json')
        packed=[v for v in footprint['tensors'] if v['name'].endswith('ph_prefill_packed_weight')]
        assert len(packed)==144 and all(v['dtype']==dtype for v in packed)
        storage[policy]={'linear_weights':len(packed),'dtype':dtype}
    counts={};snapshots={}
    for scenario in ['prefill_only','prefill_decode']:
        manifest=read(RUN/f'configs/calibration_{scenario}.json')
        assert len(manifest)==120 and sum(e['split']=='train' for e in manifest)==90
        for entry in manifest:
            label=read(RUN/f'results/calibration/{scenario}/{entry["policy_id"]}.json')
            assert label['token_count']==(204700 if scenario=='prefill_only' else 6400)
            assert math.isfinite(label['avg_nll'])
        a=read(RUN/f'results/daso8/{scenario}/quality_model.json')
        b=read(RUN/f'results/daso5_measured/{scenario}/quality_model.json')
        assert a==b,'candidate comparison changed quality fit'
        counts[scenario]=len(manifest)
    points=read(RUN/'configs/evaluation_points.json')['points']
    speed_policies=set(read(RUN/'configs/final_protocol.json')['main_uniform'])|{p['policy_id'] for p in points}
    speed_iterations=0
    for policy in speed_policies:
        with (RUN/f'results/speed_common_gpu3/{policy}/iterations.csv').open() as f:iterations=list(csv.DictReader(f))
        assert len(iterations)==14
        for row in iterations:
            assert int(row['batch'])==8 and int(row['input_seq'])==2048
            assert int(row['generated_tokens'])==8*int(row['output_seq'])
        speed_iterations+=len(iterations)
    with (RUN/'data/local_errors.csv').open() as f:
        errors={(r['module'],r['phase'],r['method']):float(r['output_rel_mse']) for r in csv.DictReader(f)}
    with (RUN/'results/action_profile.csv').open() as f:
        costs={(r['module_type'],'prefill' if int(r['m'])==16384 else 'decode',r['method']):float(r['latency_ms']) for r in csv.DictReader(f)}
    kinds=['qkv_proj','o_proj','gate_up_proj','down_proj']
    for item in points:
        policy=read(RUN/f'configs/policies/{item["policy_id"]}.json')
        assert len(policy['method_map'])==144
        for pair in policy['method_map'].values():
            assert set(pair.values())<=methods
        if item['family']=='daso5_measured':
            assert not any(v in ['fp8','sq_int8'] for pair in policy['method_map'].values() for v in pair.values())
        if 'quality_budget' in item:
            assert item['predicted_delta_nll']<=item['quality_budget']+1.e-10
            fitted=read(RUN/f'results/{item["family"]}/{item["scenario"]}/quality_model.json')
            quality=fitted['bias'];latency=0.
            for name,pair in policy['method_map'].items():
                bucket=int(name.split('.')[2])*4//36;kind=name.rsplit('.',1)[-1]
                for pi,phase in enumerate(['prefill','decode']):
                    if item['scenario']=='prefill_only' and phase=='decode':continue
                    method=pair[phase+'_method']
                    local=0. if method=='dense_bf16' else errors[name,phase,method]
                    quality+=local*fitted['coefficients'][pi][design['methods'].index(method)][bucket][kinds.index(kind)]/fitted['feature_scale']
                    latency+=costs[kind,phase,method]*(63 if phase=='decode' else 1)
            assert abs(quality-item['predicted_delta_nll'])<1.e-10
            assert abs(latency-item['predicted_linear_ms'])<1.e-8
    source=read(RUN/'reference/sources.json')
    assert source['run001_manifest_sha256']==hashlib.sha256((RUN.parent/'001/results/manifest.json').read_bytes()).hexdigest()
    inherited=read(RUN.parent/'001/results/manifest.json')['files']
    for record in inherited:
        assert digest(RUN.parent/'001'/record['path'])==record['sha256'],record['path']
    runtime=read(RUN.parent/'001/reference/runtime_sources/index.json')['records']
    for record in runtime:
        assert digest(Path(record['source']))==record['sha256'],record['source']
    eightbit_runtime=read(RUN/'reference/fp8_runtime_sources/index.json')['records']
    for record in eightbit_runtime:
        assert digest(Path(record['source']))==record['sha256']
        assert digest(RUN/record['saved'])==record['sha256']
    for command in (RUN/'logs').glob('*.command.json'):
        value=read(command);version=value['code_version']
        for name,sha in value['code_hashes'].items():
            path=RUN/'code_versions'/version/name
            key=str(path)
            if key not in snapshots:snapshots[key]=hashlib.sha256(path.read_bytes()).hexdigest()
            assert snapshots[key]==sha
    generations=0;examples=0
    for metric in (RUN/'results/generation').glob('*/*/metrics_0000_end.json'):
        if metric.parent.parent.is_symlink():continue
        value=read(metric)
        inputs=[json.loads(line) for line in (RUN/f'data/generation_{value["dataset"]}.jsonl').read_text().splitlines()]
        answers=[json.loads(line) for line in (metric.parent/'answers_0000_end.jsonl').read_text().splitlines()]
        assert len(inputs)==len(answers)==value['samples']
        assert [x['question_id'] for x in inputs]==[x['question_id'] for x in answers]
        assert value['kv_cache_memory_bytes']==6*1024**3
        assert all(len(x['generated_token_ids'])<=256 for x in answers)
        generations+=1;examples+=len(answers)
    progress=read(RUN/'results/progress.json');assert progress['complete'],progress['missing']
    with (RUN/'results/summary_latency_nll.csv').open() as f:frozen=list(csv.DictReader(f))
    with (RUN/'results/summary_final.csv').open() as f:
        final={(r['policy_id'],r['scenario']):r for r in csv.DictReader(f)}
    for row in frozen:
        actual=final[row['policy_id'],row['scenario']]
        assert all(actual[key]==value for key,value in row.items()),row['policy_id']
    write_json(RUN/'results/final_audit.json',{'passed':True,'calibration_labels':counts,
        'policy_rows':len(points),'new_generation_files':generations,'new_generation_examples':examples,
        'unique_snapshot_files':len(snapshots),'run001_manifest_unchanged':True,
        'control_quality_fit_identical':True,'baseline_weight_storage':storage,
        'run001_files_verified':len(inherited),'runtime_sources_unchanged':len(runtime),
        'additional_eightbit_runtime_sources':len(eightbit_runtime),
        'measured_speed_policies':len(speed_policies),'speed_iterations':speed_iterations,
        'latency_nll_snapshot_unchanged':True})
    print('Final audit passed',flush=True)


if __name__=='__main__':main()
