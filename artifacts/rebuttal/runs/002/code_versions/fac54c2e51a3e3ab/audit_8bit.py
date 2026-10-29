"""Validate frozen policy/measurement coverage and source lineage."""
import hashlib
import json
import math
from pathlib import Path

from common import RUN, write_json


def read(path):return json.loads(path.read_text())


def main():
    assert read(RUN/'results/kernel_audit.json')['passed']
    assert read(RUN/'results/action_profile_final.json')['rows']==64
    design=read(RUN/'configs/extended_design.json')
    methods=set(design['methods']);assert len(methods)==7
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
    for item in points:
        policy=read(RUN/f'configs/policies/{item["policy_id"]}.json')
        assert len(policy['method_map'])==144
        for pair in policy['method_map'].values():
            assert set(pair.values())<=methods
        if item['family']=='daso5_measured':
            assert not any(v in ['fp8','sq_int8'] for pair in policy['method_map'].values() for v in pair.values())
        if 'quality_budget' in item:
            assert item['predicted_delta_nll']<=item['quality_budget']+1.e-10
    source=read(RUN/'reference/sources.json')
    assert source['run001_manifest_sha256']==hashlib.sha256((RUN.parent/'001/results/manifest.json').read_bytes()).hexdigest()
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
    write_json(RUN/'results/final_audit.json',{'passed':True,'calibration_labels':counts,
        'policy_rows':len(points),'new_generation_files':generations,'new_generation_examples':examples,
        'unique_snapshot_files':len(snapshots),'run001_manifest_unchanged':True,
        'control_quality_fit_identical':True})
    print('Final audit passed',flush=True)


if __name__=='__main__':main()
