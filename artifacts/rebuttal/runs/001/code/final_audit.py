"""Validate frozen model-evaluation outputs and their source provenance."""
import hashlib
import json
import math
from pathlib import Path

from report_results import RUN,points


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert read(RUN/'results/progress.json')['complete']
    generation_count=0;examples=0;nll_count=0
    inputs={}
    for task in ['cnn_dm_1000','dsum','IWSLT']:
        source=RUN/f'data/generation_{task}.jsonl'
        questions=[json.loads(line) for line in source.read_text().splitlines()]
        inputs[task]=(sha(source),questions)
    for point in points():
        policy=point['policy_id'];phase='prefill' if point['scenario']=='prefill_only' else 'decode'
        result=read(RUN/f'results/nll/{policy}_test_{phase}.json')
        assert abs(result['total_nll']/result['token_count']-result['avg_nll'])<1e-10
        nll_count+=1
        metadata=read(RUN/f'results/speed_common_gpu3/{policy}/metadata.json')
        assert metadata['kv_cache_memory_bytes']==3*1024**3
        assert metadata['peak_allocated_bytes']<=metadata['peak_reserved_bytes']
        cache=metadata['host_weight_cache_bytes']
        assert cache['pinned_bytes']<=cache['prefill_bytes']+cache['decode_bytes']
        if point['full'] and phase=='decode':
            for task in inputs:
                source_sha,questions=inputs[task]
                output=RUN/f'results/generation/{policy}/{task}'
                metric=read(output/'metrics_0000_end.json')
                answers=[json.loads(line) for line in (output/'answers_0000_end.jsonl').read_text().splitlines()]
                assert metric['input_sha256']==source_sha
                assert len(answers)==len(questions)==metric['samples']
                assert [a['question_id'] for a in answers]==[q['question_id'] for q in questions]
                assert len({a['question_id'] for a in answers})==len(answers)
                assert all(len(a['generated_token_ids'])<=256 for a in answers)
                assert all(a['prompt_tokens']==len(q['prompt_token_ids']) for a,q in zip(answers,questions))
                assert metric['empty_predictions']==sum(not a['prediction'].strip() for a in answers)
                assert metric['kv_cache_memory_bytes']==6*1024**3
                generation_count+=1;examples+=len(answers)
    for scenario in ['prefill_only','prefill_decode']:
        for index in range(72):
            label=read(RUN/f'results/calibration/{scenario}/cal_{scenario}_p{index:02d}.json')
            assert label['token_count']==(204700 if scenario=='prefill_only' else 6400)
            assert math.isfinite(label['avg_nll'])
    cache={};pairs=0
    for path in (RUN/'logs').glob('*.command.json'):
        record=read(path);version=record.get('code_version')
        if not version:continue
        for name,expected in record['code_hashes'].items():
            key=(version,name)
            if key not in cache:cache[key]=sha(RUN/'code_versions'/version/name)
            assert cache[key]==expected,(path,name)
            pairs+=1
    source_records=read(RUN/'reference/runtime_sources/index.json')['records']
    data_records=read(RUN/'data/likelihood_datasets/index.json')['records']
    for record in source_records+data_records:assert sha(RUN/record['saved'])==record['sha256']
    environment=read(RUN/'configs/environment.json')
    for path,expected in environment['backend_files'].items():assert sha(Path(path))==expected
    result={'passed':True,'test_nll_points':nll_count,'calibration_labels':144,
        'full_generation_files':generation_count,'generated_examples':examples,
        'command_snapshot_file_pairs':pairs,'unique_snapshot_files':len(cache),
        'reused_source_files':len(source_records),'frozen_likelihood_split_files':len(data_records),
        'generation_checks':['input hashes','exact question IDs and counts','prompt lengths','token limits','empty predictions','fixed KV'],
        'source_checks':['command snapshot hashes','reused source snapshot hashes','cached evaluation split hashes','initial vLLM hashes']}
    (RUN/'results/final_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
