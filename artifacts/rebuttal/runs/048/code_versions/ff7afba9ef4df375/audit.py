"""Seal every new workload result without changing the original-model runs."""
import hashlib
import json

from common import RUN, VLLM, write_json
from launch import read


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert read(RUN/'results/report_complete.json')['rows']==66
    design=read(RUN/'configs/design.json')
    protected=read(RUN/'reference/protected_manifests.json')
    for rid,digest in protected.items():assert sha(RUN.parent/rid/'results/manifest.json')==digest
    for pid,spec in design['policies'].items():
        mapping=read(RUN/f'configs/policies/{pid}.json')['method_map']
        assert hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()==spec['map_sha256']
        for case in design['workloads']:
            result=read(RUN/f'results/quality/{pid}/{case["label"]}.json')
            assert result['quality_sha256']==case['quality_sha256']
            assert result['quality_batch']==case['batch'] and result['admission']['whole_batch_fits']
            assert len(result['blocks'])==64 and all(b['token_count']==case['output'] for b in result['blocks'])
    for campaign,order in enumerate(design['rounds']):
        for label in order:
            raw=read(RUN/f'results/speed/round{campaign}/{label}.json')
            assert raw['complete'] and len(raw['cases'])==6
            assert raw['gpu']=='0' and raw['placement']['cpu_affinity']==list(range(8))
            for case,expected in zip(raw['cases'],design['workloads'],strict=True):
                assert case['label']==expected['label'] and case['input_sha256']==expected['input_sha256']
                assert case['admission']['whole_batch_fits'] and len(case['samples'])==8
                for sample in case['samples']:
                    assert len(sample['output_token_ids'])==expected['batch']
                    assert all(len(ids)==expected['output'] for ids in sample['output_token_ids'])
    assert sha(VLLM/'vllm/vllm/_C.abi3.so')=='a6a469b427f36a61582e1fa99d8f83cf0fdd230e836e5eae7fe01514cd65fc2c'
    write_json(RUN/'results/worker.json',{'status':'complete','stage':'audited'})
    active=set()
    failures=[]
    for path in (RUN/'logs').glob('*.command.json'):
        completion=path.with_name(path.name.replace('.command.json','.completion.json'))
        if not completion.exists():active.add(path.with_name(path.name.replace('.command.json','.log')))
        elif read(completion)['returncode']:failures.append({'command':str(path.relative_to(RUN)),**read(completion)})
    write_json(RUN/'results/retained_failures.json',failures)
    entries={}
    for path in sorted(RUN.rglob('*')):
        name=str(path.relative_to(RUN))
        if name in ['results/manifest.json','results/final_audit.json']:continue
        if path.is_symlink():entries[name]={'symlink':str(path.resolve())}
        elif path.is_dir():continue
        elif path in active or path.name=='driver_controller.log':entries[name]={'mutable_orchestration_log':True}
        else:entries[name]={'sha256':sha(path),'bytes':path.stat().st_size}
    write_json(RUN/'results/manifest.json',{'files':entries})
    write_json(RUN/'results/final_audit.json',{'complete':True,'verified':True,'rows':66,
        'retained_failed_commands':len(failures),'files':len(entries),'manifest_sha256':sha(RUN/'results/manifest.json')})
    print('Sealed run019',len(entries),'files',flush=True)


if __name__=='__main__':main()
