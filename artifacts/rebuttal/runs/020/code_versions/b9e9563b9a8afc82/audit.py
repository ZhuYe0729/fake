"""Seal complete pruning states, experiments and retained failures."""
import hashlib

from common import RUN, VLLM, write_json
from launch import read


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        while chunk:=handle.read(8*1024**2):digest.update(chunk)
    return digest.hexdigest()


def main():
    assert read(RUN/'results/report_complete.json')['all_full_tasks']
    protected=read(RUN/'reference/protected_manifests.json')
    for rid,digest in protected.items():assert sha(RUN.parent/rid/'results/manifest.json')==digest
    for method in ['wanda','magnitude']:
        meta=read(RUN/f'results/canonical_{method}.json')
        assert len(meta['modules'])==224 and all(x['max_nonzero_per_4']<=2 and x['mask_zero_fraction']==.5 for x in meta['modules'])
        assert (RUN/f'data/canonical/{method}/model.pt').stat().st_size==meta['bytes']
        for scene in ['prefill_only','prefill_decode']:
            data=read(RUN/f'results/confirmation/uniform_{method}/{scene}.json')
            assert len(data['blocks'])==64 and data['quality_score_batch']==8 and data['admission']['whole_batch_fits']
    assert sha(VLLM/'vllm/vllm/_C.abi3.so')=='a6a469b427f36a61582e1fa99d8f83cf0fdd230e836e5eae7fe01514cd65fc2c'
    write_json(RUN/'results/worker.json',{'status':'complete','stage':'audited'})
    active=set();failures=[]
    for path in (RUN/'logs').glob('*.command.json'):
        completion=path.with_name(path.name.replace('.command.json','.completion.json'))
        if not completion.exists():active.add(path.with_name(path.name.replace('.command.json','.log')))
        elif read(completion)['returncode']:failures.append({'command':str(path.relative_to(RUN)),**read(completion)})
    write_json(RUN/'results/retained_failures.json',failures)
    files={}
    for path in sorted(RUN.rglob('*')):
        name=str(path.relative_to(RUN))
        if name in ['results/manifest.json','results/final_audit.json']:continue
        if path.is_symlink():files[name]={'symlink':str(path.resolve())}
        elif path.is_dir():continue
        elif path in active or path.name=='driver_controller.log':files[name]={'mutable_orchestration_log':True}
        else:files[name]={'sha256':sha(path),'bytes':path.stat().st_size}
    write_json(RUN/'results/manifest.json',{'files':files})
    write_json(RUN/'results/final_audit.json',{'complete':True,'verified':True,'methods':2,'rows':4,
        'files':len(files),'retained_failed_commands':len(failures),'manifest_sha256':sha(RUN/'results/manifest.json')})
    print('Pruning controls sealed',len(files),'files',flush=True)


if __name__=='__main__':main()
