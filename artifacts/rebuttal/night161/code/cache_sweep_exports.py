"""Retain task-owned exports for tonight's follow-ups using exact hard links."""
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT.parent/'runs'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def metadata(source,target):
    target.mkdir(parents=True,exist_ok=True)
    for path in source.iterdir():
        if path.is_file()and path.name!='model.safetensors'and path.stat().st_size<64*1024**2:
            shutil.copy2(path,target/path.name)


def main():
    with(ROOT.parent/'night151/export.lock').open('r')as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Exporter active; cache update deferred.');return
        added=[]
        for rid in ['034','035','036']:
            run=RUNS/rid;spec=read(run/'configs/model.json');ready=set()
            for stage in ['evaluation','generation']:
                ready.update(read(p)['policy']for p in(run/f'results/sweep_completed/{stage}').glob('*.json'))
            ready.update(read(p)['policy']for p in(run/'results/measurements').glob('sweep_formal*.json'))
            for pid in ready:
                source=Path(spec['temp'])/'checkpoints'/pid;binary=source/'model.safetensors'
                target=ROOT/'checkpoints'/rid/pid;linked=target/'model.safetensors'
                if not binary.exists()or linked.exists():continue
                policy=read(run/f'configs/policies/{pid}.json')['method_map']
                assert read(source/'config.json')['quantization_config']['method_map']==policy
                metadata(source,target);os.link(binary,linked)
                assert binary.stat().st_ino==linked.stat().st_ino and binary.stat().st_dev==linked.stat().st_dev
                record={'source':str(source),'cache':str(target),'run':rid,'policy':pid,'model_id':spec['model_id'],
                    'bytes':linked.stat().st_size,'inode':linked.stat().st_ino,'physical_copy_bytes':0,
                    'config_sha256':sha(target/'config.json'),'policy_sha256':sha(run/f'configs/policies/{pid}.json'),
                    'created_at_unix':time.time(),'reason':'Exact task-owned checkpoint retained for authorized nine-hour follow-up tests; parent may unlink its own path.'}
                (target/'night161_cache.json').write_text(json.dumps(record,indent=2)+'\n');added.append((rid,pid))
        # Same source model, recipe and activation scales; preseed only baseline
        # representations before036's formal phase, without changing any policy.
        run=RUNS/'036';spec=read(run/'configs/model.json')
        assert read(RUNS/'034/reference/source_model_fingerprint.json')==read(run/'reference/source_model_fingerprint.json')
        for name in ['common.py','eightbit.py','export_model.py']:
            assert sha(RUNS/'034/code'/name)==sha(run/'code'/name)
        assert sha(RUNS/'034/data/activation_scales.pt')==sha(run/'data/activation_scales.pt')
        seeded=[]
        for pid in ['uniform_dense_bf16','uniform_int8','uniform_sq_int8','uniform_fp8','phase_sq_int8_w4a16_ours']:
            source=ROOT/'checkpoints/034'/pid;binary=source/'model.safetensors'
            target=Path(spec['temp'])/'checkpoints'/pid
            if not binary.exists()or target.exists():continue
            assert read(source/'config.json')['quantization_config']['method_map']==read(run/f'configs/policies/{pid}.json')['method_map']
            metadata(source,target);os.link(binary,target/'model.safetensors')
            directory=run/'results/sweep_checkpoint_reuse';directory.mkdir(parents=True,exist_ok=True)
            (directory/(pid+'.json')).write_text(json.dumps({'source':str(source),'target':str(target),
                'exact_hardlink':True,'inode':binary.stat().st_ino,'physical_copy_bytes':0,
                'config_sha256':sha(target/'config.json'),'effective_map_equal':True},indent=2)+'\n');seeded.append(pid)
        print('Retained',added,'; seeded036 baselines',seeded,flush=True)
        # Follow-ups use the five anchor exports, not sealed exploratory maps.
        # Release only our last hard link after the source campaign is audited.
        released=[]
        for rid in ['034','035','036']:
            run=RUNS/rid;audit=run/'results/sweep_final_audit.json'
            if not audit.exists()or not read(audit)['verified']:continue
            spec=read(run/'configs/model.json')
            for item in read(run/'configs/sweep_points.json'):
                pid=item['policy'];path=ROOT/'checkpoints'/rid/pid/'model.safetensors'
                if not path.exists():continue
                own=Path(spec['temp'])/'checkpoints'/pid/'model.safetensors'
                assert not own.exists()and not path.is_symlink()
                stat=path.stat()
                if stat.st_nlink!=1:continue
                assert read(path.parent/'config.json')['quantization_config']['method_map']==read(run/f'configs/policies/{pid}.json')['method_map']
                record={'path':str(path),'bytes':stat.st_size,'inode':stat.st_ino,'source_audit_sha256':sha(audit),
                    'reason':'Own reproducible exploratory export, source campaign complete; queued follow-ups reference only five retained anchors. All code/logs/CSV/docs/configuration retained.',
                    'deleted':False,'created_at_unix':time.time()}
                directory=ROOT/'cleanup';directory.mkdir(exist_ok=True);note=directory/f'{rid}_{pid}_{time.time_ns()}.json'
                note.write_text(json.dumps(record,indent=2)+'\n');path.unlink();record['deleted']=True
                note.write_text(json.dumps(record,indent=2)+'\n');released.append((rid,pid,stat.st_size))
        if released:print('Released sealed exploratory binaries',released,flush=True)


if __name__=='__main__':main()
