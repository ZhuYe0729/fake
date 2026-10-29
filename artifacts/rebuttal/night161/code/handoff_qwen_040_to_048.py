"""Reuse identical completed exports without touching timing or model values."""
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

root=Path(__file__).resolve().parents[1];runs=root.parent/'runs';src_run=runs/'040';dst_run=runs/'048'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=read(src_run/'configs/model.json');target=read(dst_run/'configs/model.json')
assert source['model_path']==target['model_path']and source['canonical_root']==target['canonical_root']
assert sha(src_run/'data/activation_scales.pt')==sha(dst_run/'data/activation_scales.pt')
for name in ['common.py','eightbit.py','export_model.py','runtime_extensions.py']:
    assert(src_run/'code'/name).read_bytes()==(dst_run/'code'/name).read_bytes()
assert read(src_run/'results/quality_complete.json')['complete']
assert not list((dst_run/'results/generation').glob('*/*/*.json'))
with(root.parent/'night151/measurement.lock').open('a')as phase:
    fcntl.flock(phase,fcntl.LOCK_SH)
    with(root.parent/'night151/export.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for pid in read(src_run/'configs/design.json')['policies']:
            src=Path(source['temp'])/'checkpoints'/pid;dst=Path(target['temp'])/'checkpoints'/pid
            assert read(src/'config.json')['quantization_config']['method_map']==read(dst_run/f'configs/policies/{pid}.json')['method_map']
            dst.mkdir(parents=True,exist_ok=True)
            for p in src.iterdir():
                if p.is_file()and p.name!='model.safetensors':shutil.copy2(p,dst/p.name)
            binary=dst/'model.safetensors'
            if not binary.exists():os.link(src/'model.safetensors',binary)
            stat=binary.stat();assert stat.st_ino==(src/'model.safetensors').stat().st_ino
            receipt=dst_run/f'results/checkpoint_reuse/{pid}.json';receipt.parent.mkdir(parents=True,exist_ok=True)
            receipt.write_text(json.dumps({'source':str(src),'target':str(dst),'exact_hardlink':True,'inode':stat.st_ino,
                'bytes':stat.st_size,'physical_copy_bytes':0,'representation_source_run':'040','unix':time.time()},indent=2)+'\n')
            print('040→048 exact representation',pid,'inode',stat.st_ino,flush=True)
