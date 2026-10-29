"""Preserve exact primary exports on the large project XFS mount."""
import hashlib
import os
import shutil
import time
from pathlib import Path


def preserve(path,run,write):
    primary=run/'configs/primary_policies.json'
    if not primary.exists() or path.parent.name not in __import__('json').loads(primary.read_text())['policies']:
        return False
    if path.is_symlink():return True
    root=run.parents[1]/'night151/primary_weight_cache'/run.name
    root.mkdir(parents=True,exist_ok=True)
    size=path.stat().st_size
    assert root.stat().st_dev!=Path('/').stat().st_dev, 'Never cache on the nearly full root overlay.'
    if shutil.disk_usage(root).free<size+75*1024**3:return False
    destination=root/(path.parent.name+'.safetensors')
    assert not destination.exists(), 'Inspect a prior interrupted cache attempt before overwriting anything.'
    temporary=destination.with_name(destination.name+'.partial_'+str(time.time_ns()))
    started=time.perf_counter();digest=hashlib.sha256();stat=path.stat()
    with path.open('rb') as source,temporary.open('xb') as target:
        while chunk:=source.read(8*1024**2):digest.update(chunk);target.write(chunk)
        target.flush();os.fsync(target.fileno())
    copied=hashlib.sha256()
    with temporary.open('rb') as handle:
        while chunk:=handle.read(8*1024**2):copied.update(chunk)
    assert copied.hexdigest()==digest.hexdigest() and temporary.stat().st_size==size
    assert path.stat().st_mtime_ns==stat.st_mtime_ns
    temporary.rename(destination)
    policy=run/f'configs/policies/{path.parent.name}.json'
    write(run/f'results/primary_weight_cache/{path.parent.name}.json',{
        'source':str(path),'cache':str(destination),'bytes':size,'sha256':digest.hexdigest(),
        'policy_sha256':hashlib.sha256(policy.read_bytes()).hexdigest(),'copy_and_verify_seconds':time.perf_counter()-started,
        'scope':'Exact exported bytes reused across confirmation/formal/full-task jobs; no quantization or inference change. All primary baselines and DASO treated equally.'})
    link=path.with_name(path.name+'.cached_'+str(time.time_ns()))
    link.symlink_to(destination);link.replace(path)
    return True
