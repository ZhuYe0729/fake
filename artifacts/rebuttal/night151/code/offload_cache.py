"""Relocate explicit completed-run binaries, preserving bytes and original paths."""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    '/dev/shm/daso_rebuttal_007_20261025/canonical/wanda/model.pt',
    '/dev/shm/daso_rebuttal_007_20261025/canonical/magnitude/model.pt',
    '/dev/shm/daso_rebuttal_001_20261025/canonical/sparse_bf16/model.pt',
    '/dev/shm/daso_rebuttal_001_20261025/canonical/sparse_nvfp4/model.pt',
    '/dev/shm/daso_rebuttal_001_20261025/checkpoints/uniform_sparse_bf16/model.safetensors',
]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(8*1024**2):digest.update(chunk)
    return digest.hexdigest()


def main():
    pending = [Path(value) for value in CANDIDATES if not Path(value).is_symlink()]
    required = sum(path.stat().st_size for path in pending)
    available = shutil.disk_usage(ROOT).free
    # The project is on the large XFS mount, distinct from the nearly full /.
    assert ROOT.stat().st_dev != Path('/').stat().st_dev, 'Do not copy weights onto the root overlay.'
    assert available > required + 32*1024**3, {'target':str(ROOT),'available':available,'required':required}
    print('Capacity gate',str(ROOT),'free GiB',available/1024**3,'copy GiB',required/1024**3,flush=True)
    for value in CANDIDATES:
        source = Path(value)
        destination = ROOT / 'cache' / source.relative_to('/dev/shm')
        record_path = ROOT / 'cache_records' / (str(source.relative_to('/dev/shm')).replace('/','_')+'.json')
        if source.is_symlink():
            assert source.resolve()==destination and record_path.exists()
            continue
        assert source.is_file() and not destination.exists()
        destination.parent.mkdir(parents=True,exist_ok=True)
        record_path.parent.mkdir(parents=True,exist_ok=True)
        started = time.perf_counter()
        stat = source.stat()
        partial = destination.with_name(destination.name+'.partial')
        assert not partial.exists()
        digest = hashlib.sha256()
        with source.open('rb') as src, partial.open('xb') as dst:
            while chunk := src.read(8*1024**2):
                digest.update(chunk);dst.write(chunk)
            dst.flush();os.fsync(dst.fileno())
        expected = digest.hexdigest()
        assert partial.stat().st_size==stat.st_size and sha(partial)==expected
        assert source.stat().st_mtime_ns==stat.st_mtime_ns
        partial.rename(destination)
        record = {'source':str(source),'destination':str(destination),'bytes':stat.st_size,
            'source_mtime_ns':stat.st_mtime_ns,'sha256':expected,'copy_verified':True,
            'reason':'Completed task-owned weight cache relocated off constrained tmpfs; original logical path remains usable; code/logs/results/documents untouched.'}
        record_path.write_text(json.dumps(record,indent=2)+'\n')
        link = source.with_name(source.name+'.night151-link')
        link.symlink_to(destination)
        link.replace(source)
        assert source.is_symlink() and source.stat().st_size==stat.st_size
        print('Preserved and relocated',source,round(stat.st_size/1024**3,3),'GiB',round(time.perf_counter()-started,1),'seconds',flush=True)


if __name__ == '__main__':main()
