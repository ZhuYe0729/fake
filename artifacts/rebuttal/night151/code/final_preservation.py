"""Recheck all archived hashes without modifying any sealed run."""
import hashlib
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        while block:=handle.read(8*1024**2):digest.update(block)
    return digest.hexdigest()


def main():
    rows,errors=[],[];started=time.perf_counter()
    for manifest in sorted((ROOT.parent/'runs').glob('*/results/manifest.json')):
        run=manifest.parents[1];items=json.loads(manifest.read_text())['files']
        entries=items.items() if isinstance(items,dict) else ((r['path'],r) for r in items)
        checked,links,mutable,total=0,0,0,0
        for name,record in entries:
            path=run/name
            if record.get('mutable_orchestration_log'):
                mutable+=1;continue
            if 'sha256' in record:
                try:
                    size=path.stat().st_size
                    if 'bytes' in record:assert size==record['bytes'],'size mismatch'
                    assert sha(path)==record['sha256'],'hash mismatch'
                    checked+=1;total+=size
                except (OSError,AssertionError) as error:
                    errors.append({'run':run.name,'path':name,'error':str(error)})
            elif 'symlink' in record:
                try:
                    assert path.is_symlink(),'missing symlink'
                    assert str(path.resolve())==record['symlink'],'symlink target changed'
                    links+=1
                except (OSError,AssertionError) as error:
                    errors.append({'run':run.name,'path':name,'error':str(error)})
            else:
                mutable+=1
        rows.append({'run':run.name,'checked_file_hashes':checked,'checked_symlinks':links,
            'mutable_or_other_entries':mutable,'bytes_hashed':total,'manifest_sha256':sha(manifest)})
        print(run.name,'hashes',checked,'links',links,'errors',len(errors),flush=True)
    result={'verified':not errors,'runs':rows,'errors':errors,'seconds':time.perf_counter()-started,
        'scope':'All sha256 entries and explicitly recorded symlink targets in existing run001–020 manifests; mutable orchestration logs excluded according to original manifests. New runs021–026 retain their own narrower audits.'}
    (ROOT/'final_preservation_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    assert not errors,errors[:20]


if __name__=='__main__':main()
