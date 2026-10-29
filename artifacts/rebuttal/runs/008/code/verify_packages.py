"""Check final reviewer copies, manifests and relative source links."""
import hashlib
import json
from pathlib import Path

RUNS=Path(__file__).resolve().parents[2]
RESPONSES=RUNS.parent/'responses'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main():
    records=[]
    for path in sorted(RESPONSES.glob('reviewer*/overnight_run*/source.json')):
        r=json.loads(path.read_text());root=RUNS/r['run'];package=path.parent
        assert digest(root/'results/manifest.json')==r['manifest_sha256']
        for relative,expected in r['copies'].items():
            original='README.md' if relative=='RUN_README.md' else relative
            assert digest(package/relative)==expected==digest(root/original),(package,relative)
        for name,target in r['links'].items():
            link=package/name;assert link.is_symlink() and link.exists() and link.readlink()==Path(target)
        records.append({'package':str(package.relative_to(RESPONSES)),'run':r['run'],
            'copies_verified':len(r['copies']),'links_verified':len(r['links']),'manifest_sha256':r['manifest_sha256']})
    assert len(records)==21,len(records)
    result={'passed':True,'overnight_packages':len(records),'runs':sorted(set(r['run'] for r in records)),
        'records':records,'scope':'All reviewer copy bytes and source manifests/relative links verified; full run content verified by each sealing step.'}
    (RESPONSES/'overnight_packaging_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'packages':len(records),'copied_files':sum(r['copies_verified'] for r in records),'links':sum(r['links_verified'] for r in records)}))


if __name__=='__main__':main()
