"""Read-only verification of sealed historical evidence and shared runtimes."""
import hashlib
import json
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
RUNS=RUN.parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main():
    third=RUNS/'003';expected={'003':'ab9022f99d3dc81d507f2c3acb5468e80cd5905a38ec10edb8e610f7dc163fbc'}
    assert digest(third/'results/manifest.json')==expected['003']
    expected.update(json.loads((third/'reference/sources.json').read_text())['inherited_manifests'])
    counts={};manifests={}
    for rid in ['001','002','003']:
        root=RUNS/rid;manifest=root/'results/manifest.json';actual=digest(manifest);assert actual==expected[rid],rid
        entries=json.loads(manifest.read_text())['files']
        for entry in entries:assert digest(root/entry['path'])==entry['sha256'],(rid,entry['path'])
        counts[rid]=len(entries);manifests[rid]=actual;print(rid,len(entries),'verified',flush=True)
    records=[]
    for relative in ['001/reference/runtime_sources/index.json','002/reference/fp8_runtime_sources/index.json']:
        for record in json.loads((RUNS/relative).read_text())['records']:
            assert digest(Path(record['source']))==record['sha256'],record['source'];records.append(record)
    environment=json.loads((RUNS/'002/configs/environment.json').read_text())
    for source,expected_hash in environment['compiled_vllm_ops'].items():assert digest(Path(source))==expected_hash['sha256'],source
    result={'passed':True,'protected_archive_file_counts':counts,'protected_manifest_sha256':manifests,
        'shared_runtime_sources_verified':len(records),'compiled_vllm_ops_verified':environment['compiled_vllm_ops'],
        'scope':'Historical workspace archives and shared runtime source/compiled ops unchanged. Reproducible checkpoint binaries outside the archives may be released only with separate cleanup records.'}
    (RUN/'results/protected_integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'files':sum(counts.values()),'runtime_sources':len(records)}),flush=True)


if __name__=='__main__':main()
