"""Freeze source identities and protected deliveries before this run's GPU jobs."""
import hashlib
import json
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
REPO=RUN.parents[3]


def sha(path):
    with path.open('rb')as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    spec=json.loads((RUN/'configs/model.json').read_text());files={}
    model=Path(spec['model_path'])
    if spec['source_run']=='018':
        pinned=RUN.parent/'018/reference/source_model_download.json'
        manifest=json.loads(pinned.read_text())
        files.update({str((model/r['Name']).resolve()):r['Sha256']for r in manifest['files']})
    else:
        for path in sorted(model.glob('*')):
            if path.is_file()and path.suffix in ['.json','.safetensors','.model']:
                files[str(path.resolve())]=sha(path)
                print('Frozen source',path.name,flush=True)
    for path in sorted(Path(spec['canonical_root']).glob('*/model.pt')):
        files[str(path.resolve())]=sha(path)
        print('Frozen canonical',path.parent.name,flush=True)
    target=RUN/'reference/source_model_fingerprint.json'
    target.write_text(json.dumps({'source_run':spec['source_run'],'files_sha256':files},indent=2)+'\n')
    design_path=RUN/'configs/sweep_design.json';design=json.loads(design_path.read_text())
    design['input_sha256'][str(target.resolve())]=sha(target)
    for rid in ['017','018','032','033']:
        path=RUN.parent/rid/'results/final_audit.json';design['input_sha256'][str(path.resolve())]=sha(path)
    for folder in ['llama31_8b_instruct_table2','llama2_7b_chat_table2','qwen3_table2_run016']:
        for name in ['README.md','main_table.csv','provenance.json']:
            path=REPO/'artifacts/rebuttal/responses/reviewer2'/folder/name
            design['input_sha256'][str(path.resolve())]=sha(path)
    design_path.write_text(json.dumps(design,indent=2)+'\n')
    print('Frozen source/protected delivery identities',RUN.name,len(files),flush=True)


if __name__=='__main__':main()
