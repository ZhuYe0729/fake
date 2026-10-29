"""Archive reused source files and the exact cached likelihood evaluation splits."""
import hashlib
import json
import shutil
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
REPO=RUN.parents[3]


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    return {'source':str(source),'saved':str(destination.relative_to(RUN)),
        'bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}


def main():
    environment=json.loads((RUN/'configs/environment.json').read_text())
    wrapper=Path(environment['wrapper_root']);vllm=Path(environment['vllm_root'])
    records=[]
    groups={'daso_src':REPO/'daso/src','paper_runtime':REPO/'daso/workflows/paper/runtime',
        'pmpd_eval':REPO/'references/pmpd_eval_kit','cutlass_wrapper':wrapper/'cutlass_wrapper',
        'standalone_marlin_int4':wrapper/'standalone_marlin_int4',
        'standalone_w4a4_int4':wrapper/'standalone_w4a4_int4'}
    for name,root in groups.items():
        for path in sorted(root.rglob('*')):
            if path.is_file() and (path.suffix in ['.py','.cpp','.cu','.cuh','.h'] or path.name.startswith('LICENSE')):
                records.append(copy(path,RUN/'reference/runtime_sources'/name/path.relative_to(root)))
    for name,expected in environment['backend_files'].items():
        path=Path(name);assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
        records.append(copy(path,RUN/'reference/runtime_sources/vllm'/path.relative_to(vllm)))
    for relative in ['_custom_ops.py','scalar_type.py']:
        path=vllm/'vllm/vllm'/relative
        if path.exists():records.append(copy(path,RUN/'reference/runtime_sources/vllm'/path.relative_to(vllm)))
    save(RUN/'reference/runtime_sources/index.json',{'scope':'final reused sources; five original vLLM files match initial hashes',
        'records':records,'wrapper_commit':environment['wrapper_commit'],'vllm_commit':environment['vllm_commit']})
    configs={}
    for task in ['wikitext','winogrande','arc_easy','arc_challenge','mmlu']:
        result=json.loads((RUN/f'results/tasks/uniform_dense_bf16/{task}/full/result.json').read_text())
        configs.update(result['raw_lm_eval']['configs'])
    save(RUN/'data/likelihood_task_configs.json',configs)
    builders={}
    for config in configs.values():
        name=config.get('dataset_path')
        if isinstance(name,str):builders.setdefault(name.replace('/','___').lower(),set()).add(config.get('dataset_name'))
    cache=Path('/root/wja/.cache/huggingface/datasets');datasets=[]
    for root in sorted(cache.iterdir()):
        if root.name.lower() not in builders:continue
        names=builders[root.name.lower()]
        for path in sorted(root.rglob('*.arrow')):
            if not any(split in path.name for split in ['-test','-validation']):continue
            if None not in names and path.relative_to(root).parts[0] not in names:continue
            datasets.append(copy(path,RUN/'data/likelihood_datasets'/path.relative_to(cache)))
    assert datasets,'no evaluation dataset files archived'
    save(RUN/'data/likelihood_datasets/index.json',{'scope':'cached test/validation Arrow splits matching the recorded task configurations',
        'records':datasets,'total_bytes':sum(row['bytes'] for row in datasets)})
    (RUN/'data/likelihood_datasets/README.md').write_text(
        'Frozen cached test/validation splits for the recorded likelihood task definitions.\n'
        'Use datasets.Dataset.from_file(path) to inspect each Arrow file.\n'
        'Task templates/settings are in ../likelihood_task_configs.json; dependency versions are in ../../configs/package_versions.json.\n')
    print('archived sources',len(records),'evaluation splits',len(datasets),'bytes',sum(row['bytes'] for row in datasets),flush=True)


if __name__=='__main__':main()
