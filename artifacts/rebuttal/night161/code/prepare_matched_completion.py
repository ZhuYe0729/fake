"""Copy frozen017 matched maps and NLL without resolving either allocator."""
import hashlib
import json
import random
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT.parent/'runs';RUN=RUNS/'041'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    assert not RUN.exists()
    for name in ['code','configs/policies','data','reference','results','logs','extensions/sparse_nvfp4']:(RUN/name).mkdir(parents=True,exist_ok=True)
    source=RUNS/'017';current=RUNS/'034'
    for name in ['common.py','eightbit.py','export_model.py','runtime_extensions.py','arena_runtime.py','graph_bindings.py',
        'int4_marlin.py','telemetry.py','admission_check.py','benchmark.py','launch.py']:
        shutil.copy2(current/'code'/name,RUN/'code'/name)
    for name in ['csrc','compat']:shutil.copytree(current/'code'/name,RUN/'code'/name,dirs_exist_ok=True)
    for path in(source/'data').iterdir():
        if path.is_file()and path.suffix in ['.pt','.json','.jsonl','.csv']:(RUN/'data'/path.name).symlink_to(path.resolve())
    spec=read(current/'configs/model.json');spec.update(run='041',temp=str((RUN/'data/work').resolve()));write(RUN/'configs/model.json',spec)
    shutil.copy2(current/'configs/speed_inputs.json',RUN/'configs/speed_inputs.json')
    shutil.copy2(current/'reference/source_model_fingerprint.json',RUN/'reference/source_model_fingerprint.json')
    points=[x for x in read(source/'configs/allocation_pool.json')['points']if x['family']in['daso matched','hawq matched']]
    assert len(points)==18
    pids=['uniform_dense_bf16']+[p['policy_id']for p in points]
    source_hashes={};policy_sha={}
    for pid in pids:
        src=source/f'configs/policies/{pid}.json';dst=RUN/f'configs/policies/{pid}.json';shutil.copy2(src,dst);policy_sha[pid]=sha(dst)
        scenes=['prefill_only','prefill_decode']if pid=='uniform_dense_bf16'else[next(x['scenario']for x in points if x['policy_id']==pid)]
        for scene in scenes:
            src=source/f'results/confirmation/{pid}/{scene}.json';assert src.exists()
            dst=RUN/f'results/confirmation/{pid}/{scene}.json';dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src.resolve())
            source_hashes[str(src.resolve())]=sha(src)
    rounds=[]
    for i in range(3):
        middle=pids[1:].copy();random.Random(166041+i).shuffle(middle)
        rounds.append(['uniform_dense_bf16',*middle,'uniform_dense_bf16_bracket_end'])
    protected=[source/'results/final_audit.json',current/'results/sweep_final_audit.json',source/'configs/allocation_pool.json',source/'configs/frozen_design.json']
    design={'points':points,'policy_sha256':policy_sha,'rounds':rounds,'input_sha256':sha(RUN/'configs/speed_inputs.json'),
        'nll_sources_sha256':source_hashes,'protected_sha256':{str(p.resolve()):sha(p)for p in protected},
        'scope':'All18 existing matched signed-Hessian/DASO maps, no new solver/quality fitting; new3-round speed, original absolute target retained.'}
    write(RUN/'configs/design.json',design)
    resource=(current/'code/sweep_resources.py').read_text().replace("['034','035']","['038','039','040','041']")
    (RUN/'code/resources.py').write_text(resource)
    (RUN/'README.md').write_text('# Llama-3.1同候选分配与反约束实际时延\n\n冻结017全部18个同候选分配点，复用原64窗口质量，在040结束后补统一三轮实际速度及上限超限诊断。\n\n[冻结点](configs/design.json) · [计划](../../../../dev/plans/166_llama_matched_allocator_latency_completion_plan.md)\n')
    print('Prepared041:18 points/19maps/60 formal processes; original NLL frozen',flush=True)


if __name__=='__main__':main()
