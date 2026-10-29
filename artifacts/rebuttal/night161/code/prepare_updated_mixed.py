"""Freeze two original mixed cases and all nine updated Qwen maps."""
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT.parent/'runs'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    run=RUNS/'047';assert not run.exists()
    for name in ['code','configs/policies','data','reference','results','logs','extensions/sparse_nvfp4']:
        (run/name).mkdir(parents=True,exist_ok=True)
    inputs={}
    for name in ['common.py','eightbit.py','export_model.py','runtime_extensions.py','int4_marlin.py','launch.py','telemetry.py']:
        path=RUNS/'040/code'/name;shutil.copy2(path,run/'code'/name);inputs[str(path.resolve())]=sha(path)
    for name in ['compat','csrc']:shutil.copytree(RUNS/'040/code'/name,run/'code'/name)
    for name in ['resident.py','mixed_runtime.py','teacher_force_mixed.py']:
        path=RUNS/'006/code'/name;shutil.copy2(path,run/'code'/name);inputs[str(path.resolve())]=sha(path)
    spec=read(RUNS/'040/configs/model.json');spec.update(run='047',temp=str((run/'data/work').resolve()))
    write(run/'configs/model.json',spec)
    source=read(RUNS/'040/configs/design.json');rows=source['rows'];policies={}
    for row in rows:
        pid=row['policy'];path=RUNS/f'040/configs/policies/{pid}.json'
        shutil.copy2(path,run/f'configs/policies/{pid}.json');policies[pid]=sha(path);inputs[str(path.resolve())]=sha(path)
    assert len(policies)==9
    for name,src in [('activation_scales.pt',RUNS/'002/data/activation_scales.pt'),
        ('quality_64_4160.pt',RUNS/'019/data/long_quality.pt')]:
        (run/'data'/name).symlink_to(src.resolve());inputs[str(src.resolve())]=sha(src)
    original=read(RUNS/'006/configs/design.json')
    cases=[c for c in original['cases']if c['token_budget']==1024];assert len(cases)==2
    rounds=[]
    for ri in range(3):
        middle=[p for p in policies if p!='uniform_dense_bf16'];random.Random(170047+ri).shuffle(middle)
        rounds.append(['uniform_dense_bf16',*middle,'uniform_dense_bf16_bracket_end'])
    protected=[RUNS/'006/REPORT.md',RUNS/'040/configs/design.json',
        RUNS/'032/results/final_audit.json',RUNS/'033/results/final_audit.json']
    design={'created_at_unix':time.time(),'rows':rows,'policies':policies,'cases':cases,'rounds':rounds,
        'warmups':2,'iterations':5,'kv_bytes':6*1024**3,'engine':{'max_model_len':2304,'max_num_seqs':8},
        'speed_seed':original['speed_seed'],'quality_blocks':64,'quality_file':str((run/'data/quality_64_4160.pt').resolve()),
        'quality_sha256':sha(run/'data/quality_64_4160.pt'),'input_sha256':inputs,
        'protected_sha256':{str(p.resolve()):sha(p)for p in protected},
        'scope':'Frozen nine-map transfer to two closed-loop mixed bursts, dual-resident eager variant; no retuning, production/open-loop or nominal q guarantee.'}
    write(run/'configs/design.json',design)
    (run/'README.md').write_text('# Qwen新点：真实chunked/mixed调度\n\n固定040同九map、006两代表case，先64窗口实际mixed teacher forcing与BF16 no-op校验，再同卡三轮2warmup+5timed burst。双表示驻留变体单列内存，不混入主表，不称open-loop生产服务。等待044/045/046完成后开始。\n\n[设计](configs/design.json) · [计划170](../../../../dev/plans/170_updated_qwen_mixed_scheduler_plan.md)\n')
    print('Prepared047: nine maps, two mixed cases,64 paired windows,three rounds',flush=True)


if __name__=='__main__':main()
