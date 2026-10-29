"""Freeze existing Qwen maps and original six workloads in isolated040."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT.parent/'runs';RUN=RUNS/'040'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    assert not RUN.exists()
    for name in ['code','configs/policies','data','reference','results','logs','extensions/sparse_nvfp4']:(RUN/name).mkdir(parents=True,exist_ok=True)
    shutil.copytree(RUNS/'019/code',RUN/'code',dirs_exist_ok=True)
    numeric=['common.py','eightbit.py','runtime_extensions.py','arena_runtime.py','graph_bindings.py','export_model.py','int4_marlin.py','telemetry.py']
    for name in numeric:shutil.copy2(RUNS/'033/code'/name,RUN/'code'/name)
    for name in ['csrc','compat']:shutil.copytree(RUNS/'033/code'/name,RUN/'code'/name,dirs_exist_ok=True)
    for path in(RUNS/'033/data').iterdir():
        if path.is_file()and path.suffix in ['.pt','.json','.jsonl','.csv']:(RUN/'data'/path.name).symlink_to(path.resolve())
    spec={'run':'040','model_id':'qwen3_8b','model_path':'/root/wja/data/models/Qwen/Qwen3-8B',
        'temp':str((RUN/'data/work').resolve()),'canonical_root':str((ROOT.parent/'night151/cache/daso_rebuttal_001_20261025/canonical').resolve())}
    write(RUN/'configs/model.json',spec)
    shutil.copy2(RUNS/'034/code/launch.py',RUN/'code/launch.py')
    resource=(RUNS/'034/code/sweep_resources.py').read_text().replace("['034','035']","['035','036','038','039','040']")
    (RUN/'code/resources.py').write_text(resource)
    entries=[('BF16','032','uniform_dense_bf16'),('INT8 W8A8','032','uniform_int8'),
        ('SmoothQuant INT8','032','uniform_sq_int8'),('FP8 W8A8','019','uniform_fp8'),
        ('Uniform SQ-INT8 prefill / W4A16 decode','019','fixed_sq_int8_w4a16'),
        ('HAWQ-FP4 PTQ r=.80','019','hawq_signed_prefill_decode_phase_pair_r0.8'),
        ('Common hardware-minimum control','019','hardware_minimum_prefill_decode_phase_pair'),
        ('DASO exploratory w1 q=.01','032','w1_e0.01'),('DASO exploratory w32 q=.03','033','w32_e0.03')]
    rows=[];policies={}
    for label,rid,pid in entries:
        source=RUNS/rid/f'configs/policies/{pid}.json';dest=RUN/f'configs/policies/{pid}.json'
        shutil.copy2(source,dest);mapping=read(dest)['method_map'];assert len(mapping)==144
        policies[pid]={'source_run':rid,'source_policy':pid,'source_sha256':sha(source),
            'map_sha256':hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()}
        rows.append({'method':label,'policy':pid,'source_run':rid,'budget':.01 if pid=='w1_e0.01'else .03 if pid=='w32_e0.03'else None})
    old=read(RUNS/'019/configs/design.json')
    workloads=old['workloads']
    for case in workloads:
        assert sha(Path(case['input']))==case['input_sha256']
        assert sha(Path(case['quality_file']))==case['quality_sha256']
    import random
    rounds=[]
    for index in range(3):
        middle=[p for p in policies if p!='uniform_dense_bf16'];random.Random(165040+index).shuffle(middle)
        rounds.append(['uniform_dense_bf16',*middle,'uniform_dense_bf16_bracket_end'])
    protected=[RUNS/r/'results/final_audit.json'for r in ['019','032','033']]
    protected+=[RUNS/'033/code'/name for name in numeric]
    inputs=[Path(c[k])for c in workloads for k in ['input','quality_file']]
    inputs+=[RUN/'data/activation_scales.pt']
    design={'rows':rows,'policies':policies,'workloads':workloads,'rounds':rounds,'warmups':3,'iterations':5,
        'quality_windows':64,'kv_gib':6,'protected_sha256':{str(p.resolve()):sha(p)for p in protected},
        'input_sha256':{str(p.resolve()):sha(p)for p in inputs},
        'scope':'Existing center policies, fixed six-load transfer; previously observed public test corpus; no load-specific tuning or newblind claim.'}
    write(RUN/'configs/design.json',design)
    write(RUN/'configs/design_checks.json',{'verified':True,'maps':9,'workloads':6,'quality_rows':54,'formal_processes':30})
    (RUN/'README.md').write_text('# Qwen新点六负载迁移\n\n019六种负载，九map；包含032/033两个既有探索点及强基线。质量逐负载新测，速度同轮三轮；不按负载重新选点。等待037后质量，等待038/039后正式测速。\n\n[冻结设计](configs/design.json) · [计划](../../../../dev/plans/165_qwen_updated_points_six_loads_plan.md)\n')
    print('Prepared040',len(policies),'maps',len(workloads),'loads',flush=True)


if __name__=='__main__':main()
