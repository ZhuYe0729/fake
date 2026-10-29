"""Freeze overhead controls before new GPU measurements; reuse immutable inputs."""
import hashlib
import json
import shutil
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
OLD=RUN.parent/'003'


def main():
    for d in ['configs/policies','data','reference','results','logs','code/compat']:
        (RUN/d).mkdir(parents=True,exist_ok=True)
    for name in ['common.py','eightbit.py','runtime_extensions.py','export_model.py','int4_marlin.py']:
        shutil.copy2(OLD/'code'/name,RUN/'code'/name)
    shutil.copytree(OLD/'code/compat',RUN/'code/compat',dirs_exist_ok=True)
    for name in ['profile_weights.pt','activation_scales.pt']:
        (RUN/'data'/name).symlink_to(OLD/'data'/name)
    for name in ['action_profile.csv','profile_complete.json','allocation_replay_audit.json']:
        (RUN/'reference'/name).symlink_to(OLD/'results'/name)
    policies=['uniform_dense_bf16','uniform_sq_int8','uniform_fp8',
              'transfer_daso5_measured_prefill_decode_04','transfer_daso8_prefill_decode_01',
              'phase_sq_int8_w4a16_ours']
    for name in policies:shutil.copy2(OLD/f'configs/policies/{name}.json',RUN/f'configs/policies/{name}.json')
    design={'policies':policies,'modes':['async','sync','resident'],
        'uniform_modes':['async'],'warmups':2,'iterations':5,'kv_bytes':6*1024**3,
        'workloads':[{'id':'standard','batch':8,'input':2048,'output':64},
                     {'id':'long_prompt','batch':8,'input':4096,'output':64},
                     {'id':'decode_heavy','batch':1,'input':512,'output':256}],
        'engine':{'max_model_len':4360,'max_num_seqs':16,'max_num_batched_tokens':32768},
        'latency_holdout':{'train_m':[1,8,2048,16384],'test_m':[4,16,512,4096,8192,32768]},
        'mlp_chain':{'m':[8,16384],'methods':['dense_bf16','dense_nvfp4','sparse_bf16',
                      'sparse_nvfp4','w4a16_ours','sq_int8','fp8']},
        'scope':'untraced full generation and restore-cycle latency; separate short profiler traces; same frozen policies',
        'resident':'both phase representations on GPU; no per-batch DMA; extra memory reported',
        'selection':'fixed before run004 results; no test-quality tuning'}
    (RUN/'configs/design.json').write_text(json.dumps(design,indent=2)+'\n')
    inputs={str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in ['code','configs'] for p in (RUN/directory).rglob('*') if p.is_file()}
    (RUN/'reference/initial_hashes.json').write_text(json.dumps(inputs,indent=2)+'\n')
    (RUN/'README.md').write_text('# Run004 — 人工 reviewer 开销证据\n\n'
        '计划：[144](../../../../dev/plans/144_human_reviewer_rebuttal_overnight_plan.md)。'
        '固定策略、三种负载、异步/同步/双驻留控制。003完成后启动GPU测试。'
        '历史准备成本、严格shape留出预测误差、真实MLP接口与运行时开销分别报告。\n')
    print(json.dumps({'frozen':True,'policies':len(policies),'runtime_combinations':36}))


if __name__=='__main__':main()
